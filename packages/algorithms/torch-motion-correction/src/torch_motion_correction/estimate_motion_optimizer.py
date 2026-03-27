"""Estimate local motion using a deformation field."""

from typing import Any, cast

import einops
import torch
import torch.utils.checkpoint as checkpoint
import tqdm
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d
from torch_fourier_filter.envelopes import b_envelope
from torch_fourier_shift import fourier_shift_dft_2d
from torch_grid_utils import circle

from torch_motion_correction.deformation_field_utils import (
    resample_deformation_field,
)
from torch_motion_correction.optimization_state import OptimizationTracker
from torch_motion_correction.patch_grid import (
    patch_grid_centers,
)
from torch_motion_correction.patch_utils import ImagePatchIterator
from torch_motion_correction.utils import (
    normalize_image,
    prepare_bandpass_filter,
)


def estimate_local_motion(
    image: torch.Tensor,  # (t, H, W)
    pixel_spacing: float,  # Angstroms
    patch_shape: tuple[int, int],  # (ph, pw)
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    initial_deformation_field: torch.Tensor | None,  # (yx, nt, nh, nw)
    device: torch.device = None,
    n_iterations: int = 100,
    b_factor: float = 500,
    frequency_range: tuple[float, float] = (300, 10),  # angstroms
    optimizer_type: str = "adam",
    grid_type: str = "catmull_rom",
    loss_type: str = "mse",
    optimizer_kwargs: dict | None = None,
    return_trajectory: bool = False,
    trajectory_kwargs: dict | None = None,
) -> torch.Tensor | tuple[torch.Tensor, OptimizationTracker]:
    """
    Estimate motion.

    Parameters
    ----------
    image: torch.Tensor
        (t, H, W) image to estimate motion from where t is the number of frames,
        H is the height, and W is the width.
    pixel_spacing: float
        Pixel spacing in Angstroms.
    patch_shape: tuple[int, int]
        Size of the patches to extract (ph, pw) in terms of pixels.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw) where nt is the number of
        time points, nh is the number of control points in height, and nw is the
        number of control points in width.
    initial_deformation_field: torch.Tensor | None
        Initial deformation field to start from with shape (2, nt, nh, nw) where 2
        corresponds to (y, x) shifts. If None, initializes to zero shifts.
    device: torch.device, optional
        Device to perform computation on. If None, uses the device of the input image.
    n_iterations: int
        Number of iterations for the optimization process. Default is 100.
    b_factor: float
        B-factor to apply in Fourier space to downweight high frequencies.
        Default is 500.
    frequency_range: tuple[float, float]
        Frequency range in Angstroms for bandpass filtering (low, high).
        Default is (300, 10).
    loss_type: str
        Type of loss to use ('mse', 'cc' or 'ncc'). Default is 'mse'.
    optimizer_type: str
        Type of optimizer to use ('adam' or 'lbfgs'). Default is 'adam'.
    grid_type: str
        Type of grid to use ('catmull_rom' or 'bspline'). Default is 'catmull_rom'.
    optimizer_kwargs: dict | None
        Additional keyword arguments for the optimizer. If None, uses defaults.
    return_trajectory: bool
        Whether to return the optimization trajectory. Default is False. If true, a
        second return value will be provided which is an OptimizationTrajectory object.
    trajectory_kwargs: dict | None
        Additional keyword arguments for the trajectory tracking. If None, uses
        defaults.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, OptimizationTracker]
        The estimated deformation field with shape (2, nt, nh, nw) where 2 corresponds
        to (y, x) shifts. If `return_trajectory` is True, also returns an
        OptimizationTrajectory object containing the optimization history.
    """
    device = device if device is not None else image.device
    image = image.to(device)
    t, h, w = image.shape
    ph, pw = patch_shape

    if return_trajectory:
        trajectory_kwargs = trajectory_kwargs if trajectory_kwargs is not None else {}
        # Provide sensible defaults when not supplied
        trajectory_kwargs.setdefault("sample_every_n_steps", 1)
        trajectory_kwargs.setdefault("total_steps", n_iterations)
        trajectory = OptimizationTracker(**trajectory_kwargs)

    # Normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # Create the patch grid
    patch_positions = patch_grid_centers(
        image_shape=(t, h, w),
        patch_shape=(1, ph, pw),
        patch_step=(1, ph // 2, pw // 2),  # Default 50% overlap
        distribute_patches=True,
        device=device,
    )  # (t, gh, gw, 3)

    # gh, gw = patch_positions.shape[1:3]

    if grid_type == "catmull_rom":
        new_deformation_field = CubicCatmullRomGrid3d(
            resolution=deformation_field_resolution, n_channels=2
        ).to(device)
    elif grid_type == "bspline":
        new_deformation_field = CubicBSplineGrid3d(
            resolution=deformation_field_resolution, n_channels=2
        ).to(device)
    else:
        raise ValueError(
            f"Invalid grid type: {grid_type}. Must be 'catmull_rom' or 'bspline'."
        )

    if initial_deformation_field is None:
        deformation_field_data = torch.zeros(
            size=(2, *deformation_field_resolution), device=device
        )
    elif initial_deformation_field is not None:
        deformation_field_data = resample_deformation_field(
            deformation_field=initial_deformation_field.detach(),
            target_resolution=(
                deformation_field_resolution[0],
                deformation_field_resolution[1],
                deformation_field_resolution[2],
            ),
        )
        deformation_field_data -= torch.mean(deformation_field_data)
        # deformation_field_data *= -1

    if grid_type == "catmull_rom":
        deformation_field = CubicCatmullRomGrid3d.from_grid_data(
            deformation_field_data
        ).to(device)
    elif grid_type == "bspline":
        deformation_field = CubicBSplineGrid3d.from_grid_data(
            deformation_field_data
        ).to(device)

    # Reusable masks and Fourier filters
    # NOTE: This is assuming square patches... revisit if needed
    circle_mask = circle(
        radius=patch_shape[1] / 4,
        image_shape=patch_shape,
        smoothing_radius=patch_shape[1] / 4,
        device=device,
    )

    b_factor_envelope = b_envelope(
        B=b_factor,
        image_shape=patch_shape,
        pixel_size=pixel_spacing,
        rfft=True,
        fftshift=False,
        device=device,
    )

    bandpass_filter = prepare_bandpass_filter(
        frequency_range=frequency_range,
        patch_shape=patch_shape,
        pixel_spacing=pixel_spacing,
        refinement_fraction=1.0,  # Not used in this context
        device=device,
    )

    # Instantiate the patch iterator (mini-batch like data-loader)
    image_patch_iterator = ImagePatchIterator(
        image=image,
        patch_size=patch_shape,
        control_points=patch_positions,
    )

    motion_optimizer = _setup_optimizer(
        optimizer_type=optimizer_type,
        parameters=new_deformation_field.parameters(),
        **(optimizer_kwargs if optimizer_kwargs is not None else {}),
    )

    # For LBFGS, optionally subsample patches per closure to reduce memory
    # This uses a subset of patches for each closure call, which is common practice
    lbfgs_patch_subsample = None
    use_checkpointing = True
    if optimizer_type.lower() == "lbfgs":
        lbfgs_patch_subsample = (
            optimizer_kwargs.get("lbfgs_patch_subsample", None)
            if optimizer_kwargs
            else None
        )
        # Check if gradient checkpointing is enabled (default: True for memory savings)
        use_checkpointing = (
            optimizer_kwargs.get("use_gradient_checkpointing", True)
            if optimizer_kwargs
            else True
        )

    # "Training" loop going over all patched n_iterations times
    pbar = tqdm.tqdm(range(n_iterations))
    for iter_idx in pbar:
        if optimizer_type.lower() == "lbfgs":

            def compute_patch_loss(
                patch_batch: torch.Tensor, patch_batch_centers: torch.Tensor
            ) -> torch.Tensor:
                """
                Helper function for gradient checkpointing.

                Parameters
                ----------
                patch_batch: torch.Tensor
                    (b, t, ph, pw) image patches
                patch_batch_centers: torch.Tensor
                    (b, t, 3) normalized control point centers

                Returns
                -------
                batch_loss: torch.Tensor
                    (1,) loss value
                """
                # Apply circular mask in real space; then FFT to rfft with
                # shape (b, t, ph, pw//2 + 1)
                patch_batch_masked = patch_batch * circle_mask
                patch_batch_fft = torch.fft.rfftn(patch_batch_masked, dim=(-2, -1))

                shifted_patches, _predicted_shifts_angstroms = (
                    _compute_shifted_patches_and_shifts(
                        initial_deformation_field=deformation_field,
                        new_deformation_field=new_deformation_field,
                        patch_batch=patch_batch_fft,
                        patch_batch_centers=patch_batch_centers,
                        pixel_spacing=pixel_spacing,
                        ph=ph,
                        pw=pw,
                        b_factor_envelope=b_factor_envelope,
                        bandpass=bandpass_filter,
                    )
                )

                # shifted_patches: (b, t, ph, pw//2 + 1)
                # For each frame, use mean of all OTHER frames as reference
                # -> (b, t, ph, pw//2 + 1)
                total_sum = torch.sum(
                    shifted_patches, dim=1, keepdim=True
                )  # (b, 1, ph, pw//2 + 1)
                if t > 1:
                    reference_patches = (total_sum - shifted_patches) / (
                        t - 1
                    )  # (b, t, ph, pw//2 + 1)
                else:
                    reference_patches = shifted_patches
                # Original MSE loss in Fourier domain; normalized by number of
                # pixels for scale stability
                batch_loss = _compute_loss(
                    shifted_patches, reference_patches, ph, pw, loss_type=loss_type
                )
                return batch_loss

            def closure() -> torch.Tensor:
                """
                Closure function for LBFGS optimizer.

                Returns
                -------
                avg_loss: torch.Tensor
                    (1,) average loss value
                """
                motion_optimizer.zero_grad()
                # Process batches one at a time to minimize memory usage
                # Accumulate weighted loss sum (maintaining computation graph)
                # instead of stacking
                weighted_loss_sum = None
                n_batches = 0
                # Iterate over mini-batches of patches (deterministic order for LBFGS)
                # patch_batch: (b, t, ph, pw)
                # patch_batch_centers: (b, t, 3)
                iterator = image_patch_iterator.get_iterator(
                    batch_size=1, randomized=True
                )
                for idx, (patch_batch, patch_batch_centers) in enumerate(iterator):
                    # Optional subsampling: skip patches if specified (reduces memory)
                    if (
                        lbfgs_patch_subsample is not None
                        and idx >= lbfgs_patch_subsample
                    ):
                        break

                    # Use gradient checkpointing to trade compute for memory
                    # (~50% memory savings)
                    if use_checkpointing:
                        batch_loss = checkpoint.checkpoint(
                            compute_patch_loss,
                            patch_batch,
                            patch_batch_centers,
                            use_reentrant=False,  # Required for LBFGS compatibility
                        )
                    else:
                        # No checkpointing - faster but uses more memory
                        batch_loss = compute_patch_loss(
                            patch_batch, patch_batch_centers
                        )

                    # Accumulate weighted sum (still in computation graph) to
                    # avoid stacking
                    if weighted_loss_sum is None:
                        weighted_loss_sum = batch_loss
                    else:
                        weighted_loss_sum = weighted_loss_sum + batch_loss
                    n_batches += 1

                if n_batches == 0:
                    return torch.tensor(0.0, device=device, requires_grad=True)
                # Compute average loss (still connected to all batch computation graphs)
                # weighted_loss_sum cannot be None here because n_batches > 0
                assert weighted_loss_sum is not None
                avg_loss = weighted_loss_sum / n_batches
                # Call backward once on the average to accumulate all gradients
                avg_loss.backward()
                return avg_loss

            avg_loss_tensor = motion_optimizer.step(closure)
            avg_loss_value = (
                float(avg_loss_tensor.detach())
                if isinstance(avg_loss_tensor, torch.Tensor)
                else float(avg_loss_tensor)
            )

            # Clear CUDA cache after optimizer step (safer - gradients already computed)
            # if device.type == "cuda":
            #    torch.cuda.empty_cache()

            # update tqdm with current running average loss for this iteration
            pbar.set_postfix({"avg_batch_loss": f"{avg_loss_value:.6f}"})

            # Record trajectory if requested
            if return_trajectory and trajectory.sample_this_step(iter_idx):
                trajectory.add_checkpoint(
                    deformation_field=new_deformation_field.data,
                    loss=avg_loss_value,
                    step=iter_idx,
                )

        else:
            patch_iter = image_patch_iterator.get_iterator(batch_size=8)  # TODO: expose
            total_loss = 0.0
            n_batches = 0
            # Iterate over mini-batches of patches
            # patch_batch: (b, t, ph, pw)
            # patch_batch_centers: (b, t, 3)
            for patch_batch, patch_batch_centers in patch_iter:
                # Apply circular mask in real space; then FFT to rfft with
                # shape (b, t, ph, pw//2 + 1)
                patch_batch = patch_batch * circle_mask
                patch_batch = torch.fft.rfftn(patch_batch, dim=(-2, -1))

                shifted_patches, _predicted_shifts_angstroms = (
                    _compute_shifted_patches_and_shifts(
                        initial_deformation_field=deformation_field,
                        new_deformation_field=new_deformation_field,
                        patch_batch=patch_batch,
                        patch_batch_centers=patch_batch_centers,
                        pixel_spacing=pixel_spacing,
                        ph=ph,
                        pw=pw,
                        b_factor_envelope=b_factor_envelope,
                        bandpass=bandpass_filter,
                    )
                )

                # shifted_patches: (b, t, ph, pw//2 + 1)
                # For each frame, use mean of all OTHER frames as reference
                # -> (b, t, ph, pw//2 + 1)
                total_sum = torch.sum(
                    shifted_patches, dim=1, keepdim=True
                )  # (b, 1, ph, pw//2 + 1)
                if t > 1:
                    reference_patches = (total_sum - shifted_patches) / (
                        t - 1
                    )  # (b, t, ph, pw//2 + 1)
                else:
                    reference_patches = shifted_patches
                # Original MSE loss in Fourier domain; normalized by number of
                # pixels for scale stability
                loss = _compute_loss(
                    shifted_patches, reference_patches, ph, pw, loss_type=loss_type
                )

                # Use gradient accumulation to optimize over all patches simultaneously
                loss.backward()
                loss_value = (
                    loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                )
                total_loss += loss_value
                n_batches += 1

            # Step the optimizer after each pass over whole image
            motion_optimizer.step()
            motion_optimizer.zero_grad()

            # update tqdm with current running average loss for this iteration
            current_avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
            pbar.set_postfix({"avg_batch_loss": f"{current_avg_loss:.6f}"})

            # Record trajectory if requested
            average_loss = total_loss / n_batches if n_batches > 0 else 0.0
            if return_trajectory and trajectory.sample_this_step(iter_idx):
                trajectory.add_checkpoint(
                    deformation_field=new_deformation_field.data,
                    loss=average_loss,
                    step=iter_idx,
                )

    # Return final deformation field
    final_deformation_field = new_deformation_field.data + deformation_field.data
    average_shift = torch.mean(final_deformation_field.data)
    final_deformation_field = final_deformation_field.data - average_shift

    if return_trajectory:
        return final_deformation_field, trajectory
    else:
        return final_deformation_field


def _compute_shifted_patches_and_shifts(
    initial_deformation_field: CubicCatmullRomGrid3d,
    new_deformation_field: CubicCatmullRomGrid3d,
    patch_batch: torch.Tensor,
    patch_batch_centers: torch.Tensor,
    pixel_spacing: float,
    ph: int,
    pw: int,
    b_factor_envelope: torch.Tensor = None,
    bandpass: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the forward pass for motion estimation for a batch of patches.

    Parameters
    ----------
    initial_deformation_field : CubicCatmullRomGrid3d
        The deformation field model to predict shifts.
    new_deformation_field : CubicCatmullRomGrid3d
        The new deformation field model to predict shifts.
    patch_batch : torch.Tensor
        A batch of image patches in Fourier space with shape (b, t, ph, pw).
    patch_batch_centers : torch.Tensor
        Normalized control point centers for the batch with shape (b, t, 3).
    pixel_spacing : float
        Pixel spacing in Angstroms.
    ph : int
        Patch height in pixels.
    pw : int
        Patch width in pixels.
    b_factor_envelope : torch.Tensor | None
        The B-factor envelope to apply in Fourier space with shape (ph, pw//2 + 1).
        If None, no envelope is applied.
    bandpass : torch.Tensor | None
        The bandpass filter to apply in Fourier space with shape (ph, pw//2 + 1).
        If None, no bandpass is applied.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing:
        - shifted_patches: The shifted patches after applying predicted shifts and
          filters, with shape (b, t, ph, pw//2 + 1).
        - predicted_shifts: The predicted shifts from the deformation field,
          with shape (b, t, 2).
    """
    predicted_shifts = -1 * (
        new_deformation_field(patch_batch_centers)
        + initial_deformation_field(patch_batch_centers)
    )
    predicted_shifts = einops.rearrange(predicted_shifts, "b t yx -> t b yx")
    predicted_shifts_px = predicted_shifts / pixel_spacing

    # Shift the patches by the predicted shifts
    shifted_patches = fourier_shift_dft_2d(
        dft=patch_batch,
        image_shape=(ph, pw),
        shifts=predicted_shifts_px,
        rfft=True,
        fftshifted=False,
    )  # (b, t, ph, pw//2 + 1)

    # Apply Fourier filters
    if bandpass is not None:
        shifted_patches = shifted_patches * bandpass

    if b_factor_envelope is not None:
        shifted_patches = shifted_patches * b_factor_envelope

    return shifted_patches, predicted_shifts


def _setup_optimizer(
    optimizer_type: str,
    parameters: list[torch.Tensor],
    **kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Helper function to setup optimizer with given parameters and kwargs.

    Parameters
    ----------
    optimizer_type: str
        Type of optimizer to use ('adam', 'sgd', 'rmsprop', or 'lbfgs').
    parameters: list[torch.Tensor]
        List of parameters to optimize.
    **kwargs: dict[str, Any]
        Additional keyword arguments for the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        The optimizer object.
    """
    if optimizer_type.lower() == "adam":
        lr = kwargs.get("lr", 0.01)
        betas = kwargs.get("betas", (0.9, 0.999))
        eps = kwargs.get("eps", 1e-08)
        weight_decay = kwargs.get("weight_decay", 0)
        amsgrad = kwargs.get("amsgrad", False)
        return torch.optim.Adam(
            params=parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_type.lower() == "sgd":
        lr = kwargs.get("lr", 0.01)
        momentum = kwargs.get("momentum", 0.9)  # Default momentum for stability
        weight_decay = kwargs.get("weight_decay", 0)
        dampening = kwargs.get("dampening", 0)
        nesterov = kwargs.get("nesterov", True)
        return torch.optim.SGD(
            params=parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
    elif optimizer_type.lower() == "rmsprop":
        lr = kwargs.get("lr", 0.01)
        alpha = kwargs.get("alpha", 0.99)
        eps = kwargs.get("eps", 1e-08)
        weight_decay = kwargs.get("weight_decay", 0)
        momentum = kwargs.get("momentum", 0)
        centered = kwargs.get("centered", False)
        return torch.optim.RMSprop(
            params=parameters,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
    elif optimizer_type.lower() == "lbfgs":
        lr = kwargs.get("lr", 1)
        max_iter = cast(
            "int", kwargs.get("max_iter", 1)
        )  # Minimal line search to reduce memory usage
        max_eval = cast("int | None", kwargs.get("max_eval", None))
        tolerance_grad = kwargs.get("tolerance_grad", 1e-11)
        tolerance_change = kwargs.get("tolerance_change", 1e-11)
        history_size = kwargs.get(
            "history_size", 5
        )  # Reduced from default 100 to save memory
        # Limit max_eval to prevent excessive closure calls (defaults to max_iter * 2)
        if max_eval is None:
            max_eval = max(1, int(max_iter * 1.25))  # Minimal evaluations
        line_search_fn = kwargs.get("line_search_fn", "strong_wolfe")
        return torch.optim.LBFGS(
            params=parameters,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. "
            f"Choose 'adam', 'sgd', 'rmsprop', or 'lbfgs'."
        )


def _compute_loss(
    shifted_patches: torch.Tensor,
    reference_patches: torch.Tensor,
    ph: int,
    pw: int,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Compute the loss for a batch of shifted patches and reference patches.

    Parameters
    ----------
    shifted_patches : torch.Tensor
        The shifted patches with shape (b, t, ph, pw//2 + 1).
    reference_patches : torch.Tensor
        The reference patches with shape (b, t, ph, pw//2 + 1).
    ph : int
        Patch height in pixels.
    pw : int
        Patch width in pixels.
    loss_type : str, optional
        The type of loss to compute. Default is "mse". Other option is
        normalized cross-correlation (ncc).
    """
    if loss_type == "mse":
        return torch.mean((shifted_patches - reference_patches).abs() ** 2) / (ph * pw)
    elif loss_type == "ncc":
        # Inputs are in rFFT space with shapes:
        # shifted_patches: (b, t, ph, pw//2 + 1)
        # reference_patches: (b, t, ph, pw//2 + 1)
        # Convert to real space for NCC computation
        shifted_real = torch.fft.irfftn(shifted_patches, s=(ph, pw), dim=(-2, -1))
        reference_real = torch.fft.irfftn(reference_patches, s=(ph, pw), dim=(-2, -1))
        # Compute normalized cross-correlation over spatial dims for each (b, t)
        eps = 1e-8
        x = shifted_real  # (b, t, ph, pw)
        y = reference_real  # (b, t, ph, pw)
        x_mean = x.mean(dim=(-2, -1), keepdim=True)
        y_mean = y.mean(dim=(-2, -1), keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        numerator = (x_centered * y_centered).sum(dim=(-2, -1))  # (b, t)
        denom = torch.sqrt(
            (x_centered.square().sum(dim=(-2, -1)) + eps)
            * (y_centered.square().sum(dim=(-2, -1)) + eps)
        )
        ncc = numerator / denom  # (b, t)
        return -ncc.mean()
    elif loss_type == "cc":
        # Inputs are in rFFT space with shapes:
        # shifted_patches: (b, t, ph, pw//2 + 1)
        # reference_patches: (b, t, ph, pw//2 + 1)
        # Convert to real space for CC computation
        shifted_real = torch.fft.irfftn(shifted_patches, s=(ph, pw), dim=(-2, -1))
        reference_real = torch.fft.irfftn(reference_patches, s=(ph, pw), dim=(-2, -1))

        # Compute unnormalized cross-correlation over spatial dims
        # (b, t, ph, pw) * (b, t, ph, pw) → (b, t)
        cc = (shifted_real * reference_real).sum(dim=(-2, -1))

        # Optionally: mean over batch and time; negate to make it a loss
        return -cc.mean()
