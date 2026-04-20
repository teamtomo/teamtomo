"""Estimate local motion using a deformation field."""

from collections.abc import Callable
from typing import Any, cast

import einops
import torch
import torch.utils.checkpoint as checkpoint
import tqdm
from torch_fourier_shift import fourier_shift_dft_2d

from torch_motion_correction.deformation_field import DeformationField
from torch_motion_correction.optimization_state import OptimizationTracker
from torch_motion_correction.patch_utils import ImagePatchIterator
from torch_motion_correction.types import (
    FourierFilterConfig,
    OptimizationConfig,
    PatchSamplingConfig,
)
from torch_motion_correction.utils import normalize_image, prepare_patch_filters


def estimate_local_motion(
    image: torch.Tensor,  # (t, H, W)
    pixel_spacing: float,  # Angstroms
    deformation_field_resolution: tuple[int, int, int],  # (nt, nh, nw)
    patch_sampling: PatchSamplingConfig,
    initial_deformation_field: DeformationField | None = None,
    fourier_filter: FourierFilterConfig | None = None,
    optimization: OptimizationConfig | None = None,
    device: torch.device | None = None,
    trajectory_kwargs: dict | None = None,
) -> tuple[DeformationField, OptimizationTracker]:
    """Estimate local motion using a gradient-based deformation field optimization.

    Parameters
    ----------
    image: torch.Tensor
        (t, H, W) image to estimate motion from where t is the number of frames,
        H is the height, and W is the width.
    pixel_spacing: float
        Pixel spacing in Angstroms.
    deformation_field_resolution: tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw) where nt is the number of
        time points, nh is the number of control points in height, and nw is the
        number of control points in width.
    patch_sampling: PatchSamplingConfig
        Patch extraction configuration, including patch shape and overlap fraction.
    initial_deformation_field: DeformationField | None
        Initial deformation field to start from. If None, initializes to zero shifts.
    fourier_filter: FourierFilterConfig | None
        Fourier-space filtering parameters (b_factor and frequency_range).
        Defaults to ``FourierFilterConfig()`` when None.
    optimization: OptimizationConfig | None
        Optimization hyper-parameters (n_iterations, optimizer, loss, grid type).
        Defaults to ``OptimizationConfig()`` when None.
    device: torch.device | None
        Device to perform computation on. If None, uses the device of the input image.
    trajectory_kwargs: dict | None
        Additional keyword arguments for the trajectory tracking. If None, uses
        defaults.

    Returns
    -------
    tuple[DeformationField, OptimizationTracker]
        The estimated deformation field and an OptimizationTracker containing the
        optimization history.
    """
    if fourier_filter is None:
        fourier_filter = FourierFilterConfig()
    if optimization is None:
        optimization = OptimizationConfig()

    # Deconstruct config objects
    patch_shape = patch_sampling.patch_shape
    ph, pw = patch_shape
    # b_factor = fourier_filter.b_factor
    # frequency_range = fourier_filter.frequency_range
    n_iterations = optimization.n_iterations
    optimizer_type = optimization.optimizer_type
    loss_type = optimization.loss_type
    grid_type = optimization.grid_type
    optimizer_kwargs = optimization.optimizer_kwargs

    device = device if device is not None else image.device
    image = image.to(device)
    t, _h, _w = image.shape

    trajectory_kwargs = trajectory_kwargs if trajectory_kwargs is not None else {}
    trajectory_kwargs.setdefault("sample_every_n_steps", 1)
    trajectory_kwargs.setdefault("total_steps", n_iterations)
    trajectory = OptimizationTracker(**trajectory_kwargs)

    # Normalize image based on stats from central 50% of image
    image = normalize_image(image)

    # Create the patch grid via PatchSamplingConfig
    image_patch_iterator = patch_sampling.get_patch_iterator(image=image, device=device)

    new_deformation_field, deformation_field = DeformationField.from_initial_field(
        resolution=deformation_field_resolution,
        initial_field=initial_deformation_field,
        grid_type=grid_type,
        device=device,
    )

    # Reusable masks and Fourier filters
    circle_mask, b_factor_envelope, bandpass_filter = prepare_patch_filters(
        shape=patch_shape,
        pixel_spacing=pixel_spacing,
        fourier_filter=fourier_filter,
        mask_smoothing_fraction=1.0,  # optimizer historically uses radius == smoothing
        device=device,
    )

    motion_optimizer = _setup_optimizer(
        optimizer_type=optimizer_type,
        parameters=list(new_deformation_field.parameters()),
        **(optimizer_kwargs if optimizer_kwargs is not None else {}),
    )

    # For LBFGS, optionally subsample patches per closure to reduce memory
    lbfgs_patch_subsample = None
    use_checkpointing = True
    if optimizer_type.lower() == "lbfgs":
        lbfgs_patch_subsample = (
            optimizer_kwargs.get("lbfgs_patch_subsample", None)
            if optimizer_kwargs
            else None
        )
        use_checkpointing = (
            optimizer_kwargs.get("use_gradient_checkpointing", True)
            if optimizer_kwargs
            else True
        )

    # Helper inner function to to have all other arguments fixed
    def process_batch(
        patch_batch: torch.Tensor,
        patch_batch_centers: torch.Tensor,
    ) -> torch.Tensor:
        return _process_patch_batch(
            patch_batch=patch_batch,
            patch_batch_centers=patch_batch_centers,
            circle_mask=circle_mask,
            b_factor_envelope=b_factor_envelope,
            bandpass=bandpass_filter,
            base_deformation_field=deformation_field,
            new_deformation_field=new_deformation_field,
            pixel_spacing=pixel_spacing,
            ph=ph,
            pw=pw,
            loss_type=loss_type,
            t=t,
        )

    # "Training" loop going over all patches n_iterations times
    pbar = tqdm.tqdm(range(n_iterations))
    for iter_idx in pbar:
        if optimizer_type.lower() == "lbfgs":
            avg_loss = _run_lbfgs_step(
                motion_optimizer=motion_optimizer,
                image_patch_iterator=image_patch_iterator,
                process_batch_fn=process_batch,
                lbfgs_patch_subsample=lbfgs_patch_subsample,
                use_checkpointing=use_checkpointing,
                device=device,
            )
        else:
            avg_loss = _run_standard_step(
                motion_optimizer=motion_optimizer,
                image_patch_iterator=image_patch_iterator,
                process_batch_fn=process_batch,
            )

        pbar.set_postfix({"avg_batch_loss": f"{avg_loss:.6f}"})
        if trajectory.sample_this_step(iter_idx):
            trajectory.add_checkpoint(
                deformation_field=new_deformation_field.data.detach(),
                loss=avg_loss,
                step=iter_idx,
            )

    # Return final deformation field
    final_data = new_deformation_field.data.detach() + deformation_field.data
    average_shift = torch.mean(final_data)
    final_data = final_data - average_shift

    result = DeformationField(data=final_data, grid_type=grid_type)

    return result, trajectory


def _process_patch_batch(
    patch_batch: torch.Tensor,
    patch_batch_centers: torch.Tensor,
    circle_mask: torch.Tensor,
    b_factor_envelope: torch.Tensor,
    bandpass: torch.Tensor,
    base_deformation_field: DeformationField,
    new_deformation_field: DeformationField,
    pixel_spacing: float,
    ph: int,
    pw: int,
    loss_type: str,
    t: int,
) -> torch.Tensor:
    """Apply mask, FFT, shift-predict, assemble reference, and compute loss.

    Parameters
    ----------
    patch_batch : torch.Tensor
        (b, t, ph, pw) real-space image patches.
    patch_batch_centers : torch.Tensor
        (b, t, 3) normalized (t, y, x) coordinates of each patch center.
    circle_mask : torch.Tensor
        (ph, pw) real-space circular apodisation mask.
    b_factor_envelope : torch.Tensor
        (ph, pw//2+1) rFFT-space B-factor envelope.
    bandpass : torch.Tensor
        (ph, pw//2+1) rFFT-space bandpass filter.
    base_deformation_field : DeformationField
        Frozen initial deformation field.
    new_deformation_field : DeformationField
        Optimisable deformation field increment.
    pixel_spacing : float
        Pixel spacing in Angstroms.
    ph : int
        Patch height in pixels.
    pw : int
        Patch width in pixels.
    loss_type : str
        Loss function name ("mse", "ncc", or "cc").
    t : int
        Number of frames (used to scale the reference mean).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    patch_batch_fft = torch.fft.rfftn(patch_batch * circle_mask, dim=(-2, -1))

    shifted_patches, _ = _compute_shifted_patches_and_shifts(
        initial_deformation_field=base_deformation_field,
        new_deformation_field=new_deformation_field,
        patch_batch=patch_batch_fft,
        patch_batch_centers=patch_batch_centers,
        pixel_spacing=pixel_spacing,
        ph=ph,
        pw=pw,
        b_factor_envelope=b_factor_envelope,
        bandpass=bandpass,
    )

    total_sum = torch.sum(shifted_patches, dim=1, keepdim=True)
    if t > 1:
        reference_patches = (total_sum - shifted_patches) / (t - 1)
    else:
        reference_patches = shifted_patches

    return _compute_loss(
        shifted_patches, reference_patches, ph, pw, loss_type=loss_type
    )


def _run_lbfgs_step(
    motion_optimizer: torch.optim.LBFGS,
    image_patch_iterator: ImagePatchIterator,
    process_batch_fn: Callable,
    lbfgs_patch_subsample: int | None,
    use_checkpointing: bool,
    device: torch.device,
) -> float:
    """Execute one LBFGS step over all (or a subset of) patches.

    Parameters
    ----------
    motion_optimizer : torch.optim.LBFGS
        The LBFGS optimizer.
    image_patch_iterator : ImagePatchIterator
        Iterator that yields (patch_batch, patch_centers) mini-batches.
    process_batch_fn : Callable
        Partially-applied ``_process_patch_batch`` with all frozen args bound.
    lbfgs_patch_subsample : int | None
        If set, only the first ``lbfgs_patch_subsample`` batches are used per
        closure call to reduce memory usage.
    use_checkpointing : bool
        Whether to apply gradient checkpointing inside the closure.
    device : torch.device
        Device used to construct the zero-loss fallback tensor.

    Returns
    -------
    float
        Average per-batch loss for this step.
    """

    def closure() -> torch.Tensor:
        motion_optimizer.zero_grad()
        weighted_loss_sum = None
        n_batches = 0
        iterator = image_patch_iterator.get_iterator(batch_size=1, randomized=True)
        for idx, (patch_batch, patch_batch_centers) in enumerate(iterator):
            if lbfgs_patch_subsample is not None and idx >= lbfgs_patch_subsample:
                break
            if use_checkpointing:
                batch_loss = checkpoint.checkpoint(
                    process_batch_fn,
                    patch_batch,
                    patch_batch_centers,
                    use_reentrant=False,
                )
            else:
                batch_loss = process_batch_fn(patch_batch, patch_batch_centers)

            weighted_loss_sum = (
                batch_loss
                if weighted_loss_sum is None
                else weighted_loss_sum + batch_loss
            )
            n_batches += 1

        if n_batches == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        assert weighted_loss_sum is not None
        avg_loss = weighted_loss_sum / n_batches
        avg_loss.backward()
        return avg_loss

    avg_loss_tensor = motion_optimizer.step(closure)
    return (
        float(avg_loss_tensor.detach())
        if isinstance(avg_loss_tensor, torch.Tensor)
        else float(avg_loss_tensor)
    )


def _run_standard_step(
    motion_optimizer: torch.optim.Optimizer,
    image_patch_iterator: ImagePatchIterator,
    process_batch_fn: Callable,
) -> float:
    """Execute one gradient-accumulation step for Adam/SGD/RMSprop.

    Parameters
    ----------
    motion_optimizer : torch.optim.Optimizer
        The optimizer (Adam, SGD, or RMSprop).
    image_patch_iterator : ImagePatchIterator
        Iterator that yields (patch_batch, patch_centers) mini-batches.
    process_batch_fn : Callable
        Partially-applied ``_process_patch_batch`` with all frozen args bound.

    Returns
    -------
    float
        Average per-batch loss for this step.
    """
    patch_iter = image_patch_iterator.get_iterator(batch_size=8)  # TODO: expose
    total_loss = 0.0
    n_batches = 0
    for patch_batch, patch_batch_centers in patch_iter:
        loss = process_batch_fn(patch_batch, patch_batch_centers)
        loss.backward()
        total_loss += loss.item()
        n_batches += 1
    motion_optimizer.step()
    motion_optimizer.zero_grad()
    return total_loss / n_batches if n_batches > 0 else 0.0


def _compute_shifted_patches_and_shifts(
    initial_deformation_field: DeformationField,
    new_deformation_field: DeformationField,
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
    initial_deformation_field : DeformationField
        The deformation field model to predict shifts.
    new_deformation_field : DeformationField
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
        + initial_deformation_field(patch_batch_centers).detach()
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
