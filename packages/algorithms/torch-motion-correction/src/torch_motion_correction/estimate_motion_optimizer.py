"""Estimate local motion using a deformation field."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import einops
import torch
import torch.utils.checkpoint as checkpoint
import tqdm
from torch_fourier_rescale import fourier_rescale_2d

from torch_motion_correction.deformation_field import (
    DeformationField,
    _subtract_mean_shift,
)
from torch_motion_correction.optimization_state import OptimizationTracker
from torch_motion_correction.types import (
    FourierFilterConfig,
    OptimizationConfig,
    PatchSamplingConfig,
)
from torch_motion_correction.utils import prepare_patch_filters


@dataclass
class _PrecomputedPatchState:
    """Iteration-invariant state prepared once before the optimization loop."""

    new_deformation_field: DeformationField
    base_deformation_field: DeformationField
    cached_fft: torch.Tensor  # (n_patches, t, ph, pw//2+1), bandpass and B applied
    centers_norm: torch.Tensor  # (t, n_patches, 3)
    base_shifts_cache: torch.Tensor  # (t, n_patches, 2), detached
    pixel_spacing: float  # possibly Fourier-cropped
    ph: int  # possibly Fourier-cropped
    pw: int  # possibly Fourier-cropped


# Patches per extract/mask/crop/rfft chunk when caching; small to bound peak memory
_PRECOMPUTE_CHUNK_SIZE = 2


def _prepare_patch_state(
    image: torch.Tensor,  # (t, H, W)
    pixel_spacing: float,
    deformation_field_resolution: tuple[int, int, int],
    patch_sampling: PatchSamplingConfig,
    fourier_filter: FourierFilterConfig,
    grid_type: str,
    initial_deformation_field: DeformationField | None,
    device: torch.device,
) -> _PrecomputedPatchState:
    """Extract, mask, Fourier-crop, and cache all patches once, in small chunks.

    Notes
    -----
    Move is static across entire optimization which means patch extraction plus any
    masking and filtering operations can happen once rather than within optimization
    loop. Additionally, band-pass filtering zeros many pixels in Fourier space. Can crop
    down to new resolution (``fourier_filter.frequency_range[1]``) without losing any
    information to dramatically speed up per-iteration work. The bandpass and B-factor
    are applied to the cached patches here too: they commute with the phase shifts
    applied in the loop.

    Parameters
    ----------
    image : torch.Tensor
        (t, H, W) movie. May reside on a different device than ``device`` (e.g.
        large super-res movie stored on CPU). Patches are extracted onto the image's
        own device and moved to ``device`` one chunk at a time. Normalized
        internally, per patch chunk.
    pixel_spacing : float
        Pixel spacing in Angstroms.
    deformation_field_resolution : tuple[int, int, int]
        Resolution of the deformation field (nt, nh, nw).
    patch_sampling : PatchSamplingConfig
        Patch extraction configuration.
    fourier_filter : FourierFilterConfig
        Fourier-space filtering parameters (b_factor and frequency_range).
    grid_type : str
        Cubic spline interpolation type for the deformation field.
    initial_deformation_field : DeformationField | None
        Initial deformation field to start from. If None, initializes to zero.
    device : torch.device
        Device to perform computation on.

    Returns
    -------
    _PrecomputedPatchState
        Bundle of precomputed, iteration-invariant tensors and the two
        deformation fields (frozen base + optimizable increment).
    """
    patch_shape = patch_sampling.patch_shape
    ph, pw = patch_shape

    _, h, w = image.shape
    hl, hu = int(0.25 * h), int(0.75 * h)
    wl, wu = int(0.25 * w), int(0.75 * w)
    norm_std, norm_mean = torch.std_mean(image[:, hl:hu, wl:wu], dim=(-3, -2, -1))
    norm_std = norm_std.to(device)
    norm_mean = norm_mean.to(device)

    image_patch_iterator = patch_sampling.get_patch_iterator(image=image, device=device)

    new_deformation_field, base_deformation_field = DeformationField.from_initial_field(
        resolution=deformation_field_resolution,
        initial_field=initial_deformation_field,
        grid_type=grid_type,
        device=device,
    )

    # Real-space apodization mask at full (uncropped) resolution
    circle_mask, _, _ = prepare_patch_filters(
        shape=patch_shape,
        pixel_spacing=pixel_spacing,
        fourier_filter=fourier_filter,
        mask_smoothing_fraction=1.0,  # optimizer historically uses radius == smoothing
        device=device,
    )

    crop_pixel_spacing = fourier_filter.frequency_range[1] / 2
    will_crop = crop_pixel_spacing > pixel_spacing  # Only if crop would reduce size
    if will_crop:
        ph_eff = round(ph * pixel_spacing / crop_pixel_spacing)
        pw_eff = round(pw * pixel_spacing / crop_pixel_spacing)
        pixel_spacing_eff = pixel_spacing * (ph / ph_eff)
    else:
        ph_eff, pw_eff = ph, pw
        pixel_spacing_eff = pixel_spacing

    fft_chunks = []
    centers_chunks = []
    chunk_iter = image_patch_iterator.get_iterator(
        batch_size=_PRECOMPUTE_CHUNK_SIZE, randomized=False
    )
    for patch_chunk, centers_chunk in chunk_iter:
        patch_chunk = patch_chunk.to(device)
        patch_chunk = (patch_chunk - norm_mean) / norm_std
        masked_chunk = patch_chunk * circle_mask
        if will_crop:
            cropped_chunk, _ = fourier_rescale_2d(
                masked_chunk, target_shape=(ph_eff, pw_eff)
            )
        else:
            cropped_chunk = masked_chunk
        fft_chunks.append(torch.fft.rfftn(cropped_chunk, dim=(-2, -1)))
        centers_chunks.append(centers_chunk)

    cached_fft = torch.cat(fft_chunks, dim=0)  # (n_patches, t, ph_eff, pw_eff//2+1)
    all_centers_norm = torch.cat(centers_chunks, dim=1)  # (t, n_patches, 3)

    # Fourier filters, rebuilt at the (possibly cropped) resolution/spacing.
    _, b_factor_envelope, bandpass_filter = prepare_patch_filters(
        shape=(ph_eff, pw_eff),
        pixel_spacing=pixel_spacing_eff,
        fourier_filter=fourier_filter,
        mask_smoothing_fraction=1.0,
        device=device,
    )
    cached_fft *= bandpass_filter * b_factor_envelope

    with torch.no_grad():
        base_shifts_cache = base_deformation_field(all_centers_norm).detach()

    return _PrecomputedPatchState(
        new_deformation_field=new_deformation_field,
        base_deformation_field=base_deformation_field,
        cached_fft=cached_fft,
        centers_norm=all_centers_norm,
        base_shifts_cache=base_shifts_cache,
        pixel_spacing=pixel_spacing_eff,
        ph=ph_eff,
        pw=pw_eff,
    )


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
        H is the height, and W is the width. May reside on a different device than
        ``device`` (e.g. large super-res movie stored on CPU).
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
        Optimization hyper-parameters (max_iterations, optimizer, loss, grid type).
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
    max_iterations = optimization.max_iterations
    optimizer_type = optimization.optimizer_type
    loss_type = optimization.loss_type
    grid_type = optimization.grid_type
    optimizer_kwargs = optimization.optimizer_kwargs

    device = device if device is not None else image.device
    t, _h, _w = image.shape

    trajectory_kwargs = trajectory_kwargs if trajectory_kwargs is not None else {}
    trajectory_kwargs.setdefault("sample_every_n_steps", 1)
    trajectory_kwargs.setdefault("total_steps", max_iterations)
    trajectory = OptimizationTracker(**trajectory_kwargs)

    # NOTE: All patch extraction, masking, Fourier-cropping, and filter/cache setup is
    # iteration-invariant, so compute once into data object
    state = _prepare_patch_state(
        image=image,
        pixel_spacing=pixel_spacing,
        deformation_field_resolution=deformation_field_resolution,
        patch_sampling=patch_sampling,
        fourier_filter=fourier_filter,
        grid_type=grid_type,
        initial_deformation_field=initial_deformation_field,
        device=device,
    )
    new_deformation_field = state.new_deformation_field
    deformation_field = state.base_deformation_field

    motion_optimizer = _setup_optimizer(
        optimizer_type=optimizer_type,
        parameters=list(new_deformation_field.parameters()),
        **(optimizer_kwargs if optimizer_kwargs is not None else {}),
    )

    cached_fft = state.cached_fft
    centers_norm = state.centers_norm
    base_shifts_cache = state.base_shifts_cache

    # LBFGS options: optionally fit a fixed subset of patches to reduce memory
    use_checkpointing = False
    if optimizer_type.lower() == "lbfgs":
        kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        use_checkpointing = kwargs.get("use_gradient_checkpointing", True)
        n_subsample = kwargs.get("lbfgs_patch_subsample", None)
        if n_subsample is not None and n_subsample < cached_fft.shape[0]:
            # The same patches every step, as the line search and curvature history
            # need one objective; spread evenly over the image
            idx = torch.linspace(0, cached_fft.shape[0] - 1, steps=n_subsample)
            idx = idx.round().long().unique().to(cached_fft.device)
            cached_fft = cached_fft[idx]
            centers_norm = centers_norm[:, idx]
            base_shifts_cache = base_shifts_cache[:, idx]

    batch_loss_fn = (
        torch.compile(_process_patch_batch)
        if optimization.compile_loss
        else _process_patch_batch
    )

    # Helper inner function to to have all other arguments fixed
    def process_batch(
        fft_batch: torch.Tensor,
        patch_batch_centers: torch.Tensor,
        base_shifts_batch: torch.Tensor,
    ) -> torch.Tensor:
        return batch_loss_fn(
            fft_batch=fft_batch,
            patch_batch_centers=patch_batch_centers,
            base_shifts_batch=base_shifts_batch,
            new_deformation_field=new_deformation_field,
            pixel_spacing=state.pixel_spacing,
            ph=state.ph,
            pw=state.pw,
            loss_type=loss_type,
            t=t,
        )

    def loss_and_backward() -> torch.Tensor:
        return _accumulate_loss_and_gradients(
            process_batch_fn=process_batch,
            cached_fft=cached_fft,
            centers_norm=centers_norm,
            base_shifts_cache=base_shifts_cache,
            patch_batch_size=optimization.patch_batch_size,
            use_checkpointing=use_checkpointing,
        )

    early_stopper = optimization.build_early_stopper()

    pbar = tqdm.tqdm(range(max_iterations))
    for iter_idx in pbar:
        if optimizer_type.lower() == "lbfgs":
            avg_loss = _run_lbfgs_step(motion_optimizer, loss_and_backward)
        else:
            avg_loss = _run_standard_step(motion_optimizer, loss_and_backward)

        pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})
        if trajectory.sample_this_step(iter_idx):
            # Record the field this step stands for: base plus increment, mean removed
            trajectory.add_checkpoint(
                deformation_field=_subtract_mean_shift(
                    new_deformation_field.data.detach() + deformation_field.data
                ),
                loss=avg_loss,
                step=iter_idx,
            )

        # Break loop if early stopping criterion is met
        if early_stopper is not None and early_stopper(avg_loss):
            pbar.write(f"Early stopping at iter {iter_idx}. avg_loss={avg_loss:.6f}")
            break

    # Return final deformation field
    final_data = new_deformation_field.data.detach() + deformation_field.data.detach()
    final_data = _subtract_mean_shift(final_data)

    result = DeformationField(data=final_data, grid_type=grid_type)

    return result, trajectory


def _phase_shift_rfft_2d(
    dft: torch.Tensor, shifts: torch.Tensor, image_shape: tuple[int, int]
) -> torch.Tensor:
    """Shift 2D images by multiplying their rFFTs with separable phase ramps.

    Gives the same result as ``torch_fourier_shift.fourier_shift_dft_2d`` with
    ``rfft=True, fftshifted=False``, without building a full 2D angle grid per image.

    Parameters
    ----------
    dft : torch.Tensor
        (..., h, w//2+1) rFFTs of images with shape ``image_shape``.
    shifts : torch.Tensor
        (..., 2) shifts in pixels along (h, w).
    image_shape : tuple[int, int]
        (h, w) of the images.

    Returns
    -------
    torch.Tensor
        (..., h, w//2+1) rFFTs of the shifted images.
    """
    h, w = image_shape
    freq_y = torch.fft.fftfreq(h, device=dft.device)
    freq_x = torch.fft.rfftfreq(w, device=dft.device)
    phase_y = torch.polar(
        torch.ones_like(freq_y), -2 * torch.pi * shifts[..., 0, None] * freq_y
    )
    phase_x = torch.polar(
        torch.ones_like(freq_x), -2 * torch.pi * shifts[..., 1, None] * freq_x
    )
    return dft * phase_y[..., :, None] * phase_x[..., None, :]


def _process_patch_batch(
    fft_batch: torch.Tensor,
    patch_batch_centers: torch.Tensor,
    base_shifts_batch: torch.Tensor,
    new_deformation_field: DeformationField,
    pixel_spacing: float,
    ph: int,
    pw: int,
    loss_type: str,
    t: int,
) -> torch.Tensor:
    """Shift-predict, assemble reference, and compute loss for a batch.

    Parameters
    ----------
    fft_batch : torch.Tensor
        (b, t, ph, pw//2+1) rFFT of masked (and possibly resolution-cropped) patches,
        bandpass and B-factor applied, precomputed once.
    patch_batch_centers : torch.Tensor
        (t, b, 3) normalized (t, y, x) coordinates of each patch center.
    base_shifts_batch : torch.Tensor
        (t, b, 2) frozen base deformation field shifts in Angstroms, precomputed once
        and detached.
    new_deformation_field : DeformationField
        Optimisable deformation field increment.
    pixel_spacing : float
        Pixel spacing in Angstroms (of the, possibly cropped, patches).
    ph : int
        Patch height in pixels (possibly cropped).
    pw : int
        Patch width in pixels (possibly cropped).
    loss_type : str
        Loss function name ("mse", "ncc", or "cc").
    t : int
        Number of frames (used to scale the reference mean).

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    predicted_shifts = -1 * (
        new_deformation_field(patch_batch_centers) + base_shifts_batch
    )
    predicted_shifts = einops.rearrange(predicted_shifts, "t b yx -> b t yx")
    predicted_shifts_px = predicted_shifts / pixel_spacing

    shifted_patches = _phase_shift_rfft_2d(
        dft=fft_batch, shifts=predicted_shifts_px, image_shape=(ph, pw)
    )  # (b, t, ph, pw//2 + 1)

    total_sum = torch.sum(shifted_patches, dim=1, keepdim=True)
    if t > 1:
        reference_patches = (total_sum - shifted_patches) / (t - 1)
    else:
        reference_patches = shifted_patches

    return _compute_loss(
        shifted_patches, reference_patches, ph, pw, loss_type=loss_type, t=t
    )


def _accumulate_loss_and_gradients(
    process_batch_fn: Callable,
    cached_fft: torch.Tensor,
    centers_norm: torch.Tensor,
    base_shifts_cache: torch.Tensor,
    patch_batch_size: int | None,
    use_checkpointing: bool = False,
) -> torch.Tensor:
    """Accumulate the gradient of the mean loss over all patches, batch by batch.

    Each batch's mean loss is weighted by its share of the patches, so every patch
    counts the same whatever the batch size, including a short last batch.

    Parameters
    ----------
    process_batch_fn : Callable
        Partially-applied ``_process_patch_batch`` with all frozen args bound.
    cached_fft : torch.Tensor
        (n_patches, t, ph, pw//2+1) precomputed rFFT of masked patches.
    centers_norm : torch.Tensor
        (t, n_patches, 3) normalized patch center coordinates.
    base_shifts_cache : torch.Tensor
        (t, n_patches, 2) frozen base deformation field shifts, detached.
    patch_batch_size : int | None
        Number of patches per batch. None processes all patches in one batch.
    use_checkpointing : bool
        Whether to recompute each batch's activations in the backward pass instead
        of keeping them, to save memory. Default is False.

    Returns
    -------
    torch.Tensor
        Detached scalar mean loss over all patches.
    """
    n_patches = cached_fft.shape[0]
    batch_size = n_patches if patch_batch_size is None else patch_batch_size
    total_loss = torch.zeros((), device=cached_fft.device)
    for start in range(0, n_patches, batch_size):
        stop = min(start + batch_size, n_patches)
        batch = (
            cached_fft[start:stop],
            centers_norm[:, start:stop],
            base_shifts_cache[:, start:stop],
        )
        if use_checkpointing:
            loss = checkpoint.checkpoint(process_batch_fn, *batch, use_reentrant=False)
        else:
            loss = process_batch_fn(*batch)
        loss = loss * ((stop - start) / n_patches)
        loss.backward()
        total_loss += loss.detach()
    return total_loss


def _run_lbfgs_step(
    motion_optimizer: torch.optim.LBFGS,
    loss_and_backward: Callable[[], torch.Tensor],
) -> float:
    """Execute one LBFGS step.

    Parameters
    ----------
    motion_optimizer : torch.optim.LBFGS
        The LBFGS optimizer.
    loss_and_backward : Callable[[], torch.Tensor]
        Computes the mean loss over the patches, accumulates its gradient, and
        returns the detached loss.

    Returns
    -------
    float
        Mean loss over the patches at the start of this step.
    """

    def closure() -> torch.Tensor:
        motion_optimizer.zero_grad()
        return loss_and_backward()

    return float(motion_optimizer.step(closure))


def _run_standard_step(
    motion_optimizer: torch.optim.Optimizer,
    loss_and_backward: Callable[[], torch.Tensor],
) -> float:
    """Execute one gradient step for Adam/SGD/RMSprop.

    Parameters
    ----------
    motion_optimizer : torch.optim.Optimizer
        The optimizer (Adam, SGD, or RMSprop).
    loss_and_backward : Callable[[], torch.Tensor]
        Computes the mean loss over the patches, accumulates its gradient, and
        returns the detached loss.

    Returns
    -------
    float
        Mean loss over the patches at the start of this step.
    """
    loss = loss_and_backward()
    motion_optimizer.step()
    motion_optimizer.zero_grad()
    return float(loss)


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
    t: int | None = None,
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
    t : int, optional
        Number of frames. When provided (and > 1) for "ncc"/"cc", real-space reference
        is derived from the real-space shifted patches as their leave-one-out mean. Only
        valid when ``reference_patches`` constructed via
        ``(sum(shifted_patches, dim=1) - shifted_patches) / (t - 1)``. If ``t`` is None,
        falls back to irfft-ing ``reference_patches`` directly.
    """
    if loss_type == "mse":
        difference = shifted_patches - reference_patches
        squared_magnitude = difference.real.square() + difference.imag.square()
        return torch.mean(squared_magnitude) / (ph * pw)
    elif loss_type in ("ncc", "cc"):
        # Inputs are in rFFT space with shapes:
        # shifted_patches: (b, t, ph, pw//2 + 1)
        # reference_patches: (b, t, ph, pw//2 + 1)
        shifted_real = torch.fft.irfftn(shifted_patches, s=(ph, pw), dim=(-2, -1))
        if t is not None and t > 1:
            sum_real = shifted_real.sum(dim=1, keepdim=True)
            reference_real = (sum_real - shifted_real) / (t - 1)
        else:
            reference_real = torch.fft.irfftn(
                reference_patches, s=(ph, pw), dim=(-2, -1)
            )

        if loss_type == "ncc":
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
        else:
            # Compute unnormalized cross-correlation over spatial dims
            # (b, t, ph, pw) * (b, t, ph, pw) → (b, t)
            cc = (shifted_real * reference_real).sum(dim=(-2, -1))
            # Optionally: mean over batch and time; negate to make it a loss
            return -cc.mean()
