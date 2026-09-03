"""Estimate local motion using a deformation field."""

import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import einops
import torch
import torch.utils.checkpoint as checkpoint
import tqdm
from torch_fourier_rescale import fourier_rescale_2d
from torch_fourier_shift import fourier_shift_dft_2d

from torch_motion_correction.deformation_field import DeformationField
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
    cached_fft: torch.Tensor  # (n_patches, t, ph, pw//2+1)
    centers_norm: torch.Tensor  # (t, n_patches, 3)
    base_shifts_cache: torch.Tensor  # (t, n_patches, 2), detached
    b_factor_envelope: torch.Tensor | None  # (ph, pw//2+1)
    bandpass_filter: torch.Tensor | None  # (ph, pw//2+1)
    pixel_spacing: float  # possibly Fourier-cropped
    ph: int  # possibly Fourier-cropped
    pw: int  # possibly Fourier-cropped


_PRECOMPUTE_CHUNK_SIZE = 8  # num patches processed per extract/mask/crop/rfft chunk


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
    information to dramatically speed up per-iteration work.

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

    _t, h, w = image.shape
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

    with torch.no_grad():
        base_shifts_cache = base_deformation_field(all_centers_norm).detach()

    return _PrecomputedPatchState(
        new_deformation_field=new_deformation_field,
        base_deformation_field=base_deformation_field,
        cached_fft=cached_fft,
        centers_norm=all_centers_norm,
        base_shifts_cache=base_shifts_cache,
        b_factor_envelope=b_factor_envelope,
        bandpass_filter=bandpass_filter,
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
        fft_batch: torch.Tensor,
        patch_batch_centers: torch.Tensor,
        base_shifts_batch: torch.Tensor,
    ) -> torch.Tensor:
        return _process_patch_batch(
            fft_batch=fft_batch,
            patch_batch_centers=patch_batch_centers,
            base_shifts_batch=base_shifts_batch,
            b_factor_envelope=state.b_factor_envelope,
            bandpass=state.bandpass_filter,
            new_deformation_field=new_deformation_field,
            pixel_spacing=state.pixel_spacing,
            ph=state.ph,
            pw=state.pw,
            loss_type=loss_type,
            t=t,
        )

    early_stopper = optimization.build_early_stopper()

    pbar = tqdm.tqdm(range(max_iterations))
    for iter_idx in pbar:
        if optimizer_type.lower() == "lbfgs":
            avg_loss = _run_lbfgs_step(
                motion_optimizer=motion_optimizer,
                cached_fft=state.cached_fft,
                centers_norm=state.centers_norm,
                base_shifts_cache=state.base_shifts_cache,
                process_batch_fn=process_batch,
                lbfgs_patch_subsample=lbfgs_patch_subsample,
                use_checkpointing=use_checkpointing,
                device=device,
            )
        else:
            avg_loss = _run_standard_step(
                motion_optimizer=motion_optimizer,
                cached_fft=state.cached_fft,
                centers_norm=state.centers_norm,
                base_shifts_cache=state.base_shifts_cache,
                process_batch_fn=process_batch,
            )

        pbar.set_postfix({"avg_batch_loss": f"{avg_loss:.6f}"})
        if trajectory.sample_this_step(iter_idx):
            trajectory.add_checkpoint(
                deformation_field=new_deformation_field.data.detach(),
                loss=avg_loss,
                step=iter_idx,
            )

        # Break loop if early stopping criterion is met
        if early_stopper is not None and early_stopper(avg_loss):
            pbar.write(f"Early stopping at iter {iter_idx}. avg_loss={avg_loss:.6f}")
            break

    # Return final deformation field
    final_data = new_deformation_field.data.detach() + deformation_field.data
    average_shift = torch.mean(final_data)
    final_data = final_data - average_shift

    result = DeformationField(data=final_data, grid_type=grid_type)

    return result, trajectory


def _process_patch_batch(
    fft_batch: torch.Tensor,
    patch_batch_centers: torch.Tensor,
    base_shifts_batch: torch.Tensor,
    b_factor_envelope: torch.Tensor,
    bandpass: torch.Tensor,
    new_deformation_field: DeformationField,
    pixel_spacing: float,
    ph: int,
    pw: int,
    loss_type: str,
    t: int,
) -> torch.Tensor:
    """Shift-predict, filter, assemble reference, and compute loss for a batch.

    Parameters
    ----------
    fft_batch : torch.Tensor
        (b, t, ph, pw//2+1) rFFT of masked (and possibly resolution-cropped) patches,
        precomputed once.
    patch_batch_centers : torch.Tensor
        (t, b, 3) normalized (t, y, x) coordinates of each patch center.
    base_shifts_batch : torch.Tensor
        (t, b, 2) frozen base deformation field shifts in Angstroms, precomputed once
        and detached.
    b_factor_envelope : torch.Tensor
        (ph, pw//2+1) rFFT-space B-factor envelope.
    bandpass : torch.Tensor
        (ph, pw//2+1) rFFT-space bandpass filter.
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
    predicted_shifts = einops.rearrange(predicted_shifts, "b t yx -> t b yx")
    predicted_shifts_px = predicted_shifts / pixel_spacing

    shifted_patches = fourier_shift_dft_2d(
        dft=fft_batch,
        image_shape=(ph, pw),
        shifts=predicted_shifts_px,
        rfft=True,
        fftshifted=False,
    )  # (b, t, ph, pw//2 + 1)

    if bandpass is not None:
        shifted_patches = shifted_patches * bandpass
    if b_factor_envelope is not None:
        shifted_patches = shifted_patches * b_factor_envelope

    total_sum = torch.sum(shifted_patches, dim=1, keepdim=True)
    if t > 1:
        reference_patches = (total_sum - shifted_patches) / (t - 1)
    else:
        reference_patches = shifted_patches

    return _compute_loss(
        shifted_patches, reference_patches, ph, pw, loss_type=loss_type, t=t
    )


def _iterate_cached_batches(
    cached_fft: torch.Tensor,
    centers_norm: torch.Tensor,
    base_shifts_cache: torch.Tensor,
    batch_size: int,
    randomized: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield mini-batches from precomputed, cached per-patch tensors.

    Parameters
    ----------
    cached_fft : torch.Tensor
        (n_patches, t, ph, pw//2+1) rFFT of masked (and possibly resolution-cropped)
        patches.
    centers_norm : torch.Tensor
        (t, n_patches, 3) normalized (t, y, x) patch center coordinates.
    base_shifts_cache : torch.Tensor
        (t, n_patches, 2) frozen base deformation field shifts, detached.
    batch_size : int
        Number of patches per mini-batch.
    randomized : bool
        Whether to shuffle patch order. Default is True.

    Yields
    ------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (fft_batch, centers_batch, base_shifts_batch) with shapes (b, t, ph, pw//2+1),
        (t, b, 3), and (t, b, 2) respectively.
    """
    n_patches = cached_fft.shape[0]
    indices = list(range(n_patches))
    if randomized:
        random.shuffle(indices)
    for i in range(0, n_patches, batch_size):
        idx = indices[i : i + batch_size]
        yield cached_fft[idx], centers_norm[:, idx], base_shifts_cache[:, idx]


def _run_lbfgs_step(
    motion_optimizer: torch.optim.LBFGS,
    cached_fft: torch.Tensor,
    centers_norm: torch.Tensor,
    base_shifts_cache: torch.Tensor,
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
    cached_fft : torch.Tensor
        (n_patches, t, ph, pw//2+1) precomputed rFFT of masked patches.
    centers_norm : torch.Tensor
        (t, n_patches, 3) normalized patch center coordinates.
    base_shifts_cache : torch.Tensor
        (t, n_patches, 2) frozen base deformation field shifts, detached.
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
        iterator = _iterate_cached_batches(
            cached_fft, centers_norm, base_shifts_cache, batch_size=1, randomized=True
        )
        for idx, (fft_batch, centers_batch, base_shifts_batch) in enumerate(iterator):
            if lbfgs_patch_subsample is not None and idx >= lbfgs_patch_subsample:
                break
            if use_checkpointing:
                batch_loss = checkpoint.checkpoint(
                    process_batch_fn,
                    fft_batch,
                    centers_batch,
                    base_shifts_batch,
                    use_reentrant=False,
                )
            else:
                batch_loss = process_batch_fn(
                    fft_batch, centers_batch, base_shifts_batch
                )

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
    cached_fft: torch.Tensor,
    centers_norm: torch.Tensor,
    base_shifts_cache: torch.Tensor,
    process_batch_fn: Callable,
) -> float:
    """Execute one gradient-accumulation step for Adam/SGD/RMSprop.

    Parameters
    ----------
    motion_optimizer : torch.optim.Optimizer
        The optimizer (Adam, SGD, or RMSprop).
    cached_fft : torch.Tensor
        (n_patches, t, ph, pw//2+1) precomputed rFFT of masked patches.
    centers_norm : torch.Tensor
        (t, n_patches, 3) normalized patch center coordinates.
    base_shifts_cache : torch.Tensor
        (t, n_patches, 2) frozen base deformation field shifts, detached.
    process_batch_fn : Callable
        Partially-applied ``_process_patch_batch`` with all frozen args bound.

    Returns
    -------
    float
        Average per-batch loss for this step.
    """
    patch_iter = _iterate_cached_batches(
        cached_fft,
        centers_norm,
        base_shifts_cache,
        batch_size=8,  # TODO: expose
    )
    total_loss = 0.0
    n_batches = 0
    for fft_batch, centers_batch, base_shifts_batch in patch_iter:
        loss = process_batch_fn(fft_batch, centers_batch, base_shifts_batch)
        loss.backward()
        total_loss += loss.item()
        n_batches += 1
    motion_optimizer.step()
    motion_optimizer.zero_grad()
    return total_loss / n_batches if n_batches > 0 else 0.0


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
        return torch.mean((shifted_patches - reference_patches).abs() ** 2) / (ph * pw)
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
