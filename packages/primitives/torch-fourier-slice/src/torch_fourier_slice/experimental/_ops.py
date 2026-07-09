"""Non-differentiable kernel operations: forward projection and scatter.

The imperative layer between the public API / autograd Functions and the Mojo
kernels. Each function validates inputs, materialises contiguous float32
buffers on the input's device, dispatches to the CPU or GPU kernel, and returns
a tensor on that device.

The GPU kernels read and write the memory backing torch device tensors directly
-- no host round-trip. Every buffer (inputs *and* pre-zeroed outputs) is placed
on the compute device here, and its raw device address is handed to Mojo via the
``addrs`` tuple (see ``_gpu.device_address`` / ``projectors.mojo``). Output
buffers are allocated and zeroed on the device up front because the kernels
either leave radius-cut pixels untouched (forward) or atomically accumulate
(scatter / gradients).
"""

from __future__ import annotations

import math

import torch

from ._gpu import device_address, pre_launch_sync
from ._kernels import device_session, kernels
from ._validation import (
    interp_code,
    prep_poses,
    prep_rotations,
    prep_shifts_2d,
    prep_shifts_3d,
    validate_reconstruction,
)

# ---------------------------------------------------------------------------
# Forward projection (3D volume -> 2D central slices)
# ---------------------------------------------------------------------------


def forward_project(
    reconstruction: torch.Tensor,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    output_shape: tuple[int, int] | None,
    oversampling: float,
    fourier_radius_cutoff: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project an rfft volume to 2D central slices (CPU or GPU per device)."""
    device = reconstruction.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    reconstruction, bv, sidelength, _sh = validate_reconstruction(reconstruction)
    rot, shifts_2d_t, shifts_3d_t, proj_r, params = prep_poses(
        bv,
        sidelength,
        rotations,
        shifts_2d,
        output_shape,
        oversampling,
        fourier_radius_cutoff,
        interpolation,
        ewald_curvature,
        shifts_3d,
        device=tgt,
    )

    rec_r = torch.view_as_real(
        reconstruction.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    if use_gpu:
        bufs = (rec_r, rot, shifts_2d_t, shifts_3d_t, proj_r)
        addrs = tuple(device_address(t) for t in bufs)
        pre_launch_sync(device, *bufs)
        kernels().extract_central_slices_rfft_3d_gpu(
            device_session(), rec_r, rot, shifts_2d_t, shifts_3d_t, proj_r, params, addrs
        )
        return torch.view_as_complex(proj_r)
    kernels().extract_central_slices_rfft_3d(
        rec_r, rot, shifts_2d_t, shifts_3d_t, proj_r, params
    )
    return torch.view_as_complex(proj_r)


# ---------------------------------------------------------------------------
# Scatter (2D slices -> 3D volume); used for autograd adjoints and reconstruction
# ---------------------------------------------------------------------------


def run_scatter(
    slices: torch.Tensor,
    rotations: torch.Tensor,
    volume_shape: tuple[int, int, int],
    *,
    shifts_2d: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    oversampling: float = 1.0,
    fourier_radius_cutoff: float | None = None,
    interpolation: str = "linear",
    friedel_double: bool = False,
    skip_redundant: bool = False,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scatter complex ``slices`` ``(bv, bp, h, w)`` into a 3D rfft volume.

    ``volume_shape`` is ``(d, sidelength, sidelength_half)`` (no batch).
    ``shifts_2d`` applies the *conjugate* phase ramp (adjoint of the forward shift).
    ``friedel_double`` also inserts the Hermitian x=0 counterpart and
    ``skip_redundant`` skips the redundant half of the x=0 line (both for
    reconstruction). Returns ``(volume, weight_volume)`` -- complex
    ``(bv, *volume_shape)`` and real weights if ``weights`` was given else
    ``None`` -- on the input device.
    """
    device = slices.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if slices.dim() == 3:
        slices = slices.unsqueeze(0)
    if slices.dim() != 4 or not slices.is_complex():
        raise ValueError("slices must be complex (bv, bp, h, w)")
    bv, bp, h, w = slices.shape
    d, sidelength, sidelength_half = volume_shape

    rot, _bv_rot, bp_rot = prep_rotations(rotations, bv, tgt)
    if bp_rot != bp:
        raise ValueError("rotations pose count must match slices")
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)

    slices_r = torch.view_as_real(
        slices.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    vol_r = torch.zeros(bv, d, sidelength, sidelength_half, 2, dtype=torch.float32, device=tgt)
    wvol = torch.zeros(bv, d, sidelength, sidelength_half, dtype=torch.float32, device=tgt)
    if weights is not None:
        w_in = weights.to(device=tgt, dtype=torch.float32).contiguous()
    else:
        w_in = torch.zeros(bv, bp, h, w, dtype=torch.float32, device=tgt)  # unused

    radius = (
        sidelength / 2.0
        if fourier_radius_cutoff is None
        else float(fourier_radius_cutoff)
    )
    params = (
        float(oversampling),
        float(radius * radius),
        int(has_shifts_2d),
        int(weights is not None),
        int(bool(friedel_double)),
        int(bool(skip_redundant)),
        interp_code(interpolation),
        float(ewald_curvature),
        int(has_shifts_3d),
    )
    bufs = (slices_r, w_in, rot, shifts_2d_t, shifts_3d_t, vol_r, wvol)
    if use_gpu:
        addrs = tuple(device_address(t) for t in bufs)
        pre_launch_sync(device, *bufs)
        kernels().insert_central_slices_rfft_3d_gpu(
            device_session(), bufs, params, addrs
        )
    else:
        kernels().insert_central_slices_rfft_3d(bufs, params)

    out_vol = torch.view_as_complex(vol_r)
    out_weights = wvol if weights is not None else None
    return out_vol, out_weights


def run_pose_grad(
    volume: torch.Tensor,
    rotations: torch.Tensor,
    shifts_2d: torch.Tensor | None,
    pixels: torch.Tensor,
    *,
    oversampling: float,
    fourier_radius_cutoff: float | None,
    interpolation: str,
    backproject: bool,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Gradients w.r.t. rotations and shifts_2d (rotation/shift backward kernel).

    ``volume`` is the field whose spatial gradient is sampled (the rfft
    ``reconstruction`` for the forward projection, or the data-volume gradient
    ``grad_data_rec`` for the backprojection); ``pixels`` is the per-pixel
    cotangent (``grad_projections`` forward / ``projections`` backproject).
    Returns ``(grad_rotations (bv_rot, bp, 3, 3), grad_shifts (bv_shift_2d, bp, 2)
    or None)`` on the CPU.
    """
    device = pixels.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    bv, _d, _sidelength, _sh = volume.shape
    rot, bv_rot, bp = prep_rotations(rotations, bv, tgt)
    shifts_2d_t, has_shifts_2d = prep_shifts_2d(shifts_2d, bv, bp, tgt)
    shifts_3d_t, has_shifts_3d = prep_shifts_3d(shifts_3d, bv, bp, tgt)
    bv_shift_2d = shifts_2d_t.shape[0]
    bv_shift_3d = shifts_3d_t.shape[0]
    if pixels.dim() == 3:
        pixels = pixels.unsqueeze(0)
    proj_sidelength = int(pixels.shape[2])

    vol_r = torch.view_as_real(
        volume.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    pix_r = torch.view_as_real(
        pixels.to(device=tgt, dtype=torch.complex64).contiguous()
    ).contiguous()
    grad_rot = torch.zeros(bv_rot, bp, 3, 3, dtype=torch.float32, device=tgt)
    grad_shift = torch.zeros(bv_shift_2d, bp, 2, dtype=torch.float32, device=tgt)
    grad_shift_3d = torch.zeros(bv_shift_3d, bp, 3, dtype=torch.float32, device=tgt)

    radius = (
        proj_sidelength / 2.0
        if fourier_radius_cutoff is None
        else float(fourier_radius_cutoff)
    )
    params = (
        float(oversampling),
        float(radius * radius),
        int(has_shifts_2d),
        interp_code(interpolation),
        float(ewald_curvature),
        int(has_shifts_3d),
    )
    bufs = (
        vol_r,
        rot,
        shifts_2d_t,
        shifts_3d_t,
        pix_r,
        grad_rot,
        grad_shift,
        grad_shift_3d,
    )
    if backproject:
        fn = (
            kernels().insert_central_slices_rfft_3d_pose_grad_gpu
            if use_gpu
            else kernels().insert_central_slices_rfft_3d_pose_grad
        )
    else:
        fn = (
            kernels().extract_central_slices_rfft_3d_pose_grad_gpu
            if use_gpu
            else kernels().extract_central_slices_rfft_3d_pose_grad
        )
    if use_gpu:
        addrs = tuple(device_address(t) for t in bufs)
        pre_launch_sync(device, *bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return (
        grad_rot,
        (grad_shift if has_shifts_2d else None),
        (grad_shift_3d if has_shifts_3d else None),
    )


def run_weight_grad(
    grad_weight_vol: torch.Tensor,
    rotations: torch.Tensor,
    proj_sidelength: int,
    *,
    oversampling: float,
    fourier_radius_cutoff: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
) -> torch.Tensor:
    """Gradient w.r.t. backprojection weights: gather ``grad_weight_vol``.

    Adjoint of the (signed) weight splat. Returns real ``(bv, bp, h, w)`` on the
    CPU.
    """
    device = grad_weight_vol.device
    use_gpu = device.type != "cpu"
    tgt = device if use_gpu else torch.device("cpu")
    if grad_weight_vol.dim() == 3:
        grad_weight_vol = grad_weight_vol.unsqueeze(0)
    bv = grad_weight_vol.shape[0]
    rot, _bv_rot, bp = prep_rotations(rotations, bv, tgt)
    proj_sidelength_half = proj_sidelength // 2 + 1

    gwvol = grad_weight_vol.to(device=tgt, dtype=torch.float32).contiguous()
    grad_weight = torch.zeros(
        bv, bp, proj_sidelength, proj_sidelength_half, dtype=torch.float32, device=tgt
    )
    shifts_dummy = torch.zeros(1, bp, 2, dtype=torch.float32, device=tgt)
    shifts_3d_dummy = torch.zeros(1, bp, 3, dtype=torch.float32, device=tgt)
    radius = (
        proj_sidelength / 2.0
        if fourier_radius_cutoff is None
        else float(fourier_radius_cutoff)
    )
    params = (
        float(oversampling),
        float(radius * radius),
        0,
        interp_code(interpolation),
        float(ewald_curvature),
        0,
    )
    bufs = (gwvol, rot, shifts_dummy, shifts_3d_dummy, grad_weight)
    fn = (
        kernels().insert_central_slices_rfft_3d_weight_grad_gpu
        if use_gpu
        else kernels().insert_central_slices_rfft_3d_weight_grad
    )
    if use_gpu:
        addrs = tuple(device_address(t) for t in bufs)
        pre_launch_sync(device, *bufs)
        fn(device_session(), bufs, params, addrs)
    else:
        fn(bufs, params)
    return grad_weight


def reconstruction_volume_shape(
    proj_sidelength: int, oversampling: float
) -> tuple[int, int, int]:
    """Cubic rfft volume (d, h, w) for backprojecting ``proj_sidelength`` slices."""
    raw = math.ceil(proj_sidelength * oversampling)
    side = raw + (raw % 2)  # ensure even
    return (side, side, side // 2 + 1)


def backproject_scatter(
    projections: torch.Tensor,
    rotations: torch.Tensor,
    weights: torch.Tensor | None,
    shifts_2d: torch.Tensor | None,
    oversampling: float,
    fourier_radius_cutoff: float | None,
    interpolation: str,
    ewald_curvature: float = 0.0,
    shifts_3d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scatter projection slices into a Hermitian 3D rfft volume (+ weights)."""
    if projections.dim() == 3:
        projections = projections.unsqueeze(0)
        if weights is not None and weights.dim() == 3:
            weights = weights.unsqueeze(0)
    if projections.dim() != 4 or not projections.is_complex():
        raise ValueError("projections must be complex (bv, bp, h, w)")
    _bv, _bp, h, w = projections.shape
    if h % 2 != 0 or h != (w - 1) * 2:
        raise ValueError("projections must be square with even box (h == 2*(w-1))")
    if weights is not None and weights.shape != projections.shape:
        raise ValueError("weights must match projections shape")

    return run_scatter(
        projections,
        rotations,
        reconstruction_volume_shape(h, oversampling),
        shifts_2d=shifts_2d,
        weights=weights,
        oversampling=oversampling,
        fourier_radius_cutoff=fourier_radius_cutoff,
        interpolation=interpolation,
        friedel_double=True,
        skip_redundant=True,
        ewald_curvature=ewald_curvature,
        shifts_3d=shifts_3d,
    )
