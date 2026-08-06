"""Tests for the experimental Mojo-backed central-slice kernels.

The Mojo kernels work in rfft layout with DC at the origin. We validate them
against the package's own ``extract_central_slices_rfft_3d`` /
``insert_central_slices_rfft_3d`` (fftshifted rfft layout) -- the same operations
differing only by an ``fftshift`` of the non-redundant dims, so within the
Nyquist band they must agree.
"""

import numpy as np
import pytest
import torch

from torch_fourier_slice.experimental import (
    extract_central_slices_rfft_3d,
    insert_central_slices_rfft_3d,
    mojo_kernels_available,
)
from torch_fourier_slice.slice_extraction import (
    extract_central_slices_rfft_3d as canonical_extract_central_slices_rfft_3d,
)
from torch_fourier_slice.slice_insertion import (
    insert_central_slices_rfft_3d as canonical_insert_central_slices_rfft_3d,
)


def _linear_grad_ratio_ok(loss_fn, param, n=4, scale=0.05, tol=2e-2):
    """For an R-linear loss, the autograd grad must predict the loss change exactly."""
    L = loss_fn(param)
    L.backward()
    g = param.grad.clone()
    torch.manual_seed(123)
    ok = True
    for _ in range(n):
        delta = torch.randn_like(param) * scale
        with torch.no_grad():
            dA = (loss_fn(param.detach() + delta) - L.detach()).item()
        dP = torch.real(torch.sum(torch.conj(g) * delta)).item()
        ok = ok and (dA == 0 or abs(dP / dA - 1) < tol)
    return ok


pytestmark = pytest.mark.skipif(
    not mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
)


def _rfft_layouts(volume: torch.Tensor):
    """Return (rfft, rfft_shifted) for a real cubic volume.

    rfft         : rfft with DC at origin (unshifted)  -- experimental layout
    rfft_shifted : fftshifted on z/y (DC centered)     -- teamtomo layout
    """
    v = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    full = torch.fft.rfftn(v, dim=(-3, -2, -1))
    rfft_shifted = torch.fft.fftshift(full, dim=(-3, -2))
    rfft = torch.fft.ifftshift(rfft_shifted, dim=(-3, -2)).contiguous()
    return rfft, rfft_shifted


def _xyz_to_zyx(rot: torch.Tensor) -> torch.Tensor:
    """Convert an xyz-convention rotation matrix to our kernel's zyx convention.

    Our kernel multiplies coordinate vectors ordered (z, y, x); reversing both
    the rows and the columns of an xyz matrix gives the matrix for the same
    physical rotation acting on zyx-ordered vectors. The canonical reference
    kernels take xyz matrices by default, so wrap our calls with this.
    """
    return torch.flip(rot, dims=(-2, -1)).contiguous()


def _radius_mask(boxsize: int, fftfreq_max: float) -> torch.Tensor:
    """Boolean (H, W//2+1) mask of rfft pixels with |k| <= fftfreq_max."""
    h = torch.fft.fftshift(torch.fft.fftfreq(boxsize))  # matches teamtomo h-fftshift
    w = torch.fft.rfftfreq(boxsize)
    ky, kx = torch.meshgrid(h, w, indexing="ij")
    return (ky**2 + kx**2).sqrt() <= fftfreq_max


def test_identity_exact_in_band():
    """Identity rotation needs no interpolation -> in-band agreement is exact.

    (Only the even-box Nyquist edge can differ, where the canonical kernel
    zero-pads out-of-bounds samples while the Mojo kernel clamps to the edge
    voxel.)
    """
    torch.manual_seed(0)
    d = 32
    volume = torch.randn(d, d, d, dtype=torch.float32)
    rfft, shifted = _rfft_layouts(volume)

    rot = torch.eye(3).reshape(1, 3, 3)
    ref = canonical_extract_central_slices_rfft_3d(
        volume_rfft=shifted, rotation_matrices=rot
    )
    out = extract_central_slices_rfft_3d(rfft, rotations=rot, fourier_radius_cutoff=d)
    out_shifted = torch.fft.fftshift(out, dim=-2)

    mask = _radius_mask(d, fftfreq_max=0.45)
    assert torch.allclose(out_shifted[:, mask], ref[:, mask], atol=1e-4)


def test_matches_teamtomo_within_nyquist():
    """Random rotations agree with teamtomo inside the Nyquist circle."""
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    d = 32
    volume = torch.randn(d, d, d, dtype=torch.float32)
    rfft, shifted = _rfft_layouts(volume)

    rot = torch.tensor(
        Rotation.random(8, random_state=2).as_matrix(), dtype=torch.float32
    )
    ref = canonical_extract_central_slices_rfft_3d(
        volume_rfft=shifted, rotation_matrices=rot
    )
    out = extract_central_slices_rfft_3d(
        rfft, rotations=_xyz_to_zyx(rot)
    )  # (1, 8, d, d//2+1)
    out_shifted = torch.fft.fftshift(out, dim=-2)

    mask = _radius_mask(d, fftfreq_max=0.4)  # stay clear of the out-of-band corners
    diff = (out_shifted - ref).abs()[:, mask]
    scale = ref.abs()[:, mask].mean()
    assert diff.max() < 1e-2 * scale.clamp(min=1.0) + 1e-3
    assert diff.mean() < 1e-4 * scale.clamp(min=1.0)


def test_shifts_apply_phase_ramp():
    """A 2D shift must apply the expected Fourier phase ramp."""
    torch.manual_seed(1)
    d = 16
    volume = torch.randn(d, d, d, dtype=torch.float32)
    rfft, _ = _rfft_layouts(volume)
    rot = torch.eye(3).reshape(1, 3, 3)

    no_shift = extract_central_slices_rfft_3d(rfft, rotations=rot)
    shift = torch.tensor([[[2.0, -3.0]]])  # (1, 1, 2) xy
    with_shift = extract_central_slices_rfft_3d(rfft, rotations=rot, shifts_2d=shift)

    # Build the expected phase ramp: phase = -2pi/box * (ky*sx + kx*sy)
    ky = torch.fft.fftfreq(d)[:, None] * d  # rfft-ordered cycles (DC at origin)
    kx = torch.fft.rfftfreq(d)[None, :] * d
    phase = -2.0 * np.pi / d * (ky * shift[0, 0, 0] + kx * shift[0, 0, 1])
    ramp = torch.complex(torch.cos(phase), torch.sin(phase))
    assert torch.allclose(with_shift[0], no_shift[0] * ramp, atol=1e-4)


def _gpu_device() -> str | None:
    """A torch GPU device string to dispatch the Mojo GPU kernel, or None."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _gpu_usable() -> bool:
    """True if a GPU device is present and a tiny device-dispatched projection runs."""
    dev = _gpu_device()
    if dev is None:
        return False
    try:
        vol = torch.randn(8, 8, 8)
        rfft, _ = _rfft_layouts(vol)
        extract_central_slices_rfft_3d(
            rfft.to(dev), rotations=torch.eye(3).reshape(1, 3, 3)
        )
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
def test_gpu_matches_cpu():
    """A GPU-device input reproduces the CPU result and returns on that device."""
    from scipy.spatial.transform import Rotation

    dev = _gpu_device()
    torch.manual_seed(0)
    d = 48
    volume = torch.randn(d, d, d, dtype=torch.float32)
    rfft, _ = _rfft_layouts(volume)
    rot = torch.tensor(
        Rotation.random(6, random_state=3).as_matrix(), dtype=torch.float32
    )

    cpu = extract_central_slices_rfft_3d(
        rfft, rotations=rot
    )  # CPU tensor -> CPU kernel
    gpu = extract_central_slices_rfft_3d(
        rfft.to(dev), rotations=rot
    )  # GPU tensor -> GPU kernel
    assert gpu.device.type == dev
    assert gpu.shape == cpu.shape
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-4)

    # with shifts (GPU transcendental precision differs slightly)
    shift = torch.randn(1, 6, 2)
    cpu_s = extract_central_slices_rfft_3d(rfft, rotations=rot, shifts_2d=shift)
    gpu_s = extract_central_slices_rfft_3d(rfft.to(dev), rotations=rot, shifts_2d=shift)
    assert torch.allclose(gpu_s.cpu(), cpu_s, atol=1e-3)


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
def test_gpu_repeated_and_inplace():
    """Zero-copy: repeated projection matches CPU and reflects in-place edits.

    The GPU kernel reads the volume's device memory in place (no upload/cache),
    so projecting the same tensor twice is consistent and an in-place ``mul_``
    is picked up on the next call with no invalidation step.
    """
    dev = _gpu_device()
    torch.manual_seed(0)
    d = 32
    volume = torch.randn(d, d, d, dtype=torch.float32)
    rfft, _ = _rfft_layouts(volume)
    gpu_rfft = rfft.to(dev)
    rot = torch.eye(3).reshape(1, 3, 3)

    cpu = extract_central_slices_rfft_3d(rfft, rotations=rot)
    first = extract_central_slices_rfft_3d(gpu_rfft, rotations=rot)
    second = extract_central_slices_rfft_3d(gpu_rfft, rotations=rot)
    assert torch.allclose(first.cpu(), cpu, atol=1e-4)
    assert torch.allclose(second.cpu(), cpu, atol=1e-4)

    # in-place mutation is read live on the next call
    gpu_rfft.mul_(2.0)
    after = extract_central_slices_rfft_3d(gpu_rfft, rotations=rot)
    assert torch.allclose(after.cpu(), 2 * cpu, atol=1e-3)


def test_forward_projection_gradient():
    """Autograd grad of the forward projection (adjoint scatter) is exact."""
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    d = 24
    rot = torch.tensor(
        Rotation.random(5, random_state=1).as_matrix(), dtype=torch.float32
    ).unsqueeze(0)
    cut = d / 4.0  # interior samples (avoids the boundary clamp/drop asymmetry)
    w = torch.randn(1, 5, d, d // 2 + 1, dtype=torch.complex64)

    def loss(rec):
        proj = extract_central_slices_rfft_3d(
            rec, rotations=rot, fourier_radius_cutoff=cut
        )
        return torch.real(torch.sum(torch.conj(w) * proj))

    rec = torch.randn(d, d, d // 2 + 1, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(loss, rec)


def test_backprojection_matches_teamtomo_within_nyquist():
    """The 2D->3D insertion agrees with the canonical kernel inside the band.

    Slices are the rfft of real images, so they carry the Hermitian symmetry
    both kernels assume on the kx=0 plane. The DC voxel is excluded: it picks up
    an (unphysical) imaginary part from neighbouring samples splatting onto it,
    and the two kernels distribute that differently.
    """
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    d = 24
    P = 6
    fftfreq_max = 0.4
    rot = torch.tensor(
        Rotation.random(P, random_state=1).as_matrix(), dtype=torch.float32
    )
    images = torch.randn(P, d, d, dtype=torch.float32)
    proj = torch.fft.rfftn(images, dim=(-2, -1))  # Hermitian, DC at origin

    mine, mine_w = insert_central_slices_rfft_3d(
        proj,
        rotations=_xyz_to_zyx(rot),
        weights=torch.ones_like(proj, dtype=torch.float32),
        fourier_radius_cutoff=fftfreq_max * d,
    )
    ref, ref_w = canonical_insert_central_slices_rfft_3d(
        image_rfft=torch.fft.fftshift(proj, dim=-2),  # canonical: DC centred on h
        volume_shape=(d, d, d),
        rotation_matrices=rot,
        fftfreq_max=fftfreq_max,
    )
    ref = torch.fft.ifftshift(ref, dim=(-3, -2))  # canonical -> DC at origin
    ref_w = torch.fft.ifftshift(ref_w, dim=(-3, -2))
    assert mine.shape == ref.shape

    kz = torch.fft.fftfreq(d)[:, None, None]
    ky = torch.fft.fftfreq(d)[None, :, None]
    kx = torch.fft.rfftfreq(d)[None, None, :]
    in_band = (kz**2 + ky**2 + kx**2).sqrt() < 0.3
    in_band[0, 0, 0] = False  # DC, see docstring

    scale = ref[in_band].abs().mean()
    assert (mine - ref).abs()[in_band].max() < 5e-3 * scale
    # the accumulated density (weights of ones) must match closely everywhere
    assert torch.allclose(mine_w[in_band], ref_w[in_band], atol=1e-4)


def test_backprojection_gradient_and_weights():
    """Backprojection grad (exact adjoint = forward projection) and weight output."""
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    d = 24
    P = 6
    dh = d // 2 + 1
    rot = torch.tensor(
        Rotation.random(P, random_state=1).as_matrix(), dtype=torch.float32
    ).unsqueeze(0)
    cut = d / 4.0
    w_func = torch.randn(d, d, dh, dtype=torch.complex64)

    def loss(p):
        dvol, _ = insert_central_slices_rfft_3d(
            p, rotations=rot, fourier_radius_cutoff=cut
        )
        return torch.real(torch.sum(torch.conj(w_func) * dvol))

    proj = torch.randn(P, d, dh, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(loss, proj)

    # weight accumulation returns a matching real volume
    weights = torch.rand(P, d, dh, dtype=torch.float32)
    data_vol, weight_vol = insert_central_slices_rfft_3d(
        proj.detach(), rotations=rot, weights=weights
    )
    assert weight_vol is not None
    assert weight_vol.shape == data_vol.shape
    assert weight_vol.dtype == torch.float32
    # no weights -> None
    _, none_w = insert_central_slices_rfft_3d(proj.detach(), rotations=rot)
    assert none_w is None


def test_rfft_layer_rank_single_and_multivolume():
    """rfft-layer extract/insert: single (squeeze) vs multivolume (transpose)."""
    from torch_fourier_slice.experimental import (
        extract_central_slices_rfft_3d,
        extract_central_slices_rfft_3d_multivolume,
        insert_central_slices_rfft_3d,
        insert_central_slices_rfft_3d_multivolume,
    )

    torch.manual_seed(0)
    d, P, bv = 24, 5, 3
    dh = d // 2 + 1
    rot = _rand_rot(P, 1)  # (P, 3, 3) -- shared poses
    vols = torch.randn(bv, d, d, dh, dtype=torch.complex64)

    # single extract: (d, h, w) -> (P, h, w)
    s0 = extract_central_slices_rfft_3d(vols[0], rot)
    assert s0.shape == (P, d, dh)

    # multivolume, shared poses: (bv, d, h, w) -> (P, bv, h, w)
    sm = extract_central_slices_rfft_3d_multivolume(vols, rot)
    assert sm.shape == (P, bv, d, dh)
    for i in range(bv):
        assert torch.allclose(sm[:, i], extract_central_slices_rfft_3d(vols[i], rot))

    # multivolume, per-volume poses: rotations (bv, P, 3, 3)
    rots_pv = torch.stack([_rand_rot(P, i + 1) for i in range(bv)])
    smp = extract_central_slices_rfft_3d_multivolume(vols, rots_pv)
    assert smp.shape == (P, bv, d, dh)
    for i in range(bv):
        ref = extract_central_slices_rfft_3d(vols[i], rots_pv[i])
        assert torch.allclose(smp[:, i], ref)

    # single insert: (P, h, w) -> (d, h, w)
    imgs = torch.randn(P, d, dh, dtype=torch.complex64)
    v0, w0 = insert_central_slices_rfft_3d(imgs, rot)
    assert v0.shape == (d, d, dh) and w0 is None

    # multivolume insert: (P, bv, h, w) -> (bv, d, h, w)
    imgs_m = torch.randn(P, bv, d, dh, dtype=torch.complex64)
    vm, wm = insert_central_slices_rfft_3d_multivolume(imgs_m, rot)
    assert vm.shape == (bv, d, d, dh) and wm is None
    for i in range(bv):
        vi, _ = insert_central_slices_rfft_3d(imgs_m[:, i], rot)
        assert torch.allclose(vm[i], vi)

    # gradients flow through the rank adaptation (squeeze / transpose)
    v = vols[0].clone().requires_grad_(True)
    extract_central_slices_rfft_3d(v, rot).abs().pow(2).sum().backward()
    assert v.grad is not None and v.grad.shape == vols[0].shape


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
def test_gpu_scatter_matches_cpu():
    """The GPU scatter (backprojection) reproduces the CPU kernel + grad on device."""
    from scipy.spatial.transform import Rotation

    dev = _gpu_device()
    torch.manual_seed(0)
    d = 24
    P = 8
    dh = d // 2 + 1
    rot = torch.tensor(
        Rotation.random(P, random_state=1).as_matrix(), dtype=torch.float32
    ).unsqueeze(0)
    proj = torch.randn(P, d, dh, dtype=torch.complex64)
    w = torch.rand(P, d, dh, dtype=torch.float32)

    cpu_v, cpu_w = insert_central_slices_rfft_3d(proj, rotations=rot, weights=w)
    gpu_v, gpu_w = insert_central_slices_rfft_3d(
        proj.to(dev), rotations=rot, weights=w.to(dev)
    )
    assert gpu_v.device.type == dev
    assert torch.allclose(gpu_v.cpu(), cpu_v, atol=1e-4)
    assert torch.allclose(gpu_w.cpu(), cpu_w, atol=1e-4)

    # forward-projection gradient on device exercises the GPU scatter as backward
    cut = d / 4.0
    wf = torch.randn(P, d, dh, dtype=torch.complex64, device=dev)

    def loss(rec):
        proj = extract_central_slices_rfft_3d(
            rec, rotations=rot, fourier_radius_cutoff=cut
        )
        return torch.real(torch.sum(torch.conj(wf) * proj))

    rec = torch.randn(d, d, dh, dtype=torch.complex64, device=dev, requires_grad=True)
    assert _linear_grad_ratio_ok(loss, rec)


def test_invalid_interpolation_raises():
    """An unknown interpolation name is rejected."""
    vol = torch.randn(8, 8, 8)
    rfft, _ = _rfft_layouts(vol)
    with pytest.raises(ValueError, match="interpolation"):
        extract_central_slices_rfft_3d(
            rfft, rotations=torch.eye(3).reshape(1, 3, 3), interpolation="nope"
        )


def test_cubic_is_more_accurate_than_linear():
    """Tricubic interpolation beats trilinear against an exact ground truth.

    The rfft of an isotropic blob is itself isotropic, so its exact central slice
    is the *same* for every rotation -- and the identity rotation samples the
    volume on integer coordinates, i.e. with no interpolation at all. That makes
    the identity slice an exact reference, and any deviation of a rotated slice
    from it pure interpolation error.
    """
    from scipy.spatial.transform import Rotation

    d = 48
    axis = torch.arange(d, dtype=torch.float32) - d // 2
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    volume = torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * 3.0**2))
    rfft, _ = _rfft_layouts(volume)
    cutoff = d / 4  # stay well inside the band, away from the boundary clamp

    truth = extract_central_slices_rfft_3d(
        rfft, rotations=torch.eye(3).reshape(1, 3, 3), fourier_radius_cutoff=cutoff
    )[0]
    rotations = torch.tensor(
        Rotation.random(8, random_state=5).as_matrix(), dtype=torch.float32
    )

    errors = {}
    for interpolation in ("linear", "cubic"):
        out = extract_central_slices_rfft_3d(
            rfft,
            rotations=rotations,
            fourier_radius_cutoff=cutoff,
            interpolation=interpolation,
        )
        errors[interpolation] = (out - truth[None]).abs().mean()

    scale = truth.abs().mean()
    assert errors["cubic"] < 0.01 * scale
    assert 3 * errors["cubic"] < errors["linear"]


def test_cubic_gradients():
    """Tricubic forward and backprojection gradients are exact."""
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    d = 24
    dh = d // 2 + 1
    P = 5
    rot = torch.tensor(
        Rotation.random(P, random_state=1).as_matrix(), dtype=torch.float32
    ).unsqueeze(0)
    cut = d / 4.0

    wf = torch.randn(P, d, dh, dtype=torch.complex64)
    wv = torch.randn(d, d, dh, dtype=torch.complex64)

    def forward_loss(rec):
        proj = extract_central_slices_rfft_3d(
            rec, rotations=rot, fourier_radius_cutoff=cut, interpolation="cubic"
        )
        return torch.real(torch.sum(torch.conj(wf) * proj))

    def backproject_loss(proj):
        vol, _ = insert_central_slices_rfft_3d(
            proj, rotations=rot, fourier_radius_cutoff=cut, interpolation="cubic"
        )
        return torch.real(torch.sum(torch.conj(wv) * vol))

    rec = torch.randn(d, d, dh, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(forward_loss, rec)
    proj = torch.randn(P, d, dh, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(backproject_loss, proj)


def _fd_ratio(loss_fn, param, eps, seed=0):
    """|autograd directional deriv / central-difference - 1| for a real param."""
    loss_fn(param).backward()
    g = param.grad.clone()
    torch.manual_seed(seed)
    direction = torch.randn_like(param)
    with torch.no_grad():
        plus = loss_fn(param.detach() + eps * direction).item()
        minus = loss_fn(param.detach() - eps * direction).item()
    fd = (plus - minus) / (2 * eps)
    pred = torch.sum(g * direction).item()
    return abs(pred) if abs(fd) < 1e-5 else abs(pred / fd - 1)


def _rand_rot(n, seed):
    from scipy.spatial.transform import Rotation

    return torch.tensor(
        Rotation.random(n, random_state=seed).as_matrix(), dtype=torch.float32
    )


def _herm_projection(n, box, seed):
    """A physically valid (Hermitian) rfft projection stack: the rfft of a real
    image. Backprojection assumes Hermitian input (it skips the redundant rfft
    half), so shift gradients are only exact for such projections -- an arbitrary
    complex ``randn`` stack is not a valid projection and breaks that assumption.
    """
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(n, box, box, generator=g)
    return torch.fft.rfftn(img, dim=(-2, -1)).contiguous()


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_forward_pose_gradients(interp):
    """Forward-projection gradients w.r.t. rotations and shifts (finite difference)."""
    torch.manual_seed(0)
    d, P = 20, 3
    cut = d / 4.0  # interior samples (avoids the boundary clamp/drop asymmetry)
    vol = torch.randn(d, d, d // 2 + 1, dtype=torch.complex64)
    rot0 = _rand_rot(P, 1).unsqueeze(0)
    sh0 = torch.randn(1, P, 2) * 0.5
    target = torch.randn(P, d, d // 2 + 1, dtype=torch.complex64)

    def proj_loss(rot, sh):
        p = extract_central_slices_rfft_3d(
            vol,
            rotations=rot,
            shifts_2d=sh,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return ((p - target).abs() ** 2).sum()

    rot = rot0.clone().requires_grad_(True)
    assert _fd_ratio(lambda r: proj_loss(r, None), rot, eps=3e-4) < 5e-2

    shift = sh0.clone().requires_grad_(True)
    assert _fd_ratio(lambda s: proj_loss(rot0, s), shift, eps=1e-3) < 3e-2


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_backprojection_pose_and_weight_gradients(interp):
    """Backprojection gradients w.r.t. rotations, shifts and weights (finite diff)."""
    torch.manual_seed(0)
    d, P = 20, 3
    dh = d // 2 + 1
    cut = d / 4.0
    proj = _herm_projection(P, d, 7)  # Hermitian: a valid projection stack
    rot0 = _rand_rot(P, 1).unsqueeze(0)
    sh0 = torch.randn(1, P, 2) * 0.5
    wts0 = torch.rand(P, d, dh)
    data_t = torch.randn(d, d, dh, dtype=torch.complex64)
    weight_t = torch.randn(d, d, dh)

    def data_loss(rot, sh):
        dvol, _ = insert_central_slices_rfft_3d(
            proj,
            rotations=rot,
            shifts_2d=sh,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return ((dvol - data_t).abs() ** 2).sum()

    def weight_loss(wts):
        _, wvol = insert_central_slices_rfft_3d(
            proj,
            rotations=rot0,
            weights=wts,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return ((wvol - weight_t) ** 2).sum()

    rot = rot0.clone().requires_grad_(True)
    assert _fd_ratio(lambda r: data_loss(r, None), rot, eps=3e-4) < 5e-2

    shift = sh0.clone().requires_grad_(True)
    assert _fd_ratio(lambda s: data_loss(rot0, s), shift, eps=1e-3) < 3e-2

    wts = wts0.clone().requires_grad_(True)
    assert _fd_ratio(weight_loss, wts, eps=1e-3) < 2e-2


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_ewald_curvature_gradients(interp):
    """Ewald curvature bends the slice; FD-check the now-active z-column Jacobian.

    A flat slice leaves the rotation's z-input column with zero gradient; a
    non-zero ``ewald_curvature`` gives every pixel a z-offset, so that column
    becomes active in both the forward and backprojection pose-gradient kernels.
    """
    torch.manual_seed(0)
    d, P = 20, 3
    dh = d // 2 + 1
    cut = d / 4.0
    ewald = 0.02
    vol = torch.randn(d, d, dh, dtype=torch.complex64)
    rot0 = _rand_rot(P, 1).unsqueeze(0)
    sh0 = torch.randn(1, P, 2) * 0.5
    target = torch.randn(P, d, dh, dtype=torch.complex64)
    proj = torch.randn(P, d, dh, dtype=torch.complex64)
    data_t = torch.randn(d, d, dh, dtype=torch.complex64)

    # curvature must actually change the projection (otherwise the test is vacuous)
    flat = extract_central_slices_rfft_3d(
        vol, rotations=rot0, fourier_radius_cutoff=cut, interpolation=interp
    )
    curved = extract_central_slices_rfft_3d(
        vol,
        rotations=rot0,
        fourier_radius_cutoff=cut,
        interpolation=interp,
        ewald_curvature=ewald,
    )
    assert not torch.allclose(flat, curved, atol=1e-4)

    def proj_loss(rot, sh):
        p = extract_central_slices_rfft_3d(
            vol,
            rotations=rot,
            shifts_2d=sh,
            fourier_radius_cutoff=cut,
            interpolation=interp,
            ewald_curvature=ewald,
        )
        return ((p - target).abs() ** 2).sum()

    def data_loss(rot):
        dvol, _ = insert_central_slices_rfft_3d(
            proj,
            rotations=rot,
            fourier_radius_cutoff=cut,
            interpolation=interp,
            ewald_curvature=ewald,
        )
        return ((dvol - data_t).abs() ** 2).sum()

    rot = rot0.clone().requires_grad_(True)
    assert _fd_ratio(lambda r: proj_loss(r, None), rot, eps=3e-4) < 5e-2

    shift = sh0.clone().requires_grad_(True)
    assert _fd_ratio(lambda s: proj_loss(rot0, s), shift, eps=1e-3) < 3e-2

    rot_b = rot0.clone().requires_grad_(True)
    assert _fd_ratio(data_loss, rot_b, eps=3e-4) < 5e-2


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_shifts_3d_gradients(interp):
    """3D (zyx, pre-rotation) shift: forward equivalence, grads, rotation coupling."""
    torch.manual_seed(0)
    d, P = 20, 3
    dh = d // 2 + 1
    cut = d / 4.0
    vol = torch.randn(d, d, dh, dtype=torch.complex64)
    rot0 = _rand_rot(P, 1).unsqueeze(0)
    target = torch.randn(P, d, dh, dtype=torch.complex64)
    proj = _herm_projection(P, d, 1)  # Hermitian: a valid projection stack
    data_t = torch.randn(d, d, dh, dtype=torch.complex64)
    s3_0 = torch.randn(1, P, 3) * 0.5

    # at identity rotation (no oversampling) a zyx shift (0, ty, tx) applies the
    # same phase ramp to the same samples as the 2D yx shift (ty, tx).
    eye = torch.eye(3).reshape(1, 1, 3, 3).expand(1, P, 3, 3).contiguous()
    s2 = torch.randn(1, P, 2)
    s3 = torch.zeros(1, P, 3)
    s3[..., 1:] = s2
    a = extract_central_slices_rfft_3d(
        vol, rotations=eye, shifts_2d=s2, interpolation=interp
    )
    b = extract_central_slices_rfft_3d(
        vol, rotations=eye, shifts_3d=s3, interpolation=interp
    )
    assert torch.allclose(a, b, atol=1e-4)

    def proj_loss(rot, s3d):
        p = extract_central_slices_rfft_3d(
            vol,
            rotations=rot,
            shifts_3d=s3d,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return ((p - target).abs() ** 2).sum()

    s3v = s3_0.clone().requires_grad_(True)
    assert _fd_ratio(lambda s: proj_loss(rot0, s), s3v, eps=3e-3) < 3e-2

    # rotation grad with a 3D shift active -> exercises the coupling term, both
    # for the forward projection and (below) for the backprojection.
    rot = rot0.clone().requires_grad_(True)
    assert _fd_ratio(lambda r: proj_loss(r, s3_0), rot, eps=3e-4) < 5e-2

    def data_loss(rot, s3d):
        dvol, _ = insert_central_slices_rfft_3d(
            proj,
            rotations=rot,
            shifts_3d=s3d,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return ((dvol - data_t).abs() ** 2).sum()

    rotb = rot0.clone().requires_grad_(True)
    assert _fd_ratio(lambda r: data_loss(r, s3_0), rotb, eps=3e-4) < 5e-2
    s3vb = s3_0.clone().requires_grad_(True)
    assert _fd_ratio(lambda s: data_loss(rot0, s), s3vb, eps=3e-3) < 3e-2


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
def test_gpu_pose_weight_gradients_match_cpu():
    """The GPU rotation/shift/weight backward kernels reproduce the CPU grads."""
    dev = _gpu_device()
    torch.manual_seed(0)
    d, P = 24, 4
    dh = d // 2 + 1
    vol = torch.randn(d, d, dh, dtype=torch.complex64)
    rot0 = _rand_rot(P, 1).unsqueeze(0)
    sh0 = torch.randn(1, P, 2) * 0.5
    proj = torch.randn(P, d, dh, dtype=torch.complex64)
    wts0 = torch.rand(P, d, dh)
    target = torch.randn(P, d, dh, dtype=torch.complex64)

    def fwd_grads(device):
        r = rot0.clone().to(device).requires_grad_(True)
        s = sh0.clone().to(device).requires_grad_(True)
        p = extract_central_slices_rfft_3d(vol.to(device), rotations=r, shifts_2d=s)
        ((p - target.to(device)).abs() ** 2).sum().backward()
        return r.grad.cpu(), s.grad.cpu()

    def bp_grads(device):
        r = rot0.clone().to(device).requires_grad_(True)
        s = sh0.clone().to(device).requires_grad_(True)
        w = wts0.clone().to(device).requires_grad_(True)
        dvol, _wvol = insert_central_slices_rfft_3d(
            proj.to(device), rotations=r, weights=w, shifts_2d=s
        )
        (dvol.abs() ** 2).sum().backward()
        return r.grad.cpu(), s.grad.cpu(), w.grad.cpu()

    cr, cs = fwd_grads("cpu")
    gr, gs = fwd_grads(dev)
    assert torch.allclose(gr, cr, atol=1e-3 * cr.abs().max())
    assert torch.allclose(gs, cs, atol=1e-3 * cs.abs().max())

    cr, cs, cw = bp_grads("cpu")
    gr, gs, gw = bp_grads(dev)
    assert torch.allclose(gr, cr, atol=1e-3 * cr.abs().max())
    assert torch.allclose(gs, cs, atol=1e-3 * cs.abs().max() + 1e-6)
    assert torch.allclose(gw, cw, atol=1e-4)


def test_output_shape_and_batching():
    """Multi-volume / per-volume poses and custom output_shape (pose-major out)."""
    from torch_fourier_slice.experimental import (
        extract_central_slices_rfft_3d_multivolume,
    )

    torch.manual_seed(2)
    d = 16
    vols = torch.randn(2, d, d, d, dtype=torch.float32)
    # (bv=2, d, d, d//2+1)
    rfft = torch.stack([_rfft_layouts(v)[0] for v in vols])

    rot = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 3, 1, 1)  # (bv=2, P=3, 3, 3)
    out = extract_central_slices_rfft_3d_multivolume(rfft, rot)
    assert out.shape == (3, 2, d, d // 2 + 1)  # (P, bv, h, w)
    assert out.is_complex()

    out_small = extract_central_slices_rfft_3d_multivolume(
        rfft, rot, output_shape=(8, 8)
    )
    assert out_small.shape == (3, 2, 8, 8 // 2 + 1)
