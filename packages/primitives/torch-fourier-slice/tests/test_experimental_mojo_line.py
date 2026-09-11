"""Tests for the experimental Mojo-backed central-*line* projectors (3D <-> 1D).

A central line is the degenerate central slice whose in-plane (y) axis is
collapsed to the single DC row, sampled along a **direction** ``u`` (a zyx unit
vector) rather than a rotation matrix. The line kernels are validated
**internally** (no external reference): against the trusted 3D<->2D slice kernels
on the shared geometry, by the adjointness of the autograd data gradients, by
finite-difference checks of the direction / shift / weight gradients, and by a
density-compensated proj/backproj round trip.
"""

import pytest
import torch

from torch_fourier_slice.experimental import (
    extract_central_line_rfft_3d,
    extract_central_line_rfft_3d_multivolume,
    extract_central_slices_rfft_3d,
    insert_central_line_rfft_3d,
    insert_central_line_rfft_3d_multivolume,
    mojo_kernels_available,
)

pytestmark = pytest.mark.skipif(
    not mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
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
    return abs(pred) if abs(fd) < 1e-6 else abs(pred / fd - 1)


def _rand_rot(n, seed):
    from scipy.spatial.transform import Rotation

    return torch.tensor(
        Rotation.random(n, random_state=seed).as_matrix(), dtype=torch.float32
    )


def _rand_dirs(n, seed):
    """n random zyx unit directions (the line's ``u = R @ x_hat``)."""
    return _rand_rot(n, seed)[:, :, 2].contiguous()  # third column of the zyx matrix


def _rfft(volume: torch.Tensor) -> torch.Tensor:
    """rfft with DC at origin (the experimental layout), fftshifted real input."""
    v = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    return torch.fft.rfftn(v, dim=(-3, -2, -1)).contiguous()


def _herm_lines(n, box, seed):
    """A valid (Hermitian) line stack: the rfft of a real 1D signal per node."""
    g = torch.Generator().manual_seed(seed)
    return torch.fft.rfft(torch.randn(n, box, generator=g), dim=-1).contiguous()


def _gpu_device() -> str | None:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _gpu_usable() -> bool:
    dev = _gpu_device()
    if dev is None:
        return False
    try:
        rfft = _rfft(torch.randn(8, 8, 8))
        extract_central_line_rfft_3d(
            rfft.to(dev), directions=torch.tensor([0.0, 0.0, 1.0])
        )
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Parity with the trusted slice kernels (the line IS the slice's y=0 DC row)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_line_equals_slice_dc_row(interp):
    """A line at ``u = R @ x_hat`` equals the y=0 (DC) row of the R-slice, exactly."""
    torch.manual_seed(0)
    d = 32
    rfft = _rfft(torch.randn(d, d, d, dtype=torch.float32))
    rot = _rand_rot(6, 2)
    u = rot[:, :, 2].contiguous()  # third column = the slice's x-axis direction
    cut = d / 4.0
    sl = extract_central_slices_rfft_3d(
        rfft, rotations=rot, fourier_radius_cutoff=cut, interpolation=interp
    )  # (6, d, d//2+1)
    ln = extract_central_line_rfft_3d(
        rfft, directions=u, fourier_radius_cutoff=cut, interpolation=interp
    )  # (6, d//2+1)
    assert ln.shape == (6, d // 2 + 1)
    assert torch.equal(ln, sl[:, 0, :])


def test_line_shifts_3d_equals_slice_dc_row():
    """The line's 3D (u.t) shift phase matches the slice kernel on the DC row."""
    torch.manual_seed(0)
    d = 32
    rfft = _rfft(torch.randn(d, d, d, dtype=torch.float32))
    rot = _rand_rot(5, 2)
    u = rot[:, :, 2].contiguous()
    s3 = torch.randn(1, 5, 3) * 0.4
    cut = d / 4.0
    sl = extract_central_slices_rfft_3d(
        rfft, rotations=rot, shifts_3d=s3, fourier_radius_cutoff=cut
    )
    ln = extract_central_line_rfft_3d(
        rfft, directions=u, shifts_3d=s3, fourier_radius_cutoff=cut
    )
    assert torch.equal(ln, sl[:, 0, :])


# ---------------------------------------------------------------------------
# Autograd data gradients (= exact adjoints)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_line_extract_gradient(interp):
    """Autograd grad of line extraction (adjoint line scatter) is exact."""
    torch.manual_seed(0)
    d, P = 24, 8
    dh = d // 2 + 1
    u = _rand_dirs(P, 1)
    cut = d / 4.0
    w = torch.randn(P, dh, dtype=torch.complex64)

    def loss(rec):
        ln = extract_central_line_rfft_3d(
            rec, directions=u, fourier_radius_cutoff=cut, interpolation=interp
        )
        return torch.real(torch.sum(torch.conj(w) * ln))

    rec = torch.randn(d, d, dh, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(loss, rec)


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_line_insert_gradient_and_weights(interp):
    """Line insertion grad (exact adjoint = line extraction) and weight output."""
    torch.manual_seed(0)
    d, P = 24, 8
    dh = d // 2 + 1
    u = _rand_dirs(P, 1)
    cut = d / 4.0
    wv = torch.randn(d, d, dh, dtype=torch.complex64)

    def loss(lines):
        vol, _ = insert_central_line_rfft_3d(
            lines, directions=u, fourier_radius_cutoff=cut, interpolation=interp
        )
        return torch.real(torch.sum(torch.conj(wv) * vol))

    lines = torch.randn(P, dh, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(loss, lines)

    weights = torch.rand(P, dh)
    data_vol, weight_vol = insert_central_line_rfft_3d(
        lines.detach(), directions=u, weights=weights
    )
    assert weight_vol is not None
    assert weight_vol.shape == data_vol.shape and weight_vol.dtype == torch.float32
    _, none_w = insert_central_line_rfft_3d(lines.detach(), directions=u)
    assert none_w is None


# ---------------------------------------------------------------------------
# Direction / shift / weight gradients (finite difference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_line_extract_pose_gradients(interp):
    """Forward-line gradients w.r.t. directions and 3D shifts (finite difference)."""
    torch.manual_seed(0)
    d, P = 20, 3
    dh = d // 2 + 1
    cut = d / 4.0
    vol = torch.randn(d, d, dh, dtype=torch.complex64)
    u0 = _rand_dirs(P, 1)
    s0 = torch.randn(P, 3) * 0.5
    target = torch.randn(P, dh, dtype=torch.complex64)

    def loss(u, s):
        p = extract_central_line_rfft_3d(
            vol,
            directions=u,
            shifts_3d=s,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return ((p - target).abs() ** 2).sum()

    u = u0.clone().requires_grad_(True)
    assert _fd_ratio(lambda x: loss(x, None), u, eps=3e-4) < 5e-2

    s = s0.clone().requires_grad_(True)
    assert _fd_ratio(lambda x: loss(u0, x), s, eps=3e-3) < 3e-2

    # direction grad with a 3D shift active exercises the coupling term
    u = u0.clone().requires_grad_(True)
    assert _fd_ratio(lambda x: loss(x, s0), u, eps=3e-4) < 5e-2


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_line_insert_pose_and_weight_gradients(interp):
    """Line-insertion gradients w.r.t. directions, shifts and weights (finite diff)."""
    torch.manual_seed(0)
    d, P = 20, 3
    dh = d // 2 + 1
    cut = d / 4.0
    lines = _herm_lines(P, d, 7)  # Hermitian: a valid line stack
    u0 = _rand_dirs(P, 1)
    s0 = torch.randn(P, 3) * 0.5
    data_t = torch.randn(d, d, dh, dtype=torch.complex64)
    weight_cotangent = torch.randn(d, d, dh)

    def data_loss(u, s):
        dvol, _ = insert_central_line_rfft_3d(
            lines,
            directions=u,
            shifts_3d=s,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return ((dvol - data_t).abs() ** 2).sum()

    u = u0.clone().requires_grad_(True)
    assert _fd_ratio(lambda x: data_loss(x, None), u, eps=3e-4) < 5e-2

    s = s0.clone().requires_grad_(True)
    assert _fd_ratio(lambda x: data_loss(u0, x), s, eps=3e-3) < 3e-2

    # weights: the weight volume is linear in the weights, so its gradient is an
    # exact adjoint -- test that directly with an R-linear loss (a dot-product
    # test, robust to the float32 cancellation a quadratic FD would suffer).
    def weight_loss(wts):
        _, wvol = insert_central_line_rfft_3d(
            lines,
            directions=u0,
            weights=wts,
            fourier_radius_cutoff=cut,
            interpolation=interp,
        )
        return (wvol * weight_cotangent).sum()

    wts = torch.rand(P, dh, requires_grad=True)
    assert _linear_grad_ratio_ok(weight_loss, wts)


# ---------------------------------------------------------------------------
# proj / backproj round trip (density-compensated reconstruction)
# ---------------------------------------------------------------------------


def test_line_roundtrip_reconstruction():
    """Lines covering the sphere, inserted + density-compensated, recover the volume."""
    torch.manual_seed(0)
    d = 32
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(d) - d / 2] * 3, indexing="ij"), 0
    ).float()
    vol = torch.zeros(d, d, d)
    for c in [(-6, 2, 4), (5, -3, -2), (0, 7, -5)]:
        r2 = sum((grid[i] - c[i]) ** 2 for i in range(3))
        vol += torch.exp(-r2 / 8.0)
    rfft = _rfft(vol)

    u = _rand_dirs(4000, 7)
    lines = extract_central_line_rfft_3d(rfft, directions=u)
    weights = torch.ones_like(lines.real)
    data_vol, wvol = insert_central_line_rfft_3d(lines, directions=u, weights=weights)
    recon = data_vol / wvol.clamp(min=1.0)

    kz = torch.fft.fftfreq(d)[:, None, None] * d
    ky = torch.fft.fftfreq(d)[None, :, None] * d
    kx = torch.fft.rfftfreq(d)[None, None, :] * d
    kr = (kz**2 + ky**2 + kx**2).sqrt()
    covered = wvol > 0.5

    def corr(a, b, m):
        a, b = a[m], b[m]
        num = torch.real(torch.sum(torch.conj(a) * b))
        return (
            num / (a.abs().pow(2).sum().sqrt() * b.abs().pow(2).sum().sqrt())
        ).item()

    for lo, hi in [(2, 5), (5, 8), (8, 11)]:
        m = covered & (kr >= lo) & (kr < hi)
        assert corr(rfft, recon, m) > 0.85

    real_recon = torch.fft.ifftshift(
        torch.fft.irfftn(recon, s=(d, d, d), dim=(-3, -2, -1)), dim=(-3, -2, -1)
    )
    c = torch.corrcoef(torch.stack([vol.flatten(), real_recon.flatten()]))[0, 1]
    assert c > 0.9


# ---------------------------------------------------------------------------
# Rank adaptation and batching
# ---------------------------------------------------------------------------


def test_line_rank_single_and_multivolume():
    """rfft-layer extract/insert: single (squeeze) vs multivolume (transpose)."""
    torch.manual_seed(0)
    d, P, bv = 24, 5, 3
    dh = d // 2 + 1
    u = _rand_dirs(P, 1)
    vols = torch.randn(bv, d, d, dh, dtype=torch.complex64)

    s0 = extract_central_line_rfft_3d(vols[0], u)
    assert s0.shape == (P, dh)

    sm = extract_central_line_rfft_3d_multivolume(vols, u)
    assert sm.shape == (P, bv, dh)
    for i in range(bv):
        assert torch.allclose(sm[:, i], extract_central_line_rfft_3d(vols[i], u))

    dirs_pv = torch.stack([_rand_dirs(P, i + 1) for i in range(bv)])
    smp = extract_central_line_rfft_3d_multivolume(vols, dirs_pv)
    assert smp.shape == (P, bv, dh)
    for i in range(bv):
        assert torch.allclose(
            smp[:, i], extract_central_line_rfft_3d(vols[i], dirs_pv[i])
        )

    lines = torch.randn(P, dh, dtype=torch.complex64)
    v0, w0 = insert_central_line_rfft_3d(lines, u)
    assert v0.shape == (d, d, dh) and w0 is None

    lines_m = torch.randn(P, bv, dh, dtype=torch.complex64)
    vm, wm = insert_central_line_rfft_3d_multivolume(lines_m, u)
    assert vm.shape == (bv, d, d, dh) and wm is None
    for i in range(bv):
        vi, _ = insert_central_line_rfft_3d(lines_m[:, i], u)
        assert torch.allclose(vm[i], vi)

    # gradients flow through the rank adaptation (squeeze / transpose)
    v = vols[0].clone().requires_grad_(True)
    extract_central_line_rfft_3d(v, u).abs().pow(2).sum().backward()
    assert v.grad is not None and v.grad.shape == vols[0].shape


def test_line_output_length():
    """A shorter output_length yields a shorter half-line matching the slice row."""
    torch.manual_seed(0)
    d = 32
    rfft = _rfft(torch.randn(d, d, d, dtype=torch.float32))
    rot = _rand_rot(4, 1)
    u = rot[:, :, 2].contiguous()
    L = 20
    ln = extract_central_line_rfft_3d(rfft, directions=u, output_length=L)
    assert ln.shape == (4, L // 2 + 1)
    sl = extract_central_slices_rfft_3d(rfft, rotations=rot, output_shape=(L, L))
    assert torch.equal(ln, sl[:, 0, :])


# ---------------------------------------------------------------------------
# GPU parity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_gpu_line_matches_cpu(interp):
    """The GPU line kernels reproduce the CPU result and stay on device."""
    dev = _gpu_device()
    torch.manual_seed(0)
    d, P = 48, 10
    dh = d // 2 + 1
    rfft = _rfft(torch.randn(d, d, d, dtype=torch.float32))
    u = _rand_dirs(P, 3)

    cpu = extract_central_line_rfft_3d(rfft, directions=u, interpolation=interp)
    gpu = extract_central_line_rfft_3d(rfft.to(dev), directions=u, interpolation=interp)
    assert gpu.device.type == dev
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-3)

    lines = torch.randn(P, dh, dtype=torch.complex64)
    w = torch.rand(P, dh)
    cv, cw = insert_central_line_rfft_3d(
        lines, directions=u, weights=w, interpolation=interp
    )
    gv, gw = insert_central_line_rfft_3d(
        lines.to(dev), directions=u, weights=w.to(dev), interpolation=interp
    )
    assert gv.device.type == dev
    assert torch.allclose(gv.cpu(), cv, atol=1e-4)
    assert torch.allclose(gw.cpu(), cw, atol=1e-4)


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
def test_gpu_line_pose_weight_gradients_match_cpu():
    """The GPU direction/shift/weight backward kernels reproduce the CPU grads."""
    dev = _gpu_device()
    torch.manual_seed(0)
    d, P = 24, 4
    dh = d // 2 + 1
    vol = torch.randn(d, d, dh, dtype=torch.complex64)
    u0 = _rand_dirs(P, 1)
    s0 = torch.randn(P, 3) * 0.5
    lines = _herm_lines(P, d, 7)
    wts0 = torch.rand(P, dh)
    target = torch.randn(P, dh, dtype=torch.complex64)

    def fwd_grads(device):
        u = u0.clone().to(device).requires_grad_(True)
        s = s0.clone().to(device).requires_grad_(True)
        p = extract_central_line_rfft_3d(vol.to(device), directions=u, shifts_3d=s)
        ((p - target.to(device)).abs() ** 2).sum().backward()
        return u.grad.cpu(), s.grad.cpu()

    def bp_grads(device):
        u = u0.clone().to(device).requires_grad_(True)
        s = s0.clone().to(device).requires_grad_(True)
        w = wts0.clone().to(device).requires_grad_(True)
        dvol, _ = insert_central_line_rfft_3d(
            lines.to(device), directions=u, shifts_3d=s, weights=w
        )
        (dvol.abs() ** 2).sum().backward()
        return u.grad.cpu(), s.grad.cpu(), w.grad.cpu()

    cu, cs = fwd_grads("cpu")
    gu, gs = fwd_grads(dev)
    assert torch.allclose(gu, cu, atol=1e-3 * cu.abs().max())
    assert torch.allclose(gs, cs, atol=1e-3 * cs.abs().max())

    cu, cs, cw = bp_grads("cpu")
    gu, gs, gw = bp_grads(dev)
    assert torch.allclose(gu, cu, atol=1e-3 * cu.abs().max())
    assert torch.allclose(gs, cs, atol=1e-3 * cs.abs().max() + 1e-6)
    assert torch.allclose(gw, cw, atol=1e-4)
