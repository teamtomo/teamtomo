"""Tests for the experimental Mojo-backed 2D->1D central-line projectors.

A 2D central line samples an image's rfft along a direction ``u = (u_y, u_x)`` on
the circle: ``line[s] = F(s*u)``, the 1D FT of the image's projection onto the u
axis (Radon row). Validated internally: against the 3D line kernel via the
projection-slice bridge (2D-crop line == 3D-volume line), by adjointness of the
data gradients, by a disk projection-slice check, and an inverse-Radon round trip.
"""

import numpy as np
import pytest
import torch

from torch_fourier_slice.experimental import (
    extract_central_line_rfft_2d,
    extract_central_line_rfft_2d_multivolume,
    extract_central_line_rfft_3d,
    extract_central_slices_rfft_3d,
    insert_central_line_rfft_2d,
    insert_central_line_rfft_2d_multivolume,
    mojo_kernels_available,
)

pytestmark = pytest.mark.skipif(
    not mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
)


def _linear_grad_ratio_ok(loss_fn, param, n=4, scale=0.05, tol=2e-2):
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


def _img_rfft(image: torch.Tensor) -> torch.Tensor:
    return torch.fft.rfftn(
        torch.fft.fftshift(image, dim=(-2, -1)), dim=(-2, -1)
    ).contiguous()


def _rand_dirs(n, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, 2, generator=g)
    return (u / u.norm(dim=-1, keepdim=True)).contiguous()


def _gpu_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


def _gpu_usable():
    dev = _gpu_device()
    if dev is None:
        return False
    try:
        r = _img_rfft(torch.randn(16, 16))
        extract_central_line_rfft_2d(r.to(dev), torch.tensor([0.0, 1.0]))
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Projection-slice bridge: 2D-crop line == 3D-volume line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_2d_crop_line_matches_3d_volume_line(interp):
    """A 2D line through a crop's FT equals the 3D line at u3d = R @ (0, uy, ux)."""
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    d = 40
    g = torch.stack(
        torch.meshgrid(*[torch.arange(d) - d / 2] * 3, indexing="ij"), 0
    ).float()
    vol = sum(
        torch.exp(-((g[0] - a) ** 2 + (g[1] - b) ** 2 + (g[2] - c) ** 2) / 9.0)
        for a, b, c in [(-5, 3, 4), (4, -3, -2), (0, 5, -4)]
    )
    vr = torch.fft.rfftn(
        torch.fft.fftshift(vol, dim=(-3, -2, -1)), dim=(-3, -2, -1)
    ).contiguous()
    R = torch.tensor(
        Rotation.random(1, random_state=3).as_matrix(), dtype=torch.float32
    )
    crop = extract_central_slices_rfft_3d(vr, rotations=R)[0]  # 2D rfft crop
    cut = d / 4.0
    for uy, ux in [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8), (-0.7, 0.7)]:
        u2 = torch.tensor([uy, ux])
        u2 = u2 / u2.norm()
        l2 = extract_central_line_rfft_2d(
            crop, u2, fourier_radius_cutoff=cut, interpolation=interp
        ).reshape(-1)
        u3 = R[0] @ torch.tensor([0.0, u2[0], u2[1]])
        l3 = extract_central_line_rfft_3d(
            vr, directions=u3, fourier_radius_cutoff=cut, interpolation=interp
        ).reshape(-1)
        m = torch.arange(len(l2)) <= cut
        rel = (l2[m] - l3[m]).abs().sum() / (l3[m].abs().sum() + 1e-9)
        assert rel < 0.05  # interp-of-interp vs direct


def test_2d_line_disk_projection_slice():
    """irfft of a disk's 2D line == the analytic chord profile, for any direction."""
    N, r = 128, 20.0
    yy, xx = torch.meshgrid(
        torch.arange(N) - N / 2, torch.arange(N) - N / 2, indexing="ij"
    )
    disk = ((yy**2 + xx**2).sqrt() < r).float()
    dr = _img_rfft(disk)
    t = torch.arange(N) - N / 2
    chord = 2 * torch.sqrt(torch.clamp(r**2 - t**2, min=0))
    for uy, ux in [(1.0, 0.0), (0.0, 1.0), (0.707, 0.707)]:
        u = torch.tensor([uy, ux])
        u = u / u.norm()
        line = extract_central_line_rfft_2d(dr, u).reshape(-1)
        prof = torch.fft.fftshift(torch.fft.irfft(line, n=N))
        c = torch.corrcoef(torch.stack([prof, chord]))[0, 1]
        assert c > 0.99


# ---------------------------------------------------------------------------
# Autograd data gradients + round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_2d_line_extract_gradient(interp):
    torch.manual_seed(0)
    H, P = 64, 6
    Hh = H // 2 + 1
    u = _rand_dirs(P, 1)
    cut = H / 4.0
    w = torch.randn(P, Hh, dtype=torch.complex64)

    def loss(img):
        ln = extract_central_line_rfft_2d(
            img, u, fourier_radius_cutoff=cut, interpolation=interp
        )
        return torch.real(torch.sum(torch.conj(w) * ln))

    img = torch.randn(H, Hh, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(loss, img)


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_2d_line_insert_gradient_and_weights(interp):
    torch.manual_seed(0)
    H, P = 64, 6
    Hh = H // 2 + 1
    u = _rand_dirs(P, 1)
    cut = H / 4.0
    wv = torch.randn(H, Hh, dtype=torch.complex64)

    def loss(lines):
        img, _ = insert_central_line_rfft_2d(
            lines, u, fourier_radius_cutoff=cut, interpolation=interp
        )
        return torch.real(torch.sum(torch.conj(wv) * img))

    lines = torch.randn(P, Hh, dtype=torch.complex64, requires_grad=True)
    assert _linear_grad_ratio_ok(loss, lines)

    weights = torch.rand(P, Hh)
    img, wimg = insert_central_line_rfft_2d(lines.detach(), u, weights=weights)
    assert wimg is not None and wimg.shape == img.shape and wimg.dtype == torch.float32
    _, none_w = insert_central_line_rfft_2d(lines.detach(), u)
    assert none_w is None


def test_2d_line_roundtrip_reconstruction():
    """Lines over 180 directions, inserted + density-compensated, recover the image."""
    N = 48
    yy, xx = torch.meshgrid(
        torch.arange(N) - N / 2, torch.arange(N) - N / 2, indexing="ij"
    )
    ph = ((yy**2 + xx**2).sqrt() < 10).float()
    ph += 0.5 * ((torch.abs(yy - 8) < 3) & (torch.abs(xx + 6) < 9)).float()
    pr = _img_rfft(ph)
    th = torch.linspace(0, np.pi, 180)[:-1]
    U = torch.stack([torch.sin(th), torch.cos(th)], -1)
    lines = extract_central_line_rfft_2d(pr, U)
    img, wimg = insert_central_line_rfft_2d(
        lines, U, weights=torch.ones_like(lines.real)
    )
    recon = torch.fft.ifftshift(
        torch.fft.irfftn(img / wimg.clamp(min=1.0), s=(N, N), dim=(-2, -1)),
        dim=(-2, -1),
    )
    c = torch.corrcoef(torch.stack([ph.flatten(), recon.flatten()]))[0, 1]
    assert c > 0.95


# ---------------------------------------------------------------------------
# Direction + weight gradients
# ---------------------------------------------------------------------------


def _fd_ratio(loss_fn, param, eps, seed=0):
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


def _herm_lines(n, box, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.fft.rfft(torch.randn(n, box, generator=g), dim=-1).contiguous()


def test_2d_line_extract_direction_gradient():
    """Forward 2D-line direction gradient (cubic FD; bicubic is C1 so FD is clean)."""
    torch.manual_seed(0)
    H, P = 40, 3
    Hh = H // 2 + 1
    cut = H / 4.0
    img = torch.randn(H, Hh, dtype=torch.complex64)
    u0 = _rand_dirs(P, 1)
    target = torch.randn(P, Hh, dtype=torch.complex64)

    def loss(u):
        ln = extract_central_line_rfft_2d(
            img, u, fourier_radius_cutoff=cut, interpolation="cubic"
        )
        return ((ln - target).abs() ** 2).sum()

    u = u0.clone().requires_grad_(True)
    assert _fd_ratio(loss, u, eps=3e-4) < 5e-2


def test_2d_line_insert_direction_and_weight_gradients():
    """Insertion direction gradient (cubic FD) and weight gradient (dot-product)."""
    torch.manual_seed(0)
    H, P = 40, 3
    Hh = H // 2 + 1
    cut = H / 4.0
    lines = _herm_lines(P, H, 7)
    u0 = _rand_dirs(P, 1)
    data_t = torch.randn(H, Hh, dtype=torch.complex64)

    def data_loss(u):
        img, _ = insert_central_line_rfft_2d(
            lines, u, fourier_radius_cutoff=cut, interpolation="cubic"
        )
        return ((img - data_t).abs() ** 2).sum()

    u = u0.clone().requires_grad_(True)
    assert _fd_ratio(data_loss, u, eps=3e-4) < 5e-2

    # weight image is linear in the weights -> exact adjoint (dot-product test)
    wc = torch.randn(H, Hh)

    def weight_loss(wts):
        _, wimg = insert_central_line_rfft_2d(
            lines, u0, weights=wts, fourier_radius_cutoff=cut
        )
        return (wimg * wc).sum()

    wts = torch.rand(P, Hh, requires_grad=True)
    assert _linear_grad_ratio_ok(weight_loss, wts)


# ---------------------------------------------------------------------------
# Rank adaptation and GPU parity
# ---------------------------------------------------------------------------


def test_2d_line_rank_single_and_multivolume():
    torch.manual_seed(0)
    H, P, bv = 32, 5, 3
    Hh = H // 2 + 1
    u = _rand_dirs(P, 1)
    imgs = torch.randn(bv, H, Hh, dtype=torch.complex64)

    s0 = extract_central_line_rfft_2d(imgs[0], u)
    assert s0.shape == (P, Hh)

    sm = extract_central_line_rfft_2d_multivolume(imgs, u)
    assert sm.shape == (P, bv, Hh)
    for i in range(bv):
        assert torch.allclose(sm[:, i], extract_central_line_rfft_2d(imgs[i], u))

    lines = torch.randn(P, Hh, dtype=torch.complex64)
    v0, w0 = insert_central_line_rfft_2d(lines, u)
    assert v0.shape == (H, Hh) and w0 is None

    lines_m = torch.randn(P, bv, Hh, dtype=torch.complex64)
    vm, wm = insert_central_line_rfft_2d_multivolume(lines_m, u)
    assert vm.shape == (bv, H, Hh) and wm is None

    v = imgs[0].clone().requires_grad_(True)
    extract_central_line_rfft_2d(v, u).abs().pow(2).sum().backward()
    assert v.grad is not None and v.grad.shape == imgs[0].shape


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_gpu_2d_line_matches_cpu(interp):
    dev = _gpu_device()
    torch.manual_seed(0)
    H, P = 96, 8
    Hh = H // 2 + 1
    rfft = _img_rfft(torch.randn(H, H))
    u = _rand_dirs(P, 3)

    cpu = extract_central_line_rfft_2d(rfft, u, interpolation=interp)
    gpu = extract_central_line_rfft_2d(rfft.to(dev), u, interpolation=interp)
    assert gpu.device.type == dev
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-3)

    lines = torch.randn(P, Hh, dtype=torch.complex64)
    w = torch.rand(P, Hh)
    cv, cw = insert_central_line_rfft_2d(lines, u, weights=w, interpolation=interp)
    gv, gw = insert_central_line_rfft_2d(
        lines.to(dev), u, weights=w.to(dev), interpolation=interp
    )
    assert gv.device.type == dev
    assert torch.allclose(gv.cpu(), cv, atol=1e-4)
    assert torch.allclose(gw.cpu(), cw, atol=1e-4)


@pytest.mark.skipif(not _gpu_usable(), reason="no usable Mojo GPU device")
def test_gpu_2d_line_grads_match_cpu():
    """GPU direction / weight backward kernels reproduce the CPU grads."""
    dev = _gpu_device()
    torch.manual_seed(0)
    H, P = 48, 4
    Hh = H // 2 + 1
    img = torch.randn(H, Hh, dtype=torch.complex64)
    u0 = _rand_dirs(P, 1)
    lines = _herm_lines(P, H, 7)
    wts0 = torch.rand(P, Hh)
    target = torch.randn(P, Hh, dtype=torch.complex64)

    def fwd_dir(device):
        u = u0.clone().to(device).requires_grad_(True)
        ln = extract_central_line_rfft_2d(img.to(device), directions=u)
        ((ln - target.to(device)).abs() ** 2).sum().backward()
        return u.grad.cpu()

    def bp_grads(device):
        u = u0.clone().to(device).requires_grad_(True)
        w = wts0.clone().to(device).requires_grad_(True)
        dv, _ = insert_central_line_rfft_2d(lines.to(device), directions=u, weights=w)
        (dv.abs() ** 2).sum().backward()
        return u.grad.cpu(), w.grad.cpu()

    cu, gu = fwd_dir("cpu"), fwd_dir(dev)
    assert torch.allclose(gu, cu, atol=1e-3 * cu.abs().max())
    (cbu, cbw), (gbu, gbw) = bp_grads("cpu"), bp_grads(dev)
    assert torch.allclose(gbu, cbu, atol=1e-3 * cbu.abs().max())
    assert torch.allclose(gbw, cbw, atol=1e-4)


# ---------------------------------------------------------------------------
# 2D image-translation shifts
# ---------------------------------------------------------------------------


def test_2d_line_shift_phase_ramp():
    """A 2D shift t applies exp(-2pi/N * s*(u.t)) to the line."""
    torch.manual_seed(0)
    H = 48
    cut = H / 4.0
    img = torch.randn(H, H // 2 + 1, dtype=torch.complex64)
    u = _rand_dirs(4, 1)
    t = torch.randn(4, 2)
    l0 = extract_central_line_rfft_2d(img, u, fourier_radius_cutoff=cut)
    ls = extract_central_line_rfft_2d(img, u, shifts_2d=t, fourier_radius_cutoff=cut)
    s = torch.arange(H // 2 + 1).float()
    udott = (u * t).sum(-1)
    phase = -2 * np.pi / H * s[None, :] * udott[:, None]
    ramp = torch.complex(torch.cos(phase), torch.sin(phase))
    m = s <= cut
    assert torch.allclose(ls[:, m], (l0 * ramp)[:, m], atol=1e-4)


def test_2d_line_shift_gradients():
    """Extract and insert gradients w.r.t. the 2D shift (cubic FD)."""
    torch.manual_seed(0)
    H, P = 48, 3
    Hh = H // 2 + 1
    cut = H / 4.0
    img = torch.randn(H, Hh, dtype=torch.complex64)
    u = _rand_dirs(P, 1)
    t0 = torch.randn(P, 2) * 0.5
    target = torch.randn(P, Hh, dtype=torch.complex64)

    def eloss(sh):
        ln = extract_central_line_rfft_2d(
            img, u, shifts_2d=sh, fourier_radius_cutoff=cut, interpolation="cubic"
        )
        return ((ln - target).abs() ** 2).sum()

    sh = t0.clone().requires_grad_(True)
    assert _fd_ratio(eloss, sh, eps=1e-3) < 3e-2

    lines = _herm_lines(P, H, 7)
    data_t = torch.randn(H, Hh, dtype=torch.complex64)

    def iloss(sh):
        img2, _ = insert_central_line_rfft_2d(
            lines, u, shifts_2d=sh, fourier_radius_cutoff=cut, interpolation="cubic"
        )
        return ((img2 - data_t).abs() ** 2).sum()

    sh = t0.clone().requires_grad_(True)
    assert _fd_ratio(iloss, sh, eps=1e-3) < 3e-2
