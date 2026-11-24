"""Sampling at the Nyquist edge of the DFT, on both backends.

The DFT of a sampled volume repeats every ``n`` voxels, so an interpolation stencil
that reaches past the stored rfft has to read the periodic copies (and, for negative
kx, the Hermitian conjugates), not zeros. For even ``n`` this is what fills in the
Nyquist plane, which an fftshifted rfft only stores at ``-n/2`` and a DC-at-origin
rfft only at ``+n/2``. Sample points outside the Nyquist box ``|k_i| <= n/2`` read
as 0.

These compare both backends against ``_reference_sample``, a direct implementation
of that rule, with random rotations and no radius cutoff so that many samples sit
next to the edge, and check the two backends agree everywhere, edge included.
"""

import itertools

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from torch_fourier_slice import (
    extract_central_slices_rfft_2d,
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
    insert_central_slices_rfft_3d,
)
from torch_fourier_slice.experimental import (
    extract_central_line_rfft_2d as mojo_extract_central_line_rfft_2d,
)
from torch_fourier_slice.experimental import (
    extract_central_slices_rfft_3d as mojo_extract_central_slices_rfft_3d,
)
from torch_fourier_slice.experimental import (
    insert_central_slices_rfft_3d as mojo_insert_central_slices_rfft_3d,
)
from torch_fourier_slice.experimental import mojo_kernels_available

requires_mojo = pytest.mark.skipif(
    not mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
)

N_ROTATIONS = 24
# a line only reaches the edge near the axis directions, so use many more
N_LINE_ROTATIONS = 256


def _linear_kernel(s: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - np.abs(s))


def _cubic_kernel(s: np.ndarray) -> np.ndarray:
    """Catmull-Rom (a = -0.5), as in the Mojo kernels."""
    s = np.abs(s)
    a = -0.5
    near = (a + 2) * s**3 - (a + 3) * s**2 + 1
    far = a * s**3 - 5 * a * s**2 + 8 * a * s - 4 * a
    return np.where(s <= 1, near, np.where(s <= 2, far, 0.0))


def _stencil(interpolation: str):
    if interpolation == "cubic":
        return _cubic_kernel, (-1, 0, 1, 2)
    return _linear_kernel, (0, 1)


def _reference_sample(
    dft: np.ndarray, k: np.ndarray, interpolation: str = "linear"
) -> np.ndarray:
    """Interpolate a DC-at-origin rfft of a square/cube at sample coordinates ``k``.

    ``k`` is ``(..., ndim)`` in voxels, ordered like the array's axes. Frequencies
    past the stored grid are read periodically, negative kx as the conjugate at -k,
    and points outside the Nyquist box are 0.
    """
    n = dft.shape[0]
    lo = n // 2 - n + 1

    def wrap(i):
        return np.mod(i - lo, n) + lo

    kernel, offsets = _stencil(interpolation)
    lower = np.floor(k)
    frac = k - lower
    lower = lower.astype(int)
    ndim = k.shape[-1]
    out = np.zeros(k.shape[:-1], dtype=np.complex128)
    for corner in itertools.product(offsets, repeat=ndim):
        weight = np.prod(
            [kernel(frac[..., i] - corner[i]) for i in range(ndim)], axis=0
        )
        idx = [wrap(lower[..., i] + corner[i]) for i in range(ndim)]
        conj = idx[-1] < 0
        idx = [np.where(conj, wrap(-i), i) for i in idx[:-1]] + [np.abs(idx[-1])]
        values = dft[(*(np.mod(i, n) for i in idx[:-1]), idx[-1])]
        out += weight * np.where(conj, values.conj(), values)
    in_box = np.all(np.abs(k) <= n / 2 + 1e-3, axis=-1)
    return np.where(in_box, out, 0)


def _n_edge_samples(k: np.ndarray, n: int, interpolation: str = "linear") -> int:
    """Samples inside the Nyquist box whose stencil leaves the stored grid."""
    _, offsets = _stencil(interpolation)
    lower = np.floor(k).astype(int)
    lo, hi = n // 2 - n + 1, n // 2
    outside = (lower + offsets[0] < lo) | (lower + offsets[-1] > hi)
    in_box = np.all(np.abs(k) <= n / 2 + 1e-3, axis=-1)
    return int(np.sum(np.any(outside, axis=-1) & in_box))


def _rotations(ndim: int) -> torch.Tensor:
    """Random float32 rotation matrices acting on zyx (yx) column vectors."""
    if ndim == 3:
        matrices = Rotation.random(N_ROTATIONS, random_state=7).as_matrix()
    else:
        angles = np.random.default_rng(7).uniform(0, 2 * np.pi, N_LINE_ROTATIONS)
        c, s = np.cos(angles), np.sin(angles)
        matrices = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    return torch.tensor(matrices, dtype=torch.float32)


def _slice_coordinates(n: int, rotations: torch.Tensor) -> np.ndarray:
    """(b, n, n//2+1, 3) zyx sample coordinates of a central slice, DC at origin."""
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.rfftfreq(n) * n, indexing="ij")
    grid = np.stack([np.zeros_like(ky), ky, kx], axis=-1)
    return np.einsum("bij,hwj->bhwi", rotations.double().numpy(), grid)


def _line_coordinates(n: int, rotations: torch.Tensor) -> np.ndarray:
    """(b, n//2+1, 2) yx sample coordinates of a central line, DC at origin."""
    kx = np.fft.rfftfreq(n) * n
    grid = np.stack([np.zeros_like(kx), kx], axis=-1)
    return np.einsum("bij,wj->bwi", rotations.double().numpy(), grid)


def _random_rfft(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """rfft (DC at origin) of a random real array, so Hermitian symmetric."""
    generator = torch.Generator().manual_seed(seed)
    return torch.fft.rfftn(torch.randn(shape, generator=generator))


def _assert_matches(output: torch.Tensor, reference: np.ndarray, dft: torch.Tensor):
    scale = dft.abs().max().item()
    error = np.abs(output.numpy() - reference).max() / scale
    assert error < 1e-5


@pytest.mark.parametrize("sidelength", [16, 15])
def test_torch_slices_match_periodic_reference(sidelength):
    dft = _random_rfft((sidelength,) * 3)
    rotations = _rotations(ndim=3)
    k = _slice_coordinates(sidelength, rotations)
    assert _n_edge_samples(k, sidelength) > 100

    slices = extract_central_slices_rfft_3d(
        torch.fft.fftshift(dft, dim=(-3, -2)), rotations, zyx_matrices=True
    )
    slices = torch.fft.ifftshift(slices, dim=-2)
    _assert_matches(slices, _reference_sample(dft.numpy(), k), dft)


def test_torch_multichannel_slices_match_periodic_reference():
    sidelength = 16
    dft = torch.stack([_random_rfft((sidelength,) * 3, seed=i) for i in range(2)])
    rotations = _rotations(ndim=3)
    k = _slice_coordinates(sidelength, rotations)

    slices = extract_central_slices_rfft_3d_multichannel(
        torch.fft.fftshift(dft, dim=(-3, -2)), rotations, zyx_matrices=True
    )  # (b, c, h, w)
    slices = torch.fft.ifftshift(slices, dim=-2)
    for channel in range(2):
        reference = _reference_sample(dft[channel].numpy(), k)
        _assert_matches(slices[:, channel], reference, dft)


@pytest.mark.parametrize("sidelength", [16, 15])
def test_torch_lines_match_periodic_reference(sidelength):
    dft = _random_rfft((sidelength,) * 2)
    rotations = _rotations(ndim=2)
    k = _line_coordinates(sidelength, rotations)
    if sidelength % 2 == 0:  # odd lines stop at (n - 1) / 2, inside the stored grid
        assert _n_edge_samples(k, sidelength) > 50

    lines = extract_central_slices_rfft_2d(
        torch.fft.fftshift(dft, dim=-2),
        image_shape=(sidelength, sidelength),
        rotation_matrices=rotations,
        yx_matrices=True,
    )
    _assert_matches(lines, _reference_sample(dft.numpy(), k), dft)


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_mojo_slices_match_periodic_reference(interpolation):
    sidelength = 16
    dft = _random_rfft((sidelength,) * 3)
    rotations = _rotations(ndim=3)
    k = _slice_coordinates(sidelength, rotations)
    assert _n_edge_samples(k, sidelength, interpolation) > 100

    # a cutoff past the slice corners, so the Nyquist box is what limits sampling
    slices = mojo_extract_central_slices_rfft_3d(
        dft.contiguous(),
        rotations,
        fourier_radius_cutoff=sidelength,
        interpolation=interpolation,
    )
    reference = _reference_sample(dft.numpy(), k, interpolation)
    _assert_matches(slices, reference, dft)


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_mojo_lines_match_periodic_reference(interpolation):
    sidelength = 16
    dft = _random_rfft((sidelength,) * 2)
    rotations = _rotations(ndim=2)
    k = _line_coordinates(sidelength, rotations)
    assert _n_edge_samples(k, sidelength, interpolation) > 50

    # a line along direction u samples s * u, the second column of the rotation
    directions = rotations[..., :, 1].contiguous()
    lines = mojo_extract_central_line_rfft_2d(
        dft.contiguous(), directions, interpolation=interpolation
    )
    reference = _reference_sample(dft.numpy(), k, interpolation)
    _assert_matches(lines, reference, dft)


@requires_mojo
def test_backends_agree_on_slices():
    """Every pixel, including the slice corners and the Nyquist row."""
    sidelength = 16
    dft = _random_rfft((sidelength,) * 3)
    rotations = _rotations(ndim=3)

    mojo = mojo_extract_central_slices_rfft_3d(
        dft.contiguous(), rotations, fourier_radius_cutoff=sidelength
    )
    canonical = extract_central_slices_rfft_3d(
        torch.fft.fftshift(dft, dim=(-3, -2)), rotations, zyx_matrices=True
    )
    canonical = torch.fft.ifftshift(canonical, dim=-2)
    scale = dft.abs().max()
    assert torch.allclose(mojo, canonical, atol=1e-5 * scale.item(), rtol=0)


@requires_mojo
def test_backends_agree_on_backprojection():
    """Every voxel, except the Friedel bookkeeping of self-conjugate voxels.

    On the kx = 0 plane the voxels at DC and, for even n, at the Nyquist
    frequencies are their own Friedel mirror. The torch backend adds the mirrored
    conjugate into the Nyquist ones (value 2 Re(v), weight 2w) and the Mojo
    backend doesn't (v, w). After dividing by the weights the real parts match,
    and those are all `irfftn` reads, so the reconstructions agree.
    """
    sidelength = 16
    rotations = _rotations(ndim=3)
    generator = torch.Generator().manual_seed(1)
    images = torch.randn((N_ROTATIONS, sidelength, sidelength), generator=generator)
    images_rfft = torch.fft.rfftn(images, dim=(-2, -1))

    mojo, mojo_weights = mojo_insert_central_slices_rfft_3d(
        images_rfft,
        rotations,
        weights=torch.ones_like(images_rfft, dtype=torch.float32),
        fourier_radius_cutoff=sidelength,
    )
    canonical, canonical_weights = insert_central_slices_rfft_3d(
        image_rfft=torch.fft.fftshift(images_rfft, dim=-2),
        volume_shape=(sidelength,) * 3,
        rotation_matrices=rotations,
        zyx_matrices=True,
    )
    canonical = torch.fft.ifftshift(canonical, dim=(-3, -2))
    canonical_weights = torch.fft.ifftshift(canonical_weights, dim=(-3, -2))

    self_conjugate = torch.zeros_like(mojo_weights, dtype=torch.bool)
    for z, y in itertools.product((0, sidelength // 2), repeat=2):
        self_conjugate[z, y, 0] = True
    scale = canonical.abs().max().item()
    assert torch.allclose(
        mojo[~self_conjugate], canonical[~self_conjugate], atol=1e-5 * scale, rtol=0
    )
    assert torch.allclose(
        mojo_weights[~self_conjugate],
        canonical_weights[~self_conjugate],
        atol=1e-5,
        rtol=0,
    )

    def reconstruct(dft, weights):
        normalised = torch.where(weights > 1e-3, dft / weights, 0)
        return torch.fft.irfftn(normalised, s=(sidelength,) * 3)

    mojo_volume = reconstruct(mojo, mojo_weights)
    canonical_volume = reconstruct(canonical, canonical_weights)
    assert torch.allclose(
        mojo_volume, canonical_volume, atol=1e-5 * canonical_volume.abs().max().item()
    )
