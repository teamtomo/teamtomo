"""Tests for the experimental real-space projection / backprojection layer.

These cover the wrapper around the rfft kernels: padding, the FFTs, and the
gridding correction that undoes the interpolation kernel's apodization.

The workhorse fixture is an isotropic Gaussian blob. Its projection is the same
in every direction, so ``volume.sum(dim=0)`` is an exact analytic ground truth
for a projection at *any* rotation -- which makes interpolation and correction
error directly measurable.
"""

import pytest
import torch
from scipy.spatial.transform import Rotation

from torch_fourier_slice import project_2d_to_1d as canonical_project_2d_to_1d
from torch_fourier_slice.experimental import (
    backproject_2d_to_3d,
    backproject_2d_to_3d_multichannel,
    mojo_kernels_available,
    project_2d_to_1d,
    project_2d_to_1d_multichannel,
    project_3d_to_2d,
    project_3d_to_2d_multichannel,
)
from torch_fourier_slice.experimental._gridding import (
    _kernel_transform,
    gridding_correction,
)

pytestmark = pytest.mark.skipif(
    not mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
)


def _blob(sidelength: int, sigma: float = 4.0) -> torch.Tensor:
    """Isotropic Gaussian centred in a cubic box."""
    axis = torch.arange(sidelength, dtype=torch.float32) - sidelength // 2
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))


def _rotations(n: int, seed: int = 7) -> torch.Tensor:
    """Random rotation matrices."""
    xyz = torch.tensor(
        Rotation.random(n, random_state=seed).as_matrix(), dtype=torch.float32
    )
    return torch.flip(xyz, dims=(-2, -1)).contiguous()


def _interpolation_kernel(u: torch.Tensor, interpolation: str) -> torch.Tensor:
    """The interpolation kernels themselves, as implemented in `_common.mojo`."""
    a = u.abs()
    if interpolation == "linear":
        return torch.clamp(1 - a, min=0)
    inner = 1.5 * a**3 - 2.5 * a**2 + 1  # Catmull-Rom, Keys a = -1/2
    outer = -0.5 * a**3 + 2.5 * a**2 - 4 * a + 2
    return torch.where(a < 1, inner, torch.where(a < 2, outer, torch.zeros_like(a)))


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_kernel_transform_is_the_fourier_transform_of_the_kernel(interpolation):
    """`_kernel_transform` must be the continuous FT of the interpolation kernel.

    Checked against direct numerical quadrature -- this is what makes the cubic
    correction trustworthy, since it is not the familiar `sinc**2`.
    """
    u = torch.linspace(-2, 2, 400_001, dtype=torch.float64)
    k = _interpolation_kernel(u, interpolation)

    # an interpolating kernel integrates to 1, so K(0) == 1
    assert torch.trapezoid(k, u).item() == pytest.approx(1.0, abs=1e-6)

    for frequency in (0.0, 0.1, 0.25, 0.4, 0.5):
        quadrature = torch.trapezoid(k * torch.cos(2 * torch.pi * frequency * u), u)
        closed_form = _kernel_transform(
            torch.tensor(frequency, dtype=torch.float64), interpolation
        )
        assert closed_form.item() == pytest.approx(quadrature.item(), abs=1e-6)


def test_gridding_correction_is_separable_and_unit_at_the_origin():
    """The correction is the outer product of the per-axis 1D transforms."""
    d = 16
    correction = gridding_correction(d, "cubic")
    assert correction.shape == (d, d, d)
    assert correction[d // 2, d // 2, d // 2].item() == pytest.approx(1.0)

    frequency = torch.fft.fftshift(torch.fft.fftfreq(d))
    k = _kernel_transform(frequency, "cubic")
    assert torch.allclose(
        correction, k[:, None, None] * k[None, :, None] * k[None, None, :]
    )


def test_gridding_correction_rejects_unknown_interpolation():
    with pytest.raises(ValueError, match="interpolation"):
        gridding_correction(8, "nope")


@pytest.mark.parametrize("d", [64, 63])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_projection_matches_analytic_projection(interpolation, d):
    """Projections of an isotropic blob match its exact analytic projection.

    An odd box too: the blob sits on voxel ``d // 2`` either way.
    """
    volume = _blob(d)
    truth = volume.sum(dim=0)
    scale = truth.abs().max()

    images = project_3d_to_2d(volume, _rotations(8), interpolation=interpolation)
    assert images.shape == (8, d, d)
    assert not images.is_complex()

    error = (images - truth[None]).pow(2).mean().sqrt() / scale
    assert error < 1e-3


def test_cubic_projection_beats_linear():
    """The correction follows the interpolation, so cubic stays the better one.

    Applying `sinc**2` (the linear kernel's transform) to a cubic projection
    would leave it *worse* than linear, so this pins the correction to the
    kernel actually used.
    """
    volume = _blob(64)
    truth = volume.sum(dim=0)[None]
    rotations = _rotations(8)

    errors = {
        interpolation: (
            project_3d_to_2d(volume, rotations, interpolation=interpolation) - truth
        )
        .pow(2)
        .mean()
        .sqrt()
        for interpolation in ("linear", "cubic")
    }
    assert 5 * errors["cubic"] < errors["linear"]


@pytest.mark.parametrize("d", [48, 47])
def test_project_backproject_round_trip(d):
    """Reconstructing from dense views recovers the volume."""
    volume = _blob(d, sigma=3.0)
    rotations = _rotations(300, seed=1)

    images = project_3d_to_2d(volume, rotations)
    reconstruction = backproject_2d_to_3d(images, rotations)
    assert reconstruction.shape == (d, d, d)
    assert not reconstruction.is_complex()

    # a weighted backprojection recovers the volume up to a global scale
    reconstruction = reconstruction / reconstruction.max() * volume.max()
    error = (reconstruction - volume).pow(2).mean().sqrt() / volume.max()
    assert error < 1e-3


def _magnified_projection(d: int, centre, matrix: torch.Tensor) -> torch.Tensor:
    """Exact projection along z of the ``_blob(d, 2.0)`` at ``centre`` (zyx), read
    at ``A r``: an image magnified by ``A`` (yx), as ``transform_matrix`` makes it."""
    axis = torch.arange(d, dtype=torch.float32) - d // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    r = torch.stack([yy, xx], dim=-1) @ matrix.T
    sigma = 2.0
    profile = (r[..., 0] - centre[1]) ** 2 + (r[..., 1] - centre[2]) ** 2
    return (2 * torch.pi) ** 0.5 * sigma * torch.exp(-profile / (2 * sigma**2))


def _offset_blob(d: int, centre, sigma: float = 2.0) -> torch.Tensor:
    axis = torch.arange(d, dtype=torch.float32) - d // 2
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    r2 = (zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2
    return torch.exp(-r2 / (2 * sigma**2))


MAGNIFICATIONS = {
    "stretch": torch.tensor([[1.1, 0.0], [0.0, 0.95]]),
    "shear": torch.tensor([[1.04, 0.03], [-0.02, 0.97]]),
}


@pytest.mark.parametrize("d", [64, 63])
@pytest.mark.parametrize("magnification", MAGNIFICATIONS)
@pytest.mark.parametrize(
    ("interpolation", "tolerance"), [("linear", 3e-4), ("cubic", 3e-5)]
)
def test_magnified_projection_matches_analytic_projection(
    interpolation, tolerance, magnification, d
):
    """With ``transform_matrix`` A the image at r is the projection at A r."""
    centre = (2.0, -3.0, 1.0)
    matrix = MAGNIFICATIONS[magnification]
    images = project_3d_to_2d(
        _offset_blob(d, centre),
        torch.eye(3),
        transform_matrix=matrix,
        interpolation=interpolation,
    )
    truth = _magnified_projection(d, centre, matrix)
    error = (images[0] - truth).pow(2).mean().sqrt() / truth.abs().max()
    assert error < tolerance


def test_identity_magnification_changes_nothing():
    volume = _blob(32, sigma=3.0)
    rotations = _rotations(4)
    plain = project_3d_to_2d(volume, rotations)
    identity = project_3d_to_2d(volume, rotations, transform_matrix=torch.eye(2))
    assert torch.equal(plain, identity)


@pytest.mark.parametrize("magnification", MAGNIFICATIONS)
def test_magnified_round_trip(magnification):
    """Backprojecting magnified images with the same matrix recovers the volume,
    scale included: the backprojection undoes the images' 1 / |det A|."""
    d = 48
    volume = _offset_blob(d, (2.0, -3.0, 1.0), sigma=3.0)
    rotations = _rotations(300, seed=1)
    matrix = MAGNIFICATIONS[magnification]

    images = project_3d_to_2d(volume, rotations, transform_matrix=matrix)
    reconstruction = backproject_2d_to_3d(images, rotations, transform_matrix=matrix)
    plain = backproject_2d_to_3d(project_3d_to_2d(volume, rotations), rotations)
    assert torch.allclose(reconstruction, plain, atol=1e-3 * volume.max())


def test_rank_forms_agree():
    """The multichannel forms reproduce the single-volume ones, volume by volume."""
    d, bv, bp = 32, 3, 5
    volumes = torch.stack([_blob(d, sigma=2.0 + i) for i in range(bv)])
    rotations = _rotations(bp)

    stacked = project_3d_to_2d_multichannel(volumes, rotations)
    assert stacked.shape == (bp, bv, d, d)  # pose-major
    for i in range(bv):
        single = project_3d_to_2d(volumes[i], rotations)
        assert torch.allclose(stacked[:, i], single, atol=1e-4)

    volumes_out = backproject_2d_to_3d_multichannel(stacked, rotations)
    assert volumes_out.shape == (bv, d, d, d)
    for i in range(bv):
        single = backproject_2d_to_3d(stacked[:, i], rotations)
        assert torch.allclose(volumes_out[i], single, atol=1e-4)


def test_rank_errors():
    """Passing the wrong rank names the function that does handle it."""
    rotations = _rotations(2)
    with pytest.raises(ValueError, match="project_3d_to_2d_multichannel"):
        project_3d_to_2d(torch.randn(2, 8, 8, 8), rotations)
    with pytest.raises(ValueError, match="backproject_2d_to_3d_multichannel"):
        backproject_2d_to_3d(torch.randn(2, 3, 8, 8), rotations)
    with pytest.raises(ValueError, match="pad_factor"):
        project_3d_to_2d(torch.randn(8, 8, 8), rotations, pad_factor=0.5)
    with pytest.raises(ValueError, match="project_2d_to_1d_multichannel"):
        project_2d_to_1d(torch.randn(2, 8, 8), torch.tensor([1.0, 0.0]))


def test_real_space_layer_is_differentiable():
    """Gradients flow through the FFTs and the gridding correction."""
    d = 24
    volume = _blob(d, sigma=3.0).requires_grad_(True)
    rotations = _rotations(4)

    project_3d_to_2d(volume, rotations).pow(2).sum().backward()
    assert volume.grad is not None
    assert volume.grad.shape == (d, d, d)
    assert torch.isfinite(volume.grad).all()
    assert volume.grad.abs().sum() > 0

    images = torch.randn(4, d, d, requires_grad=True)
    backproject_2d_to_3d(images, rotations).pow(2).sum().backward()
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()


def _gpu_device() -> str | None:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return None


@pytest.mark.skipif(_gpu_device() is None, reason="no GPU device")
@pytest.mark.parametrize("d", [32, 31])
def test_gpu_matches_cpu(d):
    """The real-space layer follows the input device and agrees with the CPU."""
    device = _gpu_device()
    volume = _blob(d, sigma=3.0)
    rotations = _rotations(6)

    cpu = project_3d_to_2d(volume, rotations)
    gpu = project_3d_to_2d(volume.to(device), rotations)
    assert gpu.device.type == device
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-3)

    cpu_volume = backproject_2d_to_3d(cpu, rotations)
    gpu_volume = backproject_2d_to_3d(gpu, rotations)
    assert gpu_volume.device.type == device
    assert torch.allclose(gpu_volume.cpu(), cpu_volume, atol=1e-3)

    matrix = MAGNIFICATIONS["shear"]
    cpu = project_3d_to_2d(volume, rotations, transform_matrix=matrix)
    gpu = project_3d_to_2d(volume.to(device), rotations, transform_matrix=matrix)
    assert torch.allclose(gpu.cpu(), cpu, atol=1e-3)
    cpu_volume = backproject_2d_to_3d(cpu, rotations, transform_matrix=matrix)
    gpu_volume = backproject_2d_to_3d(gpu, rotations, transform_matrix=matrix)
    assert torch.allclose(gpu_volume.cpu(), cpu_volume, atol=1e-3)

    image = _blob_2d(d, sigma=3.0)
    directions = _directions_2d(6)
    cpu_lines = project_2d_to_1d(image, directions)
    gpu_lines = project_2d_to_1d(image.to(device), directions)
    assert gpu_lines.device.type == device
    assert torch.allclose(gpu_lines.cpu(), cpu_lines, atol=1e-3)


# --- 2D image -> 1D lines -----------------------------------------------------------


def _blob_2d(
    sidelength: int, sigma: float = 4.0, centre: tuple[float, float] = (0.0, 0.0)
) -> torch.Tensor:
    """Isotropic 2D Gaussian on pixel ``sidelength // 2``, or ``centre`` (yx) off it."""
    axis = torch.arange(sidelength, dtype=torch.float32) - sidelength // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.exp(-((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) / (2 * sigma**2))


def _angles(n: int, seed: int = 7) -> torch.Tensor:
    return 2 * torch.pi * torch.rand(n, generator=torch.Generator().manual_seed(seed))


def _directions_2d(n: int, seed: int = 7) -> torch.Tensor:
    """Random xy unit vectors."""
    angles = _angles(n, seed)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


@pytest.mark.parametrize("d", [64, 63])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_line_projection_matches_analytic_projection(interpolation, d):
    """Lines of an isotropic 2D blob match its exact projection, in any direction."""
    image = _blob_2d(d)
    truth = image.sum(dim=0)

    lines = project_2d_to_1d(image, _directions_2d(8), interpolation=interpolation)
    assert lines.shape == (8, d)
    assert not lines.is_complex()

    error = (lines - truth[None]).pow(2).mean().sqrt() / truth.abs().max()
    assert error < 1e-3


@pytest.mark.parametrize("d", [64, 63])
def test_line_projection_matches_canonical(d):
    """The canonical function takes an active rotation R; the line is along R's
    first row. (An off-centre blob well inside the box: the canonical function
    also subtracts the image's edge mean before padding and adds back a constant,
    which only matches for edges at ~0.)"""
    image = _blob_2d(d, sigma=3.0, centre=(4.0, -3.0))
    angles = _angles(8)
    cos, sin = angles.cos(), angles.sin()
    rotations = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
    )

    canonical = canonical_project_2d_to_1d(image, rotations)
    lines = project_2d_to_1d(image, rotations[..., 0, :])
    assert torch.allclose(lines, canonical, atol=1e-5 * canonical.abs().max())


def test_line_projection_shift_translates_the_image():
    """``shifts_2d`` (xy) moves the image by that much before it is projected."""
    d, n = 64, 6
    image = _blob_2d(d, sigma=3.0)
    directions = _directions_2d(n)
    shifts = torch.tensor([3.0, -2.0]).expand(n, 2)

    shifted = project_2d_to_1d(
        image, directions, shifts_2d=shifts, interpolation="cubic"
    )
    moved = torch.roll(image, shifts=(-2, 3), dims=(-2, -1))  # (y, x)
    expected = project_2d_to_1d(moved, directions, interpolation="cubic")
    assert torch.allclose(shifted, expected, atol=1e-3 * expected.abs().max())


def test_line_projection_rank_forms_agree():
    """The multichannel form reproduces the single-image one, image by image."""
    d, bv, bp = 32, 3, 5
    images = torch.stack([_blob_2d(d, sigma=2.0 + i) for i in range(bv)])
    directions = _directions_2d(bp)

    stacked = project_2d_to_1d_multichannel(images, directions)
    assert stacked.shape == (bp, bv, d)  # pose-major
    for i in range(bv):
        single = project_2d_to_1d(images[i], directions)
        assert torch.allclose(stacked[:, i], single, atol=1e-4)
