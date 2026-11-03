import torch
from torch_image_interpolation import insert_into_image_2d
from torch_tilt_series import TiltSeries

from torch_projection_matching import align_by_projection_matching, backproject, splat

SIZE = 128
SPACING = 20.0
TILT_ANGLES = torch.arange(-60.0, 61.0, 3.0)
AXIS = 85.0


def render(geometry, points_zyx, amplitudes, sigma_px):
    projected = geometry.project_points(points_zyx) / SPACING
    center = torch.tensor([SIZE // 2, SIZE // 2]).float()
    images = []
    for t in range(projected.shape[1]):
        image, _ = insert_into_image_2d(
            amplitudes,
            coordinates=projected[:, t] + center,
            image=torch.zeros(SIZE, SIZE),
        )
        images.append(image)
    images = torch.stack(images)
    fy = torch.fft.fftfreq(SIZE)[:, None]
    fx = torch.fft.rfftfreq(SIZE)[None, :]
    blur = torch.exp(-2 * (torch.pi * sigma_px) ** 2 * (fy**2 + fx**2))
    return torch.fft.irfft2(torch.fft.rfft2(images) * blur, s=(SIZE, SIZE))


def geometry_with(shifts):
    return TiltSeries(
        tilt_angles=TILT_ANGLES,
        tilt_axis_angle=torch.tensor(AXIS),
        sample_translations=shifts,
        pixel_spacing=SPACING,
    )


def rigid_free_rms(a, b):
    th = torch.deg2rad(TILT_ANGLES).double()
    basis = torch.stack([torch.ones_like(th), torch.cos(th), torch.sin(th)], -1)
    d = (a - b).double()
    coef = torch.linalg.lstsq(basis, d).solution
    return float(torch.sqrt(((d - basis @ coef) ** 2).mean()))


def test_splat_is_adjoint_of_backproject():
    g = torch.Generator().manual_seed(0)
    volume = torch.randn(8, 20, 24, generator=g)
    image = torch.randn(18, 22, generator=g)
    positions = torch.rand(8, 20, 24, 2, generator=g) * torch.tensor([17.0, 21.0])
    lhs = (splat(volume, positions, (18, 22)) * image).sum()
    rhs = (volume * backproject(image, positions)).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_projection_matching_recovers_shifts():
    g = torch.Generator().manual_seed(1)
    n = len(TILT_ANGLES)
    true_shifts = torch.randn(n, 2, generator=g) * 100
    n_points = 4000
    points = torch.empty(n_points, 3)
    points[:, 0] = (torch.rand(n_points, generator=g) - 0.5) * 600
    points[:, 1:] = (torch.rand(n_points, 2, generator=g) - 0.5) * 2.2 * SIZE * SPACING
    images = render(
        geometry_with(true_shifts), points, torch.randn(n_points, generator=g), 1.0
    )
    images += 0.05 * torch.randn(images.shape, generator=g)
    start = geometry_with(true_shifts + torch.randn(n, 2, generator=g) * 50)
    result = align_by_projection_matching(
        images, start, thickness=1000.0, lowpass_angstroms=60.0
    )
    start_error = rigid_free_rms(start.sample_translations, true_shifts)
    error = rigid_free_rms(result.tilt_series.sample_translations, true_shifts)
    assert start_error > 30.0
    assert error < 5.0  # A, a quarter of a pixel
    assert result.correlation.min() > 0.2
    assert result.shift_updates.shape[1:] == (n, 2)
