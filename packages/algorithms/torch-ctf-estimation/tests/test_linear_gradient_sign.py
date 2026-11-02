"""The linear defocus model must report a gradient that reproduces the fitted field."""

import math

import pytest
import torch
from torch_ctf import calculate_ctf_2d

from torch_ctf_estimation.estimate_ctf import estimate_ctf
from torch_ctf_estimation.models import CTFFittingParams, OpticalParams

PIXEL_SPACING = 2.0
SIDELENGTH = 768
BLOCKS = 12
DEFOCUS_0 = 2.0
GRADIENT = 0.6  # µm per normalised unit


def _linear_defocus(model_angle_deg: float, row: float, col: float) -> float:
    """Defocus of the package's linear model; row and col are normalised to [0, 1]."""
    angle = math.radians(model_angle_deg)
    projected = (row - 0.5) * math.cos(angle) + (col - 0.5) * math.sin(angle)
    return DEFOCUS_0 + GRADIENT * projected


def _tilted_ctf_image(model_angle_deg: float, seed: int = 0) -> torch.Tensor:
    """Noise filtered by a CTF whose defocus varies linearly, built block by block."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn((SIDELENGTH, SIDELENGTH), generator=generator)
    spectrum = torch.fft.rfft2(noise)
    image = torch.zeros(SIDELENGTH, SIDELENGTH)
    block = SIDELENGTH // BLOCKS
    for i in range(BLOCKS):
        for j in range(BLOCKS):
            defocus = _linear_defocus(
                model_angle_deg, (i + 0.5) / BLOCKS, (j + 0.5) / BLOCKS
            )
            ctf = calculate_ctf_2d(
                defocus=defocus,
                astigmatism=0.0,
                astigmatism_angle=0.0,
                voltage=300.0,
                spherical_aberration=2.7,
                amplitude_contrast=0.1,
                phase_shift=0.0,
                pixel_size=PIXEL_SPACING,
                image_shape=(SIDELENGTH, SIDELENGTH),
                rfft=True,
                fftshift=False,
            )
            filtered = torch.fft.irfft2(spectrum * ctf, s=(SIDELENGTH, SIDELENGTH))
            rows = slice(i * block, (i + 1) * block)
            cols = slice(j * block, (j + 1) * block)
            image[rows, cols] = filtered[rows, cols]
    return image + 0.3 * torch.randn((SIDELENGTH, SIDELENGTH), generator=generator)


@pytest.mark.parametrize(
    ("angle_deg", "magnitude", "expected_angle", "expected_magnitude"),
    [
        (30.0, 1.0, 30.0, 1.0),
        (210.0, 1.0, 30.0, -1.0),
        (-30.0, 1.0, 150.0, -1.0),
        (-150.0, -2.0, 30.0, 2.0),
        (180.0, 0.5, 0.0, -0.5),
    ],
)
def test_fold_gradient_direction_keeps_the_same_field(
    angle_deg, magnitude, expected_angle, expected_magnitude
):
    """Folding into [0, 180) flips the magnitude so the gradient vector is unchanged."""
    from torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils import (
        _fold_gradient_direction,
    )

    angle, mag = _fold_gradient_direction(angle_deg, magnitude)
    assert angle == pytest.approx(expected_angle)
    assert mag == pytest.approx(expected_magnitude)
    assert 0.0 <= angle < 180.0

    # the folded pair describes the same gradient vector
    def vector(a, m):
        return (m * math.cos(math.radians(a)), m * math.sin(math.radians(a)))

    assert vector(angle, mag) == pytest.approx(vector(angle_deg, magnitude), abs=1e-9)


def test_fold_gradient_direction_tensor():
    """The helper folds per element for tensors (per-knot spline summaries)."""
    from torch_ctf_estimation.estimate_ctf_2d.estimate_ctf_2d_utils import (
        _fold_gradient_direction,
    )

    angles = torch.tensor([30.0, 210.0, -30.0, 180.0])
    magnitudes = torch.tensor([1.0, 1.0, 1.0, 0.5])
    folded, mags = _fold_gradient_direction(angles, magnitudes)
    torch.testing.assert_close(folded, torch.tensor([30.0, 30.0, 150.0, 0.0]))
    torch.testing.assert_close(mags, torch.tensor([1.0, -1.0, -1.0, -0.5]))


@pytest.mark.parametrize("use_1d_defocus_for_spatial", [False, True])
@pytest.mark.parametrize("model_angle_deg", [60.0, 150.0, -30.0, -120.0])
def test_linear_model_reproduces_gradient_direction(
    model_angle_deg, use_1d_defocus_for_spatial
):
    """The returned model matches the true defocus at both ends of the gradient.

    Fitted directions outside [0, 180) used to be folded into that range without
    flipping the magnitude, which reversed the reported gradient. The per-patch 1D
    path used to refine every patch from the centre defocus without a search, so
    patches stayed near the centre and most of the gradient was lost.
    """
    image = _tilted_ctf_image(model_angle_deg)
    optical = OpticalParams(
        pixel_spacing_angstroms=PIXEL_SPACING,
        amplitude_contrast_fraction=0.1,
        target_pixel_spacing_angstroms=PIXEL_SPACING,
    )
    fitting = CTFFittingParams(
        defocus_grid_resolution=(1, 3, 3),
        frequency_fit_range_angstroms=(30.0, 5.0),
        defocus_range_microns=(0.5, 4.0),
        patch_sidelength=128,
        defocus_model="linear",
        optimize_astigmatism=False,
        use_1d_defocus_for_spatial=use_1d_defocus_for_spatial,
    )
    _mean_ps, _result1d, result2d = estimate_ctf(
        image, optical, fitting, device=torch.device("cpu")
    )
    model = result2d.defocus_model
    assert 0.0 <= model.defocus_gradient_angle < 180.0
    angle = math.radians(model_angle_deg)
    fitted_angle = math.radians(model.defocus_gradient_angle)
    fitted_cos, fitted_sin = math.cos(fitted_angle), math.sin(fitted_angle)
    expected, fitted = [], []
    for sign in (1.0, -1.0):
        row = 0.5 + sign * 0.4 * math.cos(angle)
        col = 0.5 + sign * 0.4 * math.sin(angle)
        expected.append(_linear_defocus(model_angle_deg, row, col))
        fitted.append(
            model.defocus_0
            + model.defocus_gradient_magnitude
            * ((row - 0.5) * fitted_cos + (col - 0.5) * fitted_sin)
        )
    assert fitted == pytest.approx(expected, abs=0.06)
