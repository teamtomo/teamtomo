import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_fourier_filter.envelopes import (
    Cc_envelope,
    Cs_envelope,
    _b_envelope,
    _Cc_envelope,
    _Cs_envelope,
    _dose_envelope,
    b_envelope,
    dose_envelope,
)


def test_bfactor_2d():
    # Generate an image
    image = torch.zeros((4, 4))
    image[1:, :] = 1
    # apply bfactor
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    b_env = b_envelope(
        B=10,
        image_shape=image.shape,
        pixel_size=1,
        rfft=True,
        fftshift=False,
        device=dft.device,
    )
    bfactored_dft = dft * b_env
    result = torch.real(torch.fft.irfftn(bfactored_dft, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [
                0.18851198,
                0.18851198,
                0.18851198,
                0.18851198,
            ],
            [0.88381535, 0.88381535, 0.88381535, 0.88381535],
            [1.0438573, 1.0438573, 1.0438573, 1.0438573],
            [0.88381535, 0.88381535, 0.88381535, 0.88381535],
        ]
    )
    assert torch.allclose(result, expected, atol=1e-3)


def test_dose_envelope():
    # Generate an image
    fluence = 20
    image = torch.zeros((4, 4))
    image[1:, :] = 1
    # apply bfactor
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    dose_env = dose_envelope(
        fluence=fluence,
        image_shape=image.shape,
        pixel_size=1,
        rfft=True,
        fftshift=False,
        device=dft.device,
    )
    dft_env = dft * dose_env
    result = torch.real(torch.fft.irfftn(dft_env, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [0.7495, 0.7495, 0.7495, 0.7495],
            [0.7500, 0.7500, 0.7500, 0.7500],
            [0.7505, 0.7505, 0.7505, 0.7505],
            [0.7500, 0.7500, 0.7500, 0.7500],
        ]
    )
    assert torch.allclose(result, expected, atol=1e-3)


def test_Cc_envelope():
    chromatic_aberration = 2.0  # in mm
    image_shape = (4, 4)
    pixel_size = 1.0  # in angstroms
    rfft = True
    fftshift = False
    device = torch.device("cpu")
    voltage = 300  # in kV
    energy_spread = 0.7  # in eV
    deltaV_V = 0.06e-6
    deltaI_I = 0.01e-6

    Cc_env = Cc_envelope(
        chromatic_aberration,
        image_shape,
        pixel_size,
        rfft,
        fftshift,
        device,
        voltage,
        energy_spread,
        deltaV_V,
        deltaI_I,
    )

    assert isinstance(Cc_env, torch.Tensor), "Output is not a torch.Tensor"
    assert Cc_env.min() >= 0.0 and Cc_env.max() <= 1.0, (
        "Values are out of expected range"
    )

    # Test with real values
    image = torch.zeros((4, 4))
    image[1:, :] = 1
    dft = torch.fft.rfftn(image, dim=(-2, -1))

    Cc_dft = dft * Cc_env
    result = torch.real(torch.fft.irfftn(Cc_dft, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [0.0654, 0.0654, 0.0654, 0.0654],
            [0.9427, 0.9427, 0.9427, 0.9427],
            [1.0493, 1.0493, 1.0493, 1.0493],
            [0.9427, 0.9427, 0.9427, 0.9427],
        ]
    )

    assert torch.allclose(result, expected, atol=1e-3)


def test_bfactor_2d_low_level():
    # low-level math util applied to a manually-built frequency grid should
    # reproduce the same result as the high-level wrapper in test_bfactor_2d,
    # since callers with a cached grid call it directly.
    image = torch.zeros((4, 4))
    image[1:, :] = 1
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    frequency_grid_px = fftfreq_grid(
        image_shape=image.shape, rfft=True, fftshift=False, norm=True
    )
    b_env = _b_envelope(frequency_grid_px, B=10)
    bfactored_dft = dft * b_env
    result = torch.real(torch.fft.irfftn(bfactored_dft, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [
                0.18851198,
                0.18851198,
                0.18851198,
                0.18851198,
            ],
            [0.88381535, 0.88381535, 0.88381535, 0.88381535],
            [1.0438573, 1.0438573, 1.0438573, 1.0438573],
            [0.88381535, 0.88381535, 0.88381535, 0.88381535],
        ]
    )
    assert torch.allclose(result, expected, atol=1e-3)


def test_dose_envelope_low_level():
    fluence = 20
    image = torch.zeros((4, 4))
    image[1:, :] = 1
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    frequency_grid_px = fftfreq_grid(
        image_shape=image.shape, rfft=True, fftshift=False, norm=True
    )
    dose_env = _dose_envelope(frequency_grid_px, fluence=fluence)
    dft_env = dft * dose_env
    result = torch.real(torch.fft.irfftn(dft_env, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [0.7495, 0.7495, 0.7495, 0.7495],
            [0.7500, 0.7500, 0.7500, 0.7500],
            [0.7505, 0.7505, 0.7505, 0.7505],
            [0.7500, 0.7500, 0.7500, 0.7500],
        ]
    )
    assert torch.allclose(result, expected, atol=1e-3)


def test_Cc_envelope_low_level():
    chromatic_aberration = 2.0  # in mm
    image_shape = (4, 4)
    voltage = 300  # in kV
    energy_spread = 0.7  # in eV
    deltaV_V = 0.06e-6
    deltaI_I = 0.01e-6

    frequency_grid_px = fftfreq_grid(
        image_shape=image_shape, rfft=True, fftshift=False, norm=True
    )
    Cc_env = _Cc_envelope(
        frequency_grid_px,
        chromatic_aberration,
        voltage,
        energy_spread,
        deltaV_V,
        deltaI_I,
    )

    assert isinstance(Cc_env, torch.Tensor), "Output is not a torch.Tensor"
    assert Cc_env.min() >= 0.0 and Cc_env.max() <= 1.0, (
        "Values are out of expected range"
    )

    image = torch.zeros((4, 4))
    image[1:, :] = 1
    dft = torch.fft.rfftn(image, dim=(-2, -1))

    Cc_dft = dft * Cc_env
    result = torch.real(torch.fft.irfftn(Cc_dft, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [0.0654, 0.0654, 0.0654, 0.0654],
            [0.9427, 0.9427, 0.9427, 0.9427],
            [1.0493, 1.0493, 1.0493, 1.0493],
            [0.9427, 0.9427, 0.9427, 0.9427],
        ]
    )

    assert torch.allclose(result, expected, atol=1e-3)


def test_Cs_envelope_low_level():
    spherical_aberration = 2.0  # in mm
    defocus = 1.0  # in microns,
    image_shape = (4, 4)
    voltage = 300  # in kV
    alpha = 0.005  # in mrad

    frequency_grid_px = fftfreq_grid(
        image_shape=image_shape, rfft=True, fftshift=False, norm=True
    )
    Cs_env = _Cs_envelope(
        frequency_grid_px,
        spherical_aberration,
        defocus,
        voltage,
        alpha,
    )

    assert isinstance(Cs_env, torch.Tensor), "Output is not a torch.Tensor"
    assert Cs_env.min() >= 0.0 and Cs_env.max() <= 1.0, (
        "Values are out of expected range"
    )

    image = torch.zeros((4, 4))
    image[1:, :] = 1
    dft = torch.fft.rfftn(image, dim=(-2, -1))

    Cs_dft = dft * Cs_env
    result = torch.real(torch.fft.irfftn(Cs_dft, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [0.0030, 0.0030, 0.0030, 0.0030],
            [0.9978, 0.9978, 0.9978, 0.9978],
            [1.0013, 1.0013, 1.0013, 1.0013],
            [0.9978, 0.9978, 0.9978, 0.9978],
        ]
    )

    assert torch.allclose(result, expected, atol=1e-3)


def test_Cs_envelope():
    spherical_aberration = 2.0  # in mm
    defocus = 1.0  # in microns,
    image_shape = (4, 4)
    pixel_size = 1.0  # in angstroms
    rfft = True
    fftshift = False
    device = torch.device("cpu")
    voltage = 300  # in kV
    alpha = 0.005  # in mrad

    Cs_env = Cs_envelope(
        spherical_aberration,
        defocus,
        image_shape,
        pixel_size,
        rfft,
        fftshift,
        device,
        voltage,
        alpha,
    )

    assert isinstance(Cs_env, torch.Tensor), "Output is not a torch.Tensor"
    assert Cs_env.min() >= 0.0 and Cs_env.max() <= 1.0, (
        "Values are out of expected range"
    )

    # Test with real values
    image = torch.zeros((4, 4))
    image[1:, :] = 1
    dft = torch.fft.rfftn(image, dim=(-2, -1))

    Cs_dft = dft * Cs_env
    result = torch.real(torch.fft.irfftn(Cs_dft, dim=(-2, -1)))

    expected = torch.tensor(
        [
            [0.0030, 0.0030, 0.0030, 0.0030],
            [0.9978, 0.9978, 0.9978, 0.9978],
            [1.0013, 1.0013, 1.0013, 1.0013],
            [0.9978, 0.9978, 0.9978, 0.9978],
        ]
    )

    assert torch.allclose(result, expected, atol=1e-3)
