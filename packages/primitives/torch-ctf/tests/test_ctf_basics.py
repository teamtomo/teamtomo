"""Tests for baseline 1D/2D CTF and phase utilities."""

import pytest
import torch

from torch_ctf import (
    apply_odd_zernikes,
    calculate_additional_phase_shift,
    calculate_amplitude_contrast_equivalent_phase_shift,
    calculate_ctf_1d,
    calculate_ctf_2d,
    calculate_defocus_phase_aberration,
    calculate_relativistic_electron_wavelength,
    calculate_total_phase_shift,
)

EXPECTED_2D = torch.tensor(
    [
        [
            [
                0.1000,
                0.2427,
                0.6287,
                0.9862,
                0.6624,
                -0.5461,
                0.6624,
                0.9862,
                0.6287,
                0.2427,
            ],
            [
                0.2427,
                0.3802,
                0.7344,
                0.9998,
                0.5475,
                -0.6611,
                0.5475,
                0.9998,
                0.7344,
                0.3802,
            ],
            [
                0.6287,
                0.7344,
                0.9519,
                0.9161,
                0.1449,
                -0.9151,
                0.1449,
                0.9161,
                0.9519,
                0.7344,
            ],
            [
                0.9862,
                0.9998,
                0.9161,
                0.4211,
                -0.5461,
                -0.9531,
                -0.5461,
                0.4211,
                0.9161,
                0.9998,
            ],
            [
                0.6624,
                0.5475,
                0.1449,
                -0.5461,
                -0.9998,
                -0.2502,
                -0.9998,
                -0.5461,
                0.1449,
                0.5475,
            ],
            [
                -0.5461,
                -0.6611,
                -0.9151,
                -0.9531,
                -0.2502,
                0.8651,
                -0.2502,
                -0.9531,
                -0.9151,
                -0.6611,
            ],
            [
                0.6624,
                0.5475,
                0.1449,
                -0.5461,
                -0.9998,
                -0.2502,
                -0.9998,
                -0.5461,
                0.1449,
                0.5475,
            ],
            [
                0.9862,
                0.9998,
                0.9161,
                0.4211,
                -0.5461,
                -0.9531,
                -0.5461,
                0.4211,
                0.9161,
                0.9998,
            ],
            [
                0.6287,
                0.7344,
                0.9519,
                0.9161,
                0.1449,
                -0.9151,
                0.1449,
                0.9161,
                0.9519,
                0.7344,
            ],
            [
                0.2427,
                0.3802,
                0.7344,
                0.9998,
                0.5475,
                -0.6611,
                0.5475,
                0.9998,
                0.7344,
                0.3802,
            ],
        ],
        [
            [
                0.1000,
                0.3351,
                0.8755,
                0.7628,
                -0.7326,
                -0.1474,
                -0.7326,
                0.7628,
                0.8755,
                0.3351,
            ],
            [
                0.3351,
                0.5508,
                0.9657,
                0.5861,
                -0.8741,
                0.0932,
                -0.8741,
                0.5861,
                0.9657,
                0.5508,
            ],
            [
                0.8755,
                0.9657,
                0.8953,
                -0.0979,
                -0.9766,
                0.7290,
                -0.9766,
                -0.0979,
                0.8953,
                0.9657,
            ],
            [
                0.7628,
                0.5861,
                -0.0979,
                -0.9648,
                -0.1474,
                0.8998,
                -0.1474,
                -0.9648,
                -0.0979,
                0.5861,
            ],
            [
                -0.7326,
                -0.8741,
                -0.9766,
                -0.1474,
                0.9995,
                -0.5378,
                0.9995,
                -0.1474,
                -0.9766,
                -0.8741,
            ],
            [
                -0.1474,
                0.0932,
                0.7290,
                0.8998,
                -0.5378,
                -0.3948,
                -0.5378,
                0.8998,
                0.7290,
                0.0932,
            ],
            [
                -0.7326,
                -0.8741,
                -0.9766,
                -0.1474,
                0.9995,
                -0.5378,
                0.9995,
                -0.1474,
                -0.9766,
                -0.8741,
            ],
            [
                0.7628,
                0.5861,
                -0.0979,
                -0.9648,
                -0.1474,
                0.8998,
                -0.1474,
                -0.9648,
                -0.0979,
                0.5861,
            ],
            [
                0.8755,
                0.9657,
                0.8953,
                -0.0979,
                -0.9766,
                0.7290,
                -0.9766,
                -0.0979,
                0.8953,
                0.9657,
            ],
            [
                0.3351,
                0.5508,
                0.9657,
                0.5861,
                -0.8741,
                0.0932,
                -0.8741,
                0.5861,
                0.9657,
                0.5508,
            ],
        ],
    ]
)


def test_1d_ctf_single():
    """Test 1D CTF calculation with single values."""
    result = calculate_ctf_1d(
        defocus=1.5,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        n_samples=10,
        oversampling_factor=3,
    )
    expected = torch.tensor(
        [
            0.1033,
            0.1476,
            0.2784,
            0.4835,
            0.7271,
            0.9327,
            0.9794,
            0.7389,
            0.1736,
            -0.5358,
        ]
    )
    assert torch.allclose(result, expected, atol=1e-4)


def test_1d_ctf_batch():
    """Test 1D CTF calculation with batched inputs."""
    result = calculate_ctf_1d(
        defocus=[[[1.5, 2.5]]],
        pixel_size=[[[8, 8]]],
        voltage=[[[300, 300]]],
        spherical_aberration=[[[2.7, 2.7]]],
        amplitude_contrast=[[[0.1, 0.1]]],
        phase_shift=[[[0, 0]]],
        n_samples=10,
        oversampling_factor=1,
    )
    expected = torch.tensor(
        [
            [
                0.1000,
                0.1444,
                0.2755,
                0.4819,
                0.7283,
                0.9385,
                0.9903,
                0.7519,
                0.1801,
                -0.5461,
            ],
            [
                0.1000,
                0.1738,
                0.3880,
                0.6970,
                0.9617,
                0.9237,
                0.3503,
                -0.5734,
                -0.9877,
                -0.1474,
            ],
        ]
    )
    assert result.shape == (1, 1, 2, 10)
    assert torch.allclose(result, expected, atol=1e-4)


def test_calculate_relativistic_electron_wavelength():
    """Check function matches expected value from literature.

    De Graef, Marc (2003-03-27).
    Introduction to Conventional Transmission Electron Microscopy.
    Cambridge University Press. doi:10.1017/cbo9780511615092
    """
    result = calculate_relativistic_electron_wavelength(300e3)
    expected = 1.969e-12
    assert abs(result - expected) < 1e-15


def test_calculate_relativistic_electron_wavelength_tensor():
    """Test relativistic electron wavelength with tensor input."""
    voltages = torch.tensor([100e3, 200e3, 300e3])
    result = calculate_relativistic_electron_wavelength(voltages)
    assert result.shape == (3,)
    assert torch.all(result > 0)
    # Higher voltage should give shorter wavelength
    assert result[0] > result[1] > result[2]


def test_2d_ctf_batch():
    """Test 2D CTF calculation with batched inputs."""
    result = calculate_ctf_2d(
        defocus=[[[1.5, 2.5]]],
        astigmatism=[[[0, 0]]],
        astigmatism_angle=[[[0, 0]]],
        pixel_size=[[[8, 8]]],
        voltage=[[[300, 300]]],
        spherical_aberration=[[[2.7, 2.7]]],
        amplitude_contrast=[[[0.1, 0.1]]],
        phase_shift=[[[0, 0]]],
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
    )
    expected = EXPECTED_2D
    assert result.shape == (1, 1, 2, 10, 10)
    assert torch.allclose(result[0, 0], expected, atol=1e-4)


def test_2d_ctf_astigmatism():
    """Test 2D CTF with astigmatism at different angles."""
    result = calculate_ctf_2d(
        defocus=[2.0, 2.0, 2.5, 2.0],
        astigmatism=[0.5, 1.0, 0.5, 0.5],
        astigmatism_angle=[0, 30, 45, 90],
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
    )
    assert result.shape == (4, 10, 10)

    # First case:
    # Along the X axis the powerspectrum should be like the 2.5 um defocus one
    assert torch.allclose(result[0, 0, :], EXPECTED_2D[1][0, :], atol=1e-4)
    # Along the Y axis the powerspectrum should be like the 1.5 um defocus one
    assert torch.allclose(result[0, :, 0], EXPECTED_2D[0][:, 0], atol=1e-4)

    # Second case:
    # At 30 degrees, X and Y should get half of the astigmatism (cos(60)=0.5),
    # so we still get the same powerspectrum along the axes as in the first case,
    # since the astigmatism is double.
    assert torch.allclose(result[1, 0, :], EXPECTED_2D[1][0, :], atol=1e-4)
    assert torch.allclose(result[1, :, 0], EXPECTED_2D[0][:, 0], atol=1e-4)

    # Third case:
    # At 45 degrees, the powerspectrum should be the same in X and Y and exactly
    # the average defocus (2.5)
    assert torch.allclose(result[2, 0, :], EXPECTED_2D[1][0, :], atol=1e-4)
    assert torch.allclose(result[2, :, 0], EXPECTED_2D[1][:, 0], atol=1e-4)

    # Fourth case:
    # At 90 degrees, we should get 2.5 um defocus in the Y axis
    # and 1.5 um defocus in the X axis.
    assert torch.allclose(result[3, 0, :], EXPECTED_2D[0][0, :], atol=1e-4)
    assert torch.allclose(result[3, :, 0], EXPECTED_2D[1][:, 0], atol=1e-4)


def test_2d_ctf_rfft():
    """Test 2D CTF with rfft=True."""
    result = calculate_ctf_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
    )
    # rfft should give different shape (only non-redundant half)
    assert result.shape == (10, 6)  # (h, w//2+1)


def test_2d_ctf_fftshift():
    """Test 2D CTF with fftshift=True."""
    result = calculate_ctf_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        image_shape=(10, 10),
        rfft=False,
        fftshift=True,
    )
    assert result.shape == (10, 10)
    # With fftshift, DC should be in center
    assert torch.allclose(result[5, 5], torch.tensor(0.1), atol=1e-2)


def test_2d_ctf_transform_matrix():
    """Test 2D CTF with transform_matrix for anisotropic magnification."""
    # Create a simple scaling matrix (1.02x scaling in x, 1.01x scaling in y)
    # This represents anisotropic magnification
    transform_matrix = torch.tensor([[1.02, 0.0], [0.0, 1.01]])

    # Calculate CTF without transform matrix
    result_no_transform = calculate_ctf_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
    )

    # Calculate CTF with transform matrix
    result_with_transform = calculate_ctf_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        transform_matrix=transform_matrix,
    )

    # Both should have the same shape
    assert result_no_transform.shape == (10, 10)
    assert result_with_transform.shape == (10, 10)

    # Both should be finite
    assert torch.all(torch.isfinite(result_no_transform))
    assert torch.all(torch.isfinite(result_with_transform))

    # The transform matrix should change the output (they should be different)
    assert not torch.allclose(result_no_transform, result_with_transform, atol=1e-6)


def test_calculate_defocus_phase_aberration():
    """Test defocus phase aberration calculation."""
    defocus_um = torch.tensor(1.5)
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)
    fftfreq_grid_squared = torch.tensor([0.01, 0.02, 0.03])

    result = calculate_defocus_phase_aberration(
        defocus_um=defocus_um,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
        fftfreq_grid_angstrom_squared=fftfreq_grid_squared,
    )

    assert result.shape == fftfreq_grid_squared.shape
    assert torch.all(torch.isfinite(result))


def test_calculate_additional_phase_shift():
    """Test additional phase shift calculation."""
    phase_shift_degrees = torch.tensor([0.0, 45.0, 90.0, 180.0])
    result = calculate_additional_phase_shift(phase_shift_degrees)

    expected = torch.deg2rad(phase_shift_degrees)
    assert torch.allclose(result, expected)
    assert result.shape == phase_shift_degrees.shape


def test_calculate_amplitude_contrast_equivalent_phase_shift():
    """Test amplitude contrast equivalent phase shift."""
    amplitude_contrast = torch.tensor([0.0, 0.1, 0.2, 0.5])
    result = calculate_amplitude_contrast_equivalent_phase_shift(amplitude_contrast)

    assert result.shape == amplitude_contrast.shape
    assert torch.all(torch.isfinite(result))
    # Should be increasing with amplitude contrast
    assert torch.all(result[1:] > result[:-1])


def test_calculate_total_phase_shift():
    """Test total phase shift calculation."""
    defocus_um = torch.tensor(1.5)
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)
    phase_shift_degrees = torch.tensor(0.0)
    amplitude_contrast_fraction = torch.tensor(0.1)
    fftfreq_grid_squared = torch.tensor([0.01, 0.02, 0.03])

    result = calculate_total_phase_shift(
        defocus_um=defocus_um,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
        phase_shift_degrees=phase_shift_degrees,
        amplitude_contrast_fraction=amplitude_contrast_fraction,
        fftfreq_grid_angstrom_squared=fftfreq_grid_squared,
    )

    assert result.shape == fftfreq_grid_squared.shape
    assert torch.all(torch.isfinite(result))


def test_2d_ctf_with_zernikes():
    """Test 2D CTF calculation with Zernike coefficients."""
    with pytest.warns(RuntimeWarning, match="Both beam tilt and Zernike"):
        result = calculate_ctf_2d(
            defocus=1.5,
            astigmatism=0,
            astigmatism_angle=0,
            voltage=300,
            spherical_aberration=2.7,
            amplitude_contrast=0.1,
            phase_shift=0,
            pixel_size=8,
            image_shape=(10, 10),
            rfft=False,
            fftshift=False,
            beam_tilt_mrad=torch.tensor([[1.0, 2.0]]),
            even_zernike_coeffs={"Z44c": torch.tensor(0.1), "Z60": torch.tensor(0.2)},
            odd_zernike_coeffs={"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)},
        )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # With Zernikes, result should be complex
    assert torch.is_complex(result)


def test_2d_ctf_with_beam_tilt_only():
    """Test 2D CTF calculation with only beam tilt."""
    result = calculate_ctf_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        beam_tilt_mrad=torch.tensor([[1.0, 2.0]]),
    )

    # When scalar inputs are used, output shape is (h, w) not (batch, h, w)
    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    assert torch.is_complex(result)


def test_calculate_ctf_2d_return_complex_ctf_symmetric_path():
    """e^(-i*chi) has unit modulus; imag equals WPOA CTF if antisymmetric phase is 0."""
    common = {
        "defocus": 1.5,
        "astigmatism": 0.0,
        "astigmatism_angle": 0.0,
        "pixel_size": 8.0,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "phase_shift": 15.0,
        "image_shape": (10, 10),
        "rfft": False,
        "fftshift": False,
    }
    ctf_c = calculate_ctf_2d(**common, return_complex_ctf=True)
    ctf_real = calculate_ctf_2d(**common, return_complex_ctf=False)

    assert torch.is_complex(ctf_c)
    assert ctf_c.shape == ctf_real.shape
    assert torch.all(torch.isfinite(ctf_c))
    assert torch.allclose(torch.abs(ctf_c), torch.ones_like(ctf_c.real), atol=1e-5)
    assert torch.allclose(ctf_c.imag, ctf_real, atol=1e-5)


def test_calculate_ctf_2d_return_complex_ctf_rfft_shape_and_unit_modulus():
    """With rfft, output uses the non-redundant half shape and has unit modulus."""
    ctf_c = calculate_ctf_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        pixel_size=8,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        return_complex_ctf=True,
    )
    assert ctf_c.shape == (10, 6)
    assert torch.is_complex(ctf_c)
    assert torch.allclose(torch.abs(ctf_c), torch.ones_like(ctf_c.real), atol=1e-5)


def test_calculate_ctf_2d_return_complex_ctf_beam_tilt_matches_manual_phase():
    """return_complex_ctf includes antisymmetric phase in e^(-i*(chi_s+chi_a))."""
    from torch_ctf._ctf_core import _setup_ctf_context_2d

    kwargs = {
        "defocus": 1.5,
        "astigmatism": 0.0,
        "astigmatism_angle": 0.0,
        "voltage": 300.0,
        "spherical_aberration": 2.7,
        "amplitude_contrast": 0.1,
        "phase_shift": 0.0,
        "pixel_size": 8.0,
        "image_shape": (10, 10),
        "rfft": False,
        "fftshift": False,
        "beam_tilt_mrad": torch.tensor([[1.0, 2.0]]),
    }
    out = calculate_ctf_2d(**kwargs, return_complex_ctf=True)

    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        _,  # fft_freq_grid not used here
        fft_freq_grid_squared,
        rho,
        theta,
    ) = _setup_ctf_context_2d(
        defocus=kwargs["defocus"],
        astigmatism=kwargs["astigmatism"],
        astigmatism_angle=kwargs["astigmatism_angle"],
        voltage=kwargs["voltage"],
        spherical_aberration=kwargs["spherical_aberration"],
        amplitude_contrast=kwargs["amplitude_contrast"],
        phase_shift=kwargs["phase_shift"],
        pixel_size=kwargs["pixel_size"],
        image_shape=kwargs["image_shape"],
        rfft=kwargs["rfft"],
        fftshift=kwargs["fftshift"],
        transform_matrix=None,
    )
    total_phase_shift = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=phase_shift,
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=fft_freq_grid_squared,
    )
    antisymmetric_phase_shift = apply_odd_zernikes(
        odd_zernikes=None,
        rho=rho,
        theta=theta,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        beam_tilt_mrad=kwargs["beam_tilt_mrad"],
    )
    expected = torch.exp(-1j * (total_phase_shift + antisymmetric_phase_shift))
    assert torch.allclose(out, expected, atol=1e-5)
