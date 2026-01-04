import warnings

import torch

from torch_fourier_filter.ctf import (
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


def test_2d_ctf_batch():
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


def test_calculate_additional_phase_shift():
    """Test calculate_additional_phase_shift function."""
    phase_shift_degrees = torch.tensor([0.0, 45.0, 90.0, 180.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_additional_phase_shift(phase_shift_degrees)

        # Check deprecation warning is issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "deprecated" in str(w[0].message).lower()

    # Check result: degrees to radians conversion
    expected = torch.deg2rad(phase_shift_degrees)
    assert torch.allclose(result, expected, atol=1e-6)


def test_calculate_amplitude_contrast_equivalent_phase_shift():
    """Test calculate_amplitude_contrast_equivalent_phase_shift function."""
    amplitude_contrast = torch.tensor([0.0, 0.1, 0.5, 0.9])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_amplitude_contrast_equivalent_phase_shift(amplitude_contrast)

        # Check deprecation warning is issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "deprecated" in str(w[0].message).lower()

    # Check result: arctan formula
    expected = torch.arctan(amplitude_contrast / torch.sqrt(1 - amplitude_contrast**2))
    assert torch.allclose(result, expected, atol=1e-6)


def test_calculate_defocus_phase_aberration():
    """Test calculate_defocus_phase_aberration function."""
    defocus_um = torch.tensor([1.5, 2.0])
    voltage_kv = torch.tensor([300.0, 300.0])
    spherical_aberration_mm = torch.tensor([2.7, 2.7])
    # Create a simple frequency grid (squared)
    fftfreq_grid_angstrom_squared = torch.tensor([[0.01, 0.02], [0.03, 0.04]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_defocus_phase_aberration(
            defocus_um,
            voltage_kv,
            spherical_aberration_mm,
            fftfreq_grid_angstrom_squared,
        )

        # Check deprecation warning is issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "deprecated" in str(w[0].message).lower()

    # Check result shape matches input
    assert result.shape == fftfreq_grid_angstrom_squared.shape
    # Result should be a tensor
    assert isinstance(result, torch.Tensor)


def test_calculate_total_phase_shift():
    """Test calculate_total_phase_shift function."""
    defocus_um = torch.tensor([1.5])
    voltage_kv = torch.tensor([300.0])
    spherical_aberration_mm = torch.tensor([2.7])
    phase_shift_degrees = torch.tensor([0.0])
    amplitude_contrast_fraction = torch.tensor([0.1])
    fftfreq_grid_angstrom_squared = torch.tensor([[0.01, 0.02], [0.03, 0.04]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_total_phase_shift(
            defocus_um,
            voltage_kv,
            spherical_aberration_mm,
            phase_shift_degrees,
            amplitude_contrast_fraction,
            fftfreq_grid_angstrom_squared,
        )

        # Check deprecation warning is issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "deprecated" in str(w[0].message).lower()

    # Check result shape matches frequency grid
    assert result.shape == fftfreq_grid_angstrom_squared.shape
    # Result should be a tensor
    assert isinstance(result, torch.Tensor)


def test_deprecation_warnings():
    """Test that all functions issue deprecation warnings."""
    functions_to_test = [
        (calculate_relativistic_electron_wavelength, (300e3,)),
        (calculate_additional_phase_shift, (torch.tensor([45.0]),)),
        (calculate_amplitude_contrast_equivalent_phase_shift, (torch.tensor([0.1]),)),
        (
            calculate_defocus_phase_aberration,
            (
                torch.tensor([1.5]),
                torch.tensor([300.0]),
                torch.tensor([2.7]),
                torch.tensor([[0.01]]),
            ),
        ),
        (
            calculate_total_phase_shift,
            (
                torch.tensor([1.5]),
                torch.tensor([300.0]),
                torch.tensor([2.7]),
                torch.tensor([0.0]),
                torch.tensor([0.1]),
                torch.tensor([[0.01]]),
            ),
        ),
        (
            calculate_ctf_1d,
            {
                "defocus": 1.5,
                "voltage": 300,
                "spherical_aberration": 2.7,
                "amplitude_contrast": 0.1,
                "phase_shift": 0,
                "pixel_size": 8,
                "n_samples": 10,
                "oversampling_factor": 1,
            },
        ),
        (
            calculate_ctf_2d,
            {
                "defocus": 1.5,
                "astigmatism": 0.0,
                "astigmatism_angle": 0.0,
                "voltage": 300,
                "spherical_aberration": 2.7,
                "amplitude_contrast": 0.1,
                "phase_shift": 0,
                "pixel_size": 8,
                "image_shape": (10, 10),
                "rfft": False,
                "fftshift": False,
            },
        ),
    ]

    for func, args in functions_to_test:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if isinstance(args, dict):
                func(**args)
            else:
                func(*args)

            # Check that at least one FutureWarning was issued
            future_warnings = [
                warn for warn in w if issubclass(warn.category, FutureWarning)
            ]
            assert (
                len(future_warnings) > 0
            ), f"{func.__name__} did not issue FutureWarning"
            assert "deprecated" in str(future_warnings[0].message).lower()


def test_calculate_relativistic_electron_wavelength_tensor():
    """Test calculate_relativistic_electron_wavelength with tensor input."""
    energy = torch.tensor([100e3, 200e3, 300e3])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_relativistic_electron_wavelength(energy)

        # Check deprecation warning is issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)

    # Check result shape
    assert result.shape == energy.shape
    # Check that all wavelengths are positive and reasonable
    assert torch.all(result > 0)
    assert torch.all(result < 1e-10)  # Wavelengths should be very small


def test_calculate_ctf_2d_rfft():
    """Test calculate_ctf_2d with rfft=True."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_ctf_2d(
            defocus=1.5,
            astigmatism=0.0,
            astigmatism_angle=0.0,
            voltage=300,
            spherical_aberration=2.7,
            amplitude_contrast=0.1,
            phase_shift=0,
            pixel_size=8,
            image_shape=(10, 10),
            rfft=True,
            fftshift=False,
        )

        # Check deprecation warning is issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)

    # With rfft=True, the result should have shape (..., H, W//2+1)
    assert result.shape[-1] == 6  # (10//2+1) = 6 for rfft


def test_calculate_ctf_2d_fftshift():
    """Test calculate_ctf_2d with fftshift=True."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = calculate_ctf_2d(
            defocus=1.5,
            astigmatism=0.0,
            astigmatism_angle=0.0,
            voltage=300,
            spherical_aberration=2.7,
            amplitude_contrast=0.1,
            phase_shift=0,
            pixel_size=8,
            image_shape=(10, 10),
            rfft=False,
            fftshift=True,
        )

        # Check deprecation warning is issued
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)

    # Result should have correct shape
    assert result.shape[-2:] == (10, 10)
