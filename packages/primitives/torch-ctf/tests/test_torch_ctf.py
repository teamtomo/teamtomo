"""Tests for torch_ctf package."""

import pytest
import torch

from torch_ctf import (
    apply_even_zernikes,
    apply_odd_zernikes,
    beam_tilt_to_zernike_coeffs,
    calc_LPP_ctf_2D,
    calc_LPP_phase,
    calculate_additional_phase_shift,
    calculate_amplitude_contrast_equivalent_phase_shift,
    calculate_ctf_1d,
    calculate_ctf_2d,
    calculate_ctfp_and_ctfq_2d,
    calculate_defocus_phase_aberration,
    calculate_relativistic_beta,
    calculate_relativistic_electron_wavelength,
    calculate_relativistic_gamma,
    calculate_total_phase_shift,
    get_ctf_weighting,
    get_eta,
    get_eta0_from_peak_phase_deg,
    initialize_laser_params,
    make_laser_coords,
    resolve_odd_zernikes,
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


def test_calculate_ctfp_and_ctfq_2d():
    """Test CTFP and CTFQ calculation for Ewald sphere correction."""
    ctfp, ctfq = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        discontinuity_angle=0.0,
    )

    # rfft shape: (h, w//2+1)
    assert ctfp.shape == (10, 6)
    assert ctfq.shape == (10, 6)
    assert torch.is_complex(ctfp)
    assert torch.is_complex(ctfq)
    # CTFP and CTFQ should have same magnitude but different phase
    assert torch.allclose(torch.abs(ctfp), torch.abs(ctfq), atol=1e-6)


def test_calculate_ctfp_and_ctfq_2d_discontinuity_angle_zero():
    """Test CTFP and CTFQ with discontinuity_angle=0 (default behavior)."""
    ctfp, ctfq = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        discontinuity_angle=0.0,
    )

    assert ctfp.shape == (10, 6)
    assert ctfq.shape == (10, 6)
    assert torch.is_complex(ctfp)
    assert torch.is_complex(ctfq)

    # With discontinuity_angle=0, should be same as default
    ctfp_default, ctfq_default = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
    )

    assert torch.allclose(ctfp, ctfp_default, atol=1e-6)
    assert torch.allclose(ctfq, ctfq_default, atol=1e-6)


def test_calculate_ctfp_and_ctfq_2d_discontinuity_angle_22_5():
    """Test CTFP and CTFQ with discontinuity_angle=22.5 degrees."""
    ctfp, ctfq = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        discontinuity_angle=22.5,
    )

    assert ctfp.shape == (10, 6)
    assert ctfq.shape == (10, 6)
    assert torch.is_complex(ctfp)
    assert torch.is_complex(ctfq)

    # Get angles to verify mixing behavior
    from torch_ctf.ctf_ewald import _get_fourier_angle

    angles = _get_fourier_angle(
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
    )

    # At angles >= 22.5: ctfp should use ctfp, ctfq should use ctfq
    # At angles < 22.5: ctfp should use ctfq, ctfq should use ctfp
    angle_mask = angles >= 22.5

    # Get reference ctfp and ctfq without mixing
    ctfp_ref, ctfq_ref = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        discontinuity_angle=0.0,
    )

    # Verify mixing: at angles >= 22.5, ctfp should match ctfp_ref
    # and at angles < 22.5, ctfp should match ctfq_ref
    ctfp_high_angle = ctfp[angle_mask]
    ctfp_ref_high_angle = ctfp_ref[angle_mask]
    assert torch.allclose(ctfp_high_angle, ctfp_ref_high_angle, atol=1e-6)

    ctfp_low_angle = ctfp[~angle_mask]
    ctfq_ref_low_angle = ctfq_ref[~angle_mask]
    assert torch.allclose(ctfp_low_angle, ctfq_ref_low_angle, atol=1e-6)

    # Verify ctfq mixing (opposite of ctfp)
    ctfq_high_angle = ctfq[angle_mask]
    ctfq_ref_high_angle = ctfq_ref[angle_mask]
    assert torch.allclose(ctfq_high_angle, ctfq_ref_high_angle, atol=1e-6)

    ctfq_low_angle = ctfq[~angle_mask]
    ctfp_ref_low_angle = ctfp_ref[~angle_mask]
    assert torch.allclose(ctfq_low_angle, ctfp_ref_low_angle, atol=1e-6)


def test_calculate_ctfp_and_ctfq_2d_discontinuity_angle_with_blur():
    """Test CTFP and CTFQ with discontinuity_angle and blurring enabled."""
    # Test with blurring enabled
    ctfp_blur, ctfq_blur = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        discontinuity_angle=22.5,
        blur_at_discontinuity=True,
        blur_distance_degrees=5.0,
    )

    assert ctfp_blur.shape == (10, 6)
    assert ctfq_blur.shape == (10, 6)
    assert torch.is_complex(ctfp_blur)
    assert torch.is_complex(ctfq_blur)

    # Get angles to verify blurring behavior
    from torch_ctf.ctf_ewald import _get_fourier_angle

    angles = _get_fourier_angle(
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
    )

    # Get reference without blurring (sharp transition)
    ctfp_sharp, ctfq_sharp = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        discontinuity_angle=22.5,
        blur_at_discontinuity=False,
    )

    # Verify blurring creates smooth transition
    # Transition region: 22.5 - 5.0 = 17.5 to 22.5 + 5.0 = 27.5
    transition_mask = (angles >= 17.5) & (angles <= 27.5)
    below_transition = angles < 17.5
    above_transition = angles > 27.5

    # Below transition: should match sharp version (both use ctfq for ctfp)
    assert torch.allclose(
        ctfp_blur[below_transition], ctfp_sharp[below_transition], atol=1e-6
    )

    # Above transition: should match sharp version (both use ctfp for ctfp)
    assert torch.allclose(
        ctfp_blur[above_transition], ctfp_sharp[above_transition], atol=1e-6
    )

    # In transition: should be different from sharp (smooth vs sharp)
    # The blurred version should be a weighted mix, not a hard switch
    if transition_mask.any():
        # At the exact discontinuity angle (22.5), should be 50:50 mix
        at_discontinuity = torch.abs(angles - 22.5) < 0.1
        if at_discontinuity.any():
            # Get reference values
            ctfp_ref, ctfq_ref = calculate_ctfp_and_ctfq_2d(
                defocus=1.5,
                astigmatism=0,
                astigmatism_angle=0,
                voltage=300,
                spherical_aberration=2.7,
                amplitude_contrast=0.1,
                phase_shift=0,
                pixel_size=8,
                image_shape=(10, 10),
                rfft=True,
                fftshift=False,
                discontinuity_angle=0.0,
            )

            # At discontinuity, blurred should be approximately 50:50 mix
            # ctfp_blur ≈ 0.5 * ctfp_ref + 0.5 * ctfq_ref
            expected_mix = (
                0.5 * ctfp_ref[at_discontinuity] + 0.5 * ctfq_ref[at_discontinuity]
            )
            assert torch.allclose(ctfp_blur[at_discontinuity], expected_mix, atol=1e-5)

    # Test that blurring with distance=0 is same as no blurring
    ctfp_blur_zero, ctfq_blur_zero = calculate_ctfp_and_ctfq_2d(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        phase_shift=0,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
        discontinuity_angle=22.5,
        blur_at_discontinuity=True,
        blur_distance_degrees=0.0,
    )

    assert torch.allclose(ctfp_blur_zero, ctfp_sharp, atol=1e-6)
    assert torch.allclose(ctfq_blur_zero, ctfq_sharp, atol=1e-6)


def test_get_ctf_weighting():
    """Test CTF weighting factor calculation for Ewald sphere correction."""
    W = get_ctf_weighting(
        defocus_um=1.5,
        voltage_kv=300,
        spherical_aberration_mm=2.7,
        pixel_size=8,
        particle_diameter_angstrom=500.0,
        image_shape=(10, 10),
        rfft=True,
        fftshift=False,
    )

    # Check shape matches rfft output
    assert W.shape == (10, 6)  # (h, w//2+1)

    # W should be real-valued
    assert not torch.is_complex(W)
    assert torch.is_floating_point(W)

    # W should be finite
    assert torch.all(torch.isfinite(W))

    # W should be in reasonable range (typically 0 to 2 based on equation)
    # W = 1 + A(2|sin(χ)| - 1), where A is 0 to 1 and |sin(χ)| is 0 to 1
    # So W ranges from 1 + 0*(2*0 - 1) = 1 to 1 + 1*(2*1 - 1) = 2
    assert torch.all(W >= 0.0)
    assert torch.all(W <= 2.0)

    # Test with different parameters
    W2 = get_ctf_weighting(
        defocus_um=2.0,
        voltage_kv=200,
        spherical_aberration_mm=3.0,
        pixel_size=5,
        particle_diameter_angstrom=1000.0,
        image_shape=(20, 20),
        rfft=True,
        fftshift=False,
    )

    assert W2.shape == (20, 11)  # (h, w//2+1)
    assert torch.all(torch.isfinite(W2))
    assert torch.all(W2 >= 0.0)
    assert torch.all(W2 <= 2.0)

    # Test with fftshift=True
    W3 = get_ctf_weighting(
        defocus_um=1.5,
        voltage_kv=300,
        spherical_aberration_mm=2.7,
        pixel_size=8,
        particle_diameter_angstrom=500.0,
        image_shape=(10, 10),
        rfft=True,
        fftshift=True,
    )

    assert W3.shape == (10, 6)
    assert torch.all(torch.isfinite(W3))


def test_beam_tilt_to_zernike_coeffs():
    """Test beam tilt to Zernike coefficients conversion."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = beam_tilt_to_zernike_coeffs(
        beam_tilt_mrad=beam_tilt_mrad,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z31c" in result
    assert "Z31s" in result
    assert result["Z31c"].shape == (1,)
    assert result["Z31s"].shape == (1,)
    assert torch.all(torch.isfinite(result["Z31c"]))
    assert torch.all(torch.isfinite(result["Z31s"]))


def test_resolve_odd_zernikes_beam_tilt_only():
    """Test resolve_odd_zernikes with only beam tilt."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=beam_tilt_mrad,
        odd_zernike_coeffs=None,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z31c" in result
    assert "Z31s" in result


def test_resolve_odd_zernikes_zernike_only():
    """Test resolve_odd_zernikes with only Zernike coefficients."""
    odd_zernike_coeffs = {"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)}
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=None,
        odd_zernike_coeffs=odd_zernike_coeffs,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert result["Z31c"] == odd_zernike_coeffs["Z31c"]
    assert result["Z31s"] == odd_zernike_coeffs["Z31s"]


def test_resolve_odd_zernikes_both():
    """Test resolve_odd_zernikes with both beam tilt and Zernike coefficients."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    odd_zernike_coeffs = {"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)}
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    with pytest.warns(RuntimeWarning, match="Both beam tilt and Zernike"):
        result = resolve_odd_zernikes(
            beam_tilt_mrad=beam_tilt_mrad,
            odd_zernike_coeffs=odd_zernike_coeffs,
            voltage_kv=voltage_kv,
            spherical_aberration_mm=spherical_aberration_mm,
        )

    # Zernike coefficients should override beam tilt
    assert isinstance(result, dict)
    assert result["Z31c"] == odd_zernike_coeffs["Z31c"]
    assert result["Z31s"] == odd_zernike_coeffs["Z31s"]


def test_resolve_odd_zernikes_none():
    """Test resolve_odd_zernikes with no inputs."""
    result = resolve_odd_zernikes(
        beam_tilt_mrad=None,
        odd_zernike_coeffs=None,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result is None


def test_apply_odd_zernikes():
    """Test applying odd Zernike coefficients."""
    odd_zernikes = {
        "Z31c": torch.tensor(0.1),
        "Z31s": torch.tensor(0.2),
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_trefoil():
    """Test applying trefoil Zernike coefficients."""
    odd_zernikes = {
        "Z33c": torch.tensor(0.1),
        "Z33s": torch.tensor(0.2),
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_invalid():
    """Test applying invalid Zernike coefficient raises error."""
    odd_zernikes = {"Z99": torch.tensor(0.1)}
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(ValueError, match="Unknown odd Zernike"):
        apply_odd_zernikes(
            odd_zernikes=odd_zernikes,
            rho=rho,
            theta=theta,
            voltage_kv=torch.tensor(300.0),
            spherical_aberration_mm=torch.tensor(2.7),
        )


def test_apply_even_zernikes():
    """Test applying even Zernike coefficients."""
    even_zernikes = {
        "Z44c": 0.1,  # plain float
        "Z44s": 0.2,  # plain float
        "Z60": 0.3,  # plain float
    }
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_even_zernikes(
        even_zernikes=even_zernikes,
        total_phase_shift=total_phase_shift,
        rho=rho,
        theta=theta,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # Result should be different from input
    assert not torch.allclose(result, total_phase_shift)


def test_apply_even_zernikes_invalid():
    """Test applying invalid even Zernike coefficient raises error."""
    even_zernikes = {"Z99": 0.1}  # plain float
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(ValueError, match="Unknown even Zernike"):
        apply_even_zernikes(
            even_zernikes=even_zernikes,
            total_phase_shift=total_phase_shift,
            rho=rho,
            theta=theta,
        )


def test_apply_even_zernikes_tensor_coeffs():
    """Test applying even Zernike coefficients with tensor values."""
    even_zernikes = {
        "Z44c": torch.tensor(0.1),
        "Z44s": torch.tensor(0.2),
        "Z60": torch.tensor(0.3),
    }
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_even_zernikes(
        even_zernikes=even_zernikes,
        total_phase_shift=total_phase_shift,
        rho=rho,
        theta=theta,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # Result should be different from input
    assert not torch.allclose(result, total_phase_shift)


def test_apply_even_zernikes_mixed_coeffs():
    """Test applying even Zernike coefficients with mixed float and tensor values."""
    even_zernikes = {
        "Z44c": 0.1,  # float
        "Z44s": torch.tensor(0.2),  # tensor
        "Z60": 0.3,  # float
    }
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_even_zernikes(
        even_zernikes=even_zernikes,
        total_phase_shift=total_phase_shift,
        rho=rho,
        theta=theta,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_even_zernikes_type_error():
    """Test applying even Zernike coefficients with invalid type raises TypeError."""
    even_zernikes = {"Z44c": "invalid"}  # string instead of float/tensor
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(
        TypeError, match=r"Zernike coefficient must be float or torch\.Tensor"
    ):
        apply_even_zernikes(
            even_zernikes=even_zernikes,
            total_phase_shift=total_phase_shift,
            rho=rho,
            theta=theta,
        )


def test_apply_odd_zernikes_none():
    """Test applying odd Zernike coefficients with None returns zeros."""
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=None,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.allclose(result, torch.zeros_like(rho))


def test_apply_odd_zernikes_float_coeffs():
    """Test applying odd Zernike coefficients with float values."""
    odd_zernikes = {
        "Z31c": 0.1,  # plain float
        "Z31s": 0.2,  # plain float
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_mixed_coeffs():
    """Test applying odd Zernike coefficients with mixed float and tensor values."""
    odd_zernikes = {
        "Z31c": 0.1,  # float
        "Z31s": torch.tensor(0.2),  # tensor
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_with_beam_tilt():
    """Test applying odd Zernike coefficients with beam_tilt_mrad parameter."""
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])

    result = apply_odd_zernikes(
        odd_zernikes=None,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
        beam_tilt_mrad=beam_tilt_mrad,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # Should not be all zeros when beam tilt is provided
    assert not torch.allclose(result, torch.zeros_like(rho))


def test_apply_odd_zernikes_type_error():
    """Test applying odd Zernike coefficients with invalid type raises TypeError."""
    odd_zernikes = {"Z31c": "invalid"}  # string instead of float/tensor
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(
        TypeError, match=r"Zernike coefficient must be float or torch\.Tensor"
    ):
        apply_odd_zernikes(
            odd_zernikes=odd_zernikes,
            rho=rho,
            theta=theta,
            voltage_kv=torch.tensor(300.0),
            spherical_aberration_mm=torch.tensor(2.7),
        )


def test_resolve_odd_zernikes_with_trefoil():
    """Test resolve_odd_zernikes with trefoil coefficients (Z33c, Z33s)."""
    odd_zernike_coeffs = {
        "Z33c": torch.tensor(0.1),
        "Z33s": torch.tensor(0.2),
    }
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=None,
        odd_zernike_coeffs=odd_zernike_coeffs,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z33c" in result
    assert "Z33s" in result
    assert result["Z33c"] == odd_zernike_coeffs["Z33c"]
    assert result["Z33s"] == odd_zernike_coeffs["Z33s"]


def test_resolve_odd_zernikes_beam_tilt_with_trefoil():
    """Test resolve_odd_zernikes with beam tilt and trefoil coefficients."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    odd_zernike_coeffs = {
        "Z33c": torch.tensor(0.1),
        "Z33s": torch.tensor(0.2),
    }
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=beam_tilt_mrad,
        odd_zernike_coeffs=odd_zernike_coeffs,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    # Should have both beam tilt coefficients and trefoil
    assert "Z31c" in result
    assert "Z31s" in result
    assert "Z33c" in result
    assert "Z33s" in result
    assert result["Z33c"] == odd_zernike_coeffs["Z33c"]
    assert result["Z33s"] == odd_zernike_coeffs["Z33s"]


def test_beam_tilt_to_zernike_coeffs_broadcasting():
    """Test beam_tilt_to_zernike_coeffs with different tensor shapes."""
    # Test with batched beam tilt
    beam_tilt_mrad = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = beam_tilt_to_zernike_coeffs(
        beam_tilt_mrad=beam_tilt_mrad,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z31c" in result
    assert "Z31s" in result
    assert result["Z31c"].shape == (2,)
    assert result["Z31s"].shape == (2,)
    assert torch.all(torch.isfinite(result["Z31c"]))
    assert torch.all(torch.isfinite(result["Z31s"]))


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


# ============================================================================
# LPP (Laser Phase Plate) Tests
# ============================================================================


def test_calculate_relativistic_gamma():
    """Test relativistic gamma (Lorentz factor) calculation."""
    result = calculate_relativistic_gamma(300e3)
    assert isinstance(result, torch.Tensor)
    assert result > 1.0  # Gamma should always be > 1 for relativistic electrons
    assert torch.all(torch.isfinite(result))

    # Test with tensor input
    voltages = torch.tensor([100e3, 200e3, 300e3])
    result = calculate_relativistic_gamma(voltages)
    assert result.shape == (3,)
    # Higher voltage should give higher gamma
    assert torch.all(result[1:] > result[:-1])


def test_calculate_relativistic_beta():
    """Test relativistic beta (v/c) calculation."""
    result = calculate_relativistic_beta(300e3)
    assert isinstance(result, torch.Tensor)
    assert 0 < result < 1  # Beta should be between 0 and 1
    assert torch.all(torch.isfinite(result))

    # Test with tensor input
    voltages = torch.tensor([100e3, 200e3, 300e3])
    result = calculate_relativistic_beta(voltages)
    assert result.shape == (3,)
    # Higher voltage should give higher beta
    assert torch.all(result[1:] > result[:-1])


def test_initialize_laser_params():
    """Test laser parameter initialization."""
    NA = 0.1
    laser_wavelength_angstrom = 5000.0

    beam_waist, rayleigh_range = initialize_laser_params(NA, laser_wavelength_angstrom)

    assert isinstance(beam_waist, float)
    assert isinstance(rayleigh_range, float)
    assert beam_waist > 0
    assert rayleigh_range > 0
    assert rayleigh_range > beam_waist  # Rayleigh range should be larger


def test_make_laser_coords():
    """Test laser coordinate transformation."""
    # Create a simple frequency grid
    fft_freq_grid = torch.zeros((10, 10, 2))
    fft_freq_grid[..., 0] = torch.linspace(-0.5, 0.5, 10).unsqueeze(1).expand(10, 10)
    fft_freq_grid[..., 1] = torch.linspace(-0.5, 0.5, 10).unsqueeze(0).expand(10, 10)

    electron_wavelength_angstrom = 0.025
    focal_length_angstrom = 1e6
    laser_xy_angle_deg = 45.0
    laser_long_offset_angstrom = 0.0
    laser_trans_offset_angstrom = 0.0
    beam_waist_angstroms = 1000.0
    rayleigh_range_angstroms = 10000.0

    laser_coords = make_laser_coords(
        fft_freq_grid_angstrom=fft_freq_grid,
        electron_wavelength_angstrom=electron_wavelength_angstrom,
        focal_length_angstrom=focal_length_angstrom,
        laser_xy_angle_deg=laser_xy_angle_deg,
        laser_long_offset_angstrom=laser_long_offset_angstrom,
        laser_trans_offset_angstrom=laser_trans_offset_angstrom,
        beam_waist_angstroms=beam_waist_angstroms,
        rayleigh_range_angstroms=rayleigh_range_angstroms,
    )

    assert laser_coords.shape == (10, 10, 2)
    assert torch.all(torch.isfinite(laser_coords))


def test_get_eta():
    """Test eta (phase modulation) calculation."""
    eta0 = 0.1
    laser_coords = torch.zeros((10, 10, 2))
    laser_coords[..., 0] = torch.linspace(-1, 1, 10).unsqueeze(1).expand(10, 10)
    laser_coords[..., 1] = torch.linspace(-1, 1, 10).unsqueeze(0).expand(10, 10)

    beta = 0.5
    NA = 0.1
    pol_angle_deg = 0.0
    xz_angle_deg = 0.0
    laser_phi_deg = 0.0

    eta = get_eta(
        eta0=eta0,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )

    assert eta.shape == (10, 10)
    assert torch.all(torch.isfinite(eta))
    assert torch.all(eta >= 0)  # Eta should be non-negative


def test_get_eta0_from_peak_phase_deg():
    """Test eta0 calibration from peak phase."""
    peak_phase_deg = 90.0
    laser_coords = torch.zeros((10, 10, 2))
    laser_coords[..., 0] = torch.linspace(-1, 1, 10).unsqueeze(1).expand(10, 10)
    laser_coords[..., 1] = torch.linspace(-1, 1, 10).unsqueeze(0).expand(10, 10)

    beta = 0.5
    NA = 0.1
    pol_angle_deg = 0.0
    xz_angle_deg = 0.0
    laser_phi_deg = 0.0

    eta0 = get_eta0_from_peak_phase_deg(
        peak_phase_deg=peak_phase_deg,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )

    assert isinstance(eta0, torch.Tensor)
    assert torch.all(torch.isfinite(eta0))
    assert eta0 > 0

    # Verify that the calibrated eta0 produces the desired peak phase
    eta = get_eta(
        eta0=eta0,
        laser_coords=laser_coords,
        beta=beta,
        NA=NA,
        pol_angle_deg=pol_angle_deg,
        xz_angle_deg=xz_angle_deg,
        laser_phi_deg=laser_phi_deg,
    )
    actual_peak_deg = torch.rad2deg(eta.max())
    assert torch.allclose(actual_peak_deg, torch.tensor(peak_phase_deg), atol=1.0)


def test_calc_LPP_phase():
    """Test LPP phase calculation."""
    # Create frequency grid
    fft_freq_grid = torch.zeros((10, 10, 2))
    fft_freq_grid[..., 0] = torch.linspace(-0.5, 0.5, 10).unsqueeze(1).expand(10, 10)
    fft_freq_grid[..., 1] = torch.linspace(-0.5, 0.5, 10).unsqueeze(0).expand(10, 10)

    NA = 0.1
    laser_wavelength_angstrom = 5000.0
    focal_length_angstrom = 1e6
    laser_xy_angle_deg = 0.0
    laser_xz_angle_deg = 0.0
    laser_long_offset_angstrom = 0.0
    laser_trans_offset_angstrom = 0.0
    laser_polarization_angle_deg = 0.0
    peak_phase_deg = 90.0
    voltage = 300.0

    lpp_phase = calc_LPP_phase(
        fft_freq_grid=fft_freq_grid,
        NA=NA,
        laser_wavelength_angstrom=laser_wavelength_angstrom,
        focal_length_angstrom=focal_length_angstrom,
        laser_xy_angle_deg=laser_xy_angle_deg,
        laser_xz_angle_deg=laser_xz_angle_deg,
        laser_long_offset_angstrom=laser_long_offset_angstrom,
        laser_trans_offset_angstrom=laser_trans_offset_angstrom,
        laser_polarization_angle_deg=laser_polarization_angle_deg,
        peak_phase_deg=peak_phase_deg,
        voltage=voltage,
    )

    assert lpp_phase.shape == (10, 10)
    assert torch.all(torch.isfinite(lpp_phase))


def test_calc_LPP_ctf_2D():
    """Test LPP-modified CTF calculation."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # LPP CTF should be real (not complex) when no odd Zernikes
    assert not torch.is_complex(result)


def test_calc_LPP_ctf_2D_with_zernikes():
    """Test LPP CTF with Zernike coefficients."""
    with pytest.warns(RuntimeWarning, match="Both beam tilt and Zernike"):
        result = calc_LPP_ctf_2D(
            defocus=1.5,
            astigmatism=0,
            astigmatism_angle=0,
            voltage=300,
            spherical_aberration=2.7,
            amplitude_contrast=0.1,
            pixel_size=8,
            image_shape=(10, 10),
            rfft=False,
            fftshift=False,
            NA=0.1,
            laser_wavelength_angstrom=5000.0,
            focal_length_angstrom=1e6,
            laser_xy_angle_deg=0.0,
            laser_xz_angle_deg=0.0,
            laser_long_offset_angstrom=0.0,
            laser_trans_offset_angstrom=0.0,
            laser_polarization_angle_deg=0.0,
            peak_phase_deg=90.0,
            beam_tilt_mrad=torch.tensor([[1.0, 2.0]]),
            even_zernike_coeffs={"Z44c": torch.tensor(0.1), "Z60": torch.tensor(0.2)},
            odd_zernike_coeffs={"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)},
        )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # With Zernikes, result should be complex
    assert torch.is_complex(result)


def test_calc_LPP_ctf_2D_with_beam_tilt_only():
    """Test LPP CTF with only beam tilt."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        beam_tilt_mrad=torch.tensor([[1.0, 2.0]]),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    assert torch.is_complex(result)


def test_calc_LPP_ctf_2D_with_even_zernikes():
    """Test LPP CTF with only even Zernike coefficients."""
    result = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
        even_zernike_coeffs={"Z44c": torch.tensor(0.1), "Z60": torch.tensor(0.2)},
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # With only even Zernikes, result should be real
    assert not torch.is_complex(result)


def test_calc_LPP_ctf_2D_with_transform_matrix():
    """Test LPP CTF with transform_matrix for anisotropic magnification."""
    # Create a simple scaling matrix (1.02x scaling in x, 1.01x scaling in y)
    # This represents anisotropic magnification
    transform_matrix = torch.tensor([[1.02, 0.0], [0.0, 1.01]])

    # Calculate LPP CTF without transform matrix
    result_no_transform = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
    )

    # Calculate LPP CTF with transform matrix
    result_with_transform = calc_LPP_ctf_2D(
        defocus=1.5,
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300,
        spherical_aberration=2.7,
        amplitude_contrast=0.1,
        pixel_size=8,
        image_shape=(10, 10),
        rfft=False,
        fftshift=False,
        NA=0.1,
        laser_wavelength_angstrom=5000.0,
        focal_length_angstrom=1e6,
        laser_xy_angle_deg=0.0,
        laser_xz_angle_deg=0.0,
        laser_long_offset_angstrom=0.0,
        laser_trans_offset_angstrom=0.0,
        laser_polarization_angle_deg=0.0,
        peak_phase_deg=90.0,
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
