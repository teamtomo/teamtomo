"""Tests for Ewald-sphere CTF outputs and weighting."""

import torch

from torch_ctf import calculate_ctfp_and_ctfq_2d, get_ctf_weighting


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
