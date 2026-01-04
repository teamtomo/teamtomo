import torch

from torch_fourier_filter.dose_weight import (
    critical_exposure,
    critical_exposure_bfactor,
    cumulative_dose_filter_3d,
    dose_weight_movie,
)


def test_critical_exposure():
    fft_freq = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    expected_output = torch.tensor([14.1383, 6.3823, 4.6287, 3.9365, 3.5869])
    output = critical_exposure(fft_freq)
    assert torch.allclose(
        output, expected_output, atol=1e-3
    ), "critical_exposure output mismatch"


def test_critical_exposure_bfactor():
    fft_freq = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    bfac = 1.0
    expected_output = torch.tensor([400.0, 100.0, 44.444, 25.0, 16.0])
    output = critical_exposure_bfactor(fft_freq, bfac)
    assert torch.allclose(
        output, expected_output, atol=1e-3
    ), "critical_exposure_bfactor output mismatch"


def test_dose_weight_movie():
    # Test parameters
    n_frames = 5
    image_shape = (16, 16)
    pixel_size = 1.0
    pre_exposure = 0.0
    dose_per_frame = 2.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a simple movie in Fourier space (rfft)
    real_frames = torch.rand((n_frames, *image_shape), device=device)
    movie_dft = torch.fft.rfft2(real_frames)

    # Test basic functionality
    weighted_movie = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        rfft=True,
        device=device,
    )

    # Check output shape matches input shape
    assert weighted_movie.shape == movie_dft.shape, "Output shape mismatch"

    # Check output is complex
    assert weighted_movie.dtype == movie_dft.dtype, "Output dtype mismatch"

    # Check that all values are finite
    assert torch.all(torch.isfinite(weighted_movie)), "Non-finite values in output"

    # Test with different voltage settings (200kV should apply damage correction)
    weighted_movie_200kv = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        voltage=200.0,
        rfft=True,
        device=device,
    )

    # Results should be different due to voltage correction
    assert not torch.allclose(
        weighted_movie, weighted_movie_200kv
    ), "200kV correction not applied"

    # Test with 100kV voltage
    weighted_movie_100kv = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        voltage=100.0,
        rfft=True,
        device=device,
    )

    # Results should be different due to voltage correction
    assert not torch.allclose(
        weighted_movie, weighted_movie_100kv
    ), "100kV correction not applied"

    # Test with custom B-factor
    weighted_movie_bfactor = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        crit_exposure_bfactor=10.0,
        rfft=True,
        device=device,
    )

    # Results should be different with custom B-factor
    assert not torch.allclose(
        weighted_movie, weighted_movie_bfactor
    ), "Custom B-factor not applied"

    # Test with full FFT (not rfft)
    movie_dft_full = torch.fft.fft2(real_frames)
    weighted_movie_full = dose_weight_movie(
        movie_dft=movie_dft_full,
        image_shape=image_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        rfft=False,
        device=device,
    )

    assert (
        weighted_movie_full.shape == movie_dft_full.shape
    ), "Full FFT output shape mismatch"

    # Test device handling - movie on different device than specified
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda:0")

        # Create movie on CPU
        cpu_movie = torch.fft.rfft2(torch.rand((n_frames, *image_shape)))

        # Process with CUDA device specified - should work and move tensor
        weighted_cuda = dose_weight_movie(
            movie_dft=cpu_movie,
            image_shape=image_shape,
            pixel_size=pixel_size,
            device=cuda_device,
        )

        assert weighted_cuda.device == cuda_device, "Output not on specified device"

    # Test error handling - wrong input dimensions
    try:
        dose_weight_movie(
            movie_dft=movie_dft[0],  # Remove frames dimension
            image_shape=image_shape,
            pixel_size=pixel_size,
            device=device,
        )
        raise AssertionError("Should have raised ValueError for wrong dimensions")
    except ValueError as e:
        assert "3D tensor" in str(e), "Wrong error message for dimension check"

    # Test error handling - invalid B-factor
    try:
        dose_weight_movie(
            movie_dft=movie_dft,
            image_shape=image_shape,
            pixel_size=pixel_size,
            crit_exposure_bfactor=-10.0,  # Invalid negative B-factor
            device=device,
        )
        raise AssertionError("Should have raised ValueError for invalid B-factor")
    except ValueError as e:
        assert "B-factor must be positive" in str(
            e
        ), "Wrong error message for B-factor check"


def test_cumulative_dose_filter_3d():
    # Test parameters
    volume_shape = (32, 32, 32)
    start_exposure = 0.0
    end_exposure = 10.0
    pixel_size = 1.0
    crit_exposure_bfactor = -1
    rfft = True
    fftshift = False

    # Call the function
    dose_filter = cumulative_dose_filter_3d(
        volume_shape=volume_shape,
        pixel_size=pixel_size,
        start_exposure=start_exposure,
        end_exposure=end_exposure,
        crit_exposure_bfactor=crit_exposure_bfactor,
        rfft=rfft,
        fftshift=fftshift,
    )

    # Check if the values are within a reasonable range
    # TODO: Test these against known, static values rather than just range
    assert torch.all(dose_filter >= 0) and torch.all(
        dose_filter <= 1
    ), "Dose filter values out of range"

    # Test with different bfac values
    # TODO: Test these against known, static values rather than just range
    crit_bfac_values = [0.5, 1.0, 2.0]
    for bfac in crit_bfac_values:
        dose_filter = cumulative_dose_filter_3d(
            volume_shape=volume_shape,
            pixel_size=pixel_size,
            start_exposure=start_exposure,
            end_exposure=end_exposure,
            crit_exposure_bfactor=bfac,
            rfft=rfft,
            fftshift=fftshift,
        )
        assert torch.all(dose_filter >= 0) and torch.all(
            dose_filter <= 1
        ), f"Dose filter values out of range for bfac={bfac}"


def test_memory_efficient_consistency():
    """Test that memory-efficient method produces same results as original method."""
    # Test parameters
    n_frames = 10
    image_shape = (32, 32)
    pixel_size = 1.0
    pre_exposure = 0.0
    dose_per_frame = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a test movie in Fourier space
    real_frames = torch.rand((n_frames, *image_shape), device=device)
    movie_dft = torch.fft.rfft2(real_frames)

    # Test with original method
    result_original = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        memory_efficient=False,
        rfft=True,
        device=device,
    )

    # Test with memory-efficient method
    result_memory_efficient = dose_weight_movie(
        movie_dft=movie_dft,
        image_shape=image_shape,
        pixel_size=pixel_size,
        pre_exposure=pre_exposure,
        dose_per_frame=dose_per_frame,
        memory_efficient=True,
        chunk_size=3,  # Use small chunk size to test chunking
        rfft=True,
        device=device,
    )

    # Results should be identical
    assert torch.allclose(
        result_original, result_memory_efficient, atol=1e-6
    ), "Memory-efficient method produces different results than original method"

    # Test with different voltage settings
    for voltage in [100.0, 200.0, 300.0]:
        result_orig_voltage = dose_weight_movie(
            movie_dft=movie_dft,
            image_shape=image_shape,
            pixel_size=pixel_size,
            pre_exposure=pre_exposure,
            dose_per_frame=dose_per_frame,
            voltage=voltage,
            memory_efficient=False,
            rfft=True,
            device=device,
        )

        result_mem_voltage = dose_weight_movie(
            movie_dft=movie_dft,
            image_shape=image_shape,
            pixel_size=pixel_size,
            pre_exposure=pre_exposure,
            dose_per_frame=dose_per_frame,
            voltage=voltage,
            memory_efficient=True,
            chunk_size=3,
            rfft=True,
            device=device,
        )

        assert torch.allclose(
            result_orig_voltage, result_mem_voltage, atol=1e-6
        ), f"Memory-efficient method differs from original for voltage={voltage}"

    # Test with different chunk sizes to ensure chunking works correctly
    for chunk_size in [1, 2, 5, 8]:
        result_chunked = dose_weight_movie(
            movie_dft=movie_dft,
            image_shape=image_shape,
            pixel_size=pixel_size,
            pre_exposure=pre_exposure,
            dose_per_frame=dose_per_frame,
            memory_efficient=True,
            chunk_size=chunk_size,
            rfft=True,
            device=device,
        )

        assert torch.allclose(
            result_original, result_chunked, atol=1e-6
        ), f"Memory-efficient method differs with chunk_size={chunk_size}"
