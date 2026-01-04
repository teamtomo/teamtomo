import torch

from torch_fourier_filter.bandpass import (
    bandpass_filter,
)


def test_bandpass_filter():
    freqs = torch.fft.fftfreq(20)
    in_band_idx = torch.logical_and(freqs >= 0.2, freqs <= 0.4)
    lower_idx = torch.logical_and(freqs >= 0.1, freqs <= 0.2)
    upper_idx = freqs > 0.4
    lower_falloff = torch.cos((torch.pi / 2) * ((freqs[lower_idx] - 0.2) / 0.1))
    upper_falloff = torch.cos((torch.pi / 2) * ((freqs[upper_idx] - 0.4) / 0.1))

    bp_filter = bandpass_filter(
        low=0.2,
        high=0.4,
        falloff=0.1,
        image_shape=(20, 1),
        rfft=False,
        fftshift=False,
        device=freqs.device,
    )

    assert torch.all(bp_filter[in_band_idx] == 1)
    assert torch.allclose(bp_filter[lower_idx], lower_falloff.view((-1, 1)), atol=1e-6)
    assert torch.allclose(bp_filter[upper_idx], upper_falloff.view((-1, 1)), atol=1e-6)
