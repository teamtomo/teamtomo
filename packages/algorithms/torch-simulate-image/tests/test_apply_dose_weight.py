import torch
from torch_fourier_filter.dose_weight import dose_weight_2d

from torch_simulate_image import DoseWeightConfig, apply_dose_weight


def test_apply_dose_weight_matches_primitive():
    image = torch.rand(32, 32)
    pixel_size = 1.2
    dose = 25.0
    image_dft = torch.fft.rfft2(image)
    expected = torch.fft.irfft2(
        dose_weight_2d(
            image_dft=image_dft,
            image_shape=(32, 32),
            pixel_size=pixel_size,
            dose=dose,
            voltage=300.0,
            rfft=True,
            fftshift=False,
        ),
        s=(32, 32),
    )
    config = DoseWeightConfig(apply=True, dose_start=dose, dose_end=dose)
    result = apply_dose_weight(image, config, pixel_size=pixel_size, voltage_kv=300.0)
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)
