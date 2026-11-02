# torch-fourier-rescale

[![License](https://img.shields.io/pypi/l/torch-fourier-rescale.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-fourier-rescale/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fourier-rescale.svg?color=green)](https://pypi.org/project/torch-fourier-rescale)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fourier-rescale.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

Rescale 2D and 3D images by Fourier padding/cropping in PyTorch.

Downsampling crops the Fourier transform and upsampling pads it, so rescaling
is exact up to the target Nyquist frequency and introduces no interpolation blur.
The output size is rounded to a whole number of pixels, so the achieved pixel
spacing is returned alongside the rescaled image.

## Installation

```bash
pip install torch-fourier-rescale
```

## Usage

```python
import torch
from torch_fourier_rescale import fourier_rescale_2d, fourier_rescale_3d

image = torch.rand(512, 512)

# rescale from 1.35 Å/px to 4 Å/px
rescaled, new_spacing = fourier_rescale_2d(image, source_spacing=1.35, target_spacing=4.0)

# or to an exact shape
rescaled_128, spacing_128 = fourier_rescale_2d(image, source_spacing=1.35, target_shape=(128, 128))

volume = torch.rand(64, 64, 64)
upsampled, _ = fourier_rescale_3d(volume, source_spacing=2.0, target_spacing=1.0)
```

`fourier_rescale_2d` and `fourier_rescale_3d` accept batches through leading
dimensions. `fourier_rescale_rfft_2d` and `fourier_rescale_rfft_3d` operate
directly on rfft arrays, for when the data is already in Fourier space.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
