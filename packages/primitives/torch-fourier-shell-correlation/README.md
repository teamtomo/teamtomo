# torch-fourier-shell-correlation

[![License](https://img.shields.io/pypi/l/torch-fourier-shell-correlation.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-fourier-shell-correlation/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fourier-shell-correlation.svg?color=green)](https://pypi.org/project/torch-fourier-shell-correlation)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fourier-shell-correlation.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

Compute the Fourier shell correlation (FSC) and Fourier ring correlation (FRC)
in PyTorch.

The FSC between two independent half-maps measures how well they agree as a
function of spatial frequency, and is the standard resolution measure in
cryo-EM.

## Installation

```bash
pip install torch-fourier-shell-correlation
```

## Usage

```python
import torch
from torch_fourier_shell_correlation import fsc

half_map_1 = torch.rand(64, 64, 64)
half_map_2 = half_map_1 + 0.5 * torch.rand(64, 64, 64)

# one correlation value per Fourier shell, from DC to Nyquist
correlation = fsc(half_map_1, half_map_2)  # (33,)

# the same function computes Fourier ring correlation for 2D images
frc = fsc(torch.rand(128, 128), torch.rand(128, 128))
```

Shell `i` of the output is at frequency `i / n` cycles per pixel for a box of
size `n`. Convert to resolution in Å with your pixel size, for example using
`torch_grid_utils.fftfreq_to_spatial_frequency`. An optional `fft_mask` restricts
which Fourier components contribute, and `fourier_correlation` takes
pre-computed Fourier transforms.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
