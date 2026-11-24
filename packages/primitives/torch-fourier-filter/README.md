# torch-fourier-filter

[![License](https://img.shields.io/pypi/l/torch-fourier-filter.svg?color=green)](https://github.com/jdickerson95/torch-fourier-filter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-fourier-filter.svg?color=green)](https://pypi.org/project/torch-fourier-filter)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-fourier-filter.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/torch-fourier-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/torch-fourier-filter/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/torch-fourier-filter/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/torch-fourier-filter)

Fourier space filters for image and volumes in pyTorch

Install via source using
```zsh
pip install -e .
```
And for development and testing use
```zsh
pip install -e ".[dev,test]"
```

Make sure to run tests before any commits:
```zsh
python -m pytest
```

## Changed in 0.7.0

- `power_spectral_density` and `whitening_filter` take `image_shape`, which is
  required for rfft input (the default) and raises `ValueError` when missing:
  real widths `2k` and `2k + 1` give the same rfft width. Odd widths now work.
- `whitening.real_space_shape_from_dft_shape` and `dft_utils.rfft_shape` are
  removed. Use `torch_grid_utils.image_shape_from_dft_shape` and
  `torch_grid_utils.rfft_shape`.
- `phase_randomize` keeps the spectrum Hermitian, so it preserves magnitudes and
  returns real images for any size. Random outputs differ from earlier versions.
- `rotational_average_dft_2d` and `rotational_average_dft_3d` bin centers are at
  `k / n`, which changes the frequencies for odd sizes only.
