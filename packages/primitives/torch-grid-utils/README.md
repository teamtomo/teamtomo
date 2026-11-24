# torch-grid-utils

[![License](https://img.shields.io/pypi/l/torch-grids.svg?color=green)](https://github.com/alisterburt/torch-grids/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-grids.svg?color=green)](https://pypi.org/project/torch-grids)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-grids.svg?color=green)](https://python.org)
[![CI](https://github.com/alisterburt/torch-grids/actions/workflows/ci.yml/badge.svg)](https://github.com/alisterburt/torch-grids/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alisterburt/torch-grids/branch/main/graph/badge.svg)](https://codecov.io/gh/alisterburt/torch-grids)

*torch-grid-utils* provides grids for 2D/3D image manipulations in PyTorch.

## Changed in 0.7.0

- New FFT shape and frequency helpers used across teamtomo:
  `image_shape_from_dft_shape` (real-space shape of a DFT; an rfft needs an
  explicit `image_shape` or `assume_square=True`, since widths `2k` and `2k + 1`
  give the same number of columns), `fftfreq_shell_centers` (shell centers at
  `k / n`) and `fftfreq_to_dft_coordinates` (frequencies to array indices).
  `fftfreq_grid` now also builds 1D grids.
- `polar_grid` now centers at `(h // 2, w // 2)` by default, as documented and as
  the other grids do. Odd sizes move by half a pixel; even sizes are unchanged.
