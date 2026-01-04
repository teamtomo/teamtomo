# Torch Find Peaks Documentation

Welcome to the documentation for the `torch-find-peaks` library.

## Overview

The `torch-find-peaks` library provides utilities for detecting and refining peaks in 2D and 3D data using PyTorch. It includes methods for peak detection, Gaussian fitting, and more.

## Installation

To install the library, use:

```bash
pip install torch-find-peaks
```

## Usage

Here are some of the key functionalities provided by the library:

- **Peak Detection**: Detect peaks in 2D images or 3D volumes.
- **Gaussian Fitting**: Fit 2D or 3D Gaussian functions to refine peak positions.

## API Reference

### `torch_find_peaks.find_peaks`

::: torch_find_peaks.find_peaks
    handler: python
    selection:
      members:
        - find_peaks_2d
        - find_peaks_3d

### `torch_find_peaks.refine_peaks`

::: torch_find_peaks.refine_peaks
    handler: python
    selection:
      members:
        - refine_peaks_2d
        - refine_peaks_3d

### `torch_find_peaks.gaussians`

::: torch_find_peaks.gaussians
    handler: python
    selection:
      members:
        - Gaussian2D
        - Gaussian3D
