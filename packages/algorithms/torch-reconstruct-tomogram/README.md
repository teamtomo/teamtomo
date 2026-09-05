# torch-reconstruct-tomogram

[![License](https://img.shields.io/pypi/l/torch-reconstruct-tomogram.svg?color=green)](https://github.com/teamtomo/torch-reconstruct-tomogram/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-reconstruct-tomogram.svg?color=green)](https://pypi.org/project/torch-reconstruct-tomogram)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-reconstruct-tomogram.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-reconstruct-tomogram/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-reconstruct-tomogram/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-reconstruct-tomogram/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-reconstruct-tomogram)

(sub-)Tomogram reconstruction and subtilt extraction for cryo-ET.

## Overview

This package provides (sub-)tomogram reconstruction and subtilt extraction driven entirely from a [`torch-tilt-series`](https://github.com/teamtomo/torch-tilt-series) `TiltSeries`. It supports

* `extract_particle_tilt_series()`: extract a subtilt-series at 3D location(s) in the sample
* `reconstruct_subvolume()`: rank-polymorphic reconstruction of 3D patch(es) at location(s) in the sample
* `reconstruct_tomogram()`: full volume reconstruction by tiling reconstructed patches in 3D

`TiltSeries` holds alignment geometry (in Angstroms) plus `image_path`/`image_indices`/`pixel_spacing` metadata. The functions above take a `TiltSeries`, and load and (by default) preprocess the matching raw images internally via `torch_tilt_series.load_tilt_series_images()` / `preprocess_tilt_series_images()` (plane subtraction, a DC-excluding bandpass with no low-pass, i.e. up to Nyquist, and central-crop normalization).  `output_pixel_spacing` lets both local (`reconstruct_subvolume`) and global (`reconstruct_tomogram`) reconstruction target an arbitrary output voxel size. Reconstruction happens at the input pixel spacing and is Fourier-rescaled to the requested output size. Reconstruction is performed in Fourier space using central slice insertion. Positions are in `zyx` coordinates, in Angstroms, relative to the tomogram center.

## Installation

```bash
pip install torch-reconstruct-tomogram
```

To load a tilt series from AreTomo or ETOMO output, also install the IO dependencies for [`torch-tilt-series`](https://github.com/teamtomo/torch-tilt-series):

```bash
pip install torch-tilt-series[io]
```

## Examples

See the [`examples/`](examples/) folder for scripts showing how to load a tilt series, reconstruct subvolumes and tomograms, and save the result.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
