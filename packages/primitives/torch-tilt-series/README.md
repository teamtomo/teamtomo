# torch-tilt-series

[![License](https://img.shields.io/pypi/l/torch-tilt-series.svg?color=green)](https://github.com/teamtomo/torch-tilt-series/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-tilt-series.svg?color=green)](https://pypi.org/project/torch-tilt-series)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-tilt-series.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-tilt-series/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-tilt-series/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-tilt-series/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-tilt-series)

Tilt series data structure, projection and subtilt extraction for cryo-ET.

## Overview

This package provides a `TiltSeries` class for working with cryo-ET tilt series alignment geometry in PyTorch. It supports

* loading alignment metadata from AreTomo (`.aln`) and ETOMO directories using [`alnfile`](https://github.com/teamtomo/alnfile) and [`etomofiles`](https://github.com/teamtomo/etomofiles)
* storing tilt series alignment parameters (tilt angles, tilt axis angles, translations, x-tilts) and coordinate-space transforms (see the `TiltSeries` docstring)
* computing projection matrices and projecting 3D points into 2D detector coordinates

All 3D/2D positions are in `zyx`/`yx` coordinates, in Angstroms, relative to the tomogram/detector center. Translations are stored in Angstroms as `(y, x)`.

Subtilt/subvolume extraction and full volume reconstruction live in [`torch-reconstruct-tomogram`](https://github.com/teamtomo/torch-reconstruct-tomogram).


## Installation

```bash
pip install torch-tilt-series
```

To load alignment data from AreTomo or ETOMO files, install the optional IO dependencies:

```bash
pip install torch-tilt-series[io]
```

## Examples

See the [`examples/`](examples/) folder for scripts showing how to load a tilt series and use the API.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
