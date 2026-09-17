# torch-motion-correction

[![License](https://img.shields.io/pypi/l/torch-motion-correction.svg?color=green)](https://github.com/teamtomo/torch-motion-correction/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-motion-correction.svg?color=green)](https://pypi.org/project/torch-motion-correction)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-motion-correction.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-motion-correction/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-motion-correction/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-motion-correction/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-motion-correction)

Movie motion estimation and correction using PyTorch. Implements a spline-based deformation field to model the beam-induced motion over movie collection.

## Implemented algorithms

- Cross-correlation-based global motion estimation (see `estimate_global_motion` in [estimate_motion_xc.py](src/torch_motion_correction/estimate_motion_xc.py))
- Spline-based local motion estimation (see `estimate_local_motion` in [estimate_motion_optimizer.py](src/torch_motion_correction/estimate_motion_optimizer.py))
- Motion correction of movie (not summation to final micrograph) using estimated deformation field (see `correct_motion` in [correct_motion.py](src/torch_motion_correction/correct_motion.py))

## Authors

- Alister Burt (started when working at MRC-LMB, continued at Genentech)
- Josh Dickerson (UC Berkeley)
- Matthew Giammar (UC Berkeley)

