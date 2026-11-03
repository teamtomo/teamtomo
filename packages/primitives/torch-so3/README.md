# torch-so3

[![License](https://img.shields.io/pypi/l/torch-so3.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-so3/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-so3.svg?color=green)](https://pypi.org/project/torch-so3)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-so3.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

Generate uniform sets of proper Euler angles (ZYZ format) over the SO(3) group using PyTorch.
Note that Euler angles are returned in the order of $(\phi, \theta, \psi)$.

## Examples

See the `/examples` directory for brief notebooks outlining the usage of the package.

## Installation

Install via pip
```zsh
pip install torch-so3
```

To install from source, clone the [TeamTomo monorepo](https://github.com/teamtomo/teamtomo), where this package is developed:

```bash
git clone https://github.com/teamtomo/teamtomo.git
cd teamtomo
uv sync --all-packages
```

See the [contributing guide](../../../CONTRIBUTING.md) for the development setup.

## Usage

A basic example of generating uniform Euler angles in 4.0 and 6.0 degree increments across the entire SO(3) group is shown below.

```python
from torch_so3 import get_uniform_euler_angles

angles = get_uniform_euler_angles(
    psi_step=4.0,  # units of degrees
    theta_step=6.0,
)
angles.shape  # (103320, 3)
```
