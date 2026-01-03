# torch-so3

[![License](https://img.shields.io/pypi/l/torch-so3.svg?color=green)](https://github.com/teamtomo/torch-so3/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-so3.svg?color=green)](https://pypi.org/project/torch-so3)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-so3.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-so3/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-so3/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-so3/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-so3)

Generate uniform sets of proper Euler angles (ZYZ format) over the SO(3) group using PyTorch.
Note that Euler angles are returned in the order of $(\phi, \theta, \psi)$.

## Examples

See the `/examples` directory for brief notebooks outlining the usage of the package.

## Installation

Install via pip
```zsh
pip install torch-so3
```

Install via source by first cloning the repository then running.
```zsh
git clone https://github.com/teamtomo/torch-so3.git
cd torch-so3
pip install -e .
```
And for development and testing use
```zsh
pip install -e ".[dev,test]"
```

For those contributing make sure to run tests before, and to adhere to the pre-commit hooks.
```zsh
python -m pytest
pre-commit run
```

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
