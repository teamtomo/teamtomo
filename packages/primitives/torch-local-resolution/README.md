# localResolution-TeamTomo

[![License](https://img.shields.io/pypi/l/localResolution-TeamTomo.svg?color=green)](https://github.com/DavidKart/localResolution-TeamTomo/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/localResolution-TeamTomo.svg?color=green)](https://pypi.org/project/localResolution-TeamTomo)
[![Python Version](https://img.shields.io/pypi/pyversions/localResolution-TeamTomo.svg?color=green)](https://python.org)
[![CI](https://github.com/DavidKart/localResolution-TeamTomo/actions/workflows/ci.yml/badge.svg)](https://github.com/DavidKart/localResolution-TeamTomo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DavidKart/localResolution-TeamTomo/branch/main/graph/badge.svg)](https://codecov.io/gh/DavidKart/localResolution-TeamTomo)

Core functionality for local resolution estimation of cryo-EM half-maps

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork DavidKart/localResolution-TeamTomo --clone
# or just
# gh repo clone DavidKart/localResolution-TeamTomo
cd localResolution-TeamTomo
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
