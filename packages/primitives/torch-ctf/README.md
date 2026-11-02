# torch-ctf

[![License](https://img.shields.io/pypi/l/torch-ctf.svg?color=green)](https://github.com/teamtomo/teamtomo/blob/main/packages/primitives/torch-ctf/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-ctf.svg?color=green)](https://pypi.org/project/torch-ctf)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-ctf.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/teamtomo/actions/workflows/ci.yml)

CTF calculation for cryoEM in torch

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork jdickerson95/torch-ctf --clone
# or just
# gh repo clone jdickerson95/torch-ctf
cd torch-ctf
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
