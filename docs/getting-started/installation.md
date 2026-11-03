# Installation

TeamTomo packages need Python 3.11 or newer.

## From PyPI

Install individual packages by name:

```bash
pip install torch-grid-utils
```

or every package through the metapackage:

```bash
pip install teamtomo
```

The metapackage also exposes the packages under one namespace, for example
`teamtomo.primitives.torch_ctf`, but importing the packages directly
(`import torch_ctf`) is equivalent and is what the rest of these docs use.

## PyTorch and GPUs

Packages depend on `torch` but do not pin a build. `pip` installs the default build
for your platform. For a specific CUDA version, install PyTorch first by following
the [PyTorch instructions](https://pytorch.org/get-started/locally/), then install
TeamTomo packages.

## From source

The packages live in one repository, managed as a [uv](https://docs.astral.sh/uv/)
workspace:

```bash
git clone https://github.com/teamtomo/teamtomo.git
cd teamtomo
uv sync --all-packages
```

`--all-packages` installs every package in editable mode. Contributors should also
read the [contributing guide](../developer/index.md).

## Building these docs

The documentation is built with [Zensical](https://zensical.org). Package pages
are generated from each package's `README.md`, docstrings and example notebooks
before the build:

```bash
uv run --isolated --only-group docs python scripts/generate_docs.py
uv run --isolated --only-group docs zensical serve
```
