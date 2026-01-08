# TeamTomo

TeamTomo is a set of modular Python package for cryo-EM and cryo-ET for the modern scientific computing environment.
This unified repository contains most of the core TeamTomo data processing functionality under a single umbrella for better maintainability and cross-package development.

## List of Packages

TODO

## Getting Started

### Installation for Developers

The development workspace depends on the [uv](https://github.com/astral-sh/uv) tool for Python environment/package management.

First, clone and navigate into the root of the repository:

```bash
git clone https://github.com/teamtomo/teamtomo.git
cd teamtomo
```

Then, create and activate a new virtual environment:

```bash
uv venv
source .venv/bin/activate
```

And finally sync with the repository to install packages

```bash
uv sync --all-extras --all-packages
```

Note that the `--all-extras` and `--all-packages` flags install the development and testing requirements for all sub-packages. More granular install options are possible, if your system requires it.
