# TeamTomo

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18405652.svg)](https://doi.org/10.5281/zenodo.18405652)

TeamTomo is a set of modular Python package for cryo-EM and cryo-ET for the modern scientific computing environment.

This unified repository contains the core TeamTomo data processing functionality under a single umbrella for better maintainability and cross-package development. File I/O packages are distributed separately and can be found under the [organizations repositories](https://github.com/orgs/teamtomo/repositories).

## Getting Started

Using TeamTomo packages is easy within the Python ecosystem. Simply install the latest release of packages from PyPI (e.g. ``pip install torch-grid-utils``). A complete list of packages can be found on [teamtomo.org](https://teamtomo.org).

If you're interested in contributing to TeamTomo, please look through the [CONTRIBUTING](CONTRIBUTING.md) guide for more info on adding, migrating, or updating packages. Source installation instructions for different use-cases are listed below. Note that the development workspace depends on the [`uv`](https://github.com/astral-sh/uv) tool for Python environment/package management.

### Basic source installation

```bash
git clone https://github.com/teamtomo/teamtomo.git
cd teamtomo
uv sync
```

### For contributing developers

We _strongly_ recommend making your own fork of the monorepo if you plan on contributing changes. Code should be edited in your repo (origin) and pull requests made into the TeamTomo repo (upstream).

First, fork the repository on GitHub (button near top-right of the repo home page), and then clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/teamtomo.git
cd teamtomo
```

Then, add the TeamTomo repository as an upstream remote to keep your fork up to date:

```bash
git remote add upstream https://github.com/teamtomo/teamtomo.git
git remote -v
# Verify remote setup looks like this
# origin        https://github.com/YOUR_USERNAME/teamtomo.git (fetch)
# origin        https://github.com/YOUR_USERNAME/teamtomo.git (push)
# upstream      https://github.com/teamtomo/teamtomo.git (fetch)
# upstream      https://github.com/teamtomo/teamtomo.git (push)
```

#### Package setting up via `uv`

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

### Editable install for downstream development

For developing programs atop TeamTomo infrastructure, it's useful to have your program _and_ TeamTomo both installed in editable mode in the same environment. Assuming you've cloned the `teamtomo/teamtomo` repo and your program uses `uv` for packaging, add the following to your packages `pyproject.toml` file:

```toml
[tool.uv.sources]
# (other lines for tool.uv.sources)
torch-grid-utils =          { path = "/path/to/teamtomo/packages/primitives/torch-grid-utils", editable = true }
torch-affine-utils =        { path = "/path/to/teamtomo/packages/primitives/torch-affine-utils", editable = true }
torch-image-interpolation = { path = "/path/to/teamtomo/packages/primitives/torch-image-interpolation", editable = true }
# ...
# Add other TeamTomo packages you want editable

# (continuation of pyproject.toml)
```

Then, in your project folder, run ``uv sync`` which will install those dependencies _plus_ the listed teamtomo packages in editable mode. Changes made to TeamTomo packages will reflect in your project's environment.

## List of Packages

🚧 Coming Soon 🚧
