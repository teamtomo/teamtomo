---
hide:
  - navigation
  - toc
---

# TeamTomo

TeamTomo is a set of small, modular Python packages for cryo-EM and cryo-ET image
processing, built on [PyTorch](https://pytorch.org). Every package works on plain
`torch.Tensor`s, runs on CPU or GPU, and handles batches of images through leading
dimensions, so the pieces compose without glue code.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting started**

    ---

    Install packages, take a short tour, and learn the array and coordinate
    conventions shared by every package.

    [:octicons-arrow-right-24: Getting started](getting-started/index.md)

-   :material-cube-outline: **Primitives**

    ---

    Building blocks: Fourier-space operations, grids, interpolation, CTFs,
    rotations, scattering potentials and more.

    [:octicons-arrow-right-24: Primitives](primitives/index.md)

-   :material-cog-outline: **Algorithms**

    ---

    Complete processing steps built from the primitives: motion correction, CTF
    estimation, tilt-series alignment, reconstruction, template matching.

    [:octicons-arrow-right-24: Algorithms](algorithms/index.md)

-   :material-file-swap-outline: **I/O**

    ---

    Read and write the files other software produces: STAR, IMOD models, mdoc,
    AreTomo and etomo alignments, Dynamo tables, EER, EM and structure files.

    [:octicons-arrow-right-24: I/O](io/index.md)

-   :material-source-pull: **Contributing**

    ---

    Set up the monorepo, add or migrate a package, and cut a release.

    [:octicons-arrow-right-24: Contributing](developer/index.md)

</div>

## Install

Each package is published to PyPI on its own, so install only what you need:

```bash
pip install torch-fourier-slice torch-ctf
```

or everything at once with the `teamtomo` metapackage:

```bash
pip install teamtomo
```

See [Installation](getting-started/installation.md) for GPU builds of PyTorch and
installing from source.

## Citing

If TeamTomo packages are useful in your work, please cite the
[Zenodo record](https://doi.org/10.5281/zenodo.18405652). File I/O packages
(`mrcfile`, `starfile`, ...) live in their own repositories and are listed on
[teamtomo.org](https://teamtomo.org).
