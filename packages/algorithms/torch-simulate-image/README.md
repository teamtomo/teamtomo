# torch-simulate-image

Cryo-EM **2D micrograph** simulation from complex **exit waves** in PyTorch.

This algorithm package sits downstream of `torch-scattering` and orchestrates
`torch-ctf` and `torch-fourier-filter` into a configurable pipeline:

```text
exit wave ψ → objective aperture → CTF → intensity → dose weight / envelopes
            → fluence scaling → Poisson noise → DQE → micrograph
```

## Specimen tilt

`tilt_volume` implements the cryo-ET stage geometry, `Rz(detector_rotation) @ Ry(tilt)`, as a
single resample:

```python
from torch_simulate_image import tilt_volume

tilted = tilt_volume(
    volume,                        # (d, h, w) electrostatic potential, zyx
    tilt_deg=30.0,                 # stage tilt about y
    detector_rotation_deg=-5.0,    # tilt-axis angle as recorded in an mdoc, from +y
    fill_value=3.6,                # pad out-of-bounds voxels with bulk ice
)
```

The tilt axis is **y** because that is a property of the instrument, not the specimen, and
because it is what downstream consumers assume — `torch_tilt_series.TiltSeries` builds
`Rz(tilt_axis_angle) @ Ry(tilt_angle) @ Rx(x_tilt)`. Tilting about any other in-plane axis also
produces a valid tilt series, but the reconstruction rotates the specimen onto y on the way out,
so the tomogram no longer shares a frame with the input volume.

## Examples

Minimal API usage (voltage lives on `CtfConfig` and is reused for dose
weighting / aperture / envelopes):

```python
import torch
from torch_simulate_image import (
    CtfConfig,
    FluenceConfig,
    MicrographSimulationConfig,
    PoissonConfig,
    simulate_micrograph,
)

exit_wave = ...  # complex tensor (..., H, W) from torch_scattering.multislice
config = MicrographSimulationConfig(
    pixel_size=1.0,
    ctf=CtfConfig(defocus_um=1.5, voltage_kv=300.0),
    fluence=FluenceConfig(dose_e_per_A2=30.0),
    poisson=PoissonConfig(apply=False),
)
micrograph = simulate_micrograph(exit_wave, config)
```

End-to-end notebooks (PDB → ESP → multislice → micrograph):

- Dry atoms: [`examples/simulate_micrograph_from_pdb.ipynb`](examples/simulate_micrograph_from_pdb.ipynb)
- Continuum ice comparison (`none` / `constant` / `shang_sigworth`): [`examples/simulate_micrograph_with_solvent.ipynb`](examples/simulate_micrograph_with_solvent.ipynb)
- Ice slab + tilt series (CPU Shang–Sigworth: 5 particles, 256³, −60…+60°/3°): [`examples/simulate_tilt_series_slab.ipynb`](examples/simulate_tilt_series_slab.ipynb)
- End-to-end differentiability (shift + angle pose GD through ESP → multislice → micrograph): [`examples/differentiate_pose.ipynb`](examples/differentiate_pose.ipynb)
- Whole stack, simulation → tomogram (dose-fractionated movie per tilt with beam-induced motion and gold fiducials, then motion correction, alignment and reconstruction, every stage scored against ground truth): [`examples/simulate_movies_to_tomogram.ipynb`](examples/simulate_movies_to_tomogram.ipynb)

```bash
# from the monorepo root, or this package directory
uv sync --group examples
# then open a notebook in Jupyter / VS Code
```

## Scope

- **In scope:** exit wave → micrograph (optics + detector physics), and the
  specimen-tilt geometry that feeds it (`tilt_volume`)
- **Out of scope:** 3D potentials, wave propagation, structure I/O, CLI

See `notes/torch-simulate-image-plan.md` in the monorepo for the full design.
