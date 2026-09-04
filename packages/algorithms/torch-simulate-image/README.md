# torch-simulate-image

Cryo-EM **2D micrograph** simulation from complex **exit waves** in PyTorch.

This algorithm package sits downstream of `torch-scattering` and orchestrates
`torch-ctf` and `torch-fourier-filter` into a configurable pipeline:

```text
exit wave ψ → objective aperture → CTF → intensity → dose weight / envelopes
            → fluence scaling → Poisson noise → DQE → micrograph
```

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

End-to-end notebook (PDB → ESP → multislice → micrograph):
[`examples/simulate_micrograph_from_pdb.ipynb`](examples/simulate_micrograph_from_pdb.ipynb).

```bash
# from the monorepo root, or this package directory
uv sync --group examples
# then open examples/simulate_micrograph_from_pdb.ipynb in Jupyter / VS Code
```

## Scope

- **In scope:** exit wave → micrograph (optics + detector physics)
- **Out of scope:** 3D potentials, wave propagation, structure I/O, CLI

See `notes/torch-simulate-image-plan.md` in the monorepo for the full design.
