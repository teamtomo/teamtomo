# torch-structure-manipulation

Typed, in-memory utilities for atomic structures:

- `AtomicStructure.from_dataframe` converts mmdf-compatible DataFrames into
  device-aware tensors, with positions stored in `(z, y, x)` array order.
- `AtomicStructure.from_annotated_dataframe` annotates bonding metadata and
  then constructs the structure in one step.
- `annotate_bonding_environments` adds template-derived bonding and
  `protein`/`rna`/`other` molecule annotations without reading files.
- `classify_structure_composition` and `get_scattering_provider_keys` expose
  aggregate composition labels and per-atom Peng provider keys respectively.
- Centering, rotation, translation, atom selection, and coordinate conversion
  helpers are re-exported from the package root (also available under
  ``structure_transforms``).

```python
import mmdf

from torch_structure_manipulation import (
    AtomicStructure,
    annotate_bonding_environments,
    center_structure,
)

# File I/O belongs to the caller; mmdf is not a runtime dependency.
atoms = mmdf.read("structure.cif")
structure = AtomicStructure.from_annotated_dataframe(atoms, include_hydrogens=False)

# Or annotate first when you still need the DataFrame:
annotated = annotate_bonding_environments(atoms)
centered = center_structure(annotated, center_point=(0.0, 0.0, 0.0), zyx=False)
structure = AtomicStructure.from_dataframe(centered)
```

## License

This project is licensed under the BSD 3-Clause License; see `LICENSE`.
