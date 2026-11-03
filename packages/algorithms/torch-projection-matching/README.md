# torch-projection-matching

Marker-free tilt-series alignment by projection matching in PyTorch.

`torch-projection-matching` refines the per-tilt shifts of a
`torch_tilt_series.TiltSeries` by matching each tilt image to a reprojection
of the tomogram reconstructed from all the other tilts, the fine-alignment
approach of AreTomo. It needs no fiducials.

## How it works

- Each sweep builds a tomogram outwards from the tilt closest to 0. Every
  other tilt, in order of increasing tilt angle, is cross-correlated with the
  projection (at its own geometry) of the tomogram reconstructed from the lower
  tilts; its shift is updated by the correlation peak and it is then added to
  the tomogram. Sweeps repeat until the updates are below a tolerance (usually
  2-3 sweeps).
- Matching against the lower tilts anchors the alignment at 0 degrees. Matching
  each tilt against all the other tilts instead lets the alignment drift by
  tens of Angstroms over successive sweeps on real data.
- The tomogram is a weighted backprojection in real space (ramp filter
  perpendicular to the tilt axis) using the `TiltSeries` projection, widened
  perpendicular to the axis so every ray at every tilt crosses it.
- At each sweep the specimen slab is found in a thick reconstruction, and
  matching uses only that slab. A slab much thicker than the specimen weakens
  the 3D constraint; one thinner than the specimen biases the alignment.
- The part of the updates that only translates the tomogram is removed, so
  the tomogram stays where the starting geometry put it.

## Usage

```python
from torch_projection_matching import align_by_projection_matching

# images: (n_tilts, h, w) at low resolution (about 20 A pixels), in the order of
# tilt_series.tilt_angles; tilt_series.pixel_spacing is the images' spacing
result = align_by_projection_matching(images, tilt_series, thickness=4000.0)

result.tilt_series  # refined TiltSeries
result.slab  # (z_center, thickness) of the slab used, Angstroms
result.correlation  # (n_tilts,) final correlation with the reprojection
```

`thickness` is the range searched for the specimen; it should comfortably
contain it.
