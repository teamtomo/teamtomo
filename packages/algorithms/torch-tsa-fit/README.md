# torch-tsa-fit

Fit tilt-series alignment to tracked 2D positions of 3D points in PyTorch.

`torch-tsa-fit` is the model-fitting step of fiducial (or patch) based
tilt-series alignment, the equivalent of IMOD's `tiltalign`. Given the
detector positions of points tracked through a tilt series, it finds the 3D
position of every point and the per-tilt alignment that best explain them.
The points can be gold fiducials (see `torch-tsa-fiducials`), patch centers
(see `torch_tiltxcorr.tiltxcorr_patches`), or anything else tracked through
the series.

The geometry is a `torch_tilt_series.TiltSeries`; projection uses
`TiltSeries.project_points`, so the fitted alignment is directly usable for
reconstruction and subtilt extraction.

## What is fitted

- 3D point positions (always)
- per-tilt shifts (`refine_shifts=True`, the default)
- one in-plane rotation common to all tilts (`refine_tilt_axis_angle=True`)
- per-tilt tilt angles, with the tilt closest to 0 held fixed
  (`refine_tilt_angles=True`)

With only shifts refined the problem is linear least squares. Outlying
observations are down-weighted with a Tukey biweight (`robust=True`), as in
`tiltalign`'s robust fitting. A rigid translation of the tomogram leaves every
projection unchanged; it is fixed so the tomogram stays where the starting
geometry put it.

## Usage

```python
from torch_tsa_fit import fit_tilt_series

# tracks_yx: (n_points, n_tilts, 2) detector positions in Angstroms,
#            relative to the detector center (NaN where not tracked)
result = fit_tilt_series(tilt_series, tracks_yx, refine_tilt_axis_angle=True)

result.tilt_series  # fitted TiltSeries
result.points_zyx  # (n_points, 3) 3D positions, Angstroms
result.rms_residual  # Angstroms
```
