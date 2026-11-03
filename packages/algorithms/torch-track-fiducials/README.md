# torch-track-fiducials

Gold fiducial tracking through cryo-ET tilt series in PyTorch.

`torch-track-fiducials` finds gold beads and tracks them through a tilt
series, starting from a coarse alignment (e.g. from `torch-tiltxcorr`). The
tracks are fitted with `torch-tiltalign`, so the result is a refined
`torch_tilt_series.TiltSeries` together with the bead tracks: the TeamTomo
equivalent of IMOD's seeding, beadtrack and tiltalign steps.

## How it works

1. **Bead correlation.** Every tilt is band-pass filtered and correlated with
   a dark disc of the bead diameter; maps are scaled by their robust standard
   deviation.
2. **Seeding.** The strongest peaks in the tilt closest to 0 are seeds. Each
   seed's depth is found by scoring points along its ray, projected with the
   starting geometry into the tilts within +-40 degrees.
3. **Track and fit.** Over a few rounds that widen the tilt range and shrink
   the search radius:
   - project every bead with the current geometry;
   - correct each tilt by the offset common to all its beads (the peak of the
     correlation summed around their predictions), so tilts whose starting
     shift is far off are still tracked;
   - take each bead's correlation peak within the search radius, dropping
     weak peaks, overlapping beads and beads near the image edge;
   - fit the geometry and 3D bead positions with
     `torch_tiltalign.fit_tilt_series`.
4. Beads tracked in too few tilts, or mostly rejected by the robust fit, are
   dropped and the final tracks are fitted again.

## Usage

```python
from torch_track_fiducials import track_fiducials

# images: (n_tilts, h, w) in the order of tilt_series.tilt_angles, raw
# contrast; tilt_series.pixel_spacing is the images' pixel spacing
result = track_fiducials(images, tilt_series, fiducial_diameter=100.0)

result.alignment.tilt_series  # refined TiltSeries
result.tracks_yx  # (n_beads, n_tilts, 2), Angstroms from the image center
result.alignment.rms_residual  # Angstroms
```

Only shifts are refined by default; pass `refine_tilt_axis_angle=True` or
`refine_tilt_angles=True` to refine those too.
