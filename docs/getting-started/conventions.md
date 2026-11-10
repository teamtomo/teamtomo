# Conventions

TeamTomo packages share a small set of conventions. Knowing them up front avoids
the most common mistakes when passing data between packages.

## Array layout

Images are `(h, w)` tensors and volumes are `(d, h, w)` tensors, matching
`torch.fft` and NumPy indexing. Batches are expressed with leading dimensions,
`(..., h, w)` and `(..., d, h, w)`, and most functions accept any number of them.

## Coordinate order

Coordinates follow the array dimensions they index: `[y, x]` for images and
`[z, y, x]` for volumes. A point is stored in the last dimension, so `n` points in a
volume are a `(n, 3)` tensor of `zyx` coordinates. Shifts follow the same order:
`torch_fourier_shift.fourier_shift_image_2d` takes `(..., 2)` shifts in `h` and `w`.

Coordinates are in pixels (voxels) unless a parameter says otherwise, and the
centre of pixel `i` is at coordinate `i`.

## The centre of an image

The centre of a box of size `n` is pixel `n // 2` in each dimension; for a
`(64, 64)` image that is `[32, 32]`. This is where the DC component of a centred
(`fftshift`ed) Fourier transform lands, so rotations about the image centre and
Fourier-space operations agree. Use `torch_grid_utils.image_center` rather than
computing it by hand.

## Transformation matrices

Affine transforms are matrices that left-multiply homogeneous column vectors, so
in `A @ B @ x` the transform `B` is applied first. `torch-affine-utils` builds them:

- `torch_affine_utils.transforms_3d` returns 4x4 matrices and
  `torch_affine_utils.transforms_2d` returns 3x3 matrices.
- They default to `xyzw` (`xyw`) coordinates. Pass `zyx=True` (`yx=True` in 2D) to
  get matrices that act on `zyxw` (`yxw`) coordinates instead, matching array
  indexing.

Functions that consume matrices, for example in `torch-fourier-slice` and
`torch-transform-image`, say which ordering they expect, usually with a
`zyx_matrices` argument.

## Rotations and Euler angles

Where Euler angles are used, for example by `torch-so3`, they are proper ZYZ Euler
angles given in degrees in the order $(\phi, \theta, \psi)$.

## Fourier space

- **Frequencies are in cycles per pixel** (`fftfreq`), between -0.5 and 0.5, as
  returned by `torch.fft.fftfreq`. Parameters named `fftfreq_*` use these units.
  Convert to and from cycles per Å with
  `torch_grid_utils.fftfreq_to_spatial_frequency` and
  `torch_grid_utils.spatial_frequency_to_fftfreq`.
- **`rfft` and `fftshift` are explicit.** Functions that build Fourier-space
  arrays (filters, CTFs, frequency grids) take `rfft: bool` and `fftshift: bool`
  so the output matches the transform you will multiply it with. An `rfft` of an
  `(h, w)` image has shape `(h, w // 2 + 1)`; `torch_grid_utils.rfft_shape`
  computes it.
- **Physical units** are Å for lengths and pixel sizes, µm for defocus, mm for
  spherical aberration, kV for voltage and degrees for angles, unless a parameter
  says otherwise.
