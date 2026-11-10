# Getting started

TeamTomo packages are deliberately small. Each does one job, takes and returns
`torch.Tensor`s, and avoids its own data structures where a tensor will do. A
typical workflow imports functions from several packages and chains them.

## A short tour

The example below builds a test image, shifts it with subpixel precision, bins it
by Fourier cropping and applies a contrast transfer function (CTF). It uses four
primitive packages:

```bash
pip install torch-grid-utils torch-fourier-shift torch-fourier-rescale torch-ctf
```

```python
import torch
from torch_ctf import calculate_ctf_2d
from torch_fourier_rescale import fourier_rescale_2d
from torch_fourier_shift import fourier_shift_image_2d
from torch_grid_utils import circle

# A 2D image (h, w): a soft-edged disc in the middle of a 128 x 128 box
image = circle(radius=20, image_shape=(128, 128), smoothing_radius=3)

# Shift it by (y, x) = (5.5, -3) pixels with subpixel precision
shifted = fourier_shift_image_2d(image, shifts=torch.tensor([5.5, -3.0]))

# Bin from 1.0 Å/px to 2.0 Å/px by cropping in Fourier space
binned, new_spacing = fourier_rescale_2d(shifted, source_spacing=1.0, target_spacing=2.0)

# A CTF for the rfft of the binned image, then apply it
ctf = calculate_ctf_2d(
    defocus=1.5,              # µm, positive is underfocus
    astigmatism=0.1,          # µm
    astigmatism_angle=30,     # degrees
    voltage=300,              # kV
    spherical_aberration=2.7, # mm
    amplitude_contrast=0.07,
    phase_shift=0,            # degrees
    pixel_size=new_spacing[0],
    image_shape=binned.shape,
    rfft=True,
    fftshift=False,
)
filtered = torch.fft.irfftn(torch.fft.rfftn(binned) * ctf, s=binned.shape)
```

These functions accept extra leading dimensions: a `(n, 128, 128)` stack of images
with `(n, 2)` shifts, or a tensor of `n` defocus values, is processed in one call.
Move the input tensors to a GPU with `.to("cuda")` and the computation follows.

## Next steps

- [Installation](installation.md): PyTorch builds, the metapackage, and source installs.
- [Conventions](conventions.md): array layout, coordinate order and Fourier-space
  conventions shared by all packages. Worth reading before mixing packages.
- Browse the [primitives](../primitives/index.md), [algorithms](../algorithms/index.md)
  and [I/O packages](../io/index.md).
