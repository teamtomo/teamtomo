"""The Mojo backend: everything between the public API and the Mojo kernels.

One module per operator family, mirroring the Mojo sources in ``_mojo/``:

- ``_slice_3d`` -- 3D volume <-> 2D central slices
- ``_line_3d``  -- 3D volume <-> 1D central lines
- ``_line_2d``  -- 2D image  <-> 1D central lines

Each holds the family's non-differentiable ``run_*`` ops and the
``torch.autograd.Function`` wrappers built on them. The ops are the imperative
layer over the kernels: the gathers (``run_extract_*``), their scatter adjoints
(``run_insert_*``), and the pose / weight gradient kernels (``run_*_grad``).
Each validates inputs, materialises contiguous float32 buffers on the input's
device, dispatches to the CPU or GPU kernel, and returns tensors on that device.

The GPU kernels read and write the memory backing torch device tensors directly
-- no host round-trip. Every buffer (inputs *and* pre-zeroed outputs) is placed
on the compute device, and its raw device address is handed to Mojo via the
``addrs`` tuple (see ``_device.prepare_launch`` / ``fourier_slice_kernels.mojo``).
Output buffers are allocated and zeroed on the device up front because the
kernels either leave radius-cut pixels untouched (gathers) or atomically
accumulate (scatters / gradients).

Every extraction and its matching insertion are adjoints of one another, so each
one's *data* gradient is the other's kernel:

- d/d(volume) of an extraction = the scatter (pure adjoint) of grad_output
- d/d(slices)  of an insertion = the gather of grad_output

The insertions used for reconstruction also Hermitian double-insert on the kx=0
plane, whose adjoint is a symmetrisation of the volume/image gradient before the
gather (see ``_common.symmetrise_kx0_plane`` / ``_line_2d._symmetrise_kx0_column``).

Gradients w.r.t. the pose (``rotations`` or ``directions``), the 2D / 3D shifts
and the insertion ``weights`` come from dedicated backward kernels: the pose grad
chains the analytical spatial gradient of the interpolated field through the
rotated sample coordinate; the shift grad differentiates the phase ramp; the
weight grad is the adjoint of the weight splat.

``_loader`` compiles + loads the Mojo extension, ``_device`` resolves device
addresses, ``_validation`` prepares buffers and ``_common`` holds shared helpers.
"""
