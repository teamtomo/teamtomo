"""Hermitian bookkeeping of the insertions (backprojections), on both backends.

An rfft stores the kx >= 0 half of a Hermitian spectrum. On the kx = 0 and (even n)
kx = n/2 planes both voxels of a conjugate pair are stored, since -0 = 0 and -n/2 is
the same frequency as +n/2, so an insertion has to write both. The input's own
kx = 0 and kx = n/2 columns likewise store both members of each pair, so each input
sample is weighted to count its pair once (see `_input_pixel_scale`).

Done right, an insertion is the exact adjoint of the matching extraction as a map
between real arrays, with no radius cutoff and without excluding any boundary
frequency. That is what these tests check, plus that the weights follow the data
and that the autograd backward passes are exact transposes.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from torch_fourier_slice import (
    extract_central_slices_rfft_3d,
    insert_central_slices_rfft_3d,
)
from torch_fourier_slice.experimental import (
    extract_central_line_rfft_2d as mojo_extract_central_line_rfft_2d,
)
from torch_fourier_slice.experimental import (
    extract_central_line_rfft_3d as mojo_extract_central_line_rfft_3d,
)
from torch_fourier_slice.experimental import (
    extract_central_slices_rfft_3d as mojo_extract_central_slices_rfft_3d,
)
from torch_fourier_slice.experimental import (
    insert_central_line_rfft_2d as mojo_insert_central_line_rfft_2d,
)
from torch_fourier_slice.experimental import (
    insert_central_line_rfft_3d as mojo_insert_central_line_rfft_3d,
)
from torch_fourier_slice.experimental import (
    insert_central_slices_rfft_3d as mojo_insert_central_slices_rfft_3d,
)
from torch_fourier_slice.experimental import mojo_kernels_available

requires_mojo = pytest.mark.skipif(
    not mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
)

N_POSES = 8
# larger than any frequency, so nothing is cut off
NO_CUTOFF = 1e3


def _rotations(seed: int = 3) -> torch.Tensor:
    matrices = Rotation.random(N_POSES, random_state=seed).as_matrix()
    return torch.tensor(matrices, dtype=torch.float32)


def _directions(ndim: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(N_POSES, ndim, generator=generator)
    return (directions / directions.norm(dim=-1, keepdim=True)).contiguous()


class _Insertion:
    """An insertion and its extraction for fixed poses, on DC-at-origin rffts.

    `insert(inputs, weights=None)` returns `(data, weights)`, `extract(array)` the
    extracted inputs. Inputs are rffts of real arrays of `input_shape`, inserted into
    an array of `output_shape`.
    """

    def __init__(self, insert, extract, input_shape, output_shape):
        self.insert = insert
        self.extract = extract
        self.input_shape = input_shape
        self.output_shape = output_shape


def _torch_slices(sidelength: int, rotations: torch.Tensor) -> _Insertion:
    shape = (sidelength,) * 3

    def insert(slices, weights=None):
        data, weight = insert_central_slices_rfft_3d(
            torch.fft.fftshift(slices, dim=-2), shape, rotations, zyx_matrices=True
        )
        return (
            torch.fft.ifftshift(data, dim=(-3, -2)),
            torch.fft.ifftshift(weight, dim=(-3, -2)),
        )

    def extract(volume):
        slices = extract_central_slices_rfft_3d(
            torch.fft.fftshift(volume, dim=(-3, -2)), rotations, zyx_matrices=True
        )
        return torch.fft.ifftshift(slices, dim=-2)

    return _Insertion(insert, extract, (sidelength,) * 2, shape)


def _mojo_slices(
    sidelength: int,
    rotations: torch.Tensor,
    interpolation: str = "linear",
    cutoff: float = NO_CUTOFF,
) -> _Insertion:
    def insert(slices, weights=None):
        return mojo_insert_central_slices_rfft_3d(
            slices,
            rotations,
            weights=weights,
            fourier_radius_cutoff=cutoff,
            interpolation=interpolation,
        )

    def extract(volume):
        return mojo_extract_central_slices_rfft_3d(
            volume.contiguous(),
            rotations,
            fourier_radius_cutoff=cutoff,
            interpolation=interpolation,
        )

    return _Insertion(insert, extract, (sidelength,) * 2, (sidelength,) * 3)


def _mojo_lines(
    sidelength: int,
    directions: torch.Tensor,
    interpolation: str = "linear",
    cutoff: float = NO_CUTOFF,
) -> _Insertion:
    ndim = directions.shape[-1]
    if ndim == 3:
        insert_fn, extract_fn = (
            mojo_insert_central_line_rfft_3d,
            mojo_extract_central_line_rfft_3d,
        )
    else:
        insert_fn, extract_fn = (
            mojo_insert_central_line_rfft_2d,
            mojo_extract_central_line_rfft_2d,
        )

    def insert(lines, weights=None):
        return insert_fn(
            lines,
            directions,
            weights=weights,
            fourier_radius_cutoff=cutoff,
            interpolation=interpolation,
        )

    def extract(array):
        return extract_fn(
            array.contiguous(),
            directions,
            fourier_radius_cutoff=cutoff,
            interpolation=interpolation,
        )

    return _Insertion(insert, extract, (sidelength,), (sidelength,) * ndim)


CASES = [
    pytest.param(lambda: _torch_slices(16, _rotations()), id="torch-slices-even"),
    pytest.param(lambda: _torch_slices(15, _rotations()), id="torch-slices-odd"),
]
for _interpolation in ("linear", "cubic"):
    CASES += [
        pytest.param(
            lambda i=_interpolation: _mojo_slices(16, _rotations(), i),
            id=f"mojo-slices-{_interpolation}",
            marks=requires_mojo,
        ),
        pytest.param(
            lambda i=_interpolation: _mojo_lines(16, _directions(3), i),
            id=f"mojo-lines-3d-{_interpolation}",
            marks=requires_mojo,
        ),
        pytest.param(
            lambda i=_interpolation: _mojo_lines(16, _directions(2), i),
            id=f"mojo-lines-2d-{_interpolation}",
            marks=requires_mojo,
        ),
    ]


def _rfft(x: torch.Tensor, ndim: int) -> torch.Tensor:
    return torch.fft.rfftn(x, dim=tuple(range(-ndim, 0)))


def _irfft(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.fft.irfftn(x, s=shape, dim=tuple(range(-len(shape), 0)))


def _random_real(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("make_case", CASES)
def test_insertion_is_real_space_adjoint_of_extraction(make_case):
    """<irfft(insert(rfft(p))), v> = c <p, irfft(extract(rfft(v)))> for real p, v.

    Both sides are real-space inner products, so every stored frequency counts with
    its multiplicity, edges included. `c` is the ratio of the array sizes, from the
    unnormalised forward FFTs.
    """
    case = make_case()
    inputs = _random_real((N_POSES, *case.input_shape), seed=0)
    output = _random_real(case.output_shape, seed=1)
    inserted, _ = case.insert(_rfft(inputs, len(case.input_shape)))
    extracted = case.extract(_rfft(output, len(case.output_shape)))

    lhs = (_irfft(inserted, case.output_shape) * output).sum()
    rhs = (_irfft(extracted, case.input_shape) * inputs).sum()
    lhs = lhs * np.prod(case.output_shape)
    rhs = rhs * np.prod(case.input_shape)
    assert abs(lhs - rhs) < 1e-5 * abs(rhs)


@pytest.mark.parametrize("make_case", CASES)
def test_weights_follow_the_data(make_case):
    """Inserting delta functions (rfft = 1) puts the weights into the data."""
    case = make_case()
    ones = torch.ones(
        (N_POSES, *case.input_shape[:-1], case.input_shape[-1] // 2 + 1),
        dtype=torch.complex64,
    )
    data, weights = case.insert(ones, torch.ones(ones.shape))
    scale = weights.abs().max()
    assert torch.allclose(data, weights.to(data.dtype), atol=1e-6 * scale)


@pytest.mark.parametrize("make_case", CASES)
def test_insertion_is_hermitian(make_case):
    """On the planes that store their own mirror, voxels pair up as conjugates."""
    case = make_case()
    inputs = _random_real((N_POSES, *case.input_shape), seed=0)
    data, _ = case.insert(_rfft(inputs, len(case.input_shape)))
    full_dims = tuple(range(data.dim() - 1))
    n = case.output_shape[-1]
    for x in (0, n // 2) if n % 2 == 0 else (0,):
        plane = data[..., x]
        mirror = plane.flip(dims=full_dims).roll(
            shifts=(1,) * len(full_dims), dims=full_dims
        )
        assert torch.allclose(plane, mirror.conj(), atol=1e-5 * data.abs().max())


@pytest.mark.parametrize(
    "make_case",
    [
        pytest.param(lambda r: _torch_slices(8, r), id="torch"),
        pytest.param(lambda r: _mojo_slices(8, r), id="mojo", marks=requires_mojo),
    ],
)
def test_untilted_slice_inserts_the_image_with_unit_weights(make_case):
    """An untilted slice lands on the kz = 0 plane with weight 1 at every voxel.

    Including the voxels that are their own mirror (DC and the Nyquist corners),
    which an input sample reaches through both of its writes.
    """
    case = make_case(torch.eye(3)[None])
    image = _rfft(_random_real((1, 8, 8), seed=0), 2)
    data, weights = case.insert(image, torch.ones(image.shape))
    assert torch.allclose(weights[0], torch.ones_like(weights[0]))
    assert torch.allclose(data[0], image[0], atol=1e-5 * image.abs().max())


def _linear_backward_ok(loss_fn, param, n=4, scale=0.05, tol=1e-3):
    """For a loss linear in `param`, autograd must predict every change exactly."""
    loss = loss_fn(param)
    loss.backward()
    grad = param.grad.clone()
    generator = torch.Generator().manual_seed(123)
    for _ in range(n):
        delta = torch.randn(param.shape, dtype=param.dtype, generator=generator)
        delta = delta * scale
        with torch.no_grad():
            change = (loss_fn(param.detach() + delta) - loss.detach()).item()
        predicted = torch.real(torch.sum(torch.conj(grad) * delta)).item()
        if abs(predicted / change - 1) > tol:
            return False
    return True


MOJO_CASES = [case for case in CASES if case.id.startswith("mojo")]


@pytest.mark.parametrize("make_case", MOJO_CASES)
def test_insertion_backward_is_exact_transpose(make_case):
    """The data and weight gradients of the insertion, with no radius cutoff."""
    case = make_case()
    ndim = len(case.input_shape)
    inputs = _rfft(_random_real((N_POSES, *case.input_shape), seed=0), ndim)
    weights = torch.rand(inputs.shape, generator=torch.Generator().manual_seed(2))
    data_target = _rfft(_random_real(case.output_shape, seed=3), len(case.output_shape))
    weight_target = torch.randn(
        data_target.shape, generator=torch.Generator().manual_seed(4)
    )

    def data_loss(x):
        data, _ = case.insert(x)
        return torch.real(torch.sum(torch.conj(data_target) * data))

    def weight_loss(w):
        _, weight = case.insert(inputs, w)
        return torch.sum(weight_target * weight)

    assert _linear_backward_ok(data_loss, inputs.clone().requires_grad_(True))
    assert _linear_backward_ok(weight_loss, weights.clone().requires_grad_(True))


def _finite_difference_error(loss_fn, param, eps):
    """|autograd directional derivative / central difference - 1|."""
    loss_fn(param).backward()
    direction = torch.randn(param.shape, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        plus = loss_fn(param.detach() + eps * direction).item()
        minus = loss_fn(param.detach() - eps * direction).item()
    predicted = torch.sum(param.grad * direction).item()
    return abs(predicted / ((plus - minus) / (2 * eps)) - 1)


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_insertion_pose_gradient_includes_self_mirror_samples(interpolation):
    """Pose gradients of the insertion with the cutoff at the Nyquist radius n/2.

    The sphere |k| <= n/2 keeps every sample inside the Nyquist box, where the
    insertion is smooth in the pose, and it still includes the self-mirror samples
    at |k| = 0 and n/2 that the insertion scales by 0.5.
    """
    n = 16
    target = _rfft(_random_real((n, n, n), seed=3), 3)
    slices = _rfft(_random_real((N_POSES, n, n), seed=0), 2)
    lines = _rfft(_random_real((N_POSES, n), seed=0), 1)

    def slice_loss(rotations):
        data, _ = _mojo_slices(n, rotations, interpolation, cutoff=n / 2).insert(slices)
        return ((data - target).abs() ** 2).sum()

    def line_loss(directions):
        data, _ = _mojo_lines(n, directions, interpolation, cutoff=n / 2).insert(lines)
        return ((data - target).abs() ** 2).sum()

    rotations = _rotations().requires_grad_(True)
    assert _finite_difference_error(slice_loss, rotations, eps=3e-4) < 5e-2
    directions = _directions(3).requires_grad_(True)
    assert _finite_difference_error(line_loss, directions, eps=3e-4) < 5e-2
