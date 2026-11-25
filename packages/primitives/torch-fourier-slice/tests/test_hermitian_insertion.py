"""Hermitian bookkeeping of the insertions (backprojections), on every backend.

An rfft stores the kx >= 0 half of a Hermitian spectrum. On the kx = 0 and (even n)
kx = n/2 planes both voxels of a conjugate pair are stored, since -0 = 0 and -n/2 is
the same frequency as +n/2, so an insertion has to write both. The input's own
kx = 0 and kx = n/2 columns likewise store both members of each pair, so each input
sample is weighted to count its pair once (see `_input_pixel_scale`).

Done right, an insertion is the exact adjoint of the matching extraction as a map
between real arrays, with no radius cutoff and without excluding any boundary
frequency. That is what these tests check, plus that the weights follow the data
and that the autograd backward passes are exact transposes. They run on torch, the
experimental Mojo kernels and the experimental torch reference.
"""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from torch_fourier_slice import (
    extract_central_slices_rfft_3d,
    insert_central_slices_rfft_3d,
)
from torch_fourier_slice.experimental import (
    extraction,
    insertion,
    mojo_kernels_available,
    reference,
)

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


def _experimental_slices(
    module,
    sidelength: int,
    rotations: torch.Tensor,
    interpolation: str = "linear",
    cutoff: float = NO_CUTOFF,
    oversampling: float = 1.0,
    **shifts,
) -> _Insertion:
    """`module` provides the slice functions; `cutoff` is in input pixels.

    A `cutoff` of None leaves `fftfreq_max` at its default (Nyquist).

    `shifts` are `shifts_2d` / `shifts_3d`, passed to both operations. Poses and
    shifts are in the kernels' zyx / yx order.
    """
    volume_shape = (round(sidelength * oversampling),) * 3
    options = {
        "fftfreq_max": None if cutoff is None else cutoff / sidelength,
        "zyx_matrices": True,
        "zyx_shifts": True,
        "yx_shifts": True,
        "interpolation": interpolation,
        **shifts,
    }

    def insert(slices, weights=None):
        return module.insert_central_slices_rfft_3d(
            slices, volume_shape, rotations, weights=weights, **options
        )

    def extract(volume):
        return module.extract_central_slices_rfft_3d(
            volume.contiguous(),
            rotations,
            output_shape=(sidelength, sidelength),
            **options,
        )

    return _Insertion(insert, extract, (sidelength,) * 2, volume_shape)


# the Mojo rfft-layer functions, by the names `reference` gives them
_MOJO = SimpleNamespace(
    **{
        name: getattr(insertion if name.startswith("insert") else extraction, name)
        for name in reference.__all__
        if name.startswith(("extract_central", "insert_central"))
    }
)


def _mojo_slices(*args, **kwargs) -> _Insertion:
    return _experimental_slices(_MOJO, *args, **kwargs)


def _reference_slices(*args, **kwargs) -> _Insertion:
    return _experimental_slices(reference, *args, **kwargs)


def _experimental_lines(
    module,
    sidelength: int,
    directions: torch.Tensor,
    interpolation: str = "linear",
    cutoff: float = NO_CUTOFF,
    oversampling: float = 1.0,
    **shifts,
) -> _Insertion:
    """`module` provides the line functions; `cutoff` is in input pixels.

    A `cutoff` of None leaves `fftfreq_max` at its default (Nyquist).

    `shifts` are `shifts_3d` (3D lines) / `shifts_2d` (2D lines). Directions and
    shifts are in zyx / yx order.
    """
    ndim = directions.shape[-1]
    insert_fn = getattr(module, f"insert_central_lines_rfft_{ndim}d")
    extract_fn = getattr(module, f"extract_central_lines_rfft_{ndim}d")
    if ndim == 3:
        order = {"zyx_directions": True, "zyx_shifts": True}
    else:
        order = {"yx_directions": True, "yx_shifts": True}
    output_shape = (round(sidelength * oversampling),) * ndim
    options = {
        "fftfreq_max": None if cutoff is None else cutoff / sidelength,
        "interpolation": interpolation,
        **order,
        **shifts,
    }

    def insert(lines, weights=None):
        return insert_fn(
            lines,
            output_shape,
            directions,
            weights=weights,
            line_length=sidelength,
            **options,
        )

    def extract(array):
        return extract_fn(
            array.contiguous(), directions, output_length=sidelength, **options
        )

    return _Insertion(insert, extract, (sidelength,), output_shape)


def _mojo_lines(*args, **kwargs) -> _Insertion:
    return _experimental_lines(_MOJO, *args, **kwargs)


def _reference_lines(*args, **kwargs) -> _Insertion:
    return _experimental_lines(reference, *args, **kwargs)


CASES = [
    pytest.param(lambda: _torch_slices(16, _rotations()), id="torch-slices-even"),
    pytest.param(lambda: _torch_slices(15, _rotations()), id="torch-slices-odd"),
]
for _interpolation, _n in itertools.product(("linear", "cubic"), (16, 15)):
    _parity = "even" if _n % 2 == 0 else "odd"
    CASES += [
        pytest.param(
            lambda i=_interpolation, n=_n: _reference_slices(n, _rotations(), i),
            id=f"reference-slices-{_parity}-{_interpolation}",
        ),
        pytest.param(
            lambda i=_interpolation, n=_n: _mojo_slices(n, _rotations(), i),
            id=f"mojo-slices-{_parity}-{_interpolation}",
            marks=requires_mojo,
        ),
        pytest.param(
            lambda i=_interpolation, n=_n: _reference_lines(n, _directions(3), i),
            id=f"reference-lines-3d-{_parity}-{_interpolation}",
        ),
        pytest.param(
            lambda i=_interpolation, n=_n: _reference_lines(n, _directions(2), i),
            id=f"reference-lines-2d-{_parity}-{_interpolation}",
        ),
        pytest.param(
            lambda i=_interpolation, n=_n: _mojo_lines(n, _directions(3), i),
            id=f"mojo-lines-3d-{_parity}-{_interpolation}",
            marks=requires_mojo,
        ),
        pytest.param(
            lambda i=_interpolation, n=_n: _mojo_lines(n, _directions(2), i),
            id=f"mojo-lines-2d-{_parity}-{_interpolation}",
            marks=requires_mojo,
        ),
    ]


def _shifts(ndim: int, seed: int = 5) -> torch.Tensor:
    """Random non-integer shifts in pixels, so the Nyquist phases are not trivial."""
    return 1.3 * torch.randn(
        N_POSES, ndim, generator=torch.Generator().manual_seed(seed)
    )


# shifts and oversampling change the data but not the Hermitian bookkeeping, so
# only the tests that hold for any linear insertion run on them
OPTION_CASES = [
    pytest.param(
        lambda: _reference_slices(16, _rotations(), shifts_2d=_shifts(2)),
        id="reference-slices-shifts-2d",
    ),
    pytest.param(
        lambda: _reference_slices(16, _rotations(), shifts_3d=_shifts(3)),
        id="reference-slices-shifts-3d",
    ),
    pytest.param(
        lambda: _reference_slices(16, _rotations(), oversampling=2.0),
        id="reference-slices-oversampling-2",
    ),
    pytest.param(
        lambda: _reference_slices(16, _rotations(), oversampling=1.5),
        id="reference-slices-oversampling-1.5",
    ),
    pytest.param(
        lambda: _reference_slices(16, _rotations(), cutoff=None, oversampling=2.0),
        id="reference-slices-oversampling-2-nyquist",
    ),
    pytest.param(
        lambda: _reference_slices(15, _rotations(), oversampling=16 / 15),
        id="reference-slices-odd-into-even",
    ),
    pytest.param(
        lambda: _reference_lines(15, _directions(3), oversampling=16 / 15),
        id="reference-lines-3d-odd-into-even",
    ),
    pytest.param(
        lambda: _reference_lines(16, _directions(2), oversampling=15 / 16),
        id="reference-lines-2d-even-into-odd",
    ),
    pytest.param(
        lambda: _reference_lines(16, _directions(3), cutoff=None, oversampling=2.0),
        id="reference-lines-3d-oversampling-2-nyquist",
    ),
    pytest.param(
        lambda: _mojo_slices(16, _rotations(), shifts_2d=_shifts(2)),
        id="mojo-slices-shifts-2d",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_slices(16, _rotations(), shifts_3d=_shifts(3)),
        id="mojo-slices-shifts-3d",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_slices(16, _rotations(), oversampling=2.0),
        id="mojo-slices-oversampling-2",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_slices(16, _rotations(), oversampling=1.5),
        id="mojo-slices-oversampling-1.5",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_slices(16, _rotations(), cutoff=None, oversampling=2.0),
        id="mojo-slices-oversampling-2-nyquist",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_slices(15, _rotations(), oversampling=16 / 15),
        id="mojo-slices-odd-into-even",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_lines(15, _directions(3), oversampling=16 / 15),
        id="mojo-lines-3d-odd-into-even",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_lines(16, _directions(2), oversampling=15 / 16),
        id="mojo-lines-2d-even-into-odd",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_lines(16, _directions(3), cutoff=None, oversampling=2.0),
        id="mojo-lines-3d-oversampling-2-nyquist",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _reference_lines(16, _directions(3), shifts_3d=_shifts(3)),
        id="reference-lines-3d-shifts",
    ),
    pytest.param(
        lambda: _reference_lines(16, _directions(2), shifts_2d=_shifts(2)),
        id="reference-lines-2d-shifts",
    ),
    pytest.param(
        lambda: _reference_lines(16, _directions(3), oversampling=2.0),
        id="reference-lines-3d-oversampling-2",
    ),
    pytest.param(
        lambda: _reference_lines(16, _directions(2), oversampling=2.0),
        id="reference-lines-2d-oversampling-2",
    ),
    pytest.param(
        lambda: _mojo_lines(16, _directions(3), shifts_3d=_shifts(3)),
        id="mojo-lines-3d-shifts",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_lines(16, _directions(2), shifts_2d=_shifts(2)),
        id="mojo-lines-2d-shifts",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_lines(16, _directions(3), oversampling=2.0),
        id="mojo-lines-3d-oversampling-2",
        marks=requires_mojo,
    ),
    pytest.param(
        lambda: _mojo_lines(16, _directions(2), oversampling=2.0),
        id="mojo-lines-2d-oversampling-2",
        marks=requires_mojo,
    ),
]


def _rfft(x: torch.Tensor, ndim: int) -> torch.Tensor:
    return torch.fft.rfftn(x, dim=tuple(range(-ndim, 0)))


def _irfft(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.fft.irfftn(x, s=shape, dim=tuple(range(-len(shape), 0)))


def _random_real(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("make_case", CASES + OPTION_CASES)
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


@pytest.mark.parametrize(
    "make_case", [case for case in CASES if not case.id.startswith("torch")]
)
def test_omitted_weights_are_unit_weights(make_case):
    """Without ``weights`` the insertion still returns weights: those of ones.

    (Equal up to the order of the Mojo CPU kernel's parallel accumulation.)
    """
    case = make_case()
    ndim = len(case.input_shape)
    inputs = _rfft(_random_real((N_POSES, *case.input_shape), seed=0), ndim)
    data, weights = case.insert(inputs)
    unit_data, unit_weights = case.insert(inputs, torch.ones(inputs.shape))
    assert torch.allclose(data, unit_data, atol=1e-6 * data.abs().max())
    assert torch.allclose(weights, unit_weights, atol=1e-6 * weights.max())


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
        pytest.param(lambda r: _reference_slices(8, r), id="reference"),
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
    """For a loss linear in `param`, autograd must predict every change exactly.

    The losses here are homogeneous, `loss(x + delta) - loss(x) = loss(delta)`, so
    the change is computed as `loss(delta)`: subtracting two large float32 losses
    would cancel to about the tolerance.
    """
    loss = loss_fn(param)
    loss.backward()
    grad = param.grad.clone()
    generator = torch.Generator().manual_seed(123)
    for _ in range(n):
        delta = torch.randn(param.shape, dtype=param.dtype, generator=generator)
        delta = delta * scale
        with torch.no_grad():
            change = loss_fn(delta).item()
        predicted = torch.real(torch.sum(torch.conj(grad) * delta)).item()
        if abs(predicted / change - 1) > tol:
            return False
    return True


EXPERIMENTAL_CASES = [
    case for case in CASES + OPTION_CASES if not case.id.startswith("torch")
]


@pytest.mark.parametrize("make_case", EXPERIMENTAL_CASES)
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


@pytest.mark.parametrize(
    "make_slices",
    [
        pytest.param(_mojo_slices, id="mojo", marks=requires_mojo),
        pytest.param(_reference_slices, id="reference"),
    ],
)
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("n", [16, 15])
def test_insertion_pose_gradient_includes_self_mirror_samples(
    make_slices, interpolation, n
):
    """Pose gradients of the insertion with the cutoff at the Nyquist radius n/2.

    The sphere |k| <= n/2 keeps every sample inside the Nyquist box, where the
    insertion is smooth in the pose, and it still includes the self-mirror samples
    at |k| = 0 and (even n) n/2 that the insertion scales by 0.5.
    """
    target = _rfft(_random_real((n, n, n), seed=3), 3)
    slices = _rfft(_random_real((N_POSES, n, n), seed=0), 2)

    def slice_loss(rotations):
        data, _ = make_slices(n, rotations, interpolation, cutoff=n / 2).insert(slices)
        return ((data - target).abs() ** 2).sum()

    rotations = _rotations().requires_grad_(True)
    assert _finite_difference_error(slice_loss, rotations, eps=3e-4) < 5e-2


@pytest.mark.parametrize(
    "make_lines",
    [
        pytest.param(_mojo_lines, id="mojo", marks=requires_mojo),
        pytest.param(_reference_lines, id="reference"),
    ],
)
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("n", [16, 15])
def test_line_insertion_pose_gradient_includes_self_mirror_samples(
    make_lines, interpolation, n
):
    """The line analogue of the test above."""
    target = _rfft(_random_real((n, n, n), seed=3), 3)
    lines = _rfft(_random_real((N_POSES, n), seed=0), 1)

    def line_loss(directions):
        data, _ = make_lines(n, directions, interpolation, cutoff=n / 2).insert(lines)
        return ((data - target).abs() ** 2).sum()

    directions = _directions(3).requires_grad_(True)
    assert _finite_difference_error(line_loss, directions, eps=3e-4) < 5e-2
