"""The torch reference backend against the Mojo kernels.

``torch_fourier_slice.experimental.reference`` implements the experimental API in
plain torch, and ``torch_fourier_slice.experimental`` falls back to it without the
Mojo kernels. These pin that the two have the same signatures, give the same
values and the same gradients, and that the fallback happens, for the central
slices, the central lines and the real-space layer.
"""

import inspect
import math
import subprocess
import sys

import pytest
import torch
from scipy.spatial.transform import Rotation

from torch_fourier_slice import project_2d_to_1d as canonical_project_2d_to_1d
from torch_fourier_slice.experimental import (
    backproject,
    extraction,
    insertion,
    mojo_kernels_available,
    project,
    reference,
)

requires_mojo = pytest.mark.skipif(
    not mojo_kernels_available(),
    reason="mojo package not installed / kernels failed to compile",
)

SIDELENGTH = 16
VOLUME_SHAPE = (SIDELENGTH,) * 3
# an anisotropic magnification with shear, so it mixes y and x
MAGNIFICATION = torch.tensor([[1.04, 0.03], [-0.02, 0.97]])
N_POSES = 5
N_CHANNELS = 2
MOJO_MODULES = (extraction, insertion, project, backproject)


def _mojo(name):
    """The Mojo implementation of a public function, even when falling back."""
    for module in MOJO_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AssertionError(f"no Mojo implementation of {name}")


def _rotations(n, seed, batch=()):
    matrices = Rotation.random(n * math.prod(batch), random_state=seed).as_matrix()
    return torch.tensor(matrices, dtype=torch.float32).reshape(*batch, n, 3, 3)


def _volume_rfft(*batch, seed=0):
    generator = torch.Generator().manual_seed(seed)
    volume = torch.randn(*batch, *(SIDELENGTH,) * 3, generator=generator)
    return torch.fft.rfftn(volume, dim=(-3, -2, -1))


def _image_rfft(*batch, seed=1):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(*batch, SIDELENGTH, SIDELENGTH, generator=generator)
    return torch.fft.rfftn(images, dim=(-2, -1))


def _shifts(*shape, seed=2):
    generator = torch.Generator().manual_seed(seed)
    return 2.0 * torch.randn(*shape, generator=generator)


def _assert_close(actual, expected, rtol):
    """Agree to ``rtol`` of the largest magnitude (float32, different sum order)."""
    assert actual.shape == expected.shape
    scale = expected.abs().max().clamp(min=1e-12)
    assert ((actual - expected).abs().max() / scale).item() < rtol


def _loss(output, seed=3):
    """A fixed random linear functional of a (complex) output, to backpropagate."""
    generator = torch.Generator().manual_seed(seed)
    output = torch.view_as_real(output) if output.is_complex() else output
    return (output * torch.randn(output.shape, generator=generator)).sum()


def test_reference_signatures_match_mojo():
    for name in reference.__all__:
        assert inspect.signature(getattr(reference, name)) == inspect.signature(
            _mojo(name)
        ), name


def test_falls_back_to_reference_without_mojo():
    code = (
        "import sys\n"
        "sys.modules['mojo'] = None\n"  # make `import mojo.importer` fail
        "from torch_fourier_slice import experimental\n"
        "assert not experimental.mojo_kernels_available()\n"
        "from torch_fourier_slice.experimental import reference\n"
        "for name in reference.__all__:\n"
        "    assert getattr(experimental, name) is getattr(reference, name)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


EXTRACTION_CASES = {
    "plain": {},
    "shifts": {"shifts_2d": _shifts(N_POSES, 2), "shifts_3d": _shifts(N_POSES, 3)},
    "zyx": {
        "shifts_2d": _shifts(N_POSES, 2),
        "shifts_3d": _shifts(N_POSES, 3),
        "zyx_matrices": True,
        "zyx_shifts": True,
        "yx_shifts": True,
    },
    "output_shape": {"output_shape": (24, 24)},
    "oversampled": {"output_shape": (10, 10)},
    "fftfreq_max": {"fftfreq_max": 0.3},
    "ewald": {"apply_ewald_curvature": True, "ewald_px_size": 0.1},
    "magnification": {"transform_matrix": MAGNIFICATION},
    "magnification_ewald": {
        "transform_matrix": MAGNIFICATION,
        "apply_ewald_curvature": True,
        "ewald_px_size": 0.1,
    },
}


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("case", EXTRACTION_CASES)
def test_extract_slices_matches_mojo(case, interpolation):
    args = (_volume_rfft(), _rotations(N_POSES, seed=4))
    kwargs = {**EXTRACTION_CASES[case], "interpolation": interpolation}
    _assert_close(
        reference.extract_central_slices_rfft_3d(*args, **kwargs),
        extraction.extract_central_slices_rfft_3d(*args, **kwargs),
        rtol=1e-5,
    )


@requires_mojo
def test_extract_slices_multichannel_matches_mojo():
    args = (_volume_rfft(N_CHANNELS), _rotations(N_POSES, seed=4, batch=(N_CHANNELS,)))
    kwargs = {"shifts_2d": _shifts(N_CHANNELS, N_POSES, 2), "interpolation": "cubic"}
    _assert_close(
        reference.extract_central_slices_rfft_3d_multichannel(*args, **kwargs),
        extraction.extract_central_slices_rfft_3d_multichannel(*args, **kwargs),
        rtol=1e-5,
    )


INSERTION_CASES = {
    "plain": {},
    "weights": {"weights": torch.rand(N_POSES, SIDELENGTH, SIDELENGTH // 2 + 1)},
    "shifts": {"shifts_2d": _shifts(N_POSES, 2), "shifts_3d": _shifts(N_POSES, 3)},
    "oversampling_2": {"volume_shape": (2 * SIDELENGTH,) * 3},
    "oversampling_1.5": {"volume_shape": (24,) * 3},
    "fftfreq_max": {"fftfreq_max": 0.3},
    "ewald": {"apply_ewald_curvature": True, "ewald_px_size": 0.1},
    "magnification": {"transform_matrix": MAGNIFICATION},
}


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("case", INSERTION_CASES)
def test_insert_slices_matches_mojo(case, interpolation):
    options = dict(INSERTION_CASES[case])
    volume_shape = options.pop("volume_shape", VOLUME_SHAPE)
    args = (_image_rfft(N_POSES), volume_shape, _rotations(N_POSES, seed=4))
    kwargs = {**options, "interpolation": interpolation}
    volume, weights = reference.insert_central_slices_rfft_3d(*args, **kwargs)
    mojo_volume, mojo_weights = insertion.insert_central_slices_rfft_3d(*args, **kwargs)
    _assert_close(volume, mojo_volume, rtol=1e-5)
    _assert_close(weights, mojo_weights, rtol=1e-5)


@requires_mojo
def test_insert_slices_multichannel_matches_mojo():
    args = (
        _image_rfft(N_POSES, N_CHANNELS),
        VOLUME_SHAPE,
        _rotations(N_POSES, seed=4, batch=(N_CHANNELS,)),
    )
    kwargs = {
        "weights": torch.rand(N_POSES, N_CHANNELS, SIDELENGTH, SIDELENGTH // 2 + 1),
        "shifts_3d": _shifts(N_CHANNELS, N_POSES, 3),
        "interpolation": "cubic",
    }
    volume, weights = reference.insert_central_slices_rfft_3d_multichannel(
        *args, **kwargs
    )
    mojo_volume, mojo_weights = insertion.insert_central_slices_rfft_3d_multichannel(
        *args, **kwargs
    )
    _assert_close(volume, mojo_volume, rtol=1e-5)
    _assert_close(weights, mojo_weights, rtol=1e-5)


def _gradients(function, inputs, kwargs, output_index=None):
    """Gradients of ``_loss`` w.r.t. each tensor in ``inputs`` (a name -> tensor dict).

    For a function with several outputs, ``output_index`` picks the one the loss is
    taken over.
    """
    leaves = {name: t.clone().requires_grad_(True) for name, t in inputs.items()}
    output = function(**leaves, **kwargs)
    if output_index is not None:
        output = output[output_index]
    _loss(output).backward()
    return {name: leaf.grad for name, leaf in leaves.items()}


def _extraction_inputs():
    return {
        "volume_rfft": _volume_rfft(),
        "rotation_matrices": _rotations(N_POSES, seed=4),
        "shifts_2d": _shifts(N_POSES, 2),
        "shifts_3d": _shifts(N_POSES, 3),
    }


def _insertion_inputs(weighted=True):
    """Differentiable insertion inputs, with ``weights`` unless not ``weighted``."""
    generator = torch.Generator().manual_seed(9)
    inputs = {
        "image_rfft": _image_rfft(N_POSES),
        "rotation_matrices": _rotations(N_POSES, seed=4),
        "shifts_2d": _shifts(N_POSES, 2),
        "shifts_3d": _shifts(N_POSES, 3),
    }
    if weighted:
        inputs["weights"] = torch.rand(
            N_POSES, SIDELENGTH, SIDELENGTH // 2 + 1, generator=generator
        )
    return inputs


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_extract_slices_gradients_match_mojo(interpolation):
    inputs = _extraction_inputs()
    kwargs = {"interpolation": interpolation}
    expected = _gradients(extraction.extract_central_slices_rfft_3d, inputs, kwargs)
    actual = _gradients(reference.extract_central_slices_rfft_3d, inputs, kwargs)
    for name in inputs:
        _assert_close(actual[name], expected[name], rtol=1e-4)


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_extract_slices_gradients_match_mojo_with_smaller_output(interpolation):
    """With slices smaller than the volume, every gradient keeps the forward's
    Nyquist cutoff (in slice pixels)."""
    inputs = _extraction_inputs()
    kwargs = {"output_shape": (10, 10), "interpolation": interpolation}
    expected = _gradients(extraction.extract_central_slices_rfft_3d, inputs, kwargs)
    actual = _gradients(reference.extract_central_slices_rfft_3d, inputs, kwargs)
    for name in inputs:
        _assert_close(actual[name], expected[name], rtol=1e-4)


@requires_mojo
@pytest.mark.parametrize("output_index", [0, 1])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_insert_slices_gradients_match_mojo_into_larger_volume(
    interpolation, output_index
):
    """Likewise for an insertion into a volume larger than the slices."""
    inputs = _insertion_inputs()
    kwargs = {"volume_shape": (2 * SIDELENGTH,) * 3, "interpolation": interpolation}
    function, mojo_function = (
        reference.insert_central_slices_rfft_3d,
        insertion.insert_central_slices_rfft_3d,
    )
    actual = _gradients(function, inputs, kwargs, output_index)
    expected = _gradients(mojo_function, inputs, kwargs, output_index)
    # the data volume depends on all inputs but the weights, the weight volume on
    # the weights and the rotations
    names = ["weights", "rotation_matrices"] if output_index else list(inputs)[:-1]
    for name in names:
        _assert_close(actual[name], expected[name], rtol=1e-4)


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_magnified_slices_gradients_match_mojo(interpolation):
    """The pose, shift, data and weight gradients with a magnification."""
    kwargs = {
        "transform_matrix": MAGNIFICATION,
        "interpolation": interpolation,
        "apply_ewald_curvature": True,
        "ewald_px_size": 0.1,
    }
    inputs = _extraction_inputs()
    expected = _gradients(extraction.extract_central_slices_rfft_3d, inputs, kwargs)
    actual = _gradients(reference.extract_central_slices_rfft_3d, inputs, kwargs)
    for name in inputs:
        _assert_close(actual[name], expected[name], rtol=1e-4)

    inputs = _insertion_inputs()
    kwargs = {"volume_shape": VOLUME_SHAPE, **kwargs}
    functions = (
        reference.insert_central_slices_rfft_3d,
        insertion.insert_central_slices_rfft_3d,
    )
    for output_index, names in (
        (0, list(inputs)[:-1]),
        (1, ["weights", "rotation_matrices"]),
    ):
        actual = _gradients(functions[0], inputs, kwargs, output_index)
        expected = _gradients(functions[1], inputs, kwargs, output_index)
        for name in names:
            _assert_close(actual[name], expected[name], rtol=1e-4)


@pytest.mark.parametrize(
    "matrix", [torch.eye(3), torch.zeros(2, 2)], ids=["not-2x2", "singular"]
)
def test_transform_matrix_must_be_an_invertible_2x2(matrix):
    with pytest.raises(ValueError, match="transform_matrix"):
        reference.extract_central_slices_rfft_3d(
            _volume_rfft(), _rotations(N_POSES, seed=4), transform_matrix=matrix
        )


@requires_mojo
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_insert_slices_gradients_match_mojo(interpolation, weighted):
    inputs = _insertion_inputs(weighted)
    kwargs = {"volume_shape": VOLUME_SHAPE, "interpolation": interpolation}
    function, mojo_function = (
        reference.insert_central_slices_rfft_3d,
        insertion.insert_central_slices_rfft_3d,
    )
    # the data volume depends on the slices, the rotations and the shifts
    actual = _gradients(function, inputs, kwargs, output_index=0)
    expected = _gradients(mojo_function, inputs, kwargs, output_index=0)
    for name in ("image_rfft", "rotation_matrices", "shifts_2d", "shifts_3d"):
        _assert_close(actual[name], expected[name], rtol=1e-4)
    # the weight volume on the weights (and the rotations: see below)
    if weighted:
        actual = _gradients(function, inputs, kwargs, output_index=1)
        expected = _gradients(mojo_function, inputs, kwargs, output_index=1)
        _assert_close(actual["weights"], expected["weights"], rtol=1e-4)


@requires_mojo
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_insert_slices_weight_volume_rotation_gradient_matches_mojo(
    interpolation, weighted
):
    """The weights are splatted at the rotated positions, so the weight volume
    has a rotation gradient too, unit weights included."""
    inputs = _insertion_inputs(weighted)
    kwargs = {"volume_shape": VOLUME_SHAPE, "interpolation": interpolation}
    actual = _gradients(
        reference.insert_central_slices_rfft_3d, inputs, kwargs, output_index=1
    )
    expected = _gradients(
        insertion.insert_central_slices_rfft_3d, inputs, kwargs, output_index=1
    )
    _assert_close(actual["rotation_matrices"], expected["rotation_matrices"], 1e-4)


def _assert_matches_finite_differences(
    function, inputs, name, kwargs, output_index=None
):
    """Autograd agrees with central differences along random directions of ``name``.

    Tricubic interpolation is smooth enough for differences to be accurate.
    """
    grad = _gradients(function, inputs, kwargs, output_index)[name]

    def loss(perturbed):
        output = function(**{**inputs, name: perturbed}, **kwargs)
        if output_index is not None:
            output = output[output_index]
        return _loss(output).item()

    generator = torch.Generator().manual_seed(7)
    eps = 1e-3
    pairs = []
    for _ in range(3):
        direction = torch.randn(inputs[name].shape, generator=generator)
        difference = loss(inputs[name] + eps * direction)
        difference -= loss(inputs[name] - eps * direction)
        pairs.append((difference / (2 * eps), (grad * direction).sum().item()))
    scale = max(abs(fd) for fd, _ in pairs)
    for finite_difference, autograd in pairs:
        assert abs(autograd - finite_difference) < 1e-2 * scale


@pytest.mark.parametrize("name", ["rotation_matrices", "shifts_2d", "shifts_3d"])
def test_reference_extraction_gradients_match_finite_differences(name):
    _assert_matches_finite_differences(
        reference.extract_central_slices_rfft_3d,
        _extraction_inputs(),
        name,
        {"interpolation": "cubic"},
    )


@pytest.mark.parametrize(
    ("name", "output_index"),
    [
        ("rotation_matrices", 0),
        ("rotation_matrices", 1),
        ("shifts_2d", 0),
        ("shifts_3d", 0),
    ],
)
def test_reference_insertion_gradients_match_finite_differences(name, output_index):
    _assert_matches_finite_differences(
        reference.insert_central_slices_rfft_3d,
        _insertion_inputs(),
        name,
        {"volume_shape": VOLUME_SHAPE, "interpolation": "cubic"},
        output_index,
    )


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_project_and_backproject_match_mojo(interpolation):
    generator = torch.Generator().manual_seed(5)
    volume = torch.randn((SIDELENGTH,) * 3, generator=generator)
    rotations = _rotations(N_POSES, seed=4)
    kwargs = {"shifts_2d": _shifts(N_POSES, 2), "interpolation": interpolation}
    images = project.project_3d_to_2d(volume, rotations, **kwargs)
    _assert_close(
        reference.project_3d_to_2d(volume, rotations, **kwargs), images, rtol=1e-5
    )
    _assert_close(
        reference.backproject_2d_to_3d(images, rotations, **kwargs),
        backproject.backproject_2d_to_3d(images, rotations, **kwargs),
        rtol=1e-5,
    )


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_backproject_gradients_match_mojo(interpolation):
    """``backproject_2d_to_3d`` divides by the weight volume, so its rotation
    gradient also goes through the weight volume's pose dependence."""
    generator = torch.Generator().manual_seed(5)
    inputs = {
        "images": torch.randn(N_POSES, SIDELENGTH, SIDELENGTH, generator=generator),
        "rotation_matrices": _rotations(N_POSES, seed=4),
        "shifts_2d": _shifts(N_POSES, 2),
    }
    kwargs = {"interpolation": interpolation}
    actual = _gradients(reference.backproject_2d_to_3d, inputs, kwargs)
    expected = _gradients(backproject.backproject_2d_to_3d, inputs, kwargs)
    for name in inputs:
        _assert_close(actual[name], expected[name], rtol=1e-4)


@requires_mojo
@pytest.mark.parametrize("sidelength", [SIDELENGTH, SIDELENGTH - 1])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_project_2d_to_1d_matches_mojo(interpolation, sidelength):
    """Values and gradients w.r.t. the image, the directions and the shifts."""
    generator = torch.Generator().manual_seed(5)
    directions = torch.randn(N_POSES, 2, generator=generator)
    inputs = {
        "image": torch.randn(sidelength, sidelength, generator=generator),
        "directions": directions / directions.norm(dim=-1, keepdim=True),
        "shifts_2d": _shifts(N_POSES, 2),
    }
    kwargs = {"interpolation": interpolation}
    _assert_close(
        reference.project_2d_to_1d(**inputs, **kwargs),
        project.project_2d_to_1d(**inputs, **kwargs),
        rtol=1e-5,
    )
    actual = _gradients(reference.project_2d_to_1d, inputs, kwargs)
    expected = _gradients(project.project_2d_to_1d, inputs, kwargs)
    for name in inputs:
        _assert_close(actual[name], expected[name], rtol=1e-4)

    images = torch.randn(N_CHANNELS, sidelength, sidelength, generator=generator)
    _assert_close(
        reference.project_2d_to_1d_multichannel(images, inputs["directions"]),
        project.project_2d_to_1d_multichannel(images, inputs["directions"]),
        rtol=1e-5,
    )


@pytest.mark.parametrize("sidelength", [64, 63])
def test_reference_project_2d_to_1d_matches_canonical(sidelength):
    """Runs without Mojo: an off-centre blob well inside the box projects as the
    canonical function does, with u the first row of its active rotation R.

    The canonical function also subtracts the image's edge mean before padding, so
    they only match for edges at ~0. The blob's exact projection is the reference
    for the centred case (to the linear interpolation's accuracy).
    """
    axis = torch.arange(sidelength, dtype=torch.float32) - sidelength // 2
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    angles = (
        2 * torch.pi * torch.rand(N_POSES, generator=torch.Generator().manual_seed(7))
    )
    cos, sin = angles.cos(), angles.sin()
    rotations = torch.stack(
        [torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2
    )

    image = torch.exp(-((yy - 4.0) ** 2 + (xx + 3.0) ** 2) / 18.0)
    _assert_close(
        reference.project_2d_to_1d(image, rotations[..., 0, :]),
        canonical_project_2d_to_1d(image, rotations),
        rtol=1e-5,
    )

    centred = torch.exp(-(yy**2 + xx**2) / 32.0)
    lines = reference.project_2d_to_1d(centred, rotations[..., 0, :])
    _assert_close(lines, centred.sum(dim=0).expand_as(lines), rtol=1e-2)


# --- central lines ----------------------------------------------------------------

# per line dimension: the array lines are cut from, and the names of the shift and
# axis-order arguments
LINE_KINDS = {
    "3d": {
        "array": lambda *batch: _volume_rfft(*batch),
        "ndim": 3,
        "shifts": "shifts_3d",
        "order": {"zyx_directions": True, "zyx_shifts": True},
    },
    "2d": {
        "array": lambda *batch: _image_rfft(*batch),
        "ndim": 2,
        "shifts": "shifts_2d",
        "order": {"yx_directions": True, "yx_shifts": True},
    },
}


def _array_shape(kind, sidelength=SIDELENGTH):
    """The ``volume_shape`` / ``image_shape`` keyword of a line insertion."""
    if kind == "3d":
        return {"volume_shape": (sidelength,) * 3}
    return {"image_shape": (sidelength,) * 2}


def _line_function(module, operation, kind, multichannel=False):
    suffix = "_multichannel" if multichannel else ""
    return getattr(module, f"{operation}_central_lines_rfft_{kind}{suffix}")


def _directions(n, ndim, seed, batch=()):
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn(*batch, n, ndim, generator=generator)
    return directions / directions.norm(dim=-1, keepdim=True)


def _lines_rfft(*batch, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.fft.rfft(torch.randn(*batch, SIDELENGTH, generator=generator))


def _line_weights(*batch, seed=9):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(*batch, SIDELENGTH // 2 + 1, generator=generator)


def _line_case(kind, case):
    """Keyword arguments for one of the line cases below."""
    shifts, ndim = LINE_KINDS[kind]["shifts"], LINE_KINDS[kind]["ndim"]
    return {
        "plain": {},
        "shifts": {shifts: _shifts(N_POSES, ndim)},
        "axis_order": {shifts: _shifts(N_POSES, ndim), **LINE_KINDS[kind]["order"]},
        "output_length": {"output_length": 24},
        "oversampled": {"output_length": 10},
        "fftfreq_max": {"fftfreq_max": 0.3},
    }[case]


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize(
    "case",
    ["plain", "shifts", "axis_order", "output_length", "oversampled", "fftfreq_max"],
)
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_extract_lines_matches_mojo(kind, case, interpolation):
    ndim = LINE_KINDS[kind]["ndim"]
    args = (LINE_KINDS[kind]["array"](), _directions(N_POSES, ndim, seed=4))
    kwargs = {**_line_case(kind, case), "interpolation": interpolation}
    _assert_close(
        _line_function(reference, "extract", kind)(*args, **kwargs),
        _line_function(extraction, "extract", kind)(*args, **kwargs),
        rtol=1e-5,
    )


@requires_mojo
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_extract_lines_multichannel_matches_mojo(kind):
    ndim, shifts = LINE_KINDS[kind]["ndim"], LINE_KINDS[kind]["shifts"]
    args = (
        LINE_KINDS[kind]["array"](N_CHANNELS),
        _directions(N_POSES, ndim, seed=4, batch=(N_CHANNELS,)),
    )
    kwargs = {shifts: _shifts(N_CHANNELS, N_POSES, ndim), "interpolation": "cubic"}
    _assert_close(
        _line_function(reference, "extract", kind, True)(*args, **kwargs),
        _line_function(extraction, "extract", kind, True)(*args, **kwargs),
        rtol=1e-5,
    )


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "weights",
        "shifts",
        "axis_order",
        "oversampling_2",
        "oversampling_1.5",
        "fftfreq_max",
    ],
)
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_insert_lines_matches_mojo(kind, case, interpolation):
    ndim = LINE_KINDS[kind]["ndim"]
    sidelength = {"oversampling_2": 2 * SIDELENGTH, "oversampling_1.5": 24}.get(
        case, SIDELENGTH
    )
    args = (_lines_rfft(N_POSES), (sidelength,) * ndim, _directions(N_POSES, ndim, 4))
    if case == "weights":
        options = {"weights": _line_weights(N_POSES)}
    elif case.startswith("oversampling"):
        options = {}
    else:
        options = _line_case(kind, case)
    kwargs = {**options, "interpolation": interpolation}
    data, weights = _line_function(reference, "insert", kind)(*args, **kwargs)
    mojo_data, mojo_weights = _line_function(insertion, "insert", kind)(*args, **kwargs)
    _assert_close(data, mojo_data, rtol=1e-5)
    _assert_close(weights, mojo_weights, rtol=1e-5)


@requires_mojo
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_insert_lines_multichannel_matches_mojo(kind):
    ndim, shifts = LINE_KINDS[kind]["ndim"], LINE_KINDS[kind]["shifts"]
    args = (
        _lines_rfft(N_POSES, N_CHANNELS),
        (SIDELENGTH,) * ndim,
        _directions(N_POSES, ndim, seed=4, batch=(N_CHANNELS,)),
    )
    kwargs = {
        "weights": _line_weights(N_POSES, N_CHANNELS),
        shifts: _shifts(N_CHANNELS, N_POSES, ndim),
        "interpolation": "cubic",
    }
    data, weights = _line_function(reference, "insert", kind, True)(*args, **kwargs)
    mojo_data, mojo_weights = _line_function(insertion, "insert", kind, True)(
        *args, **kwargs
    )
    _assert_close(data, mojo_data, rtol=1e-5)
    _assert_close(weights, mojo_weights, rtol=1e-5)


def _line_extraction_inputs(kind):
    ndim = LINE_KINDS[kind]["ndim"]
    array_name = "volume_rfft" if kind == "3d" else "image_rfft"
    return {
        array_name: LINE_KINDS[kind]["array"](),
        "directions": _directions(N_POSES, ndim, seed=4),
        LINE_KINDS[kind]["shifts"]: _shifts(N_POSES, ndim),
    }


def _line_insertion_inputs(kind, weighted=True):
    ndim = LINE_KINDS[kind]["ndim"]
    inputs = {
        "lines": _lines_rfft(N_POSES),
        "directions": _directions(N_POSES, ndim, seed=4),
        LINE_KINDS[kind]["shifts"]: _shifts(N_POSES, ndim),
    }
    if weighted:
        inputs["weights"] = _line_weights(N_POSES)
    return inputs


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_extract_lines_gradients_match_mojo(kind, interpolation):
    inputs = _line_extraction_inputs(kind)
    kwargs = {"interpolation": interpolation}
    actual = _gradients(_line_function(reference, "extract", kind), inputs, kwargs)
    expected = _gradients(_line_function(extraction, "extract", kind), inputs, kwargs)
    for name in inputs:
        _assert_close(actual[name], expected[name], rtol=1e-4)


@requires_mojo
@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_insert_lines_gradients_match_mojo(kind, interpolation, weighted):
    inputs = _line_insertion_inputs(kind, weighted)
    kwargs = {**_array_shape(kind), "interpolation": interpolation}
    function = _line_function(reference, "insert", kind)
    mojo_function = _line_function(insertion, "insert", kind)
    # the data depends on the lines, the directions and the shifts
    actual = _gradients(function, inputs, kwargs, output_index=0)
    expected = _gradients(mojo_function, inputs, kwargs, output_index=0)
    for name in ("lines", "directions", LINE_KINDS[kind]["shifts"]):
        _assert_close(actual[name], expected[name], rtol=1e-4)
    # the weights, unit ones included, are splatted along the directions too
    actual = _gradients(function, inputs, kwargs, output_index=1)
    expected = _gradients(mojo_function, inputs, kwargs, output_index=1)
    for name in ("weights", "directions") if weighted else ("directions",):
        _assert_close(actual[name], expected[name], rtol=1e-4)


@pytest.mark.parametrize("name", ["directions", "shifts"])
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_reference_line_extraction_gradients_match_finite_differences(kind, name):
    if name == "shifts":
        name = LINE_KINDS[kind]["shifts"]
    _assert_matches_finite_differences(
        _line_function(reference, "extract", kind),
        _line_extraction_inputs(kind),
        name,
        {"interpolation": "cubic"},
    )


@pytest.mark.parametrize(
    ("name", "output_index"), [("directions", 0), ("directions", 1), ("shifts", 0)]
)
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_reference_line_insertion_gradients_match_finite_differences(
    kind, name, output_index
):
    if name == "shifts":
        name = LINE_KINDS[kind]["shifts"]
    _assert_matches_finite_differences(
        _line_function(reference, "insert", kind),
        _line_insertion_inputs(kind),
        name,
        {**_array_shape(kind), "interpolation": "cubic"},
        output_index,
    )


# --- odd sizes ----------------------------------------------------------------

# (array side, projection side): odd throughout, an odd slice or line of an even
# array, and an even one of an odd array
ODD_SIDES = {"odd": (15, 15), "odd_projection": (16, 11), "odd_array": (15, 10)}


def _random_rfft(shape, ndim, seed):
    """The rfft over the last ``ndim`` axes of a random real array."""
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(shape, generator=generator)
    return torch.fft.rfftn(real, dim=tuple(range(-ndim, 0)))


def _assert_gradients_match(function, mojo_function, inputs, kwargs, groups):
    """Gradients agree for each ``output_index -> input names`` in ``groups``."""
    for output_index, names in groups.items():
        actual = _gradients(function, inputs, kwargs, output_index)
        expected = _gradients(mojo_function, inputs, kwargs, output_index)
        for name in names:
            _assert_close(actual[name], expected[name], rtol=1e-4)


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("sides", ODD_SIDES.values(), ids=ODD_SIDES.keys())
def test_odd_slices_match_mojo(sides, interpolation):
    """Values and gradients of slices cut from and inserted into odd arrays."""
    side, proj_side = sides
    poses = {
        "rotation_matrices": _rotations(N_POSES, seed=4),
        "shifts_2d": _shifts(N_POSES, 2),
        "shifts_3d": _shifts(N_POSES, 3),
    }
    inputs = {"volume_rfft": _random_rfft((side,) * 3, 3, seed=0), **poses}
    kwargs = {"output_shape": (proj_side,) * 2, "interpolation": interpolation}
    functions = (
        reference.extract_central_slices_rfft_3d,
        extraction.extract_central_slices_rfft_3d,
    )
    _assert_close(*(f(**inputs, **kwargs) for f in functions), rtol=1e-5)
    _assert_gradients_match(*functions, inputs, kwargs, {None: list(inputs)})

    generator = torch.Generator().manual_seed(9)
    inputs = {
        "image_rfft": _random_rfft((N_POSES, proj_side, proj_side), 2, seed=1),
        **poses,
        "weights": torch.rand(
            N_POSES, proj_side, proj_side // 2 + 1, generator=generator
        ),
    }
    kwargs = {"volume_shape": (side,) * 3, "interpolation": interpolation}
    functions = (
        reference.insert_central_slices_rfft_3d,
        insertion.insert_central_slices_rfft_3d,
    )
    (data, weights), (mojo_data, mojo_weights) = (
        f(**inputs, **kwargs) for f in functions
    )
    _assert_close(data, mojo_data, rtol=1e-5)
    _assert_close(weights, mojo_weights, rtol=1e-5)
    groups = {0: list(inputs)[:-1], 1: ["weights", "rotation_matrices"]}
    _assert_gradients_match(*functions, inputs, kwargs, groups)


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("sides", ODD_SIDES.values(), ids=ODD_SIDES.keys())
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_odd_lines_match_mojo(kind, sides, interpolation):
    """Values and gradients of lines of odd length, or of odd arrays."""
    side, line_length = sides
    ndim, shifts = LINE_KINDS[kind]["ndim"], LINE_KINDS[kind]["shifts"]
    poses = {
        "directions": _directions(N_POSES, ndim, seed=4),
        shifts: _shifts(N_POSES, ndim),
    }
    array_name = "volume_rfft" if kind == "3d" else "image_rfft"
    inputs = {array_name: _random_rfft((side,) * ndim, ndim, seed=0), **poses}
    kwargs = {"output_length": line_length, "interpolation": interpolation}
    functions = tuple(
        _line_function(module, "extract", kind) for module in (reference, extraction)
    )
    _assert_close(*(f(**inputs, **kwargs) for f in functions), rtol=1e-5)
    _assert_gradients_match(*functions, inputs, kwargs, {None: list(inputs)})

    generator = torch.Generator().manual_seed(9)
    inputs = {
        "lines": _random_rfft((N_POSES, line_length), 1, seed=1),
        **poses,
        "weights": torch.rand(N_POSES, line_length // 2 + 1, generator=generator),
    }
    kwargs = {
        **_array_shape(kind, side),
        "line_length": line_length,
        "interpolation": interpolation,
    }
    functions = tuple(
        _line_function(module, "insert", kind) for module in (reference, insertion)
    )
    (data, weights), (mojo_data, mojo_weights) = (
        f(**inputs, **kwargs) for f in functions
    )
    _assert_close(data, mojo_data, rtol=1e-5)
    _assert_close(weights, mojo_weights, rtol=1e-5)
    groups = {0: list(inputs)[:-1], 1: ["weights", "directions"]}
    _assert_gradients_match(*functions, inputs, kwargs, groups)


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(insertion, id="mojo", marks=requires_mojo),
        pytest.param(reference, id="reference"),
    ],
)
@pytest.mark.parametrize("kind", LINE_KINDS)
def test_line_length_must_fit_the_lines(module, kind):
    """A length of 2m or 2m + 1 has m + 1 rfft samples; any other is an error."""
    ndim = LINE_KINDS[kind]["ndim"]
    lines = _lines_rfft(N_POSES)  # SIDELENGTH // 2 + 1 samples
    args = (lines, (SIDELENGTH,) * ndim, _directions(N_POSES, ndim, seed=4))
    insert = _line_function(module, "insert", kind)
    insert(*args, line_length=SIDELENGTH + 1)
    with pytest.raises(ValueError, match="line_length"):
        insert(*args, line_length=SIDELENGTH + 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_reference_runs_on_gpu():
    volume, rotations, shifts = (
        _volume_rfft(),
        _rotations(N_POSES, 4),
        _shifts(N_POSES, 3),
    )
    on_cpu = reference.extract_central_slices_rfft_3d(
        volume, rotations, shifts_3d=shifts, interpolation="cubic"
    )
    on_gpu = reference.extract_central_slices_rfft_3d(
        volume.cuda(), rotations.cuda(), shifts_3d=shifts.cuda(), interpolation="cubic"
    )
    assert on_gpu.is_cuda
    _assert_close(on_gpu.cpu(), on_cpu, rtol=1e-5)

    lines, directions = _lines_rfft(N_POSES), _directions(N_POSES, 3, seed=4)
    on_cpu, _ = reference.insert_central_lines_rfft_3d(
        lines, VOLUME_SHAPE, directions, interpolation="cubic"
    )
    on_gpu, _ = reference.insert_central_lines_rfft_3d(
        lines.cuda(), VOLUME_SHAPE, directions.cuda(), interpolation="cubic"
    )
    assert on_gpu.is_cuda
    _assert_close(on_gpu.cpu(), on_cpu, rtol=1e-5)
