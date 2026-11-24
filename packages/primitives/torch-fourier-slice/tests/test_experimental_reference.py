"""The torch reference backend against the Mojo kernels.

``torch_fourier_slice.experimental.reference`` implements the experimental API in
plain torch, and ``torch_fourier_slice.experimental`` falls back to it without the
Mojo kernels. These pin that the two have the same signatures, give the same
values and the same gradients, and that the fallback happens.
"""

import inspect
import math
import subprocess
import sys

import pytest
import torch
from scipy.spatial.transform import Rotation

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
    "oversampling": {"oversampling": 1.5},
    "fftfreq_max": {"fftfreq_max": 0.3},
    "ewald": {"apply_ewald_curvature": True, "ewald_px_size": 0.1},
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
    "oversampling": {"oversampling": 2.0},
    "fftfreq_max": {"fftfreq_max": 0.3},
    "ewald": {"apply_ewald_curvature": True, "ewald_px_size": 0.1},
}


@requires_mojo
@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
@pytest.mark.parametrize("case", INSERTION_CASES)
def test_insert_slices_matches_mojo(case, interpolation):
    args = (_image_rfft(N_POSES), _rotations(N_POSES, seed=4))
    kwargs = {**INSERTION_CASES[case], "interpolation": interpolation}
    volume, weights = reference.insert_central_slices_rfft_3d(*args, **kwargs)
    mojo_volume, mojo_weights = insertion.insert_central_slices_rfft_3d(*args, **kwargs)
    _assert_close(volume, mojo_volume, rtol=1e-5)
    assert (weights is None) == (mojo_weights is None)
    if weights is not None:
        _assert_close(weights, mojo_weights, rtol=1e-5)


@requires_mojo
def test_insert_slices_multichannel_matches_mojo():
    args = (
        _image_rfft(N_POSES, N_CHANNELS),
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


def _insertion_inputs():
    generator = torch.Generator().manual_seed(9)
    return {
        "image_rfft": _image_rfft(N_POSES),
        "rotation_matrices": _rotations(N_POSES, seed=4),
        "shifts_2d": _shifts(N_POSES, 2),
        "shifts_3d": _shifts(N_POSES, 3),
        "weights": torch.rand(
            N_POSES, SIDELENGTH, SIDELENGTH // 2 + 1, generator=generator
        ),
    }


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
def test_insert_slices_gradients_match_mojo(interpolation):
    inputs = _insertion_inputs()
    kwargs = {"interpolation": interpolation}
    function, mojo_function = (
        reference.insert_central_slices_rfft_3d,
        insertion.insert_central_slices_rfft_3d,
    )
    # the data volume depends on the slices, the rotations and the shifts
    actual = _gradients(function, inputs, kwargs, output_index=0)
    expected = _gradients(mojo_function, inputs, kwargs, output_index=0)
    for name in ("image_rfft", "rotation_matrices", "shifts_2d", "shifts_3d"):
        _assert_close(actual[name], expected[name], rtol=1e-4)
    # the weight volume on the weights (and the rotations: see the xfail below)
    actual = _gradients(function, inputs, kwargs, output_index=1)
    expected = _gradients(mojo_function, inputs, kwargs, output_index=1)
    _assert_close(actual["weights"], expected["weights"], rtol=1e-4)


@requires_mojo
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="InsertSlices3D.backward drops the weight volume's dependence on the "
    "rotations, so its rotation gradient (and backproject_2d_to_3d's) is wrong",
)
def test_insert_slices_weight_volume_rotation_gradient_matches_mojo():
    inputs = _insertion_inputs()
    kwargs = {"interpolation": "cubic"}
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
        {"interpolation": "cubic"},
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
