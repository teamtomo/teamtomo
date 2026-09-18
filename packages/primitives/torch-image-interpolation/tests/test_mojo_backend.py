"""Mojo kernels vs the pure-torch reference: outputs, gradients, dispatch rules.

Every comparison runs the same public function under ``use_backend("torch")``
and ``use_backend("mojo")``. Coordinates are random continuous positions with a
deliberate fraction outside the image so the masking path is exercised.
"""

import pytest
import torch

import torch_image_interpolation as tii
from torch_image_interpolation import (
    insert_into_image_1d,
    insert_into_image_2d,
    insert_into_image_3d,
    sample_image_1d,
    sample_image_2d,
    sample_image_3d,
    use_backend,
)

pytestmark = pytest.mark.skipif(
    not tii.mojo_available("cpu"),
    reason="mojo package not installed / kernels failed to compile",
)

# GPU kernels are exercised on whichever accelerator is present and compiles
_GPU = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else None
)
DEVICES = ["cpu"] + ([_GPU] if _GPU is not None and tii.mojo_available(_GPU) else [])

SAMPLE = {1: sample_image_1d, 2: sample_image_2d, 3: sample_image_3d}
INSERT = {1: insert_into_image_1d, 2: insert_into_image_2d, 3: insert_into_image_3d}
SHAPES = {1: (37,), 2: (19, 23), 3: (11, 13, 17)}
SAMPLE_MODES = {
    1: ["nearest", "linear", "cubic"],
    2: ["nearest", "bilinear", "bicubic"],
    3: ["nearest", "trilinear"],
}
INSERT_MODES = {
    1: ["nearest", "linear"],
    2: ["nearest", "bilinear"],
    3: ["nearest", "trilinear"],
}


def _image(ndim, dtype, channels, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = SHAPES[ndim] if channels is None else (channels, *SHAPES[ndim])
    if dtype.is_complex:
        real = torch.randn(shape, generator=g)
        imag = torch.randn(shape, generator=g)
        return torch.complex(real, imag).to(dtype)
    return torch.randn(shape, generator=g, dtype=dtype)


def _coords(ndim, n=(5, 7), seed=1, spill=0.15):
    """Random coordinates over the image, ~`spill` of them outside it."""
    g = torch.Generator().manual_seed(seed)
    extents = torch.tensor(SHAPES[ndim], dtype=torch.float32) - 1
    c = (
        torch.rand((*n, ndim), generator=g) * (extents * (1 + 2 * spill))
        - extents * spill
    )
    return c.squeeze(-1) if ndim == 1 else c


def _both(fn, *args, **kwargs):
    with use_backend("torch"):
        ref = fn(*args, **kwargs)
    with use_backend("mojo"):
        out = fn(*args, **kwargs)
    return ref, out


def _cpu(g):
    """Move a gradient to the CPU; `nearest` yields no coordinate gradient in torch."""
    return None if g is None else g.cpu()


def _close(a, b, rtol=1e-4, atol=1e-4):
    assert a.shape == b.shape and a.dtype == b.dtype, (
        a.shape,
        b.shape,
        a.dtype,
        b.dtype,
    )
    scale = a.abs().max().clamp(min=1.0)
    assert torch.allclose(a, b, rtol=rtol, atol=atol * scale), (a - b).abs().max()


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("channels", [None, 3])
def test_sample_matches_torch(ndim, dtype, channels):
    image = _image(ndim, dtype, channels)
    coords = _coords(ndim)
    for mode in SAMPLE_MODES[ndim]:
        ref, out = _both(SAMPLE[ndim], image, coords, interpolation=mode)
        _close(ref, out)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_sample_integer_coordinates_are_exact(ndim):
    """Integer coordinates read the pixel itself -- for every mode."""
    image = _image(ndim, torch.float32, None)
    g = torch.Generator().manual_seed(3)
    extents = torch.tensor(SHAPES[ndim])
    coords = (torch.rand((40, ndim), generator=g) * extents).floor()
    idx = tuple(coords.long().T)
    expected = image[idx]
    coords = coords.squeeze(-1) if ndim == 1 else coords
    for mode in SAMPLE_MODES[ndim]:
        with use_backend("mojo"):
            out = SAMPLE[ndim](image, coords, interpolation=mode)
        assert torch.allclose(out, expected, atol=1e-5), mode


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_sample_gradients_match_torch(ndim, dtype):
    image = _image(ndim, dtype, 2)
    coords = _coords(ndim)
    for mode in SAMPLE_MODES[ndim]:
        grads = {}
        for backend in ("torch", "mojo"):
            img = image.clone().requires_grad_(True)
            crd = coords.clone().requires_grad_(True)
            with use_backend(backend):
                out = SAMPLE[ndim](img, crd, interpolation=mode)
            (
                out.real.sum() + 2 * out.imag.sum() if out.is_complex() else out.sum()
            ).backward()
            grads[backend] = (img.grad, crd.grad)
        _close(grads["torch"][0], grads["mojo"][0])
        if mode == "nearest":
            assert torch.count_nonzero(grads["mojo"][1]) == 0
        else:
            _close(grads["torch"][1], grads["mojo"][1], rtol=1e-3, atol=1e-3)


def test_sample_saves_no_stencil_intermediates():
    """Only image + coordinates are saved for backward (memory is O(image + n))."""
    image = _image(3, torch.complex64, None).requires_grad_(True)
    coords = _coords(3, n=(1000,)).requires_grad_(True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(t.numel()), lambda t: t
    ):
        with use_backend("mojo"):
            sample_image_3d(image, coords)
    # 2 * image (view_as_real) + 3 * n coordinates, and nothing per tap
    assert sum(saved) <= 2 * image.numel() + 3 * coords.shape[0] + 16


# ---------------------------------------------------------------------------
# insertion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("channels", [None, 3])
def test_insert_matches_torch(ndim, dtype, channels):
    coords = _coords(ndim, n=(6, 5))
    values = _image(ndim, dtype, None, seed=5)  # any tensor of the right dtype
    values = torch.randn(
        (6, 5) + (() if channels is None else (channels,)), dtype=dtype
    )
    for mode in INSERT_MODES[ndim]:
        results = {}
        for backend in ("torch", "mojo"):
            image = _image(ndim, dtype, channels, seed=7) * 0.1
            with use_backend(backend):
                img, w = INSERT[ndim](values, coords, image, interpolation=mode)
            results[backend] = (img, w)
        _close(results["torch"][0], results["mojo"][0])
        _close(results["torch"][1], results["mojo"][1])


def test_insert_is_in_place_and_accumulates_weights():
    image = torch.zeros(19, 23)
    weights = torch.zeros(19, 23)
    coords = _coords(2, n=(30,), spill=0.0)
    values = torch.ones(30)
    with use_backend("mojo"):
        out, w = insert_into_image_2d(values, coords, image, weights=weights)
        out2, w2 = insert_into_image_2d(values, coords, out, weights=w)
    assert torch.allclose(image, out2)  # same storage, updated twice
    assert torch.allclose(w2, 2 * w2 / 2) and torch.allclose(
        w2.sum(), torch.tensor(60.0)
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_insert_gradients_match_torch(ndim, dtype):
    coords = _coords(ndim, n=(6, 5))
    values = torch.randn((6, 5, 2), dtype=dtype)
    for mode in INSERT_MODES[ndim]:
        grads = {}
        for backend in ("torch", "mojo"):
            base = (_image(ndim, dtype, 2, seed=9) * 0.1).requires_grad_(True)
            image = base * 1.0  # non-leaf so the in-place insert is legal
            vals = values.clone().requires_grad_(True)
            crd = coords.clone().requires_grad_(True)
            with use_backend(backend):
                img, w = INSERT[ndim](vals, crd, image, interpolation=mode)
            loss = (
                (img.real * 1.0 + img.imag * 2.0).sum()
                if img.is_complex()
                else img.sum()
            )
            loss = loss + (w * torch.linspace(0, 1, w.numel()).reshape(w.shape)).sum()
            loss.backward()
            grads[backend] = (base.grad, vals.grad, crd.grad)
        _close(grads["torch"][0], grads["mojo"][0])
        _close(grads["torch"][1], grads["mojo"][1])
        if mode == "nearest":
            assert torch.count_nonzero(grads["mojo"][2]) == 0
        else:
            _close(grads["torch"][2], grads["mojo"][2], rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# dispatch rules
# ---------------------------------------------------------------------------


def test_unsupported_dtype_falls_back_in_auto_and_raises_in_mojo():
    image = torch.rand(19, 23, dtype=torch.float64)
    coords = _coords(2).double()
    with use_backend("auto"):
        out = sample_image_2d(image, coords)
    with use_backend("torch"):
        ref = sample_image_2d(image, coords)
    assert torch.equal(out, ref) and out.dtype == torch.float64
    with use_backend("mojo"), pytest.raises(RuntimeError, match="float32"):
        sample_image_2d(image, coords)


def test_float64_weights_fall_back_to_torch():
    image = torch.rand(4, 4)
    coords = _coords(2, n=(3,), spill=0.0)
    weights = torch.zeros(4, 4, dtype=torch.float64)
    with use_backend("auto"):
        _, w = insert_into_image_2d(torch.ones(3), coords, image, weights=weights)
    assert w.dtype == torch.float64


def test_backend_setting_roundtrip():
    assert tii.get_backend() in ("auto", "torch", "mojo")
    before = tii.get_backend()
    with use_backend("torch"):
        assert tii.get_backend() == "torch"
    assert tii.get_backend() == before
    with pytest.raises(ValueError):
        tii.set_backend("numpy")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# large n: the CPU scatter kernels switch to slab-partitioned, atomic-free
# execution above a few thousand samples -- cover it explicitly
# ---------------------------------------------------------------------------

N_PARTITIONED = (300, 250)  # 75k samples


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_insert_large_n_matches_torch(ndim, dtype):
    coords = _coords(ndim, n=N_PARTITIONED, seed=11)
    values = torch.randn((*N_PARTITIONED, 2), dtype=dtype)
    for mode in INSERT_MODES[ndim]:
        results = {}
        for backend in ("torch", "mojo"):
            image = torch.zeros((2, *SHAPES[ndim]), dtype=dtype)
            weights = torch.zeros(SHAPES[ndim])
            with use_backend(backend):
                img, w = INSERT[ndim](
                    values, coords, image, weights, interpolation=mode
                )
            results[backend] = (img, w)
        # sums of ~thousands of terms per voxel: allow float32 accumulation error
        _close(results["torch"][0], results["mojo"][0], rtol=1e-3, atol=1e-3)
        _close(results["torch"][1], results["mojo"][1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_sample_backward_large_n_matches_torch(ndim):
    image = _image(ndim, torch.float32, 2)
    coords = _coords(ndim, n=N_PARTITIONED, seed=12)
    for mode in SAMPLE_MODES[ndim]:
        grads = {}
        for backend in ("torch", "mojo"):
            img = image.clone().requires_grad_(True)
            crd = coords.clone().requires_grad_(True)
            with use_backend(backend):
                out = SAMPLE[ndim](img, crd, interpolation=mode)
            (
                out * torch.linspace(-1, 1, out.numel()).reshape(out.shape)
            ).sum().backward()
            grads[backend] = (img.grad, crd.grad)
        _close(grads["torch"][0], grads["mojo"][0], rtol=1e-3, atol=1e-3)
        if mode != "nearest":
            _close(grads["torch"][1], grads["mojo"][1], rtol=1e-3, atol=1e-3)
        # samples outside the image get exactly zero coordinate gradient
        outside = ~torch.all(
            (coords.reshape(-1, ndim) >= 0)
            & (coords.reshape(-1, ndim) <= torch.tensor(SHAPES[ndim]) - 1),
            dim=-1,
        )
        assert torch.count_nonzero(grads["mojo"][1].reshape(-1, ndim)[outside]) == 0


def test_insert_gradients_through_view_inputs():
    """In-place Functions on views may return one tensor: covers the split path."""
    coords = _coords(2, n=(6, 5))
    values = torch.randn((6, 5, 2))
    grads = {}
    for backend in ("torch", "mojo"):
        stack = (torch.randn(3, 2, *SHAPES[2]) * 0.1).requires_grad_(True)
        image = (stack * 1.0)[1]  # a view of a non-leaf
        weights = torch.zeros(2, *SHAPES[2])[0]  # a view
        vals = values.clone().requires_grad_(True)
        crd = coords.clone().requires_grad_(True)
        assert image._is_view() and weights._is_view()
        with use_backend(backend):
            img, w = insert_into_image_2d(
                vals, crd, image, weights, interpolation="bilinear"
            )
        (
            img.sum() + (w * torch.linspace(0, 1, w.numel()).reshape(w.shape)).sum()
        ).backward()
        grads[backend] = (stack.grad, vals.grad, crd.grad)
    for a, b in zip(grads["torch"], grads["mojo"], strict=True):
        _close(a, b, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# GPU kernels: same contracts as the CPU ones, reference computed on the CPU
# (torch has no 3D grid_sample backward on MPS)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_sample_matches_torch_on_device(device, ndim, dtype):
    image = _image(ndim, dtype, 2)
    coords = _coords(ndim, n=(40, 30))
    for mode in SAMPLE_MODES[ndim]:
        with use_backend("torch"):
            ref = SAMPLE[ndim](image, coords, interpolation=mode)
        with use_backend("mojo"):
            out = SAMPLE[ndim](image.to(device), coords.to(device), interpolation=mode)
        _close(ref, out.cpu())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_sample_gradients_match_torch_on_device(device, ndim, dtype):
    image = _image(ndim, dtype, 2)
    coords = _coords(ndim, n=(40, 30))
    for mode in SAMPLE_MODES[ndim]:
        grads = {}
        for backend, dev in (("torch", "cpu"), ("mojo", device)):
            img = image.detach().to(dev).requires_grad_(True)
            crd = coords.detach().to(dev).requires_grad_(True)
            with use_backend(backend):
                out = SAMPLE[ndim](img, crd, interpolation=mode)
            (
                out.real.sum() + 2 * out.imag.sum() if out.is_complex() else out.sum()
            ).backward()
            grads[backend] = (_cpu(img.grad), _cpu(crd.grad))
        _close(grads["torch"][0], grads["mojo"][0], rtol=1e-3, atol=1e-3)
        if mode != "nearest":
            _close(grads["torch"][1], grads["mojo"][1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_insert_matches_torch_on_device(device, ndim, dtype):
    coords = _coords(ndim, n=(40, 30))
    values = torch.randn((40, 30, 2), dtype=dtype)
    for mode in INSERT_MODES[ndim]:
        results = {}
        for backend, dev in (("torch", "cpu"), ("mojo", device)):
            image = torch.zeros((2, *SHAPES[ndim]), dtype=dtype, device=dev)
            weights = torch.zeros(SHAPES[ndim], device=dev)
            with use_backend(backend):
                img, w = INSERT[ndim](
                    values.to(dev), coords.to(dev), image, weights, interpolation=mode
                )
            results[backend] = (img.cpu(), w.cpu())
        _close(results["torch"][0], results["mojo"][0], rtol=1e-3, atol=1e-3)
        _close(results["torch"][1], results["mojo"][1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_insert_gradients_match_torch_on_device(device, ndim):
    coords = _coords(ndim, n=(40, 30))
    values = torch.randn((40, 30, 2))
    for mode in INSERT_MODES[ndim]:
        grads = {}
        for backend, dev in (("torch", "cpu"), ("mojo", device)):
            base = (torch.randn(2, *SHAPES[ndim]) * 0.1).to(dev).requires_grad_(True)
            image = base * 1.0
            vals = values.detach().to(dev).requires_grad_(True)
            crd = coords.detach().to(dev).requires_grad_(True)
            with use_backend(backend):
                img, w = INSERT[ndim](vals, crd, image, interpolation=mode)
            ramp = torch.linspace(0, 1, w.numel(), device=dev).reshape(w.shape)
            (img.sum() + (w * ramp).sum()).backward()
            grads[backend] = (_cpu(base.grad), _cpu(vals.grad), _cpu(crd.grad))
        _close(grads["torch"][0], grads["mojo"][0], rtol=1e-3, atol=1e-3)
        _close(grads["torch"][1], grads["mojo"][1], rtol=1e-3, atol=1e-3)
        if mode != "nearest":
            _close(grads["torch"][2], grads["mojo"][2], rtol=1e-3, atol=1e-3)
