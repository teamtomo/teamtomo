"""Shared input-contract tests for scattering propagation modes."""

from collections.abc import Callable

import pytest
import torch

from torch_scattering import firstborn, multislice, projection, rytov

PropagationMode = Callable[
    [torch.Tensor, float | torch.Tensor, float | torch.Tensor], torch.Tensor
]
MODES: tuple[PropagationMode, ...] = (multislice, rytov, firstborn, projection)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype"),
    [(torch.float32, torch.complex64), (torch.float64, torch.complex128)],
)
def test_real_potential_returns_matching_complex_precision(
    mode: PropagationMode,
    real_dtype: torch.dtype,
    complex_dtype: torch.dtype,
):
    potential = torch.zeros((2, 4, 4), dtype=real_dtype)
    wave = mode(potential, 1.0, 300.0)
    assert wave.dtype == complex_dtype
    assert torch.is_complex(wave)


@pytest.mark.parametrize("mode", MODES)
def test_complex_absorptive_potential_is_supported(mode: PropagationMode):
    potential = torch.full((2, 4, 4), 1j, dtype=torch.complex64)
    wave = mode(potential, 1.0, 300.0)
    assert torch.is_complex(wave)
    assert torch.isfinite(wave).all()
    assert torch.all(wave.abs() < 1)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "potential",
    [
        torch.zeros((4, 4)),
        torch.zeros((0, 4, 4)),
        torch.zeros((2, 0, 4)),
        torch.zeros((2, 4, 0)),
    ],
)
def test_invalid_potential_shape_is_rejected(
    mode: PropagationMode, potential: torch.Tensor
):
    with pytest.raises(ValueError):
        mode(potential, 1.0, 300.0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "potential",
    [
        torch.zeros((2, 4, 4), dtype=torch.int64),
        torch.zeros((2, 4, 4), dtype=torch.float16),
        torch.zeros((2, 4, 4), dtype=torch.bfloat16),
    ],
)
def test_unsupported_potential_dtype_is_rejected(
    mode: PropagationMode, potential: torch.Tensor
):
    with pytest.raises(TypeError):
        mode(potential, 1.0, 300.0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("pixel_size", "voltage"),
    [
        (0.0, 300.0),
        (-1.0, 300.0),
        (float("nan"), 300.0),
        (float("inf"), 300.0),
        (1.0, 0.0),
        (1.0, -300.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (torch.ones(1), 300.0),
        (1.0, torch.ones(1)),
    ],
)
def test_invalid_physical_scalars_are_rejected(
    mode: PropagationMode,
    pixel_size: float | torch.Tensor,
    voltage: float | torch.Tensor,
):
    with pytest.raises(ValueError):
        mode(torch.zeros((2, 4, 4)), pixel_size, voltage)


@pytest.mark.parametrize("mode", MODES)
def test_scalar_tensor_gradients_are_preserved(mode: PropagationMode):
    potential = torch.ones((2, 4, 4), requires_grad=True)
    pixel_size = torch.tensor(1.0, requires_grad=True)
    voltage = torch.tensor(300.0, requires_grad=True)
    wave = mode(potential, pixel_size, voltage)
    wave.real.sum().backward()
    assert potential.grad is not None
    assert pixel_size.grad is not None
    assert voltage.grad is not None
