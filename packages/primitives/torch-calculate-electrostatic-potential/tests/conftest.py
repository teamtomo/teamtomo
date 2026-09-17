"""Shared fixtures for torch_calculate_electrostatic_potential tests."""

import pytest
import torch

from torch_calculate_electrostatic_potential.utils.peng_model import (
    get_peng_scattering_parameters,
)


@pytest.fixture
def simple_atoms():
    """A small set of 3 atoms (C, N, O) with realistic Peng parameters."""
    atom_pos_zyx = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.3, -0.7], [-1.2, 2.0, 0.5]], dtype=torch.float32
    )
    atomic_numbers = torch.tensor([6, 7, 8], dtype=torch.int64)  # C, N, O
    atom_bfactors = torch.full((3,), 20.0)
    a, b = get_peng_scattering_parameters(atomic_numbers)
    return {
        "atom_pos_zyx": atom_pos_zyx,
        "atomic_numbers": atomic_numbers,
        "atom_bfactors": atom_bfactors,
        "atom_params_a": a,
        "atom_params_b": b,
    }
