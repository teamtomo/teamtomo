"""Resolve Gaussian electron-scattering-factor parameters.

The elemental table contains the Peng et al. (1996) elastic electron scattering
factor coefficients valid for ``s <= 6 Angstrom^-1``:

    f_e(s) = sum_i a_i * exp(-b_i * s^2),  s = sin(theta) / wavelength

Here ``f_e`` and ``a_i`` have units Angstrom and ``b_i`` has units Angstrom
squared. These are not X-ray form-factor coefficients. The empirical bonded
tables use the equivalent convention ``g = 2s`` and
``f_e(g) = sum_i a_i * exp(-b_i * g^2 / 4)``.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal, cast

import numpy as np
import torch

# Peng elemental tables and the current potential kernel both use five Gaussians.
PENG_GAUSSIAN_TERM_COUNT = 5


@dataclass(frozen=True, slots=True)
class BondedScatteringFactorTable:
    """Gaussian scattering factors keyed by bonded-environment identifier.

    Every environment must provide the same number of ``(a_i, b_i)`` Gaussian
    terms in ``parameters_a`` and ``parameters_b``. Use :attr:`n_gaussian_terms`
    to query that count.
    """

    parameters_a: Mapping[str, Sequence[float]]
    parameters_b: Mapping[str, Sequence[float]]

    def __post_init__(self) -> None:
        """Validate bonded-environment parameter tables."""
        if not self.parameters_a:
            raise ValueError("parameters_a must not be empty")
        if set(self.parameters_a) != set(self.parameters_b):
            raise ValueError(
                "parameters_a and parameters_b must have the same environment keys"
            )
        term_counts: set[int] = set()
        for environment in self.parameters_a:
            coeffs_a = self.parameters_a[environment]
            coeffs_b = self.parameters_b[environment]
            if len(coeffs_a) != len(coeffs_b):
                raise ValueError(
                    f"environment {environment!r}: parameters_a and parameters_b "
                    "must have the same length"
                )
            if len(coeffs_a) == 0:
                raise ValueError(
                    f"environment {environment!r}: at least one Gaussian term "
                    "is required"
                )
            term_counts.add(len(coeffs_a))
        if len(term_counts) != 1:
            raise ValueError(
                "all environments in a BondedScatteringFactorTable must use the same "
                f"number of Gaussian terms, got {sorted(term_counts)}"
            )

    @property
    def n_gaussian_terms(self) -> int:
        """Number of Gaussian terms in every bonded-environment sequence."""
        return len(next(iter(self.parameters_a.values())))


BondedScatteringFactorProviders = Mapping[str, BondedScatteringFactorTable]
ScatteringFactors = (
    Literal["peng_elemental", "peng_bonded"] | BondedScatteringFactorProviders
)
BondedFallback = Literal["elemental", "error"]


def _load_peng_element_scattering_factor_parameter_table() -> np.ndarray:
    resource = files(__package__).joinpath("peng1996_element_params.npy")
    with resource.open("rb") as stream:
        return cast("np.ndarray", np.load(stream))


def get_peng_scattering_parameters(
    atomic_numbers: torch.Tensor,
    device: torch.device | str | int | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up elemental Peng 1996 electron scattering factors.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Integer atomic numbers, any shape.
    device : torch.device, optional
        Device for returned tensors. Defaults to the atomic-number tensor's device.
    dtype : torch.dtype, optional
        Floating-point type for returned tensors.

    Returns
    -------
    a, b : torch.Tensor
        Gaussian parameters with shape ``(*atomic_numbers.shape, 5)``.
        Amplitudes ``a`` are in Angstroms and widths ``b`` in Angstroms squared.
    """
    resolved_device = atomic_numbers.device if device is None else device
    table = torch.from_numpy(_load_peng_element_scattering_factor_parameter_table()).to(
        device=resolved_device, dtype=dtype
    )
    indices = atomic_numbers.to(device=resolved_device, dtype=torch.int64)
    a, b = table[:, indices]
    return a, b


def resolve_scattering_parameters(
    atomic_numbers: torch.Tensor,
    *,
    scattering_factors: ScatteringFactors = "peng_elemental",
    bonded_environments: tuple[str, ...] | None = None,
    molecule_types: tuple[str, ...] | None = None,
    bonded_fallback: BondedFallback = "elemental",
    device: torch.device | str | int | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve an explicitly selected Peng parameter model.

    ``peng_elemental`` selects Peng et al. (1996) neutral-atom electron
    scattering factors and ignores bonding metadata. ``peng_bonded`` selects
    the bundled empirical factors of Shtyrov et al. (2026). A mapping from
    molecule-type keys to :class:`BondedScatteringFactorTable` may instead be
    supplied to select custom factors. Bonded models use the per-atom
    ``molecule_types`` and ``bonded_environments`` metadata.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        One-dimensional tensor of atomic numbers for bonded models, or any shape
        for ``peng_elemental``. Bonded lookup iterates over flat
        ``bonded_environments`` / ``molecule_types`` tuples and does not support
        batched ``(batch, n_atoms)`` atomic-number tensors.
    scattering_factors : {"peng_elemental", "peng_bonded"} or mapping
        Bundled parameter model, or custom tables keyed by molecule type such
        as ``"protein"`` and ``"rna"``.
    bonded_environments : tuple[str, ...], optional
        Canonical keys such as ``"C(HHCC)"`` produced by
        ``annotate_bonding_environments``.
    molecule_types : tuple[str, ...], optional
        Per-atom provider names: ``"protein"`` or ``"rna"``.
    bonded_fallback : {"elemental", "error"}
        Use elemental parameters with one aggregate warning, or reject unsupported
        providers and keys.
    device : torch.device, optional
        Device for returned tensors.
    dtype : torch.dtype, optional
        Floating-point type for returned tensors.

    Returns
    -------
    a, b : torch.Tensor
        Selected parameters, each with shape ``(n_atoms, 5)``.
    """
    providers: BondedScatteringFactorProviders
    if isinstance(scattering_factors, str):
        if scattering_factors == "peng_elemental":
            return get_peng_scattering_parameters(atomic_numbers, device, dtype)
        if scattering_factors != "peng_bonded":
            raise ValueError(
                f"unknown scattering_factors model: {scattering_factors!r}"
            )
        providers = _load_bonded_providers()
    else:
        providers = scattering_factors
    if atomic_numbers.ndim != 1:
        raise ValueError("bonded factors require one-dimensional atomic_numbers")
    if bonded_environments is None or molecule_types is None:
        raise ValueError(
            "bonded scattering-factor selection requires bonded_environments "
            "and molecule_types"
        )
    n_atoms = atomic_numbers.numel()
    if len(bonded_environments) != n_atoms or len(molecule_types) != n_atoms:
        raise ValueError("bonding metadata must contain one value per atom")
    if bonded_fallback not in ("elemental", "error"):
        raise ValueError(f"unknown bonded_fallback mode: {bonded_fallback!r}")

    a, b = get_peng_scattering_parameters(atomic_numbers, device, dtype)
    unsupported: list[str] = []
    for index, (environment, molecule_type) in enumerate(
        zip(bonded_environments, molecule_types, strict=True)
    ):
        provider = providers.get(molecule_type.strip().lower())
        if provider is None:
            unsupported.append(f"atom {index}: molecule type {molecule_type!r}")
            continue
        if (
            environment not in provider.parameters_a
            or environment not in provider.parameters_b
        ):
            unsupported.append(
                f"atom {index}: {molecule_type.strip().lower()} key {environment!r}"
            )
            continue
        if provider.n_gaussian_terms != PENG_GAUSSIAN_TERM_COUNT:
            raise ValueError(
                f"provider {molecule_type.strip().lower()!r} defines "
                f"{provider.n_gaussian_terms} Gaussian terms, but the potential "
                f"kernel currently requires {PENG_GAUSSIAN_TERM_COUNT}"
            )
        a[index] = torch.as_tensor(
            provider.parameters_a[environment], device=a.device, dtype=a.dtype
        )
        b[index] = torch.as_tensor(
            provider.parameters_b[environment], device=b.device, dtype=b.dtype
        )

    if unsupported:
        detail = "; ".join(unsupported[:3])
        if len(unsupported) > 3:
            detail += f"; and {len(unsupported) - 3} more"
        message = f"unsupported bonded scattering parameters ({detail})"
        if bonded_fallback == "error":
            raise ValueError(message)
        warnings.warn(f"{message}; using elemental fallback", UserWarning, stacklevel=2)
    return a, b


def _load_bonded_providers() -> dict[str, BondedScatteringFactorTable]:
    providers: dict[str, BondedScatteringFactorTable] = {}
    for molecule_type in ("protein", "rna"):
        resource = files(__package__).joinpath(
            f"elastic_scattering_bonding_{molecule_type}.json"
        )
        data: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
        providers[molecule_type] = BondedScatteringFactorTable(
            parameters_a=data["parameters_a"],
            parameters_b=data["parameters_b"],
        )
    return providers
