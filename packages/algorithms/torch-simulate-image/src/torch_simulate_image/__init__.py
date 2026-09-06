"""Cryo-EM micrograph simulation from exit waves in PyTorch."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-simulate-image")
except PackageNotFoundError:
    __version__ = "uninstalled"

from torch_simulate_image.config import (
    CtfConfig,
    DoseWeightConfig,
    DqeConfig,
    EnvelopeConfig,
    FluenceConfig,
    LppConfig,
    MicrographSimulationConfig,
    ObjectiveApertureConfig,
    PoissonConfig,
)
from torch_simulate_image.detector.dqe import apply_dqe
from torch_simulate_image.dose import apply_dose_weight
from torch_simulate_image.envelopes import apply_envelopes
from torch_simulate_image.fluence import scale_to_expected_counts
from torch_simulate_image.intensity import exit_wave_to_intensity
from torch_simulate_image.noise.poisson import poisson_sample
from torch_simulate_image.optics.aperture import apply_objective_aperture
from torch_simulate_image.optics.ctf import apply_ctf_to_exit_wave
from torch_simulate_image.pipeline import (
    simulate_micrograph,
    simulate_micrograph_from_intensity,
)

__all__ = [
    "CtfConfig",
    "DoseWeightConfig",
    "DqeConfig",
    "EnvelopeConfig",
    "FluenceConfig",
    "LppConfig",
    "MicrographSimulationConfig",
    "ObjectiveApertureConfig",
    "PoissonConfig",
    "__version__",
    "apply_ctf_to_exit_wave",
    "apply_dose_weight",
    "apply_dqe",
    "apply_envelopes",
    "apply_objective_aperture",
    "exit_wave_to_intensity",
    "poisson_sample",
    "scale_to_expected_counts",
    "simulate_micrograph",
    "simulate_micrograph_from_intensity",
]
