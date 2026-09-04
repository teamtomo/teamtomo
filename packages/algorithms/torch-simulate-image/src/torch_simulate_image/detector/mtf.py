"""MTF tensor resolution for detector modelling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch_fourier_filter.mtf import read_mtf

if TYPE_CHECKING:
    import torch

    from torch_simulate_image.config import DqeConfig


def resolve_mtf_tensors(config: DqeConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve MTF frequency and amplitude tensors from config.

    Parameters
    ----------
    config : DqeConfig
        DQE configuration with either STAR file path or explicit tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(mtf_frequencies, mtf_amplitudes)`` as 1D float tensors.
    """
    if config.mtf_frequencies is not None and config.mtf_amplitudes is not None:
        return config.mtf_frequencies, config.mtf_amplitudes
    if config.starfile_path is not None:
        return read_mtf(config.starfile_path)
    msg = "DQE config must provide MTF frequencies and amplitudes."
    raise ValueError(msg)
