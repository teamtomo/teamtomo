"""Utilities for retrieving pretrained checkpoints from Zenodo."""

from __future__ import annotations

import shutil
from pathlib import Path

import pooch

from torch_segment_tomogram_boundaries import config

_DEFAULT_REMOTE_NAME = "tomo_slab.ckpt"
_REGISTRY = {
    _DEFAULT_REMOTE_NAME: "d3bf10379a4ae804384d35aa427cfd124082bf17f9271267da725e766d69d2e9"
}
_BASE_URL = "doi:10.5281/zenodo.17344782"


def _build_fetcher(cache_dir: Path) -> pooch.Pooch:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return pooch.create(path=str(cache_dir), base_url=_BASE_URL, registry=_REGISTRY)


def get_latest_checkpoint(
    cache_dir: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Download the latest checkpoint and return its local path.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory to store the downloaded checkpoint. Defaults to
        `config.CKPT_SAVE_PATH` when omitted.
    filename : str, optional
        Desired filename for the checkpoint. When omitted, the remote filename
        (`tomo_slab.ckpt`) is used. The file is copied into ``cache_dir`` with this
        name if it differs from the remote name.
    """
    target_dir = Path(cache_dir) if cache_dir else config.CKPT_SAVE_PATH
    fetcher = _build_fetcher(target_dir)
    canonical_path = Path(fetcher.fetch(_DEFAULT_REMOTE_NAME))

    if filename and filename != _DEFAULT_REMOTE_NAME:
        destination = target_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != canonical_path.resolve():
            shutil.copy2(canonical_path, destination)
        return destination

    return canonical_path
