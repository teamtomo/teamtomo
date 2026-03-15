"""Common utility functions for the project."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def get_device(verbose: bool = False) -> torch.device:
    """
    Get the appropriate PyTorch device for computation.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU device.
    """
    if torch.cuda.is_available():
        if verbose:
            log.info("CUDA is available. Using GPU.")
        return torch.device("cuda")

    if verbose:
        log.info("CUDA not available. Using CPU.")
    return torch.device("cpu")


def save_config_snapshot(
    destination_dir: Path,
    *,
    config_module_path: Path | None = None,
    timestamp: bool = True,
) -> Path:
    """Persist the current training config alongside checkpoints.

    Parameters
    ----------
    destination_dir : Path
        Directory where the config copy should be saved.
    config_module_path : Path, optional
        Explicit path to the config module. Defaults to the packaged config.
    timestamp : bool, optional
        When ``True`` (default) append a timestamp to avoid overwriting
        previous snapshots within the same destination.

    Returns
    -------
    Path
        The path to the saved config snapshot.
    """
    if config_module_path is None:
        config_module_path = Path(__file__).resolve().parents[1] / "config.py"

    if not config_module_path.exists():
        raise FileNotFoundError(f"Config module not found at {config_module_path}.")

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    suffix = datetime.now().strftime("%Y%m%d-%H%M%S") if timestamp else "current"
    snapshot_path = destination_dir / f"config_snapshot_{suffix}.py"

    shutil.copy2(config_module_path, snapshot_path)
    log.info("Saved config snapshot to %s", snapshot_path)

    return snapshot_path
