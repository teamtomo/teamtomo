# tests/conftest.py
import warnings

import mrcfile
import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from torch_segment_tomogram_boundaries import constants
from torch_segment_tomogram_boundaries.trainer import TomoSlabTrainer


@pytest.fixture(scope="session")
def data_dirs(tmp_path_factory):
    """Creates a temporary directory structure mimicking the real data layout."""
    base_dir = tmp_path_factory.mktemp("data")
    paths = {
        "vol_dir": base_dir / "data_in" / "volumes",
        "mask_dir": base_dir / "data_in" / "masks",
        "train_dir": base_dir / "prepared_data" / "train",
        "val_dir": base_dir / "prepared_data" / "val",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


@pytest.fixture(scope="session")
def dummy_mrc_files(data_dirs):
    """Creates a pair of dummy volume and mask .mrc files."""
    shape = (32, 64, 64)
    # Create a simple volume (a ramp)
    vol_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    # Create a simple mask (a central cube)
    mask_data = np.zeros(shape, dtype=np.int8)
    mask_data[8:24, 16:48, 16:48] = 1

    vol_path = data_dirs["vol_dir"] / "dummy_tomo.mrc"
    mask_path = data_dirs["mask_dir"] / "dummy_tomo.mrc"

    with mrcfile.new(vol_path, overwrite=True) as mrc:
        mrc.set_data(vol_data)
    with mrcfile.new(mask_path, overwrite=True) as mrc:
        mrc.set_data(mask_data)

    return {"vol_path": vol_path, "mask_path": mask_path}


@pytest.fixture(scope="session")
def dummy_pt_files(data_dirs):
    """Creates dummy .pt files for training and validation."""
    # 1 training file
    data = {"image": torch.randn(2, 64, 64), "label": torch.randint(0, 2, (1, 64, 64))}
    torch.save(data, data_dirs["train_dir"] / "train_sample_0.pt")

    # 1 validation file
    data = {"image": torch.randn(2, 64, 64), "label": torch.randint(0, 2, (1, 64, 64))}
    torch.save(data, data_dirs["val_dir"] / "val_sample_0.pt")
    return data_dirs


@pytest.fixture(scope="session")
def trained_checkpoint(dummy_pt_files, tmp_path_factory):
    """
    Creates a real, lightweight model checkpoint by running training for one batch.
    This is the most reliable way to test model loading.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Override constants to point to our temporary data
        constants.TRAIN_DATA_DIR = dummy_pt_files["train_dir"]
        constants.VAL_DATA_DIR = dummy_pt_files["val_dir"]

        trainer_api = TomoSlabTrainer(
            train_data_dir=dummy_pt_files["train_dir"],
            val_data_dir=dummy_pt_files["val_dir"],
        )

        # Use Lightning's fast_dev_run to run one training and validation batch
        pl_trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
        )
        pl_trainer.fit(
            trainer_api._setup_model(), datamodule=trainer_api._setup_datamodule()
        )

        checkpoint_path = tmp_path_factory.mktemp("checkpoints") / "test_model.ckpt"
        pl_trainer.save_checkpoint(checkpoint_path)

        return checkpoint_path
