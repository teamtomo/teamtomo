# tests/test_trainer.py
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

from torch_segment_tomogram_boundaries import config
from torch_segment_tomogram_boundaries.trainer import TomoSlabTrainer


def test_trainer_initialization():
    trainer = TomoSlabTrainer()
    assert trainer.learning_rate == config.LEARNING_RATE
    assert trainer.loss_config == config.LOSS_CONFIG


def test_setup_callbacks_uses_standard_components():
    trainer = TomoSlabTrainer()
    callbacks = trainer._setup_callbacks()

    callback_types = {type(cb) for cb in callbacks}

    assert ModelCheckpoint in callback_types
    assert LearningRateMonitor in callback_types
    assert TQDMProgressBar in callback_types
    assert EarlyStopping in callback_types

    assert all(cb.__class__.__name__ != "DynamicTrainingManager" for cb in callbacks)
