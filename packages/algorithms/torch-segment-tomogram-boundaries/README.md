# torch-segment-tomogram-boundaries

[![License](https://img.shields.io/pypi/l/torch-tomo-slab.svg?color=green)](https://github.com/shahpnmlab/torch-tomo-slab/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-tomo-slab.svg?color=green)](https://pypi.org/project/torch-tomo-slab)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-tomo-slab.svg?color=green)](https://python.org)
[![CI](https://github.com/shahpnmlab/torch-tomo-slab/actions/workflows/ci.yml/badge.svg)](https://github.com/shahpnmlab/torch-tomo-slab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shahpnmlab/torch-tomo-slab/branch/main/graph/badge.svg)](https://codecov.io/gh/shahpnmlab/torch-tomo-slab)

A simple Unet application to detect boundaries of tomographic volumes

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork teamtomo/torch-segment-tomogram-boundaries --clone
# or just
# gh repo clone teamtomo/torch-segment-tomogram-boundaries
cd torch-tomo-slab
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```

## API Usage

The library is designed around three core classes: `TrainingDataGenerator`, `TomoSlabTrainer`, and `TomoSlabPredictor`. The following demonstrates the standard workflow.

### Step 1: Prepare Training Data

First, you need to convert your 3D tomogram volumes and corresponding label masks into 2D slices suitable for training.

-   Place your tomograms (`.mrc` files) in the directory specified by `constants.REFERENCE_TOMOGRAM_DIR`.
-   Place your masks (`.mrc` files) in the directory specified by `constants.MASK_OUTPUT_DIR`.

Then, run the `TrainingDataGenerator`.

```python
from torch_segment_tomogram_boundaries.processing import TrainingDataGenerator
from torch_segment_tomogram_boundaries import config

# This class reads its path configuration from the `config.py` file.
# Ensure the paths in that file are correct for your system.
print(f"Reading tomograms from: {config.TOMOGRAM_DIR}")
print(f"Reading masks from: {config.MASKS_DIR}")

# Initialize the generator
generator = TrainingDataGenerator()

# Run the data preparation pipeline
# This will process all volumes and save 2D .pt files to the
# directories specified by constants.TRAIN_DATA_DIR and constants.VAL_DATA_DIR.
generator.run()

print("Data preparation complete.")
```

### Step 2: Train a Segmentation Model

Once the data is prepared, you can train the model using the `TomoSlabTrainer`.

```python
from torch_segment_tomogram_boundaries.trainer import train
from torch_segment_tomogram_boundaries import config

# The trainer automatically loads its configuration from `config.py`.
# You can modify that file directly or override the settings (see note below).

# Start the training process
# The function will handle dataloading, model setup, and the training loop.
train(
    train_data_dir=config.TRAIN_DATA_DIR,
    val_data_dir=config.VAL_DATA_DIR,
    # Optional: override other settings
    # learning_rate=1e-4,
    # max_epochs=100,
)

print("Training finished.")
```

### Step 3: Predict on a New Tomogram

After training, use the best model checkpoint to predict a slab mask for a new, unseen tomogram.

```python
from pathlib import Path
import mrcfile
from torch_segment_tomogram_boundaries.predict import predict_binary
from torch_segment_tomogram_boundaries.fetch import get_latest_checkpoint

# 1. Specify the path to your trained model checkpoint.
# You can use a local checkpoint or fetch the latest pretrained one.
# checkpoint_path = "path/to/your/local/model.ckpt"
checkpoint_path = get_latest_checkpoint()

# 2. Specify the path to the input tomogram you want to predict
input_tomo_path = Path("/path/to/your/new_tomogram.mrc")

# 3. Define where to save the output mask
output_mask_path = Path("/path/to/your/output/predicted_mask.mrc")

# 4. Run prediction to get a binary mask
# This loads the tomogram, runs inference, and returns a binary mask.
binary_mask = predict_binary(
    input_tomogram=input_tomo_path,
    model_checkpoint_path=checkpoint_path,
    binarize_threshold=0.5,
    # Other optional args: slab_size, batch_size, smoothing_sigma
)

# 5. Save the final mask
with mrcfile.new(output_mask_path, overwrite=True) as mrc:
    mrc.set_data(binary_mask)

print(f"Prediction complete. Mask saved to {output_mask_path}")

```

### A Note on Overriding Default Configurations

The default hyperparameters (e.g., learning rate, batch size, model architecture) and file paths are stored in `src/torch_segment_tomogram_boundaries/config.py` and `src/torch_segment_tomogram_boundaries/constants.py`.

For quick experiments or when using this library as part of a larger application, it is best to **programmatically override** these settings in your script *before* you create instances of the library's classes.

Here is an example of a wrapper function that demonstrates this best practice:

```python
from torch_segment_tomogram_boundaries import config
from torch_segment_tomogram_boundaries.trainer import train
from pathlib import Path


def def run_custom_experiment(lr, batch_size, data_dir):
    """
    A wrapper to run training with custom, non-default hyperparameters.
    """
    print("--- Applying custom configuration ---")

    # Overwrite config values before calling the train function
    # This is for parameters not exposed in the `train` function signature.
    config.BATCH_SIZE = batch_size
    # For example, to change the model:
    # config.MODEL_CONFIG['channels'] = (64, 128, 256, 512)

    # Determine data paths
    prepared_data_path = Path(data_dir)
    train_path = prepared_data_path / "train"
    val_path = prepared_data_path / "val"

    print(f"Set learning rate -> {lr}")
    print(f"Set batch size -> {config.BATCH_SIZE}")
    print(f"Set training data path -> {train_path}")

    # Call the train function with arguments for exposed parameters.
    # The function will use the modified `config` for other settings.
    train(
        learning_rate=lr,
        train_data_dir=train_path,
        val_data_dir=val_path,
    )

# Example usage:
# run_custom_experiment(
#     lr=1e-5,
#     batch_size=16,
#     data_dir='/path/to/another/dataset'
# )
```
