import os
import shutil
import subprocess
from pathlib import Path


def download_training_data(output_directory: Path):
    """Download training dataset from Zenodo."""
    subprocess.run(
        [
            "zenodo_get",
            "15830961",
            "--output-dir",
            str(output_directory),
        ]
    )
    zipped_archive = output_directory / "fiducial_segmentation_images.zip"
    shutil.unpack_archive(
        zipped_archive,
        extract_dir=output_directory,
    )
    os.remove(zipped_archive)
