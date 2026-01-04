"""Example of reconstruction with SHREC'21 data."""

import time
from pathlib import Path

import einops
import mrcfile
import torch
from torch_fourier_rescale import fourier_rescale_2d

from torch_tomogram import Tomogram


def _read_shrec_alignment(file: Path) -> tuple[list]:
    with open(model_folder / "alignment_simulated.txt") as aln:
        lines = [x.split() for x in aln.readlines() if not x.startswith("#")]
        data = zip(*[(float(x[0]), float(x[1]), float(x[2])) for x in lines])
    return data


if __name__ == "__main__":
    model_folder = Path(  # downloaded from DataverseNL
        "/home/marten/data/datasets/shrec21_full_dataset_no_mirroring/model_0"
    )
    with mrcfile.open(
        model_folder / "projections_unbinned.mrc", permissive=True
    ) as mrc:
        tilt_series = torch.tensor(mrc.data)
    tilt_series, _ = fourier_rescale_2d(tilt_series, 5.0, 10.0)
    tilt_series -= einops.reduce(tilt_series, "tilt h w -> tilt 1 1", reduction="mean")
    tilt_series /= torch.std(tilt_series, dim=(-2, -1), keepdim=True)
    x_shift, y_shift, tilt_angles = _read_shrec_alignment(
        model_folder / "alignment_simulated.txt"
    )
    # invert to match reconstruction with SHREC grand_model
    tilt_angles = list(reversed(tilt_angles))
    # divide by two for 2x downsampling of tilt-series
    shifts = torch.tensor([y_shift, x_shift]).T / 2
    tilt_axis_angle = 0.0
    n_tilts = len(tilt_angles)

    start = time.time()
    tomogram = Tomogram(
        images=tilt_series,
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=shifts * -1,
        device="cuda",
    )  # invert the shifts because we employ a forward projection model!

    # 180 is the box size of the grand model
    volume = tomogram.reconstruct_tomogram((180, 512, 512), 128)
    print(time.time() - start)

    with mrcfile.open(model_folder / "grandmodel.mrc", permissive=True) as mrc:
        ground_truth = mrc.data

    # viewer = napari.Viewer()
    # viewer.add_image(ground_truth, name="ground_truth")
    # viewer.add_image(volume.cpu().numpy(), name="reconstruction")
    # napari.run()
