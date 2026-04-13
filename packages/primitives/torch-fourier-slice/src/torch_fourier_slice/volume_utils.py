import torch


def compute_cube_face_averages(volume: torch.Tensor, n: int = 1) -> float:
    """Get the average value of all voxels within n-voxels of the cube faces.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    n: int
        Number of voxels from the cube faces to include in the average.

    Returns
    -------
    float
        Average value of all voxels within n-voxels of the cube faces.
    """
    d, h, w = volume.shape[-3:]

    assert n >= 1, "n must be >= 1"
    assert len({d, h, w}) == 1, "all dimensions of volume must be equal."
    assert n < d // 2, "n must be less than half the volume size."

    total_sum = volume.sum()
    total_voxels = volume.numel()

    # Get the sum of the interior of the cube to avoid double counting
    interior = volume[n:-n, n:-n, n:-n]
    interior_sum = interior.sum()
    interior_voxels = interior.numel()

    face_sum = total_sum - interior_sum
    face_voxels = total_voxels - interior_voxels

    return float((face_sum / face_voxels).item())
