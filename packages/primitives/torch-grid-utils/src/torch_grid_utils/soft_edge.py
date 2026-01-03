import einops
import torch
from scipy import ndimage as ndi


def _add_soft_edge_single_binary_image(
    image: torch.Tensor, smoothing_radius: float
) -> torch.FloatTensor:
    if smoothing_radius == 0:
        return image.float()
    # move explicitly to cpu for scipy
    distances = ndi.distance_transform_edt(torch.logical_not(image).to("cpu"))
    distances = torch.as_tensor(distances, device=image.device).float()
    idx = torch.logical_and(distances > 0, distances <= smoothing_radius)
    output = torch.clone(image).float()
    output[idx] = torch.cos((torch.pi / 2) * (distances[idx] / smoothing_radius))
    return output


def add_soft_edge_2d(
    image: torch.Tensor, smoothing_radius: torch.Tensor | float
) -> torch.Tensor:
    image_packed, ps = einops.pack([image], "* h w")
    b = image_packed.shape[0]

    if isinstance(smoothing_radius, float | int):
        smoothing_radius = torch.as_tensor(
            data=[smoothing_radius], device=image.device, dtype=torch.float32
        )
    smoothing_radius = torch.broadcast_to(smoothing_radius, (b,))

    results = [
        _add_soft_edge_single_binary_image(_image, smoothing_radius=_smoothing_radius)
        for _image, _smoothing_radius in zip(image_packed, smoothing_radius)
    ]
    results = torch.stack(results, dim=0)
    [results] = einops.unpack(results, pattern="* h w", packed_shapes=ps)
    return results


def add_soft_edge_3d(
    image: torch.Tensor, smoothing_radius: torch.Tensor | float
) -> torch.Tensor:
    image_packed, ps = einops.pack([image], "* d h w")
    b = image_packed.shape[0]
    if isinstance(smoothing_radius, float | int):
        smoothing_radius = torch.as_tensor(
            data=[smoothing_radius], device=image.device, dtype=torch.float32
        )
    smoothing_radius = torch.broadcast_to(smoothing_radius, (b,))
    results = [
        _add_soft_edge_single_binary_image(_image, smoothing_radius=_smoothing_radius)
        for _image, _smoothing_radius in zip(image_packed, smoothing_radius)
    ]
    results = torch.stack(results, dim=0)
    [results] = einops.unpack(results, pattern="* d h w", packed_shapes=ps)
    return results
