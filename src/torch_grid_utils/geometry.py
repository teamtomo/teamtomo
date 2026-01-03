import einops
import torch


def _angle_between_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a can optionally be an nD stack, return in degrees"""
    a = torch.as_tensor(a)
    b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
    a, ps = einops.pack([a], pattern='* vec')
    dot_products = einops.einsum(a, b, 'b vec, vec -> b')
    angles = torch.rad2deg(torch.arccos(dot_products))
    [angles] = einops.unpack(angles, packed_shapes=ps, pattern='*')
    return angles


def _unique_characters(n: int) -> str:
    chars = "abcdefghijklmnopqrstuvwxyz"
    return chars[:n]
