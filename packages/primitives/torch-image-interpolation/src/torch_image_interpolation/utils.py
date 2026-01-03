import einops
import torch


def view_as_complex(tensor: torch.Tensor) -> torch.Tensor:
    """Workaround for an einops + torch.view_as_complex() issue

    c.f. https://github.com/arogozhnikov/einops/issues/370

    # works
    a = torch.rand(336, 1, 2)
    print(a.shape, ": ", a.stride(0), a.stride(1), a.stride(2))  # torch.Size([336, 1, 2]) :  2 2 1
    b = torch.view_as_complex(a)

    # errors
    a = torch.rand(size=(336, 2))
    a = einops.rearrange(a, 'b (complex c) -> b c complex', complex=2)
    print(a.shape, ": ", a.stride(0), a.stride(1), a.stride(2))  # torch.Size([336, 1, 2]) :  2 1 1
    a = a.contiguous()
    print(a.shape, ": ", a.stride(0), a.stride(1), a.stride(2))  # torch.Size([336, 1, 2]) :  2 1 1
    b = torch.view_as_complex(a)
    """
    if tensor.shape[-2] == 1:
        tensor = einops.rearrange(tensor, '... 1 complex -> ... complex')
        tensor = torch.view_as_complex(tensor)
        tensor = einops.rearrange(tensor, '... -> ... 1')
    else:
        tensor = torch.view_as_complex(tensor)
    return tensor


