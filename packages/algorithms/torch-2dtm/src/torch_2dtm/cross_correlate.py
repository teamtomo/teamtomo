"""Cross-correlation functions."""

import torch
import platform
import einops
from einops._torch_specific import allow_ops_in_compiled_graph
from torch_fourier_slice import extract_central_slices_rfft_3d

from torch_2dtm.utils import normalize_template_projection

# compile normalization utility function
allow_ops_in_compiled_graph()
if platform.system() == "Linux":
    COMPILE_BACKEND = "aot_eager"  # More stable than inductor on Linux
else:
    COMPILE_BACKEND = "inductor"  # inductor for macOS

normalize_template_projection_compiled = torch.compile(
    normalize_template_projection, backend=COMPILE_BACKEND
)


def match_template_dft_2d(
    image_dft: torch.Tensor,
    template_dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    filters: torch.Tensor,
) -> torch.Tensor:
    """Batched projection and cross-correlation with a set of filters.

    Note that this function returns a cross-correlation image which is the
    same size as the input image prior to FFT calculation.

    Parameters
    ----------
    image_dft : torch.Tensor
        `(h_im, w_im // 2 + 1)` fourier transform (rfft) of the real space image.
        Any filters etc are assumed to have already been applied to this image.
    template_dft : torch.Tensor
       `(d, h, w // 2 + 1)` fftshifted fourier transform (rfft) of the real valued template volume to take Fourier
        slices from.
    rotation_matrices : torch.Tensor
        `(b, 3, 3)` batched rotation matrices to rotate slices sampled from the template fourier transform.
    filters : torch.Tensor
        `(..., h, w // 2 + 1)` filters applied to FFT slices which are fftshifted results of a rfft.

    Returns
    -------
    torch.Tensor
        Cross-correlation of the image with the template volume for each
        orientation and defocus value. Will have shape
        (orientations, defocus_batch, H, W).
    """
    # Grab relevant dimensions
    _, h, w = template_dft.shape
    h_im, w_im = image_dft.shape
    w_im = 2 * (w_im - 1)
    w = 2 * (w - 1)

    # Extract central slice(s) from the template volume
    fourier_slices = extract_central_slices_rfft_3d(
        volume_rfft=template_dft,
        image_shape=(h,) * 3,  # NOTE: requires cubic template
        rotation_matrices=rotation_matrices,
    )  # (b, h, w)
    fourier_slices = torch.fft.ifftshift(fourier_slices, dim=(-2,))
    fourier_slices[..., 0, 0] = 0 + 0j  # zero out the DC component (mean zero)
    fourier_slices *= -1  # flip contrast

    # Apply the projective filters with broadcasting
    filters = einops.rearrange(filters, "... h w -> ... 1 h w")
    fourier_slices = fourier_slices * filters  # (..., b, h, w)

    # Inverse Fourier transform into real space and normalize
    projections = torch.fft.irfftn(fourier_slices, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))
    projections = normalize_template_projection_compiled(
        projections, (h, w), (h_im, w_im)
    )

    # Padded forward Fourier transform for cross-correlation
    projections_dft = torch.fft.rfftn(projections, dim=(-2, -1), s=(h_im, w_im))

    # Zero the DC component (set mean zero)
    projections_dft[..., 0, 0] = 0 + 0j

    # Cross correlation step by element-wise multiplication
    projections_dft = image_dft * torch.conj(projections_dft)
    cross_correlation = torch.fft.irfftn(projections_dft, dim=(-2, -1))

    return cross_correlation  # (..., h_im, w_im)
