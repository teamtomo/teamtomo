import torch
import torch.nn.functional as F
from torch_grid_utils import fftfreq_grid

from .slice_extraction import (
    extract_central_slices_rfft_2d,
    extract_central_slices_rfft_3d,
    extract_central_slices_rfft_3d_multichannel,
    transform_slice_2d,
    transform_slice_2d_multichannel,
)
from .volume_utils import compute_cube_face_averages


def project_3d_to_2d(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    transform_matrix: torch.Tensor | None = None,
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> torch.Tensor:
    """Project a cubic volume by sampling a central slice through its DFT.

    Parameters
    ----------
    volume: torch.Tensor
        `(d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of rotation matrices for extraction of `images`.
        Rotation matrices left-multiply column vectors containing xyz coordinates.
    pad_factor: float
        Factor determining how much padding to apply to each side of the volume.
        Using `pad_factor=2.0` means going from `(d, d, d)` to `(2d, 2d, 2d)` and
        and likewise `pad_factor=3.0` means going to `(3d, 3d, 3d)`, etc.
        The default value of 2.0 should suffice in most cases. See issue #24
        for more info.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection. If None,
        no masking is applied to limit the frequency content.
    zyx_matrices: bool
        Set to True if the provided matrices left multiply zyx column vectors rather
        than the default xyz column vectors.
    transform_matrix: torch.Tensor | None
        `(2, 2)` anisotropic magnification matrix in real space (yx ordering).
        If provided, applies the transformation in Fourier space to the extracted
        slices. The transformation is applied using {A^-1}T and includes proper
        scaling by 1/|det(A)| to preserve intensity.
    apply_ewald_curvature: bool
        If True, bend the central slice onto an Ewald sphere. If False (default),
        use a flat central slice.
    ewald_voltage_kv: float
        Acceleration voltage in kV. Default is 300.0 kV. Wavelength is computed
        from this using relativistic electron wavelength formula.
    ewald_flip_sign: bool
        If True, flip the sign of the Ewald curvature (apply the curve in the
        opposite direction).
    ewald_px_size: float
        Pixel size (e.g. Å / pixel). Used to convert between grid units
        (cycles / pixel) and physical spatial frequencies.

    Returns
    -------
    projections: torch.Tensor
        `(..., d, d)` array of projected images.
    """
    d, h, w = volume.shape[-3:]
    if len({d, h, w}) != 1:  # use set to remove duplicates
        raise ValueError("all dimensions of volume must be equal.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")

    # Apply edge padding to the nearest integer matching the desired pad_factor
    if pad_factor > 1.0:
        p = int((w * (pad_factor - 1.0)) // 2)
        edge_value = compute_cube_face_averages(volume, n=4)  # 4 is arbitrary
        volume = F.pad(volume, pad=[p] * 6, mode="constant", value=edge_value)

    # Track volume shape and mean as variables
    volume_shape = tuple(volume.shape[-3:])
    volume_mean = volume.mean()

    # calculate DFT
    # volume center to array origin
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))

    dft[..., 0, 0, 0] = 0.0  # Zero out mean to avoid low-res artifacts

    # fftshift the transformed volume so DC is at center
    dft = torch.fft.fftshift(dft, dim=(-3, -2))

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
        apply_ewald_curvature=apply_ewald_curvature,
        ewald_voltage_kv=ewald_voltage_kv,
        ewald_flip_sign=ewald_flip_sign,
        ewald_px_size=ewald_px_size,
    )  # (..., h, w) rfft stack

    # apply anisotropic magnification transformation if provided
    if transform_matrix is not None:
        stack_shape = tuple(rotation_matrices.shape[:-2])
        # Calculate rfft shape from volume shape
        # For rfft, width is n//2 + 1
        rfft_shape = (volume_shape[1], volume_shape[2] // 2 + 1)
        projections = transform_slice_2d(
            projection_image_dfts=projections,
            rfft_shape=rfft_shape,
            stack_shape=stack_shape,
            transform_matrix=transform_matrix,
        )

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(
        projections, dim=(-2, -1)
    )  # recenter 2D image in real space

    # unpad
    if pad_factor > 1.0:
        projections = F.pad(projections, pad=[-p] * 4)

    # Account for the subtracted off mean value for the DFT
    projections += volume_mean * d

    return projections


def project_3d_to_2d_multichannel(
    volume: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    transform_matrix: torch.Tensor | None = None,
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,
    ewald_flip_sign: bool = False,
    ewald_px_size: float = 1.0,
) -> torch.Tensor:
    """Project a multichannel cubic volume with the same rotations.

    Parameters
    ----------
    volume: torch.Tensor
        `(c, d, d, d)` volume.
    rotation_matrices: torch.Tensor
        `(..., 3, 3)` array of rotation matrices for extraction of `images`.
        Rotation matrices left-multiply column vectors containing xyz coordinates.
    pad_factor: float
        Factor determining the size after padding relative to the original size.
        A pad_factor of 2.0 doubles the box size, 3.0 triples it, etc.
        The default value of 2.0 should suffice in most cases. See issue #24
        for more info.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    zyx_matrices: bool
        Set to True if the provided matrices left multiply zyx column vectors
        instead of xyz column vectors.
    transform_matrix: torch.Tensor | None
        `(2, 2)` anisotropic magnification matrix in real space (yx ordering).
        If provided, applies the transformation in Fourier space to the extracted
        slices. The transformation is applied using {A^-1}T and includes proper
        scaling by 1/|det(A)| to preserve intensity.
    apply_ewald_curvature: bool
        If True, bend the central slice onto an Ewald sphere. If False (default),
        use a flat central slice.
    ewald_voltage_kv: float
        Acceleration voltage in kV. Default is 300.0 kV. Wavelength is computed
        from this using relativistic electron wavelength formula.
    ewald_flip_sign: bool
        If True, flip the sign of the Ewald curvature (apply the curve in the
        opposite direction).
    ewald_px_size: float
        Pixel size (e.g. Å / pixel). Used to convert between grid units
        (cycles / pixel) and physical spatial frequencies.

    Returns
    -------
    projections: torch.Tensor
        `(..., c, d, d)` tensor of multichannel projection images.
    """
    d, h, w = volume.shape[-3:]
    if len({d, h, w}) != 1:  # use set to remove duplicates
        raise ValueError("all dimensions of volume must be equal.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor > 1.0:
        p = int((volume.shape[-1] * (pad_factor - 1.0)) // 2)
        volume = F.pad(volume, pad=[p] * 6)

    # set the shape as a variable
    volume_shape = tuple(volume.shape[-3:])

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=volume_shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=volume.device,
    )
    volume = volume * torch.sinc(grid) ** 2

    # calculate DFT
    # volume center to array origin
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    # actual fftshift of 3D rfft
    dft = torch.fft.fftshift(dft, dim=(-3, -2))

    # make projections by taking central slices
    projections = extract_central_slices_rfft_3d_multichannel(
        volume_rfft=dft,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
        apply_ewald_curvature=apply_ewald_curvature,
        ewald_voltage_kv=ewald_voltage_kv,
        ewald_flip_sign=ewald_flip_sign,
        ewald_px_size=ewald_px_size,
    )  # (..., c, h, w) rfft stack

    # apply anisotropic magnification transformation if provided
    if transform_matrix is not None:
        channels = volume.shape[0]
        stack_shape = tuple(rotation_matrices.shape[:-2])
        # Calculate rfft shape from volume shape
        # For rfft, width is n//2 + 1
        rfft_shape = (volume_shape[1], volume_shape[2] // 2 + 1)
        projections = transform_slice_2d_multichannel(
            projection_image_dfts=projections,
            rfft_shape=rfft_shape,
            stack_shape=stack_shape,
            channels=channels,
            transform_matrix=transform_matrix,
        )

    # transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))  # ifftshift of 2D rfft
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(
        projections, dim=(-2, -1)
    )  # recenter 2D image in real space

    # unpad
    if pad_factor > 1.0:
        projections = F.pad(projections, pad=[-p] * 4)
    return projections


def project_2d_to_1d(
    image: torch.Tensor,
    rotation_matrices: torch.Tensor,
    pad_factor: float = 2.0,
    fftfreq_max: float | None = None,
    yx_matrices: bool = False,
) -> torch.Tensor:
    """Project a square image by sampling a central line through its DFT.

    Parameters
    ----------
    image: torch.Tensor
        `(d, d)` image.
    rotation_matrices: torch.Tensor
        `(..., 2, 2)` array of rotation matrices for extraction of `lines`.
        Rotation matrices left-multiply column vectors containing xy coordinates.
    pad_factor: float
        Factor determining the size after padding relative to the original size.
        A pad_factor of 2.0 doubles the box size, 3.0 triples it, etc.
        The default value of 2.0 should suffice in most cases. See issue #24
        for more info.
    fftfreq_max: float | None
        Maximum frequency (cycles per pixel) included in the projection.
    yx_matrices: bool
        Set to True if the provided matrices left multiply yx column vectors
        instead of xy column vectors.

    Returns
    -------
    projections: torch.Tensor
        `(..., d)` array of projected lines.
    """
    h, w = image.shape[-2:]
    if h != w:
        raise ValueError("all dimensions of image must be equal.")

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1.0")
    if pad_factor > 1.0:
        p = int((image.shape[-1] * (pad_factor - 1.0)) // 2)
        image = F.pad(image, pad=[p] * 4)

    # set the shape as a variable
    image_shape = tuple(image.shape[-2:])

    # premultiply by sinc2
    grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=False,
        fftshift=True,
        norm=True,
        device=image.device,
    )
    image = image * torch.sinc(grid) ** 2

    # calculate DFT
    dft = torch.fft.fftshift(image, dim=(-2, -1))  # image center to array origin
    dft = torch.fft.rfftn(dft, dim=(-2, -1))
    dft = torch.fft.fftshift(dft, dim=(-2,))  # actual fftshift of 2D rfft

    # make projections by taking central slices
    projections = extract_central_slices_rfft_2d(
        image_rfft=dft,
        image_shape=image_shape,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        yx_matrices=yx_matrices,
    )  # (..., w) rfft stack

    # transform back to real space
    # not needed for 1D: torch.fft.ifftshift(projections, dim=(-2,))
    projections = torch.fft.irfftn(projections, dim=(-1))
    projections = torch.fft.ifftshift(
        projections, dim=(-1)
    )  # recenter line in real space

    # unpad
    if pad_factor > 1.0:
        projections = F.pad(projections, pad=[-p] * 2)

    return projections
