import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid, transform_fftfreq_grid
from torch_image_interpolation import sample_image_2d, sample_image_3d

from .._dft_utils import _fftfreq_to_dft_coordinates
from .._grids import _apply_ewald_curvature, _central_slice_fftfreq_grid


def _determine_near_zero_conjugate_mask(
    coordinates: torch.Tensor,  # (..., n_points, 3)
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    tol: float = 1e-10,
) -> torch.Tensor:
    """Handle ambiguities in coordinates which get mapped to near-zero x-coord.

    This function operates on the principle that for a coordinate to be mapped to near
    zero in the x-direction after rotation, the following must be true since we know the
    z-coordinate of the original slice is zero:

    [a b c] [x]   [ax + by]   [x']
    [d e f] [y]   [dx + ey]   [y']
    [g h i] [0] = [gx + hy] = [z']

    where r11 = a & r12 = b are the elements of the rotation which determine the new
    x-coordinate (x'). The decision boundary for choosing if a point is conjugated
    (near zero x') is defined by a plane r11 * x + r12 * y = 0. When flipping the point
    and marking it for conjugation, we need to preserve the gradient of this decision
    boundary w.r.t the x-direction.

    The gradient of the decision boundary is d/dx = r11, and we are really only
    interested in the sign of the gradient. Thus, we use r11 to disambiguate points
    which lie close to the decision boundary.

    If r11 is also near zero using the same tolerance, we fall back to using r12 (the
    gradient w.r.t the y-direction) to determine the conjugation in the same manner.

    Parameters
    ----------
    coordinates : torch.Tensor
        Input coordinates (..., n_points, 3) which have already been rotated.
    rotation_matrices : torch.Tensor
        Rotation matrices (..., 3, 3) which were used to rotate the coordinates. Note
        that no additional rotations are applied to the coordinates, and only the
        elements of the matrix are used to determine if a conjugation is necessary.
    tol : float
        Tolerance for determining near-zero x-coordinates.

    Returns
    -------
    near_zero_conj_mask : torch.Tensor
        Mask indicating which coordinates were conjugated.
    """
    is_near_zero = torch.abs(coordinates[..., 2]) < tol  # (..., n_points)

    # Early exit if no near-zero coordinates
    if not torch.any(is_near_zero):
        return torch.zeros_like(is_near_zero, dtype=torch.bool)

    # NOTE: We are using zyx rotation matrices here, so r11 and r12 are flipped
    r11 = rotation_matrices[..., 2, 2]  # (...,)
    r12 = rotation_matrices[..., 2, 1]  # (...,)
    r11_broadcasted = einops.repeat(
        r11, "... -> ... n_points", n_points=is_near_zero.shape[-1]
    )
    r12_broadcasted = einops.repeat(
        r12, "... -> ... n_points", n_points=is_near_zero.shape[-1]
    )

    near_zero_conj_mask = torch.zeros_like(is_near_zero, dtype=torch.bool)

    # Use the sign of r11 to determine conjugation for near-zero x-coords only for
    # points not near zero for r11
    r11_is_zero = torch.abs(r11_broadcasted) <= tol * 10
    r11_decision_mask = is_near_zero & (~r11_is_zero)
    near_zero_conj_mask[r11_decision_mask] = r11_broadcasted[r11_decision_mask] < 0

    # For points where r11 is also near zero, use r12 to determine conjugation
    r12_decision_mask = is_near_zero & r11_is_zero
    near_zero_conj_mask[r12_decision_mask] = r12_broadcasted[r12_decision_mask] < 0

    return near_zero_conj_mask


def extract_central_slices_rfft_3d(
    volume_rfft: torch.Tensor,
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    x_tolerance: float = 1e-8,
    rot_tolerance: float | None = 1e-8,
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,  # in kV
    ewald_flip_sign: bool = False,  # if True, flip the sign of the Ewald curvature
    ewald_px_size: float = 1.0,  # in Angstroms / pixel
) -> torch.Tensor:
    """Extract central slice from an fftshifted rfft volume.

    NOTE: The proper pre-processing for a RFFT volume for this function is as follows:
    >>> volume = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # move center to corners
    >>> volume_rfft = torch.fft.rfftn(volume, dim=(-3, -2, -1))  # take RFFT
    >>> volume_rfft = torch.fft.fftshift(volume_rfft, dim=(-3, -2))  # center yz plane

    NOTE: This function is sensitive to small numerical precision in the rotation
    matrix for elements that are extremely close to zero. Use the 'rot_tolerance'
    parameter to zero-out elements whose absolute value is below the tolerance. Setting
    this to None will skip the zeroing step.

    Parameters
    ----------
    volume_rfft : torch.Tensor
        Input volume in fftshifted RFFT format, shape (d, d, d//2 + 1).
    rotation_matrices : torch.Tensor
        Rotation matrices (..., 3, 3) defining the orientations of the central slices
        to extract. Multiple matrices can be provided for batch extraction.
    fftfreq_max : float | None
        Maximum frequency, in terms of Nyquist frequency, to include in the extracted
        slices. For example, a value of 0.3 would discard all frequencies whose
        magnitude is grater than 0.3. The default is None and includes all frequencies.
    zyx_matrices : bool
        If True, the provided rotation matrices are assumed to be in zyx coordinates.
        If False, they are assumed to be in xyz coordinates and will be converted to
        zyx by flipping the last two axes. Default is False.
    x_tolerance : float
        Tolerance for determining near-zero x-coordinates for conjugation ambiguity
        resolution.
    rot_tolerance : float | None
        Tolerance for zeroing out near-zero elements in the rotation matrices to
        mitigate numerical precision issues. If None, no zeroing is performed.
    apply_ewald_curvature : bool
        If True, bend the central slice onto an Ewald sphere. If False (default),
        use a flat central slice.
    ewald_voltage_kv : float
        Acceleration voltage in kV. Default is 300.0 kV. Wavelength is computed
        from this using relativistic electron wavelength formula.
    ewald_flip_sign : bool
        If True, flip the sign of the Ewald curvature (apply the curve in the
        opposite direction).
    ewald_px_size : float
        Pixel size (e.g. Å / pixel). Used to convert between grid units
        (cycles / pixel) and physical spatial frequencies.

    Returns
    -------
    projection_image_dfts : torch.Tensor
        Extracted central slice DFTs in fftshifted RFFT format, shape
        (..., h, w), where h and w are the height and width of the RFFT slice.
    """
    assert volume_rfft.ndim == 3, "Input volume_rfft must be 3D tensor."

    # Get the shape of the original volume from the actual rfft tensor
    volume_shape = (
        volume_rfft.shape[-3],
        volume_rfft.shape[-2],
        (volume_rfft.shape[-1] - 1) * 2,
    )

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]   [ax + by + cz]   [x']
    # [d e f] [y]   [dx + ey + fz]   [y']
    # [g h i] [z] = [gx + hy + iz] = [z']
    #
    # zyx:
    # [i h g] [z]   [gx + hy + iz]   [z']
    # [f e d] [y]   [dx + ey + fz]   [y']
    # [c b a] [x] = [ax + by + cz] = [x']
    rotation_matrices = rotation_matrices.to(torch.float32)

    if rot_tolerance is not None:
        near_zero_elements = torch.abs(rotation_matrices) < rot_tolerance
        rotation_matrices[near_zero_elements] = 0.0

    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3) zyx coords

    # Optionally, bend the central slice into a curved surface (Ewald sphere)
    if apply_ewald_curvature:
        freq_grid = _apply_ewald_curvature(
            freq_grid=freq_grid,
            voltage_kv=ewald_voltage_kv,
            flip_sign=ewald_flip_sign,
            px_size=ewald_px_size,
        )

    # keep track of some shapes
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, *rfft_shape)

    # get (b, 3, 1) array of zyx coordinates to rotate
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=rfft_shape, dtype=torch.bool, device=volume_rfft.device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")

    # Add extra dimension onto rotation matrices and coordinates for MM broadcasting
    valid_coords = einops.rearrange(valid_coords, "b zyx -> b zyx 1")
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    rotated_coords = rotation_matrices @ valid_coords  # (..., b, zyx, 1)

    # Squeeze out single broadcast dimensions
    rotated_coords = einops.rearrange(rotated_coords, "... b zyx 1 -> ... b zyx")
    rotation_matrices = einops.rearrange(rotation_matrices, "... 1 i j -> ... i j")

    # Flip coordinates which we absolutely know end in the redundant (-x) half
    conjugate_mask = rotated_coords[..., 2] < -x_tolerance

    # Handle near-zero x-coordinates using the dedicated function
    near_zero_conj_mask = _determine_near_zero_conjugate_mask(
        coordinates=rotated_coords,
        rotation_matrices=rotation_matrices,
        tol=x_tolerance,
    )
    conjugate_mask = conjugate_mask | near_zero_conj_mask

    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = _fftfreq_to_dft_coordinates(
        frequencies=rotated_coords, image_shape=volume_shape, rfft=True
    )
    samples = sample_image_3d(
        image=volume_rfft, coordinates=rotated_coords, interpolation="trilinear"
    )  # (...) rfft

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(
        output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype
    )
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts


def extract_central_slices_rfft_3d_multichannel(
    volume_rfft: torch.Tensor,  # (c, d, d, d//2 + 1)
    rotation_matrices: torch.Tensor,  # (..., 3, 3)
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
    x_tolerance: float = 1e-8,
    rot_tolerance: float | None = 1e-8,
    apply_ewald_curvature: bool = False,
    ewald_voltage_kv: float = 300.0,  # in kV
    ewald_flip_sign: bool = False,  # if True, flip the sign of the Ewald curvature
    ewald_px_size: float = 1.0,  # in Angstroms / pixel
) -> torch.Tensor:  # (..., c, h, w)
    """Extract central slices from multichannel fftshifted rfft volume.

    NOTE: The proper pre-processing for a RFFT volume for this function is as follows:
    >>> volume = torch.fft.fftshift(volume, dim=(-3, -2, -1))  # move center to corners
    >>> volume_rfft = torch.fft.rfftn(volume, dim=(-3, -2, -1))  # take RFFT
    >>> volume_rfft = torch.fft.fftshift(volume_rfft, dim=(-3, -2))  # center yz plane

    NOTE: This function is sensitive to small numerical precision in the rotation
    matrix for elements that are extremely close to zero. Use the 'rot_tolerance'
    parameter to zero-out elements whose absolute value is below the tolerance. Setting
    this to None will skip the zeroing step.

    Parameters
    ----------
    volume_rfft : torch.Tensor
        Input multichannel volume in fftshifted RFFT format, shape (c, d, d, d//2 + 1).
    rotation_matrices : torch.Tensor
        Rotation matrices (..., 3, 3) defining the orientations of the central slices
        to extract. Multiple matrices can be provided for batch extraction.
    fftfreq_max : float | None
        Maximum frequency, in terms of Nyquist frequency, to include in the extracted
        slices. For example, a value of 0.3 would discard all frequencies whose
        magnitude is grater than 0.3. The default is None and includes all frequencies.
    zyx_matrices : bool
        If True, the provided rotation matrices are assumed to be in zyx coordinates.
        If False, they are assumed to be in xyz coordinates and will be converted to
        zyx by flipping the last two axes. Default is False.
    x_tolerance : float
        Tolerance for determining near-zero x-coordinates for conjugation ambiguity
        resolution.
    rot_tolerance : float | None
        Tolerance for zeroing out near-zero elements in the rotation matrices to
        mitigate numerical precision issues. If None, no zeroing is performed.
    apply_ewald_curvature : bool
        If True, bend the central slice onto an Ewald sphere. If False (default),
        use a flat central slice.
    ewald_voltage_kv : float
        Acceleration voltage in kV. Default is 300.0 kV. Wavelength is computed
        from this using relativistic electron wavelength formula.
    ewald_flip_sign : bool
        If True, flip the sign of the Ewald curvature (apply the curve in the
        opposite direction).
    ewald_px_size : float
        Pixel size (e.g. Å / pixel). Used to convert between grid units
        (cycles / pixel) and physical spatial frequencies.

    Returns
    -------
    projection_image_dfts : torch.Tensor
        Extracted central slice DFTs in fftshifted RFFT format, shape
        (..., c, h, w), where c is the number of channels, h and w are the height
        and width of the RFFT slice.
    """
    assert volume_rfft.ndim == 4, "Input volume_rfft must be 4D tensor."

    # Get the shape of the original volume from the actual rfft tensor
    volume_shape = (
        volume_rfft.shape[-3],
        volume_rfft.shape[-2],
        (volume_rfft.shape[-1] - 1) * 2,
    )

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]   [ax + by + cz]   [x']
    # [d e f] [y]   [dx + ey + fz]   [y']
    # [g h i] [z] = [gx + hy + iz] = [z']
    #
    # zyx:
    # [i h g] [z]   [gx + hy + iz]   [z']
    # [f e d] [y]   [dx + ey + fz]   [y']
    # [c b a] [x] = [ax + by + cz] = [x']
    rotation_matrices = rotation_matrices.to(torch.float32)

    if rot_tolerance is not None:
        near_zero_elements = torch.abs(rotation_matrices) < rot_tolerance
        rotation_matrices[near_zero_elements] = 0.0

    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=volume_rfft.device,
    )  # (h, w, 3) zyx coords

    # Optionally, bend the central slice into a curved surface (Ewald sphere)
    if apply_ewald_curvature:
        freq_grid = _apply_ewald_curvature(
            freq_grid=freq_grid,
            voltage_kv=ewald_voltage_kv,
            flip_sign=ewald_flip_sign,
            px_size=ewald_px_size,
        )

    # keep track of some shapes
    channels = volume_rfft.shape[0]
    stack_shape = tuple(rotation_matrices.shape[:-2])
    rfft_shape = freq_grid.shape[-3], freq_grid.shape[-2]
    output_shape = (*stack_shape, channels, *rfft_shape)

    # get (b, 3, 1) array of zyx coordinates to rotate
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
        valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    else:
        freq_grid_mask = torch.ones(
            size=rfft_shape, dtype=torch.bool, device=volume_rfft.device
        )
        valid_coords = einops.rearrange(freq_grid, "h w zyx -> (h w) zyx")

    # Add extra dim to rotation matrices and coordinates for broadcasting
    valid_coords = einops.rearrange(valid_coords, "hw zyx -> hw zyx 1")
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    rotated_coords = rotation_matrices @ valid_coords  # (..., hw, zyx, 1)

    # Squeeze out single broadcast dimensions
    rotated_coords = einops.rearrange(rotated_coords, "... hw zyx 1 -> ... hw zyx")
    rotation_matrices = einops.rearrange(rotation_matrices, "... 1 i j -> ... i j")

    # Flip coordinates which we absolutely know end in the redundant (-x) half
    conjugate_mask = rotated_coords[..., 2] < -x_tolerance

    # Handle near-zero x-coordinates using the dedicated function
    near_zero_conj_mask = _determine_near_zero_conjugate_mask(
        coordinates=rotated_coords,
        rotation_matrices=rotation_matrices,
        tol=x_tolerance,
    )
    conjugate_mask = conjugate_mask | near_zero_conj_mask

    rotated_coords[conjugate_mask, ...] *= -1

    # convert frequencies to array coordinates in fftshifted DFT
    rotated_coords = _fftfreq_to_dft_coordinates(
        frequencies=rotated_coords, image_shape=volume_shape, rfft=True
    )
    samples = sample_image_3d(
        image=volume_rfft, coordinates=rotated_coords, interpolation="trilinear"
    )  # shape is (..., hw, c)

    # take complex conjugate of values from redundant half transform
    samples[conjugate_mask] = torch.conj(samples[conjugate_mask])

    # Rearrange to put channels before spatial dimensions
    samples = einops.rearrange(samples, "... hw c -> ... c hw")

    # insert samples back into DFTs
    projection_image_dfts = torch.zeros(
        output_shape, device=volume_rfft.device, dtype=volume_rfft.dtype
    )
    projection_image_dfts[..., freq_grid_mask] = samples

    return projection_image_dfts


def transform_slice_2d(
    projection_image_dfts: torch.Tensor,
    rfft_shape: tuple[int, int],
    stack_shape: tuple[int, ...],
    transform_matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply anisotropic magnification transformation to a 2D slice in Fourier space.

    Parameters
    ----------
    projection_image_dfts: torch.Tensor
        Projection image DFTs with shape `(*stack_shape, h, w)` where `(h, w)`
        is the rfft shape.
    rfft_shape: tuple[int, int]
        Shape of the rfft `(h, w_rfft)` where `w_rfft = n//2 + 1`.
    stack_shape: tuple[int, ...]
        Shape of batch dimensions (can be empty for single image).
    transform_matrix: torch.Tensor
        `(2, 2)` anisotropic magnification matrix in real space (yx ordering).

    Returns
    -------
    transformed_dfts: torch.Tensor
        Transformed projection image DFTs with same shape as input.
    """
    # Calculate full 2D image shape from rfft shape
    # rfft_shape is (h, w_rfft) where w_rfft = n//2 + 1
    # Full width is w_full = 2 * (w_rfft - 1) = n
    h_full = rfft_shape[0]
    w_full = 2 * (rfft_shape[1] - 1)
    image_shape_2d = (h_full, w_full)

    # Generate 2D frequency grid
    freq_grid = fftfreq_grid(
        image_shape=image_shape_2d,
        rfft=True,
        fftshift=True,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords

    # Apply transformation matrix
    transformed_freqs = transform_fftfreq_grid(
        frequency_grid=freq_grid,
        real_space_matrix=transform_matrix,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords

    # Convert transformed frequencies to DFT array coordinates
    transformed_coords = _fftfreq_to_dft_coordinates(
        frequencies=transformed_freqs, image_shape=image_shape_2d, rfft=True
    )

    # Flatten coordinates: (h, w, 2) -> (h*w, 2)
    transformed_coords_flat = einops.rearrange(transformed_coords, "h w yx -> (h w) yx")

    # Resample each 2D DFT at the transformed frequencies
    # Need to handle batch dimensions in stack_shape
    if len(stack_shape) > 0:
        # Flatten batch dimensions for processing
        projection_image_dfts_flat = einops.rearrange(
            projection_image_dfts, "... h w -> (...) h w"
        )

        # Use channel dimension as batch: pass (b, h, w) as (c=b, h, w)
        resampled = sample_image_2d(
            image=projection_image_dfts_flat,  # (b, h, w) treated as (c=b, h, w)
            coordinates=transformed_coords_flat,  # (h*w, 2)
            interpolation="bilinear",
        )  # Returns (h*w, b) - samples for each "channel" (batch)

        # Transpose to get (b, h*w)
        n_coords = transformed_coords_flat.shape[0]
        resampled = einops.rearrange(
            resampled, "n_coords b -> b n_coords", n_coords=n_coords
        )
        # Reshape back to (b, h, w) - coordinates now match rfft_shape
        resampled = einops.rearrange(
            resampled, "b (h w) -> b h w", h=rfft_shape[0], w=rfft_shape[1]
        )
        # Reshape to original stack_shape
        projection_image_dfts = resampled.reshape(*stack_shape, *rfft_shape)
    else:
        # Single image case
        resampled = sample_image_2d(
            image=projection_image_dfts[None, ...],  # Add batch dim -> (1, h, w)
            coordinates=transformed_coords_flat,  # (h*w, 2) - no batch dim
            interpolation="bilinear",
        )  # Returns (h*w, 1) - samples for single "channel" (batch)
        resampled = resampled[:, 0]  # Remove batch dim -> (h*w,)
        projection_image_dfts = einops.rearrange(
            resampled, "(h w) -> h w", h=rfft_shape[0], w=rfft_shape[1]
        )

    # Apply the 1/|det A| scaling to preserve total intensity
    det_A = torch.linalg.det(transform_matrix)
    projection_image_dfts = projection_image_dfts / det_A

    return projection_image_dfts


def transform_slice_2d_multichannel(
    projection_image_dfts: torch.Tensor,
    rfft_shape: tuple[int, int],
    stack_shape: tuple[int, ...],
    channels: int,
    transform_matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply anisotropic magnification transformation to a multichannel 2D slice.

    Parameters
    ----------
    projection_image_dfts: torch.Tensor
        Projection image DFTs with shape `(*stack_shape, c, h, w)` where `(h, w)`
        is the rfft shape and `c` is the number of channels.
    rfft_shape: tuple[int, int]
        Shape of the rfft `(h, w_rfft)` where `w_rfft = n//2 + 1`.
    stack_shape: tuple[int, ...]
        Shape of batch dimensions (can be empty for single image).
    channels: int
        Number of channels.
    transform_matrix: torch.Tensor
        `(2, 2)` anisotropic magnification matrix in real space (yx ordering).

    Returns
    -------
    transformed_dfts: torch.Tensor
        Transformed projection image DFTs with same shape as input.
    """
    # Calculate full 2D image shape from rfft shape
    h_full = rfft_shape[0]
    w_full = 2 * (rfft_shape[1] - 1)
    image_shape_2d = (h_full, w_full)

    # Generate 2D frequency grid
    freq_grid = fftfreq_grid(
        image_shape=image_shape_2d,
        rfft=True,
        fftshift=True,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords

    # Apply transformation matrix
    transformed_freqs = transform_fftfreq_grid(
        frequency_grid=freq_grid,
        real_space_matrix=transform_matrix,
        device=projection_image_dfts.device,
    )  # (h, w, 2) yx coords

    # Convert transformed frequencies to DFT array coordinates
    transformed_coords = _fftfreq_to_dft_coordinates(
        frequencies=transformed_freqs, image_shape=image_shape_2d, rfft=True
    )

    # Flatten coordinates: (h, w, 2) -> (h*w, 2)
    transformed_coords_flat = einops.rearrange(transformed_coords, "h w yx -> (h w) yx")

    # Resample each 2D DFT at the transformed frequencies
    # Handle batch and channel dimensions
    if len(stack_shape) > 0:
        # Flatten batch dimensions but keep channels separate
        projection_image_dfts_flat = einops.rearrange(
            projection_image_dfts, "... c h w -> (...) c h w"
        )

        # Process each (batch, channel) combination
        batch_size = projection_image_dfts_flat.shape[0]

        # Flatten channels into batch for sampling - treat (b, c, h, w) as (b*c, h, w)
        projection_image_dfts_for_sampling = einops.rearrange(
            projection_image_dfts_flat, "b c h w -> (b c) h w"
        )

        # Use non-batched coordinates for all (b, c) combinations
        # sample_image_2d will return (h*w, b*c) when image is (b*c, h, w)
        # and coords are (h*w, 2)
        resampled = sample_image_2d(
            image=projection_image_dfts_for_sampling,  # (b*c, h, w)
            coordinates=transformed_coords_flat,  # (h*w, 2) - non-batched
            interpolation="bilinear",
        )  # Returns (h*w, b*c) - samples for each coordinate, for each batch/channel

        # Transpose to get (b*c, h*w)
        n_coords = transformed_coords_flat.shape[0]
        resampled = einops.rearrange(
            resampled,
            "n_coords (b c) -> (b c) n_coords",
            b=batch_size,
            c=channels,
            n_coords=n_coords,
        )

        # Reshape back to (b, c, h, w) then to original stack shape
        resampled = einops.rearrange(
            resampled,
            "(b c) (h w) -> b c h w",
            b=batch_size,
            c=channels,
            h=rfft_shape[0],
            w=rfft_shape[1],
        )
        projection_image_dfts = resampled.reshape(*stack_shape, channels, *rfft_shape)
    else:
        # Single image case with channels
        # Use non-batched coordinates - sample_image_2d will return (h*w, c)
        # when image is (c, h, w) and coords are (h*w, 2)
        resampled = sample_image_2d(
            image=projection_image_dfts,  # (c, h, w)
            coordinates=transformed_coords_flat,  # (h*w, 2) - non-batched
            interpolation="bilinear",
        )  # Returns (h*w, c) - samples for each coordinate, for each channel

        # Transpose to get (c, h*w)
        n_coords = transformed_coords_flat.shape[0]
        resampled = einops.rearrange(
            resampled, "n_coords c -> c n_coords", c=channels, n_coords=n_coords
        )
        projection_image_dfts = einops.rearrange(
            resampled, "c (h w) -> c h w", h=rfft_shape[0], w=rfft_shape[1]
        )

    # Apply the 1/|det A| scaling to preserve total intensity
    det_A = torch.linalg.det(transform_matrix)
    projection_image_dfts = projection_image_dfts / det_A

    return projection_image_dfts
