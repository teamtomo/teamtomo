import einops
import torch
from torch_grid_utils.fftfreq_grid import rfft_shape

from .._dft_utils import _fftfreq_to_dft_coordinates
from .._grids import _central_slice_fftfreq_grid


def insert_central_slices_rfft_3d(
    image_rfft: torch.Tensor,  # fftshifted rfft of (..., h, w) 2d image
    volume_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrices = rotation_matrices.to(torch.float32)

    ft_dtype = image_rfft.dtype
    device = image_rfft.device

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=device,
    )  # (h, w, 3)

    # get (b, 3, 1) array of zyx coordinates to rotate (up to fftfreq_max)
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
    else:
        freq_grid_mask = torch.ones(
            size=image_rfft.shape[-2:], dtype=torch.bool, device=device
        )

    # Only keep y >= 0 at x=0 to avoid
    # double-inserting conjugate pairs (Friedel symmetry).
    # In rfft, for x=0 plane: ft[z, y, 0] == conj(ft[-z, -y, 0])
    # freq_grid shape is (h, w, 3) where last dim is zyx
    x_coords = freq_grid[..., 2]  # (h, w)
    y_coords = freq_grid[..., 1]  # (h, w)
    friedel_redundant = (x_coords == 0) & (y_coords < 0)
    # Combine with existing frequency mask
    freq_grid_mask = freq_grid_mask & ~friedel_redundant

    valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)
    valid_coords = einops.rearrange(valid_coords, "b zyx -> b zyx 1")

    # get (..., b) array of data at each coordinate from image rffts
    valid_data = image_rfft[..., freq_grid_mask]

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]    [ax + by + cz]
    # [d e f] [y]  = [dx + ey + fz]
    # [g h i] [z]    [gx + hy + iz]
    #
    # zyx:
    # [i h g] [z]    [gx + hy + iz]
    # [f e d] [y]  = [dx + ey + fz]
    # [c b a] [x]    [ax + by + cz]
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix and remove last dim
    rotated_coordinates = einops.rearrange(
        rotation_matrices @ valid_coords, pattern="... b zyx 1 -> ... b zyx"
    )

    # flip coordinates in redundant half transform and take conjugate value
    conjugate_mask = rotated_coordinates[..., 2] < 0
    rotated_coordinates[conjugate_mask] *= -1
    valid_data[conjugate_mask] = torch.conj(valid_data[conjugate_mask])

    # calculate positions to sample in DFT array from fftfreq coordinates
    rotated_coordinates = _fftfreq_to_dft_coordinates(
        rotated_coordinates, image_shape=volume_shape, rfft=True
    )

    # initialise output volume and volume for keeping track of weights
    dft_3d = torch.zeros(
        size=rfft_shape(volume_shape),
        dtype=ft_dtype,
        device=device,
    )
    weights = torch.zeros_like(
        dft_3d,
        dtype=torch.float32 if ft_dtype == torch.complex64 else torch.float64,
        device=device,
    )

    # insert data into 3D DFT
    dft_3d, weights = _insert_into_3d_dft(
        values=valid_data,
        coordinates=rotated_coordinates,
        image=dft_3d,
        weights=weights,
    )
    return dft_3d, weights


def insert_central_slices_rfft_3d_multichannel(
    image_rfft: torch.Tensor,  # fftshifted rfft of (..., c, d, d) 2d image
    volume_shape: tuple[int, int, int],
    rotation_matrices: torch.Tensor,  # (..., 3, 3) dims need to match rfft
    fftfreq_max: float | None = None,
    zyx_matrices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrices = rotation_matrices.to(torch.float32)
    ft_dtype = image_rfft.dtype
    device = image_rfft.device
    channels = image_rfft.shape[-3]

    # generate grid of DFT sample frequencies for a central slice spanning the xy-plane
    freq_grid = _central_slice_fftfreq_grid(
        volume_shape=volume_shape,
        rfft=True,
        fftshift=True,
        device=device,
    )  # (d, d, d, 3)

    # get (b, 3, 1) array of zyx coordinates to rotate (up to fftfreq_max)
    if fftfreq_max is not None:
        normed_grid = (
            einops.reduce(freq_grid**2, "h w zyx -> h w", reduction="sum") ** 0.5
        )
        freq_grid_mask = normed_grid <= fftfreq_max
    else:
        freq_grid_mask = torch.ones(
            size=image_rfft.shape[-2:], dtype=torch.bool, device=device
        )

    # Only keep y >= 0 at x=0 to avoid
    # double-inserting conjugate pairs (Friedel symmetry).
    # In rfft, for x=0 plane: ft[z, y, 0] == conj(ft[-z, -y, 0])
    # freq_grid shape is (h, w, 3) where last dim is zyx
    x_coords = freq_grid[..., 2]  # (h, w)
    y_coords = freq_grid[..., 1]  # (h, w)
    friedel_redundant = (x_coords == 0) & (y_coords < 0)
    # Combine with existing frequency mask
    freq_grid_mask = freq_grid_mask & ~friedel_redundant

    valid_coords = freq_grid[freq_grid_mask, ...]  # (b, zyx)

    valid_coords = einops.rearrange(valid_coords, "hw zyx -> hw zyx 1")

    # get (..., b) array of data at each coordinate from image rffts
    valid_data = image_rfft[..., freq_grid_mask]

    # rotation matrices rotate xyz coordinates, make them rotate zyx coordinates
    # xyz:
    # [a b c] [x]    [ax + by + cz]
    # [d e f] [y]  = [dx + ey + fz]
    # [g h i] [z]    [gx + hy + iz]
    #
    # zyx:
    # [i h g] [z]    [gx + hy + iz]
    # [f e d] [y]  = [dx + ey + fz]
    # [c b a] [x]    [ax + by + cz]
    if not zyx_matrices:
        rotation_matrices = torch.flip(rotation_matrices, dims=(-2, -1))

    # add extra dim to rotation matrices for broadcasting
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 i j")

    # rotate all valid coordinates by each rotation matrix and remove last dim
    rotated_coordinates = einops.rearrange(
        rotation_matrices @ valid_coords, pattern="... hw zyx 1 -> ... hw zyx"
    )

    # flip coordinates in redundant half transform and take conjugate value
    conjugate_mask = rotated_coordinates[..., 2] < 0
    rotated_coordinates[conjugate_mask] *= -1
    # switch channel to end for torch-image-interpolation
    valid_data = einops.rearrange(valid_data, "... c hw -> ... hw c")
    valid_data[conjugate_mask] = torch.conj(valid_data[conjugate_mask])

    # calculate positions to sample in DFT array from fftfreq coordinates
    rotated_coordinates = _fftfreq_to_dft_coordinates(
        rotated_coordinates, image_shape=volume_shape, rfft=True
    )

    # initialise output volume and volume for keeping track of weights
    volume_dft_shape = rfft_shape(volume_shape)
    dft_3d = torch.zeros(
        size=(channels, *volume_dft_shape),
        dtype=ft_dtype,
        device=device,
    )
    weights = torch.zeros(
        size=volume_dft_shape,
        dtype=torch.float32 if ft_dtype == torch.complex64 else torch.float64,
        device=device,
    )

    # insert data into 3D DFT
    dft_3d, weights = _insert_into_3d_dft(
        values=valid_data,
        coordinates=rotated_coordinates,
        image=dft_3d,
        weights=weights,
    )
    return dft_3d, weights


def _insert_into_3d_dft(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    image: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert values into a 3D image using trilinear interpolation.

    Parameters
    ----------
    values: torch.Tensor
        `(...)` or `(..., c)` array of values to be inserted into `image`.
    coordinates: torch.Tensor
        `(..., 3)` array of 3D coordinates for each value in `data`.
        - Coordinates are ordered `zyx` and are positions in the `d`, `h` and `w`
        dimensions respectively.
        - Coordinates span the range `[0, N-1]` for a dimension of length N.
    image: torch.Tensor
        `(d, h, w)` or `(c, d, h, w)` array containing the image into which data will
        be inserted.
    weights: torch.Tensor | None
        `(d, h, w)` array containing weights associated with each voxel in `image`.
        This is useful for tracking weights across multiple calls to this function.

    Returns
    -------
    image, weights: tuple[torch.Tensor, torch.Tensor]
        The image and weights after updating with data from `data` at `coordinates`.
    """
    # keep track of a few properties of the inputs
    input_image_is_multichannel = image.ndim == 4
    d, h, w = image.shape[-3:]

    # validate inputs
    values_shape = values.shape[:-1] if input_image_is_multichannel else values.shape
    coordinates_shape, coordinates_ndim = coordinates.shape[:-1], coordinates.shape[-1]

    if values_shape != coordinates_shape:
        raise ValueError("One coordinate triplet is required for each value in data.")
    if coordinates_ndim != 3:
        raise ValueError("Coordinates must be 3D with shape (..., 3).")
    if image.dtype != values.dtype:
        raise ValueError("Image and values must have the same dtype.")

    if weights is None:
        weights = torch.zeros(size=(d, h, w), dtype=torch.float32, device=image.device)

    # add channel dim to both image and values if input image is not multichannel
    if not input_image_is_multichannel:
        image = einops.rearrange(image, "d h w -> 1 d h w")
        values = einops.rearrange(values, "... -> ... 1")

    # linearise data and coordinates
    values, _ = einops.pack([values], pattern="* c")
    coordinates, _ = einops.pack([coordinates], pattern="* zyx")
    coordinates = coordinates.float()

    # only keep data and coordinates inside the image
    image_shape = torch.tensor((d, h, w), device=image.device, dtype=torch.float32)
    upper_bound = image_shape - 1
    idx_inside = (coordinates >= 0) & (coordinates <= upper_bound)
    idx_inside = torch.all(idx_inside, dim=-1)
    values, coordinates = values[idx_inside], coordinates[idx_inside]

    # splat data onto grid using trilinear interpolation
    image, weights = _insert_into_3d_dft_linear(values, coordinates, image, weights)

    # ensure correct output image shape
    # single channel input -> (d, h, w)
    # multichannel input -> (c, d, h, w)
    if not input_image_is_multichannel:
        image = einops.rearrange(image, "1 d h w -> d h w")

    return image, weights


def _insert_into_3d_dft_linear(
    data: torch.Tensor,  # (b, c)
    coordinates: torch.Tensor,  # (b, zyx)
    image: torch.Tensor,  # (c, d, h, w)
    weights: torch.Tensor,  # (d, h, w)
) -> tuple[torch.Tensor, torch.Tensor]:
    # b is number of data points to insert per channel, c is number of channels
    b, c = data.shape

    # cache corner coordinates for each value to be inserted
    coordinates = einops.rearrange(coordinates, "b zyx -> zyx b")
    z0, y0, x0 = torch.floor(coordinates)
    z1, y1, x1 = torch.ceil(coordinates)

    # populate arrays of corner indices
    idx_z = torch.empty(size=(b, 2, 2, 2), dtype=torch.long, device=image.device)
    idx_y = torch.empty(size=(b, 2, 2, 2), dtype=torch.long, device=image.device)
    idx_x = torch.empty(size=(b, 2, 2, 2), dtype=torch.long, device=image.device)

    idx_z[:, 0, 0, 0], idx_y[:, 0, 0, 0], idx_x[:, 0, 0, 0] = z0, y0, x0  # C000
    idx_z[:, 0, 0, 1], idx_y[:, 0, 0, 1], idx_x[:, 0, 0, 1] = z0, y0, x1  # C001
    idx_z[:, 0, 1, 0], idx_y[:, 0, 1, 0], idx_x[:, 0, 1, 0] = z0, y1, x0  # C010
    idx_z[:, 0, 1, 1], idx_y[:, 0, 1, 1], idx_x[:, 0, 1, 1] = z0, y1, x1  # C011
    idx_z[:, 1, 0, 0], idx_y[:, 1, 0, 0], idx_x[:, 1, 0, 0] = z1, y0, x0  # C100
    idx_z[:, 1, 0, 1], idx_y[:, 1, 0, 1], idx_x[:, 1, 0, 1] = z1, y0, x1  # C101
    idx_z[:, 1, 1, 0], idx_y[:, 1, 1, 0], idx_x[:, 1, 1, 0] = z1, y1, x0  # C110
    idx_z[:, 1, 1, 1], idx_y[:, 1, 1, 1], idx_x[:, 1, 1, 1] = z1, y1, x1  # C111

    # calculate trilinear interpolation weights for each corner
    z, y, x = coordinates
    tz, ty, tx = z - z0, y - y0, x - x0  # fractional position between voxel corners
    w = torch.empty(size=(b, 2, 2, 2), device=image.device, dtype=weights.dtype)

    w[:, 0, 0, 0] = (1 - tz) * (1 - ty) * (1 - tx)  # C000
    w[:, 0, 0, 1] = (1 - tz) * (1 - ty) * tx  # C001
    w[:, 0, 1, 0] = (1 - tz) * ty * (1 - tx)  # C010
    w[:, 0, 1, 1] = (1 - tz) * ty * tx  # C011
    w[:, 1, 0, 0] = tz * (1 - ty) * (1 - tx)  # C100
    w[:, 1, 0, 1] = tz * (1 - ty) * tx  # C101
    w[:, 1, 1, 0] = tz * ty * (1 - tx)  # C110
    w[:, 1, 1, 1] = tz * ty * tx  # C111

    # make sure indices broadcast correctly
    idx_c = torch.arange(c, device=coordinates.device, dtype=torch.long)
    idx_c = einops.rearrange(idx_c, "c -> 1 c 1 1 1")
    idx_z = einops.rearrange(idx_z, "b z y x -> b 1 z y x")
    idx_y = einops.rearrange(idx_y, "b z y x -> b 1 z y x")
    idx_x = einops.rearrange(idx_x, "b z y x -> b 1 z y x")

    # insert weighted data and weight values at each corner
    data = einops.rearrange(data, "b c -> b c 1 1 1")
    w = einops.rearrange(w, "b z y x -> b 1 z y x")
    image.index_put_(
        indices=(idx_c, idx_z, idx_y, idx_x),
        values=data * w.to(data.dtype),
        accumulate=True,
    )
    weights.index_put_(indices=(idx_z, idx_y, idx_x), values=w, accumulate=True)

    # Enforce Friedel symmetry for insertions into x=0 plane
    # For any value inserted at (z, y, 0), also insert conj(value) at (d-z, d-y, 0)
    d, h, w_dim = weights.shape  # d, h, w dimensions of the volume

    # Find which corners are on the x=0 plane
    # idx_x has shape (b, 1, 2, 2, 2) after rearranging
    x0_mask = idx_x == 0  # (b, 1, 2, 2, 2) boolean mask

    if torch.any(x0_mask):
        # Calculate mirrored indices for Friedel symmetry
        # Mirror: (z, y, 0) -> ((d - z) % d, (d - y) % d, 0)
        idx_z_mirror = (d - idx_z) % d
        idx_y_mirror = (h - idx_y) % h

        # Only insert conjugate at mirrored positions when original is at x=0
        # Exclude the origin (d//2, h//2, 0) to avoid double insertion at DC
        # Note: after fftshift, DC is at (d//2, h//2)
        is_dc = (idx_z == d // 2) & (idx_y == h // 2) & (idx_x == 0)
        insert_conjugate_mask = x0_mask & ~is_dc

        if torch.any(insert_conjugate_mask):
            data_conj = torch.conj(data)

            # Ensure all the tensors have the same shape for masking
            # Using expand() creates views (no memory copies) instead of repeat()
            insert_conjugate_mask_expanded = insert_conjugate_mask.expand(b, c, 2, 2, 2)
            idx_c_expanded = idx_c.expand(b, c, 2, 2, 2)
            idx_z_mirror_expanded = idx_z_mirror.expand(b, c, 2, 2, 2)
            idx_y_mirror_expanded = idx_y_mirror.expand(b, c, 2, 2, 2)
            idx_x_expanded = idx_x.expand(b, c, 2, 2, 2)
            data_conj_expanded = data_conj.expand(b, c, 2, 2, 2)
            w_expanded = w.expand(b, c, 2, 2, 2).to(data_conj.dtype)

            # Insert conjugated, weighted data at mirrored positions
            image.index_put_(
                indices=(
                    idx_c_expanded[insert_conjugate_mask_expanded],
                    idx_z_mirror_expanded[insert_conjugate_mask_expanded],
                    idx_y_mirror_expanded[insert_conjugate_mask_expanded],
                    idx_x_expanded[insert_conjugate_mask_expanded],
                ),
                values=(data_conj_expanded * w_expanded)[
                    insert_conjugate_mask_expanded
                ],
                accumulate=True,
            )

            # For weights, use original mask shape (no channel dimension)
            weights.index_put_(
                indices=(
                    idx_z_mirror[insert_conjugate_mask],
                    idx_y_mirror[insert_conjugate_mask],
                    idx_x[insert_conjugate_mask],
                ),
                values=w[insert_conjugate_mask],
                accumulate=True,
            )

    return image, weights
