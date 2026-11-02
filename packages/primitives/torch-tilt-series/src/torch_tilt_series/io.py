"""Load TiltSeries instances and their raw tilt images.

Alignment metadata is loaded from external alignment formats (current
support: etomo, via etomofiles, and aretomo, via alnfile); raw tilt image
pixel data is loaded from the MRC stack a TiltSeries points to, via mrcfile.
"""

from pathlib import Path

import numpy as np
import torch

from torch_tilt_series.tilt_series import TiltSeries, _writable


def from_aretomo_output(
    aln_path: Path | str,
    pixel_spacing: float,
    image_path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> TiltSeries:
    """Initialize TiltSeries from an AreTomo .aln file.

    Tilt angles (`tilt` column) and per-tilt tilt axis angles (`rot` column)
    are taken as-is, in degrees. The `tx`, `ty` shifts (pixels) are reordered
    to yx and multiplied by `pixel_spacing` to give `sample_translations` in
    Angstroms. Only the sections listed in the .aln file are kept;
    `image_indices` is set from its 1-based `sec` column, converted to 0-based.

    Parameters
    ----------
    aln_path : Path | str
        Path to the AreTomo `.aln` alignment file.
    pixel_spacing : float
        Pixel size of the raw tilt images, in Angstroms. Used to convert the
        alignment shifts from pixels to Angstroms and stored on the result.
    image_path : Path | str | None
        Path to the raw tilt stack. Defaults to `aln_path` with its suffix
        replaced by `.mrc`. The file is not opened here.
    device : torch.device | str
        Device on which the geometry tensors are stored. Default `"cpu"`.

    Returns
    -------
    tilt_series : TiltSeries
        Tilt series with identity `sample2levelled`/`levelled2tomo` and
        `x_tilts` of 0.
    """
    import alnfile

    aln_path = Path(aln_path)
    df = alnfile.read(aln_path)

    if image_path is None:
        image_path = aln_path.with_suffix(".mrc")

    # Extract XY shifts and convert to YX convention
    corrected_shifts_xy = df[["tx", "ty"]].to_numpy()
    corrected_shifts_yx = corrected_shifts_xy[:, ::-1].copy()

    # Convert shifts from pixels to Angstroms
    corrected_shifts_yx_ang = corrected_shifts_yx * pixel_spacing

    # Valid tilt indices into the raw (unfiltered) tilt stack, 0-indexed.
    idx_valid = df["sec"].values - 1
    return TiltSeries(
        tilt_angles=df["tilt"].to_numpy(),
        tilt_axis_angle=df["rot"].to_numpy(),
        sample_translations=corrected_shifts_yx_ang,
        image_path=image_path,
        image_indices=idx_valid,
        pixel_spacing=pixel_spacing,
        device=device,
    )


def from_etomo_directory(
    etomo_dir: Path | str,
    pixel_spacing: float,
    device: torch.device | str = "cpu",
) -> TiltSeries:
    """Initialize TiltSeries from an ETOMO directory.

    Reads the IMOD alignment files with `etomofiles` and drops excluded views.
    The IMOD `.xf` transforms (which map images back to the aligned frame) are
    inverted to this package's forward model, and the shifts are converted from
    pixels to Angstroms. The global `XAXISTILT` (sign flipped to match this
    package's `Rx` convention) is added to the per-view x-tilts; if all views
    then share one value it is stored as a scalar. The raw stack path is
    resolved relative to `etomo_dir` unless absolute.

    Parameters
    ----------
    etomo_dir : Path | str
        Path to the ETOMO project directory.
    pixel_spacing : float
        Pixel size of the raw tilt images, in Angstroms. Used to convert the
        alignment shifts from pixels to Angstroms and stored on the result.
    device : torch.device | str
        Device on which the geometry tensors are stored. Default `"cpu"`.

    Returns
    -------
    tilt_series : TiltSeries
        Tilt series for the non-excluded views, with identity
        `sample2levelled`/`levelled2tomo`.
    """
    import etomofiles

    etomo_dir = Path(etomo_dir)
    df = etomofiles.read(etomo_dir)
    df = df.loc[~df["excluded"]].reset_index(drop=True)
    # Get IMOD xf components from dataframe
    # df_to_xf(df, yx=True) returns (n_tilts, 2, 3) array
    # Each matrix is [[A22, A21, DY], [A12, A11, DX]] (ready for torch-tomogram yz)
    xf = etomofiles.df_to_xf(df, yx=True)
    m, shifts = xf[:, :, :2], xf[:, :, 2]
    # Convert IMOD's backward projection model to this package's forward model
    # (scope2detector, see TiltSeries docstring).
    # IMOD: image -> scope (i.e. detector -> scope, the inverse direction).
    #   > the 2d matrix from the .xf file represents a 2d transform to align
    #   > the images so that they represent projections of a solid body
    #   > tilted around the Y axis
    # this package: scope -> image (scope2detector's forward direction)
    #   > the shifts are applied after rotation and projection and shift the
    #   > projected scope-space point to the image position
    #
    #  Rotation matrix are orthogonal, so inversion = transposition :
    #  np.einsum('nij,nj->ni', np.linalg.inv(m), shifts)
    #    = np.einsum('nji,nj->ni', m, shifts)
    #
    #  Negate shifts for forward projection model
    corrected_shifts = -np.einsum("nji,nj->ni", m, shifts)
    corrected_shifts = np.ascontiguousarray(corrected_shifts)

    # Convert shifts from pixels to Angstroms
    corrected_shifts_ang = corrected_shifts * pixel_spacing

    # X-axis tilt: IMOD combines a global scalar (XAXISTILT in tilt.com) with
    # per-view values (XTILTFILE / .xtilt). This feeds
    # sample2scope's Rx factor. IMOD's XAXISTILT sign convention is opposite
    # to our Rx convention.
    xaxistilt = -(df["xaxistilt"].iloc[0] or 0.0)
    x_tilts_perview = np.nan_to_num(df["xtilt"].to_numpy(), nan=0.0)
    x_tilts = xaxistilt + x_tilts_perview
    if np.all(x_tilts == x_tilts[0]):
        x_tilts = float(x_tilts[0])

    # Resolve the raw tilt stack path and the tilt selection/order.
    tilt_stack_name = df["image_path"][0].split("[")[0]
    tilt_stack_path = Path(tilt_stack_name)
    if not tilt_stack_path.is_absolute():
        tilt_stack_path = etomo_dir / tilt_stack_path

    # sample2levelled/levelled2tomo left at their default identity: this
    # loader draws no distinction between sample, levelled sample, and
    # tomogram space. A caller wanting a leveled correction or an arbitrary
    # reframing sets `ts.sample2levelled`/`ts.levelled2tomo` afterwards.
    return TiltSeries(
        tilt_angles=df["tlt"].to_numpy(),
        tilt_axis_angle=df["tilt_axis_angle"].to_numpy(),
        sample_translations=corrected_shifts_ang,
        x_tilts=x_tilts,
        image_path=tilt_stack_path,
        image_indices=df["idx_tilt"].to_numpy(),
        pixel_spacing=pixel_spacing,
        device=device,
    )


def load_tilt_series_images(
    tilt_series: TiltSeries,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Load the raw tilt images matching a TiltSeries' geometry.

    Reads `tilt_series.image_path` and selects/orders rows via
    `tilt_series.image_indices` so the result lines up 1:1 with
    `tilt_series.tilt_angles` etc. Returns unnormalized float32 pixel data at
    `tilt_series.pixel_spacing`; see `torch_tilt_series.utils` for the
    central-crop normalization and background subtraction used elsewhere in
    teamtomo, and `torch_tilt_series.preprocessing` for a full preprocessing
    pipeline built on top of those.

    Parameters
    ----------
    tilt_series : TiltSeries
        Tilt series whose `image_path` (and optionally `image_indices`)
        locate the images.
    device : torch.device | str | None
        Device for the returned tensor. Defaults to `tilt_series.device`.

    Returns
    -------
    images : torch.Tensor
        `(n_tilts, h, w)` float32 tilt images, ordered to match the tilt
        series (or the full stack if `image_indices` is None).

    Raises
    ------
    ValueError
        If `tilt_series.image_path` is not set.
    """
    import mrcfile

    if tilt_series.image_path is None:
        raise ValueError(
            "tilt_series.image_path is not set -> construct the TiltSeries "
            "via a torch_tilt_series loader (e.g. from_aretomo_output, "
            "from_etomo_directory) that resolves it, or load images yourself."
        )

    images = mrcfile.read(tilt_series.image_path)
    if tilt_series.image_indices is not None:
        images = images[tilt_series.image_indices.numpy()]

    if device is None:
        device = tilt_series.device
    return torch.as_tensor(_writable(images), device=device).float()
