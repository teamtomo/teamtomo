"""MTF functions for the Fourier filter.

Detector MTF curves for common cameras are bundled with the package (see
``data/mtf/README.md`` for provenance). They can be referenced by name, e.g.
``read_mtf("k3_standard_300kv_fl2")``; use :func:`list_mtf_references` to see
the available names.
"""

import os
from importlib.resources import files
from pathlib import Path

import starfile
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid

_MTF_DATA_DIR = Path(str(files("torch_fourier_filter").joinpath("data", "mtf")))

# Registry of bundled MTF star files keyed by lowercase name, derived from the
# file name without the ``mtf_`` prefix and ``.star`` suffix.
MTF_REFERENCES: dict[str, Path] = {
    path.stem.removeprefix("mtf_").lower(): path
    for path in sorted(_MTF_DATA_DIR.glob("mtf_*.star"))
}


def list_mtf_references() -> list[str]:
    """Return the names of the bundled detector MTF references."""
    return list(MTF_REFERENCES)


def get_mtf_reference_path(name: str) -> Path:
    """Return the path of a bundled detector MTF star file.

    Parameters
    ----------
    name : str
        Name of the MTF reference (case-insensitive), e.g. ``"k2_300kV"``. See
        :func:`list_mtf_references` for the available names.

    Returns
    -------
    Path
        Path to the bundled star file.

    Raises
    ------
    KeyError
        If ``name`` is not a bundled MTF reference.
    """
    key = name.lower()
    if key not in MTF_REFERENCES:
        raise KeyError(
            f"Unknown MTF reference {name!r}. "
            f"Available references: {list_mtf_references()}"
        )
    return MTF_REFERENCES[key]


def read_mtf(file_path: str | os.PathLike[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read the MTF from a star file or a bundled MTF reference.

    Args:
        file_path: The path to a star file with ``rlnResolutionInversePixel``
            and ``rlnMtfValue`` columns, or the name of a bundled MTF reference
            (case-insensitive, see :func:`list_mtf_references`). An existing
            file path takes precedence over a reference name.

    Returns
    -------
        The frequencies (0 to 0.5 at Nyquist) and MTF amps (MTF(0)=1).
    """
    if not os.path.exists(file_path) and str(file_path).lower() in MTF_REFERENCES:
        file_path = get_mtf_reference_path(str(file_path))
    df = starfile.read(file_path)
    frequencies = torch.tensor(df["rlnResolutionInversePixel"].to_numpy()).float()
    mtf_amplitudes = torch.tensor(df["rlnMtfValue"].to_numpy()).float()

    return frequencies, mtf_amplitudes


def make_mtf_grid(
    image_shape: tuple[int, int] | tuple[int, int, int],
    mtf_frequencies: torch.Tensor,  # 1D tensor
    mtf_amplitudes: torch.Tensor,  # 1D tensor
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Make a MTF grid from the frequencies and amplitudes.

    Args:
        image_shape: The shape of the image.
        pixel_size: The pixel size of the image.
        mtf_frequencies: The frequencies of the MTF.
        mtf_amplitudes: The amplitudes of the MTF.
        rfft: If the FFT is a real FFT.
        fftshift: If the FFT is shifted.
        device: The device to use for the calculation.

    Returns
    -------
        The MTF grid.
    """
    # Calculate the FFT frequency grid (normalized to physical units)
    fft_freqgrid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
        device=device,
    )  # Shape: (*image_shape), this return the frequency grid in range 0-0.5

    # Now map the MTF ampltiudes onto the corresponding frequencies
    # in the 2/3D fft_freq_px using linear interpolation

    # Clamp to the range of mtf_frequencies to avoid extrapolation issues
    fft_freq_mag_clamped = torch.clamp(
        fft_freqgrid, min=mtf_frequencies.min(), max=mtf_frequencies.max()
    )

    # Perform linear interpolation
    # Using torch.searchsorted to find bins
    indices = torch.searchsorted(mtf_frequencies, fft_freq_mag_clamped.flatten())
    indices = torch.clamp(indices, 1, len(mtf_frequencies) - 1)

    # Compute weights for linear interpolation
    x0 = mtf_frequencies[indices - 1]
    x1 = mtf_frequencies[indices]
    y0 = mtf_amplitudes[indices - 1]
    y1 = mtf_amplitudes[indices]
    weights = (fft_freq_mag_clamped.flatten() - x0) / (x1 - x0)

    # Interpolate amplitudes
    interpolated_amplitudes = y0 + weights * (y1 - y0)

    # Reshape to the original frequency grid shape
    mtf = interpolated_amplitudes.view(*fft_freq_mag_clamped.shape)

    return mtf
