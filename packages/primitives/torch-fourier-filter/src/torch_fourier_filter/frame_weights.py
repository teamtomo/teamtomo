"""Per-frame B-factor dose weighting and the matched template envelope.

Movie frames are weighted with RELION's per-frame model (Scheres, 2014)

    w_f(k) = exp(B_f k^2 / 4 + C_f)

with k the spatial frequency in 1/A, B_f each frame's B-factor and C_f its log
scale. At each frequency the weights are normalised by sqrt(sum_f w_f(k)^2), the
same convention as the Grant & Grigorieff (2015) weighting in
:mod:`torch_fourier_filter.dose_weight`, so the weighted sum keeps unit noise power
per frequency for frames of equal noise.

The per-frame values live in a :class:`FrameWeightTable`, indexed by exposure rather
than frame number, so a table can be carried to another fractionation
(:meth:`FrameWeightTable.resample`). Tables come from:

* :func:`default_frame_bfactors`: the default curve (``data/frame_weights``), for
  any exposure range and no particles;
* the user's own refit from particles (for example RIPPLE's estimator), which may
  move every frame.

A template searched against a weighted sum should carry the sum's signal envelope.
:class:`TemplateEnvelope` gives it: the sum's matched damage envelope
(:func:`matched_envelope`) times an optional smooth correction
(:class:`EnvelopeCorrection`, flat by default), as a 1D curve or as an
N-dimensional filter for a template volume.
"""

from __future__ import annotations

import functools
import json
import math
import warnings
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.interpolate import (  # type: ignore[import-untyped]
    CubicSpline,
    PchipInterpolator,
)
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_fourier_filter.utils import curve_1dim_to_ndim

_DEFAULT_CURVE = Path(
    str(
        files("torch_fourier_filter").joinpath(
            "data", "frame_weights", "default_frame_bfactors.json"
        )
    )
)
Shape = tuple[int] | tuple[int, int] | tuple[int, int, int]
_TABLE_SCHEMA = "torch-fourier-filter/frame-weight-table/1"
_ENVELOPE_SCHEMA = "torch-fourier-filter/template-envelope/1"


def frame_exposures(
    n_frames: int, pre_exposure: float = 0.0, dose_per_frame: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Start and end exposure of each frame for equal fractions, in e-/A^2.

    Parameters
    ----------
    n_frames : int
        Number of frames.
    pre_exposure : float, optional
        Exposure before the first frame. Default is 0.0.
    dose_per_frame : float, optional
        Exposure per frame. Default is 1.0.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(start, end)``, each of shape ``(n_frames,)``, float64.
    """
    start = pre_exposure + dose_per_frame * torch.arange(n_frames, dtype=torch.float64)
    return start, start + dose_per_frame


def _vector(x: Any, name: str) -> torch.Tensor:
    t = torch.as_tensor(
        np.asarray(x, dtype=np.float64) if not isinstance(x, torch.Tensor) else x
    )
    t = t.detach().to("cpu", torch.float64).reshape(-1)
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} must be finite")
    return t


@dataclass
class FrameWeightTable:
    """Per-frame B-factor and log scale, indexed by exposure.

    Parameters
    ----------
    exposure_start, exposure_end : torch.Tensor
        ``(n_frames,)`` exposure at the start and end of each frame, in e-/A^2.
    bfactor : torch.Tensor
        ``(n_frames,)`` per-frame B-factor B_f, in A^2 (negative: high resolution
        fades).
    log_scale : torch.Tensor
        ``(n_frames,)`` per-frame log scale C_f.
    voltage : float, optional
        Acceleration voltage in kV, recorded for provenance (the weights do not use
        it).
    source : str, optional
        Where the table came from, e.g. ``"default"`` or ``"estimated"``.
    provenance : dict, optional
        Free-form details: particles and micrographs used, fit diagnostics, errors.

    Only differences between frames matter: the weights are normalised at each
    frequency, so a B or C common to all frames cancels.
    """

    exposure_start: torch.Tensor
    exposure_end: torch.Tensor
    bfactor: torch.Tensor
    log_scale: torch.Tensor
    voltage: float | None = None
    source: str = "estimated"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Store the per-frame values as float64 CPU vectors and check them."""
        self.exposure_start = _vector(self.exposure_start, "exposure_start")
        self.exposure_end = _vector(self.exposure_end, "exposure_end")
        self.bfactor = _vector(self.bfactor, "bfactor")
        self.log_scale = _vector(self.log_scale, "log_scale")
        n = self.exposure_start.numel()
        if n == 0:
            raise ValueError("a frame weight table needs at least one frame")
        for name in ("exposure_end", "bfactor", "log_scale"):
            if getattr(self, name).numel() != n:
                raise ValueError(
                    f"{name} has {getattr(self, name).numel()} values for {n} frames"
                )
        if (self.exposure_end <= self.exposure_start).any():
            raise ValueError("every frame needs exposure_end > exposure_start")

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.exposure_start.numel())

    @property
    def exposure_mid(self) -> torch.Tensor:
        """Exposure at the middle of each frame."""
        return 0.5 * (self.exposure_start + self.exposure_end)

    def log_amplitudes(self, frequency: torch.Tensor) -> torch.Tensor:
        """``B_f k^2 / 4 + C_f``, ``(n_frames, *frequency.shape)``; k in 1/A."""
        k2 = frequency**2
        shape = (self.n_frames,) + (1,) * frequency.ndim
        B = self.bfactor.to(frequency.device, k2.dtype).reshape(shape)
        C = self.log_scale.to(frequency.device, k2.dtype).reshape(shape)
        return B * k2[None] / 4 + C

    def amplitudes(self, frequency: torch.Tensor) -> torch.Tensor:
        """Un-normalised amplitudes ``exp(B_f k^2 / 4 + C_f)``, one row per frame."""
        return torch.exp(self.log_amplitudes(frequency))

    def resample(self, exposure_start: Any, exposure_end: Any) -> FrameWeightTable:
        """The table at other frames (another fractionation of the same exposure).

        B and C are interpolated linearly in each frame's mid exposure, and held at
        the end values outside the table's range.
        """
        start, end = (
            _vector(exposure_start, "exposure_start"),
            _vector(exposure_end, "exposure_end"),
        )
        mid = (0.5 * (start + end)).numpy()
        x = self.exposure_mid.numpy()
        return FrameWeightTable(
            exposure_start=start,
            exposure_end=end,
            bfactor=torch.as_tensor(np.interp(mid, x, self.bfactor.numpy())),
            log_scale=torch.as_tensor(np.interp(mid, x, self.log_scale.numpy())),
            voltage=self.voltage,
            source=self.source,
            provenance={**self.provenance, "resampled_from_frames": self.n_frames},
        )

    def to_dict(self) -> dict:
        """JSON-ready dictionary."""
        return {
            "schema": _TABLE_SCHEMA,
            "source": self.source,
            "voltage": self.voltage,
            "exposure_start": self.exposure_start.tolist(),
            "exposure_end": self.exposure_end.tolist(),
            "bfactor": self.bfactor.tolist(),
            "log_scale": self.log_scale.tolist(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FrameWeightTable:
        """Inverse of :meth:`to_dict`."""
        return cls(
            exposure_start=d["exposure_start"],
            exposure_end=d["exposure_end"],
            bfactor=d["bfactor"],
            log_scale=d["log_scale"],
            voltage=d.get("voltage"),
            source=d.get("source", "estimated"),
            provenance=d.get("provenance", {}),
        )

    def to_json(self, path: str | Path) -> None:
        """Write the table as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> FrameWeightTable:
        """Read a table written by :meth:`to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@functools.lru_cache(maxsize=1)
def _default_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    d = json.loads(_DEFAULT_CURVE.read_text())
    return (
        np.asarray(d["knots_exposure"], float),
        np.asarray(d["bfactor"], float),
        np.asarray(d["log_scale"], float),
        d.get("provenance", {}),
    )


def _monotone_curve(
    knots: np.ndarray, values: np.ndarray, exposure: np.ndarray
) -> np.ndarray:
    """PCHIP through the knots; linear past the last knot; held at 0 below 0."""
    f = PchipInterpolator(knots, values, extrapolate=False)
    x = np.clip(exposure, knots[0], None)
    inside = f(np.clip(x, knots[0], knots[-1]))
    slope = (values[-1] - values[-2]) / (knots[-1] - knots[-2])
    out = np.where(x > knots[-1], values[-1] + slope * (x - knots[-1]), inside)
    return np.asarray(out, dtype=np.float64)


def default_bfactor_curve(exposure: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The default per-frame B-factor and log scale at the given exposures (e-/A^2).

    Parameters
    ----------
    exposure : array-like
        Exposure in e-/A^2 (physical; there is no voltage factor).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(bfactor, log_scale)``, float64, shaped like ``exposure``.
    """
    knots, b, c, _ = _default_curve()
    x = np.asarray(exposure, float)
    if np.any(x > knots[-1] + 1e-9):
        warnings.warn(
            f"exposure up to {x.max():.1f} e-/A^2 is past the default curve's "
            f"fitted range (0-{knots[-1]:g}); it is extended linearly",
            stacklevel=2,
        )
    return (
        torch.as_tensor(_monotone_curve(knots, b, x), dtype=torch.float64),
        torch.as_tensor(_monotone_curve(knots, c, x), dtype=torch.float64),
    )


def default_frame_bfactors(
    exposure_start: Any, exposure_end: Any, voltage: float | None = None
) -> FrameWeightTable:
    """The default frame weight table for the given frames.

    Each frame takes the default curve at its mid exposure. The curve
    (``data/frame_weights``) was fitted on yeast ribosome LSU in situ (300 kV) and
    single-particle microtubules (200 kV). It is a starting point for any dataset,
    and the fallback when no particles are available. ``voltage`` is only
    recorded: one curve in physical exposure fitted both 200 and 300 kV.

    Parameters
    ----------
    exposure_start, exposure_end : array-like
        ``(n_frames,)`` exposure at the start and end of each frame, in e-/A^2.
    voltage : float, optional
        Acceleration voltage in kV, for provenance.

    Returns
    -------
    FrameWeightTable
    """
    start, end = (
        _vector(exposure_start, "exposure_start"),
        _vector(exposure_end, "exposure_end"),
    )
    B, C = default_bfactor_curve((0.5 * (start + end)).numpy())
    return FrameWeightTable(
        exposure_start=start,
        exposure_end=end,
        bfactor=B,
        log_scale=C,
        voltage=voltage,
        source="default",
        provenance={"default_curve": _default_curve()[3]},
    )


def _frequency_grid(
    image_shape: Shape,
    pixel_size: float,
    rfft: bool,
    fftshift: bool,
    device: torch.device | None,
) -> torch.Tensor:
    """|k| in 1/A on the DFT grid of ``image_shape``."""
    return (
        fftfreq_grid(
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
            norm=True,
            device=device,
        )
        / pixel_size
    )


def frame_weights_normalization_grid(
    table: FrameWeightTable,
    image_shape: tuple[int, int],
    pixel_size: float,
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device | None = None,
    chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The frequency grid and the log normalisation per frequency, for chunks.

    Parameters
    ----------
    table : FrameWeightTable
        The movie's frame weights (one row per frame).
    image_shape : tuple[int, int]
        Real-space frame shape ``(h, w)``.
    pixel_size : float
        Pixel size in A.
    rfft, fftshift : bool, optional
        The layout of the frames' DFTs. Defaults: rfft, not shifted.
    device : torch.device, optional
        Device for the grids.
    chunk_size : int, optional
        Frames accumulated at a time. Default is 16.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(frequency, log_normalization)``, each shaped like the DFT grid: |k| in
        1/A and ``log sqrt(sum_f w_f(k)^2)``. Pass both to
        :func:`frame_weights_chunk`. The log form avoids underflow at high
        frequency, where every frame's amplitude can be below float32's range.
    """
    k = _frequency_grid(image_shape, pixel_size, rfft, fftshift, device)
    peak = torch.full_like(k, -math.inf)
    for s in range(0, table.n_frames, chunk_size):
        sub = _rows(table, s, min(s + chunk_size, table.n_frames))
        peak = torch.maximum(peak, sub.log_amplitudes(k).max(0).values)
    acc = torch.zeros_like(k)
    for s in range(0, table.n_frames, chunk_size):
        sub = _rows(table, s, min(s + chunk_size, table.n_frames))
        acc += torch.exp(2 * (sub.log_amplitudes(k) - peak[None])).sum(0)
    return k, peak + 0.5 * torch.log(acc)


def _rows(table: FrameWeightTable, start: int, end: int) -> FrameWeightTable:
    return FrameWeightTable(
        exposure_start=table.exposure_start[start:end],
        exposure_end=table.exposure_end[start:end],
        bfactor=table.bfactor[start:end],
        log_scale=table.log_scale[start:end],
        voltage=table.voltage,
        source=table.source,
    )


def frame_weights_chunk(
    chunk_dft: torch.Tensor,
    frame_start_idx: int,
    table: FrameWeightTable,
    frequency: torch.Tensor,
    log_normalization: torch.Tensor,
    in_place: bool = False,
) -> torch.Tensor:
    """Weight a contiguous chunk of a movie's frame DFTs with normalised weights.

    Parameters
    ----------
    chunk_dft : torch.Tensor
        ``(n_chunk, h, w)`` DFTs of frames ``frame_start_idx`` onwards.
    frame_start_idx : int
        Index of the chunk's first frame in the table.
    table : FrameWeightTable
        The movie's frame weights.
    frequency, log_normalization : torch.Tensor
        From :func:`frame_weights_normalization_grid`.
    in_place : bool, optional
        Multiply ``chunk_dft`` in place. Default is False.

    Returns
    -------
    torch.Tensor
        The weighted chunk.
    """
    n_chunk = chunk_dft.shape[0]
    if frame_start_idx < 0 or frame_start_idx + n_chunk > table.n_frames:
        raise ValueError(
            f"frames {frame_start_idx}-{frame_start_idx + n_chunk - 1} are outside "
            f"the table's {table.n_frames}"
        )
    sub = _rows(table, frame_start_idx, frame_start_idx + n_chunk)
    k = frequency.to(chunk_dft.device)
    log_norm = log_normalization.to(chunk_dft.device)
    w = torch.exp(sub.log_amplitudes(k) - log_norm[None])
    if in_place:
        chunk_dft *= w
        return chunk_dft
    return chunk_dft * w


def frame_weights_grid(
    table: FrameWeightTable,
    image_shape: tuple[int, int],
    pixel_size: float,
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Normalised per-frame weights on a DFT grid, with ``sum_f w_f(k)^2 = 1``.

    Parameters
    ----------
    table : FrameWeightTable
        The movie's frame weights (one row per frame).
    image_shape : tuple[int, int]
        Real-space frame shape ``(h, w)``.
    pixel_size : float
        Pixel size in A.
    rfft, fftshift : bool, optional
        The grid layout. Defaults: rfft, not shifted.
    device : torch.device, optional
        Device for the weights.

    Returns
    -------
    torch.Tensor
        ``(n_frames, *grid)``. Multiply each frame's DFT by its row and sum over
        frames for the weighted image.
    """
    k, log_norm = frame_weights_normalization_grid(
        table, image_shape, pixel_size, rfft, fftshift, device
    )
    return torch.exp(table.log_amplitudes(k) - log_norm[None])


def matched_envelope(
    frequency: torch.Tensor,
    sum_table: FrameWeightTable,
    damage_table: FrameWeightTable | None = None,
) -> torch.Tensor:
    """The signal envelope of a frame-weighted sum, normalised to E(0) = 1.

    ``E(k) = sum_f W_f a_f / sqrt(sum_f W_f^2)``, with W_f the sum's weights
    (``sum_table``) and a_f the frames' signal amplitudes under the damage
    (``damage_table``; by default the same table, when E reduces to
    ``sqrt(sum_f a_f^2)``). A template whose envelope matches the sum's is the
    matched filter for that sum.

    Parameters
    ----------
    frequency : torch.Tensor
        Spatial frequencies in 1/A, any shape.
    sum_table : FrameWeightTable
        The frame weights the sum was (or will be) made with.
    damage_table : FrameWeightTable, optional
        The frames' damage; resampled to ``sum_table``'s frames when they differ.
        Default: ``sum_table``.

    Returns
    -------
    torch.Tensor
        The envelope, shaped like ``frequency``.
    """
    if damage_table is None:
        damage_table = sum_table
    elif damage_table.n_frames != sum_table.n_frames or not torch.allclose(
        damage_table.exposure_mid, sum_table.exposure_mid
    ):
        damage_table = damage_table.resample(
            sum_table.exposure_start, sum_table.exposure_end
        )
    k = torch.cat(
        [
            torch.zeros(1, dtype=frequency.dtype, device=frequency.device),
            frequency.reshape(-1),
        ]
    )
    lw = sum_table.log_amplitudes(k)
    la = damage_table.log_amplitudes(k)
    m = (lw + la).max(0).values
    mw = lw.max(0).values
    log_e = (
        m
        + torch.log(torch.exp(lw + la - m).sum(0))
        - mw
        - 0.5 * torch.log(torch.exp(2 * (lw - mw)).sum(0))
    )
    return torch.exp(log_e[1:] - log_e[0]).reshape(frequency.shape)


@dataclass
class EnvelopeCorrection:
    """A smooth correction to a template envelope.

    A natural cubic spline of log h in sqrt(k) through the knots, flat outside them.

    Parameters
    ----------
    knot_frequency : torch.Tensor
        ``(n_knots,)`` increasing knot frequencies in 1/A.
    log_values : torch.Tensor
        ``(n_knots,)`` log h at the knots. Only the shape matters for 2DTM, where
        templates are normalised.
    """

    knot_frequency: torch.Tensor
    log_values: torch.Tensor

    def __post_init__(self) -> None:
        """Store the knots as float64 CPU vectors and check them."""
        self.knot_frequency = _vector(self.knot_frequency, "knot_frequency")
        self.log_values = _vector(self.log_values, "log_values")
        if self.knot_frequency.numel() != self.log_values.numel():
            raise ValueError("knot_frequency and log_values differ in length")
        if (
            self.knot_frequency.numel() < 2
            or (torch.diff(self.knot_frequency) <= 0).any()
        ):
            raise ValueError(
                "knot_frequency needs at least two increasing, positive values"
            )
        if (self.knot_frequency <= 0).any():
            raise ValueError("knot_frequency must be positive")

    @classmethod
    def flat(
        cls,
        n_knots: int = 5,
        low_resolution: float = 50.0,
        high_resolution: float = 2.9,
    ) -> EnvelopeCorrection:
        """No correction (h = 1), on :meth:`uniform_knots`' knots."""
        return cls.uniform_knots(n_knots, low_resolution, high_resolution)

    @classmethod
    def uniform_knots(
        cls,
        n_knots: int = 5,
        low_resolution: float = 50.0,
        high_resolution: float = 2.9,
        log_values: Any = None,
    ) -> EnvelopeCorrection:
        """Knots uniform in sqrt(k) between two resolutions (A); log values 0."""
        s = np.linspace(
            math.sqrt(1 / low_resolution), math.sqrt(1 / high_resolution), n_knots
        )
        lv = np.zeros(n_knots) if log_values is None else np.asarray(log_values)
        return cls(knot_frequency=torch.as_tensor(s**2), log_values=torch.as_tensor(lv))

    def __call__(self, frequency: torch.Tensor) -> torch.Tensor:
        """h(k), shaped like ``frequency`` (1/A)."""
        s_knots = np.sqrt(self.knot_frequency.numpy())
        lv = self.log_values.numpy()
        if np.allclose(lv, lv[0]):
            return torch.full_like(frequency, float(np.exp(lv[0])))
        spline = CubicSpline(s_knots, lv, bc_type="natural")
        s = np.sqrt(np.clip(frequency.detach().cpu().double().numpy(), 0, None))
        h = np.exp(spline(np.clip(s, s_knots[0], s_knots[-1])))
        return torch.as_tensor(h, dtype=frequency.dtype, device=frequency.device)

    def to_dict(self) -> dict:
        """JSON-ready dictionary."""
        return {
            "knot_frequency": self.knot_frequency.tolist(),
            "log_values": self.log_values.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> EnvelopeCorrection:
        """Inverse of :meth:`to_dict`."""
        return cls(knot_frequency=d["knot_frequency"], log_values=d["log_values"])


@dataclass
class TemplateEnvelope:
    """A template's radial envelope matched to a weighted sum.

    :func:`matched_envelope` of the sum's frame weights, times an optional
    correction.

    Parameters
    ----------
    frame_weights : FrameWeightTable
        The frame weights of the sums the template will be searched against.
    damage : FrameWeightTable, optional
        The frames' damage, if different from ``frame_weights`` (for example a G&G
        sum with measured damage).
    correction : EnvelopeCorrection, optional
        A smooth correction fitted from particles; ``None`` is flat (the default
        with no particles).
    provenance : dict, optional
        Free-form details of the fit.
    """

    frame_weights: FrameWeightTable
    damage: FrameWeightTable | None = None
    correction: EnvelopeCorrection | None = None
    provenance: dict = field(default_factory=dict)

    @classmethod
    def default(
        cls, dose_start: float, dose_end: float, dose_per_frame: float = 1.0
    ) -> TemplateEnvelope:
        """Default frame weights over ``[dose_start, dose_end]``, no correction."""
        n = max(1, round((dose_end - dose_start) / dose_per_frame))
        start, end = frame_exposures(n, dose_start, (dose_end - dose_start) / n)
        return cls(frame_weights=default_frame_bfactors(start, end))

    def __call__(self, frequency: torch.Tensor) -> torch.Tensor:
        """The envelope h(k), shaped like ``frequency`` (1/A)."""
        h = matched_envelope(frequency, self.frame_weights, self.damage)
        if self.correction is not None:
            h = h * self.correction(frequency)
        return h

    def grid(
        self,
        shape: Shape,
        pixel_size: float,
        rfft: bool = True,
        fftshift: bool = False,
        device: torch.device | None = None,
        n_samples: int = 2048,
    ) -> torch.Tensor:
        """The envelope on the DFT grid of a 2D image or 3D volume.

        Multiply the template's DFT by it. The curve is sampled at ``n_samples``
        frequencies up to the grid's corner and interpolated onto the grid.
        """
        k = _frequency_grid(shape, pixel_size, rfft, fftshift, device)
        k1 = torch.linspace(
            0, float(k.max()), n_samples, dtype=k.dtype, device=k.device
        )
        return curve_1dim_to_ndim(self(k1), k1, k)

    def to_dict(self) -> dict:
        """JSON-ready dictionary."""
        return {
            "schema": _ENVELOPE_SCHEMA,
            "frame_weights": self.frame_weights.to_dict(),
            "damage": None if self.damage is None else self.damage.to_dict(),
            "correction": None
            if self.correction is None
            else self.correction.to_dict(),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TemplateEnvelope:
        """Inverse of :meth:`to_dict`."""
        return cls(
            frame_weights=FrameWeightTable.from_dict(d["frame_weights"]),
            damage=None
            if d.get("damage") is None
            else FrameWeightTable.from_dict(d["damage"]),
            correction=None
            if d.get("correction") is None
            else EnvelopeCorrection.from_dict(d["correction"]),
            provenance=d.get("provenance", {}),
        )

    def to_json(self, path: str | Path) -> None:
        """Write the envelope as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def from_json(cls, path: str | Path) -> TemplateEnvelope:
        """Read an envelope written by :meth:`to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text()))
