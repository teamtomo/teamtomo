"""Pydantic models for CTF estimation results (1D, 2D, defocus/phase shift)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_serializer
from torch_cubic_spline_grids import CubicBSplineGrid1d, CubicCatmullRomGrid3d

if TYPE_CHECKING:
    from pydantic.functional_serializers import SerializerFunctionWrapHandler


@dataclass
class _Background1DResult:
    """Background fit result: model and background-subtracted 1D power spectrum."""

    rotationally_averaged_power_spectrum: torch.Tensor
    freqs: torch.Tensor
    spatial_freqs: torch.Tensor
    fit_mask: torch.Tensor
    raps_in_fit_range: torch.Tensor  # background-subtracted 1D power in the fit band
    background_model: CubicBSplineGrid1d | None


@dataclass
class _GridSearch1DResult:
    """Result of grid search over defocus and optional B-factor and phase shift."""

    best_defocus: torch.Tensor
    best_B: torch.Tensor | None
    test_defoci: torch.Tensor
    cross_correlations_1d: torch.Tensor
    cross_correlations_2d: torch.Tensor | None
    test_B_values: torch.Tensor | None
    best_phase_shift: torch.Tensor | None = None
    test_phase_shift_values: torch.Tensor | None = None


class CTF(BaseModel):
    """CTF model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defocus_um: torch.Tensor
    voltage_kev: torch.Tensor
    spherical_aberration_mm: torch.Tensor
    amplitude_contrast_fraction: torch.Tensor
    phase_shift_degrees: torch.Tensor
    envelope_B: torch.Tensor | None = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return handler(value)


class LinearDefocusModel(BaseModel):
    """Linear (tilt) defocus: defocus_0 + gradient_magnitude * direction in (x,y)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    defocus_0: float
    defocus_gradient_magnitude: float
    defocus_gradient_angle: float  # degrees
    # When t>1, optional 1D spline grid data for serialization
    defocus_0_spline_data: torch.Tensor | None = None
    gradient_magnitude_spline_data: torch.Tensor | None = None
    angle_u_spline_data: torch.Tensor | None = None
    angle_v_spline_data: torch.Tensor | None = None

    @property
    def data(self) -> torch.Tensor:
        """
        Summary tensor for compatibility with grid model API (defocus_model.data).

        - Scalar model: shape (3,) [defocus_0, gradient_magnitude, gradient_angle].
        - Spline model (t>1): shape (4, nt) with defocus_0, grad_mag, angle_u, angle_v
          per time knot.
        """
        if self.defocus_0_spline_data is not None:
            return torch.stack(
                [
                    self.defocus_0_spline_data,
                    self.gradient_magnitude_spline_data,
                    self.angle_u_spline_data,
                    self.angle_v_spline_data,
                ]
            )
        return torch.tensor(
            [
                self.defocus_0,
                self.defocus_gradient_magnitude,
                self.defocus_gradient_angle,
            ],
            dtype=torch.float32,
        )

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        return handler(value)


class QuadraticPhaseShiftModel(BaseModel):
    """
    Quadratic phase shift f(x,y) = C + g1*s + k1*s^2 + g2*t + k2*t^2.

    s = x*cos(alpha) + y*sin(alpha), t = -x*sin(alpha) + y*cos(alpha). (x,y) in [-1,1].
    Six parameters: C, alpha_rad, g1, k1, g2, k2.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    C: float  # mean phase shift at origin (degrees)
    alpha_rad: float  # orientation of s-axis (radians)
    g1: float  # linear term along s
    k1: float  # quadratic term along s
    g2: float  # linear term along t (perpendicular)
    k2: float  # quadratic term along t


class Defocus2DResults(BaseModel):
    """2D defocus result: defocus model (grid or linear) and optional traces.

    Defocus, astigmatism and defocus gradients are in µm; angles are in degrees.
    Fields marked "debug only" are populated by the 2D fitters only when they
    are called with `debug=True` and are None otherwise.

    Attributes
    ----------
    cross_correlation_final : float or None
        Mean Pearson r between patch power and simulated CTF² per time frame at
        the final parameters (heuristic fit reliability). Not comparable to the
        1D cross-correlation. None when the result was not produced by a 2D fit
        (e.g. `defocus_field_from_1d_fits`).
    defocus_model_type : {"grid", "linear"}
        Which defocus model `defocus_model` holds. Default "grid".
    defocus_model : CubicCatmullRomGrid3d or LinearDefocusModel
        Fitted defocus model. A grid model maps normalised patch positions
        `(t, y, x)` in [0, 1] to defocus in µm; a linear model stores a mean
        defocus plus a defocus gradient (magnitude in µm per normalised image
        unit, angle in degrees).
    patch_power_spectra : torch.Tensor or None
        Patch power spectra used for the fit, shape `(t, gh, gw, ph, pw // 2 + 1)`
        (debug only, also set by `defocus_field_from_1d_fits`).
    model_trace : list[torch.Tensor] or None
        Defocus model parameters after each optimisation step (debug only). For
        the grid model, a copy of the grid data; for the linear model, a tensor
        `[defocus_0, gradient_magnitude, gradient_angle_deg, 0]`.
    simulated_ctf2s : torch.Tensor or None
        Simulated CTF² (or |CTF| when fitting amplitudes) for the last frame at
        the last iteration, in rFFT patch layout (debug only).
    astigmatism : float or None
        Astigmatism magnitude in µm (difference between `defocus_u` and
        `defocus_v`).
    astigmatism_angle : float or None
        Astigmatism angle in degrees, in [-90, 90].
    astigmatism_trace : list[float] or None
        Astigmatism after each step (debug only, and only when astigmatism is
        optimised).
    astigmatism_angle_trace : list[float] or None
        Astigmatism angle after each step (debug only, and only when astigmatism
        is optimised).
    envelope_B : float or None
        Envelope B-factor in Å² used for the 2D fit.
    envelope_B_trace : list[float] or None
        Per-step envelope B-factor trace. Not populated by the current fitters.
    loss_trace : list[float] or None
        Mean loss over frames after each step (debug only).
    defocus_u : float or None
        Highest principal defocus in µm: mean defocus + astigmatism / 2.
    defocus_v : float or None
        Lowest principal defocus in µm: mean defocus - astigmatism / 2.
    tilt_axis_angle_deg : float or None
        For the linear model, the defocus gradient direction in degrees. Set by
        `estimate_ctf` / `estimate_ctf_and_thickness`; None otherwise.
    tilt_magnitude_deg : float or None
        For the linear model, the specimen tilt in degrees implied by the defocus
        gradient (see `linear_tilt_axis_and_magnitude_deg`). Set by
        `estimate_ctf` / `estimate_ctf_and_thickness`; None otherwise.
    phase_shift_degrees : float or None
        Mean (or fixed) phase shift in degrees.
    phase_shift_model_type : {"grid", "quadratic"} or None
        Spatial phase shift model type, or None if the phase shift was not
        optimised.
    phase_shift_model : CubicCatmullRomGrid3d, QuadraticPhaseShiftModel, tuple or None
        Fitted phase shift model: a `(u_grid, v_grid)` pair of
        `CubicCatmullRomGrid3d` for the grid model, a `QuadraticPhaseShiftModel`
        for the quadratic model, or None if the phase shift was not optimised.
    phase_shift_trace : list[float] or None
        Phase shift in degrees after each step, when the phase shift is optimised.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cross_correlation_final: float | None = Field(
        default=None,
        description=(
            "Mean Pearson r between patch power and simulated CTF^2 per time frame "
            "at the final parameters (heuristic fit reliability). "
            "Not comparable to 1D CC."
        ),
    )

    defocus_model_type: Literal["grid", "linear"] = "grid"
    defocus_model: CubicCatmullRomGrid3d | LinearDefocusModel
    patch_power_spectra: torch.Tensor | None = None
    model_trace: list[torch.Tensor] | None = None
    simulated_ctf2s: torch.Tensor | None = None
    astigmatism: float | None = None
    astigmatism_angle: float | None = None
    astigmatism_trace: list[float] | None = None
    astigmatism_angle_trace: list[float] | None = None
    envelope_B: float | None = None
    envelope_B_trace: list[float] | None = None
    loss_trace: list[float] | None = None
    defocus_u: float | None = None  # highest principal defocus (defocus + astig/2)
    defocus_v: float | None = None  # lowest principal defocus (defocus - astig/2)
    tilt_axis_angle_deg: float | None = None
    tilt_magnitude_deg: float | None = None
    phase_shift_degrees: float | None = None
    phase_shift_model_type: Literal["grid", "quadratic"] | None = None
    phase_shift_model: (
        CubicCatmullRomGrid3d
        | QuadraticPhaseShiftModel
        | tuple[CubicCatmullRomGrid3d, CubicCatmullRomGrid3d]
        | None
    ) = None
    phase_shift_trace: list[float] | None = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, CubicCatmullRomGrid3d):
            return value.to_dict()
        if isinstance(value, LinearDefocusModel):
            return value.model_dump()
        if isinstance(value, QuadraticPhaseShiftModel):
            return value.model_dump()
        if isinstance(value, tuple) and len(value) == 2:
            a, b = value
            if isinstance(a, CubicCatmullRomGrid3d) and isinstance(
                b, CubicCatmullRomGrid3d
            ):
                return {"u": a.to_dict(), "v": b.to_dict()}
        return handler(value)

    def linear_tilt_axis_and_magnitude_deg(
        self,
        pixel_spacing_angstroms: float,
        image_size_pixels: int,
    ) -> tuple[float, float]:
        """
        For linear defocus model: (tilt_axis_angle_deg, tilt_magnitude_deg).

        Tilt axis angle is the defocus gradient direction (degrees).
        Tilt magnitude (degrees) from defocus gradient (um per normalized unit)
        and pixel size: arctan(gradient in µm/µm) with gradient in µm/µm
        = grad_mag*1e4 / (image_size_pixels * pixel_spacing_angstroms).

        Parameters
        ----------
        pixel_spacing_angstroms : float
            Pixel spacing in Å of the image the defocus model was fitted on.
        image_size_pixels : int
            Image side length in pixels; one normalised unit spans
            `image_size_pixels * pixel_spacing_angstroms` Å.

        Returns
        -------
        tuple[float, float]
            `(tilt_axis_angle_deg, tilt_magnitude_deg)`, or `(0.0, 0.0)` if
            `defocus_model_type` is not "linear".
        """
        return linear_tilt_axis_and_magnitude_deg(
            self, pixel_spacing_angstroms, image_size_pixels
        )


def linear_tilt_axis_and_magnitude_deg(
    result2d: Defocus2DResults,
    pixel_spacing_angstroms: float,
    image_size_pixels: int,
) -> tuple[float, float]:
    """
    Tilt axis angle (degrees) and tilt magnitude (degrees) from a linear defocus result.

    The defocus gradient is in um per normalized unit (0-1 across the image).
    Tilt magnitude (deg) = arctan(gradient in um/um) * 180/pi, with
    gradient in µm/µm = defocus_gradient_magnitude * 1e4
    / (image_size_pixels * pixel_spacing_angstroms)
    (1 normalized unit = image_size_pixels * pixel_spacing_angstroms A).

    Returns (tilt_axis_angle_deg, tilt_magnitude_deg). If defocus_model_type is not
    "linear", returns (0.0, 0.0).
    """
    if result2d.defocus_model_type != "linear":
        return (0.0, 0.0)
    lm = result2d.defocus_model
    grad_mag = lm.defocus_gradient_magnitude  # microns per normalized unit
    axis_deg = lm.defocus_gradient_angle
    grad_per_micron = (grad_mag * 1e4) / (image_size_pixels * pixel_spacing_angstroms)
    tilt_rad = math.atan(grad_per_micron)
    tilt_deg = tilt_rad * 180.0 / math.pi
    return (axis_deg, tilt_deg)


class Thickness1DResults(BaseModel):
    """Results from 1D thickness estimation.

    Attributes
    ----------
    thickness_angstroms : float
        Best-fit sample thickness in Å.
    cross_correlation_final : float or None
        L2-normalised cross-correlation (cosine similarity) of the
        background-subtracted 1D power against the thickness-modulated power
        spectrum on the fit band, at the best thickness.
    frequencies_1d : torch.Tensor
        Spatial frequency (1/Å) of each 1D sample, shape `(n // 2 + 1,)` for an
        `n`-pixel power spectrum.
    powerspectrum_1d : torch.Tensor or None
        Rotationally (or equiphase) averaged power spectrum over the full
        frequency range, before background subtraction; same shape as
        `frequencies_1d`.
    background_model : CubicBSplineGrid1d or None
        Spline fitted to log power over the fit band, with the band mapped to
        [0, 1].
    test_thicknesses : torch.Tensor or None
        Thicknesses (Å) evaluated in the grid search, shape `(n_thicknesses,)`.
        None for gradient-descent refinement results.
    cross_correlations : torch.Tensor or None
        Cross-correlation for each entry of `test_thicknesses`.
    low_frequency_fit : float or None
        Low end of the fit band as a spatial frequency (1/Å).
    high_frequency_fit : float or None
        High end of the fit band as a spatial frequency (1/Å).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thickness_angstroms: float = Field(
        description="Best-fit sample thickness in Angstroms."
    )
    cross_correlation_final: float | None = Field(
        default=None,
        description=(
            "L2 NCC (cosine similarity) of background-subtracted 1D power vs "
            "thickness-modulated power spectrum on the fit band at best thickness."
        ),
    )
    frequencies_1d: torch.Tensor
    powerspectrum_1d: torch.Tensor | None = None
    background_model: CubicBSplineGrid1d | None = None
    test_thicknesses: torch.Tensor | None = None
    cross_correlations: torch.Tensor | None = None
    low_frequency_fit: float | None = None
    high_frequency_fit: float | None = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, CubicBSplineGrid1d):
            return value.to_dict()
        return handler(value)


class Thickness2DResults(BaseModel):
    """Results from 2D thickness estimation (gradient descent on spline grid).

    Attributes
    ----------
    mean_thickness : float
        Mean thickness in Å across the spline grid.
    cross_correlation_final : float or None
        Mean Pearson r between patch power and simulated thickness power spectrum
        per time frame at the final parameters (heuristic fit reliability).
    thickness_model : CubicCatmullRomGrid3d
        Thickness spline (Å) over normalised `(t, y, x)` positions in [0, 1].
    patch_power_spectra : torch.Tensor or None
        Patch power spectra used for the fit (only with `debug=True`).
    model_trace : list[torch.Tensor] or None
        Thickness grid data after each step (only with `debug=True`).
    simulated_ps : torch.Tensor or None
        Simulated power spectrum from the last iteration (only with
        `debug=True`).
    envelope_B : float or None
        Envelope B-factor in Å² used for the fit.
    envelope_B_trace : list[float] or None
        Per-step envelope B-factor trace. Not populated by the current fitters.
    loss_trace : list[float] or None
        Loss after each optimisation step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean_thickness: float = Field(
        description="Mean thickness in Angstroms across the spline grid."
    )
    cross_correlation_final: float | None = Field(
        default=None,
        description=(
            "Mean Pearson r between patch power and simulated thickness power spectrum "
            "per time frame at the final parameters (heuristic fit reliability)."
        ),
    )
    thickness_model: CubicCatmullRomGrid3d
    patch_power_spectra: torch.Tensor | None = None
    model_trace: list[torch.Tensor] | None = None
    simulated_ps: torch.Tensor | None = None
    envelope_B: float | None = None
    envelope_B_trace: list[float] | None = None
    loss_trace: list[float] | None = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, CubicCatmullRomGrid3d):
            return value.to_dict()
        return handler(value)


class Defocus1DResults(BaseModel):
    """Results from 1D defocus estimation.

    Attributes
    ----------
    cross_correlation_final : float or None
        L2-normalised cross-correlation (cosine similarity) of the
        background-subtracted 1D power against CTF² times envelope on the fit
        band at the final parameters (the 1D objective). Not comparable to the
        2D cross-correlation.
    frequencies_1d : torch.Tensor
        Spatial frequency (1/Å) of each 1D sample, shape `(n // 2 + 1,)` for an
        `n`-pixel power spectrum.
    powerspectrum_1d : torch.Tensor or None
        Rotationally (or equiphase) averaged power spectrum over the full
        frequency range, before background subtraction; same shape as
        `frequencies_1d`.
    background_model : CubicBSplineGrid1d or None
        Spline fitted to log power over the fit band, with the band mapped to
        [0, 1].
    test_defoci : torch.Tensor or None
        Defocus values (µm) evaluated in the grid search, shape `(n_defoci,)`.
        None when the grid search was skipped (`initial_defocus` given) or for
        joint defocus/thickness refinement results.
    cross_correlations : torch.Tensor or None
        Best cross-correlation for each entry of `test_defoci` (maximised over
        the other searched parameters).
    ctf_model : CTF
        Final CTF parameters: defocus (µm), voltage (keV), spherical aberration
        (mm), amplitude contrast fraction, phase shift (degrees) and optional
        envelope B-factor (Å²).
    low_frequency_fit : float or None
        Low end of the fit band as a spatial frequency (1/Å).
    high_frequency_fit : float or None
        High end of the fit band as a spatial frequency (1/Å).
    envelope_B : torch.Tensor or None
        Refined envelope B-factor in Å², or None if the envelope was not
        optimised.
    test_B_values : torch.Tensor or None
        B-factors (Å²) evaluated in the grid search, or None if the envelope was
        not optimised.
    cross_correlations_2d : torch.Tensor or None
        Cross-correlation over `(test_defoci, test_B_values)`, shape
        `(n_defoci, n_B)` (maximised over phase shift when that is also
        searched), or None if the envelope was not optimised.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cross_correlation_final: float | None = Field(
        default=None,
        description=(
            "L2 NCC (cosine similarity) of background-subtracted 1D power vs "
            "CTF^2 times envelope on the fit band at final parameters "
            "(same as 1D objective). Not comparable to 2D CC."
        ),
    )

    frequencies_1d: torch.Tensor
    powerspectrum_1d: torch.Tensor | None = None
    background_model: CubicBSplineGrid1d | None = None
    test_defoci: torch.Tensor | None = None
    cross_correlations: torch.Tensor | None = None
    ctf_model: CTF
    low_frequency_fit: float | None = None
    high_frequency_fit: float | None = None
    envelope_B: torch.Tensor | None = None
    test_B_values: torch.Tensor | None = None
    cross_correlations_2d: torch.Tensor | None = None

    @field_serializer("*", mode="wrap")  # type: ignore[misc]
    def _serialize_field(
        self, value: Any, handler: SerializerFunctionWrapHandler
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, CubicBSplineGrid1d):
            return value.to_dict()
        return handler(value)
