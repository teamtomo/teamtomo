"""Pydantic models for CTF estimation inputs (optics, fitting, laser)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable


class OpticalParams(BaseModel):
    """Optical parameters: pixel spacing, voltage, Cs, amplitude contrast."""

    pixel_spacing_angstroms: float
    voltage_kev: float = 300.0
    spherical_aberration_mm: float = 2.7
    amplitude_contrast_fraction: float = 0.07
    target_pixel_spacing_angstroms: float = Field(
        default=3.0,
        description=(
            "Internal fitting pixel spacing in Å. Images are Fourier-rescaled "
            "to max(target, source) and are never upsampled."
        ),
    )


class _EarlyStoppingMixin(BaseModel):
    """Plateau-style early stopping for Adam loops. Off by default."""

    early_stopping: bool = False
    early_stopping_patience: int = Field(default=5, ge=1)
    early_stopping_window_size: int = Field(default=3, ge=1)
    early_stopping_tolerance: float = 1e-5

    def build_early_stopper(self) -> Callable[[float], bool] | None:
        """Return a stateful early-stopping callable, or None if disabled."""
        if not self.early_stopping:
            return None
        from torch_ctf_estimation.utils.early_stopping import make_early_stopper

        return make_early_stopper(
            patience=self.early_stopping_patience,
            window_size=self.early_stopping_window_size,
            tolerance=self.early_stopping_tolerance,
        )


class CTFFittingParams(_EarlyStoppingMixin):
    """CTF fitting parameters (defocus grid, frequency range, patch size, etc.)."""

    defocus_grid_resolution: tuple[int, int, int]
    frequency_fit_range_angstroms: tuple[float, float]
    defocus_range_microns: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Defocus bounds in µm for 1D/2D fitting. Default (0, 10) when unset. "
            "Equal values, e.g. (2.5, 2.5), fix defocus without optimising it."
        ),
    )
    phase_shift_range_degrees: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Phase shift bounds in degrees for 1D/2D fitting. Default (0, 180) when unset. "
            "Equal values, e.g. (45.0, 45.0), use a known fixed phase (overrides optimize_phase_shift)."
        ),
    )
    patch_sidelength: int = 256
    debug: bool = False
    optimize_astigmatism: bool = True
    defocus_model: Literal["grid", "linear"] = "grid"
    use_1d_defocus_for_spatial: bool = False
    use_equiphase_for_1d_spatial: bool = True
    equiphase_n_theta: int = 64
    linear_fix_defocus_0_from_1x1: bool = False
    refine_steps_1d: int = 40
    n_iterations_2d: int = 100
    optimize_envelope_1d: bool = True
    b_range_1d: tuple[float, float] = (0.0, 200.0)
    b_step_1d: float = 5.0
    initial_envelope_B: float | None = None
    optimize_phase_shift: bool = False
    phase_shift_model: Literal["grid", "quadratic"] = "grid"
    phase_shift_quadratic_perpendicular_axis: bool = False
    initial_phase_shift: float = 0.0
    mask_laser_axis: bool = False
    laser_axis_mask_width: float = 0.1
    use_amplitude_2d: bool = Field(
        default=False,
        description=(
            "If True, 2D defocus/astigmatism fits sqrt(patch power) against |CTF| "
            "instead of power against CTF². 1D defocus and thickness stay on power."
        ),
    )


class LaserParams(BaseModel):
    """Laser phase plate parameters for optics groups using a laser phase plate.

    Pass this block when you need laser geometry (e.g. axis masking) and/or the
    LPP CTF model. Set ``model_laser=True`` to use the LPP CTF; with
    ``model_laser=False`` a standard CTF is used but ``laser_xy_angle_deg`` and
    ``dual_laser`` still apply when ``mask_laser_axis`` is enabled in
    ``CTFFittingParams``.

    Attributes
    ----------
    model_laser : bool
        If True, use the LPP CTF model for fitting. If False, use the standard
        CTF while still allowing laser-axis masking via ``laser_xy_angle_deg`` and
        ``dual_laser``. Default is False.
    NA : float
        Numerical aperture.
    laser_wavelength_angstrom : float
        Laser wavelength in Angstrom.
    focal_length_angstrom : float
        Focal length in Angstrom.
    laser_xy_angle_deg : float
        Laser angle in the XY plane in degrees.
    laser_xz_angle_deg : float
        Laser angle in the XZ plane in degrees.
    laser_long_offset_angstrom : float
        Longitudinal offset in Angstrom.
    laser_trans_offset_angstrom : float
        Transverse offset in Angstrom.
    laser_polarization_angle_deg : float
        Laser polarization angle in degrees.
    peak_phase_deg : float
        Peak phase in degrees.
    dual_laser : bool
        Whether a dual-laser setup is used. Default is False.
    """

    model_laser: bool = False
    NA: float = 0.055
    laser_wavelength_angstrom: float = 10640.0
    focal_length_angstrom: float = 6.8e7
    laser_xy_angle_deg: float = 0.0
    laser_xz_angle_deg: float = 0.0
    laser_long_offset_angstrom: float = 0.0
    laser_trans_offset_angstrom: float = 0.0
    laser_polarization_angle_deg: float = 90.0
    peak_phase_deg: float = 45.0
    dual_laser: bool = True


class ThicknessParams(_EarlyStoppingMixin):
    """Optional thickness estimation and optional refine after the 1D grid.

    Standalone 1D thickness is a grid search. Gradient descent is used only
    if ``refine_dim`` is not ``"none"``. ``"thickness"`` refines thickness
    only (defocus stays at the 2D mean). ``"1d"`` jointly refines defocus
    and thickness. ``"2d"`` jointly refines the defocus field and thickness.
    """

    refine_dim: Literal["none", "thickness", "1d", "2d"] = "2d"
    thickness_range_angstroms: tuple[float, float] = (300.0, 4000.0)
    thickness_step_angstroms: float = 100.0
    thickness_grid_resolution: tuple[int, int, int] | None = Field(
        default=None,
        description=(
            "Thickness spline resolution (nt, nh, nw) for 2D joint refine. "
            "If None, a scalar (1, 1, 1) thickness is used."
        ),
    )
    n_iterations: int = 100
    defocus_lr: float = 0.01
    thickness_lr: float = 50.0
    use_tilt_corrected_ps: bool = Field(
        default=False,
        description=(
            "If True, 1D thickness (grid and joint 1D) uses a CTFFind5-style "
            "tilt-corrected mean power spectrum from the 2D defocus field."
        ),
    )
    use_equiphase: bool = Field(
        default=False,
        description=(
            "If True, 1D thickness (grid and joint 1D) uses equiphase averaging "
            "so astigmatism does not smear Thon rings in the 1D profile."
        ),
    )
    frequency_fit_range_angstroms: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Fit band (low, high) in Å for thickness (1D grid, joint 1D, joint 2D). "
            "If None, fitting_config.frequency_fit_range_angstroms is used."
        ),
    )
    defocus_range_microns: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Defocus clamp (µm) for joint thickness refine. "
            "If None, fitting_config.defocus_range_microns is used."
        ),
    )
