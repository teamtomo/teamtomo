"""Configuration models for micrograph simulation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from teamtomo_basemodel import BaseModelTeamTomo, ExcludedTensor
from torch_fourier_filter.mtf import list_mtf_references


class LppConfig(BaseModelTeamTomo):
    """Laser phase plate (LPP) options for :func:`torch_ctf.calc_LPP_ctf_2D`.

    When ``apply`` is ``True``, the spatially varying LPP phase replaces the
    uniform ``CtfConfig.phase_shift_deg``.
    """

    apply: bool = False
    NA: float = Field(default=0.1, gt=0, description="Laser numerical aperture")
    laser_wavelength_angstrom: float = Field(
        default=5000.0, gt=0, description="Laser wavelength in Angstroms"
    )
    focal_length_angstrom: float = Field(
        default=1e6, gt=0, description="Focal length in Angstroms"
    )
    laser_xy_angle_deg: float = Field(
        default=0.0, description="Laser rotation in the xy plane (degrees)"
    )
    laser_xz_angle_deg: float = Field(
        default=0.0, description="Laser angle in the xz plane (degrees)"
    )
    laser_long_offset_angstrom: float = Field(
        default=0.0, description="Longitudinal laser offset in Angstroms"
    )
    laser_trans_offset_angstrom: float = Field(
        default=0.0, description="Transverse laser offset in Angstroms"
    )
    laser_polarization_angle_deg: float = Field(
        default=0.0, description="Laser polarization angle in degrees"
    )
    peak_phase_deg: float = Field(
        default=90.0, description="Desired peak LPP phase in degrees"
    )
    dual_laser: bool = Field(
        default=False,
        description="If True, add a second perpendicular laser and sum phases",
    )


class CtfConfig(BaseModelTeamTomo):
    """Contrast transfer function options applied in the wave domain.

    Parameters mirror :func:`torch_ctf.calculate_ctf_2d`. Optional fields
    (beam tilt, Zernikes, anisotropic magnification, LPP) default to unused.
    """

    apply: bool = True
    voltage_kv: float = Field(
        default=300.0, gt=0, description="Acceleration voltage (kV)"
    )
    defocus_um: float = Field(
        default=1.5, description="Defocus in µm; positive is underfocus"
    )
    astigmatism_um: float = Field(
        default=0.0,
        description="Astigmatism amplitude in µm, (defocus_u - defocus_v) / 2",
    )
    astigmatism_angle_deg: float = Field(
        default=0.0,
        description="Astigmatism angle in degrees; 0 places defocus_u along y",
    )
    spherical_aberration_mm: float = Field(
        default=2.7, description="Spherical aberration Cs in mm"
    )
    amplitude_contrast: float = Field(
        default=0.07, ge=0.0, le=1.0, description="Amplitude contrast fraction"
    )
    phase_shift_deg: float = Field(
        default=0.0,
        description=(
            "Additional uniform phase shift in degrees (e.g. Volta phase plate). "
            "Ignored when ``lpp.apply`` is True."
        ),
    )
    beam_tilt_mrad: tuple[float, float] | None = Field(
        default=None,
        description="Beam tilt [bx, by] in milliradians",
    )
    even_zernike_coeffs: dict[str, float] | None = Field(
        default=None,
        description='Even Zernike coefficients, e.g. {"Z44c": 0.1, "Z60": 0.3}',
    )
    odd_zernike_coeffs: dict[str, float] | None = Field(
        default=None,
        description='Odd Zernike coefficients, e.g. {"Z31c": 0.1, "Z33s": 0.4}',
    )
    transform_matrix: ExcludedTensor = Field(
        default=None,
        description="Optional 2x2 real-space anisotropic magnification matrix",
    )
    lpp: LppConfig = Field(default_factory=LppConfig)


class FluenceConfig(BaseModelTeamTomo):
    """Fluence scaling from relative intensity to expected electron counts.

    Expected counts per pixel are::

        λ = (I / mean(I)) * dose_e_per_A2 * pixel_size² * coincidence_loss

    where ``coincidence_loss`` is a multiplicative efficiency factor
    (``1.0`` = no loss, ``0.8`` = 20% loss).
    """

    dose_e_per_A2: float = Field(
        default=30.0, gt=0, description="Total fluence in e⁻/Å²"
    )
    coincidence_loss: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Detector coincidence-loss factor (1 = none, 0.8 = 20% loss)",
    )


class PoissonConfig(BaseModelTeamTomo):
    """Poisson shot-noise sampling options.

    Every draw is seeded separately, so draws are independent on any device.
    Without a ``seed``, draws follow ``torch.manual_seed``. With a ``seed``,
    pass a ``draw_index`` with each draw (see
    :func:`~torch_simulate_image.poisson_sample`).
    """

    apply: bool = True
    seed: int | None = Field(
        default=None,
        description="Base seed, combined with each draw's draw_index",
    )


class DqeConfig(BaseModelTeamTomo):
    """Detector quantum efficiency (MTF) options.

    When ``apply`` is ``True``, give the MTF in exactly one way:
    ``mtf_reference``, ``starfile_path``, or ``mtf_frequencies`` with
    ``mtf_amplitudes``.
    """

    apply: bool = False
    mtf_reference: str | None = Field(
        default=None,
        description=(
            "Name of a detector MTF bundled with torch-fourier-filter, e.g. "
            "'k3_standard_300kv_fl2' (case-insensitive). "
            "torch_fourier_filter.mtf.list_mtf_references() lists them."
        ),
    )
    mtf_frequencies: ExcludedTensor = Field(
        default=None,
        description="MTF spatial frequencies in ascending order, cycles/pixel",
    )
    mtf_amplitudes: ExcludedTensor = None
    starfile_path: str | None = None
    apply_before_noise: bool = False

    @model_validator(mode="after")
    def _validate_mtf_source(self) -> Self:
        if self.mtf_reference is not None:
            # a typo should fail here, not at the first simulated image
            known = list_mtf_references()
            if self.mtf_reference.lower() not in known:
                msg = (
                    f"Unknown mtf_reference {self.mtf_reference!r}. "
                    f"Bundled detector MTFs: {known}"
                )
                raise ValueError(msg)
        if not self.apply:
            return self
        has_freq = self.mtf_frequencies is not None
        has_amp = self.mtf_amplitudes is not None
        if has_freq != has_amp:
            msg = (
                "When dqe.apply is True, mtf_frequencies and mtf_amplitudes "
                "must both be set or both be None."
            )
            raise ValueError(msg)
        n_sources = sum(
            [
                self.mtf_reference is not None,
                self.starfile_path is not None,
                has_freq and has_amp,
            ]
        )
        if n_sources != 1:
            msg = (
                "When dqe.apply is True, provide exactly one of mtf_reference, "
                "starfile_path, or (mtf_frequencies, mtf_amplitudes)."
            )
            raise ValueError(msg)
        return self


class DoseWeightConfig(BaseModelTeamTomo):
    """Grant & Grigorieff dose weighting in Fourier space.

    Each frequency is scaled by the mean exposure filter over the cumulative
    dose from ``dose_start`` to ``dose_end`` (see
    :func:`~torch_simulate_image.apply_dose_weight`).
    """

    apply: bool = False
    dose_start: float = Field(
        default=0.0, description="Cumulative dose in e⁻/Å² before this image"
    )
    dose_end: float | None = Field(
        default=None,
        description="Cumulative dose in e⁻/Å² after this image; required to apply",
    )
    crit_exposure_bfactor: float = -1

    @model_validator(mode="after")
    def _validate_dose_end(self) -> Self:
        if self.apply and self.dose_end is None:
            msg = "When dose_weight.apply is True, dose_end must be set."
            raise ValueError(msg)
        return self


class EnvelopeConfig(BaseModelTeamTomo):
    """Optional envelopes applied to intensity in Fourier space.

    B-factor and dose envelopes are self-contained. Cs / Cc envelopes reuse
    defocus, spherical aberration, and voltage from :class:`CtfConfig` when
    applied via the pipeline.
    """

    apply: bool = False
    b_factor: float = Field(
        default=0.0, description="B-factor in Å²; 0 disables the B envelope"
    )
    dose_envelope: bool = Field(
        default=False, description="Apply Grant & Grigorieff dose envelope"
    )
    cs_envelope: bool = Field(
        default=False, description="Apply spatial-coherence (Cs) envelope"
    )
    cc_envelope: bool = Field(
        default=False, description="Apply temporal-coherence (Cc) envelope"
    )
    illumination_semiangle_mrad: float = Field(
        default=0.005,
        gt=0,
        description="Illumination semi-angle alpha for Cs envelope (mrad)",
    )
    chromatic_aberration_mm: float = Field(
        default=2.7, gt=0, description="Chromatic aberration Cc in mm"
    )
    energy_spread_ev: float = Field(
        default=0.7, gt=0, description="FWHM energy spread in eV (Cc envelope)"
    )
    delta_v_over_v: float = Field(
        default=0.06e-6, ge=0, description="Relative HT voltage fluctuation ΔV/V"
    )
    delta_i_over_i: float = Field(
        default=0.01e-6, ge=0, description="Relative lens-current fluctuation ΔI/I"
    )


class ObjectiveApertureConfig(BaseModelTeamTomo):
    """Circular objective aperture (pupil) applied in the diffraction plane.

    When ``apply`` is ``True``, provide exactly one cutoff specification:

    - ``outer_semiangle_mrad`` — physical aperture semi-angle; converted to
      spatial frequency via ``q_max = alpha / wavelength``.
    - ``cutoff_frequency_inv_A`` — hard cutoff in Å⁻¹ directly.
    """

    apply: bool = False
    outer_semiangle_mrad: float | None = Field(
        default=None,
        gt=0,
        description="Objective aperture outer semi-angle in mrad",
    )
    cutoff_frequency_inv_A: float | None = Field(
        default=None,
        gt=0,
        description="Cutoff spatial frequency in Å⁻¹ (alternative to semi-angle)",
    )
    soft_edge_half_width_inv_A: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Cosine soft-edge half-width in Å⁻¹: roll-off spans "
            "[q_max - w, q_max + w]; 0 is a hard mask"
        ),
    )

    @model_validator(mode="after")
    def _validate_cutoff_source(self) -> Self:
        if not self.apply:
            return self
        has_angle = self.outer_semiangle_mrad is not None
        has_freq = self.cutoff_frequency_inv_A is not None
        if has_angle == has_freq:
            msg = (
                "When objective_aperture.apply is True, provide exactly one of "
                "outer_semiangle_mrad or cutoff_frequency_inv_A."
            )
            raise ValueError(msg)
        return self


class MicrographSimulationConfig(BaseModelTeamTomo):
    """Top-level configuration for :func:`~torch_simulate_image.simulate_micrograph`.

    Acceleration voltage lives on ``ctf.voltage_kv`` and is reused for dose
    weighting, objective aperture angle conversion, and envelopes.
    """

    pixel_size: float = Field(gt=0, description="Pixel size in Angstroms")
    ctf: CtfConfig = Field(default_factory=CtfConfig)
    objective_aperture: ObjectiveApertureConfig = Field(
        default_factory=ObjectiveApertureConfig
    )
    fluence: FluenceConfig = Field(default_factory=FluenceConfig)
    poisson: PoissonConfig = Field(default_factory=PoissonConfig)
    dqe: DqeConfig = Field(default_factory=DqeConfig)
    dose_weight: DoseWeightConfig = Field(default_factory=DoseWeightConfig)
    envelope: EnvelopeConfig = Field(default_factory=EnvelopeConfig)
    return_expected_counts: bool = False
