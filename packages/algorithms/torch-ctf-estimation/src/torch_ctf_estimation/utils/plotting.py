"""Plotting utilities for CTF estimation."""

import numpy as np
import torch
from torch_ctf import calculate_total_phase_shift

from torch_ctf_estimation.models import Defocus1DResults, Defocus2DResults

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as err:
    print("For plotting please install [plot] extras")
    raise ModuleNotFoundError("For plotting please install [plot] extras") from err


def _fit_band_background_subtracted_normalized(
    results1d: Defocus1DResults,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit-band background-subtracted power, min-max normalized (same as plot_1d panel 3)."""
    if (
        results1d.background_model is None
        or results1d.low_frequency_fit is None
        or results1d.powerspectrum_1d is None
    ):
        raise ValueError(
            "Background-subtracted plot requires powerspectrum_1d, background_model, "
            "and fit-range limits on results1d."
        )

    freqs = results1d.frequencies_1d
    power_spec = results1d.powerspectrum_1d
    fit_mask = (freqs >= results1d.low_frequency_fit) & (
        freqs <= results1d.high_frequency_fit
    )
    fit_freqs = freqs[fit_mask]
    fit_power = power_spec[fit_mask]

    device = next(results1d.background_model.parameters()).device
    x = torch.linspace(0, 1, steps=len(fit_freqs), device=device)
    background = torch.exp(results1d.background_model(x).squeeze())
    corrected_power = fit_power.to(device) - background

    corrected_power_min = corrected_power.min()
    corrected_power_max = corrected_power.max()
    corrected_power_normalized = (corrected_power - corrected_power_min) / (
        corrected_power_max - corrected_power_min
    )
    return fit_freqs.detach().cpu(), corrected_power_normalized.detach().cpu()


def _simulated_ctf2_fit_band(
    results1d: Defocus1DResults,
    defocus_um: float,
    phase_shift_degrees: float,
) -> torch.Tensor:
    """Simulated CTF^2 on the 1D fit band (same method as plot_1d panel 3)."""
    if results1d.ctf_model is None or results1d.low_frequency_fit is None:
        raise ValueError("Simulated CTF^2 plot requires ctf_model and fit-range limits.")

    freqs = results1d.frequencies_1d
    fit_mask = (freqs >= results1d.low_frequency_fit) & (
        freqs <= results1d.high_frequency_fit
    )
    fit_freqs = freqs[fit_mask].detach().cpu().numpy()

    ctf_device = results1d.ctf_model.defocus_um.device
    ctf_dtype = results1d.ctf_model.defocus_um.dtype
    fftfreq_sq = torch.as_tensor(fit_freqs**2, device=ctf_device, dtype=ctf_dtype)
    simulated_ctf2 = (
        torch.sin(
            calculate_total_phase_shift(
                defocus_um=torch.tensor(defocus_um, device=ctf_device, dtype=ctf_dtype),
                fftfreq_grid_angstrom_squared=fftfreq_sq,
                voltage_kv=results1d.ctf_model.voltage_kev,
                spherical_aberration_mm=results1d.ctf_model.spherical_aberration_mm,
                amplitude_contrast_fraction=results1d.ctf_model.amplitude_contrast_fraction,
                phase_shift_degrees=torch.tensor(
                    phase_shift_degrees, device=ctf_device, dtype=ctf_dtype
                ),
            )
        )
        ** 2
    )
    return simulated_ctf2.detach().cpu()


def plot_1d_spectrum(results1d: Defocus1DResults) -> None:
    """Plot 1D power spectrum analysis results.

    Parameters
    ----------
    results1d : Defocus1DResults
        Results from 1D defocus estimation containing frequencies, power spectrum,
        background model, and CTF fitting results.
    """
    _, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: 1D Power Spectrum with background
    ax1 = axes[0, 0]
    freqs = results1d.frequencies_1d.detach().cpu().numpy()
    power_spec = results1d.powerspectrum_1d.detach().cpu().numpy()

    ax1.semilogy(freqs, power_spec, "b-", alpha=0.7, label="Power Spectrum")

    # Add background model if available
    if (
        results1d.background_model is not None
        and results1d.low_frequency_fit is not None
    ):
        # Get fitting range
        fit_mask = (freqs >= results1d.low_frequency_fit) & (
            freqs <= results1d.high_frequency_fit
        )
        fit_freqs = freqs[fit_mask]

        # Evaluate background model (use same device as model to avoid device mismatch)
        device = next(results1d.background_model.parameters()).device
        x = torch.linspace(0, 1, steps=len(fit_freqs), device=device)
        background = (
            torch.exp(results1d.background_model(x).squeeze()).detach().cpu().numpy()
        )

        ax1.semilogy(fit_freqs, background, "r--", alpha=0.8, label="Background Model")

        # Mark fitting range
        ax1.axvline(results1d.low_frequency_fit, color="gray", linestyle=":", alpha=0.5)
        ax1.axvline(
            results1d.high_frequency_fit, color="gray", linestyle=":", alpha=0.5
        )
        ax1.fill_betweenx(
            ax1.get_ylim(),
            results1d.low_frequency_fit,
            results1d.high_frequency_fit,
            alpha=0.1,
            color="gray",
            label="Fit Range",
        )

    ax1.set_xlabel("Spatial Frequency (1/Å)")
    ax1.set_ylabel("Power")
    ax1.set_title("1D Power Spectrum")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cross-correlation vs defocus
    if results1d.test_defoci is not None and results1d.cross_correlations is not None:
        ax2 = axes[0, 1]
        defoci = results1d.test_defoci.detach().cpu().numpy()
        correlations = results1d.cross_correlations.detach().cpu().numpy()

        ax2.plot(defoci, correlations, "g-", linewidth=2)

        # Mark best defocus
        best_idx = correlations.argmax()
        best_defocus = defoci[best_idx]
        ax2.axvline(
            best_defocus,
            color="red",
            linestyle="--",
            label=f"Best Defocus: {best_defocus:.3f} μm",
        )
        ax2.scatter(best_defocus, correlations[best_idx], color="red", s=100, zorder=5)

        ax2.set_xlabel("Defocus (μm)")
        ax2.set_ylabel("Cross-correlation")
        ax2.set_title("CTF Fitting Cross-correlation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Background-subtracted power spectrum in fit range
    if (
        results1d.background_model is not None
        and results1d.low_frequency_fit is not None
    ):
        ax3 = axes[1, 0]
        fit_freqs, corrected_power_normalized = _fit_band_background_subtracted_normalized(
            results1d
        )
        fit_freqs_np = fit_freqs.numpy()

        ax3.plot(
            fit_freqs_np,
            corrected_power_normalized.numpy(),
            "purple",
            linewidth=2,
            label="Background-subtracted",
        )

        if results1d.ctf_model is not None:
            defocus_um = float(results1d.ctf_model.defocus_um.detach().cpu())
            phase_shift_deg = float(
                results1d.ctf_model.phase_shift_degrees.detach().cpu()
            )
            simulated_ctf2 = _simulated_ctf2_fit_band(
                results1d, defocus_um, phase_shift_deg
            )
            ax3.plot(
                fit_freqs_np,
                simulated_ctf2.numpy(),
                "orange",
                linestyle="--",
                label="Simulated CTF^2",
            )
        ax3.set_xlabel("Spatial Frequency (1/Å)")
        ax3.set_ylabel("Corrected Power")
        ax3.set_title("Background-subtracted Power Spectrum")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: CTF parameters summary
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Display CTF parameters as text
    ctf_info = [
        f"Defocus: {results1d.ctf_model.defocus_um:.3f} μm",
        f"Voltage: {results1d.ctf_model.voltage_kev:.1f} keV",
        f"Cs: {results1d.ctf_model.spherical_aberration_mm:.2f} mm",
        f"Amplitude contrast: {results1d.ctf_model.amplitude_contrast_fraction:.3f}",
        f"Phase shift: {results1d.ctf_model.phase_shift_degrees:.1f}°",
    ]

    if results1d.low_frequency_fit is not None:
        ctf_info.extend(
            [
                "",
                f"Fit range: {results1d.low_frequency_fit:.3f} - "
                f"{results1d.high_frequency_fit:.3f} 1/Å",
            ]
        )

    ax4.text(
        0.1,
        0.9,
        "CTF Parameters:",
        fontsize=14,
        fontweight="bold",
        transform=ax4.transAxes,
        verticalalignment="top",
    )

    for i, info in enumerate(ctf_info):
        ax4.text(
            0.1,
            0.8 - i * 0.08,
            info,
            fontsize=12,
            transform=ax4.transAxes,
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.show()


def _mean_defocus_um_2d(results2d: Defocus2DResults) -> float:
    if results2d.defocus_u is not None and results2d.defocus_v is not None:
        return (results2d.defocus_u + results2d.defocus_v) / 2
    if results2d.defocus_u is not None:
        return results2d.defocus_u
    raise ValueError("plot_2d_spectrum requires defocus_u (and ideally defocus_v).")


def plot_2d_spectrum(
    results2d: Defocus2DResults,
    results1d: Defocus1DResults,
) -> None:
    """Plot 2D fit against the same 1D background-subtracted spectrum.

    Background-subtracted power is identical to ``plot_1d_spectrum`` panel 3.
    Simulated CTF^2 uses the same 1D formula with defocus and phase shift from
    the 2D fit.

    Parameters
    ----------
    results2d : Defocus2DResults
        Results from 2D defocus / phase-shift estimation.
    results1d : Defocus1DResults
        Results from 1D estimation (background, frequencies, optics).
    """
    fit_freqs, corrected_power_normalized = _fit_band_background_subtracted_normalized(
        results1d
    )
    fit_freqs_np = fit_freqs.numpy()

    defocus_um = _mean_defocus_um_2d(results2d)
    if results2d.phase_shift_degrees is not None:
        phase_shift_deg = results2d.phase_shift_degrees
    elif results1d.ctf_model is not None:
        phase_shift_deg = float(results1d.ctf_model.phase_shift_degrees.detach().cpu())
    else:
        phase_shift_deg = 0.0

    simulated_ctf2 = _simulated_ctf2_fit_band(results1d, defocus_um, phase_shift_deg)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    ax1.plot(
        fit_freqs_np,
        corrected_power_normalized.numpy(),
        "purple",
        linewidth=2,
        label="Background-subtracted",
    )
    ax1.plot(
        fit_freqs_np,
        simulated_ctf2.numpy(),
        "orange",
        linestyle="--",
        linewidth=2,
        label="Simulated CTF^2 (2D fit)",
    )
    ax1.axvline(results1d.low_frequency_fit, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(results1d.high_frequency_fit, color="gray", linestyle=":", alpha=0.5)
    ax1.fill_betweenx(
        ax1.get_ylim(),
        results1d.low_frequency_fit,
        results1d.high_frequency_fit,
        alpha=0.1,
        color="gray",
        label="Fit range",
    )
    ax1.set_xlabel("Spatial Frequency (1/Å)")
    ax1.set_ylabel("Normalized power / CTF^2")
    ax1.set_title("2D fit: background-subtracted power vs CTF^2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.axis("off")
    ctf_info = [
        f"Defocus u: {results2d.defocus_u:.3f} μm",
        f"Defocus v: {results2d.defocus_v:.3f} μm",
    ]
    if results2d.astigmatism is not None:
        ctf_info.append(f"Astigmatism: {results2d.astigmatism:.3f} μm")
    if results2d.astigmatism_angle is not None:
        ctf_info.append(f"Astigmatism angle: {results2d.astigmatism_angle:.1f}°")
    if results2d.phase_shift_degrees is not None:
        ctf_info.append(f"Phase shift: {results2d.phase_shift_degrees:.1f}°")
    if results2d.envelope_B is not None:
        ctf_info.append(f"Envelope B: {results2d.envelope_B:.1f}")
    if results2d.cross_correlation_final is not None:
        ctf_info.append(f"2D cross-correlation: {results2d.cross_correlation_final:.3f}")
    ctf_info.extend(
        [
            "",
            f"Fit range: {results1d.low_frequency_fit:.3f} - "
            f"{results1d.high_frequency_fit:.3f} 1/Å",
        ]
    )

    ax2.text(
        0.1,
        0.9,
        "2D CTF Parameters:",
        fontsize=14,
        fontweight="bold",
        transform=ax2.transAxes,
        verticalalignment="top",
    )
    for i, info in enumerate(ctf_info):
        ax2.text(
            0.1,
            0.8 - i * 0.08,
            info,
            fontsize=12,
            transform=ax2.transAxes,
            verticalalignment="top",
        )

    plt.tight_layout()
    plt.show()


def _rfft_to_display_image(spectrum_rfft: torch.Tensor) -> torch.Tensor:
    """Mirror rfft half to a symmetric display image (same layout as docs notebooks)."""
    spectrum_rfft = spectrum_rfft.detach().cpu().float()
    if spectrum_rfft.ndim != 2:
        raise ValueError(
            f"Expected 2D rfft spectrum, got shape {tuple(spectrum_rfft.shape)}"
        )
    left = torch.flip(spectrum_rfft[:, 2:], dims=(1, 0))
    return torch.hstack([left, spectrum_rfft])


def _fiji_contrast_uint8(
    image: torch.Tensor,
    saturated_percent: float = 3.0,
) -> np.ndarray:
    """Scale to uint8 using ImageJ-style saturated-percentile contrast."""
    values = image.detach().cpu().numpy()
    low = np.percentile(values, saturated_percent)
    high = np.percentile(values, 100.0 - saturated_percent)
    if high <= low:
        high = low + 1e-12
    scaled = (values - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _patch_spectrum_2d_from_results(
    results2d: Defocus2DResults,
    frame_index: int = 0,
) -> torch.Tensor:
    if results2d.patch_power_spectra is None:
        raise ValueError(
            "2D spectrum images require patch_power_spectra. "
            "Re-run estimate_ctf with debug=True in CTFFittingParams."
        )
    patch_ps = results2d.patch_power_spectra[frame_index]
    if patch_ps.ndim == 4:
        patch_ps = patch_ps.mean(dim=(0, 1))
    elif patch_ps.ndim != 2:
        raise ValueError(
            f"Unexpected patch_power_spectra shape: {tuple(patch_ps.shape)}"
        )
    return patch_ps


def _simulated_ctf2_2d_from_results(results2d: Defocus2DResults) -> torch.Tensor:
    if results2d.simulated_ctf2s is None:
        raise ValueError(
            "2D spectrum images require simulated_ctf2s. "
            "Re-run estimate_ctf with debug=True in CTFFittingParams."
        )
    simulated_ctf2 = results2d.simulated_ctf2s
    if simulated_ctf2.ndim == 4:
        simulated_ctf2 = simulated_ctf2.mean(dim=(0, 1))
    elif simulated_ctf2.ndim != 2:
        raise ValueError(
            f"Unexpected simulated_ctf2s shape: {tuple(simulated_ctf2.shape)}"
        )
    return simulated_ctf2


def plot_2d_spectrum_images(
    results2d: Defocus2DResults,
    frame_index: int = 0,
    saturated_percent: float = 3.0,
) -> None:
    """Show measured and simulated 2D spectra side by side.

    Uses background-subtracted patch power from the 2D fit and the final
    simulated CTF^2 model. Display uses symmetric rfft layout with ImageJ-style
    saturated-percentile contrast scaled to 0-255.

    Parameters
    ----------
    results2d : Defocus2DResults
        Results from 2D defocus / phase-shift estimation (``debug=True``).
    frame_index : int
        Time-frame index when ``patch_power_spectra`` includes a time dimension.
    saturated_percent : float
        Percent of pixels saturated at each histogram tail (ImageJ default is
        0.35; 3.0 matches common Fiji "Enhance Contrast" usage at 3%).
    """
    measured_rfft = _patch_spectrum_2d_from_results(results2d, frame_index)
    simulated_rfft = _simulated_ctf2_2d_from_results(results2d)

    measured_display = _rfft_to_display_image(measured_rfft)
    simulated_display = _rfft_to_display_image(simulated_rfft)

    measured_img = _fiji_contrast_uint8(measured_display, saturated_percent)
    simulated_img = _fiji_contrast_uint8(simulated_display, saturated_percent)

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(measured_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0].set_title("Measured power spectrum (2D)")
    axes[0].axis("off")

    axes[1].imshow(simulated_img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("Simulated CTF^2 (2D fit)")
    axes[1].axis("off")

    plt.suptitle(
        f"2D spectra ({saturated_percent:g}% saturated contrast, 0-255)",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
