"""CTF calculation for cryoEM in torch."""

from importlib.metadata import PackageNotFoundError, version

from torch_ctf.ctf_1d import calculate_ctf_1d
from torch_ctf.ctf_2d import calculate_ctf_2d
from torch_ctf.ctf_aberrations import (
    apply_even_zernikes,
    apply_odd_zernikes,
    beam_tilt_to_zernike_coeffs,
    calculate_defocus_phase_aberration,
    calculate_relativistic_electron_wavelength,
    resolve_odd_zernikes,
)
from torch_ctf.ctf_ewald import calculate_ctfp_and_ctfq_2d, get_ctf_weighting
from torch_ctf.ctf_lpp import (
    calc_LPP_ctf_2D,
    calc_LPP_phase,
    calculate_relativistic_beta,
    calculate_relativistic_gamma,
    get_eta,
    get_eta0_from_peak_phase_deg,
    initialize_laser_params,
    make_laser_coords,
)
from torch_ctf.ctf_utils import (
    calculate_additional_phase_shift,
    calculate_amplitude_contrast_equivalent_phase_shift,
    calculate_total_phase_shift,
)

try:
    __version__ = version("torch-ctf")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"

__all__ = [
    "apply_even_zernikes",
    "apply_odd_zernikes",
    "beam_tilt_to_zernike_coeffs",
    "calc_LPP_ctf_2D",
    "calc_LPP_phase",
    "calculate_additional_phase_shift",
    "calculate_amplitude_contrast_equivalent_phase_shift",
    "calculate_ctf_1d",
    "calculate_ctf_2d",
    "calculate_ctfp_and_ctfq_2d",
    "calculate_defocus_phase_aberration",
    "calculate_relativistic_beta",
    "calculate_relativistic_electron_wavelength",
    "calculate_relativistic_gamma",
    "calculate_total_phase_shift",
    "get_ctf_weighting",
    "get_eta",
    "get_eta0_from_peak_phase_deg",
    "initialize_laser_params",
    "make_laser_coords",
    "resolve_odd_zernikes",
]
