"""Primitive packages for cryo-EM and cryo-ET operations."""

# Import all primitive packages
try:
    import torch_affine_utils
except ImportError:
    torch_affine_utils = None  # type: ignore[assignment]

try:
    import torch_calculate_electrostatic_potential
except ImportError:
    torch_calculate_electrostatic_potential = None  # type: ignore[assignment]

try:
    import torch_ctf
except ImportError:
    torch_ctf = None  # type: ignore[assignment]

try:
    import torch_cubic_spline_grids
except ImportError:
    torch_cubic_spline_grids = None  # type: ignore[assignment]

try:
    import torch_find_peaks
except ImportError:
    torch_find_peaks = None  # type: ignore[assignment]

try:
    import torch_fourier_filter
except ImportError:
    torch_fourier_filter = None  # type: ignore[assignment]

try:
    import torch_fourier_rescale
except ImportError:
    torch_fourier_rescale = None  # type: ignore[assignment]

try:
    import torch_fourier_shell_correlation
except ImportError:
    torch_fourier_shell_correlation = None  # type: ignore[assignment]

try:
    import torch_fourier_shift
except ImportError:
    torch_fourier_shift = None  # type: ignore[assignment]

try:
    import torch_fourier_slice
except ImportError:
    torch_fourier_slice = None  # type: ignore[assignment]

try:
    import torch_grid_utils
except ImportError:
    torch_grid_utils = None  # type: ignore[assignment]

try:
    import torch_image_interpolation
except ImportError:
    torch_image_interpolation = None  # type: ignore[assignment]

try:
    import torch_scattering
except ImportError:
    torch_scattering = None  # type: ignore[assignment]

try:
    import torch_so3
except ImportError:
    torch_so3 = None  # type: ignore[assignment]

try:
    import torch_subpixel_crop
except ImportError:
    torch_subpixel_crop = None  # type: ignore[assignment]

try:
    import torch_structure_manipulation
except ImportError:
    torch_structure_manipulation = None  # type: ignore[assignment]

try:
    import torch_transform_image
except ImportError:
    torch_transform_image = None  # type: ignore[assignment]

try:
    import torch_tilt_series
except ImportError:
    torch_tilt_series = None  # type: ignore[assignment]

__all__ = [
    "torch_affine_utils",
    "torch_calculate_electrostatic_potential",
    "torch_ctf",
    "torch_cubic_spline_grids",
    "torch_find_peaks",
    "torch_fourier_filter",
    "torch_fourier_rescale",
    "torch_fourier_shell_correlation",
    "torch_fourier_shift",
    "torch_fourier_slice",
    "torch_grid_utils",
    "torch_image_interpolation",
    "torch_scattering",
    "torch_so3",
    "torch_structure_manipulation",
    "torch_subpixel_crop",
    "torch_tilt_series",
    "torch_transform_image",
]
