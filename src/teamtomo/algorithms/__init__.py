"""Algorithms packages for cryo-EM and cryo-ET operations."""

# Import all algorithm packages
try:
    import torch_2dtm
except ImportError:
    torch_2dtm = None  # type: ignore[assignment]

try:
    import torch_motion_correction
except ImportError:
    torch_motion_correction = None  # type: ignore[assignment]

try:
    import torch_cryoeraser
except ImportError:
    torch_cryoeraser = None  # type: ignore[assignment]

try:
    import torch_refine_tilt_axis_angle
except ImportError:
    torch_refine_tilt_axis_angle = None  # type: ignore[assignment]

try:
    import torch_segment_fiducials_2d
except ImportError:
    torch_segment_fiducials_2d = None  # type: ignore[assignment]

__all__ = [
    "torch_2dtm",
    "torch_motion_correction",
    "torch_cryoeraser",
    "torch_refine_tilt_axis_angle",
]
