"""2D template matching in pytorch"""

__version__ = "0.1.0"
__author__ = "Josh Dickerson"
__email__ = "jdickerson@berkeley.edu"

from .cross_correlate import match_template_dft_2d

__all__ = ["match_template_dft_2d"]
