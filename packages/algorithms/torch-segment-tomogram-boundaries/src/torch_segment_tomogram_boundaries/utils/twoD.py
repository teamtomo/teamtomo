"""2D image processing utilities for tomographic data.

This module provides the robust normalization helper used across
the tomographic segmentation pipeline.
"""

import torch


def robust_normalization(data: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor using robust statistics (median and percentile range).

    Parameters
    ----------
    data : torch.Tensor
        Input tensor to normalize.

    Returns
    -------
    torch.Tensor
        Normalized tensor with median=0 and scaled by 5th-95th percentile range.
        If range is too small, returns median-centered data without scaling.
    """
    data = data.float()
    p5, p95 = torch.quantile(data, 0.05), torch.quantile(data, 0.95)
    if p95 - p5 < 1e-5:
        return data - torch.median(data)
    return (data - torch.median(data)) / (p95 - p5)
