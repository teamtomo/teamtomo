"""Pydantic input models for local resolution estimation.

`ComputeResolutionInput` is built by `estimate_local_resolution` and
validated here so invalid configs fail before GPU or FFT work.
"""

from __future__ import annotations

from typing import Self

import torch  # runtime: arbitrary_types_allowed + Tensor fields
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ComputeResolutionInput(BaseModel):
    """Validated bundle passed to `compute_resolution`.

    Field constraints enforce positive physical parameters; `model_validator`
    ties `windows_radii` to `resolutions`, checks `(B,Z,Y,X)` tensors, and
    validates `gpu_id` against CUDA when set.

    Attributes
    ----------
    apix : float
        Voxel size in Å per pixel. Must be > 0.
    windows_radii : list[float]
        Local window radii in voxels, one per resolution shell. Must be
        non-empty, positive and the same length as `resolutions`.
    resolutions : list[float]
        Target resolutions in Å, one per entry of `windows_radii`. Must be
        positive.
    batch_half_map1 : torch.Tensor
        First half-maps, shape `(B, Z, Y, X)` with `B >= 1`. Use `Z = 1` for 2D.
    batch_half_map2 : torch.Tensor
        Second half-maps, same shape as `batch_half_map1`.
    step_size : int
        Stride in voxels between correlation samples on the output grid.
        Must be >= 1. Default 3.
    gpu_id : int or None
        CUDA device index, or None to run on CPU. Must be a valid device index
        when set. Default None.
    n_random_maps : int
        Number of surrogate (permuted) maps used to build the null distribution.
        0 means estimate the number from `reference_dist_size`, the largest
        window radius and the map size (`compute_resolution` writes the
        estimate back to this field). Default 0.
    reference_dist_size : int
        Target number of random window samples in the null distribution.
        Must be >= 1. Default 10000.
    do_phase_permutation : bool
        If True, build surrogates by permuting the Fourier phases of half-map 2;
        if False, shuffle its voxels in real space. Default True.
    batch_size : int
        Number of grid positions processed together in the local correlation
        scan (memory vs throughput). Must be >= 1. Default 4096.
    shell_size : float
        Bandpass shell half-width as a fraction of Nyquist: each shell spans
        `1 / resolution ± 0.5 * shell_size / apix` in 1/Å. Must be > 0.
        Default 0.05.
    falloff : float
        Falloff of the hyperbolic-tangent bandpass filter
        (`torch_fourier_filter.bandpass.bandpass_filter_hyptan`). Must be > 0.
        Default 1.5.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    apix: float = Field(gt=0, description="Voxel size in Å per pixel")
    windows_radii: list[float] = Field(
        min_length=1,
        description="Window radii in voxels, one per resolution shell",
    )
    resolutions: list[float] = Field(
        min_length=1,
        description="Target resolutions in Å, one per window radius",
    )
    batch_half_map1: torch.Tensor
    batch_half_map2: torch.Tensor
    step_size: int = Field(default=3, ge=1)
    gpu_id: int | None = None
    n_random_maps: int = Field(default=0, ge=0)
    reference_dist_size: int = Field(default=10_000, ge=1)
    do_phase_permutation: bool = Field(
        default=True,
        description=(
            "If True, Fourier phase permutation for null maps; else real-space shuffle"
        ),
    )
    batch_size: int = Field(default=4096, ge=1)
    shell_size: float = Field(default=0.05, gt=0)
    falloff: float = Field(default=1.5, gt=0)

    @model_validator(mode="after")
    def _check_lists_and_tensors(self) -> Self:
        # Lists, tensors, and CUDA device; scalar bounds use Field(gt=0) above.
        if len(self.windows_radii) != len(self.resolutions):
            msg = (
                "windows_radii and resolutions must have the same length, "
                f"got {len(self.windows_radii)} and {len(self.resolutions)}"
            )
            raise ValueError(msg)
        for i, w in enumerate(self.windows_radii):
            if w <= 0:
                msg = f"windows_radii[{i}] must be positive, got {w}"
                raise ValueError(msg)
        for i, r in enumerate(self.resolutions):
            if r <= 0:
                msg = f"resolutions[{i}] must be positive, got {r}"
                raise ValueError(msg)
        if self.gpu_id is not None and self.gpu_id < 0:
            msg = f"gpu_id must be non-negative when set, got {self.gpu_id}"
            raise ValueError(msg)
        if self.gpu_id is not None:
            if not torch.cuda.is_available():
                msg = "gpu_id was set but CUDA is not available"
                raise ValueError(msg)
            n_cuda = torch.cuda.device_count()
            if self.gpu_id >= n_cuda:
                msg = (
                    f"gpu_id {self.gpu_id} is out of range; "
                    f"only {n_cuda} CUDA device(s)"
                )
                raise ValueError(msg)

        for name, tensor in (
            ("batch_half_map1", self.batch_half_map1),
            ("batch_half_map2", self.batch_half_map2),
        ):
            if tensor.ndim != 4:
                msg = f"{name} must have shape (B, Z, Y, X), got ndim {tensor.ndim}"
                raise ValueError(msg)
        if self.batch_half_map1.shape != self.batch_half_map2.shape:
            msg = (
                "batch_half_map1 and batch_half_map2 shapes must match: "
                f"{tuple(self.batch_half_map1.shape)} vs "
                f"{tuple(self.batch_half_map2.shape)}"
            )
            raise ValueError(msg)
        if self.batch_half_map1.shape[0] < 1:
            raise ValueError("batch dimension B must be >= 1")
        return self

    @computed_field
    def device(self) -> torch.device:
        """`cpu` if `gpu_id` is `None`, else `cuda:{gpu_id}`."""
        if self.gpu_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{self.gpu_id}")
