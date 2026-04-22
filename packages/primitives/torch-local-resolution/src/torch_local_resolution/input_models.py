"""Pydantic input models for local resolution estimation.

:class:`ComputeResolutionInput` is built by ``estimate_local_resolution`` and
validated here so invalid configs fail before GPU or FFT work.
"""

from __future__ import annotations

import torch  # runtime: arbitrary_types_allowed + Tensor fields
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from typing_extensions import Self


class ComputeResolutionInput(BaseModel):
    """Validated bundle passed to ``compute_resolution``.

    Field constraints enforce positive physical parameters; ``model_validator``
    ties ``windows_radii`` to ``resolutions``, checks ``(B,Z,Y,X)`` tensors, and
    validates ``gpu_id`` against CUDA when set.
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
        """``cpu`` if ``gpu_id`` is ``None``, else ``cuda:{gpu_id}``."""
        if self.gpu_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{self.gpu_id}")
