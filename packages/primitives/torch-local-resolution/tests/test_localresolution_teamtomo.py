"""Basic smoke tests for estimate_local_resolution."""

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from torch_local_resolution import estimate_local_resolution

WINDOWS_RADII = [3, 5, 7]
NUM_RADII = len(WINDOWS_RADII)
RESOLUTIONS = [3.0, 4.0, 5.0]
APIX = 1.5


def _make_random_bzyx_half_maps(
    spatial_shape: tuple[int, ...],
    batch_size: int = 1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired half-maps with shape (B, Z, Y, X).

    ``spatial_shape`` is ``(Y, X)`` for 2D (stored as Z=1) or ``(Z, Y, X)`` for 3D.
    """
    generator = torch.manual_seed(seed)
    if len(spatial_shape) == 2:
        z, y, x = 1, spatial_shape[0], spatial_shape[1]
    elif len(spatial_shape) == 3:
        z, y, x = spatial_shape
    else:
        msg = f"spatial_shape must be (Y, X) or (Z, Y, X), got {spatial_shape!r}"
        raise ValueError(msg)
    noise_scale = 0.5
    shared = torch.randn(batch_size, z, y, x, generator=generator)
    noise1 = torch.randn(batch_size, z, y, x, generator=generator)
    noise2 = torch.randn(batch_size, z, y, x, generator=generator)
    half1 = shared + noise_scale * noise1
    half2 = shared + noise_scale * noise2
    return half1, half2


def _expected_shape(
    spatial_shape: tuple[int, ...],
    step_size: int,
    batch_size: int = 1,
    num_radii: int | None = None,
) -> torch.Size:
    """Expected output shape: (batch, num_radii, *spatial)."""
    if num_radii is None:
        num_radii = NUM_RADII
    spatial = tuple(math.ceil(s / step_size) for s in spatial_shape)
    return torch.Size([batch_size, num_radii, *list(spatial)])


class TestComputeResolution2D:
    """Verify that estimate_local_resolution works on 2D input."""

    def test_runs_without_error(self):
        """estimate_local_resolution should not crash on 2D input."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(64, 64))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
        )
        assert result is not None

    @pytest.mark.parametrize("step_size", [1, 3, 5])
    def test_output_shape(self, step_size):
        """Output shape should be (1, num_radii, *spatial)."""
        spatial_shape = (64, 64)
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=spatial_shape)
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            step_size=step_size,
        )
        expected = _expected_shape(spatial_shape, step_size)
        msg = f"step_size={step_size}: expected {expected}, got {result.shape}"
        assert result.shape == expected, msg

    def test_output_values_between_zero_and_one(self):
        """All output values should be in [0, 1]."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(64, 64))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
        )
        assert torch.all(result >= 0), "Found values below 0"
        assert torch.all(result <= 1), "Found values above 1"


class TestComputeResolution3D:
    """Verify that estimate_local_resolution works on 3D input."""

    def test_runs_without_error(self):
        """estimate_local_resolution should not crash on 3D input."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
        )
        assert result is not None

    @pytest.mark.parametrize("step_size", [1, 3, 5])
    def test_output_shape(self, step_size):
        """Output shape should be (1, num_radii, *spatial)."""
        spatial_shape = (32, 32, 32)
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=spatial_shape)
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            step_size=step_size,
        )
        expected = _expected_shape(spatial_shape, step_size)
        msg = f"step_size={step_size}: expected {expected}, got {result.shape}"
        assert result.shape == expected, msg

    def test_output_values_between_zero_and_one(self):
        """All output values should be in [0, 1]."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
        )
        assert torch.all(result >= 0), "Found values below 0"
        assert torch.all(result <= 1), "Found values above 1"


class TestComputeResolutionBatch:
    """Verify batched input is handled correctly."""

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch_output_shape(self, batch_size):
        """Batch dim should match number of input maps."""
        spatial_shape = (32, 32, 32)
        half1, half2 = _make_random_bzyx_half_maps(
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
        )
        expected = _expected_shape(
            spatial_shape,
            step_size=3,
            batch_size=batch_size,
        )
        msg = f"batch_size={batch_size}: expected {expected}, got {result.shape}"
        assert result.shape == expected, msg

    def test_batch_values_between_zero_and_one(self):
        """All output values should be in [0, 1]."""
        half1, half2 = _make_random_bzyx_half_maps(
            spatial_shape=(32, 32, 32),
            batch_size=3,
        )
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
        )
        assert torch.all(result >= 0), "Found values below 0"
        assert torch.all(result <= 1), "Found values above 1"


class TestPermutationModes:
    """Verify both permutation branches: phase permutation and real-space shuffle."""

    @pytest.mark.parametrize("do_phase_permutation", [True, False])
    def test_runs_without_error_2d(self, do_phase_permutation):
        """Both permutation modes should complete without error on 2D input."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(64, 64))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            do_phase_permutation=do_phase_permutation,
        )
        assert result is not None

    @pytest.mark.parametrize("do_phase_permutation", [True, False])
    def test_runs_without_error_3d(self, do_phase_permutation):
        """Both permutation modes should complete without error on 3D input."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            do_phase_permutation=do_phase_permutation,
        )
        assert result is not None

    @pytest.mark.parametrize("do_phase_permutation", [True, False])
    @pytest.mark.parametrize("step_size", [1, 3, 5])
    def test_output_shape_matches_both_modes(self, do_phase_permutation, step_size):
        """Output shape must not depend on permutation mode."""
        spatial_shape = (32, 32, 32)
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=spatial_shape)
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            step_size=step_size,
            do_phase_permutation=do_phase_permutation,
        )
        expected = _expected_shape(spatial_shape, step_size)
        msg = (
            f"do_phase_permutation={do_phase_permutation}, step_size={step_size}: "
            f"expected {expected}, got {result.shape}"
        )
        assert result.shape == expected, msg

    @pytest.mark.parametrize("do_phase_permutation", [True, False])
    def test_output_values_between_zero_and_one(self, do_phase_permutation):
        """Both permutation modes must produce values in [0, 1]."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            do_phase_permutation=do_phase_permutation,
        )
        assert torch.all(result >= 0), (
            f"do_phase_permutation={do_phase_permutation}: found values below 0"
        )
        assert torch.all(result <= 1), (
            f"do_phase_permutation={do_phase_permutation}: found values above 1"
        )


class TestSkipStatistics:
    """Verify skip_statistics=True returns raw cosine-similarity correlations."""

    def test_runs_without_error_2d(self):
        """skip_statistics=True should not crash on 2D input."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(64, 64))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            skip_statistics=True,
        )
        assert result is not None

    def test_runs_without_error_3d(self):
        """skip_statistics=True should not crash on 3D input."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            skip_statistics=True,
        )
        assert result is not None

    @pytest.mark.parametrize("step_size", [1, 3, 5])
    def test_output_shape_2d(self, step_size):
        """Output shape should be (1, num_radii, *spatial) for 2D."""
        spatial_shape = (64, 64)
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=spatial_shape)
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            step_size=step_size,
            skip_statistics=True,
        )
        expected = _expected_shape(spatial_shape, step_size)
        msg = f"step_size={step_size}: expected {expected}, got {result.shape}"
        assert result.shape == expected, msg

    @pytest.mark.parametrize("step_size", [1, 3, 5])
    def test_output_shape_3d(self, step_size):
        """Output shape should be (1, num_radii, *spatial) for 3D."""
        spatial_shape = (32, 32, 32)
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=spatial_shape)
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            step_size=step_size,
            skip_statistics=True,
        )
        expected = _expected_shape(spatial_shape, step_size)
        msg = f"step_size={step_size}: expected {expected}, got {result.shape}"
        assert result.shape == expected, msg

    def test_output_values_between_neg1_and_1_2d(self):
        """Cosine-similarity values should be in [-1, 1] for 2D."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(64, 64))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            skip_statistics=True,
        )
        assert torch.all(result >= -1), "Found values below -1"
        assert torch.all(result <= 1), "Found values above 1"

    def test_output_values_between_neg1_and_1_3d(self):
        """Cosine-similarity values should be in [-1, 1] for 3D."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            skip_statistics=True,
        )
        assert torch.all(result >= -1), "Found values below -1"
        assert torch.all(result <= 1), "Found values above 1"

    @pytest.mark.parametrize("batch_size", [2, 4])
    def test_batch_output_shape(self, batch_size):
        """Batch dim should match number of input maps."""
        spatial_shape = (32, 32, 32)
        half1, half2 = _make_random_bzyx_half_maps(
            spatial_shape=spatial_shape,
            batch_size=batch_size,
        )
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            skip_statistics=True,
        )
        expected = _expected_shape(
            spatial_shape,
            step_size=3,
            batch_size=batch_size,
        )
        msg = f"batch_size={batch_size}: expected {expected}, got {result.shape}"
        assert result.shape == expected, msg

    def test_identical_maps_high_correlation(self):
        """Identical half-maps should yield correlation values near 1."""
        half1, _ = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half1,
            skip_statistics=True,
        )
        assert torch.all(result > 0.9), (
            "Identical maps should give high correlation"
            f"got min={result.min().item():.4f}"
        )

    def test_permutation_params_do_not_affect_output(self):
        """do_phase_permutation, n_random_maps, reference_dist_size are
        irrelevant when skip_statistics=True; output must be identical."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        common_kw = {
            "apix": APIX,
            "windows_radii": WINDOWS_RADII,
            "resolutions": RESOLUTIONS,
            "batch_half_map1": half1,
            "batch_half_map2": half2,
            "skip_statistics": True,
        }
        result_a = estimate_local_resolution(
            **common_kw,
            do_phase_permutation=True,
            n_random_maps=0,
            reference_dist_size=10000,
        )
        result_b = estimate_local_resolution(
            **common_kw,
            do_phase_permutation=False,
            n_random_maps=50,
            reference_dist_size=500,
        )
        assert torch.equal(result_a, result_b), (
            "do_phase_permutation / n_random_maps / reference_dist_size should "
            "have no effect when skip_statistics=True"
        )

    def test_differs_from_default_statistics(self):
        """skip_statistics=True output should differ from skip_statistics=False."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        common_kw = {
            "apix": APIX,
            "windows_radii": WINDOWS_RADII,
            "resolutions": RESOLUTIONS,
            "batch_half_map1": half1,
            "batch_half_map2": half2,
        }
        result_skip = estimate_local_resolution(**common_kw, skip_statistics=True)
        result_default = estimate_local_resolution(**common_kw, skip_statistics=False)
        assert not torch.equal(result_skip, result_default), (
            "skip_statistics=True and False should produce different outputs"
        )

    def test_determinism_same_seed(self):
        """Two runs with the same seed must produce identical output."""
        kwargs = {
            "apix": APIX,
            "windows_radii": WINDOWS_RADII,
            "resolutions": RESOLUTIONS,
            "skip_statistics": True,
        }
        half1_a, half2_a = _make_random_bzyx_half_maps(
            spatial_shape=(32, 32, 32), seed=42
        )
        result_a = estimate_local_resolution(
            batch_half_map1=half1_a, batch_half_map2=half2_a, **kwargs
        )
        half1_b, half2_b = _make_random_bzyx_half_maps(
            spatial_shape=(32, 32, 32), seed=42
        )
        result_b = estimate_local_resolution(
            batch_half_map1=half1_b, batch_half_map2=half2_b, **kwargs
        )
        assert torch.equal(result_a, result_b), (
            "Two runs with identical seed produced different results — "
            "the computation may not be deterministic"
        )


BASELINES_DIR = Path(__file__).parent / "baselines"


class TestRegressionBaselines:
    """Regression tests: fixed seed → near-exact output.

    These tests require baseline ``.pt`` files in ``baselines/``.
    """

    @staticmethod
    def _load_baseline(name: str) -> torch.Tensor:
        path = BASELINES_DIR / name
        if not path.exists():
            pytest.skip(
                f"Baseline file {path} not found. Generate regression baselines first."
            )
        return torch.load(path, weights_only=True)

    def test_regression_3d_phase_permutation(self):
        """3D output with phase permutation must match stored baseline."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            do_phase_permutation=True,
        )
        expected = self._load_baseline("regression_3d_phase_seed42.pt")
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                "3D phase-permutation output changed vs stored baseline. "
                "If intentional, re-run generate_regression_baselines.py."
            ),
        )

    def test_regression_3d_real_space_shuffle(self):
        """3D output with real-space shuffle must match stored baseline."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            do_phase_permutation=False,
        )
        expected = self._load_baseline("regression_3d_shuffle_seed42.pt")
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                "3D real-space-shuffle output changed vs stored baseline. "
                "If intentional, re-run generate_regression_baselines.py."
            ),
        )

    def test_regression_2d_phase_permutation(self):
        """2D output with phase permutation must match stored baseline."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(64, 64))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            do_phase_permutation=True,
        )
        expected = self._load_baseline("regression_2d_phase_seed42.pt")
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                "2D phase-permutation output changed vs stored baseline. "
                "If intentional, re-run generate_regression_baselines.py."
            ),
        )

    def test_determinism_same_seed(self):
        """Two runs with the same seed must produce identical output."""
        kwargs = {
            "apix": APIX,
            "windows_radii": WINDOWS_RADII,
            "resolutions": RESOLUTIONS,
            "do_phase_permutation": True,
        }
        half1_a, half2_a = _make_random_bzyx_half_maps(
            spatial_shape=(32, 32, 32), seed=42
        )
        result_a = estimate_local_resolution(
            batch_half_map1=half1_a, batch_half_map2=half2_a, **kwargs
        )
        half1_b, half2_b = _make_random_bzyx_half_maps(
            spatial_shape=(32, 32, 32), seed=42
        )
        result_b = estimate_local_resolution(
            batch_half_map1=half1_b, batch_half_map2=half2_b, **kwargs
        )
        assert torch.equal(result_a, result_b), (
            "Two runs with identical seed produced different results — "
            "the computation may not be deterministic"
        )

    def test_regression_3d_skip_statistics(self):
        """3D output with skip_statistics must match stored baseline."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(32, 32, 32))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            skip_statistics=True,
        )
        expected = self._load_baseline("regression_3d_skip_statistics_seed42.pt")
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                "3D skip_statistics output changed vs stored baseline. "
                "If intentional, re-run generate_regression_baselines.py."
            ),
        )

    def test_regression_2d_skip_statistics(self):
        """2D output with skip_statistics must match stored baseline."""
        half1, half2 = _make_random_bzyx_half_maps(spatial_shape=(64, 64))
        result = estimate_local_resolution(
            apix=APIX,
            windows_radii=WINDOWS_RADII,
            resolutions=RESOLUTIONS,
            batch_half_map1=half1,
            batch_half_map2=half2,
            skip_statistics=True,
        )
        expected = self._load_baseline("regression_2d_skip_statistics_seed42.pt")
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
            msg=(
                "2D skip_statistics output changed vs stored baseline. "
                "If intentional, re-run generate_regression_baselines.py."
            ),
        )


class TestComputeResolutionValidation:
    """Invalid inputs raise pydantic.ValidationError."""

    def _valid_half_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)

    def _call(self, **overrides: Any) -> None:
        h1, h2 = self._valid_half_maps()
        kw: dict[str, Any] = {
            "apix": APIX,
            "windows_radii": [3.0],
            "resolutions": [5.0],
            "batch_half_map1": h1,
            "batch_half_map2": h2,
        }
        kw.update(overrides)
        estimate_local_resolution(**kw)

    def test_wrong_ndim(self):
        half1 = torch.zeros(1, 8, 8)
        half2 = torch.zeros(1, 8, 8)
        with pytest.raises(ValidationError):
            self._call(batch_half_map1=half1, batch_half_map2=half2)

    def test_shape_mismatch(self):
        half1 = torch.zeros(1, 1, 8, 8)
        half2 = torch.zeros(1, 1, 4, 4)
        with pytest.raises(ValidationError):
            self._call(batch_half_map1=half1, batch_half_map2=half2)

    def test_empty_batch(self):
        half1 = torch.zeros(0, 1, 4, 4)
        half2 = torch.zeros(0, 1, 4, 4)
        with pytest.raises(ValidationError):
            self._call(batch_half_map1=half1, batch_half_map2=half2)

    def test_apix_non_positive(self):
        with pytest.raises(ValidationError):
            self._call(apix=0.0)

    def test_empty_windows_radii(self):
        with pytest.raises(ValidationError):
            self._call(windows_radii=[], resolutions=[])

    def test_windows_resolutions_length_mismatch(self):
        with pytest.raises(ValidationError):
            self._call(windows_radii=[1.0, 2.0], resolutions=[5.0])

    def test_non_positive_window_radius(self):
        with pytest.raises(ValidationError):
            self._call(windows_radii=[0.0], resolutions=[5.0])

    def test_non_positive_resolution(self):
        with pytest.raises(ValidationError):
            self._call(windows_radii=[3.0], resolutions=[0.0])

    def test_negative_gpu_id(self):
        with pytest.raises(ValidationError):
            self._call(gpu_id=-1)

    @pytest.mark.skipif(
        torch.cuda.is_available(),
        reason="requires CUDA to be unavailable",
    )
    def test_gpu_id_requires_cuda(self):
        with pytest.raises(ValidationError, match="CUDA"):
            self._call(gpu_id=0)
