"""Tests for aberration helpers and Zernike handling."""

import pytest
import torch

from torch_ctf import (
    apply_astigmatism_to_defocus,
    apply_even_zernikes,
    apply_odd_zernikes,
    beam_tilt_to_zernike_coeffs,
    resolve_odd_zernikes,
)


def test_beam_tilt_to_zernike_coeffs():
    """Test beam tilt to Zernike coefficients conversion."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = beam_tilt_to_zernike_coeffs(
        beam_tilt_mrad=beam_tilt_mrad,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z31c" in result
    assert "Z31s" in result
    assert result["Z31c"].shape == (1,)
    assert result["Z31s"].shape == (1,)
    assert torch.all(torch.isfinite(result["Z31c"]))
    assert torch.all(torch.isfinite(result["Z31s"]))


def test_apply_astigmatism_to_defocus_direct():
    """Test public astigmatism defocus adjustment helper directly."""
    h, w = 6, 6
    fft_freq_grid = torch.zeros((h, w, 2), dtype=torch.float32)
    fft_freq_grid[..., 0] = torch.linspace(-0.2, 0.2, steps=w)
    fft_freq_grid[..., 1] = torch.linspace(-0.2, 0.2, steps=h).unsqueeze(-1)
    fft_freq_grid_squared = torch.sum(fft_freq_grid**2, dim=-1)

    defocus = torch.tensor([2.0, 3.0], dtype=torch.float32).view(2, 1, 1)
    astigmatism = torch.tensor([0.0, 0.4], dtype=torch.float32)
    astigmatism_angle = torch.tensor([0.0, 45.0], dtype=torch.float32)

    adjusted = apply_astigmatism_to_defocus(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        fft_freq_grid=fft_freq_grid,
        fft_freq_grid_squared=fft_freq_grid_squared,
    )

    assert adjusted.shape == (2, h, w)
    assert torch.all(torch.isfinite(adjusted))
    assert torch.allclose(adjusted[0], torch.full((h, w), 2.0), atol=1e-6)
    assert not torch.allclose(adjusted[1], torch.full((h, w), 3.0), atol=1e-6)


def test_resolve_odd_zernikes_beam_tilt_only():
    """Test resolve_odd_zernikes with only beam tilt."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=beam_tilt_mrad,
        odd_zernike_coeffs=None,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z31c" in result
    assert "Z31s" in result


def test_resolve_odd_zernikes_zernike_only():
    """Test resolve_odd_zernikes with only Zernike coefficients."""
    odd_zernike_coeffs = {"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)}
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=None,
        odd_zernike_coeffs=odd_zernike_coeffs,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert result["Z31c"] == odd_zernike_coeffs["Z31c"]
    assert result["Z31s"] == odd_zernike_coeffs["Z31s"]


def test_resolve_odd_zernikes_both():
    """Test resolve_odd_zernikes with both beam tilt and Zernike coefficients."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    odd_zernike_coeffs = {"Z31c": torch.tensor(0.1), "Z31s": torch.tensor(0.2)}
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    with pytest.warns(RuntimeWarning, match="Both beam tilt and Zernike"):
        result = resolve_odd_zernikes(
            beam_tilt_mrad=beam_tilt_mrad,
            odd_zernike_coeffs=odd_zernike_coeffs,
            voltage_kv=voltage_kv,
            spherical_aberration_mm=spherical_aberration_mm,
        )

    # Zernike coefficients should override beam tilt
    assert isinstance(result, dict)
    assert result["Z31c"] == odd_zernike_coeffs["Z31c"]
    assert result["Z31s"] == odd_zernike_coeffs["Z31s"]


def test_resolve_odd_zernikes_none():
    """Test resolve_odd_zernikes with no inputs."""
    result = resolve_odd_zernikes(
        beam_tilt_mrad=None,
        odd_zernike_coeffs=None,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result is None


def test_apply_odd_zernikes():
    """Test applying odd Zernike coefficients."""
    odd_zernikes = {
        "Z31c": torch.tensor(0.1),
        "Z31s": torch.tensor(0.2),
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_trefoil():
    """Test applying trefoil Zernike coefficients."""
    odd_zernikes = {
        "Z33c": torch.tensor(0.1),
        "Z33s": torch.tensor(0.2),
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_invalid():
    """Test applying invalid Zernike coefficient raises error."""
    odd_zernikes = {"Z99": torch.tensor(0.1)}
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(ValueError, match="Unknown odd Zernike"):
        apply_odd_zernikes(
            odd_zernikes=odd_zernikes,
            rho=rho,
            theta=theta,
            voltage_kv=torch.tensor(300.0),
            spherical_aberration_mm=torch.tensor(2.7),
        )


def test_apply_even_zernikes():
    """Test applying even Zernike coefficients."""
    even_zernikes = {
        "Z44c": 0.1,  # plain float
        "Z44s": 0.2,  # plain float
        "Z60": 0.3,  # plain float
    }
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_even_zernikes(
        even_zernikes=even_zernikes,
        total_phase_shift=total_phase_shift,
        rho=rho,
        theta=theta,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # Result should be different from input
    assert not torch.allclose(result, total_phase_shift)


def test_apply_even_zernikes_invalid():
    """Test applying invalid even Zernike coefficient raises error."""
    even_zernikes = {"Z99": 0.1}  # plain float
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(ValueError, match="Unknown even Zernike"):
        apply_even_zernikes(
            even_zernikes=even_zernikes,
            total_phase_shift=total_phase_shift,
            rho=rho,
            theta=theta,
        )


def test_apply_even_zernikes_tensor_coeffs():
    """Test applying even Zernike coefficients with tensor values."""
    even_zernikes = {
        "Z44c": torch.tensor(0.1),
        "Z44s": torch.tensor(0.2),
        "Z60": torch.tensor(0.3),
    }
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_even_zernikes(
        even_zernikes=even_zernikes,
        total_phase_shift=total_phase_shift,
        rho=rho,
        theta=theta,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # Result should be different from input
    assert not torch.allclose(result, total_phase_shift)


def test_apply_even_zernikes_mixed_coeffs():
    """Test applying even Zernike coefficients with mixed float and tensor values."""
    even_zernikes = {
        "Z44c": 0.1,  # float
        "Z44s": torch.tensor(0.2),  # tensor
        "Z60": 0.3,  # float
    }
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_even_zernikes(
        even_zernikes=even_zernikes,
        total_phase_shift=total_phase_shift,
        rho=rho,
        theta=theta,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_even_zernikes_type_error():
    """Test applying even Zernike coefficients with invalid type raises TypeError."""
    even_zernikes = {"Z44c": "invalid"}  # string instead of float/tensor
    total_phase_shift = torch.ones((10, 10)) * 0.5
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(
        TypeError, match=r"Zernike coefficient must be float or torch\.Tensor"
    ):
        apply_even_zernikes(
            even_zernikes=even_zernikes,
            total_phase_shift=total_phase_shift,
            rho=rho,
            theta=theta,
        )


def test_apply_odd_zernikes_none():
    """Test applying odd Zernike coefficients with None returns zeros."""
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=None,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.allclose(result, torch.zeros_like(rho))


def test_apply_odd_zernikes_float_coeffs():
    """Test applying odd Zernike coefficients with float values."""
    odd_zernikes = {
        "Z31c": 0.1,  # plain float
        "Z31s": 0.2,  # plain float
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_mixed_coeffs():
    """Test applying odd Zernike coefficients with mixed float and tensor values."""
    odd_zernikes = {
        "Z31c": 0.1,  # float
        "Z31s": torch.tensor(0.2),  # tensor
    }
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    result = apply_odd_zernikes(
        odd_zernikes=odd_zernikes,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))


def test_apply_odd_zernikes_with_beam_tilt():
    """Test applying odd Zernike coefficients with beam_tilt_mrad parameter."""
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])

    result = apply_odd_zernikes(
        odd_zernikes=None,
        rho=rho,
        theta=theta,
        voltage_kv=torch.tensor(300.0),
        spherical_aberration_mm=torch.tensor(2.7),
        beam_tilt_mrad=beam_tilt_mrad,
    )

    assert result.shape == (10, 10)
    assert torch.all(torch.isfinite(result))
    # Should not be all zeros when beam tilt is provided
    assert not torch.allclose(result, torch.zeros_like(rho))


def test_apply_odd_zernikes_type_error():
    """Test applying odd Zernike coefficients with invalid type raises TypeError."""
    odd_zernikes = {"Z31c": "invalid"}  # string instead of float/tensor
    rho = torch.ones((10, 10)) * 0.5
    theta = torch.linspace(0, 2 * torch.pi, 100).reshape(10, 10)

    with pytest.raises(
        TypeError, match=r"Zernike coefficient must be float or torch\.Tensor"
    ):
        apply_odd_zernikes(
            odd_zernikes=odd_zernikes,
            rho=rho,
            theta=theta,
            voltage_kv=torch.tensor(300.0),
            spherical_aberration_mm=torch.tensor(2.7),
        )


def test_resolve_odd_zernikes_with_trefoil():
    """Test resolve_odd_zernikes with trefoil coefficients (Z33c, Z33s)."""
    odd_zernike_coeffs = {
        "Z33c": torch.tensor(0.1),
        "Z33s": torch.tensor(0.2),
    }
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=None,
        odd_zernike_coeffs=odd_zernike_coeffs,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z33c" in result
    assert "Z33s" in result
    assert result["Z33c"] == odd_zernike_coeffs["Z33c"]
    assert result["Z33s"] == odd_zernike_coeffs["Z33s"]


def test_resolve_odd_zernikes_beam_tilt_with_trefoil():
    """Test resolve_odd_zernikes with beam tilt and trefoil coefficients."""
    beam_tilt_mrad = torch.tensor([[1.0, 2.0]])
    odd_zernike_coeffs = {
        "Z33c": torch.tensor(0.1),
        "Z33s": torch.tensor(0.2),
    }
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = resolve_odd_zernikes(
        beam_tilt_mrad=beam_tilt_mrad,
        odd_zernike_coeffs=odd_zernike_coeffs,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    # Should have both beam tilt coefficients and trefoil
    assert "Z31c" in result
    assert "Z31s" in result
    assert "Z33c" in result
    assert "Z33s" in result
    assert result["Z33c"] == odd_zernike_coeffs["Z33c"]
    assert result["Z33s"] == odd_zernike_coeffs["Z33s"]


def test_beam_tilt_to_zernike_coeffs_broadcasting():
    """Test beam_tilt_to_zernike_coeffs with different tensor shapes."""
    # Test with batched beam tilt
    beam_tilt_mrad = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
    voltage_kv = torch.tensor(300.0)
    spherical_aberration_mm = torch.tensor(2.7)

    result = beam_tilt_to_zernike_coeffs(
        beam_tilt_mrad=beam_tilt_mrad,
        voltage_kv=voltage_kv,
        spherical_aberration_mm=spherical_aberration_mm,
    )

    assert isinstance(result, dict)
    assert "Z31c" in result
    assert "Z31s" in result
    assert result["Z31c"].shape == (2,)
    assert result["Z31s"].shape == (2,)
    assert torch.all(torch.isfinite(result["Z31c"]))
    assert torch.all(torch.isfinite(result["Z31s"]))
