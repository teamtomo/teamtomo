import math

import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from torch_fourier_filter.phase_randomize import phase_permutation, phase_randomize


def test_phase_randomize():
    # Test parameters
    image_shape_2d = (64, 64)
    image_shape_3d = (32, 64, 64)
    cuton = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random 2D and 3D DFTs
    dft_2d = torch.fft.fft2(torch.rand(image_shape_2d, device=device))
    dft_3d = torch.fft.fftn(torch.rand(image_shape_3d, device=device))

    # Test 2D phase randomization preserves magnitudes
    randomized_2d = phase_randomize(dft_2d, image_shape_2d, cuton=cuton, device=device)
    assert torch.allclose(torch.abs(dft_2d), torch.abs(randomized_2d)), (
        "Magnitude spectrum not preserved for 2D input"
    )

    # Test 3D phase randomization preserves magnitudes
    randomized_3d = phase_randomize(dft_3d, image_shape_3d, cuton=cuton, device=device)
    assert torch.allclose(torch.abs(dft_3d), torch.abs(randomized_3d)), (
        "Magnitude spectrum not preserved for 3D input"
    )

    # Test rfft preserves magnitudes
    dft_2d_rfft = torch.fft.rfft2(torch.rand(image_shape_2d, device=device))
    randomized_2d_rfft = phase_randomize(
        dft_2d_rfft, image_shape_2d, rfft=True, cuton=cuton, device=device
    )
    assert torch.allclose(torch.abs(dft_2d_rfft), torch.abs(randomized_2d_rfft)), (
        "Magnitude spectrum not preserved for rfft input"
    )

    # Ensure phases are randomized only at or above cuton
    freq_grid_2d = fftfreq_grid(
        image_shape_2d, rfft=False, fftshift=False, norm=True, device=device
    )
    freq_mask_2d = torch.abs(freq_grid_2d) >= cuton
    original_phases_2d = torch.angle(dft_2d)
    randomized_phases_2d = torch.angle(randomized_2d)
    assert torch.allclose(
        original_phases_2d[~freq_mask_2d], randomized_phases_2d[~freq_mask_2d]
    ), "Phases not preserved below cuton for 2D input"
    assert not torch.allclose(
        original_phases_2d[freq_mask_2d], randomized_phases_2d[freq_mask_2d]
    ), "Phases not randomized at or above cuton for 2D input"

    freq_grid_3d = fftfreq_grid(
        image_shape_3d, rfft=False, fftshift=False, norm=True, device=device
    )
    freq_mask_3d = torch.abs(freq_grid_3d) >= cuton
    original_phases_3d = torch.angle(dft_3d)
    randomized_phases_3d = torch.angle(randomized_3d)
    assert torch.allclose(
        original_phases_3d[~freq_mask_3d], randomized_phases_3d[~freq_mask_3d]
    ), "Phases not preserved below cuton for 3D input"
    assert not torch.allclose(
        original_phases_3d[freq_mask_3d], randomized_phases_3d[freq_mask_3d]
    ), "Phases not randomized at or above cuton for 3D input"


def test_phase_randomize_cuton_zero():
    image_shape = (32, 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dft = torch.fft.fft2(torch.rand(image_shape, device=device))
    out = phase_randomize(dft, image_shape, cuton=0, device=device)
    assert torch.allclose(torch.abs(dft), torch.abs(out))


def test_phase_permutation():
    image_shape_2d = (64, 64)
    image_shape_3d = (32, 64, 64)
    cuton = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dft_2d = torch.fft.fft2(torch.rand(image_shape_2d, device=device))
    permuted_2d = phase_permutation(dft_2d, image_shape_2d, cuton=cuton, device=device)
    assert torch.allclose(torch.abs(dft_2d), torch.abs(permuted_2d))

    dft_3d = torch.fft.fftn(torch.rand(image_shape_3d, device=device))
    permuted_3d = phase_permutation(dft_3d, image_shape_3d, cuton=cuton, device=device)
    assert torch.allclose(torch.abs(dft_3d), torch.abs(permuted_3d))

    dft_2d_rfft = torch.fft.rfft2(torch.rand(image_shape_2d, device=device))
    permuted_rfft = phase_permutation(
        dft_2d_rfft, image_shape_2d, rfft=True, cuton=cuton, device=device
    )
    assert torch.allclose(torch.abs(dft_2d_rfft), torch.abs(permuted_rfft))

    freq_grid_2d = fftfreq_grid(
        image_shape_2d, rfft=False, fftshift=False, norm=True, device=device
    )
    freq_mask_2d = torch.abs(freq_grid_2d) >= cuton
    op = torch.angle(dft_2d)
    pp = torch.angle(permuted_2d)
    assert torch.allclose(op[~freq_mask_2d], pp[~freq_mask_2d])
    wo = torch.remainder(op[freq_mask_2d].flatten() + math.pi, 2 * math.pi)
    wp = torch.remainder(pp[freq_mask_2d].flatten() + math.pi, 2 * math.pi)
    assert torch.allclose(torch.sort(wo)[0], torch.sort(wp)[0], atol=1e-3, rtol=1e-4)
    assert not torch.allclose(
        torch.exp(1j * op[freq_mask_2d]), torch.exp(1j * pp[freq_mask_2d])
    )


def _max_circular_nn_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Check `a` and `b` hold the same multiset of angles, up to per-element noise.

    Avoids sort-and-compare: DC/Nyquist bins sit at angle exactly 0 or pi, so
    a CPU/CUDA rounding difference there can flip one across the 0/2pi wrap
    and cyclically shift every other bin's sorted rank, producing a large
    spurious diff. Nearest-neighbor circular distance sidesteps that.
    """
    circular_diff = (
        torch.remainder(a.unsqueeze(1) - b.unsqueeze(0) + math.pi, 2 * math.pi)
        - math.pi
    )
    dist = circular_diff.abs()
    return max(dist.min(dim=1).values.max().item(), dist.min(dim=0).values.max().item())


def test_phase_permutation_cuton_zero():
    """Full-spectrum permutation uses _permute_all_phases (no freq grid)."""
    image_shape = (24, 24)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dft = torch.fft.fft2(torch.rand(image_shape, device=device))
    out = phase_permutation(dft, image_shape, cuton=0, device=device)
    assert torch.allclose(torch.abs(dft), torch.abs(out))
    # Phase angle is ill-defined where |dft|≈0. Compare multiset only on bins
    # with meaningful magnitude, in float64.
    mag = torch.abs(dft).flatten()
    mask = mag > 1e-5
    assert mask.any()
    wo = torch.angle(dft.flatten()[mask]).double()
    wp = torch.angle(out.flatten()[mask]).double()
    # 1e-4 clears the ~1e-7 float32 noise floor (CPU and CUDA) but is still
    # far tighter than the ~0.01 rad angle spacing, so real bugs get caught.
    assert _max_circular_nn_distance(wo, wp) < 1e-4


def test_phase_randomize_fftshift():
    image_shape = (32, 32)
    cuton = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(image_shape, device=device)
    dft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    out = phase_randomize(dft, image_shape, cuton=cuton, fftshift=True, device=device)
    assert torch.allclose(torch.abs(dft), torch.abs(out))
    freq_grid = fftfreq_grid(
        image_shape, rfft=False, fftshift=True, norm=True, device=device
    )
    freq_mask = torch.abs(freq_grid) >= cuton
    op = torch.angle(dft)
    out_phases = torch.angle(out)
    assert torch.allclose(op[~freq_mask], out_phases[~freq_mask])
    assert not torch.allclose(op[freq_mask], out_phases[freq_mask])


def test_phase_permutation_fftshift():
    image_shape = (32, 32)
    cuton = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(image_shape, device=device)
    dft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    out = phase_permutation(dft, image_shape, cuton=cuton, fftshift=True, device=device)
    assert torch.allclose(torch.abs(dft), torch.abs(out))
    freq_grid = fftfreq_grid(
        image_shape, rfft=False, fftshift=True, norm=True, device=device
    )
    freq_mask = torch.abs(freq_grid) >= cuton
    op = torch.angle(dft)
    out_phases = torch.angle(out)
    assert torch.allclose(op[~freq_mask], out_phases[~freq_mask])
    wo = torch.remainder(op[freq_mask].flatten() + math.pi, 2 * math.pi)
    wp = torch.remainder(out_phases[freq_mask].flatten() + math.pi, 2 * math.pi)
    assert torch.allclose(torch.sort(wo)[0], torch.sort(wp)[0], atol=1e-3, rtol=1e-4)
