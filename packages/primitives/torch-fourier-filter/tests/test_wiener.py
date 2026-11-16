import numpy as np
import pytest
import torch
from torch_ctf import calculate_ctf_2d

from torch_fourier_filter.wiener import (
    ctf_deconvolution_filter,
    ctf_deconvolve,
    wiener_filter,
)


def _reference_filter(
    shape, pixel_size, defocus, strength, snr_falloff, highpass_nyquist
) -> np.ndarray:
    """Independent NumPy implementation on a full, unshifted DFT grid.

    The CTF is written as amplitude-contrast mixing of cos and sin (positive at
    zero frequency), with frequency indexed per axis as a fraction of Nyquist.
    """
    axes = []
    for n in shape:
        idx = np.arange(n) - n // 2  # -n/2 .. n/2 - 1 for even n
        axes.append(np.fft.ifftshift(idx / (n // 2)))
    grids = np.meshgrid(*axes, indexing="ij")
    r = np.minimum(1.0, np.sqrt(sum(g**2 for g in grids)))

    voltage, cs, amplitude = 300e3, 2.7e-3, 0.07
    wavelength = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    k2 = (r / (2 * pixel_size * 1e-10)) ** 2
    w = np.pi / 2 * (wavelength**3 * cs * k2**2 - 2 * wavelength * defocus * 1e-6 * k2)
    ctf = amplitude * np.cos(w) - np.sqrt(1 - amplitude**2) * np.sin(w)

    highpass = 1 - np.cos(np.minimum(1.0, r / highpass_nyquist) * np.pi)
    snr = 10 ** (3 * strength) * np.exp(-r * snr_falloff * 100 / pixel_size) * highpass
    return ctf * snr / (ctf**2 * snr + 1)


@pytest.mark.parametrize("shape", [(64, 64), (32, 32, 32), (16, 32, 48)])
def test_ctf_deconvolution_filter_matches_reference(shape):
    kwargs = {
        "pixel_size": 4.0,
        "defocus": 4.0,
        "strength": 0.8,
        "snr_falloff": 1.1,
        "highpass_nyquist": 0.02,
    }
    expected = _reference_filter(shape, **kwargs)
    result = ctf_deconvolution_filter(
        image_shape=shape, rfft=False, fftshift=False, **kwargs
    )
    assert result.shape == shape
    scale = np.abs(expected).max()
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=1e-4 * scale)


def test_ctf_deconvolution_filter_layouts():
    shape = (16, 32, 32)
    full = ctf_deconvolution_filter(shape, 4.0, 3.0, rfft=False, fftshift=False)
    half = ctf_deconvolution_filter(shape, 4.0, 3.0, rfft=True, fftshift=False)
    shifted = ctf_deconvolution_filter(shape, 4.0, 3.0, rfft=False, fftshift=True)
    torch.testing.assert_close(half, full[..., : shape[-1] // 2 + 1])
    torch.testing.assert_close(shifted, torch.fft.fftshift(full))


def test_ctf_deconvolution_filter_batched_parameters():
    defocus = torch.tensor([2.0, 3.0, 5.0])
    strength = torch.tensor([0.5, 0.8, 1.0])
    batched = ctf_deconvolution_filter(
        (32, 32), 2.0, defocus=defocus, strength=strength, snr_falloff=1.2
    )
    assert batched.shape == (3, 32, 17)
    for i in range(3):
        single = ctf_deconvolution_filter(
            (32, 32),
            2.0,
            defocus=float(defocus[i]),
            strength=float(strength[i]),
            snr_falloff=1.2,
        )
        torch.testing.assert_close(batched[i], single)


def test_ctf_deconvolution_filter_batched_3d():
    result = ctf_deconvolution_filter((8, 16, 16), 10.0, torch.tensor([3.0, 6.0]))
    assert result.shape == (2, 8, 16, 9)


def test_ctf_deconvolution_filter_removes_zero_frequency():
    result = ctf_deconvolution_filter((32, 32), 4.0, 3.0)
    assert result[0, 0] == 0


def test_ctf_deconvolution_filter_phase_flipped_is_non_negative():
    result = ctf_deconvolution_filter((64, 64), 2.0, 3.0, phase_flipped=True)
    assert (result >= 0).all()
    unflipped = ctf_deconvolution_filter((64, 64), 2.0, 3.0)
    assert (unflipped < 0).any()


def test_ctf_deconvolution_filter_snr_override():
    snr = 10.0
    result = ctf_deconvolution_filter((32, 32), 4.0, 3.0, snr=snr)
    ctf = -calculate_ctf_2d(
        defocus=3.0,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        phase_shift=0.0,
        pixel_size=4.0,
        image_shape=(32, 32),
        rfft=True,
        fftshift=False,
    )
    # grid corners beyond Nyquist are clamped in the filter but not in the CTF
    inside = torch.fft.rfftfreq(32)[None, :] ** 2 + torch.fft.fftfreq(32)[:, None] ** 2
    inside = inside <= 0.25
    expected = ctf * snr / (ctf**2 * snr + 1)
    torch.testing.assert_close(result[inside], expected[inside], atol=5e-5, rtol=1e-3)


def test_ctf_deconvolution_filter_is_differentiable():
    strength = torch.tensor(0.8, requires_grad=True)
    snr_falloff = torch.tensor([1.0, 1.2], requires_grad=True)
    result = ctf_deconvolution_filter(
        (32, 32), 4.0, 3.0, strength=strength, snr_falloff=snr_falloff
    )
    result.square().sum().backward()
    assert strength.grad is not None and torch.isfinite(strength.grad)
    assert snr_falloff.grad is not None and (snr_falloff.grad != 0).all()


def test_ctf_deconvolution_filter_rejects_bad_input():
    with pytest.raises(ValueError, match="2D or 3D"):
        ctf_deconvolution_filter((32,), 4.0, 3.0)
    with pytest.raises(ValueError, match="highpass_nyquist"):
        ctf_deconvolution_filter((32, 32), 4.0, 3.0, highpass_nyquist=0.0)


def test_wiener_filter_zero_snr_gives_zero():
    h = torch.linspace(-1, 1, 11)
    result = wiener_filter(h, 0.0)
    assert torch.equal(result, torch.zeros_like(h))


def test_wiener_filter_real_and_complex_agree():
    h = torch.linspace(-1, 1, 11)
    real = wiener_filter(h, 5.0)
    complex_ = wiener_filter(h.to(torch.complex64), 5.0)
    torch.testing.assert_close(complex_.real, real)
    torch.testing.assert_close(complex_.imag, torch.zeros_like(real))


def test_wiener_filter_response():
    h = torch.polar(torch.rand(64) + 0.01, torch.rand(64) * 2 * torch.pi)
    snr = torch.rand(64) * 100
    response = wiener_filter(h, snr) * h
    expected = h.abs() ** 2 * snr / (h.abs() ** 2 * snr + 1)
    torch.testing.assert_close(response.real, expected)
    torch.testing.assert_close(
        response.imag, torch.zeros_like(expected), atol=1e-6, rtol=0
    )
    assert (response.real <= 1).all()


def test_wiener_filter_batched_snr():
    h = torch.rand(3, 8, 5)
    snr = torch.tensor([1.0, 10.0, 100.0]).reshape(3, 1, 1)
    result = wiener_filter(h, snr)
    for i in range(3):
        torch.testing.assert_close(result[i], wiener_filter(h[i], float(snr[i])))


def _blobs(shape: tuple[int, int], n: int, sigma: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    centers = torch.rand(n, 2, generator=generator) * torch.tensor(shape)
    y, x = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing="ij")
    d2 = (y[None] - centers[:, 0, None, None]) ** 2
    d2 = d2 + (x[None] - centers[:, 1, None, None]) ** 2
    return torch.exp(-d2 / (2 * sigma**2)).sum(dim=0)


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def test_ctf_deconvolve_keeps_contrast_and_improves_it():
    shape, pixel_size, defocus = (128, 128), 4.0, 3.0
    potential = _blobs(shape, n=40, sigma=2.0, seed=0)
    ctf = calculate_ctf_2d(
        defocus=defocus,
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        phase_shift=0.0,
        pixel_size=pixel_size,
        image_shape=shape,
        rfft=True,
        fftshift=False,
    )
    image = torch.fft.irfftn(torch.fft.rfftn(potential) * ctf, s=shape)
    # the CTF was applied periodically, so there is no edge jump to pad away
    deconvolved = ctf_deconvolve(image, pixel_size, defocus, ndim=2, padding=0)
    # dark stays dark, and closer to the (negated) potential than before
    raw, restored = _corr(image, potential), _corr(deconvolved, potential)
    assert raw < 0
    assert restored < raw - 0.1
    assert restored < -0.95


def test_ctf_deconvolve_batch_of_images():
    images = torch.randn(3, 64, 64) * torch.tensor([1.0, 2.0, 3.0])[:, None, None]
    images = images + torch.tensor([0.0, 5.0, -5.0])[:, None, None]
    defocus = torch.tensor([2.0, 3.0, 4.0])
    result = ctf_deconvolve(images, 2.0, defocus, ndim=2)
    assert result.shape == images.shape
    torch.testing.assert_close(result.mean(dim=(-2, -1)), images.mean(dim=(-2, -1)))
    torch.testing.assert_close(result.std(dim=(-2, -1)), images.std(dim=(-2, -1)))
    for i in range(3):
        single = ctf_deconvolve(images[i], 2.0, float(defocus[i]), ndim=2)
        torch.testing.assert_close(result[i], single)


def test_ctf_deconvolve_batch_of_volumes():
    volumes = torch.randn(2, 16, 32, 32)
    result = ctf_deconvolve(volumes, 10.0, torch.tensor([3.0, 5.0]), ndim=3)
    assert result.shape == volumes.shape
    single = ctf_deconvolve(volumes[1], 10.0, 5.0, ndim=3)
    torch.testing.assert_close(result[1], single)


def test_ctf_deconvolve_without_rescaling_matches_filter():
    image = torch.randn(32, 32)
    result = ctf_deconvolve(image, 4.0, 3.0, ndim=2, padding=0, preserve_mean_std=False)
    deconvolution_filter = ctf_deconvolution_filter((32, 32), 4.0, 3.0)
    expected = torch.fft.irfftn(
        torch.fft.rfftn(image) * deconvolution_filter, s=(32, 32)
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    ("shape", "ndim"), [((2, 40, 48), 2), ((2, 12, 16, 20), 3), ((30, 33), 2)]
)
def test_ctf_deconvolve_padding_matches_manual_mirror_pad(shape, ndim):
    images = torch.randn(shape)
    defocus = torch.tensor([3.0, 5.0]) if len(shape) > ndim else 3.0
    result = ctf_deconvolve(
        images, 4.0, defocus, ndim=ndim, padding=0.25, preserve_mean_std=False
    )
    assert result.shape == images.shape

    pads = [round(0.25 * n) for n in shape[-ndim:]]
    widths = [(0, 0)] * (len(shape) - ndim) + [(p, p) for p in pads]
    padded = torch.from_numpy(np.pad(images.numpy(), widths, mode="reflect"))
    expected = ctf_deconvolve(
        padded, 4.0, defocus, ndim=ndim, padding=0, preserve_mean_std=False
    )
    crop = tuple(slice(p, p + n) for p, n in zip(pads, shape[-ndim:], strict=True))
    torch.testing.assert_close(result, expected[(..., *crop)])


def test_ctf_deconvolve_padding_suppresses_edge_artefacts():
    # a gradient across the image makes a jump between opposite edges, which
    # an unpadded FFT treats as real; compare to a heavily padded reference
    shape = (128, 128)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, shape[0]), torch.linspace(-1, 1, shape[1]), indexing="ij"
    )
    image = _blobs(shape, n=40, sigma=2.0, seed=0) + 4 * x + 2 * y
    kwargs = {"pixel_size": 4.0, "defocus": 3.0, "ndim": 2, "preserve_mean_std": False}
    reference = ctf_deconvolve(image, padding=0.9, **kwargs)
    edge = torch.ones(shape, dtype=torch.bool)
    edge[8:-8, 8:-8] = False

    def edge_error(padding: float) -> float:
        error = ctf_deconvolve(image, padding=padding, **kwargs) - reference
        return float(error[edge].abs().max() / reference.std())

    assert edge_error(0.25) < 0.25 * edge_error(0.0)


def test_ctf_deconvolve_rejects_bad_input():
    with pytest.raises(ValueError, match="ndim"):
        ctf_deconvolve(torch.randn(8, 8), 4.0, 3.0, ndim=1)
    with pytest.raises(ValueError, match="padding"):
        ctf_deconvolve(torch.randn(8, 8), 4.0, 3.0, ndim=2, padding=1.0)
    with pytest.raises(ValueError, match="padding"):
        ctf_deconvolve(torch.randn(8, 8), 4.0, 3.0, ndim=2, padding=-0.1)
