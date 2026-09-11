"""Project the 4v6x ribosome and reconstruct it back from those projections.

An end-to-end tour of the experimental real-space API:

1. **Simulate** a density map for the 80S ribosome (PDB ``4v6x``) straight from
   the RCSB entry with ``ttsim3d``.
2. **Project** it at random orientations with :func:`project_3d_to_2d` -- the
   forward model of single-particle cryo-EM.
3. **Reconstruct** the volume from those projections and their (here, known)
   orientations with :func:`backproject_2d_to_3d`.
4. **Score** the result by Fourier shell correlation against the volume we
   started from.

Run twice over, to make two different points:

- **Noise-free**, the reconstruction is correct to the Nyquist limit. That is a
  statement about the projector/backprojector pair being exact adjoints, and it
  is the run to watch if you are checking the kernels.
- **With noise** at a cryo-EM-like SNR, resolution is limited by how much signal
  you collected, and improves as you add particles. That is the run that looks
  like real life.

Prerequisites::

    uv sync --group test                                 # ttsim3d, FSC, matplotlib
    uv pip install 'mojo==1.0.0b2' --prerelease allow    # the Mojo kernels

Run::

    uv run --no-sync python \
        packages/primitives/torch-fourier-slice/examples/ribosome_reconstruction_demo.py

Environment:
    ``DEMO_DEVICE``   ``cpu`` (default), ``mps`` or ``cuda`` -- the kernels follow
                      whichever device the input tensor is on.
    ``DEMO_NO_SHOW``  set to skip the interactive window (the PNG is still written).

The PDB download and the simulated volume are cached in ``examples/.cache/``, so
only the first run pays for them.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import urllib3
from scipy.spatial.transform import Rotation
from torch_fourier_shell_correlation import fourier_shell_correlation
from ttsim3d.models import Simulator, SimulatorConfig

from torch_fourier_slice.experimental import (
    backproject_2d_to_3d,
    mojo_kernels_available,
    project_3d_to_2d,
)

PDB_ID = "4v6x"  # 80S ribosome
PIXEL_SPACING = 3.0  # angstroms per pixel -> Nyquist at 6.0 A
BOX = 128  # 384 A box, comfortably larger than the ~300 A ribosome
SNR = 0.1  # signal-to-noise (variance ratio) of the noisy projections
PARTICLE_COUNTS = (100, 400, 1600)
CACHE = Path(__file__).parent / ".cache"


def simulate_ribosome() -> torch.Tensor:
    """Download the PDB entry and simulate its density map (cached on disk)."""
    CACHE.mkdir(exist_ok=True)
    volume_path = CACHE / f"{PDB_ID}_box{BOX}_{PIXEL_SPACING:g}apx.pt"
    if volume_path.exists():
        return torch.load(volume_path)

    structure_path = CACHE / f"{PDB_ID}.cif"
    if not structure_path.exists():
        print(f"downloading {PDB_ID} from RCSB (~28 MB, once) ...")
        response = urllib3.PoolManager().request(
            "GET", f"https://files.rcsb.org/download/{PDB_ID}.cif"
        )
        structure_path.write_bytes(response.data)

    print(f"simulating a {BOX}^3 volume at {PIXEL_SPACING} A/px ...")
    simulator = Simulator(
        pdb_filepath=str(structure_path),
        pixel_spacing=PIXEL_SPACING,
        volume_shape=(BOX, BOX, BOX),
        b_factor_scaling=1.0,
        additional_b_factor=0.0,
        simulator_config=SimulatorConfig(
            voltage=300.0,
            apply_dose_weighting=True,
            dose_start=0.0,
            dose_end=35.0,
            upsampling=2,
        ),
    )
    volume = simulator.run()
    torch.save(volume, volume_path)
    return volume


def random_rotations(n: int, seed: int = 0) -> torch.Tensor:
    """``(n, 3, 3)`` uniformly random rotation matrices in the kernels' convention.

    The experimental API multiplies **zyx**-ordered coordinate vectors, while
    ``scipy`` hands back matrices for xyz-ordered ones. Reversing both the rows
    and the columns re-expresses the same physical rotation in zyx.
    """
    xyz = torch.tensor(
        Rotation.random(n, random_state=seed).as_matrix(), dtype=torch.float32
    )
    return torch.flip(xyz, dims=(-2, -1)).contiguous()


def add_noise(images: torch.Tensor, snr: float, seed: int = 0) -> torch.Tensor:
    """Add white Gaussian noise at a given signal-to-noise *variance* ratio.

    Drawn on the CPU and then moved, so a run on the GPU sees the same noise as a
    run on the CPU and the two are directly comparable.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(images.shape, generator=generator).to(images.device)
    return images + noise * (images.std() / snr**0.5)


def reconstruct(volume: torch.Tensor, n_particles: int, snr: float | None):
    """Project ``volume`` at ``n_particles`` random orientations, then rebuild it.

    This is the whole point of the example: the same rotation matrices drive the
    forward projection and the backprojection, so the reconstruction problem here
    is "known poses, unknown volume".
    """
    rotations = random_rotations(n_particles).to(volume.device)

    images = project_3d_to_2d(volume, rotations)  # (n_particles, box, box), real
    if snr is not None:
        images = add_noise(images, snr)

    reconstruction = backproject_2d_to_3d(images, rotations)  # (box, box, box)
    return images, reconstruction


def lowpass(volume: torch.Tensor, resolution: float) -> torch.Tensor:
    """Low-pass a volume to ``resolution`` angstroms.

    A reconstruction carries noise well beyond the resolution it actually
    supports, so showing the raw voxels understates it. Filtering to the measured
    resolution is what you would look at in practice.
    """
    cutoff = PIXEL_SPACING / resolution  # cycles per pixel
    kz = torch.fft.fftfreq(volume.shape[-3], device=volume.device)
    ky = torch.fft.fftfreq(volume.shape[-2], device=volume.device)
    kx = torch.fft.rfftfreq(volume.shape[-1], device=volume.device)
    radius = (
        kz[:, None, None] ** 2 + ky[None, :, None] ** 2 + kx[None, None, :] ** 2
    ).sqrt()
    dft = torch.fft.rfftn(volume, dim=(-3, -2, -1)) * (radius <= cutoff)
    return torch.fft.irfftn(dft, s=volume.shape[-3:], dim=(-3, -2, -1))


def resolution_from_fsc(
    reconstruction: torch.Tensor,
    reference: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """FSC curve, its frequency axis (1/A), and the resolution at ``threshold``."""
    curve = fourier_shell_correlation(reconstruction.cpu(), reference.cpu())
    # shell i sits at i/box cycles per pixel; divide by the pixel size for 1/A
    frequency = torch.arange(len(curve)) / (BOX * PIXEL_SPACING)
    resolved = (curve >= threshold) & (frequency > 0)
    resolution = 1 / frequency[resolved][-1].item() if resolved.any() else float("inf")
    return resolution, curve, frequency


def main() -> None:
    """Simulate, project, reconstruct, score, and plot."""
    if not mojo_kernels_available():
        raise SystemExit(
            "experimental Mojo kernels unavailable -- "
            "uv pip install 'mojo==1.0.0b2' --prerelease allow"
        )
    device = os.environ.get("DEMO_DEVICE", "cpu")

    volume = simulate_ribosome().to(device)
    nyquist = 2 * PIXEL_SPACING
    print(
        f"\nvolume {tuple(volume.shape)} at {PIXEL_SPACING} A/px "
        f"(Nyquist {nyquist:.1f} A), running on {device}\n"
    )

    # --- noise-free: how good is the projector/backprojector pair itself? ------
    _, clean_reconstruction = reconstruct(volume, 400, snr=None)
    clean_resolution, clean_curve, frequency = resolution_from_fsc(
        clean_reconstruction, volume
    )
    print("noise-free, 400 projections")
    print(f"    FSC=0.5 out to {clean_resolution:.1f} A (Nyquist is {nyquist:.1f} A)")
    print(f"    mean FSC over all shells: {clean_curve.mean():.4f}")

    # --- noisy: resolution is bought with particles ----------------------------
    print(f"\nwith noise at SNR {SNR}")
    noisy = {}
    for n_particles in PARTICLE_COUNTS:
        images, reconstruction = reconstruct(volume, n_particles, snr=SNR)
        resolution, curve, _ = resolution_from_fsc(reconstruction, volume)
        noisy[n_particles] = (images, reconstruction, curve, resolution)
        print(f"    {n_particles:5d} projections -> FSC=0.5 at {resolution:5.1f} A")

    _figure(volume, clean_curve, clean_reconstruction, noisy, frequency)


def _figure(volume, clean_curve, clean_reconstruction, noisy, frequency) -> None:
    """Example projections, central slices, and the FSC curves."""
    try:
        import matplotlib

        if os.environ.get("DEMO_NO_SHOW"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -- skipping the figure")
        return

    most = max(noisy)
    noisy_images, noisy_reconstruction, _, noisy_resolution = noisy[most]
    middle = volume.shape[0] // 2

    def slice_of(v):
        return v[middle].detach().cpu().numpy()

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5))

    # row 1 -- what the reconstruction is built from
    for ax, image in zip(
        axes[0], noisy_images[:3].detach().cpu().numpy(), strict=False
    ):
        ax.imshow(image, cmap="gray")
    axes[0, 0].set_ylabel("projections", fontsize=11)
    axes[0, 1].set_title(
        f"noisy projections at random orientations (SNR {SNR})", fontsize=11
    )

    # row 2 -- ground truth vs the two reconstructions, on a shared scale
    truth_slice = slice_of(volume)
    vmin, vmax = truth_slice.min(), truth_slice.max()
    # second title line keeps all three panels vertically aligned
    panels = [
        (truth_slice, "ground truth\n"),
        (slice_of(clean_reconstruction), "reconstruction, noise-free\n"),
        (
            slice_of(lowpass(noisy_reconstruction, noisy_resolution)),
            f"reconstruction, {most} noisy\nlow-passed to {noisy_resolution:.0f} Å",
        ),
    ]
    for ax, (image, title) in zip(axes[1], panels, strict=True):
        normalised = (image - image.mean()) / image.std()
        normalised = normalised * truth_slice.std() + truth_slice.mean()
        ax.imshow(normalised, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
    axes[1, 0].set_ylabel("central slices", fontsize=11)

    for ax in axes[:2].flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # row 3 -- FSC, spanning the full width
    for ax in axes[2]:
        ax.remove()
    fsc_ax = fig.add_subplot(3, 1, 3)
    x = frequency.numpy()
    fsc_ax.plot(x, clean_curve.numpy(), lw=2.2, color="k", label="noise-free, 400")
    for n_particles in sorted(noisy):
        fsc_ax.plot(
            x,
            noisy[n_particles][2].numpy(),
            lw=1.6,
            label=f"SNR {SNR}, {n_particles} particles",
        )
    fsc_ax.axhline(0.5, ls="--", lw=1, color="grey")
    fsc_ax.text(x[-1], 0.52, "FSC = 0.5", ha="right", fontsize=9, color="grey")
    fsc_ax.set_xlim(0, x[-1])
    fsc_ax.set_ylim(-0.05, 1.05)
    fsc_ax.set_xlabel("spatial frequency (1/Å)")
    fsc_ax.set_ylabel("FSC vs ground truth")
    fsc_ax.legend(fontsize=9, loc="lower left")

    # a second axis labelled in resolution is what people actually read; ticks are
    # placed explicitly because 1/f is unusable as a transform near f = 0
    resolution_ax = fsc_ax.twiny()
    resolution_ax.set_xlim(fsc_ax.get_xlim())
    candidates = (100, 50, 30, 20, 15, 10, 8, 2 * PIXEL_SPACING)
    ticks = [r for r in candidates if 1 / r <= x[-1]]
    resolution_ax.set_xticks([1 / r for r in ticks])
    resolution_ax.set_xticklabels([f"{r:g}" for r in ticks], fontsize=9)
    resolution_ax.set_xlabel("resolution (Å)")

    fig.suptitle(
        f"{PDB_ID} ribosome: forward-project at random orientations, "
        f"then reconstruct\nnoise-free reaches Nyquist "
        f"({2 * PIXEL_SPACING:.0f} Å); {most} noisy particles reach "
        f"{noisy_resolution:.0f} Å",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    out = Path(__file__).parent / "ribosome_reconstruction_demo.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")
    if not os.environ.get("DEMO_NO_SHOW"):
        plt.show()


if __name__ == "__main__":
    main()
