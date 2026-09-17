"""Benchmark the Mojo kernels against the pure-torch reference, and keep a history.

Times the public ``sample_image_*d`` / ``insert_into_image_*d`` functions under
``backend="torch"`` and ``backend="mojo"`` on realistic cryo-EM/ET workloads --
forward alone and forward + backward -- and prints a speed-up table. Every run
appends one JSON line per case to ``results/history.jsonl`` (git sha, timestamp,
device, timings) so the relative performance can be tracked over the course of
the port.

Usage::

    python benchmarks/bench_interpolation.py                 # all cases, cpu
    python benchmarks/bench_interpolation.py --device mps    # + gpu if available
    python benchmarks/bench_interpolation.py --cases sample_3d --quick --label wip
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import torch_image_interpolation as tii
from torch_image_interpolation import (
    insert_into_image_1d,
    insert_into_image_2d,
    insert_into_image_3d,
    sample_image_1d,
    sample_image_2d,
    sample_image_3d,
    use_backend,
)

if TYPE_CHECKING:
    from collections.abc import Callable

print = functools.partial(print, flush=True)  # noqa: A001  (progress in pipes)

HERE = Path(__file__).resolve().parent
HISTORY = HERE / "results" / "history.jsonl"


@dataclass
class Case:
    """One benchmark: how to build its inputs and how to run it."""

    name: str
    build: Callable[[torch.device, bool], dict]  # (device, quick) -> kwargs
    run: Callable[..., torch.Tensor]  # kwargs -> tensor whose sum is the loss
    grad_of: str  # which input receives requires_grad for the backward timing


def _cplx(*shape, device):
    return torch.complex(
        torch.randn(*shape, device=device), torch.randn(*shape, device=device)
    )


def _coords(n, extents, device):
    e = torch.tensor(extents, dtype=torch.float32, device=device) - 1
    return torch.rand(n, len(extents), device=device) * e


def _q(n, quick):
    return n // 8 if quick else n


def _slice_coords(n_slices, extents, device):
    """Coordinates of `n_slices` randomly rotated central slices through a volume.

    The access pattern torch-fourier-slice produces: consecutive samples are
    spatially adjacent, unlike uniformly random coordinates.
    """
    d, h, w = extents
    z, y, x = torch.meshgrid(
        torch.zeros(1, device=device),
        torch.linspace(-(h - 1) / 2, (h - 1) / 2, h, device=device),
        torch.linspace(-(w - 1) / 2, (w - 1) / 2, w, device=device),
        indexing="ij",
    )
    grid = torch.stack([z, y, x], -1).reshape(-1, 3)  # (h*w, 3) zyx, centred
    q = torch.randn(n_slices, 4, device=device)
    q = q / q.norm(dim=1, keepdim=True)
    a, b, c, d_ = q.unbind(1)
    rot = torch.stack(
        [
            1 - 2 * (c * c + d_ * d_),
            2 * (b * c - a * d_),
            2 * (b * d_ + a * c),
            2 * (b * c + a * d_),
            1 - 2 * (b * b + d_ * d_),
            2 * (c * d_ - a * b),
            2 * (b * d_ - a * c),
            2 * (c * d_ + a * b),
            1 - 2 * (b * b + c * c),
        ],
        -1,
    ).reshape(n_slices, 3, 3)
    pts = torch.einsum("sij,nj->sni", rot, grid).reshape(-1, 3)
    centre = torch.tensor([(d - 1) / 2, (h - 1) / 2, (w - 1) / 2], device=device)
    return (pts + centre).contiguous()


CASES: list[Case] = [
    # --- sampling -----------------------------------------------------------
    Case(
        "sample_3d_trilinear_complex_slices",  # torch-fourier-slice's access pattern
        lambda d, q: {
            "image": _cplx(128, 128, 128, device=d),
            "coordinates": _slice_coords(_q(64, q), (128, 128, 128), d),
            "interpolation": "trilinear",
        },
        sample_image_3d,
        "image",
    ),
    Case(
        "sample_3d_trilinear_complex_rfft",  # same size, uniformly random coords
        lambda d, q: {
            "image": _cplx(128, 128, 65, device=d),
            "coordinates": _coords(_q(40 * 128 * 65, q), (128, 128, 65), d),
            "interpolation": "trilinear",
        },
        sample_image_3d,
        "image",
    ),
    Case(
        "sample_3d_trilinear_real",
        lambda d, q: {
            "image": torch.randn(128, 128, 128, device=d),
            "coordinates": _coords(_q(1_000_000, q), (128, 128, 128), d),
            "interpolation": "trilinear",
        },
        sample_image_3d,
        "image",
    ),
    Case(
        "sample_3d_nearest_real",
        lambda d, q: {
            "image": torch.randn(128, 128, 128, device=d),
            "coordinates": _coords(_q(1_000_000, q), (128, 128, 128), d),
            "interpolation": "nearest",
        },
        sample_image_3d,
        "image",
    ),
    Case(
        "sample_3d_trilinear_real_multichannel4",
        lambda d, q: {
            "image": torch.randn(4, 64, 64, 64, device=d),
            "coordinates": _coords(_q(500_000, q), (64, 64, 64), d),
            "interpolation": "trilinear",
        },
        sample_image_3d,
        "image",
    ),
    Case(
        "sample_2d_bilinear_real",
        lambda d, q: {
            "image": torch.randn(1024, 1024, device=d),
            "coordinates": _coords(_q(1_000_000, q), (1024, 1024), d),
            "interpolation": "bilinear",
        },
        sample_image_2d,
        "image",
    ),
    Case(
        "sample_2d_bicubic_real",
        lambda d, q: {
            "image": torch.randn(1024, 1024, device=d),
            "coordinates": _coords(_q(1_000_000, q), (1024, 1024), d),
            "interpolation": "bicubic",
        },
        sample_image_2d,
        "image",
    ),
    Case(
        "sample_2d_bilinear_complex_rfft",
        lambda d, q: {
            "image": _cplx(512, 257, device=d),
            "coordinates": _coords(_q(1_000_000, q), (512, 257), d),
            "interpolation": "bilinear",
        },
        sample_image_2d,
        "image",
    ),
    Case(
        "sample_1d_linear_real",  # torch path allocates n*w: keep it small
        lambda d, q: {
            "image": torch.randn(4096, device=d),
            "coordinates": _coords(_q(40_000, q), (4096,), d)[:, 0],
            "interpolation": "linear",
        },
        sample_image_1d,
        "image",
    ),
    # --- insertion ------------------------------------------------------------
    Case(
        "insert_3d_trilinear_complex_slices",  # backprojection access pattern
        lambda d, q: {
            "values": _cplx(_q(64, q) * 128 * 128, device=d),
            "coordinates": _slice_coords(_q(64, q), (128, 128, 128), d),
            "image": torch.zeros(128, 128, 128, dtype=torch.complex64, device=d),
            "interpolation": "trilinear",
        },
        lambda **kw: insert_into_image_3d(**kw)[0],
        "values",
    ),
    Case(
        "insert_3d_trilinear_complex_rfft",  # same size, uniformly random coords
        lambda d, q: {
            "values": _cplx(_q(40 * 128 * 65, q), device=d),
            "coordinates": _coords(_q(40 * 128 * 65, q), (128, 128, 65), d),
            "image": torch.zeros(128, 128, 65, dtype=torch.complex64, device=d),
            "interpolation": "trilinear",
        },
        lambda **kw: insert_into_image_3d(**kw)[0],
        "values",
    ),
    Case(
        "insert_3d_trilinear_real",
        lambda d, q: {
            "values": torch.randn(_q(1_000_000, q), device=d),
            "coordinates": _coords(_q(1_000_000, q), (128, 128, 128), d),
            "image": torch.zeros(128, 128, 128, device=d),
            "interpolation": "trilinear",
        },
        lambda **kw: insert_into_image_3d(**kw)[0],
        "values",
    ),
    Case(
        "insert_2d_bilinear_real",
        lambda d, q: {
            "values": torch.randn(_q(1_000_000, q), device=d),
            "coordinates": _coords(_q(1_000_000, q), (1024, 1024), d),
            "image": torch.zeros(1024, 1024, device=d),
            "interpolation": "bilinear",
        },
        lambda **kw: insert_into_image_2d(**kw)[0],
        "values",
    ),
    Case(
        "insert_1d_linear_real",
        lambda d, q: {
            "values": torch.randn(_q(1_000_000, q), device=d),
            "coordinates": _coords(_q(1_000_000, q), (4096,), d)[:, 0],
            "image": torch.zeros(4096, device=d),
            "interpolation": "linear",
        },
        lambda **kw: insert_into_image_1d(**kw)[0],
        "values",
    ),
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time(
    fn: Callable[[], None], device: torch.device, repeat: int, warmup: int = 2
) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t)
    return statistics.median(samples) * 1e3  # ms


def _fresh(kwargs: dict) -> dict:
    """Insertion mutates its image in place: hand every run a zeroed copy."""
    out = dict(kwargs)
    if "values" in out:
        out["image"] = torch.zeros_like(out["image"])
    return out


def bench_case(case: Case, device: torch.device, repeat: int, quick: bool) -> dict:
    """Time one case under both backends; return a result row."""
    torch.manual_seed(0)
    kwargs = case.build(device, quick)
    n = kwargs["coordinates"].shape[0]
    row: dict = {"case": case.name, "device": device.type, "n_samples": int(n)}

    for backend in ("torch", "mojo"):
        with use_backend(backend):
            try:
                # correctness cross-check on the forward pass
                out = case.run(**_fresh(kwargs))
                fwd = _time(lambda: case.run(**_fresh(kwargs)), device, repeat)

                def fwd_bwd() -> None:
                    kw = _fresh(kwargs)
                    leaf = kw[case.grad_of].detach().requires_grad_(True)
                    kw[case.grad_of] = leaf * 1.0 if "values" in kw else leaf
                    y = case.run(**kw)
                    (
                        y.real.sum() + y.imag.sum() if y.is_complex() else y.sum()
                    ).backward()

                fb = _time(fwd_bwd, device, max(1, repeat // 2))
            except (RuntimeError, NotImplementedError) as exc:
                row[backend] = {"error": str(exc).splitlines()[0][:120]}
                continue
        row[backend] = {"fwd_ms": fwd, "fwd_bwd_ms": fb}
        row.setdefault("_out", {})[backend] = out.detach()

    outs = row.pop("_out", {})
    if len(outs) == 2:
        a, b = outs["torch"], outs["mojo"]
        row["max_abs_diff"] = float((a - b).abs().max())
    if "fwd_ms" in row.get("torch", {}) and "fwd_ms" in row.get("mojo", {}):
        row["speedup_fwd"] = row["torch"]["fwd_ms"] / row["mojo"]["fwd_ms"]
        row["speedup_fwd_bwd"] = row["torch"]["fwd_bwd_ms"] / row["mojo"]["fwd_bwd_ms"]
    return row


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "unknown"


def _fmt(row: dict, backend: str, key: str) -> str:
    v = row.get(backend, {})
    if "error" in v:
        return "n/a"
    return f"{v[key]:9.2f}"


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu", help="cpu, cuda, mps or 'all'")
    ap.add_argument(
        "--cases", default="", help="comma-separated substrings to select cases"
    )
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="1/8 of the sample counts")
    ap.add_argument("--label", default="", help="free-text tag stored with the results")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    devices = [args.device]
    if args.device == "all":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")

    selected = [
        c
        for c in CASES
        if not args.cases or any(s in c.name for s in args.cases.split(","))
    ]
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": _git_sha(),
        "label": args.label,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "cpu_count": torch.get_num_threads(),
        "quick": args.quick,
    }

    print(
        f"# torch-image-interpolation benchmark  ({meta['git_sha']}, "
        f"{meta['timestamp']}) {args.label}"
    )
    for dev in devices:
        device = torch.device(dev)
        if not tii.mojo_available(device):
            print(f"\n[{dev}] Mojo kernels unavailable -- skipping")
            continue
        print(f"\n## device: {dev}   (times in ms, median of {args.repeat})\n")
        print(
            f"{'case':42s} {'n':>9s} | {'torch fwd':>9s} {'mojo fwd':>9s} {'x':>6s} | "
            f"{'torch f+b':>9s} {'mojo f+b':>9s} {'x':>6s} | max|Δ|"
        )
        print("-" * 128)
        rows = []
        for case in selected:
            row = bench_case(case, device, args.repeat, args.quick)
            rows.append(row)
            sp_f = f"{row['speedup_fwd']:5.1f}x" if "speedup_fwd" in row else "   n/a"
            sp_b = (
                f"{row['speedup_fwd_bwd']:5.1f}x"
                if "speedup_fwd_bwd" in row
                else "   n/a"
            )
            diff = f"{row['max_abs_diff']:.1e}" if "max_abs_diff" in row else "n/a"
            tf, mf = _fmt(row, "torch", "fwd_ms"), _fmt(row, "mojo", "fwd_ms")
            tb, mb = _fmt(row, "torch", "fwd_bwd_ms"), _fmt(row, "mojo", "fwd_bwd_ms")
            print(
                f"{row['case']:42s} {row['n_samples']:9d} | {tf} {mf} {sp_f} | "
                f"{tb} {mb} {sp_b} | {diff}"
            )
            for b in ("torch", "mojo"):
                if "error" in row.get(b, {}):
                    print(f"    {b}: {row[b]['error']}")
        if not args.no_save:
            HISTORY.parent.mkdir(parents=True, exist_ok=True)
            with HISTORY.open("a") as f:
                for row in rows:
                    f.write(json.dumps({**meta, **row}) + "\n")
            print(f"\nappended {len(rows)} rows to {HISTORY.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
