"""Benchmark matrix comparing torch-image-interpolation's torch and Mojo backends.

Compares the pure-PyTorch reference implementation (``grid_sample`` /
``index_put_``) against the fused Mojo kernels for 3D sampling and 3D
insertion, forward-only and forward+backward, on CPU and/or GPU.

Output is CSV, one row per (operation, backend, device, box, n_points, interp,
dtype, phase) combination, with columns ``operation``, ``backend``, ``device``,
``box``, ``n_points``, ``interp``, ``dtype``, ``phase``, ``ms``:

- operation: ``sample`` | ``insert``
- backend:   ``torch`` | ``mojo``
- box:       the volume is ``(box, box, box)``
- n_points:  number of coordinates sampled from / inserted into it
- phase:     ``forward`` | ``forward_backward`` -- the forward pass alone (under
  ``no_grad``), and the full forward + ``autograd.grad()`` round trip. Both are
  run and timed directly rather than derived from each other by subtraction, so
  every row is a direct measurement; compute the backward-only cost yourself as
  ``forward_backward - forward`` if you want it.

Pass ``--out FILE`` to write the CSV to a file (default: stdout) and/or
``--table`` to also print a human-readable pivot table with torch/mojo
speed-ups. ``--check`` cross-checks the two backends' forward output before
timing anything.

Coordinates are drawn uniformly at random over the volume.

Usage:

- ``python benchmarks/bench_matrix.py --skip-cpu --table --out results/cuda.csv``
- ``python benchmarks/bench_matrix.py --boxes 128,256 --points 10000 --check``
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import torch_image_interpolation as tii
from torch_image_interpolation import (
    insert_into_image_3d,
    sample_image_3d,
    use_backend,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

print = functools.partial(print, flush=True)  # noqa: A001  (progress in pipes)

BOXES = (128, 256, 384, 512)
N_POINTS = (10_000, 100_000, 1_000_000, 4_000_000)
INTERPOLATIONS = ("trilinear", "nearest")
DTYPES = ("float32", "complex64")
BACKENDS = ("torch", "mojo")
PHASES = ("forward", "forward_backward")

_TORCH_DTYPES = {"float32": torch.float32, "complex64": torch.complex64}


@dataclass
class Config:
    """One point in the (box, n_points, interp, dtype) grid."""

    box: int
    n_points: int
    interp: str
    dtype: str


@dataclass
class Workload:
    """Prebuilt inputs for one (operation, config, device) plus how to run it."""

    call: Callable[[], torch.Tensor]
    leaf: torch.Tensor  # the tensor gradients are taken with respect to
    accumulators: tuple[torch.Tensor, ...] = ()  # mutated in place by `call`


def _randn(shape: tuple[int, ...], dtype: str, device: str) -> torch.Tensor:
    if dtype == "complex64":
        return torch.complex(
            torch.randn(shape, device=device), torch.randn(shape, device=device)
        )
    return torch.randn(shape, device=device)


def _coordinates(n: int, box: int, device: str) -> torch.Tensor:
    return torch.rand(n, 3, device=device) * (box - 1)


def _build_sample(cfg: Config, device: str) -> Workload:
    image = _randn((cfg.box,) * 3, cfg.dtype, device)
    coordinates = _coordinates(cfg.n_points, cfg.box, device)
    return Workload(
        call=lambda: sample_image_3d(image, coordinates, interpolation=cfg.interp),
        leaf=image,
    )


def _build_insert(cfg: Config, device: str) -> Workload:
    values = _randn((cfg.n_points,), cfg.dtype, device)
    coordinates = _coordinates(cfg.n_points, cfg.box, device)
    image = torch.zeros((cfg.box,) * 3, dtype=_TORCH_DTYPES[cfg.dtype], device=device)
    # Pass `weights` explicitly: left as None, both backends allocate a zeroed
    # box**3 float32 image on every call, which at box=512 is half a gigabyte
    # of memset per rep and has nothing to do with the kernel being timed.
    weights = torch.zeros((cfg.box,) * 3, dtype=torch.float32, device=device)
    return Workload(
        call=lambda: insert_into_image_3d(
            values, coordinates, image, weights, interpolation=cfg.interp
        )[0],
        leaf=values,
        accumulators=(image, weights),
    )


OPERATIONS: dict[str, Callable[[Config, str], Workload]] = {
    "sample": _build_sample,
    "insert": _build_insert,
}


def _loss(out: torch.Tensor) -> torch.Tensor:
    if out.is_complex():
        return out.real.sum() + out.imag.sum()
    return out.sum()


def _make_run(workload: Workload, phase: str) -> Callable[[], None]:
    """A zero-argument callable running one `phase` of `workload`."""
    if phase == "forward":

        def run() -> None:
            with torch.no_grad():
                workload.call()

        return run

    def run() -> None:
        out = workload.call()
        torch.autograd.grad(_loss(out), workload.leaf)
        # Insertion writes into `image`/`weights` in place, so after a backward
        # those tensors carry this rep's grad_fn; leaving it attached would
        # make the next rep extend an already-freed graph. Detaching in place
        # is O(1) and restores them to leaves.
        for tensor in workload.accumulators:
            tensor.detach_()

    return run


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


@dataclass
class Timing:
    """How a configuration is warmed up and timed."""

    warmup_ms: float = 150.0
    min_warmup: int = 3
    min_time_ms: float = 100.0
    min_reps: int = 3
    max_reps: int = 500


def _time(run: Callable[[], None], device: str, timing: Timing) -> tuple[float, int]:
    """Mean ms/call and the number of timed reps used."""
    # Warm up on wall time, synchronising each call so `warmup_ms` is real GPU
    # time rather than however long it took to queue a few thousand launches.
    start = time.perf_counter()
    n_warm = 0
    while True:
        run()
        _sync(device)
        n_warm += 1
        elapsed_ms = (time.perf_counter() - start) * 1000
        if n_warm >= timing.min_warmup and elapsed_ms >= timing.warmup_ms:
            break
    per_call_ms = elapsed_ms / n_warm

    # `min_time_ms` is what actually sets the rep count: sub-millisecond kernels
    # get hundreds of launches, a half-second torch call gets `min_reps` and no
    # more -- ten reps of an already-stable half-second call is just wall time.
    reps = int(timing.min_time_ms / per_call_ms) + 1 if per_call_ms > 0 else 1
    reps = max(timing.min_reps, min(timing.max_reps, reps))

    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(reps):
            run()
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / reps, reps

    start = time.perf_counter()
    for _ in range(reps):
        run()
    return (time.perf_counter() - start) / reps * 1000, reps


@dataclass
class Row:
    """One CSV row: a single directly-measured timing."""

    operation: str
    backend: str
    device: str
    box: int
    n_points: int
    interp: str
    dtype: str
    phase: str
    ms: float


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cannot allocate memory" in msg


def _free(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()


def _describe(operation: str, cfg: Config, device: str) -> str:
    return (
        f"{operation} {device} box={cfg.box} n={cfg.n_points} "
        f"interp={cfg.interp} dtype={cfg.dtype}"
    )


def _reset(workload: Workload) -> None:
    """Zero the in-place accumulators so a call starts from a clean image."""
    for tensor in workload.accumulators:
        tensor.detach_().zero_()


def _check(workload: Workload, operation: str, cfg: Config, device: str) -> None:
    """Cross-check the two backends' forward output; report to stderr."""
    outputs = {}
    for backend in BACKENDS:
        _reset(workload)
        try:
            with use_backend(backend), torch.no_grad():
                outputs[backend] = workload.call().clone()
        except (RuntimeError, NotImplementedError) as exc:
            print(
                f"check {_describe(operation, cfg, device)}: {backend} failed "
                f"({str(exc).splitlines()[0][:120]})",
                file=sys.stderr,
            )
            return
    diff = (outputs["torch"] - outputs["mojo"]).abs().max().item()
    print(
        f"check {_describe(operation, cfg, device)}: max|torch-mojo| = {diff:.3e}",
        file=sys.stderr,
    )


def _bench(
    workload: Workload,
    backend: str,
    phase: str,
    operation: str,
    cfg: Config,
    device: str,
    timing: Timing,
    verbose: bool,
) -> float | None:
    """Time one (backend, phase); return None if it OOMs or is unsupported."""
    workload.leaf.requires_grad_(phase == "forward_backward")
    _reset(workload)
    run = _make_run(workload, phase)
    try:
        with use_backend(backend):
            ms, reps = _time(run, device, timing)
    except (RuntimeError, NotImplementedError, torch.OutOfMemoryError) as exc:
        if isinstance(exc, torch.OutOfMemoryError) or _is_oom(exc):
            reason = "OOM"
        else:
            reason = str(exc).splitlines()[0][:120]
        print(
            f"skipping {operation} {backend} {device} {phase} "
            f"box={cfg.box} n={cfg.n_points} interp={cfg.interp} "
            f"dtype={cfg.dtype}: {reason}",
            file=sys.stderr,
        )
        _free(device)
        return None
    finally:
        workload.leaf.requires_grad_(False)
        for tensor in workload.accumulators:
            tensor.detach_()
    if verbose:
        print(
            f"  {backend:>5} {phase:<16} {ms:9.3f} ms  ({reps} reps)", file=sys.stderr
        )
    return ms


def collect_rows(
    devices: Iterable[str],
    configs: Iterable[Config],
    operations: Iterable[str],
    timing: Timing,
    check: bool,
    verbose: bool,
) -> list[Row]:
    """Run every (operation, config, device, phase, backend) combination."""
    rows: list[Row] = []
    for device in devices:
        backends = list(BACKENDS)
        if not tii.mojo_available(device):
            print(
                f"Mojo kernels unavailable on {device} -- torch rows only.",
                file=sys.stderr,
            )
            backends.remove("mojo")
        for operation in operations:
            build = OPERATIONS[operation]
            for cfg in configs:
                try:
                    workload = build(cfg, device)
                except (RuntimeError, torch.OutOfMemoryError) as exc:
                    if not (isinstance(exc, torch.OutOfMemoryError) or _is_oom(exc)):
                        raise
                    print(
                        f"skipping {_describe(operation, cfg, device)}: "
                        "inputs do not fit in memory",
                        file=sys.stderr,
                    )
                    _free(device)
                    continue
                if verbose:
                    print(_describe(operation, cfg, device), file=sys.stderr)
                if check:
                    _check(workload, operation, cfg, device)
                for phase in PHASES:
                    for backend in backends:
                        ms = _bench(
                            workload,
                            backend,
                            phase,
                            operation,
                            cfg,
                            device,
                            timing,
                            verbose,
                        )
                        if ms is not None:
                            rows.append(
                                Row(
                                    operation,
                                    backend,
                                    device,
                                    cfg.box,
                                    cfg.n_points,
                                    cfg.interp,
                                    cfg.dtype,
                                    phase,
                                    ms,
                                )
                            )
                del workload
                _free(device)
    return rows


def _parse_cpu_list(spec: str) -> set[int]:
    """Parse '12-19' or '0,1,2,7-9' into a set of CPU ids."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _parse_choice(spec: str, valid: tuple[str, ...], name: str) -> list[str]:
    chosen = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [c for c in chosen if c not in valid]
    if unknown:
        raise SystemExit(f"unknown {name}: {', '.join(unknown)} (valid: {valid})")
    return chosen


def render_table(rows: list[Row]) -> str:
    """Pivot: one block per (operation, device, phase), columns = backend."""
    backends_seen = [b for b in BACKENDS if any(r.backend == b for r in rows)]
    groups: dict[
        tuple[str, str, str], dict[tuple[int, int, str, str], dict[str, float]]
    ] = {}
    for r in rows:
        key = (r.operation, r.device, r.phase)
        groups.setdefault(key, {}).setdefault(
            (r.box, r.n_points, r.interp, r.dtype), {}
        )[r.backend] = r.ms

    lines: list[str] = []
    for (operation, device, phase), table in sorted(groups.items()):
        lines.append(f"\n=== {operation} / {device} / {phase} ===")
        lines.append(
            f"{'box':>5} {'n_points':>10} {'interp':<10} {'dtype':<10}"
            + "".join(f"{b:>16}" for b in backends_seen)
            + f"{'speedup':>10}"
        )
        for (box, n_points, interp, dtype), vals in sorted(table.items()):
            cells = "".join(
                f"{vals[b]:>13.3f} ms" if b in vals else f"{'--':>16}"
                for b in backends_seen
            )
            if "torch" in vals and "mojo" in vals and vals["mojo"] > 0:
                speedup = f"{vals['torch'] / vals['mojo']:>9.2f}x"
            else:
                speedup = f"{'--':>10}"
            lines.append(
                f"{box:>5} {n_points:>10} {interp:<10} {dtype:<10}{cells}{speedup}"
            )
    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, run the matrix, write CSV (and optionally a table)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--boxes",
        default=",".join(str(b) for b in BOXES),
        help=f"Comma-separated cubic volume edge lengths (default: {BOXES}).",
    )
    parser.add_argument(
        "--points",
        default=",".join(str(n) for n in N_POINTS),
        help=f"Comma-separated coordinate counts (default: {N_POINTS}).",
    )
    parser.add_argument(
        "--interp",
        default=",".join(INTERPOLATIONS),
        help=f"Comma-separated interpolation modes (default: {INTERPOLATIONS}).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help=f"Comma-separated dtypes, any of {DTYPES} (default: float32).",
    )
    parser.add_argument(
        "--operations",
        default=",".join(OPERATIONS),
        help=f"Comma-separated operations (default: {tuple(OPERATIONS)}).",
    )
    parser.add_argument(
        "--cpu-affinity",
        default=None,
        help="Logical CPU ids to pin the CPU pass to, e.g. '12-19' or '0,1,2,3'. "
        "Use ids mapping to distinct physical cores (see `lscpu -e`) for a "
        "thread-matched comparison; see the module docstring.",
    )
    parser.add_argument("--warmup-ms", type=float, default=150.0)
    parser.add_argument("--min-time-ms", type=float, default=100.0)
    parser.add_argument("--min-reps", type=int, default=3)
    parser.add_argument("--max-reps", type=int, default=500)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Cross-check the backends' forward output before timing each config.",
    )
    parser.add_argument(
        "--out", default=None, help="CSV output path (default: stdout)."
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Also print a human-readable pivot table (after the CSV, or to "
        "stderr if the CSV itself is going to stdout).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Progress on stderr."
    )
    args = parser.parse_args()

    devices = []
    if not args.skip_cpu:
        devices.append("cpu")
    if not args.skip_gpu:
        if torch.cuda.is_available():
            devices.append("cuda")
        else:
            print("CUDA not available, skipping GPU.", file=sys.stderr)
    if not devices:
        print("Nothing to run: no devices selected.", file=sys.stderr)
        return

    boxes = [int(b) for b in args.boxes.split(",") if b.strip()]
    points = [int(n) for n in args.points.split(",") if n.strip()]
    interps = _parse_choice(args.interp, INTERPOLATIONS, "interpolation")
    dtypes = _parse_choice(args.dtype, DTYPES, "dtype")
    operations = _parse_choice(args.operations, tuple(OPERATIONS), "operation")
    configs = [
        Config(box, n, interp, dtype)
        for box in boxes
        for n in points
        for interp in interps
        for dtype in dtypes
    ]

    if "cpu" in devices and args.cpu_affinity:
        cores = _parse_cpu_list(args.cpu_affinity)
        os.sched_setaffinity(0, cores)
        torch.set_num_threads(len(cores))

    timing = Timing(
        warmup_ms=args.warmup_ms,
        min_time_ms=args.min_time_ms,
        min_reps=args.min_reps,
        max_reps=args.max_reps,
    )
    rows = collect_rows(devices, configs, operations, timing, args.check, args.verbose)

    out_fh = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.writer(out_fh)
    writer.writerow(
        [
            "operation",
            "backend",
            "device",
            "box",
            "n_points",
            "interp",
            "dtype",
            "phase",
            "ms",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.operation,
                r.backend,
                r.device,
                r.box,
                r.n_points,
                r.interp,
                r.dtype,
                r.phase,
                f"{r.ms:.4f}",
            ]
        )
    if args.out:
        out_fh.close()
        print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)

    if args.table:
        dest = sys.stdout if args.out else sys.stderr
        print(render_table(rows), file=dest)


if __name__ == "__main__":
    main()
