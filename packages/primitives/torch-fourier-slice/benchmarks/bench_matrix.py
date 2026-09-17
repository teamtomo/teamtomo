"""Benchmark matrix comparing torch-fourier-slice's canonical and experimental backends.

Compares torch-fourier-slice canonical (pure PyTorch) vs. experimental (Mojo)
vs., optionally, torch-projectors -- for central-slice insertion and
extraction, forward-only and forward+backward, on CPU and/or GPU.

Assumes ``torch-fourier-slice[mojo]`` is installed. ``torch-projectors`` is
optional: if it isn't importable, its rows are simply omitted from the
matrix -- torch-fourier-slice canonical vs experimental is always compared,
with or without torch-projectors installed.

Output is CSV, one row per (operation, backend, device, box, bp, interp,
phase) combination:

    operation,backend,device,box,bp,interp,phase,ms

- operation: ``insertion`` | ``extraction``
- backend:   ``canonical`` | ``experimental`` | ``torch_projectors``
- phase:     ``forward`` | ``forward_backward`` -- the forward pass alone,
  and the full forward+``backward()`` round trip. Both are run and timed
  directly (not derived from each other by subtraction), so every row is a
  direct measurement; compute the backward-only cost yourself as
  ``forward_backward - forward`` if you want it (subtracting two
  independently-averaged numbers adds noise, so it's left as a row-level CSV
  operation rather than baked into what gets measured).

Pass ``--out FILE`` to write the CSV to a file (default: stdout) and/or
``--table`` to also print a human-readable pivot table.

torch-fourier-slice's canonical (non-experimental) API has no
``interpolation`` choice -- extraction hardcodes (tri/bi)linear sampling via
``torch_image_interpolation`` and insertion hardcodes trilinear splatting --
so canonical rows only ever report ``interp=linear``; ``cubic`` configs are
skipped for that backend.

CPU thread matching
--------------------
torch-fourier-slice's Mojo kernels pick their own worker-thread count from
the number of *physical* cores implied by this process's CPU affinity mask
(Mojo's ``num_physical_cores()``), not from ``torch.set_num_threads()``.
torch-projectors' CPU kernel instead uses torch's own intraop thread pool.

For a genuinely matched CPU comparison this script pins the process with
``--cpu-affinity`` and sets ``torch.set_num_threads()`` to the same count --
but on a hyperthreaded CPU, a naive contiguous core range (e.g. ``0-7``) can
cover only half as many *physical* cores as logical ones (each hyperthread
pair contributes 2 logical IDs for 1 physical core), silently under-counting
Mojo's side relative to torch's. Check ``lscpu -e`` on your machine and pick
logical CPU IDs that map to distinct physical cores (e.g. the non-hyperthreaded
E-cores on a hybrid Intel part) for a fair comparison. Without ``--cpu-affinity``
the CPU pass just runs with whatever the OS schedules by default.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass

import torch

from torch_fourier_slice import (
    extract_central_slices_rfft_3d as canonical_extract,
)
from torch_fourier_slice import (
    insert_central_slices_rfft_3d as canonical_insert,
)
from torch_fourier_slice.experimental import (
    extract_central_slices_rfft_3d as experimental_extract,
)
from torch_fourier_slice.experimental import (
    insert_central_slices_rfft_3d as experimental_insert,
)

try:
    import torch_projectors

    HAVE_TORCH_PROJECTORS = True
except ImportError:
    HAVE_TORCH_PROJECTORS = False

CONFIGS = [
    # (box, bp, interpolation)
    (128, 41, "linear"),
    (128, 41, "cubic"),
    (256, 41, "linear"),
    (256, 41, "cubic"),
    (256, 200, "linear"),
    (256, 200, "cubic"),
]


def _random_rotations(n: int, device: str) -> torch.Tensor:
    a = torch.randn(n, 3, 3, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[..., 0] *= det.sign().unsqueeze(-1)
    return q.contiguous()


def _time(run, device: str, n_reps: int, warmup: int) -> float:
    """Mean ms/call, using CUDA events on GPU and perf_counter on CPU."""
    for _ in range(warmup):
        run()
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_reps):
            run()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / n_reps
    start = time.perf_counter()
    for _ in range(n_reps):
        run()
    return (time.perf_counter() - start) / n_reps * 1000


# ---------------------------------------------------------------------------
# Run-builders: one per (operation, backend). Each returns a differentiable
# output tensor for the given (box, bp, interp, device); backends that don't
# support an `interp` choice (canonical) just ignore it.
# ---------------------------------------------------------------------------


def _insert_canonical(box: int, bp: int, interp: str, device: str) -> torch.Tensor:
    h, w = box, box // 2 + 1
    image_rfft = torch.randn(
        bp, h, w, dtype=torch.complex64, device=device, requires_grad=True
    )
    rot = _random_rotations(bp, device).requires_grad_(True)
    vol, _ = canonical_insert(image_rfft, (box, box, box), rot)
    return vol


def _insert_experimental(box: int, bp: int, interp: str, device: str) -> torch.Tensor:
    h, w = box, box // 2 + 1
    image_rfft = torch.randn(
        bp, h, w, dtype=torch.complex64, device=device, requires_grad=True
    )
    rot = _random_rotations(bp, device).requires_grad_(True)
    vol, _ = experimental_insert(image_rfft, rot, interpolation=interp)
    return vol


def _insert_torch_projectors(
    box: int, bp: int, interp: str, device: str
) -> torch.Tensor:
    h, w = box, box // 2 + 1
    proj = torch.randn(
        1, bp, h, w, dtype=torch.complex64, device=device, requires_grad=True
    )
    rot = _random_rotations(bp, device).unsqueeze(0).requires_grad_(True)
    vol, _ = torch_projectors.backproject_2d_to_3d_forw(proj, rot, interpolation=interp)
    return vol


def _extract_canonical(box: int, bp: int, interp: str, device: str) -> torch.Tensor:
    d, dh = box, box // 2 + 1
    vol = torch.randn(
        d, d, dh, dtype=torch.complex64, device=device, requires_grad=True
    )
    rot = _random_rotations(bp, device).requires_grad_(True)
    return canonical_extract(vol, rot)


def _extract_experimental(box: int, bp: int, interp: str, device: str) -> torch.Tensor:
    d, dh = box, box // 2 + 1
    vol = torch.randn(
        d, d, dh, dtype=torch.complex64, device=device, requires_grad=True
    )
    rot = _random_rotations(bp, device).requires_grad_(True)
    return experimental_extract(vol, rot, interpolation=interp)


def _extract_torch_projectors(
    box: int, bp: int, interp: str, device: str
) -> torch.Tensor:
    d, dh = box, box // 2 + 1
    vol = torch.randn(
        1, d, d, dh, dtype=torch.complex64, device=device, requires_grad=True
    )
    rot = _random_rotations(bp, device).unsqueeze(0).requires_grad_(True)
    return torch_projectors.project_3d_to_2d_forw(vol, rot, interpolation=interp)


BACKENDS = {
    "insertion": {
        "canonical": _insert_canonical,
        "experimental": _insert_experimental,
        "torch_projectors": _insert_torch_projectors,
    },
    "extraction": {
        "canonical": _extract_canonical,
        "experimental": _extract_experimental,
        "torch_projectors": _extract_torch_projectors,
    },
}


@dataclass
class Row:
    """One CSV row: a single directly-measured timing."""

    operation: str
    backend: str
    device: str
    box: int
    bp: int
    interp: str
    phase: str
    ms: float


def _bench_phase(
    make_run,
    box: int,
    bp: int,
    interp: str,
    device: str,
    n_reps: int,
    warmup: int,
    phase: str,
) -> float:
    """Time one phase ('forward' or 'forward_backward'). Raises on failure."""
    if phase == "forward":

        def run():
            make_run(box, bp, interp, device)

    else:

        def run():
            out = make_run(box, bp, interp, device)
            out.abs().pow(2).sum().backward()

    return _time(run, device, n_reps, warmup)


def _try_bench_phase(
    make_run,
    operation: str,
    backend: str,
    box: int,
    bp: int,
    interp: str,
    device: str,
    reps: int,
    warmup: int,
    phase: str,
) -> float | None:
    """Like `_bench_phase`, but return None on OOM instead of raising.

    Prints a warning to stderr instead -- so, e.g., a backward-only OOM
    doesn't also cost you the (perfectly fine) forward-only number for the
    same config.
    """
    try:
        return _bench_phase(make_run, box, bp, interp, device, reps, warmup, phase)
    except torch.cuda.OutOfMemoryError:
        pass
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" not in msg and "cannot allocate memory" not in msg:
            raise
    print(
        f"OOM, skipping: {operation} {backend} {device} {phase} "
        f"box={box} bp={bp} interp={interp} "
        "(the canonical pure-PyTorch backend allocates far more intermediate "
        "memory than the fused kernels)",
        file=sys.stderr,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
    return None


def collect_rows(devices: list[str], reps: int, warmup: int) -> list[Row]:
    """Run every (operation, backend, config, device, phase) combination."""
    rows: list[Row] = []
    for operation, backends in BACKENDS.items():
        for backend, make_run in backends.items():
            if backend == "torch_projectors" and not HAVE_TORCH_PROJECTORS:
                continue
            for box, bp, interp in CONFIGS:
                if backend == "canonical" and interp != "linear":
                    continue  # canonical has no interpolation choice
                for device in devices:
                    for phase in ("forward", "forward_backward"):
                        ms = _try_bench_phase(
                            make_run,
                            operation,
                            backend,
                            box,
                            bp,
                            interp,
                            device,
                            reps,
                            warmup,
                            phase,
                        )
                        if ms is not None:
                            rows.append(
                                Row(
                                    operation,
                                    backend,
                                    device,
                                    box,
                                    bp,
                                    interp,
                                    phase,
                                    ms,
                                )
                            )
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


def render_table(rows: list[Row]) -> str:
    """Render a simple text pivot.

    One block per (operation, device, phase), rows = (box, bp, interp),
    columns = backend.
    """
    backends_seen = sorted({r.backend for r in rows})
    groups: dict[
        tuple[str, str, str], dict[tuple[int, int, str], dict[str, float]]
    ] = {}
    for r in rows:
        key = (r.operation, r.device, r.phase)
        groups.setdefault(key, {}).setdefault((r.box, r.bp, r.interp), {})[
            r.backend
        ] = r.ms

    lines: list[str] = []
    for (operation, device, phase), table in sorted(groups.items()):
        lines.append(f"\n=== {operation} / {device} / {phase} ===")
        header = f"{'box':>5} {'bp':>5} {'interp':<7}" + "".join(
            f"{b:>18}" for b in backends_seen
        )
        lines.append(header)
        for (box, bp, interp), vals in sorted(table.items()):
            cells = "".join(
                f"{vals[b]:>15.3f} ms" if b in vals else f"{'--':>18}"
                for b in backends_seen
            )
            lines.append(f"{box:>5} {bp:>5} {interp:<7}{cells}")
    return "\n".join(lines)


def main() -> None:
    """Parse CLI args, run the matrix, write CSV (and optionally a table)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default=None,
        help="Logical CPU ids to pin the CPU pass to, e.g. '12-19' or '0,1,2,3'. "
        "Use ids mapping to distinct physical cores (see `lscpu -e`) for a "
        "thread-matched comparison; see the module docstring.",
    )
    parser.add_argument("--reps", type=int, default=10, help="Timed reps per config.")
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup calls per config."
    )
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument(
        "--out", type=str, default=None, help="CSV output path (default: stdout)."
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Also print a human-readable pivot table (after the CSV, or to "
        "stderr if the CSV itself is going to stdout).",
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

    if "cpu" in devices and args.cpu_affinity:
        cores = _parse_cpu_list(args.cpu_affinity)
        os.sched_setaffinity(0, cores)
        torch.set_num_threads(len(cores))

    if not HAVE_TORCH_PROJECTORS:
        print(
            "torch_projectors not installed -- its rows are omitted "
            "(torch-fourier-slice canonical vs experimental still runs).",
            file=sys.stderr,
        )

    rows = collect_rows(devices, args.reps, args.warmup)

    out_fh = open(args.out, "w", newline="") if args.out else sys.stdout
    writer = csv.writer(out_fh)
    writer.writerow(
        ["operation", "backend", "device", "box", "bp", "interp", "phase", "ms"]
    )
    for r in rows:
        writer.writerow(
            [
                r.operation,
                r.backend,
                r.device,
                r.box,
                r.bp,
                r.interp,
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
