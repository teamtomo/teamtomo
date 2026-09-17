# GPU atomic-contention reductions

The scatter (insertion) and pose-gradient kernels have many GPU threads
accumulating into a shared handful of memory locations via atomics. This
documents the reduction strategies for cutting that contention: how each
works, and whether it's actually used.

## Warp-level reduction — shipped, used for pose gradients

Every pixel of a pose-gradient kernel targets the same ~14-scalar
accumulator, so contention is severe (plain atomics measured 3-13x *slower*
on GPU than the CPU fallback). `_grad_add` (`_common.mojo`) sums each value
across a warp (`sum()` from `std.gpu.primitives.warp`) and has a single lane
issue one atomic per warp instead of one per pixel — a 32x (`WARP_SIZE`) cut
to the atomic count. `_warp_pose_uniform` detects whether a warp's threads
all share one pose (needed since a warp can straddle a pose boundary).

## Occupancy/ILP coarsening — shipped, used for the scatter kernels

The scatter kernels don't reduce atomics at all; contention there is spread
across the volume rather than concentrated on one accumulator, so instead
each thread handles several pixels via an unrolled loop
(`SCATTER_COARSEN_CUBIC`/`SCATTER_COARSEN_LINEAR`), giving more independent
in-flight atomics per thread to hide latency. Data-driven per interpolation
mode — see that constant's own comment.

## Block-level reduction — tried, reverted (slower)

One block per pose (`grid = (bp, bv)`) instead of the flat 1D grid the other
kernels use; each thread strides over that pose's pixels into a register (no
atomics needed, disjoint work), then a `barrier()` and a single thread sums
all per-thread partials via shared memory before one atomic per pose. This is
what torch-projectors' own backward CUDA kernel does for its pose gradients
(confirmed directly in its source — and confirmed it never combines this
with warp-level reduction).

Correct, but 10-25% *slower* than the warp-level kernel across every config
tested. Root cause: `(bp, bv)` only launches as many blocks as there are
poses (82-400 in the tested configs) versus the warp kernel's tens of
thousands of blocks — far worse GPU occupancy, which outweighs the saved
atomics. The final reduction was also serial (one thread summing all
per-thread partials), not a tree. Warp-level reduction is also simply less
code: a drop-in helper needing no grid-layout change, versus a new 2D grid,
a new launcher, and manual shared-memory sizing.

## `match_any` warp reduction for scatter atomics — tried, reverted (compile blowup)

`match_any` (`std.gpu.primitives.warp`, available since the mojo/max split
brought `mojo==1.0.0`) returns a bitmask of every warp lane whose value
matches the calling lane's — verified to behave exactly like CUDA's
`__match_any_sync`. The standard "warp-aggregated atomics" technique groups
lanes by their target address this way, sums within each group via
`shuffle_idx`, and has one leader lane per group issue a single atomic —
verified correct in isolation before touching real code.

Reverted because of compile time, not correctness: `_atomic_add_at`
(`_common.mojo`) is `@always_inline` and called once per splat corner per
component — tricubic 3D alone is 64 corners x 2, doubled again by
`SCATTER_COARSEN_CUBIC`. Every one of those 100+ call sites in a kernel got
the full leader-election-and-shuffle loop inlined into it instead of one
instruction, and the resulting kernel didn't finish compiling within 150s
even for the smallest possible test case. A viable version would need to
shrink what's inlined per call site — e.g. drop `@always_inline` so it
compiles once as a real function, or gather a pixel's corner contributions
and reduce once per pixel instead of once per corner. Not attempted.
