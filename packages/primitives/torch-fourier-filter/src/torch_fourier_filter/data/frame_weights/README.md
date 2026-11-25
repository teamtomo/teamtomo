# Default per-frame B-factor curve

`default_frame_bfactors.json` is the default for
`torch_fourier_filter.frame_weights.default_frame_bfactors`: RELION's per-frame
model, w_f(k) = exp(B_f k²/4 + C_f), as a B-factor curve and a log-scale curve in
physical exposure (e/Å²).

It was fitted on the large ribosomal subunit (LSU) of yeast ribosomes in situ at
300 kV and on single-particle microtubule specimens at 200 kV. It is a starting
point; refit B and C for your own data where you can.
