# torch-ctf

[![License](https://img.shields.io/pypi/l/torch-ctf.svg?color=green)](https://github.com/jdickerson95/torch-ctf/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-ctf.svg?color=green)](https://pypi.org/project/torch-ctf)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-ctf.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/torch-ctf/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/torch-ctf/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/torch-ctf/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/torch-ctf)

CTF calculation for cryoEM in torch

## Sign convention

torch-ctf uses the convention that **positive electrostatic potential multiplied
by the CTF at underfocus gives a dark image**, as in a raw cryo-EM micrograph.

- `defocus > 0` is underfocus and `defocus < 0` is overfocus (micrometers).
- The CTF is `sin(chi)`, with

  `chi = -pi * lambda * defocus * k^2 + pi / 2 * Cs * lambda^3 * k^4 - phase_shift - arcsin(amplitude_contrast)`

  `chi` is what `calculate_total_phase_shift` returns. It is negative at low
  frequency for underfocus.
- With no phase shift, the CTF is `-amplitude_contrast` at zero frequency and, at
  underfocus, its first extremum is `-1`. For 3 µm underfocus, 300 kV, Cs 2.7 mm
  and amplitude contrast 0.07, the CTF starts at -0.07 and goes to -1 first.
- For a projection `p` of positive potential, `irfftn(rfftn(p) * ctf)` is dark
  where the particle is. Multiply the image by -1 if you want white particles on
  a dark background, as RELION expects.

This matches CTFFIND4 and cisTEM, which write the CTF as `-sin(chi')` with
`chi' = -chi`. It is the negative of RELION's CTF, which works with
contrast-inverted particles.

Every CTF in the package follows it:

| Function | Output |
| --- | --- |
| `calculate_ctf_1d`, `calculate_ctf_2d`, `calc_LPP_ctf_2D` | `sin(chi_s)`, times `exp(i chi_a)` when beam tilt or odd Zernikes are given |
| `return_complex_ctf=True` | `H = exp(-i (chi_s - chi_a))`, the wave aberration function. Applied to an exit wave `exp(+i sigma V)` (multislice simulation, Ewald sphere correction) it gives dark particles at underfocus. The real CTF is `i (H(k) - conj(H(-k))) / 2`, which is `-imag(H)` without odd aberrations |
| `calculate_ctfp_and_ctfq_2d` | `ctfp = sin(chi_s) - i cos(chi_s)` and `ctfq = sin(chi_s) + i cos(chi_s)`, both times `exp(i chi_a)`. `(ctfp + ctfq) / 2` is the real CTF, `ctfq = i H(k)` and `ctfp = -i conj(H(-k))` |
| `calculate_ctf_thickness_*` with `return_power_spectrum=False` | the CTF times `sinc(pi * lambda * k^2 * t / 2)`, so equal to the CTF at zero thickness |
| `calculate_ctf_thickness_*` with `return_power_spectrum=True` | not affected by the sign |

`chi_s` is the symmetric phase above (plus any even Zernikes) and `chi_a` the
antisymmetric phase from beam tilt and odd Zernikes. Odd aberrations multiply the
real CTF by `exp(+i chi_a)` as in RELION, so beam tilt and odd Zernike
coefficients mean the same as they do there.

### Changed in 0.7.0

Before 0.7.0 the CTF had the opposite sign: `-sin(chi)`, starting at
`+amplitude_contrast` and going to +1 first (RELION's convention). To reproduce
earlier results:

```python
ctf = -calculate_ctf_2d(...)  # also for calculate_ctf_1d and calc_LPP_ctf_2D
ctfp, ctfq = calculate_ctfp_and_ctfq_2d(...)
ctfp, ctfq = -ctfp, -ctfq
```

The complex CTF (`return_complex_ctf=True`) already gave dark particles and is
unchanged without odd aberrations. With beam tilt or odd Zernikes it used to be
`exp(-i (chi_s + chi_a))`; the odd phase now has the sign that matches the real
CTF and `ctfp`/`ctfq`. The thickness amplitude CTF already used the new sign and
has not changed; it now agrees with `calculate_ctf_1d` and `calculate_ctf_2d`. The phase helpers
(`calculate_total_phase_shift`, `calculate_defocus_phase_aberration`, the Zernike
functions) are unchanged. If you multiplied projections by -1 before applying the
CTF to get dark particles, drop that flip.

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork jdickerson95/torch-ctf --clone
# or just
# gh repo clone jdickerson95/torch-ctf
cd torch-ctf
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
