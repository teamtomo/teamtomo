import math

import numpy as np
import pytest
import torch

from torch_fourier_filter.dose_weight import dose_weight_movie
from torch_fourier_filter.frame_weights import (
    EnvelopeCorrection,
    FrameWeightTable,
    TemplateEnvelope,
    default_bfactor_curve,
    default_frame_bfactors,
    frame_exposures,
    frame_weights_chunk,
    frame_weights_grid,
    frame_weights_normalization_grid,
    matched_envelope,
)


def _table(n=8, seed=0, b_scale=300.0):
    g = torch.Generator().manual_seed(seed)
    start, end = frame_exposures(n, 0.0, 1.0)
    B = -b_scale * torch.rand(n, generator=g, dtype=torch.float64)
    C = -0.5 * torch.rand(n, generator=g, dtype=torch.float64)
    return FrameWeightTable(start, end, B, C, voltage=300.0)


def test_table_validation():
    start, end = frame_exposures(3)
    with pytest.raises(ValueError):
        FrameWeightTable(start, end, [0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        FrameWeightTable(end, start, [0.0] * 3, [0.0] * 3)
    t = FrameWeightTable(start, end, [0, -10, -20], [0, -0.1, -0.2])
    assert t.n_frames == 3
    assert torch.allclose(
        t.exposure_mid, torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
    )


def test_table_json_roundtrip(tmp_path):
    t = _table()
    t.provenance = {"particles": 100}
    t.to_json(tmp_path / "w.json")
    u = FrameWeightTable.from_json(tmp_path / "w.json")
    for name in ("exposure_start", "exposure_end", "bfactor", "log_scale"):
        assert torch.equal(getattr(t, name), getattr(u, name))
    assert u.voltage == 300.0 and u.provenance == {"particles": 100}


def test_weights_are_normalised_and_match_the_model():
    t = _table()
    shape, px = (16, 16), 1.3
    w = frame_weights_grid(t, shape, px)
    assert w.shape == (t.n_frames, 16, 9)
    assert torch.allclose((w**2).sum(0), torch.ones(16, 9), atol=1e-5)
    k = (
        torch.sqrt(
            torch.fft.fftfreq(16)[:, None] ** 2 + torch.fft.rfftfreq(16)[None, :] ** 2
        ).double()
        / px
    )
    a = torch.exp(t.bfactor[:, None, None] * k**2 / 4 + t.log_scale[:, None, None])
    expected = a / torch.sqrt((a**2).sum(0))
    assert torch.allclose(w.double(), expected, atol=1e-5)


def test_common_offset_cancels():
    t = _table()
    u = FrameWeightTable(
        t.exposure_start, t.exposure_end, t.bfactor - 123.0, t.log_scale + 0.7
    )
    assert torch.allclose(
        frame_weights_grid(t, (12, 12), 1.0),
        frame_weights_grid(u, (12, 12), 1.0),
        atol=1e-6,
    )


def test_equal_frames_get_equal_weights():
    start, end = frame_exposures(5)
    t = FrameWeightTable(start, end, torch.zeros(5), torch.zeros(5))
    w = frame_weights_grid(t, (8, 8), 1.0)
    assert torch.allclose(w, torch.full_like(w, 1 / math.sqrt(5)))


def test_no_underflow_at_high_frequency():
    start, end = frame_exposures(4)
    t = FrameWeightTable(
        start, end, [-3000.0, -3100.0, -3500.0, -4000.0], [0.0, 0.0, 0.0, 0.0]
    )
    w = frame_weights_grid(
        t, (32, 32), 0.5
    )  # |k| up to 1.4 / A: exp(-3000 k^2 / 4) underflows float32
    assert torch.isfinite(w).all()
    assert torch.allclose((w**2).sum(0), torch.ones_like(w[0]), atol=1e-5)
    assert torch.allclose(
        w[0, 16, 16], torch.tensor(1.0), atol=1e-4
    )  # the least damaged frame takes it all


def test_dose_weight_movie_with_a_table():
    t = _table(n=7, seed=3)
    shape, px = (20, 20), 1.1
    g = torch.Generator().manual_seed(1)
    movie = torch.fft.rfftn(torch.randn(7, *shape, generator=g), dim=(-2, -1))
    expected = movie * frame_weights_grid(t, shape, px)
    full = dose_weight_movie(movie, shape, px, frame_weights=t)
    chunked = dose_weight_movie(
        movie, shape, px, frame_weights=t, memory_efficient=True, chunk_size=3
    )
    assert torch.allclose(full, expected, atol=1e-6)
    assert torch.allclose(chunked, expected, atol=1e-6)
    k, log_norm = frame_weights_normalization_grid(t, shape, px, chunk_size=2)
    pieces = [
        frame_weights_chunk(movie[s : s + 2], s, t, k, log_norm) for s in range(0, 7, 2)
    ]
    assert torch.allclose(torch.cat(pieces), expected, atol=1e-6)
    with pytest.raises(ValueError):
        dose_weight_movie(movie[:5], shape, px, frame_weights=t)


def test_default_curve():
    D = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 56.0])
    B, C = default_bfactor_curve(D)
    assert torch.allclose(
        B,
        torch.tensor(
            [0.0, -15.925, -76.436, -172.553, -326.917, -488.124], dtype=torch.float64
        ),
    )
    assert C[0] == 0 and (C[1:] < 0).all()
    x = np.linspace(0, 56, 400)
    B, C = default_bfactor_curve(x)
    assert (torch.diff(B) <= 1e-9).all() and (torch.diff(C) <= 1e-9).all()
    with pytest.warns(UserWarning):
        B70, _ = default_bfactor_curve([70.0])
    assert B70[0] < -488.124


def test_default_table_across_fractionations():
    a = default_frame_bfactors(*frame_exposures(50, 0.0, 1.0), voltage=200.0)
    b = default_frame_bfactors(*frame_exposures(100, 0.0, 0.5))
    assert a.source == "default" and a.voltage == 200.0
    # every frame takes the curve at its mid exposure, whatever the fractionation
    for t in (a, b):
        B, C = default_bfactor_curve(t.exposure_mid.numpy())
        assert torch.allclose(t.bfactor, B) and torch.allclose(t.log_scale, C)
    assert torch.allclose(a.exposure_mid[10], b.exposure_mid[20:22].mean())
    # a table carried to another fractionation keeps the curve
    r = a.resample(b.exposure_start, b.exposure_end)
    inside = (b.exposure_mid >= a.exposure_mid[0]) & (
        b.exposure_mid <= a.exposure_mid[-1]
    )
    assert torch.allclose(r.bfactor[inside], b.bfactor[inside], atol=0.5)
    same = a.resample(a.exposure_start, a.exposure_end)
    assert torch.allclose(same.bfactor, a.bfactor) and torch.allclose(
        same.log_scale, a.log_scale
    )


def test_matched_envelope():
    t = _table(n=10, seed=5)
    k = torch.linspace(0, 0.4, 41, dtype=torch.float64)
    e = matched_envelope(k, t)
    a = t.amplitudes(k)
    expected = torch.sqrt((a**2).sum(0)) / torch.sqrt((a[:, :1] ** 2).sum(0))
    assert torch.allclose(e, expected, atol=1e-10)
    assert e[0] == pytest.approx(1.0)
    # a sum weighted unlike the damage has a smaller envelope than the matched one
    # (Cauchy-Schwarz),
    # relative to the same normalisation at k = 0 when the scales agree there
    start, end = frame_exposures(10)
    flat = FrameWeightTable(start, end, torch.zeros(10), t.log_scale)
    e_mismatched = matched_envelope(k, flat, damage_table=t)
    assert (e_mismatched <= e + 1e-12).all()


def test_envelope_correction():
    flat = EnvelopeCorrection.flat()
    k = torch.linspace(0.001, 0.5, 50)
    assert torch.allclose(flat(k), torch.ones_like(k))
    c = EnvelopeCorrection.uniform_knots(
        5, 50.0, 2.9, log_values=[0.0, -0.1, 0.2, 0.1, -0.3]
    )
    assert torch.allclose(c(c.knot_frequency), torch.exp(c.log_values), atol=1e-10)
    below = c(torch.tensor([1 / 100.0, 1 / 60.0], dtype=torch.float64))
    above = c(torch.tensor([1 / 2.5, 1 / 2.0], dtype=torch.float64))
    assert torch.allclose(below, below[:1].expand(2)) and torch.allclose(
        above, above[:1].expand(2)
    )
    with pytest.raises(ValueError):
        EnvelopeCorrection([0.1, 0.05], [0.0, 0.0])


def test_template_envelope_default_and_grid(tmp_path):
    env = TemplateEnvelope.default(0.0, 50.0)
    assert env.frame_weights.n_frames == 50 and env.correction is None
    k = torch.linspace(0, 0.5, 101, dtype=torch.float64)
    h = env(k)
    assert h[0] == pytest.approx(1.0) and (torch.diff(h) <= 1e-12).all()
    g = env.grid((32, 32, 32), pixel_size=1.0)
    assert g.shape == (32, 32, 17)
    assert g[0, 0, 0] == pytest.approx(1.0, abs=1e-6)
    axis = env(torch.arange(17, dtype=torch.float64) / 32)
    assert torch.allclose(g[0, 0, :].double(), axis, atol=1e-3)
    env.correction = EnvelopeCorrection.uniform_knots(
        log_values=[0.0, 0.1, 0.2, 0.0, -0.2]
    )
    env.to_json(tmp_path / "env.json")
    back = TemplateEnvelope.from_json(tmp_path / "env.json")
    assert torch.allclose(back(k), env(k))
