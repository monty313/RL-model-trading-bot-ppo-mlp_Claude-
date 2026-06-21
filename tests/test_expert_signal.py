"""Tests for the SOFT expert-signal observation layer (Phase 1, pure engine).

Covers: the block contract (names/dim/bounds), the vote aggregation against hand-crafted
law states, the per-STRAT toggles, the STRAT-006 volatility blend, the session weight, the
soft do_not_trade, and an end-to-end run on a real precomputed matrix (finite + bounded +
law_states passthrough equivalence). No wiring is exercised — this layer touches nothing in
the existing observation yet. See docs/EXPERT_SIGNAL_DESIGN.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantra.locked_core.laws.laws import compute_law_states
from quantra.market_pipeline.expert_signal import (
    DEFAULT_EXPERT_CONFIG,
    EXPERT_DIM,
    EXPERT_NAMES,
    ExpertSignalConfig,
    compute_expert_signals,
    expert_signals_dict,
)
from quantra.market_pipeline.expert_signal.engine import (
    _LAW_BB,
    _LAW_CCI,
    _LAW_SSMA,
)
from quantra.market_pipeline.feature_builder.builder import build_market_matrix
from quantra.market_pipeline.feature_builder.schema import (
    PRECOMPUTED_DIM,
    PRECOMPUTED_NAMES,
)

# Feature index within an output row, by name.
_OUT = {name: i for i, name in enumerate(EXPERT_NAMES)}


def _zeros_row() -> np.ndarray:
    return np.zeros((1, PRECOMPUTED_DIM), dtype=np.float32)


def _set(mat: np.ndarray, **cols) -> np.ndarray:
    for name, val in cols.items():
        mat[0, PRECOMPUTED_NAMES.index(name)] = val
    return mat


def _laws(direction9, signals3=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Build a (1,12) law-state row: 9 directional + 3 market-condition signals."""
    return np.array(list(direction9) + list(signals3), dtype=np.float32).reshape(1, 12)


def _sincos(hour: float):
    ang = 2.0 * np.pi * hour / 24.0
    return float(np.sin(ang)), float(np.cos(ang))


# --------------------------------------------------------------------------- contract
def test_block_contract():
    assert EXPERT_DIM == len(EXPERT_NAMES) == 8
    assert len(set(EXPERT_NAMES)) == EXPERT_DIM          # unique
    assert all(n.startswith("expert_") for n in EXPERT_NAMES)


def test_shapes_row_and_matrix():
    row = _zeros_row()[0]
    out_row = compute_expert_signals(row)
    assert out_row.shape == (EXPERT_DIM,)
    assert out_row.dtype == np.float32

    mat = np.zeros((7, PRECOMPUTED_DIM), dtype=np.float32)
    out_mat = compute_expert_signals(mat)
    assert out_mat.shape == (7, EXPERT_DIM)


def test_expert_signals_dict_keys():
    d = expert_signals_dict(compute_expert_signals(_zeros_row()[0]))
    assert list(d.keys()) == EXPERT_NAMES
    assert all(isinstance(v, float) for v in d.values())


# --------------------------------------------------------------- vote aggregation
def _full_vol_row():
    """A row whose ATR/ADX columns give volatility_ok == 1 (so tradeability == confidence)."""
    mat = _zeros_row()
    return _set(mat, atr_dev_5m=1.0, adx5_30m=0.2, adx15_30m=0.0)


def test_all_long_laws():
    mat = _full_vol_row()
    out = compute_expert_signals(mat, law_states=_laws([1] * 9))[0]
    assert out[_OUT["expert_regime_bias"]] == pytest.approx(1.0)
    assert out[_OUT["expert_confidence"]] == pytest.approx(1.0)
    assert out[_OUT["expert_trend_strength"]] == pytest.approx(1.0)
    assert out[_OUT["expert_volatility_ok"]] == pytest.approx(1.0)
    assert out[_OUT["expert_do_not_trade"]] == pytest.approx(0.0)
    assert out[_OUT["expert_long"]] == pytest.approx(1.0)
    assert out[_OUT["expert_short"]] == pytest.approx(0.0)


def test_all_short_laws():
    mat = _full_vol_row()
    out = compute_expert_signals(mat, law_states=_laws([-1] * 9))[0]
    assert out[_OUT["expert_regime_bias"]] == pytest.approx(-1.0)
    assert out[_OUT["expert_confidence"]] == pytest.approx(1.0)
    assert out[_OUT["expert_short"]] == pytest.approx(1.0)
    assert out[_OUT["expert_long"]] == pytest.approx(0.0)


def test_conflict_lowers_confidence():
    # super=+1 (×3), trend=-1 (×3), pullback=0 (×3). Net long but contested.
    mat = _full_vol_row()
    out = compute_expert_signals(mat, law_states=_laws([1, 1, 1, -1, -1, -1, 0, 0, 0]))[0]
    cfg = DEFAULT_EXPERT_CONFIG
    wtot = 3 * (cfg.w_super + cfg.w_trend + cfg.w_pullback)
    expected_conf = (3 * cfg.w_super - 3 * cfg.w_trend) / wtot
    assert out[_OUT["expert_regime_bias"]] == pytest.approx(1.0)
    assert out[_OUT["expert_confidence"]] == pytest.approx(expected_conf, abs=1e-5)
    assert 0.0 < out[_OUT["expert_confidence"]] < 1.0
    # strong families: 3 of 6 agree with +1 -> 0.5
    assert out[_OUT["expert_trend_strength"]] == pytest.approx(0.5)


def test_all_zero_laws_is_full_chop():
    out = compute_expert_signals(_zeros_row(), law_states=_laws([0] * 9))[0]
    assert out[_OUT["expert_regime_bias"]] == pytest.approx(0.0)
    assert out[_OUT["expert_confidence"]] == pytest.approx(0.0)
    assert out[_OUT["expert_trend_strength"]] == pytest.approx(0.0)
    assert out[_OUT["expert_do_not_trade"]] == pytest.approx(1.0)
    assert out[_OUT["expert_long"]] == pytest.approx(0.0)
    assert out[_OUT["expert_short"]] == pytest.approx(0.0)


# ------------------------------------------------------------------- per-STRAT toggles
def test_cci_toggle_zeroes_cci_only_signal():
    # Only CCI laws fire long; everything else 0.
    direction = [0] * 9
    for i in _LAW_CCI:
        direction[i] = 1
    mat = _full_vol_row()
    on = compute_expert_signals(mat, law_states=_laws(direction))[0]
    assert on[_OUT["expert_regime_bias"]] == pytest.approx(1.0)

    off = compute_expert_signals(
        mat, law_states=_laws(direction),
        cfg=ExpertSignalConfig(use_cci=False),
    )[0]
    assert off[_OUT["expert_regime_bias"]] == pytest.approx(0.0)
    assert off[_OUT["expert_confidence"]] == pytest.approx(0.0)


def test_bb_and_ssma_toggles_independent():
    # BB long only -> disabling SSMA leaves BB bias intact; disabling BB removes it.
    direction = [0] * 9
    for i in _LAW_BB:
        direction[i] = 1
    mat = _full_vol_row()
    assert compute_expert_signals(
        mat, law_states=_laws(direction), cfg=ExpertSignalConfig(use_ssma=False)
    )[0][_OUT["expert_regime_bias"]] == pytest.approx(1.0)
    assert compute_expert_signals(
        mat, law_states=_laws(direction), cfg=ExpertSignalConfig(use_bb=False)
    )[0][_OUT["expert_regime_bias"]] == pytest.approx(0.0)


def test_all_strategies_off_is_neutral():
    cfg = ExpertSignalConfig(use_bb=False, use_cci=False, use_ssma=False)
    out = compute_expert_signals(_full_vol_row(), law_states=_laws([1] * 9), cfg=cfg)[0]
    assert out[_OUT["expert_regime_bias"]] == pytest.approx(0.0)
    assert out[_OUT["expert_confidence"]] == pytest.approx(0.0)


# ------------------------------------------------------------------- volatility (STRAT-006)
def test_volatility_ok_blend():
    # Only ATR expanding -> w_atr share; add ADX rising -> full.
    atr_only = compute_expert_signals(
        _set(_zeros_row(), atr_dev_5m=1.0), law_states=_laws([1] * 9)
    )[0][_OUT["expert_volatility_ok"]]
    assert atr_only == pytest.approx(DEFAULT_EXPERT_CONFIG.w_atr, abs=1e-5)

    both = compute_expert_signals(
        _set(_zeros_row(), atr_dev_5m=1.0, adx5_30m=0.5, adx15_30m=0.0),
        law_states=_laws([1] * 9),
    )[0][_OUT["expert_volatility_ok"]]
    assert both == pytest.approx(1.0)


def test_volatility_quiet_market_low():
    out = compute_expert_signals(
        _set(_zeros_row(), atr_dev_5m=-0.5, adx5_30m=0.0, adx15_30m=0.3),
        law_states=_laws([1] * 9),
    )[0]
    assert out[_OUT["expert_volatility_ok"]] == pytest.approx(0.0)
    # confident regime but no movement -> tradeability 0 -> full chop
    assert out[_OUT["expert_do_not_trade"]] == pytest.approx(1.0)


def test_volatility_toggle_off_is_neutral_one():
    out = compute_expert_signals(
        _zeros_row(), law_states=_laws([1] * 9),
        cfg=ExpertSignalConfig(use_adx_atr=False),
    )[0]
    assert out[_OUT["expert_volatility_ok"]] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- session
def test_session_peak_vs_offhours():
    s, c = _sincos(10.0)                      # London/NY core
    peak = compute_expert_signals(
        _set(_zeros_row(), time_sin_hour=s, time_cos_hour=c), law_states=_laws([0] * 9)
    )[0][_OUT["expert_session_ok"]]
    assert peak == pytest.approx(1.0)

    s, c = _sincos(2.0)                       # overnight / Asia
    off = compute_expert_signals(
        _set(_zeros_row(), time_sin_hour=s, time_cos_hour=c), law_states=_laws([0] * 9)
    )[0][_OUT["expert_session_ok"]]
    assert off == pytest.approx(DEFAULT_EXPERT_CONFIG.session_default)


def test_session_toggle_off_is_one():
    s, c = _sincos(2.0)
    out = compute_expert_signals(
        _set(_zeros_row(), time_sin_hour=s, time_cos_hour=c),
        law_states=_laws([0] * 9), cfg=ExpertSignalConfig(use_session=False),
    )[0]
    assert out[_OUT["expert_session_ok"]] == pytest.approx(1.0)


# -------------------------------------------------------------- end-to-end on real matrix
def test_end_to_end_real_matrix_bounds(make_1m):
    mm = build_market_matrix(make_1m(n_bars=6000, seed=3))
    out = compute_expert_signals(mm.matrix)
    assert out.shape == (mm.matrix.shape[0], EXPERT_DIM)
    assert np.isfinite(out).all()
    rb = out[:, _OUT["expert_regime_bias"]]
    assert np.isin(rb, (-1.0, 0.0, 1.0)).all()
    # every non-bias feature is bounded to [0,1]
    others = np.delete(out, _OUT["expert_regime_bias"], axis=1)
    assert (others >= 0.0).all() and (others <= 1.0).all()


def test_law_states_passthrough_matches_internal(make_1m):
    mm = build_market_matrix(make_1m(n_bars=4000, seed=5))
    internal = compute_expert_signals(mm.matrix)
    passthrough = compute_expert_signals(mm.matrix, law_states=compute_law_states(mm.matrix))
    np.testing.assert_array_equal(internal, passthrough)


def test_deterministic(make_1m):
    mm = build_market_matrix(make_1m(n_bars=3000, seed=7))
    a = compute_expert_signals(mm.matrix)
    b = compute_expert_signals(mm.matrix)
    np.testing.assert_array_equal(a, b)
