"""QUANTRA MASTER TEST SUITE — the one big runnable test for the whole folder.

WHY THIS FILE EXISTS (read before adding tests)
-----------------------------------------------
STANDING RULE [2026-06-13]: this is the SINGLE place every Quantra test lives.
All future tests — for any milestone (laws, env, agent, reward, telemetry, ...) —
are APPENDED here under the matching section, never scattered into new files. One
command runs the entire folder's guarantees:

    pytest tests/test_ftmo_master_suite.py        # or just: pytest

WHAT IT PROVES (everything is judged against repeated FTMO passing, not PnL)
---------------------------------------------------------------------------
The suite is organised along the interpretability chain from
``docs/MLP_INTERPRETABILITY_LAYER.md`` (State Vector -> Law -> Hidden State ->
Heads -> Risk -> Reward -> Outcome), plus the cost/speed substrate that lets us
afford enough seeds and walk-forward windows to *establish* a pass rate:

    SECTION A  Runtime / hardware efficiency   (afford the validation budget)
    SECTION B  Data pipeline - loader          (faithful FTMO situation in)
    SECTION C  Data pipeline - resampler       (no lookahead -> edge transfers)
    (SECTION D+ added as M2+ land: features, laws, env, agent, reward, telemetry)

HOW THE LLM RISK DOCTOR SHOULD USE THIS FILE
--------------------------------------------
A green master suite means the *substrate* the bot trains on is faithful and fast;
it does NOT by itself prove the policy passes. When triangulating a pass-rate
regression, first confirm this suite is green — a red data/resampler test means the
bot's world is corrupt (e.g. lookahead leakage masquerading as a learned edge), and
that is the break, not the actor/critic. Only if the substrate is sound do you walk
the chain into hidden-state / reward / critic diagnostics.

Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest
import torch

import quantra.runtime.config as cfg
from quantra.market_pipeline.data_loader import OHLCV_COLUMNS, load_symbol, parse_mt5_csv
from quantra.market_pipeline.feature_builder import (
    EXPECTED_WIDTHS,
    MARKET_DIM,
    MARKET_NAMES,
    PRECOMPUTED_DIM,
    PRECOMPUTED_NAMES,
    RAW_FEATURE_NAMES,
    SCHEMA,
    STATE_DIM,
    assemble_state,
    build_market_matrix,
)
from quantra.market_pipeline.resampler import (
    as_of_higher_tf,
    build_all_timeframes,
    resample_ohlcv,
)
from quantra.locked_core.laws import LAW_NAMES, compute_law_states
from quantra.locked_core.cost_layer import CostLayer
from quantra.locked_core.risk_manager import RiskManager
from quantra.ftmo_passing import ChallengeState
from quantra.runtime.config import ChallengeConfig, RiskConfig
from quantra.env import SymbolData, TradingEnv
from quantra.learning_system.ppo_agent import ActorCritic, PPOAgent, ppo_loss
from quantra.learning_system.rollout_buffer import FIELDS, RolloutBuffer
from quantra.market_pipeline.law_mask_engine import (
    CLOSE,
    HOLD,
    MODE_LIVE,
    MODE_SCHOOL,
    OPEN_LONG,
    OPEN_SHORT,
    LawMask,
    build_direction_mask,
    build_pointer_mask,
)
from quantra.runtime import HardwareConfig, RuntimeConfig, plan
from quantra.runtime.autoscale import plan_cpu_scale
from quantra.runtime.device import RepresentativePolicy, available_devices
from quantra.runtime.throughput_benchmark import race_devices
from tools import impact, snapshot  # change-impact tracker (repo root on sys.path via conftest)


# =============================================================================
# SECTION A — RUNTIME / HARDWARE EFFICIENCY
# FTMO link: the validation that selects brains by pass-rate needs many windows x
# 7 seeds. These tests guarantee we run on the cheapest fast device at ~80% load,
# so that budget is affordable and no paid GPU is wasted on a 3x256 MLP.
# =============================================================================
def test_representative_policy_matches_locked_architecture():
    """The benchmark net must mirror the locked four-head 3x256 design.

    FTMO link: an honest device race must time the REAL workload shape (direction
    {HOLD,OPEN_LONG,OPEN_SHORT,CLOSE} + Beta size + 5-slot pointer + value), or we
    might pick a device that is slower for the actual policy and waste training time.
    """
    import torch

    net = RepresentativePolicy(state_dim=145)
    d, s, p, v = net(torch.randn(8, 145))
    assert d.shape == (8, 4)
    assert s.shape == (8, 2)
    assert p.shape == (8, 5)
    assert v.shape == (8, 1)
    linears = [m for m in net.trunk if m.__class__.__name__ == "Linear"]
    assert len(linears) == 3 and all(l.out_features == 256 for l in linears)


def test_cpu_is_always_a_candidate_device():
    devs = available_devices()
    assert devs and devs[0].kind == "cpu", "CPU must be the first/default candidate"


def test_benchmark_reports_positive_throughput():
    results = race_devices(state_dim=64, batch=256, seconds=0.3)
    cpu = next(r for r in results if r.kind == "cpu")
    assert cpu.steps_per_sec > 0


def test_autoscale_targets_80pct_with_headroom():
    """n_envs/threads never exceed (cores - reserved); ~80% target, kernel stays alive."""
    hw = HardwareConfig(utilization_target=0.80, reserved_cores=1)
    sp = plan_cpu_scale(hw)
    cores = os.cpu_count() or 1
    assert hw.min_envs <= sp.n_envs <= hw.max_envs
    assert sp.torch_threads <= max(1, cores - hw.reserved_cores)


def test_plan_prefers_cpu_when_no_accelerator():
    """On a CPU-only box the plan MUST pick CPU (the cheap default) — money guard."""
    import torch

    p = plan(state_dim=RuntimeConfig().nominal_state_dim, benchmark_seconds=0.3)
    assert p.n_envs >= 1
    if not torch.cuda.is_available():
        assert p.device_kind == "cpu" and p.device == "cpu"


def test_challenge_defaults_are_the_locked_ftmo_values():
    """Defaults must be 2.5%/4.0% (SOW-A3). A wrong default silently trains/judges
    against the wrong wall, so every 'pass' would be meaningless."""
    c = RuntimeConfig().challenge
    assert c.daily_target_pct == 2.5
    assert c.daily_risk_pct == 4.0
    assert c.phase_b_trailing_pct == 1.0
    assert c.hard_wall_pct == 4.0


# =============================================================================
# SECTION B — DATA PIPELINE: LOADER
# FTMO link: the bot can only learn to pass the challenge it is shown. Faithful
# bars (correct stamps, real spread, deduped, monotonic) are the ground truth the
# laws, costs, and risk walls all read.
# =============================================================================
def test_parse_tab_delimited_mt5_export(make_1m, write_mt5_csv):
    df = make_1m(n_bars=600, seed=1)
    parsed, meta = parse_mt5_csv(write_mt5_csv(df, sep="\t"))
    assert list(parsed.columns) == OHLCV_COLUMNS
    assert meta["delimiter"] == "\t" and meta["had_spread"] is True
    assert parsed.index.is_monotonic_increasing and len(parsed) == 600
    assert abs(parsed["close"].iloc[-1] - df["close"].iloc[-1]) < 1e-4


def test_parse_comma_delimited_variant(make_1m, write_mt5_csv):
    parsed, meta = parse_mt5_csv(write_mt5_csv(make_1m(n_bars=200, seed=2), sep=","))
    assert meta["delimiter"] == "," and len(parsed) == 200


def test_parser_drops_duplicate_timestamps(make_1m, write_mt5_csv):
    df = make_1m(n_bars=100, seed=3)
    dup = pd.concat([df, df.iloc[:10]]).sort_index()
    parsed, meta = parse_mt5_csv(write_mt5_csv(dup, sep="\t"))
    assert meta["dropped_duplicates"] == 10 and parsed.index.is_unique


def test_load_symbol_local_source(make_1m, write_mt5_csv):
    out, rep = load_symbol("EURUSD", path=write_mt5_csv(make_1m(n_bars=150, seed=4)),
                           use_cache=False)
    assert rep.rows == 150 and rep.source == "local"
    assert list(out.columns) == OHLCV_COLUMNS


def test_parquet_cache_used_on_second_load(make_1m, write_mt5_csv, monkeypatch):
    """Efficiency: first load parses CSV + writes Parquet; second reads the cache.

    FTMO link: cheap re-loads are what make many walk-forward windows affordable;
    a broken cache would 10x the cost of establishing a pass rate.
    """
    name = cfg.DRIVE_FILENAMES["EURUSD"]
    path = write_mt5_csv(make_1m(n_bars=300, seed=11), name=name)
    monkeypatch.setattr(cfg, "RAW_DIR", path.parent)
    monkeypatch.setattr(cfg, "PARQUET_DIR", path.parent)
    out1, rep1 = load_symbol("EURUSD")
    out2, rep2 = load_symbol("EURUSD")
    assert rep1.source == "local" and rep2.source == "cache"
    assert len(out1) == len(out2) == 300
    assert (cfg.PARQUET_DIR / "EURUSD_1m.parquet").exists()


def test_parse_throughput_is_reasonable(make_1m, write_mt5_csv):
    """Efficiency guard: parsing 20k bars must not regress into minutes.

    FTMO link: the offline data step must stay cheap so iteration time (and cost)
    per validated seed stays low. Bound is generous — it only catches pathologies.
    """
    path = write_mt5_csv(make_1m(n_bars=20_000, seed=12))
    t0 = time.perf_counter()
    parsed, _ = parse_mt5_csv(path)
    assert len(parsed) == 20_000
    assert (time.perf_counter() - t0) < 10.0


# =============================================================================
# SECTION C — DATA PIPELINE: RESAMPLER (NO LOOKAHEAD)
# FTMO link: lookahead is the deadliest trading-RL bug. A bot that peeks at an
# unfinished 5m/30m/4H bar learns a fantasy edge that vanishes live and walks into
# the 4% wall. These tests make peeking structurally impossible.
# =============================================================================
def test_resample_5m_ohlc_aggregation(make_1m):
    df = make_1m(n_bars=60, seed=5, start="2021-01-04 00:00:00")
    five = resample_ohlcv(df, "5min")
    block = df.iloc[0:5]
    assert five.index[0] == pd.Timestamp("2021-01-04 00:05:00")  # close-time stamp
    assert five.iloc[0]["open"] == block["open"].iloc[0]
    assert five.iloc[0]["high"] == block["high"].max()
    assert five.iloc[0]["low"] == block["low"].min()
    assert five.iloc[0]["close"] == block["close"].iloc[-1]
    assert five.iloc[0]["tick_volume"] == block["tick_volume"].sum()


def test_build_all_timeframes_keys(make_1m):
    assert set(build_all_timeframes(make_1m(n_bars=1000, seed=6))) == {"1m", "5m", "30m", "4H"}


def test_as_of_merge_has_no_lookahead(make_1m):
    df = make_1m(n_bars=120, seed=7)
    five = resample_ohlcv(df, "5min")
    merged = as_of_higher_tf(df.index, five, suffix="5m")
    for t in [df.index[3], df.index[7], df.index[42], df.index[99]]:
        val = merged.loc[t, "close_5m"]
        eligible = five[five.index <= t]
        if eligible.empty:
            assert pd.isna(val)
        else:
            assert val == eligible["close"].iloc[-1]  # last CLOSED bar only


# =============================================================================
# SECTION D — FEATURE BUILDER + STATE VECTOR (~145 scalars)
# FTMO link: a complete, bounded, lookahead-free observation is what lets the MLP
# tell breach-risk from safe trading (Term 1). Drift, leakage, or NaN here is a top
# root cause of inconsistent passing — these guards make all three impossible.
# =============================================================================
def test_schema_total_is_207_with_raw_bollinger_and_training_wheels():
    # market 131 (prev 128 + 3 new 5m-ATR feats for market_volatility_obs) + 18 raw
    # price-SMA + 12 law + 35 trade + 3 portfolio + 8 account (C12 +acct_dist_to_perm_dd) = 207
    assert STATE_DIM == 207
    for name, width in EXPECTED_WIDTHS.items():
        s, e = SCHEMA.block_spans[name]
        assert e - s == width, f"block {name} width drifted"
    assert EXPECTED_WIDTHS["market"] == 131 and EXPECTED_WIDTHS["market_raw"] == 18
    # the two training-wheel block flags exist (the operator's enforced "acts")
    assert "tw_cci_block" in SCHEMA.feature_names and "tw_bb_block" in SCHEMA.feature_names
    assert len(SCHEMA.feature_names) == STATE_DIM
    assert len(set(SCHEMA.feature_names)) == STATE_DIM  # names unique


def test_config_nominal_state_dim_matches_schema():
    """The hardware race must time the TRUE observation width (no wasted spend)."""
    from quantra.runtime import RuntimeConfig

    assert RuntimeConfig().nominal_state_dim == STATE_DIM


def test_precomputed_matrix_shape_finite_and_normalized_clipped(make_1m):
    df = make_1m(n_bars=4000, seed=20)
    mm = build_market_matrix(df)
    assert mm.matrix.shape == (4000, PRECOMPUTED_DIM)     # (T, 119) = market 89 + raw 30
    assert mm.matrix.dtype == np.float32
    assert np.isfinite(mm.matrix).all()                   # no NaN/inf reaches the policy
    # ONLY the normalized columns are clipped to ±10; raw levels are exempt.
    norm_idx = [i for i, n in enumerate(PRECOMPUTED_NAMES) if n not in RAW_FEATURE_NAMES]
    assert np.abs(mm.matrix[:, norm_idx]).max() <= 10.0 + 1e-4
    assert mm.names == PRECOMPUTED_NAMES                   # telemetry block labels intact


def test_raw_block_present_finite_and_unclipped(make_1m):
    """Raw SMA/CCI block exists (30), is finite, and is NOT squashed to ±10.

    FTMO link: the operator wants un-transformed levels; clipping would destroy
    them. They're flagged (RAW_FEATURE_NAMES) so the M5 agent standardizes them and
    the LLM can attribute any instability to the raw block specifically.
    """
    mm = build_market_matrix(make_1m(n_bars=4000, seed=26))
    raw_idx = [i for i, n in enumerate(PRECOMPUTED_NAMES) if n in RAW_FEATURE_NAMES]
    assert len(raw_idx) == 68      # 18 raw price-SMA + 24 raw CCI + 18 raw Bollinger + 8 raw wheel CCI
    assert np.isfinite(mm.matrix[:, raw_idx]).all()
    # a raw Bollinger band level exists and is a real price (unclipped, > the ±10 clip)
    boll_raw = [i for i, n in enumerate(PRECOMPUTED_NAMES) if n.startswith("boll_") and "_raw_" in n]
    assert len(boll_raw) == 18
    # raw CCI value (not the cci_sma) is unbounded -> proves no clip on the raw block
    cci_cols = [i for i, n in enumerate(PRECOMPUTED_NAMES)
                if n in RAW_FEATURE_NAMES and n.startswith("cci") and "_sma_" not in n]
    assert np.abs(mm.matrix[1000:, cci_cols]).max() > 10.0


def test_market_features_carry_signal(make_1m):
    """Fast 1m features must vary after warmup (real signal, not a dead constant)."""
    mm = build_market_matrix(make_1m(n_bars=4000, seed=21))
    for feat in ["z10_1m", "cci10_1m", "candle_return_1m"]:
        col = mm.matrix[1000:, MARKET_NAMES.index(feat)]
        assert col.std() > 0, f"{feat} is constant"


def test_feature_builder_has_no_lookahead(make_1m):
    """Building on a truncated series reproduces the same row -> no forward peek.

    This is THE guard against a fantasy edge: if features at bar t changed when
    future bars exist, the bot would 'see' the future, pass in backtest, and breach
    live. Closed-bar-only resampling + rolling/shift make that impossible.
    """
    df = make_1m(n_bars=4000, seed=22)
    full = build_market_matrix(df).matrix
    for i in (1500, 2731, 3990):
        trunc = build_market_matrix(df.iloc[: i + 1]).matrix
        assert np.allclose(full[i], trunc[-1], atol=1e-5), f"lookahead at row {i}"


def test_time_features_are_cyclical(make_1m):
    mm = build_market_matrix(make_1m(n_bars=300, seed=23))
    s = mm.matrix[:, MARKET_NAMES.index("time_sin_hour")]
    c = mm.matrix[:, MARKET_NAMES.index("time_cos_hour")]
    assert np.allclose(s ** 2 + c ** 2, 1.0, atol=1e-4)


def test_assemble_state_full_width_and_block_validation(make_1m):
    mm = build_market_matrix(make_1m(n_bars=500, seed=24))
    state = assemble_state(mm.matrix[400])                 # law/trade/acct zero-filled
    assert state.shape == (STATE_DIM,) and state.dtype == np.float32
    assert mm.matrix.shape[1] == PRECOMPUTED_DIM           # row width feeds assemble
    with pytest.raises(ValueError):                        # wrong block size fails loud
        assemble_state(mm.matrix[400], law_flags=np.zeros(3))   # law block must be 12
    with pytest.raises(ValueError):                        # wrong precomputed width
        assemble_state(mm.matrix[400][:50])


def test_valid_from_after_warmup(make_1m):
    """With enough bars the non-4H features warm up, so valid_from < T (env has bars)."""
    df = make_1m(n_bars=8000, seed=25)
    mm = build_market_matrix(df)
    assert 0 <= mm.valid_from < len(df)


# =============================================================================
# SECTION E — CHANGE-IMPACT TRACKER (observation/pipeline drift guard)
# FTMO link: the observation is the policy's whole world. A silent change to it
# invalidates normalization, the agent input dim, telemetry, and checkpoints — any
# of which can quietly wreck pass-rate. These guards make such a change impossible
# to do accidentally and give the LLM an FTMO-framed map of the blast radius.
# =============================================================================
def test_state_vector_snapshot_matches():
    """The live schema must match the committed snapshot, or fail with a checklist."""
    deltas = snapshot.diff(snapshot.load(), snapshot.build())
    msg = (
        "STATE-VECTOR DRIFT:\n  " + "\n  ".join(deltas)
        + "\n\nFollow-ups (relative to passing FTMO):\n  "
        + "\n  ".join(snapshot.checklist(deltas))
        + "\n\nIf this change is intended: `python tools/snapshot.py --update`"
    )
    assert not deltas, msg


def test_snapshot_carries_llm_interpretation():
    """The snapshot must stay LLM-readable + FTMO-framed (operator requirement)."""
    snap = snapshot.load()
    assert snap["state_dim"] == STATE_DIM
    interp = snap["_llm_interpretation"]
    assert "market_raw" in interp["blocks"] and "account" in interp["blocks"]
    assert interp["drift_means"]  # concrete follow-ups present


def test_impact_graph_traces_schema_dependents():
    """The AST graph must resolve relative imports and find schema's dependents."""
    graph = impact.build_graph()
    schema_mod = impact._module_name(
        impact.REPO_ROOT / "quantra" / "market_pipeline" / "feature_builder" / "schema.py"
    )
    affected = impact.reverse_closure({schema_mod}, graph)
    assert any(m.endswith("builder") for m in affected), "schema dependents not traced"


def test_impact_report_is_ftmo_framed():
    rpt = impact.report(["quantra/market_pipeline/feature_builder/schema.py"])
    assert "passing FTMO" in rpt
    assert "snapshot.py --check" in rpt  # actionable follow-up present


# =============================================================================
# SECTION F — LAWMASK (9 laws + 3 gates, two enforcement modes)
# FTMO link: laws are the bot's spine — they forbid the wrong direction with logit
# -1e9 BEFORE the policy acts. A correct mask is the mechanical reason the bot can't
# trade itself into the 4% wall. Laws are masks, never rewards (SOW R5).
# =============================================================================
_LIDX = {n: i for i, n in enumerate(LAW_NAMES)}


def _feat_row(**named):
    """A PRECOMPUTED_DIM zero feature row with specific named features set."""
    r = np.zeros(PRECOMPUTED_DIM, dtype=np.float32)
    for k, v in named.items():
        r[PRECOMPUTED_NAMES.index(k)] = v
    return r


def _law(row, name):
    return compute_law_states(row)[_LIDX[name]]


def test_law_neutral_row_all_directional_inactive():
    s = compute_law_states(_feat_row())
    assert s.shape == (12,)
    assert np.all(s[:9] == 0)  # no directional law active on a zero row


def test_super_trend_bb_buy_and_sell():
    buy = _feat_row(boll_bb20_up_5m=0.5, boll_bb200_up_5m=0.5,
                    boll_bb20_up_30m=0.5, boll_bb200_up_30m=0.5)
    assert _law(buy, "law_super_trend_bb") == 1
    sell = _feat_row(boll_bb20_lo_5m=-0.5, boll_bb200_lo_5m=-0.5,
                     boll_bb20_lo_30m=-0.5, boll_bb200_lo_30m=-0.5)
    assert _law(sell, "law_super_trend_bb") == -1


def test_super_trend_cci_requires_above_100():
    # RAW CCI: all four above their SMA but NOT above +100 -> super-trend OFF, trend ON
    above_sma = _feat_row(cci30_5m=20, cci30_sma_5m=10, cci100_5m=20, cci100_sma_5m=10,
                          cci30_30m=20, cci30_sma_30m=10, cci100_30m=20, cci100_sma_30m=10)
    assert _law(above_sma, "law_super_trend_cci") == 0   # raw CCI 20 is not > +100
    assert _law(above_sma, "law_trend_cci") == 1         # trend (no +100) DOES fire
    # raw CCI above +100 AND above its SMA -> super-trend ON
    extreme = _feat_row(cci30_5m=150, cci30_sma_5m=120, cci100_5m=150, cci100_sma_5m=120,
                        cci30_30m=150, cci30_sma_30m=120, cci100_30m=150, cci100_sma_30m=120)
    assert _law(extreme, "law_super_trend_cci") == 1


def test_shifted_sma_laws_use_align_flags():
    buy = _feat_row(ssma_align_1m=1, ssma_align_5m=1, ssma_align_30m=1)
    assert _law(buy, "law_super_trend_ssma") == 1       # needs 1m+5m+30m
    assert _law(buy, "law_trend_ssma") == 1             # needs 5m+30m
    pb = _feat_row(ssma_align_5m=1, ssma_align_30m=1, ssma_align_1m=-1)
    assert _law(pb, "law_pullback_ssma") == 1           # 1m pulls back inside HTF up
    assert _law(pb, "law_super_trend_ssma") == 0


def test_pullback_cci_desync():
    # RAW: 30m both CCIs above their SMA; 5m large(100) above while small(10) below
    buy = _feat_row(cci10_30m=20, cci10_sma_30m=10, cci100_30m=20, cci100_sma_30m=10,
                    cci100_5m=20, cci100_sma_5m=10, cci10_5m=-5, cci10_sma_5m=10)
    assert _law(buy, "law_pullback_cci") == 1


def test_market_condition_signals_volatility_spread_stationarity():
    # Renamed gates -> observation signals (2026-06-18). Volatility now reads 5m + 4H ATR.
    g = _feat_row(atr_dev_5m=0.1, atr_dev_4H=0.1, spread_range_ratio_1m=0.3, adf_stat_1m=-3.5)
    assert _law(g, "market_volatility_obs") == 1
    assert _law(g, "market_spread_obs") == 1
    assert _law(g, "market_stationarity_obs") == 1      # adf below -2.86 -> stationary
    closed = _feat_row(atr_dev_5m=-0.1, atr_dev_4H=-0.1, spread_range_ratio_1m=2.0, adf_stat_1m=0.0)
    assert _law(closed, "market_volatility_obs") == 0
    assert _law(closed, "market_spread_obs") == 0
    assert _law(closed, "market_stationarity_obs") == 0


def _gates_open():
    s = np.zeros(12, dtype=np.float32)
    s[9] = s[10] = s[11] = 1  # volatility, spread, stationarity favorable; no directional law
    return s


def test_mask_position_legality_and_hold_always_legal():
    """The SOW §2.3 acceptance: -1e9 on every forbidden action per position state."""
    s = _gates_open()
    m = build_direction_mask(s, position=0, n_open=0)          # FLAT
    assert m[HOLD] == 0 and m[OPEN_LONG] == 0 and m[OPEN_SHORT] == 0
    assert m[CLOSE] < -1e8                                     # nothing to close
    m = build_direction_mask(s, position=1, n_open=1)          # LONG
    assert m[OPEN_SHORT] < -1e8 and m[CLOSE] == 0 and m[HOLD] == 0
    m = build_direction_mask(s, position=-1, n_open=1)         # SHORT
    assert m[OPEN_LONG] < -1e8 and m[CLOSE] == 0
    m = build_direction_mask(s, position=1, n_open=5)          # slots full
    assert m[OPEN_LONG] < -1e8 and m[OPEN_SHORT] < -1e8
    m = build_direction_mask(s, position=0, n_open=0)
    assert m[CLOSE] < -1e8                                     # 0 open -> CLOSE masked


def test_mask_live_buy_law_bans_shorts():
    s = _gates_open(); s[_LIDX["law_super_trend_bb"]] = 1
    m = build_direction_mask(s, position=0, n_open=0, mode=MODE_LIVE)
    assert m[OPEN_SHORT] < -1e8 and m[OPEN_LONG] == 0 and m[HOLD] == 0


def test_phase_gated_stationarity_blocks_opens_only_when_constrained():
    # 2026-06-18: the 3 market-condition signals are observation-only by default (PHASE_FREE);
    # only stationarity re-enforces, and only in PHASE_CONSTRAINED.
    import quantra.runtime.config as _cfg
    s = _gates_open(); s[11] = 0  # market_stationarity_obs = 0 (non-stationary)
    # PHASE_FREE (default): observation-only -> the signal does NOT block opens.
    m_free = build_direction_mask(s, position=0, n_open=0, mode=MODE_LIVE)
    assert m_free[OPEN_LONG] == 0 and m_free[OPEN_SHORT] == 0 and m_free[HOLD] == 0
    # PHASE_CONSTRAINED: stationarity re-enforces -> opens blocked when non-stationary.
    _orig = _cfg.TRAINING_PHASE
    try:
        _cfg.TRAINING_PHASE = _cfg.PHASE_CONSTRAINED
        m_con = build_direction_mask(s, position=0, n_open=0, mode=MODE_LIVE)
        assert m_con[OPEN_LONG] < -1e8 and m_con[OPEN_SHORT] < -1e8 and m_con[HOLD] == 0
    finally:
        _cfg.TRAINING_PHASE = _orig


def test_mask_school_permits_only_required_direction():
    s = _gates_open(); s[_LIDX["law_trend_bb"]] = 1
    m = build_direction_mask(s, position=0, n_open=0, mode=MODE_SCHOOL,
                             required_laws=["law_trend_bb"])
    assert m[OPEN_LONG] == 0 and m[OPEN_SHORT] < -1e8       # buy permission only
    m2 = build_direction_mask(_gates_open(), position=0, n_open=0, mode=MODE_SCHOOL,
                              required_laws=["law_trend_bb"])
    assert m2[OPEN_LONG] < -1e8 and m2[OPEN_SHORT] < -1e8   # law inactive -> no permission
    assert m2[HOLD] == 0


def test_pointer_mask_targets_occupied_slots_only():
    pm = build_pointer_mask([1, 0, 1, 0, 0])
    assert pm[0] == 0 and pm[2] == 0
    assert pm[1] < -1e8 and pm[3] < -1e8 and pm[4] < -1e8


def test_lawmask_wrapper_end_to_end():
    row = _feat_row(atr_dev_1m=0.1, atr_dev_30m=0.1, spread_range_ratio_1m=0.3, adf_stat_1m=-3.5,
                    boll_bb20_up_5m=0.5, boll_bb200_up_5m=0.5,
                    boll_bb20_up_30m=0.5, boll_bb200_up_30m=0.5)
    res = LawMask(mode=MODE_LIVE).step(row, position=0, occupied=[0, 0, 0, 0, 0])
    assert res.opens_allowed is True
    assert res.direction_mask[OPEN_SHORT] < -1e8           # buy super-trend bans shorts
    assert res.direction_mask[OPEN_LONG] == 0


# =============================================================================
# SECTION TW — TRAINING WHEELS (operator counter-trend OPEN blocks, 2026-06-15)
# FTMO link: these semi-permanent masks forbid opening AGAINST a strong 30m+4H trend
# (CCI 5/15 vs SMA20-sh0; price vs BB10/100 dev0.5). Fewer breach-bound counter-trend
# opens => fewer wasted episodes => faster convergence to a consistently-passing brain.
# They are isolated from the locked 9 laws and removable via config.TRAINING_WHEELS.
# =============================================================================
def _wheel_df(n, seed):
    """A long 1m frame with a slow oscillation over a random walk — long enough to warm
    the 4H bands and varied enough that CCI diverges from its SMA in both directions, so
    the wheel flags actually fire (the slow 4H BB100 needs ~24k 1m bars to come alive)."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1min")
    rng = np.random.RandomState(seed)
    steps = rng.randn(n) * 0.05 + np.sin(np.arange(n) / 3000.0) * 0.04
    close = 100.0 + np.cumsum(steps)
    return pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1,
                         "close": close, "tick_volume": np.full(n, 100.0),
                         "spread": np.full(n, 10.0)}, index=idx)


def test_wheel_ingredients_and_flags_present_and_bounded():
    """The wheel ingredients + the two block flags exist, are finite, and ∈ {-1,0,1}."""
    mm = build_market_matrix(_wheel_df(9000, seed=3))
    for tf in ("30m", "4H"):
        for p in (5, 15):
            assert f"tw_cci{p}_{tf}" in PRECOMPUTED_NAMES and f"tw_cci{p}_sma_{tf}" in PRECOMPUTED_NAMES
        for p in (10, 100):
            assert f"tw_bb{p}_up_{tf}" in PRECOMPUTED_NAMES and f"tw_bb{p}_lo_{tf}" in PRECOMPUTED_NAMES
    for flag in ("tw_cci_block", "tw_bb_block"):
        col = mm.matrix[:, PRECOMPUTED_NAMES.index(flag)]
        assert np.isfinite(col).all()
        assert set(np.unique(col)).issubset({-1.0, 0.0, 1.0})


def test_wheel_flags_fire_in_both_directions():
    """Over a long oscillating series the CCI wheel flags BOTH uptrend (+1) and downtrend
    (-1), and the BB wheel flags an uptrend (+1) — the precomputed 'acts' are real signals,
    not dead constants. (Needs ~24k bars so the 4H BB100 warms.)"""
    mm = build_market_matrix(_wheel_df(26000, seed=1))
    cci = mm.matrix[mm.valid_from:, PRECOMPUTED_NAMES.index("tw_cci_block")]
    bb = mm.matrix[mm.valid_from:, PRECOMPUTED_NAMES.index("tw_bb_block")]
    assert (cci == 1).sum() > 0 and (cci == -1).sum() > 0, "CCI wheel never fired both ways"
    assert (bb == 1).sum() > 0, "BB wheel never flagged an uptrend breakout"


def test_wheel_mask_blocks_counter_trend_open_each_direction():
    """Uptrend flag bans OPEN_SHORT; downtrend bans OPEN_LONG; CCI/BB ban independently."""
    s = _gates_open()                                   # gates open, no directional law
    up_cci = build_direction_mask(s, 0, 0, training_wheels=True, wheel_states=np.array([1.0, 0.0]))
    assert up_cci[OPEN_SHORT] < -1e8 and up_cci[OPEN_LONG] == 0 and up_cci[HOLD] == 0
    down_bb = build_direction_mask(s, 0, 0, training_wheels=True, wheel_states=np.array([0.0, -1.0]))
    assert down_bb[OPEN_LONG] < -1e8 and down_bb[OPEN_SHORT] == 0
    # independent: only the BB wheel up -> only shorts banned (CCI flag neutral)
    bb_only = build_direction_mask(s, 0, 0, training_wheels=True, wheel_states=np.array([0.0, 1.0]))
    assert bb_only[OPEN_SHORT] < -1e8 and bb_only[OPEN_LONG] == 0


def test_wheels_off_is_a_noop_and_hold_always_legal():
    s = _gates_open()
    on = build_direction_mask(s, 0, 0, training_wheels=True, wheel_states=np.array([1.0, -1.0]))
    off = build_direction_mask(s, 0, 0, training_wheels=False, wheel_states=np.array([1.0, -1.0]))
    assert off[OPEN_LONG] == 0 and off[OPEN_SHORT] == 0   # disabled -> no bans
    assert on[HOLD] == 0 and off[HOLD] == 0               # HOLD always legal either way


def test_wheels_only_remove_never_reopen_a_law_ban():
    """Applied last + additive: a law-banned direction stays banned; wheels can't re-open."""
    s = _gates_open(); s[_LIDX["law_super_trend_bb"]] = -1   # sell law -> bans OPEN_LONG
    # a -1 (downtrend) BB wheel also bans OPEN_LONG; a +1 CCI wheel would ban OPEN_SHORT.
    m = build_direction_mask(s, 0, 0, mode=MODE_LIVE, training_wheels=True,
                             wheel_states=np.array([1.0, -1.0]))
    assert m[OPEN_LONG] < -1e8 and m[OPEN_SHORT] < -1e8 and m[HOLD] == 0


def test_env_direction_mask_applies_wheels_when_enabled():
    """End-to-end through the env: the precomputed tw flags drive the live mask."""
    row = _feat_row(atr_dev_1m=0.1, atr_dev_30m=0.1, spread_range_ratio_1m=0.3,
                    adf_stat_1m=-3.5, tw_cci_block=1.0)     # gates open + CCI uptrend wheel
    T = 3
    mat = np.tile(row, (T, 1)).astype(np.float32)
    sd = SymbolData(mat, close=np.full(T, 100.0), atr=np.full(T, 1.0),
                    spread=np.full(T, 10.0), valid_from=0)
    on = TradingEnv({"EURUSD": sd}, training_wheels=True)
    assert on.direction_mask("EURUSD")[OPEN_SHORT] < -1e8   # uptrend wheel bans shorts
    assert on.direction_mask("EURUSD")[OPEN_LONG] == 0
    off = TradingEnv({"EURUSD": sd}, training_wheels=False)
    assert off.direction_mask("EURUSD")[OPEN_SHORT] == 0    # wheels off -> short legal


# =============================================================================
# SECTION G — ENV + RISKMANAGER + COSTLAYER (M4)
# FTMO link: this is the challenge physics. The B5 invariant — 4 symbols can't
# collectively overshoot the daily-risk buffer in one bar — plus real costs and the
# hard wall are the mechanical reasons the bot can learn to PASS rather than just
# look profitable. No risk/sizing rule may ever overshoot.
# =============================================================================
def _open_gate_matrix(T):
    """A (T, PRECOMPUTED_DIM) feature matrix with all 3 gates OPEN, no directional
    law active — so opens are legal in both directions and we can exercise the env."""
    m = np.zeros((T, PRECOMPUTED_DIM), dtype=np.float32)
    for name, val in [("atr_dev_1m", 0.1), ("atr_dev_30m", 0.1),
                      ("spread_range_ratio_1m", 0.3), ("adf_stat_1m", -3.5)]:
        m[:, PRECOMPUTED_NAMES.index(name)] = val
    return m


def _sym(T=60, price=1.20, atr=0.001, spread=2e-5, drift=0.0):
    close = (price + drift * np.arange(T)).astype(float)
    return SymbolData(matrix=_open_gate_matrix(T), close=close,
                      atr=np.full(T, atr), spread=np.full(T, spread), valid_from=0)


# ---- CostLayer ----
def test_cost_forex_pays_commission_metals_indices_dont():
    cl = CostLayer()
    # forex: close pays the $5 RT/lot commission
    assert cl.close_cost("EURUSD", 2.0).commission == 10.0
    # metals + indices: no per-trade commission
    assert cl.close_cost("XAUUSD", 2.0).commission == 0.0
    assert cl.close_cost("US30", 2.0).commission == 0.0
    # open cost carries spread + slippage but NO commission (charged once, on close)
    oc = cl.open_cost("EURUSD", 1.0, spread_price=2e-5)
    assert oc.commission == 0.0 and oc.spread > 0 and oc.slippage > 0


def test_cost_spread_scales_with_lots_and_contract():
    cl = CostLayer()
    one = cl.open_cost("EURUSD", 1.0, 2e-5).spread
    two = cl.open_cost("EURUSD", 2.0, 2e-5).spread
    assert abs(two - 2 * one) < 1e-9          # linear in lots
    assert abs(one - 2e-5 * 100_000 * 1.0) < 1e-9  # spread_price * contract * lots


# ---- RiskManager (no overshoot) ----
def test_riskmanager_never_exceeds_available_budget():
    rm = RiskManager(account_size=10_000)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        raw = float(rng.random())
        atr = float(rng.uniform(1e-4, 5e-3))
        budget = float(rng.uniform(0, 500))
        sr = rm.size("EURUSD", raw, atr, budget)
        assert sr.committed_risk <= budget + 1e-6   # THE invariant, fuzzed
        if sr.feasible:
            assert sr.lots >= rm.cfg.min_lot


def test_riskmanager_refuses_when_buffer_too_small():
    rm = RiskManager(account_size=10_000)
    sr = rm.size("EURUSD", raw_size=1.0, atr_price=1e-3, available_budget=0.0)
    assert not sr.feasible and sr.lots == 0.0


# ---- B5: 4 symbols cannot collectively overshoot in one bar ----
def test_b5_four_symbols_cannot_overshoot_buffer_in_one_bar():
    data = {s: _sym(atr=0.001) for s in ("EURUSD", "XAUUSD", "GBPUSD", "US30")}
    # Large per-trade cap so each symbol WANTS more than its share -> threading must
    # be what prevents overshoot, not the per-trade cap.
    env = TradingEnv(data, risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    buffer0 = env.account.remaining_buffer
    # all 4 symbols open max-size long in the same bar
    for _ in range(4):
        env.step((OPEN_LONG, 1.0, 0))
    committed = env._committed_risk()
    assert committed <= buffer0 + 1e-6          # collective risk never exceeds buffer
    assert committed > 0                          # at least one symbol got filled


def test_true_sequential_buffer_visible_to_later_symbols():
    data = {s: _sym(atr=0.001) for s in ("EURUSD", "XAUUSD", "GBPUSD", "US30")}
    env = TradingEnv(data, risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    buf_before = env.account.remaining_buffer - env._committed_risk()
    env.step((OPEN_LONG, 1.0, 0))               # symbol 0 opens
    buf_after = env.account.remaining_buffer - env._committed_risk()
    assert buf_after < buf_before               # symbol 1 sees a reduced buffer


# ---- slot mechanics ----
def test_slot_open_fills_next_free_and_masks_at_five():
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4)})
    for _ in range(7):                           # try to open 7 (only 5 slots)
        env.step((OPEN_LONG, 0.3, 0))
    assert env._n_open("EURUSD") == 5            # OPEN masked once full


def test_close_routes_to_pointer_slot():
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4)})
    for _ in range(3):
        env.step((OPEN_LONG, 0.3, 0))
    assert env._n_open("EURUSD") == 3
    env.step((CLOSE, 0.0, 1))                     # close slot index 1
    assert not env.slots["EURUSD"][1].occupied and env._n_open("EURUSD") == 2


# ---- costs reduce equity ----
def test_round_trip_costs_reduce_equity():
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4, drift=0.0)})  # flat price
    eq0 = env.account.equity
    env.step((OPEN_LONG, 0.5, 0))
    env.step((CLOSE, 0.0, 0))
    # flat price -> no PnL, only costs -> equity strictly lower (no costless world)
    assert env.account.equity < eq0


# ---- hard wall ----
def test_hard_wall_force_flattens_and_locks_out_day():
    # C10: price crashes after entry -> equity hits the 4% wall -> breach FORCE-FLATTENS all +
    # LOCKS OUT the rest of the day. The breach no longer ENDS the episode (without a calendar-day
    # boundary the account just stays flat + locked until end-of-data).
    T = 12
    close = np.concatenate([np.full(3, 1.20), np.linspace(1.20, 1.10, T - 3)]).astype(float)
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), close, np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0)}
    env = TradingEnv(data, risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    env.step((OPEN_LONG, 1.0, 0))
    breached_at = None
    done = False
    for i in range(T):
        if done:
            break
        _, _, done, _info = env.step((HOLD, 0.0, 0))
        if env.account.breached and breached_at is None:
            breached_at = i
            assert env.account.locked_out            # the DAY is locked out...
            assert not done                          # ...but the breach did NOT end the episode (C10)
            assert env._n_open("EURUSD") == 0        # force-flattened on breach
    assert breached_at is not None                   # the wall was hit
    assert env.account.breached                      # no day boundary here -> stays breached


# ---- observation shape + law/mask visibility ----
def test_env_observation_shape_and_law_block_populated():
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4)})
    obs = env.reset()
    assert obs.shape == (STATE_DIM,)             # full 179
    law_block = obs[SCHEMA.block_spans["law"][0]:SCHEMA.block_spans["law"][1]]
    assert law_block.shape == (12,)
    acct_start = SCHEMA.block_spans["account"][0]
    assert abs(obs[acct_start] - 1.0) < 1e-5    # equity_norm == 1.0 at reset


def test_env_enforces_mask_coerces_forbidden_action():
    # flat position: CLOSE is illegal (nothing to close) -> env coerces to HOLD
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4)})
    _, _, _, info = env.step((CLOSE, 0.0, 0))
    assert info["coerced"] is True and info["executed"] == "HOLD"


# =============================================================================
# SECTION H — PPOAGENT + ROLLOUTBUFFER + PPO LOSS (M5)
# FTMO link: the brain that turns the 179-dim observation into a legal, sized,
# slot-aware action. Masks are applied to the LOGITS here, so the policy can never
# sample a breach-bound action; the summed 3-head log-prob is the PPO loss contract.
# =============================================================================
def test_actor_critic_matches_locked_architecture():
    net = ActorCritic(state_dim=STATE_DIM)
    d, s, p, v = net(torch.randn(4, STATE_DIM))
    assert d.shape == (4, 4)          # {HOLD, OPEN_LONG, OPEN_SHORT, CLOSE}
    assert s.shape == (4, 2)          # Beta alpha/beta params
    assert p.shape == (4, 5)          # 5 pointer slots
    assert v.shape == (4,)            # V(s)
    linears = [m for m in net.trunk if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3 and all(l.out_features == 256 for l in linears)


def test_agent_act_respects_direction_mask():
    """A forbidden direction must never be sampled (masks are applied to logits)."""
    agent = PPOAgent(state_dim=STATE_DIM)
    n = 1000
    obs = torch.randn(n, STATE_DIM)
    dm = torch.zeros(n, 4)
    dm[:, OPEN_SHORT] = -1e9           # forbid OPEN_SHORT everywhere
    pm = torch.zeros(n, 5)
    step = agent.act(obs, dm, pm)
    assert (step.a_direction != OPEN_SHORT).all()
    assert step.a_size.min() >= 0.0 and step.a_size.max() <= 1.0
    assert step.a_pointer.min() >= 0 and step.a_pointer.max() <= 4


def test_summed_log_prob_gates_size_on_open_and_pointer_on_close():
    """Size head contributes only on OPEN, pointer only on CLOSE (the lock)."""
    agent = PPOAgent(state_dim=STATE_DIM)
    obs = torch.randn(8, STATE_DIM)
    dm, pm = torch.zeros(8, 4), torch.zeros(8, 5)
    a_ptr = torch.zeros(8, dtype=torch.long)
    size_a, size_b = torch.full((8,), 0.3), torch.full((8,), 0.7)

    hold = torch.full((8,), HOLD, dtype=torch.long)
    lp_a, _, _ = agent.evaluate_actions(obs, dm, pm, hold, size_a, a_ptr)
    lp_b, _, _ = agent.evaluate_actions(obs, dm, pm, hold, size_b, a_ptr)
    assert torch.allclose(lp_a, lp_b)   # size gated OFF for HOLD -> logp unchanged

    opn = torch.full((8,), OPEN_LONG, dtype=torch.long)
    lp_a2, _, _ = agent.evaluate_actions(obs, dm, pm, opn, size_a, a_ptr)
    lp_b2, _, _ = agent.evaluate_actions(obs, dm, pm, opn, size_b, a_ptr)
    assert not torch.allclose(lp_a2, lp_b2)  # size gated ON for OPEN -> logp changes


def test_act_deterministic_is_argmax_and_beta_mean():
    agent = PPOAgent(state_dim=STATE_DIM)
    obs = torch.randn(5, STATE_DIM)
    dm = torch.zeros(5, 4); dm[:, OPEN_LONG] = -1e9   # forbid OPEN_LONG
    pm = torch.zeros(5, 5)
    a_dir, a_size, a_ptr, value = agent.act_deterministic(obs, dm, pm)
    assert (a_dir != OPEN_LONG).all()                  # argmax respects the mask
    assert (a_size >= 0).all() and (a_size <= 1).all()  # Beta mean in [0,1]


def test_rollout_buffer_stores_ten_fields_and_summed_logp():
    assert len(FIELDS) == 10                           # SOW §2.9 ten fields
    agent = PPOAgent(state_dim=STATE_DIM)
    buf = RolloutBuffer(capacity=3, state_dim=STATE_DIM)
    obs = torch.randn(STATE_DIM)
    dm, pm = torch.zeros(4), torch.zeros(5)
    step = agent.act(obs, dm, pm)
    buf.add(obs, step.a_direction.item(), step.a_size.item(), step.a_pointer.item(),
            reward=1.0, next_obs=obs, logp_old=step.log_prob.item(),
            value_old=step.value.item(), done=0.0, dir_mask=dm, ptr_mask=pm)
    assert len(buf) == 1
    d = buf.get()
    assert d["logp_old"].shape == (1,) and d["dir_mask"].shape == (1, 4)
    assert abs(float(d["logp_old"][0]) - float(step.log_prob)) < 1e-5
    buf.clear()
    assert len(buf) == 0


def test_rollout_buffer_no_replay_full_raises():
    buf = RolloutBuffer(capacity=1, state_dim=STATE_DIM)
    z4, z5, o = torch.zeros(4), torch.zeros(5), torch.zeros(STATE_DIM)
    buf.add(o, 0, 0.5, 0, 0.0, o, -1.0, 0.0, 0.0, z4, z5)
    with pytest.raises(RuntimeError):
        buf.add(o, 0, 0.5, 0, 0.0, o, -1.0, 0.0, 0.0, z4, z5)  # on-policy: no overflow


def _collect(agent, n=24):
    buf = RolloutBuffer(n, STATE_DIM)
    for _ in range(n):
        obs = torch.randn(STATE_DIM)
        dm, pm = torch.zeros(4), torch.zeros(5)
        st = agent.act(obs, dm, pm)
        buf.add(obs, st.a_direction.item(), st.a_size.item(), st.a_pointer.item(),
                reward=float(np.random.randn()), next_obs=torch.randn(STATE_DIM),
                logp_old=st.log_prob.item(), value_old=st.value.item(),
                done=0.0, dir_mask=dm, ptr_mask=pm)
    return buf


def test_ppo_loss_matches_old_logp_right_after_collection():
    """Evaluating the same net on the just-collected actions -> ratio≈1, KL≈0."""
    torch.manual_seed(0)
    agent = PPOAgent(state_dim=STATE_DIM)
    batch = _collect(agent).get()
    adv = torch.randn(len(batch["obs"]))
    ret = torch.randn(len(batch["obs"]))
    loss, diag = ppo_loss(agent, batch, adv, ret)
    assert abs(diag["ratio_mean"] - 1.0) < 1e-4
    assert abs(diag["approx_kl"]) < 1e-4
    assert 0.0 <= diag["clip_frac"] <= 1.0


def test_ppo_loss_backprops_into_the_trunk():
    agent = PPOAgent(state_dim=STATE_DIM)
    batch = _collect(agent).get()
    n = len(batch["obs"])
    loss, _ = ppo_loss(agent, batch, torch.randn(n), torch.randn(n))
    agent.net.zero_grad()
    loss.backward()
    grads = [p.grad for p in agent.net.trunk.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)   # learning signal reaches the trunk


def test_agent_drives_env_end_to_end():
    """Integration: the agent acts on real env observations + masks without crashing."""
    env = TradingEnv({"EURUSD": _sym(T=30, atr=1e-4)})
    agent = PPOAgent(state_dim=STATE_DIM)
    obs = env.reset()
    for _ in range(10):
        dm = torch.tensor(env.direction_mask("EURUSD"))
        sym = env.symbols[env.cursor]
        occ = [s.occupied for s in env.slots[sym]]
        pm = torch.tensor(build_pointer_mask(occ))
        a_dir, a_size, a_ptr, _ = agent.act_deterministic(obs, dm, pm)
        obs, reward, done, info = env.step((int(a_dir), float(a_size), int(a_ptr)))
        assert not info["coerced"]          # agent only ever picked legal actions
        if done:
            break


# =============================================================================
# SECTION I — REWARDENGINE (L0-L6 + QUAD) + E8 DOMINANCE (M6)
# FTMO link: the objective. Layer 0 (net PnL) MUST dominate, or the bot games a
# shaper while losing the trading game. E8 is the invariant that forbids that.
# =============================================================================
from quantra.learning_system.reward_engine import (  # noqa: E402
    DailyMetrics,
    QuadBonus,
    RewardContext,
    RewardEngine,
)


def test_e8_layer0_dominates_over_1000_rollouts():
    """Over 1000 random rollouts, cumulative |L0| exceeds every shaping layer's."""
    eng = RewardEngine()
    rng = np.random.default_rng(7)
    fails = 0
    for _ in range(1000):
        sums = {k: 0.0 for k in ("L0", "L1", "L2", "L3", "L4")}
        for _ in range(256):                       # a rollout of 256 steps
            ctx = RewardContext(
                net_pnl_delta=float(rng.normal(0, 2e-3)),   # realistic per-step PnL
                in_position=bool(rng.random() < 0.6),
                momentum_aligned=bool(rng.random() < 0.5),
                stagnation=bool(rng.random() < 0.3),
                drawdown_pct=float(rng.uniform(0, 4.0)),
                day_progress=float(rng.uniform(-1, 1)),
                breach_risk=bool(rng.random() < 0.2),
            )
            d = eng.decompose(ctx)
            for k in sums:
                sums[k] += abs(d[k])
        l0 = sums["L0"]
        if any(sums[k] > l0 for k in ("L1", "L2", "L3", "L4")):
            fails += 1
    assert fails == 0, f"E8 violated in {fails}/1000 rollouts (a shaping layer outweighed L0)"


def test_pain_zone_is_zero_below_threshold_and_monotonic():
    eng = RewardEngine()
    assert eng._pain(3.0) == 0.0                    # below 3.5% start -> no pain
    p35, p38, p40 = eng._pain(3.5), eng._pain(3.8), eng._pain(4.0)
    assert 0.0 <= p35 < p38 < p40                   # exponential, increasing to the wall
    assert abs(p40 - 1.0) < 1e-9


def test_reward_layer0_passthrough_dominates_a_single_step():
    eng = RewardEngine()
    # a meaningful L0 with all shaping on -> total is dominated by L0's sign/scale
    d = eng.decompose(RewardContext(net_pnl_delta=0.01, in_position=True,
                                    momentum_aligned=True, day_progress=1.0))
    assert d["L0"] == 0.01
    assert abs(d["shaped"]) < abs(d["L0"])          # shaping is a whisper


def test_quad_bonus_respects_95pct_ceiling():
    """Even with a huge flow streak, the QUAD bonus stays < 1x day PnL (E8-safe)."""
    q = QuadBonus(enabled=True)
    bonus = 0.0
    for _ in range(20):                              # build a long flow streak
        bonus = q.end_of_day(DailyMetrics(
            drawdown_efficiency=float(_ + 1), law_productivity=float(_ + 1),
            target_velocity=float(_ + 1), td_stability=float(-(_ + 1)),
            day_pnl=100.0, passed=True))
    assert bonus <= 0.95 * 100.0 + 1e-9              # ceiling holds


def test_quad_bonus_zero_on_non_pass_day():
    q = QuadBonus(enabled=True)
    for i in range(10):
        q.end_of_day(DailyMetrics(i, i, i, -i, day_pnl=100.0, passed=True))
    assert q.end_of_day(DailyMetrics(1, 1, 1, -1, day_pnl=100.0, passed=False)) == 0.0
    assert q.flow_streak == 0                        # streak resets on a failed day


def test_env_uses_layered_reward_engine():
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4)})
    assert isinstance(env.reward_engine, RewardEngine)
    _, reward, _, _ = env.step((HOLD, 0.0, 0))
    assert isinstance(reward, float)                 # layered reward returned, no crash


# =============================================================================
# SECTION J — CURRICULUM + TWO-PHASE EPISODE RULE (M7)
# FTMO link: the two-phase rule banks the +2.5% behind a tight 1% wall (challenge-
# style stopping); the curriculum teaches structure-first inside law context. Both
# shape the bot toward consistent passing rather than reckless profit-chasing.
# =============================================================================
from quantra.ftmo_passing.challenge_state import ChallengeState as _CS  # noqa: E402
from quantra.learning_system.curriculum_manager import (  # noqa: E402
    DEFAULT_STAGES,
    CurriculumManager,
    Stage,
)
from quantra.market_pipeline.law_mask_engine.engine import MODE_LIVE as _LIVE  # noqa: E402
from quantra.market_pipeline.law_mask_engine.engine import MODE_SCHOOL as _SCHOOL  # noqa: E402


def test_two_phase_wall_tightens_after_target():
    cs = _CS(account_size=10_000, challenge=ChallengeConfig())
    assert cs.phase == "A"
    # Phase A wall = peak − 4% of account
    assert abs(cs.wall_equity - (10_000 - 0.04 * 10_000)) < 1e-6
    cs.enter_phase_b()
    assert cs.phase == "B"
    # Phase B wall = re-anchored peak − 1%
    assert abs(cs.wall_equity - (cs.peak_equity - 0.01 * 10_000)) < 1e-6


def test_env_auto_flats_and_enters_phase_b_at_target():
    """A +2.5% day -> auto-flat ALL + switch to Phase B (episode continues)."""
    # price climbs so a long hits the +2.5% daily target
    T = 40
    close = np.linspace(1.20, 1.26, T).astype(float)   # ~5% climb
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), close, np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0)}
    env = TradingEnv(data, risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    env.step((OPEN_LONG, 1.0, 0))
    for _ in range(T - 2):
        if env.done:
            break
        env.step((HOLD, 0.0, 0))
        if env.account.phase == "B":
            break
    assert env.account.phase == "B"                 # target banked, Phase B engaged
    assert env._n_open("EURUSD") == 0               # all positions auto-flatted
    assert not env.account.breached


def test_curriculum_stages_and_law_school_config():
    cm = CurriculumManager()
    assert [s.name for s in cm.stages] == ["trend", "reversion", "stationarity_atr"]
    cfg = cm.law_school_config()
    assert cfg["mask_mode"] == _SCHOOL and cfg["required_laws"]   # school mode w/ context
    # trend stage permits trend/super-trend laws, not pullback
    assert any("trend" in n for n in cfg["required_laws"])
    assert all("pullback" not in n for n in cfg["required_laws"])


def test_curriculum_feature_mask_zeros_1m_timing_only():
    cm = CurriculumManager()
    mask = cm.feature_mask()
    assert mask.shape == (STATE_DIM,)
    assert mask[PRECOMPUTED_NAMES.index("candle_return_1m")] == 0.0   # 1m timing masked
    assert mask[PRECOMPUTED_NAMES.index("z10_1m")] == 0.0
    assert mask[PRECOMPUTED_NAMES.index("ssma_high_dist_1m")] == 1.0  # law ingredient kept
    assert mask[PRECOMPUTED_NAMES.index("atr_dev_1m")] == 1.0         # gate ingredient kept


def test_curriculum_graduates_to_live_mode():
    cm = CurriculumManager()
    cm.graduate(); cm.graduate(); cm.graduate()      # past the 3 stages
    assert cm.graduated and cm.current_stage() is None
    assert cm.law_school_config()["mask_mode"] == _LIVE   # live-ban mode after graduation
    assert (cm.feature_mask() == 1.0).all()              # no 1m masking once graduated


# =============================================================================
# SECTION K — TRAINER + GAE + AGGRESSION SCHEDULER + G8 (M8)
# FTMO link: the PPO loop that turns physics + masks + reward into a brain that
# passes. gamma/lambda locked (patience); the scheduler keeps exploration high while
# the bot leaves premium setups on the table (G8), cooling as it captures them.
# =============================================================================
from quantra.learning_system.trainer import (  # noqa: E402
    GAMMA,
    LAMBDA,
    AggressionScheduler,
    TrainConfig,
    Trainer,
    compute_gae,
    missed_opportunity,
)
from quantra.learning_system.trainer.scheduler import AggressionRanges  # noqa: E402


def test_gae_locked_gamma_lambda_and_shapes():
    assert GAMMA == 0.997 and LAMBDA == 0.97          # 🔴 patience locked
    r = torch.ones(5)
    v = torch.zeros(5)
    d = torch.tensor([0., 0., 0., 0., 1.])
    adv, ret = compute_gae(r, v, d, last_value=0.0)
    assert adv.shape == (5,) and ret.shape == (5,)
    # with V=0 and the last step terminal, the last advantage is just its reward
    assert abs(float(adv[-1]) - 1.0) < 1e-6
    # earlier advantages accumulate discounted future reward -> strictly larger
    assert float(adv[0]) > float(adv[-1])


def test_aggression_scheduler_dials_stay_in_locked_ranges():
    rng = AggressionRanges()
    sch = AggressionScheduler(rng)
    for mr in (0.0, 0.5, 1.0):
        sch.aggression = mr
        v = sch.values()
        assert rng.entropy[0] <= v.entropy_coef <= rng.entropy[1]
        assert rng.clip[0] <= v.clip_eps <= rng.clip[1]
        assert rng.lr[0] <= v.lr <= rng.lr[1]
        assert rng.epochs[0] <= v.epochs <= rng.epochs[1]


def test_aggression_cools_as_misses_fall():
    sch = AggressionScheduler(start=1.0)
    for _ in range(50):
        sch.update(miss_rate=0.0)                      # bot now captures everything
    assert sch.aggression < 0.1                        # aggression cools toward low end


def test_g8_missed_opportunity_requires_multi_tf_agreement_flat_and_move():
    row = _feat_row(ssma_align_5m=1, ssma_align_30m=1, ssma_align_4H=1)
    assert missed_opportunity(row, was_flat=True, realized_move_atr=2.0) is True   # all align + ran 2 ATR
    assert missed_opportunity(row, was_flat=True, realized_move_atr=0.5) is False  # move < 1.5 ATR
    assert missed_opportunity(row, was_flat=False, realized_move_atr=2.0) is False  # wasn't flat
    disagree = _feat_row(ssma_align_5m=1, ssma_align_30m=-1, ssma_align_4H=1)
    assert missed_opportunity(disagree, was_flat=True, realized_move_atr=2.0) is False


def test_trainer_runs_and_checkpoints():
    """Integration: a short PPO run collects, updates, schedules, and saves a brain."""
    data = {"EURUSD": _sym(T=400, atr=1e-4, drift=1e-5)}
    env = TradingEnv(data, risk_cfg=RiskConfig(max_per_trade_risk_frac=0.2))
    trainer = Trainer(env, train_cfg=TrainConfig(rollout_size=64, minibatch=16, seed=0))
    p0 = [x.detach().clone() for x in trainer.agent.net.parameters()]
    hist = trainer.train(n_updates=2)
    assert len(hist) == 2
    assert "approx_kl" in hist[-1] and "miss_rate" in hist[-1]
    # the update actually changed the weights (a learning step happened)
    changed = any((a - b).abs().sum() > 0 for a, b in zip(trainer.agent.net.parameters(), p0))
    assert changed
    path = trainer.checkpoint("test_brain")
    assert path.exists()


# =============================================================================
# SECTION L — TELEMETRYLOGGER (versioned data contract, round-trip) (M9)
# FTMO link: no telemetry -> no diagnosis. A breach/pass-day must be fully
# reconstructable so the Risk Doctor can find the cause before it costs a challenge.
# =============================================================================
from quantra.diagnostics.telemetry_logger import (  # noqa: E402
    SCHEMA_VERSION,
    StepPacket,
    TelemetryLogger,
)


def _demo_packet(ts=0):
    return StepPacket(
        run_id="r", seed=0, window_id="w0", episode_id=1, timestep=ts, symbol="EURUSD",
        timestamp="2021-01-04T00:00:00", bar_index=ts,
        observation=list(np.random.randn(STATE_DIM)),
        law_states=[0.0] * 12, enforcement_mode="live", legal_actions=[1, 1, 1, 0],
        pre_mask_logits=[0.1, 0.2, 0.3, 0.4], post_mask_logits=[0.1, 0.2, -1e9, 0.4],
        action_probs=[0.4, 0.3, 0.0, 0.3], chosen_action=0, pointer_output=None,
        raw_size=0.5, feasible_size=0.66, value=0.01,
        hidden_summary=[0.1, -0.2, 0.3],
        reward_decomposition={"L0": 0.01, "L1": 0.0, "L3": -0.001},
        quad_signals={"dd_eff": 1.0}, risk_context={"daily_dd": 0.5, "trailing_dd": 1.2},
        outcome={"next_bar_return": 0.0003},
    )


def test_telemetry_round_trip_preserves_every_field(tmp_path):
    log = TelemetryLogger("run_rt", out_dir=tmp_path)
    p = _demo_packet(3)
    log.log_step(p)
    log.log_trade({"trade_id": 1, "pnl": 12.3, "symbol": "EURUSD"})
    log.log_day({"day": 1, "passed": True, "day_pnl": 250.0})
    path = log.flush()

    recs = TelemetryLogger.load(path)
    header = recs[0]
    assert header["kind"] == "header" and header["schema_version"] == SCHEMA_VERSION
    assert "blocks" in header and len(header["feature_names"]) == STATE_DIM

    step = next(r for r in recs if r["kind"] == "step")
    # every contract field survives the round trip
    for fld in p.to_dict():
        assert fld in step
    assert step["chosen_action"] == 0 and step["enforcement_mode"] == "live"
    assert np.allclose(step["observation"], p.observation, atol=1e-6)
    assert step["reward_decomposition"]["L0"] == 0.01
    assert any(r["kind"] == "trade" for r in recs) and any(r["kind"] == "day" for r in recs)


def test_telemetry_header_carries_block_names_for_the_llm():
    """The LLM must be able to map any observation index to its feature block."""
    log = TelemetryLogger("run_hdr")
    recs = log._buf
    blocks = recs[0]["blocks"]
    assert set(blocks) == {"market", "market_raw", "law", "trade", "portfolio", "account"}
    assert "law_super_trend_bb" in blocks["law"]


# =============================================================================
# SECTION M — MLPINTERPRETER (the 7 required visuals) (M10)
# FTMO link: these visuals make "did the internal behaviour help the bot pass?"
# answerable - L0 dominance over time, regime separation, value near danger.
# =============================================================================
from quantra.diagnostics.mlp_interpreter import MLPInterpreter  # noqa: E402


def test_mlp_interpreter_produces_all_seven_visuals(tmp_path):
    log = TelemetryLogger("run_viz", out_dir=tmp_path)
    rng = np.random.default_rng(0)
    for t in range(40):
        p = _demo_packet(t)
        p.observation = list(rng.standard_normal(STATE_DIM))
        p.chosen_action = int(rng.integers(0, 4))
        p.value = float(rng.standard_normal())
        p.hidden_summary = list(rng.standard_normal(8))
        p.reward_decomposition = {"L0": float(rng.standard_normal() * 1e-2),
                                  "L1": 1e-4, "L3": -1e-4}
        p.risk_context = {"trailing_dd": float(abs(rng.standard_normal()))}
        log.log_step(p)
    recs = TelemetryLogger.load(log.flush())

    interp = MLPInterpreter(recs, out_dir=tmp_path / "viz")
    paths = interp.generate_all()
    assert set(paths) == {                          # all 7 required visuals
        "activation_trace", "hidden_state_projection", "action_value_timeline",
        "reward_layer_timeline", "correlation_heatmap", "failure_atlas", "pass_day_atlas",
    }
    for name, path in paths.items():
        assert path.exists() and path.stat().st_size > 0, f"{name} not written"


def test_failure_atlas_module_is_real(tmp_path):
    """The SOW-named failure_atlas module must be callable (not an empty shell)."""
    from quantra.diagnostics.failure_atlas import failure_atlas, pass_day_atlas
    log = TelemetryLogger("run_atlas", out_dir=tmp_path)
    for t in range(12):
        log.log_step(_demo_packet(t))
    recs = TelemetryLogger.load(log.flush())
    fa = failure_atlas(recs, out_dir=tmp_path / "fa")
    pa = pass_day_atlas(recs, out_dir=tmp_path / "fa")
    assert fa.exists() and pa.exists()


# =============================================================================
# SECTION N — LLMRISKDOCTOR (read-only, mandatory rulebook, 8-taxonomy) (M11)
# FTMO link: catches models that pass by luck or drift to the wall BEFORE they cost a
# challenge, with evidence-cited diagnoses - and can never touch execution.
# =============================================================================
from quantra.diagnostics.llm_risk_doctor import (  # noqa: E402
    TAXONOMY,
    UNCLASSIFIED,
    LLMRiskDoctor,
)


def test_risk_doctor_fails_loud_without_rulebook(tmp_path):
    with pytest.raises(FileNotFoundError):
        LLMRiskDoctor(rulebook=tmp_path / "missing_MLP_INTERPRETABILITY_LAYER.md")


def test_risk_doctor_is_read_only_no_write_method():
    doc = LLMRiskDoctor()                              # default rulebook exists in docs/
    assert doc.rulebook                                # it has READ the rulebook
    assert not hasattr(doc, "write")                  # NO write capability, by design
    assert not hasattr(doc, "modify") and not hasattr(doc, "execute")


def test_risk_doctor_classifies_reward_hijack_with_evidence():
    log = TelemetryLogger("run_doc")
    for t in range(30):
        p = _demo_packet(t)
        # a shaping layer (L1) dwarfs Layer 0 -> Reward Hijack
        p.reward_decomposition = {"L0": 1e-5, "L1": 0.5}
        p.value = float(np.sin(t)); p.outcome = {"next_bar_return": float(np.sin(t))}
        log.log_step(p)
    diag = LLMRiskDoctor().diagnose(log._buf)
    assert diag.classification == "Reward Hijack"
    assert diag.classification in TAXONOMY            # always one of the 8
    assert diag.evidence and "L0" in diag.evidence[0]  # cites the per-layer integral
    assert "DIAGNOSIS" in diag.render()               # follows the output template


def test_risk_doctor_never_invents_ninth_category():
    log = TelemetryLogger("run_clean")
    rng = np.random.default_rng(1)
    for t in range(40):
        p = _demo_packet(t)
        p.reward_decomposition = {"L0": float(rng.normal(0, 1e-2)), "L1": 1e-4}
        p.value = float(rng.normal()); p.outcome = {"next_bar_return": float(rng.normal())}
        p.hidden_summary = list(rng.normal(size=8))
        p.pre_mask_logits = [0.1, 0.2, 0.3, 0.4]; p.post_mask_logits = [0.1, 0.2, 0.3, 0.4]
        log.log_step(p)
    diag = LLMRiskDoctor().diagnose(log._buf)
    assert diag.classification in TAXONOMY or diag.classification == UNCLASSIFIED


# =============================================================================
# SECTION O — SCOREBOARD + WALK-FORWARD + PROMOTION GATE (M12)
# FTMO link: rank brains by PASS RATE (not PnL), validate on rolling out-of-sample
# windows x 7 seeds, and ship only robust, no-worse-breach improvements.
# =============================================================================
from quantra.ftmo_passing.validation import (  # noqa: E402
    PromotionGate,
    RunResult,
    Scoreboard,
    WalkForwardRunner,
    generate_windows,
)


def test_scoreboard_ranks_by_pass_rate_then_breach_not_pnl():
    # A: higher pass rate but huge PnL irrelevant; B: lower pass rate
    a = Scoreboard([RunResult(True, False, True, 0.02, pnl=1.0) for _ in range(3)]
                   + [RunResult(False, False, True, 0.02)])           # 75% pass
    b = Scoreboard([RunResult(True, False, True, 0.02, pnl=9999.0)]
                   + [RunResult(False, False, False, 0.03) for _ in range(3)])  # 25% pass, huge PnL
    assert a.pass_rate > b.pass_rate
    assert a.better_than(b)                       # PnL never rescues the lower passer
    # tie on pass rate -> fewer breaches wins
    c = Scoreboard([RunResult(True, True, True, 0.02), RunResult(True, False, True, 0.02)])
    d = Scoreboard([RunResult(True, False, True, 0.02), RunResult(True, False, True, 0.02)])
    assert d.better_than(c)                       # d has 0 breaches vs c's 1


def test_walk_forward_generates_12_2_1_windows():
    idx = pd.date_range("2021-01-01", "2023-06-30", freq="D")   # 2.5 years
    wins = generate_windows(idx)
    assert len(wins) >= 6                          # rolling monthly windows
    w0 = wins[0]
    # 12 months train, 2 months test
    assert (w0.train_end.year - w0.train_start.year) * 12 + (w0.train_end.month - w0.train_start.month) == 12
    assert (w0.test_end.year - w0.test_start.year) * 12 + (w0.test_end.month - w0.test_start.month) == 2
    # 1-month step between consecutive windows
    assert (wins[1].train_start - wins[0].train_start).days in (28, 29, 30, 31)


def test_walk_forward_runner_runs_all_windows_x_seeds():
    idx = pd.date_range("2021-01-01", "2022-09-30", freq="D")
    runner = WalkForwardRunner(n_seeds=7)
    calls = []

    def eval_fn(window, seed):
        calls.append(seed)
        return RunResult(passed=(seed < 4), breached=(seed == 6), target_hit=True, max_drawdown=0.02)

    sb, seed_pass = runner.run(idx, eval_fn)
    assert len(seed_pass) == 7
    assert sb.n == len(calls)                       # one result per (window, seed)
    assert sum(1 for c in seed_pass if c > 0) == 4  # seeds 0..3 passed every window


def test_promotion_gate_requires_3_seeds_improvement_no_worse_breach():
    gate = PromotionGate()
    base = Scoreboard([RunResult(True, False, True, 0.03) for _ in range(2)]
                      + [RunResult(False, True, False, 0.05)])        # 1 breach
    better = Scoreboard([RunResult(True, False, True, 0.02) for _ in range(3)])  # no breach, higher pass
    ok, _ = gate.promote(better, base, seed_pass_counts=[2, 1, 1, 0, 0, 0, 0])   # 3 seeds passed
    assert ok
    # too few seeds -> reject even if better
    no, reason = gate.promote(better, base, seed_pass_counts=[2, 0, 0, 0, 0, 0, 0])
    assert not no and "seeds" in reason
    # worse breach count -> reject
    worse_breach = Scoreboard([RunResult(True, True, True, 0.02) for _ in range(3)]
                              + [RunResult(True, True, True, 0.02)])   # 4 breaches
    no2, reason2 = gate.promote(worse_breach, base, seed_pass_counts=[3, 2, 1, 0, 0, 0, 0])
    assert not no2 and "breach" in reason2


# =============================================================================
# SECTION P — HPO (Optuna on NON-SACRED dials only) (M13)
# FTMO link: gamma/lambda + the aggression schedule are the patience that makes the
# bot pass; the guard makes them un-tunable so HPO can only help, never sabotage.
# =============================================================================
from quantra.learning_system.hpo import (  # noqa: E402
    DEFAULT_SEARCH_SPACE,
    SACRED_DIALS,
    run_study,
    suggest,
    validate_not_sacred,
)


class _FakeTrial:
    def suggest_float(self, n, lo, hi):
        return (lo + hi) / 2.0

    def suggest_categorical(self, n, choices):
        return choices[0]

    def suggest_int(self, n, lo, hi):
        return lo


def test_hpo_refuses_to_tune_sacred_dials():
    assert {"gamma", "lambda"} <= SACRED_DIALS
    with pytest.raises(ValueError):
        validate_not_sacred(["value_coef", "gamma"])      # gamma is hand-locked
    with pytest.raises(ValueError):
        suggest(_FakeTrial(), {"clip_range": (0.2, 0.4)})  # the G2 ranges are locked


def test_hpo_suggest_returns_only_non_sacred_params():
    params = suggest(_FakeTrial())
    assert set(params) == set(DEFAULT_SEARCH_SPACE)
    assert not (set(params) & SACRED_DIALS)               # never a sacred dial
    assert "value_coef" in params and params["minibatch"] in (32, 64, 128)


def test_hpo_study_maximizes_pass_rate_objective():
    """A tiny Optuna study runs over the non-sacred space and improves the objective."""
    def objective(params):
        # toy: pass-rate proxy peaks at value_coef≈0.5, grad_clip≈0.5
        return 1.0 - abs(params["value_coef"] - 0.5) - abs(params["grad_clip_norm"] - 0.5)
    study = run_study(objective, n_trials=12, seed=0)
    assert study.best_value > 0.0
    assert not (set(study.best_params) & SACRED_DIALS)    # winner uses no sacred dial


# =============================================================================
# SECTION Q — LIVE BRIDGE: ExecutionAdapter + ManualHalt + LiveRunner (M14)
# FTMO link: live execution must mirror training (5 slots, pointer-CLOSE) so the
# learned pass-behaviour reproduces, behind two hard kill switches that make a bad
# session non-fatal. Isolated from diagnostics.
# =============================================================================
from quantra.locked_core.platform_adapter import (  # noqa: E402
    BrokerAdapter,
    SimBrokerAdapter,
    make_adapter,
)
from quantra.live_bridge import ExecutionAdapter, LiveRunner, ManualHalt  # noqa: E402


def test_make_adapter_returns_a_broker_sim_default():
    # 'mt5' yields a real MT5Adapter where MetaTrader5 is installed, else a sim
    # fallback; either way it's a BrokerAdapter. 'sim' is always the paper broker.
    assert isinstance(make_adapter("mt5"), BrokerAdapter)
    assert isinstance(make_adapter("sim"), SimBrokerAdapter)
    assert make_adapter("sim").connect() is True


def test_execution_adapter_slots_and_pointer_close():
    ex = ExecutionAdapter(SimBrokerAdapter(), ["EURUSD"])
    for _ in range(7):                            # only 5 slots
        ex.open("EURUSD", side=1, lots=0.1, price=1.20)
    assert ex.n_open("EURUSD") == 5               # OPEN refused once full
    assert ex.open("EURUSD", 1, 0.1) is None
    assert ex.close("EURUSD", pointer=2, price=1.21) is True
    assert ex.slots["EURUSD"][2] is None and ex.n_open("EURUSD") == 4


def test_manual_halt_flattens_and_latches():
    broker = SimBrokerAdapter()
    ex = ExecutionAdapter(broker, ["EURUSD", "US30"])
    ex.open("EURUSD", 1, 0.1); ex.open("US30", -1, 0.1)
    halt = ManualHalt()
    closed = halt.halt(broker)
    assert closed == 2 and halt.is_halted        # flattened all + latched
    assert len(broker.positions()) == 0
    halt.reset(); assert not halt.is_halted       # manual reset only


def test_live_runner_deterministic_execution_and_kill_switches():
    broker = SimBrokerAdapter()
    ex = ExecutionAdapter(broker, ["EURUSD"])
    runner = LiveRunner(PPOAgent(state_dim=STATE_DIM),
                        ex, RiskManager(account_size=10_000))
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    dm = np.zeros(4, dtype=np.float32); dm[OPEN_SHORT] = -1e9   # forbid OPEN_SHORT
    pm = np.zeros(5, dtype=np.float32)
    info = runner.step("EURUSD", obs, dm, pm, atr_price=1e-3, price=1.20, remaining_budget=300.0)
    assert info["action"] in ("OPEN", "HOLD", "CLOSE", "OPEN_SKIPPED")
    # breach auto-flat kill switch latches and flattens
    ex.open("EURUSD", 1, 0.1, 1.20)
    assert runner.breach_autoflat(equity=9_600.0, wall_equity=9_600.0) is True
    assert runner.halt.is_halted and ex.n_open("EURUSD") == 0
    assert runner.step("EURUSD", obs, dm, pm, 1e-3, 1.20, 300.0)["action"] == "HALTED"


def test_live_bridge_isolated_from_diagnostics():
    """SOW C7: the live runner module must not import the diagnostics layer."""
    import quantra.live_bridge.live_runner as lr
    src = open(lr.__file__, encoding="utf-8").read()
    assert "quantra.diagnostics" not in src       # no diagnostics coupling at runtime


# =============================================================================
# SECTION R — END-TO-END ACCEPTANCE (M15)
# FTMO link: proves the whole mission machine composes - a brain trains under faithful
# physics, is logged, visualised, diagnosed (read-only), and ranked by PASS RATE.
# =============================================================================
from quantra.acceptance import run_acceptance  # noqa: E402
from quantra.diagnostics.llm_risk_doctor import TAXONOMY as _TAX, UNCLASSIFIED as _UNC  # noqa: E402


def test_end_to_end_acceptance_runs_the_whole_chain(tmp_path):
    """data -> features -> laws -> env -> agent -> train -> telemetry -> 7 visuals ->
    LLM diagnosis -> scoreboard, all in one run, without crashing."""
    res = run_acceptance(symbols=["EURUSD"], n_train_updates=1, eval_episodes=2,
                         bars=7000, seed=0, out_dir=tmp_path)
    # scoreboard produces the 4 ranking metrics
    s = res.scoreboard.summary()
    for k in ("pass_rate", "breaches", "target_hit_consistency", "max_drawdown_path"):
        assert k in s
    # the 7 required visuals were produced
    assert len(res.visuals) == 7
    for path in res.visuals.values():
        assert path.exists()
    # an evidence-cited LLM diagnosis following the output template
    rendered = res.diagnosis.render()
    assert "DIAGNOSIS" in rendered and "Failure classification" in rendered
    assert res.diagnosis.classification in _TAX or res.diagnosis.classification == _UNC
    # a checkpoint was saved (SOW §8.4)
    assert res.checkpoint.exists()


# =============================================================================
# SECTION S — PRODUCTION MT5 PATH: real close + hardened order-send + live loop (M14b)
# FTMO link: a trained brain can only bank a live pass if the live loop faithfully
# rebuilds the obs/masks/slots bar-by-bar and the MT5 adapter can actually open AND
# CLOSE. (Terminal-only calls are verified at source level; the loop is tested on sim.)
# =============================================================================
from quantra.live_bridge import LiveSession, ReplayBarFeed  # noqa: E402
from quantra.locked_core.platform_adapter.adapters import MT5Adapter  # noqa: E402


def test_mt5_close_and_order_are_implemented_not_stubs():
    import inspect
    close_src = inspect.getsource(MT5Adapter.close_position)
    assert "no-op stub" not in close_src                      # the old stub is GONE
    assert "position" in close_src and "_send" in close_src    # opposite deal refs the ticket
    order_src = inspect.getsource(MT5Adapter.market_order)
    assert "symbol_select" in order_src and "symbol_info_tick" in order_src and "_send" in order_src
    assert "retcode" in inspect.getsource(MT5Adapter._send)     # order_send result is checked


def test_live_session_builds_full_obs_and_decides_on_a_bar_stream(make_1m):
    df = make_1m(n_bars=6700, seed=3)
    sym = "EURUSD"
    ex = ExecutionAdapter(SimBrokerAdapter(), [sym])
    sess = LiveSession(PPOAgent(state_dim=STATE_DIM), ReplayBarFeed({sym: df}),
                       ex, RiskManager(account_size=10_000), [sym])
    obs, price, atr, law, wheel = sess._observe(sym)
    assert obs.shape == (STATE_DIM,) and law.shape == (12,)    # full STATE_DIM obs from live bars
    assert wheel.shape == (2,)                                 # tw_cci_block + tw_bb_block flags
    infos = sess.run_steps(2)                                   # process 2 closed bars
    assert infos and all("action" in i for i in infos)
    assert all(i["action"] in ("OPEN", "CLOSE", "HOLD", "OPEN_SKIPPED", "HALTED",
                               "BREACH_AUTOFLAT") for i in infos)


def test_live_session_breach_autoflat_latches(make_1m):
    df = make_1m(n_bars=6700, seed=4)
    sym = "EURUSD"
    ex = ExecutionAdapter(SimBrokerAdapter(), [sym])
    sess = LiveSession(PPOAgent(state_dim=STATE_DIM), ReplayBarFeed({sym: df}),
                       ex, RiskManager(account_size=10_000), [sym])
    slot = ex.open(sym, 1, 5.0, 1.20)                          # big long via broker
    sess.portfolio.open(sym, slot, 1, entry=1.20, lots=5.0, rpl=150.0)
    sess.portfolio.mark(sym, 1.10, 1e-3)                        # -0.10 * 100k * 5 = -$50k
    assert sess._check_breach({sym: 1.10}) is True
    assert sess.halt.is_halted and sess.portfolio.n_open(sym) == 0   # flattened + latched


# =============================================================================
# SECTION T — PER-DAY INPUTS + ftmo_mode + LEVERAGE/MARGIN (2026-06-15)
# FTMO link: the operator dials target/trailing-stop/leverage per day (per account).
# ftmo_mode ON = the 2-phase challenge (auto-flat at target, tighten to protect the
# pass). OFF = a single trailing stop that runs indefinitely (aggressive side accounts).
# Margin (1:leverage) is the REAL physical cap that lets OFF run uncapped-but-safe, and
# every cap only shrinks lots so the B5 no-overshoot invariant still holds.
# =============================================================================
def test_make_challenge_default_matches_locked_ftmo():
    c = cfg.make_challenge()
    assert (c.daily_target_pct, c.daily_risk_pct) == (2.5, 4.0)
    assert (c.hard_wall_pct, c.pain_zone_start_pct) == (4.0, 3.5)  # wall=risk, pain=7/8 wall
    assert c.ftmo_mode is True and c.leverage == 100.0


def test_make_challenge_clamps_into_mode_bounds():
    # ftmo ON: target/risk clamped to the challenge-safe band
    on = cfg.make_challenge(daily_target_pct=999, daily_risk_pct=999, ftmo_mode=True)
    assert on.daily_target_pct == cfg.FTMO_ON_BOUNDS["target"][1]   # 10.0
    assert on.daily_risk_pct == cfg.FTMO_ON_BOUNDS["risk"][1]       # 10.0
    # ftmo OFF: the wide side-account envelope; 80/20 passes through, wall pinned to risk
    off = cfg.make_challenge(daily_target_pct=80, daily_risk_pct=20, ftmo_mode=False, leverage=500)
    assert (off.daily_target_pct, off.daily_risk_pct) == (80.0, 20.0)
    assert off.hard_wall_pct == 20.0 and abs(off.pain_zone_start_pct - 17.5) < 1e-9
    assert off.leverage == 500.0
    # OFF still clamps the extremes
    assert cfg.make_challenge(daily_target_pct=999, daily_risk_pct=999,
                              ftmo_mode=False).daily_target_pct == cfg.FTMO_OFF_BOUNDS["target"][1]


def test_ftmo_off_single_trailing_and_never_autoflats():
    c = cfg.make_challenge(daily_target_pct=4, daily_risk_pct=10, ftmo_mode=False)
    cs = ChallengeState(account_size=10_000, challenge=c)
    cs.mark_to_market(500.0)                       # equity 10_500 >= target 10_400
    assert cs.target_hit is True                   # target reached...
    assert cs.should_autoflat is False             # ...but OFF never auto-flats (runs on)
    assert abs(cs.wall_equity - (cs.peak_equity - 0.10 * 10_000)) < 1e-6  # single 10% trailing
    cs.phase = "B"                                 # even if forced to B, OFF stays daily_risk_pct
    assert abs(cs.wall_equity - (cs.peak_equity - 0.10 * 10_000)) < 1e-6


def test_env_reset_injects_per_day_challenge():
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4)})
    assert env.challenge_cfg.daily_target_pct == 2.5      # default
    env.reset(challenge=cfg.make_challenge(daily_target_pct=5, daily_risk_pct=8, ftmo_mode=False))
    assert env.challenge_cfg.daily_risk_pct == 8.0 and env.challenge_cfg.ftmo_mode is False
    assert env.reward_engine.challenge.hard_wall_pct == 8.0           # reward stack tracks it
    assert abs(env.account.wall_equity - (10_000 - 0.08 * 10_000)) < 1e-6   # new trailing stop


def test_margin_ceiling_caps_lots_without_overshoot():
    rm = RiskManager(10_000, RiskConfig(max_per_trade_risk_frac=0.5))
    con = cfg.CONTRACT_SIZE["EURUSD"]
    # Big budget so the risk-buffer would allow many lots; margin must bind instead.
    sr = rm.size("EURUSD", 1.0, 1e-4, 100_000.0, price=1.20, contract=con,
                 leverage=100.0, free_margin=100.0)
    expected = math.floor((100.0 * 100.0) / (1.20 * con) / 0.01) * 0.01   # ~0.08 lots
    assert sr.feasible and sr.reason == "margin-capped"
    assert abs(sr.lots - expected) < 1e-9
    assert sr.committed_risk <= 100_000.0 + 1e-6                          # invariant intact


def test_ftmo_off_removes_per_trade_cap():
    """OFF (apply_per_trade_cap=False) lets confidence scale the WHOLE budget, so the same
    raw_size buys more than the 1%-capped ON path when the budget exceeds the cap."""
    rm = RiskManager(10_000, RiskConfig(max_per_trade_risk_frac=0.01))
    on = rm.size("EURUSD", 1.0, 1e-4, 500.0, apply_per_trade_cap=True)
    off = rm.size("EURUSD", 1.0, 1e-4, 500.0, apply_per_trade_cap=False)
    assert off.lots > on.lots
    assert off.committed_risk <= 500.0 + 1e-6 and on.committed_risk <= 500.0 + 1e-6


def test_ftmo_off_has_target_and_stop_for_day_toggle():
    """OFF keeps a target as the AIM. Default: runs PAST it (no auto-flat). With
    stop_for_day: banks and stops at the target."""
    run = ChallengeState(10_000, cfg.make_challenge(daily_target_pct=4, daily_risk_pct=10,
                                                    ftmo_mode=False))
    run.mark_to_market(500.0)                       # equity 10_500 >= target 10_400
    assert run.target_hit is True and run.should_autoflat is False   # target = aim, not a stop
    stop = ChallengeState(10_000, cfg.make_challenge(daily_target_pct=4, daily_risk_pct=10,
                                                     ftmo_mode=False, stop_for_day=True))
    stop.mark_to_market(500.0)
    assert stop.target_hit is True and stop.should_autoflat is True  # bank + stop


def test_env_off_stop_for_day_locks_out_day_at_target():
    # C10: OFF + stop_for_day banks the day at target and LOCKS OUT the rest of that day — it no
    # longer ENDS the episode (the multi-day account continues; midnight reopens it).
    T = 40
    close = np.linspace(1.20, 1.26, T).astype(float)         # climbs -> a long hits the target
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), close, np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0)}
    env = TradingEnv(data, challenge=cfg.make_challenge(daily_target_pct=2, daily_risk_pct=20,
                                                        ftmo_mode=False, stop_for_day=True),
                     risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    env.step((OPEN_LONG, 1.0, 0))
    for _ in range(T - 2):
        if env.account.locked_out:
            break
        env.step((HOLD, 0.0, 0))
    assert env.account.target_hit and env.account.locked_out  # banked + DAY locked out (not done)
    assert env._n_open("EURUSD") == 0                          # auto-flatted at target
    assert not env.done                                       # C10: the EPISODE is not ended


import math  # noqa: E402  (used by Section T margin math)


# =============================================================================
# SECTION U — LOGIC-AUDIT FIXES + CONSISTENCY DEMONSTRATION (2026-06-15)
# FTMO link: proves the post-audit logic. (1) per-symbol reward attribution is an exact
# PnL decomposition (the dominant signal reaches the right asset). (2) the daily reset
# fires on a calendar-day change (each day is its own fresh 2.5%/4% challenge). (3) THE
# DEMONSTRATION: across many challenge-days the bot banks +2.5% and NEVER breaches the 4%
# trailing wall — and on a losing day the wall caps the loss with no overshoot.
# =============================================================================
def test_per_symbol_contribution_is_an_exact_pnl_decomposition():
    """Each symbol's contribution = its own realized-net + its open uPnL; summed over the 4
    symbols it equals the total account PnL. This is the math behind correct L0 attribution
    (the whole-bar move is NO LONGER lumped onto the last symbol) [2026-06-15 fix]."""
    syms = ["EURUSD", "XAUUSD", "GBPUSD", "US30"]
    T = 50
    data = {}
    for i, s in enumerate(syms):
        close = (1.0 + 0.0002 * (i + 1) * np.arange(T)).astype(float)   # gentle, distinct trends
        data[s] = SymbolData(_open_gate_matrix(T), close, np.full(T, 1e-3),
                             np.full(T, 2e-5), valid_from=0)
    env = TradingEnv(data)
    env.reset()
    env.step((OPEN_LONG, 0.5, 0))                    # trade ONLY EURUSD (cursor 0)
    for _ in range(8):
        if env.done:
            break
        env.step((HOLD, 0.0, 0))
    # the decomposition is EXACT: per-symbol contributions sum to the total account PnL
    total_contrib = sum(env._sym_contribution(s) for s in env.symbols)
    assert abs(total_contrib - (env.account.equity - env.account.account_size)) < 1e-6
    # the PnL lands on the symbol that TRADED (EURUSD); US30 — which the OLD bug lumped the
    # whole-portfolio move onto — carries exactly 0 because it never traded.
    assert abs(env._sym_contribution("EURUSD")) > 1e-9
    assert env._sym_contribution("US30") == 0.0


def test_daily_reset_fires_on_calendar_day_change():
    """A calendar-day change re-anchors the day and returns to Phase A (fresh 2.5% target),
    instead of staying stuck in a post-day-1 Phase B [2026-06-15 fix: reset_day was dead]."""
    T = 24
    dates = np.array([0] * 10 + [1] * 14, dtype=np.int64)    # day0 then day1 at bar 10
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), np.full(T, 1.20),
                                 np.full(T, 1e-3), np.full(T, 2e-5), valid_from=0, dates=dates)}
    env = TradingEnv(data)
    env.reset()
    env.account.target_hit = True                    # simulate a day-0 pass into Phase B
    env.account.phase = "B"
    for _ in range(12):                              # cross into day 1 (1 symbol -> 1 bar/step)
        if env.done:
            break
        env.step((HOLD, 0.0, 0))
    assert env.account.phase == "A" and env.account.target_hit is False


# =============================================================================
# SECTION W — C10/C11 MULTI-DAY EPISODE + FAILED-DAY PENALTY (2026-06-19)
# FTMO link: an episode is N TRADING DAYS on ONE continuous account. A daily-wall breach LOCKS OUT
# the rest of that day (it does NOT end the episode); midnight resets the day and the account carries
# forward. A LARGE proportional penalty for missing a day's target makes CONSISTENCY the objective,
# not mere survival — which is what repeatedly passing FTMO requires.
# =============================================================================
def test_daily_target_is_relative_to_day_opening_balance():
    """C11: the day's target is daily_target_pct of THAT DAY'S opening balance (compounding), not a
    fixed % of account_size — so after a winning day the next day aims higher."""
    cs = ChallengeState(10_000, cfg.make_challenge(daily_target_pct=2.5, daily_risk_pct=4.0))
    assert abs(cs.daily_target_equity - 10_250.0) < 1e-9        # day 1: +2.5% of 10_000
    cs.mark_to_market(600.0)                                    # +6% day
    cs.reset_day()                                             # day 2 opens at 10_600 (carried forward)
    assert abs(cs.day_start_equity - 10_600.0) < 1e-9
    assert abs(cs.daily_target_equity - 10_600.0 * 1.025) < 1e-6   # +2.5% of the DAY'S opening


def test_target_hit_and_day_passed_share_identical_threshold():
    """C11 verification: 'target hit' (the latch) and 'day passed' (EOD pass / zero shortfall) must
    threshold on the SAME value — equity >= daily_target_equity — with no rounding or base skew."""
    cs = ChallengeState(10_000, cfg.make_challenge(daily_target_pct=2.5, daily_risk_pct=4.0))
    tgt = cs.daily_target_equity                       # day_start_equity * (1 + 2.5%)
    cs.mark_to_market(tgt - cs.account_size - 0.01)    # equity one cent BELOW target
    assert not cs.target_hit and not cs.day_passed and cs.day_shortfall_fraction > 0.0
    cs.mark_to_market(tgt - cs.account_size)           # equity EXACTLY at target
    assert cs.target_hit and cs.day_passed and cs.day_shortfall_fraction == 0.0


def test_day_shortfall_fraction_zero_at_target_one_when_flat():
    """C11 shortfall: 0 at/above target, 1.0 if the day ended flat, >1 if it lost money (the wall-hit
    day is punished hardest because its shortfall is largest)."""
    cs = ChallengeState(10_000, cfg.make_challenge(daily_target_pct=2.5, daily_risk_pct=4.0))
    cs.mark_to_market(0.0)                                      # flat day (0% of +2.5%)
    assert abs(cs.day_shortfall_fraction - 1.0) < 1e-9
    cs.mark_to_market(250.0)                                    # exactly at target
    assert cs.day_shortfall_fraction == 0.0
    cs.mark_to_market(-400.0)                                   # -4% day (hit the wall)
    assert cs.day_shortfall_fraction > 1.0


def test_reset_day_clears_breach_and_lockout_account_carries():
    """C10: reset_day lifts breached + locked_out and re-anchors the day to current equity (the
    LOSS carries forward), so a breached day reopens fresh next morning."""
    cs = ChallengeState(10_000, cfg.make_challenge(daily_target_pct=2.5, daily_risk_pct=4.0))
    cs.mark_to_market(-450.0)                                   # -4.5% -> breach the 4% wall
    assert cs.breached and cs.locked_out
    cs.reset_day()
    assert not cs.breached and not cs.locked_out               # fresh day
    assert abs(cs.day_start_equity - 9_550.0) < 1e-9           # the -450 carried forward


def test_no_new_opens_while_day_locked_out():
    """C10: while locked out, direction_mask forbids new opens and the env coerces a forbidden
    OPEN to HOLD (CLOSE/HOLD stay legal)."""
    env = TradingEnv({"EURUSD": _sym(T=20, atr=1e-4)})
    env.reset()
    env.account.locked_out = True
    m = env.direction_mask("EURUSD")
    assert m[OPEN_LONG] <= -1e8 and m[OPEN_SHORT] <= -1e8
    _, _, _, info = env.step((OPEN_LONG, 1.0, 0))
    assert info["coerced"] is True and info["executed"] == "HOLD"


def test_breach_locks_out_day_then_resets_next_day():
    """C10 end-to-end: a day-1 crash breaches the wall -> locked out (NOT done); at the calendar-day
    boundary the lockout lifts, the breach clears, and the day-1 loss carries into day 2."""
    T = 24
    dates = np.array([0] * 12 + [1] * 12, dtype=np.int64)
    close = np.concatenate([np.full(3, 1.20), np.linspace(1.20, 1.08, 9),
                            np.full(12, 1.08)]).astype(float)
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), close, np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0, dates=dates)}
    env = TradingEnv(data, risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    env.step((OPEN_LONG, 1.0, 0))
    saw_lockout = saw_reset = False
    day1_equity = None
    for _ in range(T):
        if env.done:
            break
        env.step((HOLD, 0.0, 0))
        if env.account.locked_out and not saw_lockout:
            saw_lockout = True
            day1_equity = env.account.equity
            assert not env.done                               # breach did NOT end the episode
        if saw_lockout and not env.account.locked_out:        # crossed midnight
            saw_reset = True
            assert not env.account.breached                   # fresh day
            break
    assert saw_lockout and saw_reset
    assert day1_equity < 10_000.0                             # the breach loss is real
    assert env.account.day_start_equity < 10_000.0           # ...and carried into day 2


def test_episode_ends_after_configured_days():
    """C10: with episode_days=2 over 3 calendar days, the episode ends at the 2nd midnight boundary
    (not at a breach, not at end-of-data)."""
    T = 36
    dates = np.array([0] * 12 + [1] * 12 + [2] * 12, dtype=np.int64)
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), np.full(T, 1.20), np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0, dates=dates)}
    env = TradingEnv(data, episode_days=2)
    steps = 0
    while not env.done and steps < T + 5:
        env.step((HOLD, 0.0, 0)); steps += 1
    assert env.done and env._days_elapsed == 2                # exactly 2 trading days


def test_failed_day_penalty_fires_proportionally_on_a_missed_day():
    """C11: a day that HOLDs flat (0% < +2.5% target) takes a LARGE failed-day penalty at the
    midnight boundary == -failed_day_penalty * shortfall (shortfall==1.0 for a flat day)."""
    T = 24
    dates = np.array([0] * 12 + [1] * 12, dtype=np.int64)
    fdp = 5.0
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), np.full(T, 1.20), np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0, dates=dates)}
    env = TradingEnv(data, challenge=cfg.make_challenge(daily_target_pct=2.5, daily_risk_pct=4.0,
                                                        failed_day_penalty=fdp))
    boundary_penalty = boundary_reward = None
    for _ in range(T):
        if env.done:
            break
        _, reward, _, info = env.step((HOLD, 0.0, 0))
        if info["day_penalty"] != 0.0:
            boundary_penalty, boundary_reward = info["day_penalty"], reward
            break
    assert boundary_penalty is not None
    assert abs(boundary_penalty - (-fdp)) < 1e-6              # flat day -> shortfall 1.0 -> -fdp
    assert boundary_reward < -1.0                             # the penalty dominates the step reward


def test_failed_day_penalty_zero_when_target_reached():
    """C11: a day that REACHES its target incurs NO failed-day penalty at the boundary."""
    T = 24
    dates = np.array([0] * 12 + [1] * 12, dtype=np.int64)
    close = np.concatenate([np.linspace(1.20, 1.30, 12), np.full(12, 1.30)]).astype(float)  # day 0 climbs
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), close, np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0, dates=dates)}
    env = TradingEnv(data, challenge=cfg.make_challenge(daily_target_pct=2.5, daily_risk_pct=8.0,
                                                        failed_day_penalty=5.0),
                     risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    env.step((OPEN_LONG, 1.0, 0))                            # ride the day-0 climb past +2.5%
    crossed = False
    for _ in range(T):
        if env.done:
            break
        _, _, _, info = env.step((HOLD, 0.0, 0))
        if info["days_elapsed"] >= 1:                        # the day-0 boundary has been crossed
            crossed = True
            assert info["day_penalty"] == 0.0                # target was reached -> no penalty
            break
    assert crossed


# =============================================================================
# SECTION V — REAL-DATA BACKTEST (2026-06-15). NO synthetic pass-claims.
# FTMO link: the ONLY performance proof is on REAL MT5 bars via scripts/real_backtest.py.
# (1) unit-test the MT5-style report math (ground-truth net = final − account, incl. the
# breach force-flatten). (2) a gated real backtest, run only when the real CSV is present
# AND QUANTRA_REAL_BACKTEST=1, so the suite stays fast + offline by default. The synthetic
# "demonstration" tests were REMOVED — a real backtest already showed the trailing wall can
# be overshot, so fake favorable-data pass claims have no place here.
# =============================================================================
from scripts.real_backtest import compute_stats  # noqa: E402


def test_backtest_report_math_is_correct():
    """MT5-style stats on a KNOWN trade list + equity curve: net is the GROUND-TRUTH equity
    change (not just the agent's closes), plus profit factor / win% / max drawdown."""
    trades = [100.0, -40.0, 60.0, -20.0]
    equity = [10000.0, 10100.0, 10060.0, 10120.0, 10100.0]
    s = compute_stats(trades, equity, account_size=10000.0)
    assert s["closed_net"] == 100.0                  # 100 - 40 + 60 - 20
    assert s["net"] == 100.0                          # final 10100 - 10000 (ground truth)
    assert s["forced"] == 0.0                          # closed == equity change here
    assert s["gp"] == 160.0 and s["gl"] == -60.0
    assert abs(s["pf"] - 160.0 / 60.0) < 1e-9
    assert abs(s["win"] - 50.0) < 1e-9                # 2 of 4 wins
    assert abs(s["maxdd"] - 40.0) < 1e-9              # peak 10100 -> dip 10060 = 40


def test_real_backtest_runs_on_real_bars_when_enabled():
    """Gated REAL proof: with data/raw/EURUSD_recent.csv present AND QUANTRA_REAL_BACKTEST=1,
    run the harness end-to-end on real bars and assert it emits a self-consistent report.
    Skipped by default (the real run is `python scripts/real_backtest.py`)."""
    if not os.path.exists("data/raw/EURUSD_recent.csv") or os.environ.get("QUANTRA_REAL_BACKTEST") != "1":
        pytest.skip("set QUANTRA_REAL_BACKTEST=1 with data/raw/EURUSD_recent.csv present")
    import subprocess
    import sys as _sys
    # Pass --path explicitly: real_backtest.py's --path now defaults to None (omitting it
    # triggers the loader's gdown auto-download fallback). This gated test guards on the
    # local CSV existing, so it must point at exactly that file to stay deterministic.
    r = subprocess.run([_sys.executable, "scripts/real_backtest.py", "--updates", "0",
                        "--path", "data/raw/EURUSD_recent.csv"],
                       capture_output=True, text=True, timeout=600)
    assert "QUANTRA BACKTEST" in r.stdout and "GROUND TRUTH" in r.stdout


# Allow `python tests/test_ftmo_master_suite.py` to run the whole suite directly.
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) — standing rule since 2026-06-13.
# Every change APPENDS a dated IRAC entry (newest last). Conclusion is ALWAYS why
# the change makes the bot pass FTMO more consistently with no bug/inefficiency.
# Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Created the single master suite (hardening pass).
#   I: Tests were split across test_runtime.py + test_data_pipeline.py; the operator
#      wanted ONE runnable suite, and future milestones risked fragmenting tests.
#   R: New standing rule — one master suite all future tests append to; every guard
#      framed against repeated FTMO passing (SOW Section 11 acceptance tests).
#   A: Consolidated all runtime + data-pipeline tests here under chain-ordered
#      sections (A runtime, B loader, C resampler); added efficiency/caching guards
#      (parquet-cache reuse, parse throughput); deleted the two old test files.
#   C: A single green command now proves the training substrate is faithful and
#      cheap, so we can afford the seeds/windows that establish a reproducible pass
#      rate — and the LLM can rule the substrate out first when triangulating.
# [2026-06-13] Added Section D - feature builder + state vector (M2).
#   I: The ~145-scalar observation needed proof it is complete (146), bounded,
#      signal-bearing, and lookahead-free before any policy trains on it.
#   R: STATE_VECTOR.md schema (146) + lookahead-safety + bounded-encoding design.
#   A: Added Section D - schema/block-width pins, config==schema, finite+clipped
#      matrix, signal-variance, truncated-vs-full no-lookahead, cyclical time,
#      assemble_state width+validation, valid_from warmup. 8 tests (23 total green).
#   C: The bot's world is now provably faithful and leak-free, so a learned edge is
#      real and transfers live - the precondition for repeatable FTMO passing.
# [2026-06-13] Added Section E - change-impact tracker + raw-input block tests.
#   I: An operator-added raw SMA/CCI block changed the observation (146->176); the
#      obs could later drift again undetected and silently degrade pass-rate.
#   R: Operator directives (raw inputs + a change-impact tracking system, LLM-readable).
#   A: Updated Section D to 176/raw-block coverage; added Section E - snapshot-match,
#      LLM-interpretation, AST dependency-graph, and FTMO-framed-report tests.
#   C: The observation is now both faithful AND change-guarded, so any future obs
#      change is caught with a concrete follow-up list before it can hurt passing.
# [2026-06-13] Added Section F - LawMask (M3) + gate ingredients (dim 176->179).
#   I: The bot needed its spine (which directions are legal) proven correct before
#      training, and the snapshot needed re-pinning after adding gate ingredients.
#   R: THE_TRADING_CODE.md (9 laws + 3 gates, exact params) + SOW C5 (-1e9) + §2.3-2.4
#      position/slot legality + the two enforcement modes.
#   A: Section F - law states per family (incl. CCI +100, ssma align, pullback desync),
#      3 gates, position/slot legality, live-ban vs school-permission, pointer mask,
#      end-to-end wrapper. Re-pinned the state-vector snapshot to 179.
#   C: The legal space is verified exactly per blueprint, so the mask mechanically
#      blocks breach-bound directions in both training and live - the foundation of
#      not breaching, which is the foundation of consistent passing.
# [2026-06-13] Added Section G - Env + RiskManager + CostLayer (M4).
#   I: The challenge physics (shared account, 5 slots, true-sequential risk, costs,
#      wall) needed proof that no risk/sizing rule can overshoot and that the loop is
#      faithful, before any policy trains on it.
#   R: SOW B5 (no collective overshoot) + H3 (sizing vs buffer) + §10.5 (costs) +
#      §2.4 (slots) + §2.7 (wall).
#   A: Section G - CostLayer (forex-only $5 RT, spread/slippage), RiskManager
#      no-overshoot (2000-case fuzz + refusal), B5 4-symbol one-bar invariant,
#      true-sequential buffer visibility, slot fill/mask/pointer-close, round-trip cost
#      bleed, hard-wall force-flatten, 179-dim obs + mask coercion. 12 tests.
#   C: The env is now provably faithful and overshoot-proof, so the behaviour the bot
#      learns here is the behaviour that passes real challenges - not a sim artifact.
# [2026-06-13] Added Section H - PPOAgent + RolloutBuffer + PPO loss (M5).
#   I: The brain that consumes the 179-dim obs and emits masked, sized, slot-aware
#      actions - plus the summed-log-prob PPO loss - needed building + proving.
#   R: PPO_ENGINE.md (3x256 trunk, 4 heads, Beta size, summed 3-head log-prob,
#      size-on-OPEN/pointer-on-CLOSE gating, -1e9 masks, live argmax/mean) + SOW §2.8/2.9.
#   A: Section H - locked architecture shape, mask-respecting sampling, the OPEN/CLOSE
#      log-prob gating, deterministic argmax/Beta-mean, 10-field buffer + no-replay,
#      PPO loss (ratio≈1/KL≈0 post-collection, grad reaches the trunk), agent-drives-env.
#   C: The policy provably cannot sample an illegal/breach-bound action and its update
#      is a correct trust-region step - so training under the M4 physics moves it toward
#      passing, not toward the wall.
# [2026-06-13] Added Section I - RewardEngine + QUAD + E8 (M6).
#   I: The objective needed Layer-0 dominance proven, plus the pain ramp and QUAD ceiling.
#   R: REWARD_DESIGN.md + E8 (L0 dominates) + E9 (QUAD 95% ceiling, pass-day gate).
#   A: Section I - E8 dominance over 1000 rollouts, pain-zone monotonicity, L0 whisper,
#      QUAD 95% ceiling + non-pass-day zero + streak reset, env-uses-engine. 6 tests.
#   C: The training signal provably can't be hijacked by a shaper, so PPO optimizes real
#      net progress inside the legal/risk-safe space - the objective that passes.
# [2026-06-13] Added Section J - curriculum + two-phase episode (M7).
#   I: The +2.5% win needed banking behind a tight wall, and training needed staging.
#   R: SOW §2.6 (two-phase) + §7 (law-gated curriculum, structure-first).
#   A: Section J - phase-B wall tightening, env auto-flat+enter-phase-B at target,
#      stage configs + school mode, 1m-timing feature mask (law ingredients kept),
#      graduation to live. 5 tests.
#   C: The bot banks pass-days behind a 1% wall and learns law-respect by habit -
#      both push it toward consistent passing rather than reckless profit-chasing.
# [2026-06-13] Added Section K - Trainer + GAE + scheduler + G8 (M8).
#   I: Nothing ran the on-policy PPO loop that turns the pieces into a trained brain.
#   R: SOW §2.1/2.8 (PPO), G2/G4 (locked gamma/lambda + ranges + 512/64), G8, §8.4.
#   A: Section K - GAE locked gamma/lambda + shapes, scheduler dials within ranges +
#      cooling, G8 multi-TF-agree/flat/>=1.5ATR, trainer-runs-updates-checkpoints. 5 tests.
#   C: Repeated patient PPO steps under faithful physics + masks produce a brain that
#      hits target without breaching, and every brain is saved for the promotion gate.
# [2026-06-13] Added Section L - TelemetryLogger round-trip (M9).
#   I: The diagnostics layer needed a versioned, complete, reconstructable record.
#   R: MLP_INTERPRETABILITY_LAYER.md data contract + versioned schema.
#   A: Section L - full-field round-trip (every contract field survives JSONL),
#      header carries schema version + grouped block names for the LLM. 2 tests.
#   C: Any breach/pass-day is fully reconstructable, so the Risk Doctor can diagnose
#      the cause - the loop that stops the same failure recurring and eroding pass-rate.
# [2026-06-13] Added Section M - MLPInterpreter 7 visuals (M10).
#   I: Telemetry arrays explain nothing without the standard visual evidence.
#   R: MLP_INTERPRETABILITY_LAYER.md REQUIRED VISUALS (the 7, tied to passing).
#   A: Section M - generate_all() writes all 7 PNGs from a telemetry run. 1 test.
#   C: Internal behaviour becomes inspectable evidence, so robust pass-behaviour is
#      distinguishable from fragile luck - protecting the pass rate.
# [2026-06-13] Added Section N - LLMRiskDoctor (M11).
#   I: The read-only/mandatory-rulebook boundary + taxonomy diagnosis needed enforcing.
#   R: MLP_INTERPRETABILITY_LAYER.md (template, 8 taxonomy, no 9th) + SOW R5/§9.3.
#   A: Section N - fail-loud without rulebook, no write/modify/execute methods,
#      Reward-Hijack classification with cited evidence + template render, never a 9th. 4 tests.
#   C: Failures get a true, evidence-backed cause + safe prescription before they cost a
#      challenge, and the doctor can never touch execution - it only protects passing.
# [2026-06-13] Added Section O - Scoreboard + walk-forward + promotion gate (M12).
#   I: Nothing ranked brains by pass-rate, validated out-of-sample, or gated promotion.
#   R: SOW §1.3/I1 (ranking) + I2 (12/2/1, 7 seeds) + I3 (>=3 seeds + improvement + no worse breach).
#   A: Section O - scoreboard ranks by pass-rate-not-PnL, 12/2/1 window generation,
#      runner over windows x seeds, promotion gate's 3 conditions. 4 tests.
#   C: Only robust, no-worse-breach improvements ship, so the deployed pass rate only
#      ratchets up - the operational definition of repeatable passing.
# [2026-06-13] Added Section P - HPO sacred-guard (M13).
#   I: Tuning was needed but a naive search could tune away the patience that passes.
#   R: SOW G6 (Optuna on non-sacred dials; gamma/lambda/scheduler hand-locked).
#   A: Section P - guard refuses sacred dials, suggest returns only non-sacred, tiny
#      Optuna study maximizes a pass-rate proxy with no sacred dial. 3 tests.
#   C: HPO improves tunable stability while the patience underpinning passing stays
#      locked - tuning can only help the pass rate, never sabotage it.
# [2026-06-13] Added Section Q - live bridge (M14).
#   I: Live execution + kill switches + diagnostics isolation needed building/proving.
#   R: SOW §2.10 (determinism) + B2 (5 slots/pointer-CLOSE) + §10.1 (kill switches) + C7 (isolation).
#   A: Section Q - adapter sim fallback, slot/pointer-close mechanics, manual-halt flatten+latch,
#      live-runner deterministic exec + breach auto-flat + halted-block, no-diagnostics-import. 5 tests.
#   C: The learned pass-behaviour reproduces live with identical masks/slots, behind two
#      hard kill switches - so passes get banked without a bad session blowing the account.
# [2026-06-13] Added Section R - end-to-end acceptance (M15). BUILD COMPLETE (M0-M15).
#   I: The 14 milestones needed proof they compose into one working mission machine.
#   R: SOW §11.3 (one run: scoreboard 4 metrics; telemetry -> 7 visuals; LLM diagnosis).
#   A: Section R - run_acceptance trains a brain, evals deterministically with full-
#      contract telemetry, emits the 7 visuals + a template diagnosis + the scoreboard. 1 test.
#   C: The whole chain runs green, so the system can be pointed at real bars + 7 seeds to
#      establish a real pass rate - the FTMO-passing machine is built and verifiable.
# [2026-06-13] Added Section S - production MT5 path (M14b).
#   I: M14's MT5 adapter couldn't close + had no live feed; it couldn't actually trade.
#   R: SOW §2.10/§10 (live determinism + kill switches) + C4 (1m) + no-lookahead.
#   A: Section S - MT5 close/order-send are real (source-verified, terminal-only), live
#      session builds the 179 obs from a bar stream + decides + breach-auto-flats. 3 tests.
#   C: A trained brain can now be driven on MT5 bar-by-bar with faithful obs/masks/slots
#      behind hard kill switches - the bridge from a trained brain to a banked live pass.
# [2026-06-15] Added Section T - per-day inputs + ftmo_mode + leverage/margin.
#   I: New per-day config injection, ftmo OFF single-trailing/no-autoflat, and the margin
#      ceiling needed locked-in proof.
#   R: Operator decision 2026-06-15 + B5 (no-overshoot must survive the new caps).
#   A: Section T - make_challenge defaults/clamps, OFF single-trailing + never-autoflat,
#      env.reset(challenge=) injection, margin caps lots, OFF removes per-trade cap. 6 tests.
#   C: The adjustable-input + margin behaviour is regression-proof, so the operator can dial
#      accounts per day with the no-overshoot guarantee intact.
# [2026-06-15] Added Section U - logic-audit fixes + consistency DEMONSTRATION.
#   I: The adversarial logic audit found a dominant-reward mis-attribution, a dead daily reset,
#      and an unbounded L4; the fixes needed locked-in proof + an end-to-end pass demonstration.
#   R: Logic audit 2026-06-15 (verified bugs) + the mission (2.5%/day, no 4% breach, consistently).
#   A: Section U - exact per-symbol PnL decomposition (US30 no longer absorbs others' PnL),
#      daily reset fires on a date change, and a 15-day demonstration: every day banks +2.5%,
#      0 breach the 4% wall, plus a losing-day protection check (loss capped at the wall). 5 tests.
#   C: The post-fix challenge LOGIC provably delivers consistent passing mechanics (banks the
#      target, never overshoots the wall) - the rails the trained policy needs to pass for real.
# [2026-06-15] Removed FAKE synthetic demonstrations; added Section V (REAL-DATA backtest).
#   I: The synthetic "banks 2.5%/3% consistently" demonstrations were rigged favorable data
#      (operator: "no fake tests"); a REAL backtest on REAL EURUSD bars even DISPROVED the
#      "loss never overshoots the wall" protection test (it hit 11% DD on a 4% wall).
#   R: Operator directive 2026-06-15: real proof only, on real MT5 bars (scripts/real_backtest.py).
#   A: Deleted the 15-day demonstration + protection + _favorable_day. Added Section V: a
#      unit test of the MT5-style report math (ground-truth net = final-account) + a gated
#      real backtest (runs only with the real CSV + QUANTRA_REAL_BACKTEST=1).
#   C: The suite no longer claims a pass it can't honestly back; performance is measured on
#      REAL bars, so what we report about passing FTMO is true - not a fitted illusion.
# [2026-06-18] Section V gated backtest now passes --path explicitly (real_backtest --path fix).
#   I: scripts/real_backtest.py's --path now defaults to None so an OMITTED --path triggers the
#      loader's Parquet/Drive/gdown fallback (operator brief Section 9). This gated test ran the
#      script with NO --path, relying on the old hardcoded default; after the fix that would
#      resolve via the fallback instead of the local CSV the test guards on.
#   R: The test only runs when data/raw/EURUSD_recent.csv exists AND QUANTRA_REAL_BACKTEST=1, so
#      it must point the harness at exactly that file to stay deterministic.
#   A: Added "--path data/raw/EURUSD_recent.csv" to the subprocess invocation; compute_stats
#      import and assertions unchanged.
#   C: The real-bar proof still measures the exact CSV it guards on, while the shipped script
#      gains the clean-checkout auto-download path - the gateway to a real FTMO pass.
