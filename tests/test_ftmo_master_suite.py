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
def test_schema_total_is_179_with_raw_block_and_gate_ingredients():
    # 89 market+time + 3 gate ingredients (M3) + 30 raw + 12 law + 35 trade + 3 + 7
    assert STATE_DIM == 179
    for name, width in EXPECTED_WIDTHS.items():
        s, e = SCHEMA.block_spans[name]
        assert e - s == width, f"block {name} width drifted"
    assert EXPECTED_WIDTHS["market"] == 92 and EXPECTED_WIDTHS["market_raw"] == 30
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
    assert len(raw_idx) == 30
    assert np.isfinite(mm.matrix[:, raw_idx]).all()
    cci_cols = [i for i, n in enumerate(PRECOMPUTED_NAMES) if n.startswith("raw_cci")]
    assert np.abs(mm.matrix[1000:, cci_cols]).max() > 10.0  # proves no clip on raw


def test_market_features_carry_signal(make_1m):
    """Fast 1m features must vary after warmup (real signal, not a dead constant)."""
    mm = build_market_matrix(make_1m(n_bars=4000, seed=21))
    for feat in ["z10_1m", "cci10_norm_1m", "candle_return_1m"]:
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
    dev_only = _feat_row(cci30_dev_5m=0.2, cci100_dev_5m=0.2, cci30_dev_30m=0.2, cci100_dev_30m=0.2,
                         cci30_norm_5m=0.5, cci100_norm_5m=0.5, cci30_norm_30m=0.5, cci100_norm_30m=0.5)
    assert _law(dev_only, "law_super_trend_cci") == 0   # not above +100
    assert _law(dev_only, "law_trend_cci") == 1         # trend (no +100) DOES fire
    full = _feat_row(cci30_dev_5m=0.2, cci100_dev_5m=0.2, cci30_dev_30m=0.2, cci100_dev_30m=0.2,
                     cci30_norm_5m=1.5, cci100_norm_5m=1.5, cci30_norm_30m=1.5, cci100_norm_30m=1.5)
    assert _law(full, "law_super_trend_cci") == 1


def test_shifted_sma_laws_use_align_flags():
    buy = _feat_row(ssma_align_1m=1, ssma_align_5m=1, ssma_align_30m=1)
    assert _law(buy, "law_super_trend_ssma") == 1       # needs 1m+5m+30m
    assert _law(buy, "law_trend_ssma") == 1             # needs 5m+30m
    pb = _feat_row(ssma_align_5m=1, ssma_align_30m=1, ssma_align_1m=-1)
    assert _law(pb, "law_pullback_ssma") == 1           # 1m pulls back inside HTF up
    assert _law(pb, "law_super_trend_ssma") == 0


def test_pullback_cci_desync():
    buy = _feat_row(cci10_dev_30m=0.2, cci100_dev_30m=0.2, cci100_dev_5m=0.2, cci10_dev_5m=-0.2)
    assert _law(buy, "law_pullback_cci") == 1


def test_gates_atr_spread_stationarity():
    g = _feat_row(atr_dev_1m=0.1, atr_dev_30m=0.1, spread_range_ratio_1m=0.3, adf_stat_1m=-3.5)
    assert _law(g, "gate_atr_liquidity") == 1
    assert _law(g, "gate_spread") == 1
    assert _law(g, "gate_stationarity") == 1            # adf below -2.86 -> stationary
    closed = _feat_row(atr_dev_1m=-0.1, spread_range_ratio_1m=2.0, adf_stat_1m=0.0)
    assert _law(closed, "gate_atr_liquidity") == 0
    assert _law(closed, "gate_spread") == 0
    assert _law(closed, "gate_stationarity") == 0


def _gates_open():
    s = np.zeros(12, dtype=np.float32)
    s[9] = s[10] = s[11] = 1  # atr, spread, stationarity open; no directional law
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


def test_mask_closed_gate_bans_new_opens():
    s = _gates_open(); s[9] = 0  # ATR gate closed
    m = build_direction_mask(s, position=0, n_open=0, mode=MODE_LIVE)
    assert m[OPEN_LONG] < -1e8 and m[OPEN_SHORT] < -1e8 and m[HOLD] == 0


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
    assert res.opens_allowed_by_gates is True
    assert res.direction_mask[OPEN_SHORT] < -1e8           # buy super-trend bans shorts
    assert res.direction_mask[OPEN_LONG] == 0


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
def test_hard_wall_force_flattens_and_ends_episode():
    # price crashes after entry -> equity hits the 4% wall -> breach + flatten + done
    T = 12
    close = np.concatenate([np.full(3, 1.20), np.linspace(1.20, 1.10, T - 3)]).astype(float)
    data = {"EURUSD": SymbolData(_open_gate_matrix(T), close, np.full(T, 1e-3),
                                 np.full(T, 2e-5), valid_from=0)}
    env = TradingEnv(data, risk_cfg=RiskConfig(max_per_trade_risk_frac=0.5))
    env.step((OPEN_LONG, 1.0, 0))
    done = False
    for _ in range(T):
        if done:
            break
        _, _, done, info = env.step((HOLD, 0.0, 0))
    assert env.account.breached and done
    assert env._n_open("EURUSD") == 0            # force-flattened


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
