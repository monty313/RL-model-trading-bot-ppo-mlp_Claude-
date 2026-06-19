"""The 9 directional laws + 3 market-condition observation signals — from the M2 matrix. 🔴

WHAT THIS MODULE DOES
---------------------
Computes the 12 law/signal STATES (the schema `law` block) from the normalized market
features, exactly per THE_TRADING_CODE.md / SOW-D4. Directional laws emit -1 (sell
context) / 0 (inactive) / +1 (buy context); the 3 market-condition signals emit
0 / 1. Vectorized over the whole matrix (offline precompute) and per-row (live).

The 9 directional laws (3 families x 3):
  Super Trend: Bollinger / CCI / Shifted-SMA   (strongest expansion; continuation only)
  Trend:       Bollinger / CCI / Shifted-SMA   (directional structure; one side legal)
  Pull Back:   Bollinger / CCI / Shifted-SMA   (retrace inside HTF direction; that side only)
The 3 market-condition signals (formerly "gates"): Volatility (ATR 5m+4H) · Spread ·
Stationarity (30-bar DF). These are OBSERVATION-ONLY by default (config.TRAINING_PHASE ==
PHASE_FREE) — the bot SEES them and learns to use them rather than being hard-blocked. Only
market_stationarity_obs re-enforces, and only in PHASE_CONSTRAINED (see engine.py). This is
the fix for the ~98.7% open-lockout that stopped the policy ever learning to pass.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Laws are the bot's SPINE. They run BEFORE the policy and define what is legal; the
mask (mask_engine) then forbids the wrong direction with logit -1e9. A clean law
state saves the bot from directional stupidity that would breach the 4% wall — the
single biggest source of avoidable breaches. Laws are NEVER reward terms (SOW R5).

4H OBSERVATION RULE 🔴: no DIRECTIONAL law reads 4H to activate. 4H is context only.
Every directional law binds exactly the timeframes in THE_TRADING_CODE.md (1m/5m/30m).
Carve-out: market_volatility_obs reads atr_dev_4H as an OBSERVATION INGREDIENT (paired
with 5m) — it never solo-triggers the mask, and it is observation-only in PHASE_FREE, so
4H still never gatekeeps a trade by itself. (Training wheels are the other 4H carve-out.)

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md`` (Term 2 Law Context). Before blaming
the actor for a "bad" action, verify what was LEGAL here: a directional law in +1
bans OPEN_SHORT; in -1 bans OPEN_LONG. If a law flag looks wrong, the break is in its
INGREDIENT (the market feature), not the actor. These states are deterministic from
price — identical across seeds — so a law-state difference across runs is a data bug.
"""

from __future__ import annotations

import numpy as np

# COUPLING [C1] -> market_pipeline/feature_builder/indicators.py: ADF_CRIT_5PCT is the
# locked 5% DF critical value the Stationarity gate compares against; changing it there
# silently shifts the stat_gate flag here (and the legal space the mask derives from it).
from quantra.market_pipeline.feature_builder.indicators import ADF_CRIT_5PCT
# COUPLING [C1] -> market_pipeline/feature_builder/schema.py: PRECOMPUTED_NAMES order +
# SCHEMA.blocks["law"] are consumed below (_IDX map + LAW_NAMES). Adding/renaming/reordering
# precomputed features or law names there must change _IDX/LAW_NAMES usage here in lockstep.
from quantra.market_pipeline.feature_builder.schema import PRECOMPUTED_NAMES, SCHEMA

# The 12 law/signal names, in schema `law` block order. compute_law_states returns
# columns in exactly this order so they drop straight into the observation.
# COUPLING [C4 in COUPLINGS.md]: this ORDER (9 directional then 3 market-condition signals)
# is sliced by position in law_mask_engine (_OBS_IDX, law_states[:9]/[9:]) and mirrored in
# schema._law_names. Reorder here => update schema._law_names + the mask engine.
# ⚠️ COMPATIBILITY [C18+] -> quantra/learning_system/policy_registry/registry.py
# (default_law_fingerprint -> compatibility_signature): LAW_NAMES is the THIRD input to a policy's
# compatibility hash. Adding/removing/renaming a law changes EVERY saved policy's signature, so the
# registry will force a fresh start on RESUME (old law-fingerprint no longer matches). Reordering only
# (same names) does NOT change the hash here, but DOES break the positional slices above — fix those too.
# ✅ SAFE TO CHANGE without a fresh start: everything OUTSIDE the law set — training_phase,
#    training_wheels, the challenge numbers, and the reward weights/math — none of those touch LAW_NAMES.
LAW_NAMES = list(SCHEMA.blocks["law"])
DIRECTIONAL_LAWS = LAW_NAMES[:9]   # -1/0/+1
MARKET_SIGNALS = LAW_NAMES[9:]     # 0/1 (volatility, spread, stationarity) — observation-only by default

# Column index of each precomputed feature name. COUPLING [C1]: this index map depends
# on schema.PRECOMPUTED_NAMES ORDER; env._COL + scheduler._COL build the same map.
_IDX = {name: i for i, name in enumerate(PRECOMPUTED_NAMES)}

# +100 / -100 CCI thresholds on the RAW CCI value [operator decision 2026-06-13: CCI
# is kept raw, so the Super-Trend ±100 check reads raw CCI vs ±100, not cci_norm vs ±1].
# COUPLING: must match SOW-D4's ±100 + the raw cci{p}_{tf} features in schema/builder.
_CCI_HI, _CCI_LO = 100.0, -100.0


def _c(mat: np.ndarray, name: str) -> np.ndarray:
    """Column accessor by feature name (raises clearly if an ingredient is missing)."""
    return mat[:, _IDX[name]]


def _state(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """-1/0/+1 from mutually-exclusive buy/sell boolean masks."""
    return np.where(buy, 1.0, np.where(sell, -1.0, 0.0)).astype(np.float32)


def compute_law_states(matrix: np.ndarray) -> np.ndarray:
    """Compute all 12 law/gate states for a (T, P) feature matrix (or a (P,) row).

    Returns (T, 12) float32 in LAW_NAMES order. Pure function of the market features,
    so it is precomputable offline and identical across seeds — exactly what makes
    law context reproducible for the Risk Doctor.
    """
    mat = np.atleast_2d(matrix).astype(np.float32)

    # ---- SUPER TREND ----
    # COUPLING [C1] -> market_pipeline/feature_builder/schema.py + feature_builder/builder.py:
    # every "boll_*"/"ssma_align_*"/"atr_dev_*"/"spread_range_ratio_*"/"adf_stat_*" string
    # below must exist (same spelling) in PRECOMPUTED_NAMES and be emitted by the builder,
    # else _c() raises KeyError on _IDX. Renaming a feature there breaks every law that reads it.
    # ST1 Bollinger: above BOTH outer upper bands (BB20+BB200) on 5m AND 30m -> buy.
    st1_buy = (_c(mat, "boll_bb20_up_5m") > 0) & (_c(mat, "boll_bb200_up_5m") > 0) \
        & (_c(mat, "boll_bb20_up_30m") > 0) & (_c(mat, "boll_bb200_up_30m") > 0)
    st1_sell = (_c(mat, "boll_bb20_lo_5m") < 0) & (_c(mat, "boll_bb200_lo_5m") < 0) \
        & (_c(mat, "boll_bb20_lo_30m") < 0) & (_c(mat, "boll_bb200_lo_30m") < 0)

    # ST2 CCI: all four CCIs (30,100 on 5m,30m) above applied SMA AND above +100.
    # CCI is RAW now: "above applied SMA" = raw cci > raw cci_sma; "above +100" = raw cci > 100.
    # COUPLING: column names mirror schema/builder; thresholds mirror SOW-D4 (±100).
    # COUPLING [C7] -> market_pipeline/feature_builder/schema.py + feature_builder/builder.py:
    # the cci{p}_{tf} / cci{p}_sma_{tf} name templates (periods 30,100 here; 10,100 in PB2)
    # must match schema._market_names / RAW_FEATURE_NAMES and what builder emits; curriculum
    # masks cci{p}_1m. Change a CCI period/name there => update these f-string templates.
    _cci_pairs = [(p, tf) for tf in ("5m", "30m") for p in (30, 100)]
    cci_above = [_c(mat, f"cci{p}_{tf}") > _c(mat, f"cci{p}_sma_{tf}") for (p, tf) in _cci_pairs]
    cci_below = [_c(mat, f"cci{p}_{tf}") < _c(mat, f"cci{p}_sma_{tf}") for (p, tf) in _cci_pairs]
    cci_hi = [_c(mat, f"cci{p}_{tf}") > _CCI_HI for (p, tf) in _cci_pairs]
    cci_lo = [_c(mat, f"cci{p}_{tf}") < _CCI_LO for (p, tf) in _cci_pairs]
    st2_buy = np.all(cci_above, axis=0) & np.all(cci_hi, axis=0)
    st2_sell = np.all(cci_below, axis=0) & np.all(cci_lo, axis=0)

    # ST3 Shifted SMA: price above both shifted lines on 1m, 5m AND 30m.
    st3_buy = (_c(mat, "ssma_align_1m") == 1) & (_c(mat, "ssma_align_5m") == 1) \
        & (_c(mat, "ssma_align_30m") == 1)
    st3_sell = (_c(mat, "ssma_align_1m") == -1) & (_c(mat, "ssma_align_5m") == -1) \
        & (_c(mat, "ssma_align_30m") == -1)

    # ---- TREND ----
    # T1 Bollinger: above MIDLINES of BB20+BB200 on 5m AND 30m.
    t1_buy = (_c(mat, "boll_bb20_mid_5m") > 0) & (_c(mat, "boll_bb200_mid_5m") > 0) \
        & (_c(mat, "boll_bb20_mid_30m") > 0) & (_c(mat, "boll_bb200_mid_30m") > 0)
    t1_sell = (_c(mat, "boll_bb20_mid_5m") < 0) & (_c(mat, "boll_bb200_mid_5m") < 0) \
        & (_c(mat, "boll_bb20_mid_30m") < 0) & (_c(mat, "boll_bb200_mid_30m") < 0)

    # T2 CCI: all four CCIs above their applied SMA (no +100 requirement).
    t2_buy = np.all(cci_above, axis=0)
    t2_sell = np.all(cci_below, axis=0)

    # T3 Shifted SMA: price above both lines on 5m AND 30m.
    t3_buy = (_c(mat, "ssma_align_5m") == 1) & (_c(mat, "ssma_align_30m") == 1)
    t3_sell = (_c(mat, "ssma_align_5m") == -1) & (_c(mat, "ssma_align_30m") == -1)

    # ---- PULL BACK ----
    # PB1 Bollinger: 30m above BOTH mids; 5m above BB200 mid but BELOW BB20 mid.
    pb1_buy = (_c(mat, "boll_bb20_mid_30m") > 0) & (_c(mat, "boll_bb200_mid_30m") > 0) \
        & (_c(mat, "boll_bb200_mid_5m") > 0) & (_c(mat, "boll_bb20_mid_5m") < 0)
    pb1_sell = (_c(mat, "boll_bb20_mid_30m") < 0) & (_c(mat, "boll_bb200_mid_30m") < 0) \
        & (_c(mat, "boll_bb200_mid_5m") < 0) & (_c(mat, "boll_bb20_mid_5m") > 0)

    # PB2 CCI: 30m both CCIs (10,100) above SMA; 5m LARGE(100) above while SMALL(10) below.
    # (RAW: "above/below SMA" = raw cci vs raw cci_sma.)
    def _cci_above(p, tf):
        return _c(mat, f"cci{p}_{tf}") > _c(mat, f"cci{p}_sma_{tf}")
    pb2_buy = _cci_above(10, "30m") & _cci_above(100, "30m") \
        & _cci_above(100, "5m") & ~_cci_above(10, "5m")
    pb2_sell = ~_cci_above(10, "30m") & ~_cci_above(100, "30m") \
        & ~_cci_above(100, "5m") & _cci_above(10, "5m")

    # PB3 Shifted SMA: 5m AND 30m above both lines; 1m BELOW both lines.
    pb3_buy = (_c(mat, "ssma_align_5m") == 1) & (_c(mat, "ssma_align_30m") == 1) \
        & (_c(mat, "ssma_align_1m") == -1)
    pb3_sell = (_c(mat, "ssma_align_5m") == -1) & (_c(mat, "ssma_align_30m") == -1) \
        & (_c(mat, "ssma_align_1m") == 1)

    # ---- MARKET-CONDITION OBSERVATION SIGNALS (0/1) — formerly "gates" ----
    # These are OBSERVATION-ONLY by default (config.TRAINING_PHASE == PHASE_FREE): the bot
    # SEES the condition and learns to use it instead of being hard-blocked. Only
    # market_stationarity_obs re-enforces, and only in PHASE_CONSTRAINED (engine.py).
    # Volatility: ATR expanding on BOTH the 5m execution frame AND the 4H macro regime.
    # (4H is an OBSERVATION INGREDIENT here, never a solo mask trigger — see the 4H rule.)
    # Companion raw magnitudes atr_level_5m / atr_level_4H are in the market block.
    market_volatility_obs = ((_c(mat, "atr_dev_5m") > 0) & (_c(mat, "atr_dev_4H") > 0)).astype(np.float32)
    # Spread: current spread < last 1m candle range (ratio < 1). Raw ATR-cost = spread_atr_1m.
    market_spread_obs = (_c(mat, "spread_range_ratio_1m") < 1.0).astype(np.float32)
    # Stationarity: 30-bar DF stat below the 5% critical value => stationary (1). Raw stat =
    # adf_stat_1m (in the market block). The ONLY signal that re-enforces in PHASE_CONSTRAINED.
    market_stationarity_obs = (_c(mat, "adf_stat_1m") < ADF_CRIT_5PCT).astype(np.float32)

    # COUPLING [C4] -> market_pipeline/law_mask_engine/engine.py + feature_builder/builder.py:
    # this stack ORDER (9 directional then volatility/spread/stationarity) IS the law block;
    # the mask engine slices law_states[:9]/[9:] + _OBS_IDX by position and the builder drops
    # these 12 columns into the schema "law" block. Reorder => desyncs both; keep == LAW_NAMES.
    states = np.stack([
        _state(st1_buy, st1_sell), _state(st2_buy, st2_sell), _state(st3_buy, st3_sell),
        _state(t1_buy, t1_sell), _state(t2_buy, t2_sell), _state(t3_buy, t3_sell),
        _state(pb1_buy, pb1_sell), _state(pb2_buy, pb2_sell), _state(pb3_buy, pb3_sell),
        market_volatility_obs, market_spread_obs, market_stationarity_obs,
    ], axis=1).astype(np.float32)

    return states[0] if matrix.ndim == 1 else states


def law_states_dict(row_states: np.ndarray) -> dict:
    """Name->value for one row's 12 states (telemetry + LLM Risk Doctor readability)."""
    # COUPLING [C8] -> diagnostics/telemetry_logger/logger.py + llm_risk_doctor/doctor.py:
    # the keys are LAW_NAMES (Term-2 law-context fields). The Risk Doctor reads these exact
    # names; renaming a law in schema._law_names changes these keys and breaks that contract.
    return {name: float(v) for name, v in zip(LAW_NAMES, row_states)}


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M3 — implemented the 9 laws + 3 gates.
#   I: The bot had no spine — nothing defined which directions were legal, so a
#      policy could open straight into the 4% wall.
#   R: THE_TRADING_CODE.md (9 laws + 3 gates, exact TFs/params, SOW-D4); laws are
#      masks never rewards; 4H observed-never-required.
#   A: Vectorized compute_law_states reading the M2 features -> -1/0/+1 directional
#      states + 0/1 gates, in schema law-block order; deterministic from price.
#   C: The legal space is now defined exactly per the blueprint, so the mask can
#      forbid breach-bound directions before the policy ever acts — the foundation
#      of not breaching, which is the foundation of passing.
# [2026-06-13] CCI laws read RAW CCI (operator un-normalized CCI).
#   I: ST2/T2/PB2 read cci_dev>0 and cci_norm>±1; those features became raw.
#   R: Operator CCI-raw decision; legal space must stay IDENTICAL (sign-preserving).
#   A: ST2/T2 read raw cci{p}_{tf} vs raw cci{p}_sma_{tf} (was cci_dev>0) + raw cci>±100
#      (was cci_norm>±1; _CCI_HI/_CCI_LO 1.0->100.0); PB2 likewise. Same comparisons.
#   C: The masks forbid exactly the same directions as before (Section F tests verify),
#      so the breach protection is unchanged while CCI is exposed raw.
# [2026-06-18] Gates -> market-condition OBSERVATION signals (operator/Perplexity redesign).
#   I: The 3 hard gates (ATR-liquidity, spread, stationarity) shut ~98.7% of opens on real
#      EURUSD, so the policy never traded enough to learn to pass — a calibration failure of
#      treating market context as a hard blocker instead of something the bot should LEARN.
#   R: Operator decision 2026-06-18 (Option 1 / phase-gated): the 3 signals become
#      observation-only; volatility re-reads 5m+4H (execution frame + macro regime); the DF
#      window shrinks 100->30 (faster, more responsive); enforcement is phase-gated and, this
#      session, only stationarity re-enforces in PHASE_CONSTRAINED (engine.py).
#   A: Renamed gate_* -> market_volatility_obs/market_spread_obs/market_stationarity_obs (same
#      law-block slots, 12-wide); volatility now (atr_dev_5m>0 & atr_dev_4H>0); GATES export
#      -> MARKET_SIGNALS; the directional laws + their masks are UNCHANGED.
#   C: In PHASE_FREE the bot can finally trade and learn which conditions help it pass FTMO,
#      while the stationarity guard can be re-armed late (PHASE_CONSTRAINED) once disciplined.
