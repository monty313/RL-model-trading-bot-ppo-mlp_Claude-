"""FeatureBuilder — assemble the ~145-scalar observation; precompute market block offline.

WHAT THIS MODULE DOES
---------------------
Turns clean 1m bars into the policy's observation:
  1. resample to 5m/30m/4H (lookahead-safe, M1),
  2. compute the locked indicators per timeframe (indicators.py),
  3. as-of merge higher-TF features onto the 1m clock (closed bars only),
  4. assemble the precomputed block in canonical schema order (normalized `market`
     89 + RAW `market_raw` 30 = 119 with raw inputs on),
  5. clean (inf/NaN -> 0); clip ONLY the normalized block (RAW levels pass through
     for the M5 agent to standardize),
  6. cache the result to a float32 memmap so the RL loop never recomputes it.

The action-DEPENDENT blocks (law flags, per-slot trade ×5, portfolio, account) are
filled at env-step time (M3/M4); ``assemble_state`` concatenates everything into the
full 146-vector in schema order.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The 89 market features are a pure function of price, so computing them ONCE offline
(not every step) is the single biggest training-speed win — it makes many
walk-forward windows × 7 seeds affordable (the user's cost mandate) and keeps the
hot loop to array-indexing + a microsecond MLP. Bounded, NaN-free inputs keep the
tiny 3×256 trunk stable so it can actually learn the breach-risk vs safe distinction
that passing depends on.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. The observation you analyse comes
from here. If hidden states can't separate regimes (Representation Collapse), check
``valid_from`` (warmup) and whether the market block is constant/zeroed for the
episode window before blaming the trunk. Features are clipped to ±CLIP; a feature
pinned at ±CLIP across an episode is a saturation signal worth flagging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from quantra.runtime import config as cfg
from quantra.market_pipeline.data_loader import load_symbol
from quantra.market_pipeline.resampler import build_all_timeframes

from quantra.runtime.config import INCLUDE_RAW_INPUTS

# COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py: builder is bound to
# schema's PRECOMPUTED_NAMES order (output column order), RAW_FEATURE_NAMES (clip-exempt set),
# RAW_SMA_TFS (which TFs emit raw_sma*), and STATE_DIM/PRECOMPUTED_DIM (width asserts).
# Any rename/reorder in schema must be reflected by the column emitters in _compute_tf_features.
from . import indicators as ind
from .schema import (
    MARKET_DIM,
    MARKET_NAMES,
    PRECOMPUTED_DIM,
    PRECOMPUTED_NAMES,
    RAW_FEATURE_NAMES,
    RAW_SMA_TFS,
    SCHEMA,
    STATE_DIM,
)

# Continuous features are clipped to +/- CLIP after normalization. Real ATR-scaled
# distances / z-scores live well inside this; the clip only kills warmup blow-ups
# and flat-market division spikes that would destabilise the small MLP.
CLIP = 10.0


def _flag_three_way(cond_pos: pd.Series, cond_neg: pd.Series) -> pd.Series:
    """Encode a directional ingredient as -1 / 0 / +1 (the regime-flag convention)."""
    out = pd.Series(0.0, index=cond_pos.index)
    out[cond_pos] = 1.0
    out[cond_neg] = -1.0
    return out


def _compute_tf_features(df: pd.DataFrame, tf: str, point_size: float = 1e-5) -> pd.DataFrame:
    """Compute every schema market feature that belongs to timeframe ``tf``.

    Returns a frame indexed like ``df`` with columns named exactly per schema
    (e.g. ``boll_bb20_mid_5m``), so assembly is a pure column-select — the naming
    alignment the telemetry contract relies on.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    # COUPLING -> quantra/market_pipeline/feature_builder/indicators.py: the locked params
    # (ATR_PERIOD, BB_FAST/BB_SLOW/BB_DEV, CCI_PERIODS, CCI_APPLIED_SMA, SHIFT, SSMA_PERIOD,
    # ATR_REF_PERIOD) consumed below define the legal space; laws.py reads the same ingredients.
    # Changing a param in indicators.py shifts which bars fire => retrain. Do not diverge.
    atr_tf = ind.atr(h, l, c, ind.ATR_PERIOD).replace(0, np.nan)
    out: Dict[str, pd.Series] = {}

    # --- Bollinger (5m/30m/4H): price-vs-band distance in ATR units ---
    if tf in ("5m", "30m", "4H"):
        bb20_mid, bb20_up, bb20_lo = ind.bollinger(c, ind.BB_FAST)
        bb200_mid, bb200_up, bb200_lo = ind.bollinger(c, ind.BB_SLOW)
        for base, line in [
            ("bb20_mid", bb20_mid), ("bb20_up", bb20_up), ("bb20_lo", bb20_lo),
            ("bb200_mid", bb200_mid), ("bb200_up", bb200_up), ("bb200_lo", bb200_lo),
        ]:
            out[f"boll_{base}_{tf}"] = (c - line) / atr_tf     # NORMALIZED distance
            # RAW band price level [operator 2026-06-15: keep both]. UNCLIPPED via
            # RAW_FEATURE_NAMES. COUPLING: name mirrors schema._market_names boll_*_raw_*.
            out[f"boll_{base}_raw_{tf}"] = line

    # --- CCI (1m/5m/30m/4H): RAW value + RAW shifted-forward SMA + sync flags ---
    # CCI is kept RAW [operator decision 2026-06-13]: NO /100, NO normalized deviation.
    # We expose the raw CCI and the raw shifted SMA(CCI,2,sh4) so the policy compares
    # current CCI to its smoothed location 4 bars ago itself.
    # COUPLING: these column names (cci{p}_{tf}, cci{p}_sma_{tf}) MUST match schema
    # ._market_names + RAW_FEATURE_NAMES; the CCI laws (laws.py) read them; curriculum
    # masks cci{p}_1m. Rename here => change all four.
    if tf in ("1m", "5m", "30m", "4H"):
        cci_vals, cci_smas = {}, {}
        for p in ind.CCI_PERIODS:
            cp = ind.cci(h, l, c, p)
            csma = ind.applied_sma_shift(cp, ind.CCI_APPLIED_SMA, ind.SHIFT)
            out[f"cci{p}_{tf}"] = cp            # RAW CCI value (unclipped via RAW_FEATURE_NAMES)
            out[f"cci{p}_sma_{tf}"] = csma       # RAW shifted-forward SMA(CCI, 2, shift 4)
            cci_vals[p], cci_smas[p] = cp, csma
        # Flags from the RAW CCI-vs-SMA comparison (sign-identical to the old (CCI-SMA)).
        all_pos = (cci_vals[10] > cci_smas[10]) & (cci_vals[30] > cci_smas[30]) & (cci_vals[100] > cci_smas[100])
        all_neg = (cci_vals[10] < cci_smas[10]) & (cci_vals[30] < cci_smas[30]) & (cci_vals[100] < cci_smas[100])
        out[f"cci_sync_{tf}"] = _flag_three_way(all_pos, all_neg)
        if tf == "5m":
            # Pull Back ingredient: large CCIs above their SMA while small (10) is below.
            buy = (cci_vals[30] > cci_smas[30]) & (cci_vals[100] > cci_smas[100]) & (cci_vals[10] < cci_smas[10])
            sell = (cci_vals[30] < cci_smas[30]) & (cci_vals[100] < cci_smas[100]) & (cci_vals[10] > cci_smas[10])
            out["cci_pullback_5m"] = _flag_three_way(buy, sell)

    # --- TRAINING-WHEEL ingredients (30m/4H) [operator 2026-06-15] ---
    # NEW additive configs (NOT the locked SOW-D4 params): CCI 5/15 with applied SMA
    # period 20 shift 0 (RAW value + SMA), and Bollinger 10/100 dev 0.5 upper/lower
    # distances in ATR units. These feed the two counter-trend block flags computed in
    # build_market_matrix. 4H is used here by operator override (training wheels only).
    # COUPLING [C9] -> indicators.WHEEL_* + schema._WHEEL_TFS + law_mask_engine (reads flags).
    if tf in ind.WHEEL_TFS:
        for p in ind.WHEEL_CCI_PERIODS:                 # (5, 15)
            cp = ind.cci(h, l, c, p)
            out[f"tw_cci{p}_{tf}"] = cp                  # RAW (unclipped via RAW_FEATURE_NAMES)
            out[f"tw_cci{p}_sma_{tf}"] = ind.applied_sma_shift(cp, ind.WHEEL_CCI_SMA, ind.WHEEL_CCI_SHIFT)
        for p in (ind.WHEEL_BB_FAST, ind.WHEEL_BB_SLOW):  # (10, 100)
            _mid, up, lo = ind.bollinger(c, p, ind.WHEEL_BB_DEV)
            out[f"tw_bb{p}_up_{tf}"] = (c - up) / atr_tf   # >0 => price above upper band
            out[f"tw_bb{p}_lo_{tf}"] = (c - lo) / atr_tf   # <0 => price below lower band

    # --- ATR regime (1m/30m/4H): level, shifted ref, normalized deviation ---
    if tf in ("1m", "30m", "4H"):
        atr_t = ind.atr(h, l, c, ind.ATR_PERIOD)
        baseline = atr_t.rolling(100, min_periods=20).mean().replace(0, np.nan)
        ref = atr_t.rolling(ind.ATR_REF_PERIOD).mean().shift(ind.SHIFT)
        out[f"atr_level_{tf}"] = atr_t / baseline
        out[f"atr_ref_{tf}"] = ref / baseline
        out[f"atr_dev_{tf}"] = (atr_t - ref) / ref.replace(0, np.nan)

    # --- Shifted SMA (1m/5m/30m/4H) on high/low + alignment flag ---
    if tf in ("1m", "5m", "30m", "4H"):
        ssma_h = ind.shifted_sma(h, ind.SSMA_PERIOD, ind.SHIFT)
        ssma_l = ind.shifted_sma(l, ind.SSMA_PERIOD, ind.SHIFT)
        out[f"ssma_high_dist_{tf}"] = (c - ssma_h) / atr_tf
        out[f"ssma_low_dist_{tf}"] = (c - ssma_l) / atr_tf
        out[f"ssma_align_{tf}"] = _flag_three_way((c > ssma_h) & (c > ssma_l),
                                                  (c < ssma_h) & (c < ssma_l))

    # --- Z-scores (1m/5m/30m/4H): dual lookback ---
    if tf in ("1m", "5m", "30m", "4H"):
        out[f"z10_{tf}"] = ind.zscore(c, 10)
        out[f"z100_{tf}"] = ind.zscore(c, 100)

    # --- ADX (1m/30m/4H): trend strength, normalized /100 ---
    if tf in ("1m", "30m", "4H"):
        out[f"adx5_{tf}"] = ind.adx(h, l, c, 5) / 100.0
        out[f"adx15_{tf}"] = ind.adx(h, l, c, 15) / 100.0

    # --- Candle structure + time context: 1m only ---
    if tf == "1m":
        ret, rng_atr, uwick, lwick = ind.candle_structure(o, h, l, c, atr_tf)
        out["candle_return_1m"] = ret
        out["candle_range_1m"] = rng_atr
        out["candle_uwick_1m"] = uwick
        out["candle_lwick_1m"] = lwick
        idx = df.index
        hour = idx.hour + idx.minute / 60.0
        out["time_sin_hour"] = pd.Series(np.sin(2 * np.pi * hour / 24.0), index=idx)
        out["time_cos_hour"] = pd.Series(np.cos(2 * np.pi * hour / 24.0), index=idx)
        out["time_dow"] = pd.Series((idx.dayofweek / 4.0) * 2.0 - 1.0, index=idx)
        # Gate ingredients [M3]. MT5 <SPREAD> is in POINTS; convert to price via
        # point_size to compare against ATR + candle range (Spread Filter). adf_stat
        # is the rolling Dickey-Fuller stat (Stationarity gate).
        spread_price = df["spread"].astype(float) * point_size
        out["spread_atr_1m"] = spread_price / atr_tf
        out["spread_range_ratio_1m"] = spread_price / (h - l).replace(0, np.nan)
        out["adf_stat_1m"] = ind.rolling_df_stat(c, 100)

    # --- RAW SMA inputs (operator override, 2026-06-13): UNNORMALIZED price-level
    # SMAs on 5m/30m/4H. SMA period 1 = price, so shifts 0-3 give a 4-tap price
    # ladder; sma30/sma50 are medium trend levels. These bypass the ±CLIP downstream
    # and MUST be standardized by the M5 agent (see RAW_INPUTS.md). Observation-only,
    # never law ingredients (laws stay on the locked SOW-D4 params). ---
    if INCLUDE_RAW_INPUTS and tf in RAW_SMA_TFS:
        out[f"raw_sma1_sh0_{tf}"] = c
        out[f"raw_sma1_sh1_{tf}"] = c.shift(1)
        out[f"raw_sma1_sh2_{tf}"] = c.shift(2)
        out[f"raw_sma1_sh3_{tf}"] = c.shift(3)
        out[f"raw_sma30_{tf}"] = c.rolling(30, min_periods=30).mean()
        out[f"raw_sma50_{tf}"] = c.rolling(50, min_periods=50).mean()

    return pd.DataFrame(out, index=df.index)


def _asof_onto(base_index: pd.DatetimeIndex, feat: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of merge a higher-TF feature frame onto the 1m index (no peek)."""
    base = pd.DataFrame(index=base_index)
    base.index.name = "time"
    merged = pd.merge_asof(
        base.reset_index(), feat.reset_index(), on="time", direction="backward"
    ).set_index("time")
    return merged


@dataclass
class MarketMatrix:
    """The precomputed market block + provenance for the env / telemetry."""

    # COUPLING [C1] -> quantra/env/trading_env.py + quantra/live_bridge/live_session.py:
    # these field names (matrix/index/valid_from/names) are read when the env/live session
    # slices a precomputed row; matrix width is PRECOMPUTED_DIM in PRECOMPUTED_NAMES order.
    matrix: np.ndarray          # (T, PRECOMPUTED_DIM) float32, precomputed-block order
    index: pd.DatetimeIndex     # 1m timestamps aligned to rows
    valid_from: int             # first row with non-4H normalized features ready
    names: list                 # PRECOMPUTED_NAMES (market + market_raw) for telemetry


def build_market_matrix(df_1m: pd.DataFrame, point_size: float = 1e-5) -> MarketMatrix:
    """Compute the precomputed (action-independent) feature matrix from 1m bars.

    Width = PRECOMPUTED_DIM (normalized `market` 92 + RAW `market_raw` 30 = 122 with
    raw inputs on). ``point_size`` converts MT5 spread points -> price for the Spread
    Filter ingredients. Pure (no IO): used by tests and by ``precompute_symbol``.
    Higher-TF features are as-of merged so row t only sees closed 5m/30m/4H bars.
    """
    frames = build_all_timeframes(df_1m)  # {1m,5m,30m,4H}
    cols = pd.DataFrame(index=df_1m.index)

    one_min = _compute_tf_features(frames["1m"], "1m", point_size=point_size)
    cols = cols.join(one_min)
    for tf in ("5m", "30m", "4H"):
        feat = _compute_tf_features(frames[tf], tf)
        cols = cols.join(_asof_onto(df_1m.index, feat))

    # TRAINING-WHEEL block flags [operator 2026-06-15]. Computed here (not per-TF)
    # because each flag aggregates BOTH 30m AND 4H, which only coexist on the 1m index
    # after the as-of merges above. +1 = uptrend context (blocks OPEN_SHORT later) when
    # the condition holds on BOTH timeframes; -1 = downtrend (blocks OPEN_LONG); else 0.
    # NaN during warmup => comparisons are False => flag 0 (safe: no wheel before warmup).
    # COUPLING [C9] -> law_mask_engine/engine.py reads tw_cci_block/tw_bb_block by name.
    cci_above = np.logical_and.reduce([
        (cols[f"tw_cci{p}_{tf}"] > cols[f"tw_cci{p}_sma_{tf}"]).to_numpy()
        for tf in ("30m", "4H") for p in (5, 15)])
    cci_below = np.logical_and.reduce([
        (cols[f"tw_cci{p}_{tf}"] < cols[f"tw_cci{p}_sma_{tf}"]).to_numpy()
        for tf in ("30m", "4H") for p in (5, 15)])
    cols["tw_cci_block"] = np.where(cci_above, 1.0, np.where(cci_below, -1.0, 0.0))
    # BB: "price above both BBs" = above the upper band of bb10 AND bb100 on both TFs.
    bb_above = np.logical_and.reduce([
        (cols[f"tw_bb{p}_up_{tf}"] > 0).to_numpy()
        for tf in ("30m", "4H") for p in (10, 100)])
    bb_below = np.logical_and.reduce([
        (cols[f"tw_bb{p}_lo_{tf}"] < 0).to_numpy()
        for tf in ("30m", "4H") for p in (10, 100)])
    cols["tw_bb_block"] = np.where(bb_above, 1.0, np.where(bb_below, -1.0, 0.0))

    # Order to the canonical precomputed-block schema; missing = coding error.
    missing = [n for n in PRECOMPUTED_NAMES if n not in cols.columns]
    if missing:
        raise RuntimeError(f"FeatureBuilder produced no values for: {missing}")
    ordered = cols[PRECOMPUTED_NAMES]

    # valid_from = first row where all NON-4H NORMALIZED features are ready (warmup
    # end). 4H is observation-only and may stay 0 during its long BB200 warmup, so we
    # don't make the env wait ~33 days for it; we wait only for the law-relevant
    # 1m/5m/30m normalized features. Keeps usable training bars per window high
    # (more pass/fail samples per seed). RAW features warm fast and aren't gated on.
    non_4h_norm = [n for n in MARKET_NAMES if not n.endswith("_4H")]
    valid_mask = ordered[non_4h_norm].notna().all(axis=1).to_numpy()
    valid_from = int(np.argmax(valid_mask)) if valid_mask.any() else len(ordered)

    # Clean: inf->NaN->0 for all; clip ONLY the normalized block. RAW features are
    # unbounded price/CCI levels and must NOT be clipped (the M5 agent standardizes
    # them) — clipping raw price to ±10 would destroy it.
    clean = ordered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    norm_cols = [n for n in PRECOMPUTED_NAMES if n not in RAW_FEATURE_NAMES]
    clean[norm_cols] = clean[norm_cols].clip(-CLIP, CLIP)
    matrix = clean[PRECOMPUTED_NAMES].to_numpy(dtype=np.float32)
    return MarketMatrix(matrix, df_1m.index, valid_from, list(PRECOMPUTED_NAMES))


def precompute_symbol(symbol: str, force: bool = False) -> MarketMatrix:
    """Build + memmap-cache the market block for ``symbol`` (Drive/local via M1).

    First call computes and writes ``data/features/{symbol}_market.npy`` (+ a
    timestamp sidecar); later calls memory-map it. This is the offline step that
    keeps training cheap enough to validate a real pass rate.
    """
    cfg.ensure_dirs()
    npy = cfg.FEATURE_CACHE_DIR / f"{symbol}_market.npy"
    meta = cfg.FEATURE_CACHE_DIR / f"{symbol}_market_index.parquet"
    if npy.exists() and meta.exists() and not force:
        mat = np.load(npy, mmap_mode="r")
        idx = pd.read_parquet(meta).index
        valid_from = int(np.argmax(np.abs(mat).sum(axis=1) > 0)) if len(mat) else 0
        return MarketMatrix(mat, pd.DatetimeIndex(idx), valid_from, list(PRECOMPUTED_NAMES))

    df_1m, _ = load_symbol(symbol)
    # COUPLING [C5] -> quantra/runtime/config.py: POINT_SIZE is a per-symbol dict keyed by
    # config.SYMBOLS; a symbol missing here silently falls back to DEFAULT_POINT_SIZE, which
    # would mis-scale the spread-gate ingredients. Add new symbols to config.POINT_SIZE.
    mm = build_market_matrix(df_1m, point_size=cfg.POINT_SIZE.get(symbol, cfg.DEFAULT_POINT_SIZE))
    np.save(npy, mm.matrix)
    pd.DataFrame(index=mm.index).to_parquet(meta)
    return mm


def assemble_state(
    precomputed_row: np.ndarray,
    law_flags: Optional[np.ndarray] = None,
    trade: Optional[np.ndarray] = None,
    portfolio: Optional[np.ndarray] = None,
    account: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate blocks into the full STATE_DIM vector in canonical schema order.

    ``precomputed_row`` is one row of build_market_matrix (the market + market_raw
    blocks, width PRECOMPUTED_DIM). The env (M4) appends the live law/trade/
    portfolio/account sub-vectors; any omitted block is zero-filled (M2 tests +
    warmup). Width is asserted == STATE_DIM so a block-size drift fails loudly
    rather than silently feeding the policy a malformed world.
    """
    pre = np.asarray(precomputed_row, dtype=np.float32).ravel()
    if pre.shape[0] != PRECOMPUTED_DIM:
        raise ValueError(f"precomputed_row expected {PRECOMPUTED_DIM} values, got {pre.shape[0]}")

    def _blk(name: str, vec: Optional[np.ndarray]) -> np.ndarray:
        width = SCHEMA.block_spans[name][1] - SCHEMA.block_spans[name][0]
        if vec is None:
            return np.zeros(width, dtype=np.float32)
        v = np.asarray(vec, dtype=np.float32).ravel()
        if v.shape[0] != width:
            raise ValueError(f"block '{name}' expected {width} values, got {v.shape[0]}")
        return v

    # COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py (_BLOCK_BUILDERS) +
    # quantra/env/trading_env.py (the caller that supplies law/trade/portfolio/account vecs):
    # this concat order MUST equal schema's block order, and each _blk width is read from
    # SCHEMA.block_spans. Reorder schema blocks => reorder here or the obs is scrambled.
    state = np.concatenate([
        pre,                       # market + market_raw (PRECOMPUTED_DIM)
        _blk("law", law_flags),
        _blk("trade", trade),
        _blk("portfolio", portfolio),
        _blk("account", account),
    ])
    assert state.shape[0] == STATE_DIM, f"assembled {state.shape[0]} != {STATE_DIM}"
    return state.astype(np.float32)


# Convenience re-exports for the env/agent/telemetry.
__all__ = [
    "MarketMatrix", "build_market_matrix", "precompute_symbol", "assemble_state",
    "PRECOMPUTED_NAMES", "PRECOMPUTED_DIM", "MARKET_NAMES", "MARKET_DIM",
    "RAW_FEATURE_NAMES", "STATE_DIM", "SCHEMA", "CLIP",
]


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change APPENDS a dated IRAC entry (newest last). Conclusion is ALWAYS why
# the change makes the bot pass FTMO more consistently with no bug/inefficiency.
# Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M2 — FeatureBuilder + offline market precompute.
#   I: Computing ~89 multi-TF indicators every RL step would make walk-forward
#      validation unaffordable, and any lookahead/NaN would teach a fantasy edge
#      or destabilise the tiny MLP — both wreck consistent passing.
#   R: STATE_VECTOR.md (~145, 4H observation copies) + offline-precompute design +
#      lookahead-safe as-of merge (M1) + bounded encodings.
#   A: Vectorized per-TF compute -> backward as-of merge -> schema-ordered 89-wide
#      matrix -> inf/NaN->0, clip ±10 -> float32 memmap cache; assemble_state()
#      concatenates the full 146 with asserted block widths.
#   C: The hot loop is just array indexing + a microsecond MLP on a faithful,
#      bounded, lookahead-free observation, so we can afford the seeds/windows that
#      prove a real, transferable FTMO pass rate.
# [2026-06-13] Operator override — compute the RAW SMA + RAW CCI block.
#   I: The operator wants raw (unnormalized) SMA + CCI levels added to the obs; raw
#      price clipped to ±10 would be destroyed, and unbounded inputs can destabilise
#      the small MLP or invite shortcut learning if fed naively.
#   R: Operator directive overrides the no-raw-price rule for `market_raw`; raw
#      features bypass ±CLIP and are flagged (RAW_FEATURE_NAMES) for M5 standardization.
#   A: Added raw SMA (5m/30m/4H) + raw CCI (CCI TFs); ordered by PRECOMPUTED_NAMES;
#      clip applied ONLY to normalized columns; assemble_state now takes the 119-wide
#      precomputed row; valid_from still keys off non-4H NORMALIZED features.
#   C: The policy gets the requested raw signal without corrupting it, while the raw
#      block stays isolated + flagged so we can standardize/ablate it — protecting
#      training stability and the pass rate if raw inputs turn out to hurt.
# [2026-06-13] Operator decision — CCI computed RAW (value + raw shifted SMA).
#   I: CCI was emitted normalized (cp/100, (cp-sma)/100); operator wants raw CCI + raw
#      shifted SMA(2,sh4). raw_cci in market_raw then duplicated it.
#   R: Operator override; flags recomputed from raw CCI-vs-SMA (sign-identical, laws unchanged).
#   A: Emit cci{p}_{tf}=cp, cci{p}_sma_{tf}=applied_sma_shift(cp); flags from raw compare;
#      dropped raw_cci (now == cci{p}); these CCI cols are in RAW_FEATURE_NAMES (unclipped).
#   C: The policy sees CCI's true magnitude + its 4-bar-ago smoothed location, the
#      operator's intended trend signal, with the legal space (laws) unchanged.
# [2026-06-15] Operator decision — emit RAW Bollinger band levels (keep both).
#   I: Only the normalized (close-band)/ATR was emitted; operator wants the raw band too.
#   R: Operator override; additive + observation-only (laws read the normalized sign).
#   A: In the Bollinger loop, also emit boll_{base}_raw_{tf} = the raw band price level
#      (unclipped via RAW_FEATURE_NAMES). 18 new features; market 92->110.
#   C: The policy gets the raw band positions/width alongside the ATR-scaled distance,
#      with the masks unchanged - richer volatility/structure signal, same legal space.
# [2026-06-15] Operator decision — emit training-wheel ingredients + block flags.
#   I: The new counter-trend "training wheels" need CCI 5/15 (SMA20 sh0) + BB 10/100
#      (dev0.5) on 30m/4H as ingredients, plus the two derived block flags the mask reads.
#   R: Operator decision 2026-06-15; reuse the locked-safe vectorized indicator fns with
#      the new WHEEL_* params; flags aggregate BOTH TFs (computed post as-of-merge).
#   A: _compute_tf_features emits tw_cci{5,15}(_sma) + tw_bb{10,100}_{up,lo} for 30m/4H;
#      build_market_matrix computes tw_cci_block / tw_bb_block (+1 up / -1 down / 0) from
#      them. RAW CCI cols bypass ±CLIP; normalized BB dists are clipped; flags are ±1/0.
#   C: The wheels' uptrend/downtrend context is now both observable (acts) and available
#      to the mask (laws) as a precomputed, cache-cheap signal - no hot-loop cost, so it
#      speeds convergence to passing without slowing training.
