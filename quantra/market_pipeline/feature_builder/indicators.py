"""Vectorized technical indicators with the EXACT blueprint parameters.

WHAT THIS MODULE DOES
---------------------
Pure, lookahead-safe numpy/pandas implementations of every indicator the laws and
observation use, with the locked parameters (SOW-D4 / STATE_VECTOR.md):
  * Bollinger BB20 + BB200, deviation 1
  * CCI 10/30/100 with applied SMA period 2, shift 4
  * ATR14 with reference SMA period 4, shift 4
  * Shifted SMA period 4, shift 4 on HIGH and LOW
  * Z-scores, dual lookback 10 and 100
  * ADX 5 and ADX 15 (Wilder)
  * Candle structure

Every function consumes only past+current bars (rolling/shift never peeks forward),
so features computed on a closed bar are exactly what would have been available live.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
These are the law INGREDIENTS. If a parameter drifts (e.g. shift 4 -> shift 0) the
laws would fire on different bars than the spec, the masks would permit different
trades, and the bot would learn a different — untested — policy that may not pass.
Pinning the exact params here keeps the legal space identical to the blueprint, and
the no-lookahead construction keeps the learned edge transferable to live passing.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. If a law flag looks wrong in
telemetry, check the ingredient here first (Term 2 Law Context). A NaN/inf ingredient
(early warmup, flat market) is replaced with 0 downstream; a *constant* ingredient
across regimes points to a data problem, not an actor failure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Locked parameters (SOW-D4). Changing any of these changes the legal space and
# MUST go through Monty (🔴) — see THE_TRADING_CODE.md.
# COUPLING [C7] -> quantra/market_pipeline/feature_builder/builder.py + quantra/locked_core/laws/laws.py:
# builder._compute_tf_features reads these exact names (SHIFT, SSMA_PERIOD, ATR_PERIOD,
# ATR_REF_PERIOD, CCI_APPLIED_SMA, BB_FAST/BB_SLOW/BB_DEV, CCI_PERIODS) to emit features,
# and the laws' semantics assume the same values. Rename/change a param => fix builder + laws.
SHIFT = 4
SSMA_PERIOD = 4
ATR_PERIOD = 14
ATR_REF_PERIOD = 4
CCI_APPLIED_SMA = 2
BB_FAST, BB_SLOW, BB_DEV = 20, 200, 1.0
Z_LOOKBACKS = (10, 100)
ADX_PERIODS = (5, 15)
CCI_PERIODS = (10, 30, 100)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high, low, close, period: int = ATR_PERIOD) -> pd.Series:
    """ATR over `period` (simple mean of True Range) — the distance normalizer."""
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def bollinger(close: pd.Series, period: int, dev: float = BB_DEV):
    """Return (mid, upper, lower) Bollinger bands. mid = SMA(close, period)."""
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + dev * sd, mid - dev * sd


def cci(high, low, close, period: int) -> pd.Series:
    """Commodity Channel Index. CCI = (TP - SMA(TP)) / (0.015 * mean abs dev)."""
    tp = (high + low + close) / 3.0
    sma = tp.rolling(period, min_periods=period).mean()
    mad = (tp - sma).abs().rolling(period, min_periods=period).mean()
    out = (tp - sma) / (0.015 * mad)
    return out.replace([np.inf, -np.inf], np.nan)


def applied_sma_shift(series: pd.Series, period: int = CCI_APPLIED_SMA, shift: int = SHIFT) -> pd.Series:
    """SMA(period) then shift forward `shift` bars — the 'shifted reference' pattern.

    shift(+n) moves a past value to the current row, so this only ever uses bars
    <= t-shift: lookahead-safe by construction (critical — see module docstring).
    """
    return series.rolling(period, min_periods=period).mean().shift(shift)


def shifted_sma(price_line: pd.Series, period: int = SSMA_PERIOD, shift: int = SHIFT) -> pd.Series:
    """Shifted SMA of a price line (applied to HIGH or LOW), period 4, shift 4."""
    return price_line.rolling(period, min_periods=period).mean().shift(shift)


def zscore(close: pd.Series, lookback: int) -> pd.Series:
    """Rolling z-score: (close - mean) / std over `lookback` (pullback depth)."""
    m = close.rolling(lookback, min_periods=lookback).mean()
    s = close.rolling(lookback, min_periods=lookback).std(ddof=0)
    return ((close - m) / s).replace([np.inf, -np.inf], np.nan)


def adx(high, low, close, period: int) -> pd.Series:
    """Wilder's ADX over `period`, returned in [0, 100] (caller divides by 100).

    Trend-strength observation only (NOT a law, SOW-C6). Uses Wilder smoothing
    (ewm alpha=1/period) on +DM/-DM/TR and on DX.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)
    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=alpha, adjust=False).mean() / atr_w
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, adjust=False).mean() / atr_w
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=alpha, adjust=False).mean()


def candle_structure(o, h, l, c, atr_1m):
    """1m candle shape: return/ATR, range/ATR, upper-wick ratio, lower-wick ratio.

    Cheap rejection/expansion info the slower indicators smooth over — helps the
    bot time clean entries/exits inside the legal space without overstaying (which
    matters for not drifting toward the wall).
    """
    rng = (h - l).replace(0, np.nan)
    body_hi = pd.concat([o, c], axis=1).max(axis=1)
    body_lo = pd.concat([o, c], axis=1).min(axis=1)
    ret = (c - o) / atr_1m
    rng_atr = (h - l) / atr_1m
    uwick = (h - body_hi) / rng
    lwick = (body_lo - l) / rng
    return ret, rng_atr, uwick, lwick


# Dickey-Fuller 5% critical value (with constant). stat < this => reject unit root
# => stationary (mean-reverting). Used by the Stationarity Regime Gate (F4: p<0.05).
# COUPLING [C4] -> quantra/locked_core/laws/laws.py: imported there (gate_stationarity
# threshold compares adf_stat_1m < ADF_CRIT_5PCT). Change this value => gate fires on a
# different regime boundary; keep in sync with the F4 spec.
ADF_CRIT_5PCT = -2.86


def rolling_df_stat(close: pd.Series, window: int = 100) -> pd.Series:
    """Vectorized rolling Dickey-Fuller t-statistic (0-lag ADF) — stationarity proxy.

    Regresses dy = a + b*y_lag over each trailing `window` and returns b/SE(b). A
    more-negative stat means stronger mean reversion (stationary); a stat near/above
    0 means a random walk / trend. We use the 0-lag DF (vectorized via rolling
    moments) instead of statsmodels.adfuller per bar, which would be far too slow
    over ~1.8M bars/symbol — speed here is what keeps walk-forward affordable.
    Lookahead-safe: rolling/shift only look back.

    Serves FTMO passing: the Stationarity Regime Gate trades only in the regime a
    curriculum stage allows; a faithful, cheap regime signal keeps the bot from
    learning habits in the wrong regime (which would breach when the regime flips).
    """
    y = close.astype(float)
    ylag = y.shift(1)
    dy = y - ylag
    n = window
    sx = ylag.rolling(n, min_periods=n).sum()
    sy = dy.rolling(n, min_periods=n).sum()
    sxx = (ylag * ylag).rolling(n, min_periods=n).sum()
    sxy = (ylag * dy).rolling(n, min_periods=n).sum()
    syy = (dy * dy).rolling(n, min_periods=n).sum()
    ss_xx = sxx - sx * sx / n
    ss_xy = sxy - sx * sy / n
    ss_yy = syy - sy * sy / n
    b = ss_xy / ss_xx.replace(0, np.nan)
    ss_res = (ss_yy - b * ss_xy).clip(lower=0)
    se_b = np.sqrt((ss_res / (n - 2)) / ss_xx.replace(0, np.nan))
    return (b / se_b).replace([np.inf, -np.inf], np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change APPENDS a dated IRAC entry (newest last). Conclusion is ALWAYS why
# the change makes the bot pass FTMO more consistently with no bug/inefficiency.
# Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M2 — locked-parameter indicators.
#   I: Without a single home for the exact indicator math, params could drift and
#      the laws would fire on the wrong bars, changing the (untested) legal space.
#   R: SOW-D4 locks (BB20/200 dev1, CCI 10/30/100 appliedSMA(2,sh4), ATR14 ref(4,sh4),
#      shifted-SMA(4,sh4), z 10/100, ADX5/15) + lookahead-safety.
#   A: Implemented each indicator vectorized with the locked params; all use only
#      rolling/shift of past bars (no forward peek); inf->nan for clean downstream fill.
#   C: The legal space stays identical to the blueprint and transfers to live, so the
#      bot trains on — and keeps passing — the real challenge, not a drifted variant.
# [2026-06-13] M3 — added rolling Dickey-Fuller stat for the Stationarity gate.
#   I: The Stationarity Regime Gate (F4) needs a 100-bar ADF p<0.05 signal, but
#      statsmodels.adfuller per bar over ~1.8M bars/symbol is far too slow.
#   R: F4 (rolling ADF, 100-bar, p<0.05) + the offline-precompute / cheap-iteration design.
#   A: Vectorized 0-lag DF t-stat via rolling moments + ADF_CRIT_5PCT=-2.86.
#   C: The bot gets a faithful, cheap regime signal so it trades only in the allowed
#      regime per stage — avoiding habits that breach when the regime flips — while
#      keeping walk-forward affordable. Net: more validated seeds, steadier passing.
