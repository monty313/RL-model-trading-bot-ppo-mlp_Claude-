"""Resample 1m bars to 5m / 30m / 4H — completed-bar-only, lookahead-safe.

WHAT THIS MODULE DOES
---------------------
Aggregates the clean 1m OHLCV(+spread) series into the higher timeframes the laws
read (5m, 30m, 4H). Each higher-TF bar is indexed by its **close time** (right
edge), and OHLCV is aggregated open=first / high=max / low=min / close=last /
tick_volume=sum / spread=last. ``as_of_higher_tf`` then aligns a higher-TF series
to the 1m clock using a backward ``merge_asof`` so that at any 1m timestamp the
policy only ever sees the most recent higher-TF bar that has ALREADY CLOSED.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Lookahead is the most dangerous bug in a trading RL system: a bot that peeks at an
unfinished 5m/30m/4H bar learns a fantasy edge that vanishes live and walks it into
the 4% wall. Indexing by close time + backward as-of merge makes peeking
structurally impossible, so the structure the laws read in training is exactly the
structure available live — which is what makes the learned pass-behaviour transfer.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. If you suspect "Shortcut
Learning" or an implausibly clean equity curve, verify here first that higher-TF
features were as-of-merged (closed bars only). A leakage artifact masquerades as a
learned edge; this module is the guard. 4H is OBSERVATION ONLY and never a law
trigger (4H Observation Rule) — its presence here is context, not permission.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

# Pandas offset aliases for the higher timeframes the laws/observation use.
# 1m is the base/decision frame (PPO acts every 1m bar, SOW C4/B4).
# COUPLING [C1] -> feature_builder/builder.py (iterates ("5m","30m","4H") + frames[tf]) and feature_builder/schema.py:
# these TF keys become the _5m/_30m/_4H feature-name suffixes; adding/renaming a TF changes FEATURE_NAMES width and all C1 consumers.
TIMEFRAMES: Dict[str, str] = {"5m": "5min", "30m": "30min", "4H": "4h"}

# COUPLING [C5] -> data_loader/loader.py: these aggregation keys must match OHLCV_COLUMNS
# exactly; a column the loader emits but _AGG omits is silently dropped from every higher-TF bar.
_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "tick_volume": "sum",
    "spread": "last",
}


def resample_ohlcv(df_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate a 1m frame (indexed by bar OPEN time) to ``freq`` higher-TF bars.

    Uses ``closed='left', label='right'`` so each output bar is stamped at its
    CLOSE time. Example: the 5m bin [11:30, 11:35) (the 1m bars opening 11:30..11:34)
    is labelled 11:35 — the instant that bar completes and may legally be read.
    Empty bins (weekend gaps, holidays) are dropped so the series stays real.
    """
    out = (
        df_1m.resample(freq, closed="left", label="right")
        .agg(_AGG)
        .dropna(subset=["close"])
    )
    out.index.name = "time"
    return out


def build_all_timeframes(df_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return {"1m": df_1m, "5m": ..., "30m": ..., "4H": ...} for one symbol.

    The 1m frame is passed through unchanged (it is the decision frame); the rest
    are close-time-stamped higher-TF aggregations ready for ``as_of_higher_tf``.
    """
    # COUPLING [C1] -> feature_builder/builder.py: caller reads frames["1m"] and frames[tf]
    # for tf in ("5m","30m","4H"); these dict keys must stay identical to TIMEFRAMES + "1m" or the builder KeyErrors.
    frames: Dict[str, pd.DataFrame] = {"1m": df_1m}
    for name, freq in TIMEFRAMES.items():
        frames[name] = resample_ohlcv(df_1m, freq)
    return frames


def as_of_higher_tf(
    base_index: pd.DatetimeIndex,
    higher: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    """Align a higher-TF frame onto the 1m ``base_index`` with NO lookahead.

    For each 1m timestamp t, attaches the higher-TF bar whose CLOSE time is the
    latest <= t (backward ``merge_asof``). Columns are suffixed (e.g. ``close_5m``)
    so the feature builder can stack timeframes without collisions. This is the
    function that operationally guarantees the bot never sees an unfinished bar.
    """
    base = pd.DataFrame(index=base_index)
    base.index.name = "time"
    # COUPLING [C1] -> feature_builder/builder.py + feature_builder/schema.py: add_suffix(f"_{suffix}")
    # builds the "_5m"/"_30m"/"_4H" column names the builder selects and schema lists in FEATURE_NAMES; the suffix format is load-bearing.
    merged = pd.merge_asof(
        base.reset_index(),
        higher.add_suffix(f"_{suffix}").reset_index(),
        on="time",
        direction="backward",
    ).set_index("time")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Completed-bar-only higher TFs (no lookahead).
#   I: Peeking at an unfinished 5m/30m/4H bar teaches a fantasy edge that vanishes live and breaches.
#   R: Completed-bar-only: close-time stamping + backward merge_asof; 4H is observation-only.
#   A: resample closed='left'/label='right'; as_of_higher_tf backward merge; no-lookahead test in Section C.
#   C: Training structure == live structure, so the edge transfers and the bot keeps passing across windows.
