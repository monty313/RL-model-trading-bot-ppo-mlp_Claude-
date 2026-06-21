"""Expert Signal Layer — SOFT, observation-only features distilled from the operator's
rule-based STRAT portfolio (the "Example trading strategies portfolio" PDF).

WHAT THIS MODULE DOES
---------------------
Turns the operator's discretionary/rule-based read of the market into a small block of
compact, bounded **observation features** the PPO policy can learn to weight. It is a
*feature engine*, NOT a policy and NOT a gate: it never chooses an action, never edits a
mask, never touches sizing/execution/reward. (Design: docs/EXPERT_SIGNAL_DESIGN.md.)

THE KEY DESIGN CHOICE — aggregate the existing laws, don't re-derive indicators
-----------------------------------------------------------------------------
The 9 directional laws already ARE the operator's strategies, in three flavours each:
  * STRAT-001 (Bollinger regime)   -> law_*_bb   (super_trend / trend / pullback)
  * STRAT-002 (dual/triple CCI)     -> law_*_cci
  * STRAT-004 (SMA stack)           -> law_*_ssma
and STRAT-006 (ADX/ATR "do-not-trade vs great-movement" filter) lives in the
``atr_dev_*`` / ``adx*`` ingredients + the ``market_volatility_obs`` signal.

So this engine *reuses* ``compute_law_states`` (the canonical, locked law computation) and
aggregates those 9 directional votes + the ADX/ATR ingredients into decisive summary
features. Reusing the laws (instead of recomputing BB/CCI/SMA) guarantees the expert read
stays IDENTICAL to what the masks already enforce, and keeps this layer tiny.

Scope this round (operator decision 2026-06-21): STRAT-001 / 002 / 004 / 006 only.
LTF = 5m, HTF = 30m. (STRAT-003 trinity needs CCI 14/900 which aren't in the matrix;
news/COT/opening-bell need external data — both deferred.)

THE 8 FEATURES (EXPERT_NAMES order)
-----------------------------------
Primary (information-rich):
  expert_regime_bias    {-1,0,+1}  net directional vote across the 9 laws (toggled, weighted)
  expert_confidence     [0,1]      net weighted conviction |long-short|/total (penalises conflict)
  expert_trend_strength [0,1]      clean alignment of the STRONG families (super_trend+trend)
                                   in the regime direction (distinct from pullback noise)
  expert_volatility_ok  [0,1]      soft "is it moving" = blend(LTF ATR expansion, HTF ADX rising)
  expert_session_ok     [0,1]      session-quality weight by hour (London/NY peak high)
Soft tradeability:
  expert_do_not_trade   [0,1]      soft chop/avoid score = 1 - confidence*volatility_ok
Derived convenience:
  expert_long           [0,1]      (regime_bias>0) * tradeability
  expert_short          [0,1]      (regime_bias<0) * tradeability

All features are bounded -> they belong in the NORMAL (clipped) observation path, NOT in
RAW_FEATURE_NAMES (no standardisation needed).

PURITY / WIRING
---------------
``compute_expert_signals(matrix)`` is a pure function of the PRECOMPUTED market matrix
(same input as ``compute_law_states``) -> it is action-independent and can be precomputed
once per bar (Phase 2). It accepts an optional precomputed ``law_states`` so the env/builder
can pass the laws it already computed instead of recomputing them.

🔴 LOCKED-ITEM SAFETY: this module reads the observation; it MUST NOT be wired into a mask
or gate. ``expert_do_not_trade`` is a FEATURE, never a blocker (a hard version of it was the
``CCI_REGIME_GATE`` that caused the always-HOLD collapse — see SESSION_HANDOFF arc #1).

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
These features are a deterministic function of the price-derived law states + ATR/ADX/time
columns, so they are identical across seeds. If an expert feature looks wrong, the break is
in its INGREDIENT (a law state or an ATR/ADX column), not here — check ``laws.py`` /
``builder.py`` first, exactly as for a law flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# COUPLING [C-EXP1] -> quantra/locked_core/laws/laws.py: we reuse compute_law_states (the
# canonical 9 laws + 3 signals). DIRECTIONAL_LAWS order (super_trend bb/cci/ssma, trend
# bb/cci/ssma, pullback bb/cci/ssma) is what _LAW_FAMILY / _LAW_STRATEGY below index by
# POSITION. Reordering LAW_NAMES there => fix the index groups here.
from quantra.locked_core.laws.laws import compute_law_states
# COUPLING [C-EXP1] -> quantra/market_pipeline/feature_builder/schema.py: PRECOMPUTED_NAMES
# defines the column index map (_IDX) we read atr_dev_*/adx*/time_* from. The names below
# must exist in PRECOMPUTED_NAMES (they are all in the `market` block). Rename there => fix here.
from quantra.market_pipeline.feature_builder.schema import PRECOMPUTED_NAMES

# ---------------------------------------------------------------------------
# Public block contract (consumed by the Phase-2 schema wiring).
# ---------------------------------------------------------------------------
EXPERT_NAMES = [
    "expert_regime_bias",
    "expert_confidence",
    "expert_trend_strength",
    "expert_volatility_ok",
    "expert_session_ok",
    "expert_do_not_trade",
    "expert_long",
    "expert_short",
]
EXPERT_DIM = len(EXPERT_NAMES)

# Column index of each precomputed feature name (same construction as laws._IDX).
_IDX = {name: i for i, name in enumerate(PRECOMPUTED_NAMES)}

# Positional groups within the 9 DIRECTIONAL law states (laws.DIRECTIONAL_LAWS order):
#   super_trend = (0,1,2)  trend = (3,4,5)  pullback = (6,7,8)
#   bb = (0,3,6)  cci = (1,4,7)  ssma = (2,5,8)
_SUPER, _TREND, _PULL = (0, 1, 2), (3, 4, 5), (6, 7, 8)
_LAW_BB, _LAW_CCI, _LAW_SSMA = (0, 3, 6), (1, 4, 7), (2, 5, 8)
_STRONG = _SUPER + _TREND                       # the 6 "strong" trend laws (no pullback)


@dataclass(frozen=True)
class ExpertSignalConfig:
    """Knobs for the expert layer. Frozen so a config is a reproducible fingerprint.

    Phase 2 mirrors these into the notebook HYPERPARAMETERS -> OVERRIDES (one-line
    revertible) and config.py defaults; for Phase 1 the defaults live here.
    """

    enabled: bool = True

    # --- per-STRATEGY toggles (zero a strategy's 3 directional-law contributions) ---
    use_bb: bool = True          # STRAT-001
    use_cci: bool = True         # STRAT-002
    use_ssma: bool = True        # STRAT-004
    use_adx_atr: bool = True     # STRAT-006 (volatility_ok)
    use_session: bool = True     # session-quality weight

    # --- family weights for the directional vote (super_trend strongest) ---
    w_super: float = 1.0
    w_trend: float = 0.7
    w_pullback: float = 0.5

    # --- STRAT-006 volatility/tradeability ---
    atr_tf: str = "5m"           # LTF execution frame (atr_dev_5m exists)
    macro_atr_tf: str = "4H"     # macro regime (atr_dev_4H exists) — observation context
    adx_tf: str = "30m"          # HTF — ADX is NOT computed on 5m (ADX_TFS=1m/30m/4H)
    atr_dev_scale: float = 0.25  # atr_dev at this level -> full ATR-expansion credit
    adx_gap_scale: float = 0.10  # (adx5-adx15) at this gap -> full ADX-rising credit
    w_atr: float = 0.5           # blend weight: LTF ATR expansion
    w_adx: float = 0.5           # blend weight: HTF ADX rising

    # --- session-quality weight by hour (of the data index; MT5 bars are GMT+2) ---
    # Each tuple is (start_hour, end_hour, weight); later ranges override earlier ones.
    # Default: London+NY core high, the overlap shoulders medium, the rest low.
    session_ranges: Tuple[Tuple[float, float, float], ...] = (
        (7.0, 16.0, 1.0),    # London + NY core
        (16.0, 21.0, 0.6),   # NY afternoon
    )
    session_default: float = 0.25   # outside any range (Asia / overnight)


DEFAULT_EXPERT_CONFIG = ExpertSignalConfig()


def _col(mat: np.ndarray, name: str) -> np.ndarray:
    """Column accessor by precomputed-feature name (raises clearly if missing)."""
    return mat[:, _IDX[name]]


def _law_weights(cfg: ExpertSignalConfig) -> np.ndarray:
    """Per-law weight vector (9,) = family weight * strategy-enabled flag."""
    fam = np.array(
        [cfg.w_super] * 3 + [cfg.w_trend] * 3 + [cfg.w_pullback] * 3, dtype=np.float64
    )
    enabled = np.ones(9, dtype=np.float64)
    if not cfg.use_bb:
        enabled[list(_LAW_BB)] = 0.0
    if not cfg.use_cci:
        enabled[list(_LAW_CCI)] = 0.0
    if not cfg.use_ssma:
        enabled[list(_LAW_SSMA)] = 0.0
    return fam * enabled


def _recover_hour(mat: np.ndarray) -> np.ndarray:
    """Recover hour-of-day [0,24) from the sin/cos time encoding."""
    s = _col(mat, "time_sin_hour")
    c = _col(mat, "time_cos_hour")
    ang = np.arctan2(s, c)                       # [-pi, pi]
    return np.mod(ang / (2.0 * np.pi) * 24.0, 24.0)


def _session_weight(mat: np.ndarray, cfg: ExpertSignalConfig) -> np.ndarray:
    """Vectorised session-quality weight in [0,1]."""
    if not cfg.use_session:
        return np.ones(mat.shape[0], dtype=np.float64)
    hour = _recover_hour(mat)
    w = np.full(mat.shape[0], cfg.session_default, dtype=np.float64)
    for (a, b, weight) in cfg.session_ranges:
        w = np.where((hour >= a) & (hour < b), weight, w)
    return np.clip(w, 0.0, 1.0)


def _volatility_ok(mat: np.ndarray, cfg: ExpertSignalConfig) -> np.ndarray:
    """STRAT-006 soft tradeability in [0,1]: blend LTF ATR expansion + HTF ADX rising.

    ATR-on-5m exists (LTF); ADX is NOT computed on 5m, so the ADX component reads the
    HTF (30m) by design — "is the 5m moving AND is the 30m trend strengthening".
    """
    if not cfg.use_adx_atr:
        return np.ones(mat.shape[0], dtype=np.float64)
    atr_dev = _col(mat, f"atr_dev_{cfg.atr_tf}")                  # >0 => ATR above its ref
    atr_comp = np.clip(atr_dev / max(cfg.atr_dev_scale, 1e-9), 0.0, 1.0)
    adx_gap = _col(mat, f"adx5_{cfg.adx_tf}") - _col(mat, f"adx15_{cfg.adx_tf}")
    adx_comp = np.clip(adx_gap / max(cfg.adx_gap_scale, 1e-9), 0.0, 1.0)
    wsum = cfg.w_atr + cfg.w_adx
    if wsum <= 0:
        return np.zeros(mat.shape[0], dtype=np.float64)
    return np.clip((cfg.w_atr * atr_comp + cfg.w_adx * adx_comp) / wsum, 0.0, 1.0)


def compute_expert_signals(
    matrix: np.ndarray,
    *,
    law_states: Optional[np.ndarray] = None,
    cfg: ExpertSignalConfig = DEFAULT_EXPERT_CONFIG,
) -> np.ndarray:
    """Compute the EXPERT_DIM expert features for a (T, P) precomputed matrix (or a (P,) row).

    Args:
        matrix: the PRECOMPUTED market matrix (width PRECOMPUTED_DIM), same input as
            ``compute_law_states`` — one row per bar, columns in PRECOMPUTED_NAMES order.
        law_states: optional precomputed (T,12)/(12,) law states (the env/builder already
            has these); if None they are computed here from ``matrix``.
        cfg: the ExpertSignalConfig (toggles + thresholds).

    Returns:
        (T, EXPERT_DIM) float32 in EXPERT_NAMES order, or (EXPERT_DIM,) for a 1-D input.
        Every value is bounded: regime_bias in {-1,0,+1}; all others in [0,1].
    """
    mat = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
    T = mat.shape[0]

    # --- 9 directional law votes (reuse the canonical, locked computation) ---
    if law_states is None:
        laws = compute_law_states(mat)                  # (T, 12)
    else:
        laws = np.atleast_2d(np.asarray(law_states, dtype=np.float32))
    directional = laws[:, :9]                            # (T, 9) each in {-1,0,+1}

    w = _law_weights(cfg)                                # (9,)
    wtot = float(w.sum())
    is_long = (directional > 0).astype(np.float64)       # (T,9)
    is_short = (directional < 0).astype(np.float64)
    long_score = is_long @ w                             # (T,)
    short_score = is_short @ w
    net = long_score - short_score

    if wtot > 0:
        regime_bias = np.sign(net)                       # {-1,0,+1}
        confidence = np.clip(np.abs(net) / wtot, 0.0, 1.0)
    else:                                                # all strategies toggled off
        regime_bias = np.zeros(T, dtype=np.float64)
        confidence = np.zeros(T, dtype=np.float64)

    # --- trend_strength: clean alignment of the STRONG families in the bias direction ---
    strong_idx = [i for i in _STRONG if w[i] > 0]        # respect strategy toggles
    if strong_idx:
        strong = directional[:, strong_idx]              # (T, n_strong)
        agree = (np.sign(strong) == regime_bias[:, None]) & (regime_bias[:, None] != 0)
        trend_strength = agree.sum(axis=1) / float(len(strong_idx))
    else:
        trend_strength = np.zeros(T, dtype=np.float64)
    trend_strength = np.clip(trend_strength, 0.0, 1.0)

    # --- STRAT-006 volatility + session + soft tradeability ---
    volatility_ok = _volatility_ok(mat, cfg)
    session_ok = _session_weight(mat, cfg)
    tradeability = confidence * volatility_ok            # both must be high
    do_not_trade = np.clip(1.0 - tradeability, 0.0, 1.0)

    # --- derived convenience long/short-ness ---
    expert_long = (regime_bias > 0).astype(np.float64) * tradeability
    expert_short = (regime_bias < 0).astype(np.float64) * tradeability

    out = np.stack([
        regime_bias,
        confidence,
        trend_strength,
        volatility_ok,
        session_ok,
        do_not_trade,
        expert_long,
        expert_short,
    ], axis=1).astype(np.float32)

    return out[0] if np.asarray(matrix).ndim == 1 else out


def expert_signals_dict(row: np.ndarray) -> dict:
    """Name->value for one row's expert features (telemetry / Risk-Doctor readability)."""
    return {name: float(v) for name, v in zip(EXPERT_NAMES, np.asarray(row).ravel())}


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule. I/R/A/C; Conclusion is always why this helps the
# bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-21] Phase 1 — pure ExpertSignalGenerator (no wiring yet).
#   I: The operator can express an edge as rules (the STRAT portfolio) and wants the bot to
#      SEE that read. The bot's perception had the raw ingredients + the 9 laws, but no
#      compact, decisive expert summary — so the policy had to re-derive "what's my read"
#      from scratch via reward trial-and-error (slow; part of the "no common sense" gap).
#   R: docs/EXPERT_SIGNAL_DESIGN.md — observation-features door (NOT mask/BC); additive;
#      reuse the locked laws (don't re-derive indicators); do_not_trade stays a SOFT feature;
#      scope STRAT-001/002/004/006; LTF 5m / HTF 30m; operator convention (config + OVERRIDES).
#   A: New package quantra/market_pipeline/expert_signal. compute_expert_signals aggregates
#      compute_law_states' 9 directional votes (family-weighted, per-STRAT togglable) into
#      regime_bias/confidence/trend_strength, blends atr_dev_5m + adx(30m) into volatility_ok,
#      derives session_ok from the time encoding, and a SOFT do_not_trade = 1-conf*vol +
#      derived expert_long/short. Pure function of the precomputed matrix; optional law_states
#      passthrough for efficient wiring. Bounded outputs (-> normal clipped path, not RAW).
#   C: Phase 1 is self-contained (touches no existing file, tests stay green). When wired
#      (Phase 2) it gives the policy a compact, locked-consistent expert read it can learn to
#      weight — eyes now, with the reward still owning survival — without the always-HOLD gate
#      hazard. COUPLING (both directions): reads laws.compute_law_states (positional law
#      groups) + schema.PRECOMPUTED_NAMES (atr_dev_*/adx*/time_* columns); Phase 2 will add an
#      "expert" block to schema._BLOCK_BUILDERS + EXPECTED_WIDTHS and emit it from the builder.
