"""LawMask engine — law states -> action mask (logit -1e9 on forbidden). 🔴

WHAT THIS MODULE DOES
---------------------
Turns the 12 law/gate states (laws.py) plus the current position + slot occupancy
into the additive direction mask over {HOLD, OPEN_LONG, OPEN_SHORT, CLOSE} and the
pointer mask over the 5 slots. Forbidden actions get -1e9 so the policy can never
sample them. Supports BOTH enforcement modes:

  LIVE  — laws BAN directions (an active buy-law forbids OPEN_SHORT, etc.). The 3 market-
          condition signals are observation-only by default (PHASE_FREE); in PHASE_CONSTRAINED
          the stationarity signal bans NEW opens when non-stationary. Everything else is legal.
  SCHOOL— curriculum permission mode: OPEN is allowed ONLY in the direction the
          stage's required law(s) currently permit, and only when active.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
This is the hard gate that mechanically prevents directional breaches: the policy
cannot trade against the trend a law defines, and cannot open in dead/illiquid/
wrong-regime conditions. The SAME masks run in training and live, so the discipline
the bot learns transfers to passing real challenges. HOLD is never masked, so a legal
action always exists.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Always interpret the action
distribution against this mask (Term 4): a 0.95 probability is meaningless if it was
the only legal option. Mask Dependence = the pre-mask logits repeatedly favor an
action this engine forbids; that's an actor problem, not a mask problem. The mask
itself is correct by construction (tested). Never modify masks/sizing/walls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

# COUPLING [C4] -> quantra/locked_core/laws/laws.py: MARKET_SIGNALS + LAW_NAMES (12 names, 9
# directional then 3 market-condition signals) and compute_law_states are imported here;
# _OBS_IDX/_LAW_IDX below and the [:9] dir-slice assume that order. Reorder/rename => break this.
from quantra.locked_core.laws.laws import MARKET_SIGNALS, LAW_NAMES, compute_law_states
# config.TRAINING_PHASE drives whether the market-condition signals re-enforce (phase-gated).
from quantra.runtime import config as cfg

# Direction action indices. COUPLING [C2 in COUPLINGS.md]: these integer meanings are
# assumed by ppo_agent.agent (direction head + OPEN/CLOSE gating), runtime.device
# (RepresentativePolicy), env.trading_env._apply_action, and live_bridge.live_session.
# Reorder here => the agent opens when it means to close. Change in ALL or none.
HOLD, OPEN_LONG, OPEN_SHORT, CLOSE = 0, 1, 2, 3
N_DIR_ACTIONS = 4            # COUPLING: == ppo_agent direction head width + device mirror
N_SLOTS = 5                 # COUPLING [C3]: re-exported from schema; pointer head width
NEG = -1e9  # SOW C5: forbidden actions get logit = -1e9 before sampling

# Position encodings.
FLAT, LONG, SHORT = 0, 1, -1

# Enforcement modes (THE_TRADING_CODE.md two-mode rule).
MODE_LIVE = "live"
MODE_SCHOOL = "school"

# COUPLING [C4] -> quantra/locked_core/laws/laws.py: the hardcoded offset 9 assumes
# laws.LAW_NAMES is exactly 9 directional laws followed by the 3 MARKET_SIGNALS; this
# mirrors schema._law_names order. If laws.py changes the count/order, fix the 9 here.
_OBS_IDX = {name: 9 + i for i, name in enumerate(MARKET_SIGNALS)}  # signals are the last 3
_LAW_IDX = {name: i for i, name in enumerate(LAW_NAMES)}


@dataclass
class MaskResult:
    """Per-step mask + the inputs that produced it (logged for the Risk Doctor)."""

    # COUPLING [C2/C3] -> quantra/env/trading_env.py + quantra/ppo_agent/agent.py: these
    # field names + array widths (direction (4,) per N_DIR_ACTIONS, pointer (5,) per
    # N_SLOTS, law_states (12,)) are unpacked by the env step and added to the agent's
    # head logits; rename a field or change a width => fix the consumers + telemetry.
    direction_mask: np.ndarray   # (4,) additive {0, -1e9}
    pointer_mask: np.ndarray     # (5,) additive {0, -1e9}
    law_states: np.ndarray       # (12,) — 9 directional + 3 market-condition signals
    opens_allowed: bool          # whether the phase-gated market signals currently permit NEW opens


def _opens_allowed(states: np.ndarray) -> bool:
    """Do the (phase-gated) market-condition signals permit NEW opens right now?

    PHASE_FREE (default): always True — the 3 signals are OBSERVATION-ONLY (the bot sees
    them in the state and learns to use them; nothing is hard-blocked). This is what lets
    the policy trade enough to learn to pass FTMO (the old hard gates shut ~98.7% of opens).
    PHASE_CONSTRAINED: the stationarity signal re-enforces — opens blocked when the market is
    non-stationary. Volatility + spread enforcement are deferred (see the TODO at file end).
    """
    if cfg.TRAINING_PHASE >= cfg.PHASE_CONSTRAINED:
        return bool(states[_OBS_IDX["market_stationarity_obs"]] == 1)
    return True


def _apply_training_wheels(mask: np.ndarray, wheel_states: Optional[Sequence[float]]) -> None:
    """Operator-directed semi-permanent counter-trend OPEN blocks (mutates `mask`).

    wheel_states = (tw_cci_block, tw_bb_block), each +1 (uptrend) / 0 / -1 (downtrend).
    The two wheels are INDEPENDENT — each bans on its own (union with the rest):
      +1 (uptrend context)  -> forbid OPEN_SHORT (no selling into the 30m+4H uptrend)
      -1 (downtrend context)-> forbid OPEN_LONG  (no buying into the 30m+4H downtrend)
    HOLD/CLOSE are never touched here, and these never RE-enable an already-forbidden
    action (mask is additive -1e9). 4H is used by explicit operator override of the
    locked 4H-observation-only rule; these wheels are isolated from the 9 locked laws
    and gated by config.TRAINING_WHEELS so they can be removed. COUPLING [C9].
    """
    if wheel_states is None:
        return
    for s in wheel_states:                 # each flag bans independently
        if s == 1:
            mask[OPEN_SHORT] = NEG
        elif s == -1:
            mask[OPEN_LONG] = NEG


def build_direction_mask(
    law_states: np.ndarray,
    position: int,
    n_open: int,
    mode: str = MODE_LIVE,
    required_laws: Optional[Sequence[str]] = None,
    stationarity_mode: str = "A",
    training_wheels: bool = False,
    wheel_states: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Additive direction mask (4,) of {0, -1e9}. HOLD is never masked.

    Order of restriction: base position legality -> slot limits -> market-condition signals
    (phase-gated open-block) -> directional laws (LIVE ban / SCHOOL permission) -> training
    wheels (operator counter-trend OPEN blocks). This mirrors SOW §2.3-2.4 + THE_TRADING_CODE.md;
    the market signals only narrow the set in PHASE_CONSTRAINED (observation-only otherwise),
    and the wheels are an additive operator override applied last (only REMOVE, never re-open).
    """
    mask = np.zeros(N_DIR_ACTIONS, dtype=np.float32)

    # 1) Base position legality (SOW §2.3 action table).
    if position == FLAT:
        mask[CLOSE] = NEG                       # nothing to close
    elif position == LONG:
        mask[OPEN_SHORT] = NEG                  # CLOSE before reversing (no auto-reverse)
    elif position == SHORT:
        mask[OPEN_LONG] = NEG

    # 2) Slot mechanics (SOW §2.4): OPEN masked when all 5 full; CLOSE masked when 0 open.
    if n_open >= N_SLOTS:
        mask[OPEN_LONG] = NEG
        mask[OPEN_SHORT] = NEG
    if n_open <= 0:
        mask[CLOSE] = NEG

    # 3) Market-condition signals (formerly "gates"): PHASE-GATED. In PHASE_FREE (default)
    #    they NEVER block — they are observation-only. In PHASE_CONSTRAINED the stationarity
    #    signal forbids NEW opens when non-stationary (open-position management is unaffected).
    #    `stationarity_mode` is still accepted for call-site compatibility but no longer drives
    #    enforcement — config.TRAINING_PHASE does (see _opens_allowed + the TODO at file end).
    if not _opens_allowed(law_states):
        mask[OPEN_LONG] = NEG
        mask[OPEN_SHORT] = NEG

    # 4) Directional laws.
    dir_states = law_states[:9]
    if mode == MODE_LIVE:
        # Each active law bans the OPPOSITE direction; bans accumulate (union).
        if np.any(dir_states == 1):
            mask[OPEN_SHORT] = NEG
        if np.any(dir_states == -1):
            mask[OPEN_LONG] = NEG
    elif mode == MODE_SCHOOL:
        # Permission mode: open ONLY in the direction the required law(s) allow, and
        # only when active. No required law active (or conflict) -> no open permission.
        req = required_laws or DIRECTIONAL_DEFAULT
        permitted = set()
        for name in req:
            s = law_states[_LAW_IDX[name]]
            if s == 1:
                permitted.add(LONG)
            elif s == -1:
                permitted.add(SHORT)
        if permitted != {LONG}:
            mask[OPEN_LONG] = NEG
        if permitted != {SHORT}:
            mask[OPEN_SHORT] = NEG
    else:
        raise ValueError(f"unknown enforcement mode: {mode!r}")

    # 5) Training wheels (operator override): counter-trend OPEN blocks, applied last.
    if training_wheels:
        _apply_training_wheels(mask, wheel_states)

    mask[HOLD] = 0.0  # HOLD is always legal — a legal action always exists.
    return mask


def build_pointer_mask(occupied: Sequence[float]) -> np.ndarray:
    """Additive pointer mask (5,): legal only on OCCUPIED slots (the CLOSE targets)."""
    occ = np.asarray(occupied, dtype=np.float32).ravel()
    mask = np.where(occ > 0.5, 0.0, NEG).astype(np.float32)
    return mask


# Default "required laws" if a school stage doesn't specify (all 9 directional laws).
# COUPLING [C4] -> quantra/locked_core/laws/laws.py + quantra/learning_system/curriculum_manager/curriculum.py:
# the [:9] slice assumes the first 9 LAW_NAMES are the directional laws; curriculum
# passes required_laws by these exact law-name strings into MODE_SCHOOL masking.
DIRECTIONAL_DEFAULT: List[str] = list(LAW_NAMES[:9])


class LawMask:
    """Convenience wrapper: market features -> law states -> masks (live or school)."""

    def __init__(self, mode: str = MODE_LIVE, required_laws: Optional[Sequence[str]] = None,
                 stationarity_mode: str = "A"):
        self.mode = mode
        self.required_laws = list(required_laws) if required_laws else None
        self.stationarity_mode = stationarity_mode

    def step(self, market_row: np.ndarray, position: int, occupied: Sequence[float]) -> MaskResult:
        """Compute the full mask for one symbol at one bar."""
        # COUPLING [C1] -> quantra/locked_core/laws/laws.py + feature_builder/schema.py:
        # market_row must be in PRECOMPUTED_NAMES order — laws._IDX indexes it by feature name.
        # If schema reorders the precomputed block, laws._IDX (and this caller's row) must match.
        states = compute_law_states(np.asarray(market_row, dtype=np.float32))
        n_open = int(np.sum(np.asarray(occupied) > 0.5))
        dmask = build_direction_mask(
            states, position, n_open, self.mode, self.required_laws, self.stationarity_mode
        )
        pmask = build_pointer_mask(occupied)
        return MaskResult(dmask, pmask, states, _opens_allowed(states))


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M3 — implemented the action-mask engine (both enforcement modes).
#   I: Law states existed but nothing converted them into the -1e9 action mask, and
#      nothing enforced the live-ban vs law-school-permission distinction.
#   R: SOW C5 (logit -1e9), §2.3-2.4 (position/slot legality), THE_TRADING_CODE.md
#      two-mode rule; HOLD always legal; gates ban new opens only.
#   A: build_direction_mask (base legality -> slots -> gates -> laws, live or school)
#      + build_pointer_mask (occupied slots only) + LawMask wrapper.
#   C: Breach-bound and out-of-context directions are now mechanically impossible to
#      sample, in both training and live, so the discipline transfers and the bot
#      stops losing challenges to directional/regime mistakes.
# [2026-06-15] Operator override — training wheels (counter-trend OPEN blocks).
#   I: The operator wants semi-permanent "training wheels" forbidding opens against a
#      strong 30m+4H trend (2 CCI 5/15 above/below SMA20; price above/below BB 10/100
#      dev0.5), to stop the bot wasting episodes opening into breach-bound trends.
#   R: Operator decision 2026-06-15. Additive + applied LAST (only removes options,
#      never re-opens); the two wheels ban INDEPENDENTLY; isolated from the locked 9
#      laws (their 4H-observation-only invariant is untouched — wheels are a separate
#      operator override that does read 4H); gated by config.TRAINING_WHEELS.
#   A: build_direction_mask gained training_wheels + wheel_states; _apply_training_wheels
#      bans OPEN_SHORT on +1 (uptrend) / OPEN_LONG on -1 (downtrend) per flag.
#   C: While the wheels are on, the policy cannot open counter-trend on the 30m+4H
#      context, so fewer breaches per window and faster convergence to a passing brain —
#      with HOLD always legal and the wheels removable once discipline is learned.
# [2026-06-18] Gates -> phase-gated market-condition signals (operator/Perplexity redesign).
#   I: The 3 hard gates shut ~98.7% of opens on real EURUSD, so the policy never traded
#      enough to learn to pass. Market context should be LEARNED, not hard-blocked.
#   R: Operator decision 2026-06-18 (phase-gated): the 3 signals (now market_volatility_obs/
#      market_spread_obs/market_stationarity_obs) are observation-only in PHASE_FREE (default);
#      only stationarity re-enforces, and only in PHASE_CONSTRAINED, this session.
#   A: Replaced _gates_allow_opens(states, stationarity_mode) with _opens_allowed(states)
#      keyed off cfg.TRAINING_PHASE; renamed GATES->MARKET_SIGNALS, _GATE_IDX->_OBS_IDX,
#      MaskResult.opens_allowed_by_gates->opens_allowed. Directional-law/slot/position/wheels
#      masking is UNCHANGED. stationarity_mode kept in signatures for call-site compatibility.
#   C: In PHASE_FREE the policy can finally trade and learn which conditions help it pass
#      FTMO; the stationarity guard can be re-armed late (PHASE_CONSTRAINED) once disciplined.

# ============================================================
# TODO — FUTURE SESSION: Curriculum Reorganization
# Three phase-gated enforcement blocks are planned but deferred:
#   1. market_stationarity_obs  <- DONE (this session, CONSTRAINED phase)
#   2. market_volatility_obs    <- needs its own dedicated training session
#   3. market_spread_obs        <- needs its own dedicated training session
#
# When ready: add each signal to the CONSTRAINED open-block in _opens_allowed(),
# one session at a time, verify full suite green before proceeding.
# Also: reorganize and document the full curriculum stage progression
# in a single clean pass once all three signals are enforced.
# ============================================================
