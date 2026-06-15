"""RewardEngine — layered reward L0-L6 + QUAD, with Layer-0 dominance (E8). 🔴

WHAT THIS MODULE DOES
---------------------
Computes the layered reward (REWARD_DESIGN.md):
    r(t) = L0_dNetPnL + (a*L1_momentum - b*L2_stagnation + e*L4_target) * L5_category
           - d*L3_painzone   [+ L6 daily bonus at end of day]
Layer 0 (net PnL after costs) is the dominant driver; the shaping layers are tiny
"whispers" (small coefficients) that help timing/restraint without ever winning the
reward game while losing the trading game (the E8 rule). The QUAD daily bonus (E9)
sits on L6 with a hard 95%-of-day-PnL ceiling so it stays strictly < Layer 0.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
This is the objective the policy optimizes. Because L0 dominates, the bot is driven
to make real net money inside the legal/risk-safe space — not to game a shaper. L3's
exponential pain-zone ramp pushes it off the wall; L1/L2 sharpen entry/exit timing;
L6/QUAD reward consistent pass-days. Get this right and PPO learns to pass.

🔴 LOCKED: L0 dominance (E8), pain-zone exponential 3.5->4.0%, QUAD 95% ceiling.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md`` (Term 7 Reward Decomposition).
``decompose()`` returns every layer's contribution — if any single shaping layer's
cumulative magnitude exceeds Layer 0 over a window, that is a Reward Hijack; cite the
per-layer integral. The reward is the training signal's ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from quantra.runtime.config import ChallengeConfig

# Shaping coefficients — deliberately TINY vs typical per-step net PnL so Layer 0
# dominates (REWARD_DESIGN: "whisper, not a shout"). Units: account fraction.
ALPHA = 1e-4   # L1 momentum bonus
BETA = 1e-4    # L2 stagnation penalty
DELTA = 5e-4   # L3 pain-zone (slightly larger — it must be felt near the wall)
EPS = 5e-5     # L4 target progress (tiniest)
PAIN_K = 4.0   # exponential steepness of the pain ramp


@dataclass
class RewardContext:
    # COUPLING -> env/trading_env.py: _reward()/build-context constructs RewardContext by
    # these exact keyword field NAMES (net_pnl_delta, in_position, momentum_aligned,
    # stagnation, drawdown_pct, day_progress, breach_risk). Renaming a field breaks that call.
    """Everything the engine needs for one step — the env populates it."""

    net_pnl_delta: float           # L0: equity change after costs this step / account
    in_position: bool = False
    momentum_aligned: bool = False  # small CCI back in sync + ATR alive, in trade dir
    stagnation: bool = False        # favorable legal state, no improvement, 3x5m bars
    drawdown_pct: float = 0.0       # current DAILY drawdown % (for the pain zone)
    day_progress: float = 0.0       # day PnL vs target (1.0 = target hit)
    breach_risk: bool = False       # in pain zone / near wall (the explicit L5 category)


@dataclass
class RewardEngine:
    """Pure layered-reward computation. One instance per training run."""

    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)
    quad_enabled: bool = True       # ON in training, OFF in early law school

    def _pain(self, dd_pct: float) -> float:
        """L3 exponential ramp from pain_zone_start (3.5%) to hard_wall (4.0%)."""
        # COUPLING -> runtime/config.py: reads ChallengeConfig.pain_zone_start_pct and
        # .hard_wall_pct by attribute; renaming those fields there breaks the pain ramp.
        lo, hi = self.challenge.pain_zone_start_pct, self.challenge.hard_wall_pct
        if dd_pct <= lo:
            return 0.0
        frac = min(1.0, (dd_pct - lo) / max(1e-9, hi - lo))
        return math.expm1(PAIN_K * frac) / math.expm1(PAIN_K)   # 0..1, convex

    def decompose(self, ctx: RewardContext) -> Dict[str, float]:
        """Per-layer contributions (for telemetry + the E8 dominance proof)."""
        l0 = ctx.net_pnl_delta
        l1 = ALPHA if (ctx.in_position and ctx.momentum_aligned) else 0.0
        l2 = -BETA if ctx.stagnation else 0.0
        l3 = -DELTA * self._pain(ctx.drawdown_pct)
        l4 = EPS * max(0.0, ctx.day_progress)
        # L5 category multiplier on the dense shaping (breach-risk = protect capital:
        # damp upside shaping, keep protection). Bounded so it can't flip dominance.
        l5_mult = 0.5 if ctx.breach_risk else 1.0
        shaped = (l1 + l2 + l4) * l5_mult
        # COUPLING [C8] -> diagnostics/mlp_interpreter/interpreter.py + llm_risk_doctor/
        # doctor.py: both read the "L0".."L5_mult"/"shaped" keys by name (L0-dominance /
        # Reward-Hijack checks). "total" is consumed by reward()/env. Keep these key names.
        return {"L0": l0, "L1": l1, "L2": l2, "L3": l3, "L4": l4,
                "L5_mult": l5_mult, "shaped": shaped, "total": l0 + shaped + l3}

    def reward(self, ctx: RewardContext) -> float:
        return self.decompose(ctx)["total"]


# --------------------------- QUAD daily bonus (E9) ---------------------------
@dataclass
class DailyMetrics:
    """One day's raw inputs for the QUAD signals."""

    drawdown_efficiency: float   # cushion from the 4% wall across the day
    law_productivity: float      # closed profit from law-active allowed-direction trades
    target_velocity: float       # day net profit / bars in open positions
    td_stability: float          # TD-error / advantage line (qualifier)
    day_pnl: float               # day net PnL (the ceiling reference)
    passed: bool                 # hit target AND avoided the breach (pass-day gate)


def _sma4_above_shift4(series: List[float]) -> bool:
    """House pattern: SMA-4 above its shift-4 line. Needs >= 8 samples."""
    if len(series) < 8:
        return False
    sma4 = sum(series[-4:]) / 4.0
    shift4 = sum(series[-8:-4]) / 4.0
    return sma4 > shift4


def _sma4_below_shift4(series: List[float]) -> bool:
    if len(series) < 8:
        return False
    return (sum(series[-4:]) / 4.0) < (sum(series[-8:-4]) / 4.0)


class QuadBonus:
    """E9 QUAD bonus: 3 payable signals + TD qualifier, flow synergy, streak, 95% cap."""

    MICRO = 0.05      # each payable signal: +5% of day PnL
    FLOW = 0.05       # all 3 payable TRUE and TD qualifier TRUE: +5%
    STREAK = 0.05     # +5% per extra consecutive flow day
    CEILING = 0.95    # total bonus strictly < 1x day PnL (E8-safe)

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._hist: Dict[str, List[float]] = {k: [] for k in
                                              ("dd_eff", "law_prod", "tgt_vel", "td_stab")}
        self.flow_streak = 0

    def end_of_day(self, m: DailyMetrics) -> float:
        """Return the day's QUAD bonus in account dollars (0 unless a valid pass day)."""
        for key, val in [("dd_eff", m.drawdown_efficiency), ("law_prod", m.law_productivity),
                         ("tgt_vel", m.target_velocity), ("td_stab", m.td_stability)]:
            self._hist[key].append(val)
        if not self.enabled or not m.passed:
            self.flow_streak = 0 if not m.passed else self.flow_streak
            return 0.0

        dd = _sma4_above_shift4(self._hist["dd_eff"])
        law = _sma4_above_shift4(self._hist["law_prod"])
        tgt = _sma4_above_shift4(self._hist["tgt_vel"])
        td_ok = _sma4_below_shift4(self._hist["td_stab"])   # qualifier: BELOW its line

        bonus_frac = self.MICRO * (dd + law + tgt)          # payable micro-bonuses
        if dd and law and tgt and td_ok:                    # flow-state synergy
            bonus_frac += self.FLOW
            self.flow_streak += 1
            bonus_frac += self.STREAK * (self.flow_streak - 1)
        else:
            self.flow_streak = 0

        bonus_frac = min(bonus_frac, self.CEILING)          # E8-safe ceiling
        return bonus_frac * max(0.0, m.day_pnl)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M6 — implemented the layered reward + QUAD bonus.
#   I: The env returned a raw Layer-0 proxy; the bot needs the full layered objective
#      with Layer-0 dominance and the pain-zone/QUAD structure to learn to PASS.
#   R: REWARD_DESIGN.md (L0-L6, tiny shaping, exponential pain 3.5->4.0%) + E8 (L0
#      dominates) + E9 (QUAD: 3 payable + TD qualifier, 5/5/5%, 95% ceiling, toggle).
#   A: RewardEngine.decompose/reward with tiny locked coefficients + L5 breach-risk
#      damping; QuadBonus EOD subsystem (SMA-4 vs shift-4 signals, flow streak, 95% cap).
#   C: Layer 0 provably dominates (E8 test), so the policy optimizes REAL net progress
#      inside the legal/risk-safe space and the pain ramp keeps it off the wall - which
#      is what passing consistently requires.
