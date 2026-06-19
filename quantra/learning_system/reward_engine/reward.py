"""RewardEngine — layered reward L0-L6 + QUAD, with Layer-0 dominance (E8). 🔴

WHAT THIS MODULE DOES
---------------------
Computes the layered reward (REWARD_DESIGN.md):
    r(t) = L0_dNetPnL + (L1_momentum + L2_dailyProgress + L4_tradeQuality) * L5_category
           - L3_painzone   [+ L6 daily bonus at end of day]
(C17 2026-06-19: L2 is now per-step daily-progress, L4 is per-close trade-quality — see decompose.)
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

from quantra.runtime.config import ChallengeConfig, RewardConfig

# C16 (2026-06-19): the per-layer shaping WEIGHTS now live in RewardConfig (config.py) as
# plain-English, operator-tunable fields — net_pnl_weight / step_pnl_weight /
# daily_progress_weight / drawdown_pain_weight / drawdown_pain_steepness / trade_quality_weight.
# The MATH/STRUCTURE + decompose() layer KEYS (L0..L5) are UNCHANGED; only names + default values
# changed, so the E8 Layer-0 dominance proof still holds. (Former globals: ALPHA/BETA/DELTA/EPS/PAIN_K;
# defaults preserved except daily_progress_weight raised 1e-4 -> 1e-3 per the C16 spec.)


@dataclass
class RewardContext:
    # COUPLING -> env/trading_env.py _reward(): constructs RewardContext by these exact keyword
    # field NAMES every step; renaming/removing one here breaks that call (and vice-versa). The
    # C17 re-pointed inputs are produced upstream and flow ACROSS three files:
    #   day_pnl, day_target_equity  <- ftmo_passing/challenge_state.py (ChallengeState.day_pnl and
    #                                  .daily_target_equity properties), read in env _reward().
    #   trade_close_quality         <- env/trading_env.py _record_close_quality() (called from the
    #                                  CLOSE branch of _apply_action + from _force_flatten), summed
    #                                  per step and ALREADY signed + account-normalized there.
    """Everything the engine needs for one step — the env populates it."""

    net_pnl_delta: float            # L0: equity change after costs this step / account (dominant)
    in_position: bool = False
    momentum_aligned: bool = False  # L1: small CCI back in sync + ATR alive, in trade dir
    drawdown_pct: float = 0.0       # L3: current DAILY drawdown % (for the pain zone)
    breach_risk: bool = False       # L5: in pain zone / near wall (the explicit category multiplier)
    # --- C17 [2026-06-19] re-pointed inputs (daily-progress L2 + trade-quality L4) ---------------
    day_pnl: float = 0.0            # L2: equity - day_start_equity (USD); reward only while > 0
    day_target_equity: float = 1.0  # L2: day_start_equity*(1+target%) (USD) — the progress denominator
    trade_close_quality: float = 0.0  # L4: signed, account-normalized realized-on-close quality, summed
    #                                   over trades CLOSED this step (0 on steps with no close)


@dataclass
class RewardEngine:
    """Pure layered-reward computation. One instance per training run."""

    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)   # C16 operator-tunable weights
    quad_enabled: bool = True       # ON in training, OFF in early law school

    def _pain(self, dd_pct: float) -> float:
        """L3 exponential ramp from pain_zone_start (3.5%) to hard_wall (4.0%)."""
        # COUPLING -> runtime/config.py: reads ChallengeConfig.pain_zone_start_pct and
        # .hard_wall_pct by attribute; renaming those fields there breaks the pain ramp.
        lo, hi = self.challenge.pain_zone_start_pct, self.challenge.hard_wall_pct
        if dd_pct <= lo:
            return 0.0
        frac = min(1.0, (dd_pct - lo) / max(1e-9, hi - lo))
        k = self.reward_cfg.drawdown_pain_steepness            # C16: was the PAIN_K global
        return math.expm1(k * frac) / math.expm1(k)            # 0..1, convex

    def decompose(self, ctx: RewardContext) -> Dict[str, float]:
        """Per-layer contributions (for telemetry + the E8 dominance proof). Weights come from
        RewardConfig (C16); L2 + L4 math was RE-POINTED (C17) to literally compute daily-progress
        and trade-quality. The layer KEYS are unchanged so the Risk Doctor / E8 readers still work."""
        rc = self.reward_cfg
        l0 = rc.net_pnl_weight * ctx.net_pnl_delta             # dominant outcome base (weight 1.0)
        l1 = rc.step_pnl_weight if (ctx.in_position and ctx.momentum_aligned) else 0.0
        # L2 daily-progress (C17): every step, a positive whisper that GROWS as equity climbs toward
        # the day's target and switches OFF while the day is flat/negative. day_pnl + day_target_equity
        # come straight from ChallengeState (ftmo_passing/challenge_state.py) via env _reward(); the
        # ratio is dimensionless (~0.024 at a 2.5% target) so it stays a whisper without a clamp.
        l2 = rc.daily_progress_weight * (max(0.0, ctx.day_pnl) / max(1e-9, ctx.day_target_equity))
        l3 = -rc.drawdown_pain_weight * self._pain(ctx.drawdown_pct)
        # L4 trade-quality (C17): nonzero ONLY on bars where a trade closed (else trade_close_quality
        # is 0). env/trading_env.py _record_close_quality() already applied the operator's sign rules
        # (reward winners; penalize giving back a once-profitable trade; ignore never-profitable
        # losers) AND normalized by account_size, so here it is a pure scaled passthrough.
        l4 = rc.trade_quality_weight * ctx.trade_close_quality
        # L5 category multiplier on the dense shaping (breach-risk = protect capital:
        # damp upside shaping, keep protection). Bounded so it can't flip dominance.
        l5_mult = 0.5 if ctx.breach_risk else 1.0
        shaped = (l1 + l2 + l4) * l5_mult
        # COUPLING [C8] -> diagnostics/mlp_interpreter/interpreter.py + llm_risk_doctor/
        # doctor.py: both read the "L0".."L5_mult"/"shaped" keys by name (L0-dominance /
        # Reward-Hijack checks). "total" is consumed by reward()/env. Keep these key names.
        # ⚠️ COMPATIBILITY [C18+] -> policy_registry/registry.py (default_reward_layer_keys ->
        # compatibility_signature): the set of "L*" keys here IS the reward LAYER ARRANGEMENT, the
        # SECOND input to a policy's compatibility hash. Tuning a WEIGHT (C16) or re-pointing a term's
        # MATH (C17) keeps these keys, so old policies stay RESUME-safe. But ADDING / REMOVING / RENAMING
        # a layer changes the hash -> the registry forces a fresh start (old policies can't be resumed).
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
# STANDING RULE [2026-06-19, operator] — applies to THIS file and EVERY file going forward: keep
# SHOWING THE WORK. On every edit (1) append a DATED IRAC entry here, and (2) in the code comments
# DOCUMENT the cross-file RELATIONSHIPS the change depends on (the COUPLING) — name the other file(s)
# and the exact attr/field/key relied on, in BOTH directions — and date the re-pointed logic, so any
# future reader/editor can see what connects to what, and what breaks where, and when it changed.
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
# [2026-06-15c] Clamp L4 target-progress at 1.0.
#   I: (audit) L4 = EPS*day_progress was unbounded; ftmo-OFF day_progress can reach ~40, so the
#      whisper could rival L0 over a window and the E8 proof never sampled that range.
#   R: Logic audit 2026-06-15 (L0 dominance must hold in ALL reachable configs).
#   A: l4 = EPS*min(1.0, max(0.0, day_progress)) — caps the shaping at target in every mode.
#   C: L0 stays the dominant objective even in the new OFF configuration, so the bot keeps
#      optimizing REAL net money (no reward hijack) - the basis of consistent passing.
# [2026-06-19] C16 — reward weights renamed to a plain-English, operator-tunable RewardConfig.
#   I: The shaping weights were cryptic module globals (ALPHA/BETA/DELTA/EPS/PAIN_K) — not
#      operator-visible, not captured per run, and not matching the multi-day consistency philosophy.
#   R: Operator spec 2026-06-19 (C16: rename to plain English + retune defaults; do NOT change the
#      reward math/structure; keep the E8 Layer-0 dominance proof valid — E8 guards the per-step
#      shaping, C11's failed-day penalty stays an exempt env-level event).
#   A: Weights moved to config.RewardConfig (net_pnl_weight / step_pnl_weight / daily_progress_weight /
#      drawdown_pain_weight / drawdown_pain_steepness / trade_quality_weight + a failed_day_penalty
#      mirror); RewardEngine reads reward_cfg; decompose() math + the L0..L5 keys are UNCHANGED; only
#      daily_progress_weight was raised 1e-4->1e-3 ("matters most"), verified E8-safe (worst shaping/L0
#      ratio ~0.26 over 1000x256 rollouts).
#   C: The training objective is now legible, tunable, and captured per run, the consistency driver is
#      weighted up, and Layer-0 still provably dominates the per-step shaping — so the bot optimizes
#      real net progress toward passing while the operator can shape HOW it gets there.
# [2026-06-19] C17 — re-pointed two reward terms so their MATH matches their plain-English names.
#   I: After C16 the names were a taxonomy over proxy math: daily_progress_weight scaled the old
#      stagnation flag and trade_quality_weight scaled target-progress — neither literally measured
#      what its name said, so tuning them was misleading.
#   R: Operator spec 2026-06-19 (C17: re-point exactly these two terms, change nothing else;
#      re-verify E8; full suite; IRAC). The other terms (L0/L1/L3/L5, QUAD) are untouched.
#   A: L2 daily-progress = daily_progress_weight * max(0, day_pnl)/day_target_equity (per step, OFF
#      when the day is flat/negative). L4 trade-quality fires only on a CLOSE: +winners,
#      -gave-back-a-once-profitable-trade, 0 for never-profitable losers. CROSS-FILE WIRING:
#      RewardContext gained day_pnl/day_target_equity/trade_close_quality; env/trading_env.py reads
#      ChallengeState.day_pnl + .daily_target_equity and accrues close-quality in _record_close_quality()
#      (using Slot.mfe>0 as the "ever in profit" signal, normalized by account_size). Dropped the now-dead
#      stagnation/day_progress ctx fields. E8 re-verified at the new math (worst shaping/L0 ~0.05 over
#      1000x256 rollouts; trade-quality term is the tiniest).
#   C: The knobs now mean exactly what they say — daily_progress literally rewards getting closer to the
#      day's target and trade_quality literally rewards banking winners / discourages round-tripping them
#      — so the operator can shape consistent, winner-keeping behaviour while Layer-0 still dominates.
