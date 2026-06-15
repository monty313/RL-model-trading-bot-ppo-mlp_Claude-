"""ChallengeState — the shared account block + the FTMO buffers/wall the env reads.

WHAT THIS MODULE DOES
---------------------
Tracks the ONE shared account every symbol's decision reads (SOW B5): balance,
equity (realized + unrealized), the trailing high-water peak, the day's start equity,
the remaining daily-risk buffer (distance to the wall), and day PnL vs target. It
exposes the 7-scalar account observation block and flags the hard-wall breach.

This is Phase-A only for M4 (4% trailing wall + buffer + breach). The two-phase rule
(at +2.5% auto-flat all -> fresh 1% trailing, Phase B) is M7; the hooks (`phase`,
`target_hit`) are here so M7 slots in without reshaping the account block.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The whole mission is defined relative to this object: hit the daily target without
the equity touching the trailing wall. The remaining-buffer it computes is what the
RiskManager sizes against (so total risk never overshoots, B5), and the wall it
enforces is the hard breach line. A faithful shared-account picture is also what lets
a EURUSD decision see the risk an open XAUUSD trade already consumed (portfolio-aware).

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md`` (Term 6 Risk Context). For any
breach, walk backward from the moment ``breached`` flips: when did remaining_buffer
start collapsing, and did the action distribution adapt? The gap is the danger-
blindness window. NEVER modify the wall/buffer — read only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COUPLING [C5] -> quantra/runtime/config.py: ChallengeConfig fields read below
# (daily_risk_pct, phase_b_trailing_pct, daily_target_pct, ftmo_account_size) must keep
# these exact names/units (%) or wall/target math here silently breaks.
from quantra.runtime.config import ChallengeConfig


@dataclass
class ChallengeState:
    """Mutable shared account. One instance per episode, read by all 4 symbols."""

    # COUPLING [C5] -> quantra/env/trading_env.py: env constructs ChallengeState(account_size,
    # challenge) and reads attrs (equity, peak_equity, account_size, remaining_buffer, breached,
    # should_autoflat) + methods (mark_to_market, realize, charge, enter_phase_b, account_block).
    # Renaming any of these breaks the env's account wiring; live_bridge/live_session.py too.
    account_size: float
    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)

    balance: float = field(init=False)        # realized equity
    equity: float = field(init=False)         # realized + unrealized
    peak_equity: float = field(init=False)    # trailing high-water anchor
    day_start_equity: float = field(init=False)
    phase: str = field(init=False, default="A")
    breached: bool = field(init=False, default=False)
    locked_out: bool = field(init=False, default=False)
    target_hit: bool = field(init=False, default=False)

    def __post_init__(self):
        self.balance = self.account_size
        self.equity = self.account_size
        self.peak_equity = self.account_size
        self.day_start_equity = self.account_size

    # --- wall + buffer (two-phase, SOW §2.6) ---
    @property
    def wall_equity(self) -> float:
        """The trailing stop-loss (account-level). ftmo_mode: Phase A = peak − daily_risk_pct,
        Phase B (post auto-flat) = fresh tighter phase_b_trailing_pct. ftmo_mode OFF: a SINGLE
        trailing stop (daily_risk_pct), never tightened — the bot compounds all day
        [operator decision 2026-06-15: OFF runs indefinitely under one trailing stop]."""
        if self.challenge.ftmo_mode and self.phase == "B":
            pct = self.challenge.phase_b_trailing_pct
        else:
            pct = self.challenge.daily_risk_pct
        return self.peak_equity - (pct / 100.0) * self.account_size

    @property
    def should_autoflat(self) -> bool:
        """Auto-flat ALL when the day's target is hit (never while breached). ftmo_mode ON:
        always (Phase A) -> banks the pass behind the tighter Phase-B wall. ftmo_mode OFF:
        ONLY if stop_for_day is set; otherwise OFF runs PAST the target (the target is still
        the AIM that drives day_progress + the success-%, just not a forced stop)."""
        if self.breached or not self.target_hit:
            return False
        if self.challenge.ftmo_mode:
            return self.phase == "A"
        return self.challenge.stop_for_day

    def enter_phase_b(self) -> None:
        """After the +2.5% auto-flat: re-anchor the trailing peak to current equity and
        switch to the fresh 1% Phase-B wall. The day's target is banked."""
        self.phase = "B"
        self.peak_equity = self.equity

    @property
    def remaining_buffer(self) -> float:
        """USD the account can still lose before the wall. RiskManager sizes vs this."""
        return max(0.0, self.equity - self.wall_equity)

    @property
    def daily_target_equity(self) -> float:
        return self.day_start_equity + (self.challenge.daily_target_pct / 100.0) * self.account_size

    @property
    def day_pnl(self) -> float:
        return self.equity - self.day_start_equity

    # --- updates ---
    def mark_to_market(self, total_unrealized: float) -> None:
        """Recompute equity from realized balance + summed open-slot uPnL; update
        peak; trip the hard wall if breached. Called once per bar after prices move."""
        self.equity = self.balance + total_unrealized
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        if self.equity >= self.daily_target_equity:
            self.target_hit = True
        if self.equity <= self.wall_equity and not self.breached:
            self.breached = True
            self.locked_out = True   # force-flatten + lockout handled by the env

    def realize(self, pnl_after_costs: float) -> None:
        """Bank a closed trade's net PnL into balance (equity recomputed on m2m)."""
        self.balance += pnl_after_costs

    def charge(self, cost: float) -> None:
        """Deduct a fill cost immediately from balance (so equity reflects it)."""
        self.balance -= cost

    def reset_day(self) -> None:
        """Daily reset (00:00 CE(S)T, SOW §10.3) — START A FRESH DAY. Re-anchors day_start AND
        the trailing peak to current equity, clears target_hit, and returns to Phase A, so each
        new calendar day begins with its own 2.5% target and the wide Phase-A wall instead of
        being stuck in a post-day-1 Phase B [2026-06-15 fix: reset_day was dead code + left phase
        in B]. COUPLING -> env/trading_env.py _advance_bar() calls this on a calendar-day change."""
        self.day_start_equity = self.equity
        self.peak_equity = self.equity
        self.target_hit = False
        self.phase = "A"

    def account_block(self) -> np.ndarray:
        """The 7-scalar `account` observation block (schema order), normalized.

        Order matches schema _account_names(): equity_norm, equity_dev, equity_slope,
        trailing_buffer, daily_buffer, day_progress, overall_progress. equity_slope is
        a per-step hook (env fills it); here it's 0 (filled by the env's equity SMA).
        """
        eq_norm = self.equity / self.account_size
        eq_dev = (self.equity - self.peak_equity) / self.account_size
        trailing_buf = self.remaining_buffer / self.account_size
        daily_buf = (self.equity - self.wall_equity) / self.account_size  # NOTE: same single wall
        #     in EVERY phase, so pre-breach this == trailing_buf (audit). A genuine separate daily-
        #     loss wall lands with the scale-invariant-obs rework (task #23); kept now for obs width.
        day_progress = self.day_pnl / ((self.challenge.daily_target_pct / 100.0) * self.account_size)
        overall_progress = (self.equity - self.account_size) / self.account_size
        # COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py: this 7-scalar
        # order MUST equal schema._account_names() (equity_norm, equity_dev, equity_slope,
        # trailing_buffer, daily_buffer, day_progress, overall_progress); also consumed via
        # env/trading_env.py _obs()/_reward() (reads index [5]=day_progress). Reorder -> breaks both.
        return np.array([
            eq_norm, eq_dev, 0.0, trailing_buf, daily_buf, day_progress, overall_progress
        ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M4 — ChallengeState (shared account + Phase-A buffer/wall).
#   I: The env needs a single shared account with the remaining-risk buffer (for
#      sizing) and the hard wall (for breach), readable by all 4 symbols (B5).
#   R: SOW §2.6/2.7 (two-phase, three-zone), B5 (one shared account block), H4 walls.
#   A: equity/peak/buffer tracking + Phase-A 4% trailing wall + the 7-scalar account
#      block; hooks (phase, target_hit) reserved for the M7 two-phase rule.
#   C: Every symbol sizes against the SAME live buffer and the wall is enforced
#      centrally, so the 4 symbols can't collectively overshoot — the core of B5 and
#      of not breaching, hence of passing.
# [2026-06-13] M7 — two-phase episode rule.
#   I: Phase-A-only tracking didn't bank the +2.5% win or switch to the tighter wall.
#   R: SOW §2.6 (Phase A 4% trailing until +2.5% -> auto-flat ALL -> Phase B 1% trailing).
#   A: phase-aware wall_equity (4% in A, 1% in B), should_autoflat, enter_phase_b
#      (re-anchor peak to post-flat equity). The env force-flattens + enters Phase B.
#   C: Hitting target locks in the day's gain behind a tight 1% wall, exactly the
#      challenge-style stopping behaviour that turns a good day into a banked pass-day.
# [2026-06-15] ftmo_mode-aware wall + auto-flat (per-day inputs).
#   I: wall/auto-flat were hard-wired to the 2-phase challenge; ftmo OFF needs a single
#      trailing stop that runs indefinitely (no auto-flat at target).
#   R: Operator decision 2026-06-15 (OFF = one trailing stop, compounds all day).
#   A: wall_equity uses phase_b only when ftmo_mode AND phase B; should_autoflat requires
#      ftmo_mode (OFF never auto-flats).
#   C: ON keeps the protective 2-phase pass behaviour; OFF lets a strong policy keep
#      compounding under a single trailing stop - the operator's side-account mode.
# [2026-06-15b] OFF keeps the target as the AIM + stop_for_day toggle.
#   I: OFF must still HAVE a target (operator: "OFF has to have a target and a trailing
#      stop") - it drives day_progress + the success-%; it just isn't a forced stop.
#   R: Operator correction 2026-06-15 + the earlier "stop-for-day toggle" request.
#   A: should_autoflat: ON auto-flats at target (Phase A); OFF auto-flats ONLY if
#      challenge.stop_for_day, else returns False so the day runs PAST the target.
#   C: OFF aims at the target (measured by the success-%) yet compounds past it by default,
#      with an opt-in stop to bank a side-account day - matching how the operator trades.
# [2026-06-15c] reset_day made real (was dead code) + daily_buffer comment corrected.
#   I: (audit) reset_day was never called and left phase in B; acct_daily_buffer == trailing.
#   R: Logic audit 2026-06-15.
#   A: reset_day re-anchors day_start AND peak, clears target_hit, returns to Phase A (fresh day);
#      the env now calls it on a calendar-day change. daily_buffer comment fixed (genuine daily
#      wall deferred to the scale-invariant-obs rework).
#   C: Each calendar day is a fresh Phase-A 2.5%/4% challenge instead of a stuck post-day-1 Phase
#      B - faithful multi-day simulation, which is what an honest pass-rate metric requires.
