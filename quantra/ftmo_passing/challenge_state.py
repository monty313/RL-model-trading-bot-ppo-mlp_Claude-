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

from quantra.runtime.config import ChallengeConfig


@dataclass
class ChallengeState:
    """Mutable shared account. One instance per episode, read by all 4 symbols."""

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

    # --- wall + buffer (Phase A) ---
    @property
    def wall_equity(self) -> float:
        """Trailing wall: peak minus daily_risk_pct% of the account (4% default)."""
        return self.peak_equity - (self.challenge.daily_risk_pct / 100.0) * self.account_size

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
        """Daily reset (00:00 CE(S)T, SOW §10.3): re-anchor day start + target."""
        self.day_start_equity = self.equity
        self.target_hit = False

    def account_block(self) -> np.ndarray:
        """The 7-scalar `account` observation block (schema order), normalized.

        Order matches schema _account_names(): equity_norm, equity_dev, equity_slope,
        trailing_buffer, daily_buffer, day_progress, overall_progress. equity_slope is
        a per-step hook (env fills it); here it's 0 (filled by the env's equity SMA).
        """
        eq_norm = self.equity / self.account_size
        eq_dev = (self.equity - self.peak_equity) / self.account_size
        trailing_buf = self.remaining_buffer / self.account_size
        daily_buf = (self.equity - self.wall_equity) / self.account_size  # same anchor (Phase A)
        day_progress = self.day_pnl / ((self.challenge.daily_target_pct / 100.0) * self.account_size)
        overall_progress = (self.equity - self.account_size) / self.account_size
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
