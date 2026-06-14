"""RiskManager — raw_size in [0,1] -> lots, slot-aware, NEVER overshoots. 🔴

WHAT THIS MODULE DOES
---------------------
Converts the policy's normalized size into broker lots against the REMAINING
daily-risk buffer (SOW H3). It is the hard guarantee behind the B5 invariant: the
total risk of all open slots (across all 4 symbols) can never exceed what the
account can still afford to lose today.

The guarantee mechanism: a position's "risk" = its loss if price hits a reference
stop (stop_atr_mult * ATR). The desired risk is capped at the available budget, and
lots are rounded DOWN to the broker step — so committed risk <= desired <= available
ALWAYS. A trade that can't fit even the minimum lot is refused (0 lots), never forced.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
This is the mechanical reason the bot cannot size its way into a breach. The policy
is platform-blind (SOW H3) — it only emits raw_size; the RiskManager translates that
into a lot count that respects the wall. Round-down + per-trade cap + the shared
buffer (threaded true-sequentially across symbols by the env, B5) keep total exposure
bounded, so the 4% hard wall is rarely even approached.

🔴 The no-overshoot invariant is locked. The dials (stop mult, caps) are tunable; the
invariant is not.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. The Risk Doctor NEVER touches
sizing (hard boundary). When diagnosing Risk Blindness, compare the raw_size the
actor wanted vs the feasible lots here: if the actor stays max-size into breach-risk
but the RiskManager keeps shrinking it, the wall held — the failure is the actor's
*intent*, not the sizing. Cite SizingResult.committed_risk vs the buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quantra.runtime import config as cfg


@dataclass(frozen=True)
class SizingResult:
    """Outcome of sizing one OPEN — carries the committed risk for buffer accounting."""

    lots: float
    committed_risk: float   # USD this position loses if its reference stop is hit
    risk_per_lot: float     # USD/lot (stored on the slot for later buffer math)
    feasible: bool
    reason: str


class RiskManager:
    """Slot-aware sizing. Pure given (account_size, dials); the env owns the buffer."""

    def __init__(self, account_size: float, risk_cfg: cfg.RiskConfig | None = None):
        self.account_size = float(account_size)
        self.cfg = risk_cfg or cfg.RiskConfig()

    def risk_per_lot(self, symbol: str, atr_price: float) -> float:
        """USD lost per 1.0 lot if price moves stop_atr_mult*ATR against the trade."""
        stop_distance = max(0.0, atr_price) * self.cfg.stop_atr_mult
        return stop_distance * cfg.CONTRACT_SIZE.get(symbol, 1.0)

    def size(self, symbol: str, raw_size: float, atr_price: float,
             available_budget: float) -> SizingResult:
        """raw_size in [0,1] -> feasible lots with committed_risk <= available_budget.

        ``available_budget`` is the daily-risk buffer ALREADY reduced by every open
        slot's committed risk (and, within a bar, by prior symbols' opens — the
        true-sequential B5 threading the env performs). Rounding DOWN is what makes
        the no-overshoot guarantee exact.
        """
        rpl = self.risk_per_lot(symbol, atr_price)
        raw_size = float(min(max(raw_size, 0.0), 1.0))
        if rpl <= 0.0 or available_budget <= 0.0:
            return SizingResult(0.0, 0.0, rpl, False, "no budget or zero risk/lot")

        per_trade_cap = self.cfg.max_per_trade_risk_frac * self.account_size
        desired_risk = raw_size * min(per_trade_cap, available_budget)
        desired_risk = min(desired_risk, available_budget)   # hard ceiling

        # Round DOWN to the lot step: committed = lots*rpl <= desired <= available.
        lots = math.floor((desired_risk / rpl) / self.cfg.lot_step) * self.cfg.lot_step
        lots = min(lots, self.cfg.max_lot)
        lots = round(lots, 8)
        if lots < self.cfg.min_lot:
            return SizingResult(0.0, 0.0, rpl, False, "below min lot for the buffer")

        committed = lots * rpl
        # Invariant (must hold by construction); assert to fail loud if ever violated.
        assert committed <= available_budget + 1e-6, (
            f"RiskManager overshoot: committed {committed} > budget {available_budget}"
        )
        return SizingResult(lots, committed, rpl, True, "ok")


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M4 — implemented the RiskManager (no-overshoot sizing).
#   I: A platform-blind policy emits raw_size in [0,1]; nothing converted that into
#      lots that respect the remaining daily-risk buffer, so the bot could over-leverage
#      straight into the 4% wall.
#   R: SOW H3 (raw_size -> lots vs remaining buffer + rounding + caps) + B5 (total
#      open-slot risk never exceeds the buffer).
#   A: risk_per_lot via stop_atr_mult*ATR*contract; desired risk capped at the budget;
#      lots rounded DOWN to lot_step; sub-min refused; invariant asserted.
#   C: The bot literally cannot size its way past the wall — committed risk <= budget
#      by construction — which is the mechanical foundation of not breaching, hence passing.
