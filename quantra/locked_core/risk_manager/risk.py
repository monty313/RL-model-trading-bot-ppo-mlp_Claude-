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

# COUPLING [C5] -> runtime/config.py: this module reads cfg.CONTRACT_SIZE (per-symbol dict
# keyed by config.SYMBOLS) and cfg.RiskConfig (stop_atr_mult/max_per_trade_risk_frac/lot_step/
# max_lot/min_lot). Renaming those config fields/keys or dropping a symbol breaks sizing here.
from quantra.runtime import config as cfg


@dataclass(frozen=True)
class SizingResult:
    """Outcome of sizing one OPEN — carries the committed risk for buffer accounting."""

    # COUPLING -> env/trading_env.py: env reads SizingResult.committed_risk to debit the shared
    # B5 buffer and stores risk_per_lot on the slot; LLM Risk Doctor cites committed_risk vs the
    # buffer. Renaming/reordering these fields breaks the env's true-sequential buffer accounting.
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
        # COUPLING [C5] -> runtime/config.py: CONTRACT_SIZE[symbol] (price->USD per lot). The same
        # dict is read by cost_layer/costs.py + env/trading_env.py; a wrong/missing key here
        # silently falls back to 1.0 and mis-sizes risk. Keys must stay == config.SYMBOLS.
        return stop_distance * cfg.CONTRACT_SIZE.get(symbol, 1.0)

    def size(self, symbol: str, raw_size: float, atr_price: float,
             available_budget: float, *, apply_per_trade_cap: bool = True,
             price: float | None = None, contract: float | None = None,
             leverage: float | None = None, free_margin: float | None = None
             ) -> SizingResult:
        """raw_size in [0,1] -> feasible lots with committed_risk <= available_budget.

        ``available_budget`` is the daily-risk buffer ALREADY reduced by every open
        slot's committed risk (and, within a bar, by prior symbols' opens — the
        true-sequential B5 threading the env performs). Rounding DOWN is what makes
        the no-overshoot guarantee exact.

        Two optional caps layer ON TOP, and each only ever SHRINKS lots, so the 🔴
        no-overshoot invariant is preserved by construction:
          * apply_per_trade_cap (ftmo_mode ON): one trade <= max_per_trade_risk_frac of
            account. ftmo_mode OFF passes False -> no fixed cap; confidence scales the whole
            budget and MARGIN is the real ceiling (operator decision 2026-06-15).
          * margin (price·contract·leverage·free_margin): the broker 1:leverage ceiling.
            max lots = free_margin·leverage / (price·contract). This is the physical cap
            that lets ftmo-off run uncapped-but-safe.
        """
        rpl = self.risk_per_lot(symbol, atr_price)
        raw_size = float(min(max(raw_size, 0.0), 1.0))
        if rpl <= 0.0 or available_budget <= 0.0:
            return SizingResult(0.0, 0.0, rpl, False, "no budget or zero risk/lot")

        if apply_per_trade_cap:
            per_trade_cap = self.cfg.max_per_trade_risk_frac * self.account_size
            desired_risk = raw_size * min(per_trade_cap, available_budget)
        else:
            desired_risk = raw_size * available_budget    # confidence scales the WHOLE budget
        desired_risk = min(desired_risk, available_budget)   # hard ceiling (no overshoot)

        # Round DOWN to the lot step: committed = lots*rpl <= desired <= available.
        lots = math.floor((desired_risk / rpl) / self.cfg.lot_step) * self.cfg.lot_step

        # Margin ceiling (1:leverage). Only ever shrinks lots; never relaxes the budget.
        reason = "ok"
        if leverage and price and contract and free_margin is not None:
            margin_lots = (max(0.0, free_margin) * leverage) / (price * contract)
            margin_lots = math.floor(margin_lots / self.cfg.lot_step) * self.cfg.lot_step
            if margin_lots < lots:
                lots, reason = margin_lots, "margin-capped"

        lots = min(lots, self.cfg.max_lot)
        lots = round(lots, 8)
        if lots < self.cfg.min_lot:
            return SizingResult(0.0, 0.0, rpl, False,
                                "below min lot (margin)" if reason == "margin-capped"
                                else "below min lot for the buffer")

        committed = lots * rpl
        # Invariant (must hold by construction); assert to fail loud if ever violated.
        assert committed <= available_budget + 1e-6, (
            f"RiskManager overshoot: committed {committed} > budget {available_budget}"
        )
        return SizingResult(lots, committed, rpl, True, reason)


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
# [2026-06-15] Margin ceiling (1:leverage) + mode-aware per-trade cap.
#   I: per-trade risk was pinned to 1% of account regardless of target/DD, blocking the
#      size head from scaling with the regime; and the sim modelled NO broker margin.
#   R: Operator decision 2026-06-15 (no fixed caps when ftmo OFF; margin is the real cap;
#      different leverage per account).
#   A: size() gains apply_per_trade_cap (OFF -> confidence scales the whole budget) and an
#      optional margin ceiling (free_margin*leverage/(price*contract)); both only SHRINK lots.
#   C: The size head now scales risk with the regime up to the REAL physical (margin) limit,
#      so bigger goals are pursued proportionally without ever overshooting the wall (B5 intact).
