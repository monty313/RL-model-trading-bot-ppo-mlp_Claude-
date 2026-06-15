"""CostLayer — real FTMO costs from day 1. 🔴

WHAT THIS MODULE DOES
---------------------
Charges, in account dollars, the real frictions of every trade (SOW §10.5):
  * Spread — paid on entry (you cross the spread): spread_price * contract * lots.
  * Slippage — fixed points per symbol, adverse on EVERY fill (open AND close).
  * Commission — $5 round-trip per lot on FOREX only; metals/indices pay none.
All in USD because every traded symbol is USD-quoted (CONTRACT_SIZE handles the
price->dollars conversion).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The bot never meets a costless world (SOW C8). An edge that survives real spread +
slippage + commission is one that can actually pass a funded challenge; a costless
edge evaporates live and breaches. Charging costs at fill time also teaches the bot
to value restraint — overtrading bleeds the daily buffer toward the wall.

🔴 LOCKED (SOW §10.5): $5 RT/lot forex; metals/indices no per-trade commission.
Changing the cost structure needs Monty's approval.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Costs are subtracted inside
reward Layer 0 (net PnL). If the bot churns (many tiny trades) and bleeds, look at
cost-per-trade vs gross PnL here before blaming the reward shaping — a churning
policy is a Stagnation/Reward issue, but the *mechanism* of the bleed is this layer.
"""

from __future__ import annotations

from dataclasses import dataclass

# COUPLING [C5] -> runtime/config.py: reads the per-symbol dicts CONTRACT_SIZE / SLIPPAGE_POINTS /
# POINT_SIZE / ASSET_CLASS (all keyed by config.SYMBOLS) + DEFAULT_POINT_SIZE, and cfg.CostConfig
# (commissioned_classes, commission_per_lot_rt_forex). Renaming a config field/key breaks costing.
from quantra.runtime import config as cfg


@dataclass(frozen=True)
class FillCosts:
    """Itemised costs of one fill (open or close), in USD — logged for telemetry."""

    # COUPLING [C8] -> diagnostics/telemetry_logger/logger.py: these field names (spread/slippage/
    # commission/total) are the per-fill cost contract the telemetry logger + interpreter read.
    # Renaming a field changes the logged cost schema other diagnostics tools depend on.
    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.slippage + self.commission


class CostLayer:
    """Converts a fill (symbol, lots, spread) into USD cost. Stateless + pure."""

    def __init__(self, cost_cfg: cfg.CostConfig | None = None):
        self.cfg = cost_cfg or cfg.CostConfig()

    def _contract(self, symbol: str) -> float:
        return cfg.CONTRACT_SIZE.get(symbol, 1.0)

    def _slippage_usd(self, symbol: str, lots: float) -> float:
        """Fixed slippage points -> price -> USD, charged on every fill."""
        # COUPLING [C5] -> runtime/config.py: SLIPPAGE_POINTS[symbol] + POINT_SIZE[symbol] (per-symbol
        # dicts keyed by config.SYMBOLS). A missing key silently falls back (0.0 / DEFAULT_POINT_SIZE)
        # and undercharges; keep both dicts' keys == config.SYMBOLS when adding/removing a symbol.
        pts = cfg.SLIPPAGE_POINTS.get(symbol, 0.0)
        point = cfg.POINT_SIZE.get(symbol, cfg.DEFAULT_POINT_SIZE)
        return pts * point * self._contract(symbol) * lots

    def _pays_commission(self, symbol: str) -> bool:
        # COUPLING [C5] -> runtime/config.py: ASSET_CLASS[symbol] decides forex (pays $5 RT) vs
        # metals/indices (none); its values must match cfg.CostConfig.commissioned_classes membership.
        return cfg.ASSET_CLASS.get(symbol, "forex") in self.cfg.commissioned_classes

    def open_cost(self, symbol: str, lots: float, spread_price: float) -> FillCosts:
        """Cost to OPEN: full spread (you cross it) + entry slippage. No commission
        here — the $5 RT is charged once, on close, to avoid double-counting."""
        spread = max(0.0, spread_price) * self._contract(symbol) * lots
        return FillCosts(spread=spread, slippage=self._slippage_usd(symbol, lots))

    def close_cost(self, symbol: str, lots: float) -> FillCosts:
        """Cost to CLOSE: exit slippage + the $5 round-trip commission (forex only)."""
        commission = (self.cfg.commission_per_lot_rt_forex * lots
                      if self._pays_commission(symbol) else 0.0)
        return FillCosts(slippage=self._slippage_usd(symbol, lots), commission=commission)

    def round_trip_cost(self, symbol: str, lots: float, spread_price: float) -> float:
        """Total USD cost of a full open+close (for sizing sanity / diagnostics)."""
        return (self.open_cost(symbol, lots, spread_price).total
                + self.close_cost(symbol, lots).total)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M4 — implemented the CostLayer.
#   I: Without real costs the bot would learn a costless edge that breaches live.
#   R: SOW §10.5 ($5 RT/lot forex; metals/indices none; spread + fixed slippage) + C8.
#   A: open_cost (spread + entry slippage), close_cost (exit slippage + $5 RT forex-only),
#      all in USD via CONTRACT_SIZE; itemised FillCosts for telemetry.
#   C: Every trade pays the real friction it would pay at FTMO, so a surviving edge
#      is a passable edge and the bot learns the restraint that keeps the buffer alive.
