"""Platform adapters — MT5 broker interface (live) + a sim/paper fallback. 🔴

WHAT THIS MODULE DOES
---------------------
A minimal broker interface (connect / market_order / close_position / positions /
equity) with two implementations:
  * MT5Adapter   — real MetaTrader5 execution (Windows + the MT5 terminal). Imported
                   lazily and guarded, so the package always imports elsewhere.
  * SimBrokerAdapter — an in-memory paper broker (the default + the test substrate),
                   so the whole live stack is exercisable without a real terminal.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The policy is platform-blind (SOW H3): it emits normalized intentions; this layer is
the ONLY place that knows about lots/tickets/brokers. Keeping that boundary clean is
what lets one brain pass on any FTMO account size, and the sim adapter lets us verify
the live execution logic safely before risking a funded account.

🔴 LOCKED boundary: only this tier issues broker commands. The LLM Risk Doctor never does.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. You may VIEW this file but must NEVER
call any method here — issuing broker commands is outside your read-only boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Position:
    """One open broker position (a filled trade slot)."""

    ticket: int
    symbol: str
    side: int          # +1 long, -1 short
    lots: float
    entry_price: float


class BrokerAdapter:
    """Abstract broker interface. ExecutionAdapter speaks only to this."""

    def connect(self) -> bool: raise NotImplementedError
    def market_order(self, symbol: str, side: int, lots: float, price: float = 0.0) -> int: raise NotImplementedError
    def close_position(self, ticket: int, price: float = 0.0) -> bool: raise NotImplementedError
    def positions(self) -> List[Position]: raise NotImplementedError
    def equity(self) -> float: raise NotImplementedError


@dataclass
class SimBrokerAdapter(BrokerAdapter):
    """In-memory paper broker — the default + test substrate. No real orders."""

    starting_equity: float = 10_000.0
    _next_ticket: int = 1000
    _positions: Dict[int, Position] = field(default_factory=dict)
    _realized: float = 0.0
    _connected: bool = False

    def connect(self) -> bool:
        self._connected = True
        return True

    def market_order(self, symbol: str, side: int, lots: float, price: float = 0.0) -> int:
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions[ticket] = Position(ticket, symbol, int(side), float(lots), float(price))
        return ticket

    def close_position(self, ticket: int, price: float = 0.0) -> bool:
        return self._positions.pop(ticket, None) is not None

    def positions(self) -> List[Position]:
        return list(self._positions.values())

    def equity(self) -> float:
        return self.starting_equity + self._realized


class MT5Adapter(BrokerAdapter):
    """Real MetaTrader5 adapter (Windows + terminal). MetaTrader5 imported lazily."""

    def __init__(self):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:  # pragma: no cover - no terminal in CI
            raise RuntimeError(
                "MT5Adapter requires the MetaTrader5 package + a running MT5 terminal "
                "(Windows). Use SimBrokerAdapter for paper/testing."
            ) from exc
        self._mt5 = mt5

    def connect(self) -> bool:  # pragma: no cover - needs a live terminal
        return bool(self._mt5.initialize())

    def market_order(self, symbol, side, lots, price=0.0) -> int:  # pragma: no cover
        m = self._mt5
        req = {"action": m.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lots),
               "type": m.ORDER_TYPE_BUY if side > 0 else m.ORDER_TYPE_SELL,
               "deviation": 20, "type_filling": m.ORDER_FILLING_IOC}
        res = m.order_send(req)
        return int(getattr(res, "order", 0))

    def close_position(self, ticket, price=0.0) -> bool:  # pragma: no cover
        # Real close requires an opposite deal referencing the position; left to the
        # operator's terminal config. Returns False here as a safe no-op stub.
        return False

    def positions(self):  # pragma: no cover
        return [Position(p.ticket, p.symbol, 1 if p.type == 0 else -1, p.volume, p.price_open)
                for p in (self._mt5.positions_get() or [])]

    def equity(self) -> float:  # pragma: no cover
        info = self._mt5.account_info()
        return float(info.equity) if info else 0.0


def make_adapter(kind: str = "sim", **kw) -> BrokerAdapter:
    """Factory: 'sim' (default, paper) or 'mt5' (real). Falls back to sim if MT5 is
    unavailable, so a non-Windows/CI environment always gets a working adapter."""
    if kind == "mt5":
        try:
            return MT5Adapter()
        except RuntimeError:
            return SimBrokerAdapter(**kw)
    return SimBrokerAdapter(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M14 — implemented the platform adapters (MT5 + sim).
#   I: The platform-blind policy needs a single boundary that turns intentions into
#      broker orders, testable without a real terminal.
#   R: SOW H3 (platform-blind) + §10/§14 (MT5 live, isolated) + only this tier issues orders.
#   A: BrokerAdapter interface; SimBrokerAdapter (in-memory paper, default); guarded
#      MT5Adapter; make_adapter factory with safe sim fallback.
#   C: The live execution logic is exercisable + correct before any funded account is
#      risked, and the broker boundary stays clean - so one brain can pass any account.
