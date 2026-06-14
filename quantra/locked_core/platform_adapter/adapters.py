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
    """Real MetaTrader5 adapter (Windows + terminal). MetaTrader5 imported lazily.

    Production-capable: connect/login, hardened market_order (symbol_select, live
    tick price, per-broker filling-mode negotiation, retcode check, magic+deviation),
    and a REAL close_position (opposite deal referencing the position ticket). Every
    method that calls the terminal is pragma:no-cover here because it can only be
    validated against a live MT5 terminal on the operator's machine — never in CI.
    """

    DEVIATION = 20          # max slippage in points the broker may fill within
    MAGIC = 909313          # tag Quantra orders so we can identify/manage only ours

    def __init__(self, login: int = 0, password: str = "", server: str = "",
                 path: str = "", deviation: int = DEVIATION, magic: int = MAGIC):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:  # pragma: no cover - no terminal in CI
            raise RuntimeError(
                "MT5Adapter requires the MetaTrader5 package + a running MT5 terminal "
                "(Windows). Use SimBrokerAdapter for paper/testing."
            ) from exc
        self._mt5 = mt5
        self.login, self.password, self.server, self.path = login, password, server, path
        self.deviation, self.magic = deviation, magic

    # -- connection --
    def connect(self) -> bool:  # pragma: no cover - needs a live terminal
        m = self._mt5
        ok = m.initialize(path=self.path) if self.path else m.initialize()
        if not ok:
            return False
        if self.login:
            ok = m.login(self.login, password=self.password, server=self.server)
        return bool(ok)

    # -- helpers --
    def _filling(self, symbol):  # pragma: no cover - broker-dependent
        """Pick a filling mode the symbol/broker actually supports (IOC -> FOK -> RETURN)."""
        m = self._mt5
        info = m.symbol_info(symbol)
        modes = getattr(info, "filling_mode", 0) if info else 0
        if modes & m.ORDER_FILLING_IOC:
            return m.ORDER_FILLING_IOC
        if modes & m.ORDER_FILLING_FOK:
            return m.ORDER_FILLING_FOK
        return m.ORDER_FILLING_RETURN

    def _send(self, req) -> int:  # pragma: no cover
        """order_send + retcode check. Returns the resulting order/deal ticket or 0."""
        res = self._mt5.order_send(req)
        if res is None or res.retcode != self._mt5.TRADE_RETCODE_DONE:
            return 0
        return int(getattr(res, "order", 0) or getattr(res, "deal", 0))

    # -- orders --
    def market_order(self, symbol, side, lots, price=0.0) -> int:  # pragma: no cover
        m = self._mt5
        m.symbol_select(symbol, True)                       # ensure the symbol is in Market Watch
        tick = m.symbol_info_tick(symbol)
        if tick is None:
            return 0
        is_buy = side > 0
        px = price or (tick.ask if is_buy else tick.bid)    # buy at ask, sell at bid
        req = {
            "action": m.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(lots),
            "type": m.ORDER_TYPE_BUY if is_buy else m.ORDER_TYPE_SELL, "price": float(px),
            "deviation": self.deviation, "magic": self.magic,
            "type_time": m.ORDER_TIME_GTC, "type_filling": self._filling(symbol),
            "comment": "quantra",
        }
        return self._send(req)

    def close_position(self, ticket, price=0.0) -> bool:  # pragma: no cover
        """Close a position with the OPPOSITE deal referencing its ticket."""
        m = self._mt5
        found = m.positions_get(ticket=ticket)
        if not found:
            return False
        pos = found[0]
        m.symbol_select(pos.symbol, True)
        tick = m.symbol_info_tick(pos.symbol)
        if tick is None:
            return False
        closing_buy = pos.type == m.POSITION_TYPE_SELL     # close a short by buying
        px = price or (tick.ask if closing_buy else tick.bid)
        req = {
            "action": m.TRADE_ACTION_DEAL, "symbol": pos.symbol, "position": int(ticket),
            "volume": float(pos.volume),
            "type": m.ORDER_TYPE_BUY if closing_buy else m.ORDER_TYPE_SELL, "price": float(px),
            "deviation": self.deviation, "magic": self.magic,
            "type_time": m.ORDER_TIME_GTC, "type_filling": self._filling(pos.symbol),
            "comment": "quantra-close",
        }
        return self._send(req) != 0

    def positions(self):  # pragma: no cover
        return [Position(p.ticket, p.symbol, 1 if p.type == 0 else -1, p.volume, p.price_open)
                for p in (self._mt5.positions_get() or [])]

    def equity(self) -> float:  # pragma: no cover
        info = self._mt5.account_info()
        return float(info.equity) if info else 0.0

    def recent_bars(self, symbol: str, n: int = 600):  # pragma: no cover
        """Pull the last n CLOSED 1m bars as a clean OHLCV+spread DataFrame (live feed)."""
        import pandas as pd
        m = self._mt5
        m.symbol_select(symbol, True)
        # index 1 (not 0) so we never include the still-forming current bar (no lookahead).
        rates = m.copy_rates_from_pos(symbol, m.TIMEFRAME_M1, 1, n)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        out = pd.DataFrame({
            "open": df["open"], "high": df["high"], "low": df["low"], "close": df["close"],
            "tick_volume": df.get("tick_volume", 0.0),
            "spread": df.get("spread", 0.0).astype(float),
        })
        out.index.name = "time"
        return out


def make_adapter(kind: str = "sim", **kw) -> BrokerAdapter:
    """Factory: 'sim' (default, paper) or 'mt5' (real). Falls back to sim if MT5 is
    unavailable, so a non-Windows/CI environment always gets a working adapter."""
    if kind == "mt5":
        try:
            return MT5Adapter(**kw)          # login/password/server/path/deviation/magic
        except RuntimeError:
            return SimBrokerAdapter()        # kw was MT5-specific; sim uses its defaults
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
