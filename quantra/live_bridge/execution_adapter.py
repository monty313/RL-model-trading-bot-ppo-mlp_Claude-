"""ExecutionAdapter — maps the 5-slot policy model onto live broker tickets. 🔴

WHAT THIS MODULE DOES
---------------------
Bridges the policy's per-symbol 5-slot model (SOW B2) to the broker: OPEN fills the
next FREE slot (places an order, records its ticket), CLOSE routes to the POINTER-
selected slot (closes exactly that ticket), and OPEN is refused when all 5 are full.
Pure slot<->ticket bookkeeping on top of a BrokerAdapter.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Live execution must mirror training EXACTLY — 5 slots, next-free OPEN, pointer-routed
CLOSE — or the behaviour that passed in training won't reproduce live. This adapter is
that mirror, keeping the live slot mechanics identical to the env's (M4).

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. View only; never call open/close —
that is broker execution, outside your boundary.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# COUPLING -> quantra/locked_core/platform_adapter/adapters.py: calls BrokerAdapter.market_order
# (returns ticket int) + close_position(ticket)->bool; those signatures are unpacked below.
from quantra.locked_core.platform_adapter.adapters import BrokerAdapter

# COUPLING [C3] -> quantra/market_pipeline/feature_builder/schema.py (N_SLOTS=5) + live_session.py:
# must equal schema N_SLOTS and env/ppo pointer width; change one -> change all (trade block 7*5=35).
N_SLOTS = 5


class ExecutionAdapter:
    """Per-symbol 5-slot -> broker-ticket manager."""

    def __init__(self, broker: BrokerAdapter, symbols: List[str], n_slots: int = N_SLOTS):
        self.broker = broker
        self.n_slots = n_slots
        # slots[symbol][i] = ticket (int) or None when free
        self.slots: Dict[str, List[Optional[int]]] = {s: [None] * n_slots for s in symbols}

    def n_open(self, symbol: str) -> int:
        return sum(1 for t in self.slots[symbol] if t is not None)

    def free_slot(self, symbol: str) -> Optional[int]:
        for i, t in enumerate(self.slots[symbol]):
            if t is None:
                return i
        return None

    # COUPLING -> quantra/live_bridge/live_session.py + live_runner.py: callers depend on
    # open()->slot index|None and close()->bool; live_session mirrors the returned slot into LivePortfolio.
    def open(self, symbol: str, side: int, lots: float, price: float = 0.0) -> Optional[int]:
        """Fill the next free slot. Returns the slot index, or None if all 5 are full."""
        i = self.free_slot(symbol)
        if i is None:
            return None                       # all slots full -> OPEN refused (mirrors the mask)
        ticket = self.broker.market_order(symbol, side, lots, price)
        self.slots[symbol][i] = ticket
        return i

    def close(self, symbol: str, pointer: int, price: float = 0.0) -> bool:
        """Close the pointer-selected slot's ticket and free the slot."""
        if not (0 <= pointer < self.n_slots):
            return False
        ticket = self.slots[symbol][pointer]
        if ticket is None:
            return False
        ok = self.broker.close_position(ticket, price)
        if ok:
            self.slots[symbol][pointer] = None
        return ok

    def close_all(self, price: float = 0.0) -> int:
        """Flatten every slot across all symbols (breach auto-flat / halt)."""
        n = 0
        for symbol, slots in self.slots.items():
            for i, ticket in enumerate(slots):
                if ticket is not None and self.broker.close_position(ticket, price):
                    slots[i] = None
                    n += 1
        return n


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M14 — implemented the ExecutionAdapter.
#   I: Nothing mapped the 5-slot pointer model onto live broker tickets.
#   R: SOW B2 (5 slots; CLOSE routes to the pointer slot; OPEN fills next free / masked at 5).
#   A: slot<->ticket bookkeeping: open (next free), close (pointer slot), close_all, n_open.
#   C: Live slot mechanics mirror training exactly, so the behaviour that passed in
#      training reproduces live - the whole point of training under faithful physics.
