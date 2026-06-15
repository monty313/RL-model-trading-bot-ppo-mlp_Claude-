"""ManualHalt — the operator's always-available hard kill switch. 🔴

WHAT THIS MODULE DOES
---------------------
A hard kill switch (SOW §10.1): ``halt()`` force-flattens every open position via the
broker adapter and latches the system HALTED so no new orders can be placed until a
manual ``reset()``. Always available, always immediate.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
A funded challenge can be lost in seconds to a runaway. The manual halt is the
operator's guaranteed circuit-breaker — independent of the policy, the reward, and the
diagnostics — so a single bad session can never blow the account. Protecting the
account is the precondition for getting to pass it again.

🔴 LOCKED kill switch. The LLM Risk Doctor may never trigger or override it.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. You may recommend the operator halt;
you may NEVER call halt()/reset() yourself (no execution authority).
"""

from __future__ import annotations


class ManualHalt:
    """Operator circuit-breaker. Flattens everything and latches until reset."""

    def __init__(self):
        # COUPLING -> quantra/live_bridge/live_runner.py + live_session.py: they read .is_halted
        # and set ._halted = True directly on breach auto-flat; keep this attr name + property in sync.
        self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted

    def halt(self, broker, price: float = 0.0) -> int:
        """Force-flatten ALL open positions and latch HALTED. Returns # closed."""
        closed = 0
        # COUPLING -> quantra/locked_core/platform_adapter/adapters.py: relies on broker.positions()
        # yielding objects with a .ticket attr + close_position(ticket)->bool. Called via live_runner.manual_halt.
        for pos in list(broker.positions()):
            if broker.close_position(pos.ticket, price):
                closed += 1
        self._halted = True
        return closed

    def reset(self) -> None:
        """Manual reset only — the bot can never un-halt itself."""
        self._halted = False


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M14 — implemented the ManualHalt kill switch.
#   I: The operator had no guaranteed, immediate way to stop the bot and flatten.
#   R: SOW §10.1 (manual halt = hard kill, always available; locks until manual reset).
#   A: halt() flattens every broker position + latches HALTED; reset() is manual-only.
#   C: A runaway session can be stopped instantly, protecting the funded account - the
#      precondition for living to pass another challenge.
