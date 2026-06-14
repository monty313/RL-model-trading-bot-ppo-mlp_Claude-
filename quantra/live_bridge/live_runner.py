"""LiveRunner — deterministic live inference + execution, isolated from diagnostics. 🔴

WHAT THIS MODULE DOES
---------------------
The live deployment loop (SOW §2.10/§10/§12.1): for each symbol it runs the policy
DETERMINISTICALLY (argmax direction · Beta-mean size · argmax pointer on CLOSE), clips
the size through the RiskManager, and executes via the ExecutionAdapter. It honors the
two hard kill switches — the always-available ManualHalt and the 4% breach auto-flat —
and is fully ISOLATED from the diagnostics layer (no telemetry coupling at runtime;
the Risk Doctor only ever reads checkpointed telemetry, SOW C7/§12.3).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Live is where passes are banked. Deterministic inference + the SAME masks + the SAME
slot mechanics as training make the learned pass-behaviour reproduce live; the kill
switches make a single bad session non-fatal. Isolation keeps the supervisory LLM from
ever touching execution.

🔴 LOCKED: live determinism, RiskManager clip, the two kill switches, diagnostics isolation.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. This module is OFF-LIMITS to you at
runtime. You read only checkpointed telemetry, never the live runner.
"""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from quantra.learning_system.ppo_agent.agent import PPOAgent
from quantra.live_bridge.execution_adapter import ExecutionAdapter
from quantra.live_bridge.manual_halt import ManualHalt
from quantra.locked_core.risk_manager.risk import RiskManager
from quantra.market_pipeline.law_mask_engine.engine import CLOSE, OPEN_LONG, OPEN_SHORT


class LiveRunner:
    """Deterministic per-symbol live execution with hard kill switches."""

    def __init__(self, agent: PPOAgent, execution: ExecutionAdapter,
                 risk: RiskManager, halt: Optional[ManualHalt] = None):
        self.agent = agent
        self.exec = execution
        self.risk = risk
        self.halt = halt or ManualHalt()

    def step(self, symbol, obs, dir_mask, ptr_mask, atr_price, price, remaining_budget):
        """One deterministic live decision for ``symbol``. Returns an info dict."""
        if self.halt.is_halted:
            return {"action": "HALTED"}
        a_dir, a_size, a_ptr, value = self.agent.act_deterministic(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(dir_mask, dtype=torch.float32),
            torch.as_tensor(ptr_mask, dtype=torch.float32))
        a_dir, a_size, a_ptr = int(a_dir[0]), float(a_size[0]), int(a_ptr[0])

        if a_dir in (OPEN_LONG, OPEN_SHORT):
            sr = self.risk.size(symbol, a_size, atr_price, remaining_budget)  # RiskManager clip
            if not sr.feasible:
                return {"action": "OPEN_SKIPPED", "reason": sr.reason}
            side = 1 if a_dir == OPEN_LONG else -1
            slot = self.exec.open(symbol, side, sr.lots, price)
            return {"action": "OPEN", "side": side, "lots": sr.lots, "slot": slot}
        if a_dir == CLOSE:
            ok = self.exec.close(symbol, a_ptr, price)
            return {"action": "CLOSE", "slot": a_ptr, "ok": ok}
        return {"action": "HOLD"}

    def breach_autoflat(self, equity: float, wall_equity: float, price: float = 0.0) -> bool:
        """Hard kill switch #2: at/below the 4% wall, flatten ALL + latch halted."""
        if equity <= wall_equity:
            self.exec.close_all(price)
            self.halt._halted = True            # lock out for the day (manual reset)
            return True
        return False

    def manual_halt(self, price: float = 0.0) -> int:
        """Hard kill switch #1: operator halt -> flatten all via the broker."""
        return self.halt.halt(self.exec.broker, price)


def main() -> None:  # pragma: no cover - operator entry point (SOW §12.1)
    ap = argparse.ArgumentParser(description="Quantra live runner (deterministic).")
    ap.add_argument("--symbols", default="EURUSD,XAUUSD,GBPUSD,US30")
    ap.add_argument("--daily_target_pct", type=float, default=2.5)
    ap.add_argument("--daily_risk_pct", type=float, default=4.0)
    ap.add_argument("--ftmo_account_size", type=float, default=10_000.0)
    ap.add_argument("--broker", default="sim", choices=["sim", "mt5"])
    args = ap.parse_args()
    print("Quantra live runner configured (no live data feed wired in this build):")
    print(f"  symbols={args.symbols} target={args.daily_target_pct}% "
          f"risk={args.daily_risk_pct}% account={args.ftmo_account_size} broker={args.broker}")
    print("  Connect a live 1m feed + checkpoint to begin. Manual halt + breach auto-flat armed.")


if __name__ == "__main__":  # pragma: no cover
    main()


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M14 — implemented the LiveRunner.
#   I: Nothing ran the policy deterministically against a live broker with the kill
#      switches, isolated from diagnostics.
#   R: SOW §2.10 (live determinism) + H3 (RiskManager clip) + §10.1 (manual halt +
#      breach auto-flat) + C7/§12.3 (isolation from diagnostics).
#   A: LiveRunner.step (argmax/Beta-mean/argmax -> risk clip -> ExecutionAdapter),
#      breach_autoflat, manual_halt; argparse CLI per §12.1; no diagnostics import.
#   C: The learned pass-behaviour reproduces live with the same masks/slots, and two
#      hard kill switches make a bad session non-fatal - so passes get banked safely.
