"""Scoreboard — the ONLY ranking that matters. 🔴

WHAT THIS MODULE DOES
---------------------
Ranks runs strictly by the locked scoreboard hierarchy (SOW §1.3 / I1):
  1. FTMO pass rate          (higher better)
  2. Breach count            (lower better)
  3. Target-hit consistency  (higher better)
  4. Max drawdown path       (lower better)
Raw PnL is NEVER a ranking criterion — it is a diagnostic sanity check only.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Selecting brains by pass-rate (not profit) is THE mechanism that keeps the whole
project pointed at the mission. A bot that makes more money but breaches more, or
passes once luckily, ranks BELOW a steadier passer. This is what makes "repeatedly
pass" the optimization target rather than PnL.

🔴 LOCKED ranking order. PnL may not enter the key.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md`` (Term 8 Outcome). Cite scoreboard
metrics as pure facts in the "What happened" section; never rank by PnL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class RunResult:
    """One (window, seed) episode outcome — the raw input to the scoreboard."""

    # COUPLING -> quantra/ftmo_passing/validation/walk_forward.py: eval_fn there builds
    # RunResult(...) and WalkForwardRunner.run reads r.passed; these field NAMES + the
    # `passed` semantics are the contract. Removing/renaming a field breaks the runner.
    passed: bool          # hit DAILY_TARGET and avoided the trailing breach
    breached: bool        # touched the 4% wall
    target_hit: bool      # reached the daily target at least once
    max_drawdown: float   # worst trailing drawdown path (fraction; lower better)
    pnl: float = 0.0      # diagnostic only — NEVER used to rank


@dataclass
class Scoreboard:
    """Aggregates RunResults into the 4 ranking metrics."""

    results: List[RunResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / self.n if self.n else 0.0

    # COUPLING -> quantra/ftmo_passing/validation/walk_forward.py: PromotionGate.promote
    # compares candidate.breach_count vs baseline.breach_count (the "no worse breach" I3 gate).
    @property
    def breach_count(self) -> int:
        return sum(r.breached for r in self.results)

    @property
    def target_hit_consistency(self) -> float:
        return sum(r.target_hit for r in self.results) / self.n if self.n else 0.0

    @property
    def max_drawdown_path(self) -> float:
        return max((r.max_drawdown for r in self.results), default=0.0)

    def rank_key(self) -> tuple:
        """Lexicographic key (all 'higher is better' after sign flips). PnL excluded."""
        return (self.pass_rate, -self.breach_count,
                self.target_hit_consistency, -self.max_drawdown_path)

    # COUPLING -> quantra/ftmo_passing/validation/walk_forward.py: PromotionGate.promote
    # calls candidate.better_than(baseline) as the "scoreboard improved" I3 gate.
    def better_than(self, other: "Scoreboard") -> bool:
        return self.rank_key() > other.rank_key()

    def summary(self) -> dict:
        return {"pass_rate": self.pass_rate, "breaches": self.breach_count,
                "target_hit_consistency": self.target_hit_consistency,
                "max_drawdown_path": self.max_drawdown_path, "n": self.n}


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M12 — implemented the Scoreboard.
#   I: Nothing ranked runs by the mission metric; PnL could sneak in as a selector.
#   R: SOW §1.3/I1 (pass-rate -> breach -> consistency -> max-DD; PnL diagnostic only).
#   A: RunResult + Scoreboard metrics + lexicographic rank_key (PnL excluded) + better_than.
#   C: Brains are selected for REPEATED PASSING, not profit - the single choice that keeps
#      the whole system optimizing the mission.
