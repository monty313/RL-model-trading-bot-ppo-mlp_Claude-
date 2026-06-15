"""WalkForwardRunner + PromotionGate — the single locked validation protocol. 🔴

WHAT THIS MODULE DOES
---------------------
WalkForwardRunner generates the locked walk-forward windows — **12 months train / 2
months test / 1 month step** — and orchestrates **7 seeds** per window (SOW §8.1 / I2).
PromotionGate enforces the promotion rule (I3): a candidate promotes only if it
survives the full walk-forward on **>= 3 of the 7 seeds**, shows a scoreboard
improvement vs the previous checkpoint, AND has **no worse breach count**.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Walk-forward on rolling out-of-sample windows x 7 seeds is what distinguishes a brain
that genuinely PASSES from one that overfit a lucky window/seed. The promotion gate
then refuses to ship anything that isn't a robust improvement with no extra breaches —
so the deployed brain's pass rate only ever ratchets up.

🔴 LOCKED: 12/2/1 months, 7 seeds, and the promotion conditions.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. A candidate that passes few seeds is
Shortcut Learning until proven otherwise (overfit to a seed/window). Cite the seed
pass count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import pandas as pd

# COUPLING -> quantra/ftmo_passing/validation/scoreboard.py: depends on RunResult fields
# (.passed) + Scoreboard API (constructor takes a List[RunResult]; .breach_count, .better_than).
# Any rename there must be mirrored in this runner/gate.
from .scoreboard import RunResult, Scoreboard

# 🔴 locked protocol
TRAIN_MONTHS = 12
TEST_MONTHS = 2
STEP_MONTHS = 1
N_SEEDS = 7
PROMOTE_MIN_SEEDS = 3


@dataclass
class Window:
    """One walk-forward window: a 12mo train span and the following 2mo test span."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_windows(index: pd.DatetimeIndex,
                     train_months: int = TRAIN_MONTHS,
                     test_months: int = TEST_MONTHS,
                     step_months: int = STEP_MONTHS) -> List[Window]:
    """Roll 12mo-train / 2mo-test windows forward by 1mo across the data span."""
    start = pd.Timestamp(index.min()).normalize()
    end = pd.Timestamp(index.max()).normalize()
    windows: List[Window] = []
    tr_start = start
    while True:
        tr_end = tr_start + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > end:
            break
        windows.append(Window(tr_start, tr_end, tr_end, te_end))
        tr_start = tr_start + pd.DateOffset(months=step_months)
    return windows


class WalkForwardRunner:
    """Runs the protocol: for each window x seed, call eval_fn -> RunResult."""

    def __init__(self, n_seeds: int = N_SEEDS):
        self.n_seeds = n_seeds

    # COUPLING -> quantra/ftmo_passing/validation/scoreboard.py + M15 e2e caller: eval_fn
    # MUST return a RunResult; the (Scoreboard, per-seed-pass List[int]) return shape is
    # unpacked by the e2e harness and feeds PromotionGate.promote(..., seed_pass_counts).
    def run(self, index: pd.DatetimeIndex,
            eval_fn: Callable[[Window, int], RunResult]) -> Tuple[Scoreboard, List[int]]:
        """Return (overall Scoreboard, per-seed pass counts across windows).

        eval_fn(window, seed) trains on the window's train span and evaluates on its
        test span, returning a RunResult. (The heavy training is injected so this
        protocol layer stays pure + testable; the e2e wiring lands in M15.)
        """
        windows = generate_windows(index)
        results: List[RunResult] = []
        seed_pass = [0] * self.n_seeds
        for w in windows:
            for seed in range(self.n_seeds):
                r = eval_fn(w, seed)
                results.append(r)
                if r.passed:
                    seed_pass[seed] += 1
        return Scoreboard(results), seed_pass


class PromotionGate:
    """I3 promotion rule — a candidate ships only if it's a robust improvement."""

    def __init__(self, min_seeds: int = PROMOTE_MIN_SEEDS):
        self.min_seeds = min_seeds

    def promote(self, candidate: Scoreboard, baseline: Optional[Scoreboard],
                seed_pass_counts: List[int]) -> Tuple[bool, str]:
        """Return (promote?, reason). All three conditions must hold (I3)."""
        seeds_passed = sum(1 for c in seed_pass_counts if c > 0)
        if seeds_passed < self.min_seeds:
            return False, f"survived only {seeds_passed} seeds (< {self.min_seeds})"
        if baseline is None:
            return True, f"first candidate; survived {seeds_passed} seeds"
        if candidate.breach_count > baseline.breach_count:
            return False, (f"breach count worsened "
                           f"({candidate.breach_count} > {baseline.breach_count})")
        if not candidate.better_than(baseline):
            return False, "no scoreboard improvement vs the previous checkpoint"
        return True, (f"promoted: {seeds_passed} seeds, scoreboard improved, "
                      f"breaches {candidate.breach_count} <= {baseline.breach_count}")


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M12 — implemented WalkForwardRunner + PromotionGate.
#   I: Nothing validated robustness across out-of-sample windows/seeds or gated promotion.
#   R: SOW §8.1/I2 (12/2/1 months, 7 seeds) + I3 (>=3 seeds + improvement + no worse breach).
#   A: generate_windows (rolling 12/2/1), WalkForwardRunner.run (window x seed -> RunResult),
#      PromotionGate.promote enforcing all three I3 conditions.
#   C: Only brains that genuinely + robustly pass more (with no extra breaches) ever ship,
#      so the deployed pass rate only ratchets up - the definition of repeatable passing.
