"""Aggression scheduler (G2 ranges) + the G8 missed-opportunity metric. 🔴 logic

WHAT THIS MODULE DOES
---------------------
The hand-locked aggression scheduler moves the law-school dials within their locked
ranges (G2): entropy 0.03-0.08, clip 0.25-0.35, LR 5e-4-1e-3, epochs 10-15. It is
driven by the G8 missed-opportunity metric: while the bot misses premium legal setups,
aggression stays HIGH (explore more); as it captures them, aggression COOLS.

G8 (SOW §7.5): a missed opportunity for a symbol is TRUE when the permitted direction
agrees across 5m AND 30m AND 4H, the bot was FLAT, and price then ran >= 1.5x ATR in
the permitted direction. TRAINING-ONLY — it never touches reward, masks, or live. 4H is
a confirmation lens here only; law activation is unchanged (4H Observation Rule).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Keeping exploration high while the bot leaves money on the table — then cooling as it
learns to take premium legal setups — gets it to a capturing, disciplined policy faster
and more stably, which is what produces a high, steady pass rate per training budget.

🔴 LOCKED: the dial RANGES and the scheduler LOGIC are hand-locked (off-limits to HPO).

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. A high miss-rate that won't fall is
Stagnation Blindness (the actor won't take premium legal setups). The scheduler is the
control loop; the *cause* is in the actor/critic, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COUPLING [C1] -> market_pipeline/feature_builder/schema.py: _COL is built from
# PRECOMPUTED_NAMES order and indexed by the ssma_align_5m/30m/4H names below; if
# schema renames/reorders those columns (or the builder stops emitting them) the G8
# metric reads the wrong column. feature_builder/builder.py must emit them in this order.
from quantra.market_pipeline.feature_builder.schema import PRECOMPUTED_NAMES

_COL = {n: i for i, n in enumerate(PRECOMPUTED_NAMES)}
G8_ATR_THRESHOLD = 1.5   # 🔴 locked: price must run >= 1.5x ATR in the permitted dir


# COUPLING -> trainer/trainer.py: collect_rollout calls missed_opportunity(data.matrix
# [t], True, move_atr) positionally; market_row must be a PRECOMPUTED_NAMES-ordered row.
def missed_opportunity(market_row, was_flat: bool, realized_move_atr: float,
                       threshold: float = G8_ATR_THRESHOLD) -> bool:
    """G8: multi-TF (5m+30m+4H) directional agreement + flat + move >= 1.5x ATR.

    Direction agreement uses the shifted-SMA alignment flags across the three TFs
    (a clean per-TF +1/-1/0 signal). ``realized_move_atr`` is the SIGNED forward move
    over the window in ATR units (positive = up). Training-only diagnostic.
    """
    d5 = float(market_row[_COL["ssma_align_5m"]])
    d30 = float(market_row[_COL["ssma_align_30m"]])
    d4h = float(market_row[_COL["ssma_align_4H"]])
    if d5 == 0 or not (d5 == d30 == d4h):     # all three must agree on a non-flat dir
        return False
    if not was_flat:
        return False
    return realized_move_atr * d5 >= threshold  # ran >= 1.5x ATR in the permitted dir


@dataclass(frozen=True)
class AggressionRanges:
    # COUPLING [C6-adjacent] -> learning_system/hpo/hpo.py: these ranges + the scheduler
    # logic are the "sacred" dials HPO must never tune — hpo.SACRED_DIALS lists
    # entropy_range/clip_range/lr_range/epochs_range/scheduler_logic to guard exactly this.
    """The locked G2 law-school dial ranges (low, high)."""

    entropy: tuple = (0.03, 0.08)
    clip: tuple = (0.25, 0.35)
    lr: tuple = (5e-4, 1e-3)
    epochs: tuple = (10, 15)


@dataclass
class DialValues:
    # COUPLING -> trainer/trainer.py (+ ppo_agent/loss.py): the trainer reads dials.lr/
    # clip_eps/entropy_coef/epochs by these field NAMES and forwards clip_eps/entropy_coef
    # into ppo_loss(clip_eps=, entropy_coef=). Renaming a field breaks the update loop.
    entropy_coef: float
    clip_eps: float
    lr: float
    epochs: int


class AggressionScheduler:
    """Maps a smoothed miss-rate (0..1) to dial values inside the locked ranges."""

    def __init__(self, ranges: AggressionRanges | None = None, start: float = 1.0):
        self.ranges = ranges or AggressionRanges()
        self.aggression = float(np.clip(start, 0.0, 1.0))  # 1 = max aggression

    def update(self, miss_rate: float, momentum: float = 0.2) -> None:
        """Move aggression toward the observed miss-rate (more misses -> stay hot)."""
        self.aggression = (1 - momentum) * self.aggression + momentum * float(np.clip(miss_rate, 0, 1))

    def _lerp(self, lo_hi: tuple) -> float:
        lo, hi = lo_hi
        return lo + (hi - lo) * self.aggression   # aggression 1 -> high end of the range

    def values(self) -> DialValues:
        return DialValues(
            entropy_coef=self._lerp(self.ranges.entropy),
            clip_eps=self._lerp(self.ranges.clip),
            lr=self._lerp(self.ranges.lr),
            epochs=int(round(self._lerp(self.ranges.epochs))),
        )


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M8 — implemented the G8 metric + aggression scheduler.
#   I: Nothing measured missed premium setups or modulated exploration within the
#      locked G2 ranges.
#   R: SOW §7.5 (G8: 5m+30m+4H agree + flat + >=1.5x ATR, training-only, 4H rule
#      protected) + G2 (locked ranges + scheduler logic, off-limits to HPO).
#   A: missed_opportunity() via the shifted-SMA alignment flags; AggressionScheduler
#      mapping a smoothed miss-rate to entropy/clip/LR/epochs within the ranges.
#   C: Exploration stays high while money is left on the table and cools as the bot
#      captures setups - reaching a disciplined, capturing policy (high pass rate) faster.
