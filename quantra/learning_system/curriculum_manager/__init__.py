"""quantra.learning_system.curriculum_manager  —  SOW tier: 04_learning_system/curriculum_manager.

WHAT THIS PACKAGE DOES
----------------------
Stage controller (trend -> reversion/pullback -> stationarity+ATR) running law-school permission mode and early 1m-feature masking; full-chart always.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Structure-first learning (trade only inside the stage's law context) builds a bot that respects the laws by habit before it is allowed fine 1m timing — fewer law-adjacent mistakes, higher pass rate (SOW F1-F5).

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M7] Export the curriculum API (consumed by the trainer, M8).
from .curriculum import DEFAULT_STAGES, CurriculumManager, Stage

__all__ = ["CurriculumManager", "Stage", "DEFAULT_STAGES"]


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Package documented under the new IRAC rule.
#   I: Scaffolded in M0 with a header docstring but no standing change-log, so future FTMO-relevant implementation could drift undocumented.
#   R: SOW R2-R4 + the new IRAC update-log rule (2026-06-13).
#   A: Confirmed the header states the package's FTMO role + the LLM rulebook pointer; added this IRAC log as the permanent change-story anchor for when real code lands.
#   C: A documented, discoverable package keeps its future implementation aligned to repeated FTMO passing and prevents silent, bug-introducing drift.
