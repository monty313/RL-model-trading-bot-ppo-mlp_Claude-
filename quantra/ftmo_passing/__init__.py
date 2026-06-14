"""quantra.ftmo_passing  —  SOW tier: 02_ftmo_passing.

WHAT THIS PACKAGE DOES
----------------------
Defines what "winning" means: the ``scoreboard`` (pass-rate -> breach -> target-hit
consistency -> max-DD path), the ``challenge_state`` tracker (equity, buffers,
day PnL vs target), the two-phase ``episode_rule`` (Phase A 4% trailing until
+2.5% -> auto-flat all -> Phase B 1% trailing), and ``validation`` (walk-forward +
promotion gate).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
This tier IS the mission's measuring stick. It encodes the FTMO physics the bot
must respect and the ranking that selects brains by pass-rate rather than profit,
so training pressure always points at consistent passing.

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M4] Export the shared-account tracker (consumed by the env, M4).
from .challenge_state import ChallengeState

__all__ = ["ChallengeState"]


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
