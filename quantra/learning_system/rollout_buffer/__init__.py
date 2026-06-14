"""quantra.learning_system.rollout_buffer  —  SOW tier: 04_learning_system/rollout_buffer.

WHAT THIS PACKAGE DOES
----------------------
On-policy storage of (s, a_dir, a_size, a_ptr, r, s', logp_old, V_old, done, masks); no replay buffer.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
PPO is on-policy by design (SOW §2.1); replay would contaminate the gradient. Storing the masks + summed log-prob exactly is what keeps the patient, law-bounded policy update correct.

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M5] Export the RolloutBuffer (consumed by the trainer, M8).
from .buffer import FIELDS, RolloutBuffer

__all__ = ["RolloutBuffer", "FIELDS"]


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
