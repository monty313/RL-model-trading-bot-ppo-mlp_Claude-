"""quantra.learning_system.ppo_agent  —  SOW tier: 04_learning_system/ppo_agent.

WHAT THIS PACKAGE DOES
----------------------
The four-head PPO actor-critic: direction (categorical), size (Beta), pointer (categorical over 5 slots), value — on a shared 3x256 MLP trunk. Emits the summed log-prob of the three action heads.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The heads encode WHAT/HOW-MUCH/WHICH-to-close and the critic encodes patience; together, inside the masks, they are how the bot learns to hit target without breaching (PPO_ENGINE.md).

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M5] Export the agent + loss API (consumed by the trainer, M8).
from .agent import ActorCritic, AgentStep, PPOAgent
from .loss import ppo_loss

__all__ = ["PPOAgent", "ActorCritic", "AgentStep", "ppo_loss"]


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
