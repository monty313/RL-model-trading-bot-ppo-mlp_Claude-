"""quantra.learning_system.policy_registry — SOW tier 04_learning_system / policy_registry.

The saved IDENTITY of a trained policy ("what is this policy's perspective on how to pass the FTMO
challenge?") + a Leaderboard that ranks every saved policy by how well it actually passed. Implements
the committed contract in artifacts/policy_registry/README.md (PROJECT_GUIDE §4.11).

COUPLING (both directions):
  -> runtime/config.py: writes under cfg.POLICY_REGISTRY_DIR; auto-names diff a run's OVERRIDES vs
     the baseline built from config defaults (baseline_overrides()).
  -> learning_system/reward_engine/reward.py: the compatibility signature fingerprints the reward
     LAYER arrangement (decompose() L-keys), so re-pointing weights/math (C16/C17) stays resume-safe
     but adding/removing a layer forces a fresh start.
  -> locked_core/laws/laws.py: the signature also fingerprints LAW_NAMES.
  <- learning_system/trainer + colab/Quantra_Barbershop.ipynb (Cell 6): the WRITER — calls build_card()
     then card.record_pass()/card.save() each pass.
  <- diagnostics + the Barbershop dashboard: READERS — PolicyCard.load()/Leaderboard.from_dir().
"""

from quantra.learning_system.policy_registry.registry import (  # noqa: F401
    CompatibilityError,
    Leaderboard,
    LeaderboardRow,
    PassRecord,
    PolicyCard,
    auto_name,
    baseline_overrides,
    build_card,
    check_compatibility,
    compatibility_signature,
    default_law_fingerprint,
    default_reward_layer_keys,
)
