"""quantra.locked_core.cost_layer  —  SOW tier: 01_locked_core/cost_layer  🔴.

WHAT THIS PACKAGE DOES
----------------------
Applies real FTMO costs from day 1: $5 round-trip per lot on forex, spread, and fixed slippage; metals/indices carry no per-trade commission.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
No costless world ever exists in training (SOW C8/H2), so the learned edge is real after fees — an edge that survives costs is one that can actually pass.

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M4] Export the CostLayer API (consumed by the env, M4).
from .costs import CostLayer, FillCosts

__all__ = ["CostLayer", "FillCosts"]


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
