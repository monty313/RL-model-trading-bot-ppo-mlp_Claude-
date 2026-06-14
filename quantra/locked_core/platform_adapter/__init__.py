"""quantra.locked_core.platform_adapter  —  SOW tier: 01_locked_core/platform_adapter  🔴.

WHAT THIS PACKAGE DOES
----------------------
The MT5 broker interface (live + simulation) — the platform-blind boundary between the policy's normalized intentions and concrete broker orders.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Keeping the policy platform-blind (it never sees lots/account size) is what lets ONE brain pass challenges on any account size or supported asset (SOW A2).

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M14] Export the broker adapters (consumed by the live bridge).
from .adapters import BrokerAdapter, MT5Adapter, Position, SimBrokerAdapter, make_adapter

__all__ = ["BrokerAdapter", "SimBrokerAdapter", "MT5Adapter", "Position", "make_adapter"]


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
