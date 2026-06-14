"""quantra.diagnostics.llm_risk_doctor  —  SOW tier: 05_diagnostics/llm_risk_doctor.

WHAT THIS PACKAGE DOES
----------------------
The offline, READ-ONLY supervisory diagnosis module: reads telemetry + interpreter artifacts + (read-only) any repo file, and emits a structured diagnosis using the output template and the 8-failure taxonomy.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
It catches models that pass by luck or drift toward the wall BEFORE they cost a real challenge. It MUST read the rulebook first and MAY NEVER write, execute, or touch masks/sizing/walls (hard boundary).

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M11] Export the read-only diagnoser.
from .doctor import TAXONOMY, UNCLASSIFIED, Diagnosis, LLMRiskDoctor

__all__ = ["LLMRiskDoctor", "Diagnosis", "TAXONOMY", "UNCLASSIFIED"]


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
