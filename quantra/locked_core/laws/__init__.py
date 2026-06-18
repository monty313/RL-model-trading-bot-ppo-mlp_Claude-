"""quantra.locked_core.laws  —  SOW tier: 01_locked_core/laws  🔴.

WHAT THIS PACKAGE DOES
----------------------
The 9 directional laws (3 super-trend, 3 trend, 3 pullback) + 3 market-condition observation signals (Volatility, Spread, Stationarity — formerly "gates", now observation-only by default) and the LawMask that turns law states into pre-mask logits (logit = -1e9 on forbidden directions).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Laws run BEFORE the policy and forbid the directions that would breach the wall — directional stupidity is masked out, never merely penalised. They are masks, never reward terms (SOW R5).

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M3] Export the law-state API. laws.py computes the 12 states from
# the M2 features; the mask engine (market_pipeline/law_mask_engine) consumes them.
from .laws import (
    DIRECTIONAL_LAWS,
    LAW_NAMES,
    MARKET_SIGNALS,
    compute_law_states,
    law_states_dict,
)

__all__ = ["LAW_NAMES", "DIRECTIONAL_LAWS", "MARKET_SIGNALS", "compute_law_states", "law_states_dict"]


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
