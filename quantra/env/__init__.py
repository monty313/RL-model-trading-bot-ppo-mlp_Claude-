"""quantra.env  —  the Quantra gym environment (supports SOW J2 module: Env).

WHAT THIS PACKAGE DOES
----------------------
The real-chart trading environment: 4 symbols stepped TRUE-SEQUENTIALLY each 1m
bar, 5 trade slots per symbol, ONE shared account block read by every symbol's
decision, with within-bar account updates so the symbols cannot collectively
overshoot the daily-risk buffer (SOW B5). Supports a vectorised mode (many parallel
worlds) so the runtime optimizer can drive ~80% utilisation.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The env IS the FTMO challenge made steppable: it enforces the two-phase episode
rule, applies real costs, exposes the shared-account risk picture, and routes
CLOSE via the pointer head. Training against faithful challenge physics is what
makes the learned behaviour transfer to passing real challenges.

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M4] Export the env API. trading_env.py wires features (M2), laws
# (M3), risk + costs (M4) into the steppable challenge. Consumed by the trainer (M8).
from .trading_env import Slot, SymbolData, TradingEnv, prepare_symbol_data

__all__ = ["TradingEnv", "SymbolData", "Slot", "prepare_symbol_data"]


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
