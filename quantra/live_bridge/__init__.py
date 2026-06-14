"""quantra.live_bridge  —  SOW tier: 06_live_bridge.

WHAT THIS PACKAGE DOES
----------------------
Isolated live deployment: ``live_runner`` (deterministic inference — argmax
direction · Beta-mean size · argmax pointer on CLOSE, clipped by RiskManager),
``execution_adapter`` (manages the 5 slots/symbol on the broker, routes CLOSE to
the pointer-selected slot), and ``manual_halt`` (always-available hard kill).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Live is where passes are actually banked. This tier keeps execution deterministic
and fully separated from diagnostics (SOW C7/I4), with hard kill switches (manual
halt + breach auto-flat) so a single bad session can't blow a funded account.

HARD BOUNDARY: the LLM Risk Doctor never touches this tier in real time — it reads
only checkpointed telemetry (SOW §12.3).

BINDING RULEBOOK FOR THE LLM RISK DOCTOR: ``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

# [C - 2026-06-13, M14] Export the live-bridge API.
from .execution_adapter import ExecutionAdapter
from .live_runner import LiveRunner
from .manual_halt import ManualHalt

__all__ = ["LiveRunner", "ExecutionAdapter", "ManualHalt"]


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
