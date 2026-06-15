"""Failure Atlas + Pass-Day Atlas builders (required visuals 6 & 7).

WHAT THIS MODULE DOES
---------------------
The standardized multi-panel atlases from the data contract: the FAILURE ATLAS (breach
/ stagnation / law-adjacent episodes) and the PASS-DAY ATLAS (clean pass days). The
panel rendering lives in ``MLPInterpreter`` (so the 7 visuals share one code path); this
module is the named, callable entry point the SOW J2 module list expects, delegating to
the interpreter.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Standardizing what breaches and clean passes LOOK like internally turns one-off mistakes
into a classifiable library, so recurring failure modes get fixed and pass behaviour gets
reinforced — the loop that drives the pass rate up.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. The Failure Atlas is your first stop
when a breach episode appears; the Pass-Day Atlas shows the stable recurring signature a
healthy passer leaves behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from quantra.diagnostics.mlp_interpreter.interpreter import MLPInterpreter


# COUPLING -> quantra/diagnostics/mlp_interpreter/interpreter.py: delegates to MLPInterpreter's
# constructor (records, out_dir=) + its failure_atlas()/pass_day_atlas() method names; rename there => fix here.
def failure_atlas(records: List[dict], out_dir: Optional[Path] = None) -> Path:
    """Build the Failure Atlas PNG from a telemetry run (visual 6)."""
    return MLPInterpreter(records, out_dir=out_dir).failure_atlas()


def pass_day_atlas(records: List[dict], out_dir: Optional[Path] = None) -> Path:
    """Build the Pass-Day Atlas PNG from a telemetry run (visual 7)."""
    return MLPInterpreter(records, out_dir=out_dir).pass_day_atlas()


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M10b — gave the SOW-named failure_atlas module real content.
#   I: The failure_atlas package was an empty shell (the atlases lived only in the
#      interpreter), i.e. a dummy package in a SOW-named slot.
#   R: SOW J2 lists failure_atlas as a diagnostics module + the 7-visual contract.
#   A: Thin failure_atlas()/pass_day_atlas() builders delegating to MLPInterpreter, so
#      the named module is callable + real; no duplicate rendering code.
#   C: The atlases are reachable by their contract name with no dead shell - keeping the
#      diagnostics that catch fragile (non-passing) behaviour honest and discoverable.
