"""TelemetryLogger — the versioned per-step/-trade/-day data contract. 🔴 contract

WHAT THIS MODULE DOES
---------------------
Records the structured evidence the interpretability layer reasons from, exactly per
the REQUIRED DATA CONTRACT in ``docs/MLP_INTERPRETABILITY_LAYER.md``. Every step packet
carries: IDs (run/seed/window/episode/timestep/symbol), time, the full normalized
observation + grouped block names, law/gate states + enforcement mode + legal actions,
the policy outputs (pre/post-mask logits, probs, chosen action, pointer output, raw_size,
feasible size), V(s), trunk hidden summary, the full reward decomposition + QUAD signal
states, the risk-context snapshot (all Term 6 attributes), and the short-horizon outcome.
Serializes to JSONL (round-trip preserves every field); the schema is VERSIONED.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
No telemetry -> no interpretability -> no diagnosis (Term 9). If a breach or a pass-day
isn't fully logged, the Risk Doctor can't reconstruct WHY and the same failure recurs.
Capturing the whole chain is what lets us prove Layer-0 dominance, find the danger-
blindness window, and fix the cause before it costs a challenge.

🔴 The data-contract fields and the version may not be dropped without Monty's approval.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Before any diagnosis, check the
``schema_version`` and that the events you analyse were fully logged; a diagnosis on
partial telemetry MUST say so. The header packet holds the grouped block names so you
can map any observation index to its feature.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py: header logs
# SCHEMA.blocks/feature_names; if the observation layout changes, regenerate telemetry.
from quantra.market_pipeline.feature_builder.schema import SCHEMA
from quantra.runtime import config as cfg

# COUPLING [C8] -> quantra/diagnostics/mlp_interpreter/interpreter.py + llm_risk_doctor/doctor.py:
# they read this schema_version + StepPacket fields by exact name; bump => update both readers.
SCHEMA_VERSION = "1.0.0"   # 🔴 versioned; no field removed without approval


def _jsonable(v: Any) -> Any:
    """Make numpy arrays / scalars JSON-serializable (round-trip safe)."""
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# COUPLING [C8] -> quantra/diagnostics/mlp_interpreter/interpreter.py + llm_risk_doctor/doctor.py:
# every field name below is read by exact key there (observation, hidden_summary, value, chosen_action,
# pre/post_mask_logits, reward_decomposition, risk_context["trailing_dd"], law_states, raw_size, outcome).
# Rename/drop any field => break both readers.
@dataclass
class StepPacket:
    """One decision step — every field required by the data contract."""

    # IDs + time
    run_id: str
    seed: int
    window_id: str
    episode_id: int
    timestep: int
    symbol: str
    timestamp: str
    bar_index: int
    # observation
    observation: list                      # full normalized state vector (STATE_DIM == 215)
    # law context
    law_states: list                       # the 12 law/gate states
    enforcement_mode: str                  # "live" | "school"
    legal_actions: list                    # direction legality before sampling
    # policy outputs
    pre_mask_logits: list
    post_mask_logits: list
    action_probs: list
    chosen_action: int
    pointer_output: Optional[int]          # which slot on CLOSE (else None)
    raw_size: float
    feasible_size: float                   # lots after RiskManager
    # critic + trunk
    value: float
    hidden_summary: list                   # per-layer activation summary
    # reward + risk + outcome
    reward_decomposition: Dict[str, float]
    quad_signals: Dict[str, float]
    risk_context: Dict[str, float]         # all Term 6 attributes
    outcome: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: _jsonable(v) for k, v in asdict(self).items()}


class TelemetryLogger:
    """Appends step/trade/day packets to a versioned JSONL run file; reads them back."""

    def __init__(self, run_id: str, seed: int = 0, window_id: str = "w0",
                 out_dir: Optional[Path] = None):
        cfg.ensure_dirs()
        self.run_id, self.seed, self.window_id = run_id, seed, window_id
        self.path = (out_dir or cfg.TELEMETRY_DIR) / f"{run_id}.jsonl"
        self._buf: List[dict] = []
        # Run header: schema version + the grouped feature block names (contract).
        # COUPLING [C8] -> quantra/diagnostics/mlp_interpreter/interpreter.py + llm_risk_doctor/doctor.py:
        # they pick out kind=="header"/"step" and read run_id/schema_version; keep these key strings stable.
        self._buf.append({"kind": "header", "schema_version": SCHEMA_VERSION,
                          "run_id": run_id, "seed": seed, "window_id": window_id,
                          "blocks": {name: names for name, names in SCHEMA.blocks.items()},
                          "feature_names": SCHEMA.feature_names})

    def log_step(self, packet: StepPacket) -> None:
        # COUPLING [C8] -> quantra/diagnostics/mlp_interpreter/interpreter.py + llm_risk_doctor/doctor.py:
        # both filter records by r.get("kind")=="step"; this literal must stay "step".
        d = packet.to_dict(); d["kind"] = "step"; self._buf.append(d)

    def log_trade(self, trade: Dict[str, Any]) -> None:
        d = _jsonable(trade); d["kind"] = "trade"; self._buf.append(d)

    def log_day(self, day: Dict[str, Any]) -> None:
        d = _jsonable(day); d["kind"] = "day"; self._buf.append(d)

    def flush(self) -> Path:
        with open(self.path, "w", encoding="utf-8") as fh:
            for rec in self._buf:
                fh.write(json.dumps(rec) + "\n")
        return self.path

    @staticmethod
    def load(path: Path) -> List[dict]:
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M9 — implemented the versioned TelemetryLogger.
#   I: The interpretability layer + Risk Doctor have nothing to reason from without a
#      complete, versioned record of the decision chain.
#   R: MLP_INTERPRETABILITY_LAYER.md REQUIRED DATA CONTRACT (every step field) + a
#      versioned schema that no refactor may silently shrink.
#   A: StepPacket with every contract field + header (block names); JSONL writer/reader
#      with numpy round-trip; per-trade + per-day packets.
#   C: Any breach/pass-day is fully reconstructable, so the Risk Doctor can find the
#      cause and prescribe a fix before it costs a challenge - the loop that keeps the
#      pass rate from silently eroding.
