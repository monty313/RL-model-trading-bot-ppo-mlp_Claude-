"""MLPInterpreter — the 7 required visuals from telemetry. 🔴 contract

WHAT THIS MODULE DOES
---------------------
Converts a telemetry run (TelemetryLogger JSONL) into the 7 standard artifacts
(MLP_INTERPRETABILITY_LAYER.md "REQUIRED VISUALS"):
  1. Activation Trace        4. Reward Layer Timeline   7. Pass-Day Atlas
  2. Hidden-State Projection  5. Correlation Heatmap
  3. Action/Value Timeline    6. Failure Atlas
Uses matplotlib (Agg, no display) + a numpy-SVD PCA (no sklearn dependency).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Raw telemetry arrays explain nothing. These visuals make the FTMO question answerable:
did the internal behaviour help the bot pass? — clustering by breach-risk (collapse?),
Layer-0 dominance over time (reward hijack?), value softening near danger (critic ok?),
and recurring pass-day signatures. They are the evidence the Risk Doctor reasons from.

🔴 The 7 visuals are a contract item; each must stay tied to PASSING, not be pretty-but-
unconnected.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Read the visuals along the reverse
chain: Reward Layer Timeline (L0 dominant?) -> Action/Value Timeline (confidence/value
near danger?) -> Hidden-State Projection (breach-risk separable?) -> Failure Atlas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")           # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from quantra.runtime import config as cfg


def _pca2(x: np.ndarray) -> np.ndarray:
    """2-component PCA via SVD (no sklearn). Returns (N, 2)."""
    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2))
    xc = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:2].T


class MLPInterpreter:
    """Builds the 7 visuals from a list of telemetry records (header + step packets)."""

    def __init__(self, records: List[dict], out_dir: Optional[Path] = None):
        cfg.ensure_dirs()
        self.out_dir = out_dir or cfg.REPORT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # COUPLING [C8] -> quantra/diagnostics/telemetry_logger/logger.py: the "kind" tags
        # ("step"/"header") and every field key read below are produced by TelemetryLogger/StepPacket.
        self.steps = [r for r in records if r.get("kind") == "step"]
        self.header = next((r for r in records if r.get("kind") == "header"), {})

    # --- extractors ---
    def _col(self, key: str) -> np.ndarray:
        return np.array([s[key] for s in self.steps], dtype=float)

    def _obs(self) -> np.ndarray:
        return np.array([s["observation"] for s in self.steps], dtype=float)

    # COUPLING [C8] -> quantra/diagnostics/telemetry_logger/logger.py: reads the
    # "reward_decomposition" dict key + its layer names (e.g. "L0"); set by StepPacket.
    def _reward_layers(self) -> Dict[str, np.ndarray]:
        keys: set = set()
        for s in self.steps:
            keys |= set(s["reward_decomposition"].keys())
        return {k: np.array([s["reward_decomposition"].get(k, 0.0) for s in self.steps])
                for k in sorted(keys)}

    def _save(self, fig, name: str) -> Path:
        path = self.out_dir / f"{name}.png"
        fig.savefig(path, dpi=80, bbox_inches="tight")
        plt.close(fig)
        return path

    # --- 1. Activation Trace ---
    def activation_trace(self) -> Path:
        h = np.array([s["hidden_summary"] for s in self.steps], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 3))
        for i in range(min(h.shape[1], 8)):
            ax.plot(h[:, i], lw=0.8, label=f"h{i}")
        ax.set_title("Activation Trace (hidden summary over time)")
        ax.set_xlabel("step"); ax.legend(fontsize=6, ncol=4)
        return self._save(fig, "1_activation_trace")

    # --- 2. Hidden-State Projection (PCA) colored by action ---
    def hidden_state_projection(self) -> Path:
        proj = _pca2(self._obs())
        act = self._col("chosen_action")
        fig, ax = plt.subplots(figsize=(5, 4))
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=act, cmap="viridis", s=10)
        ax.set_title("Hidden-State Projection (PCA, colour=action)")
        fig.colorbar(sc, ax=ax, label="chosen action")
        return self._save(fig, "2_hidden_state_projection")

    # --- 3. Action/Value Timeline ---
    def action_value_timeline(self) -> Path:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
        a1.plot(self._col("value"), color="tab:blue"); a1.set_ylabel("V(s)")
        a1.set_title("Action / Value Timeline")
        a2.plot(self._col("chosen_action"), drawstyle="steps-post", color="tab:orange")
        a2.set_ylabel("action"); a2.set_xlabel("step")
        return self._save(fig, "3_action_value_timeline")

    # --- 4. Reward Layer Timeline ---
    def reward_layer_timeline(self) -> Path:
        layers = self._reward_layers()
        fig, ax = plt.subplots(figsize=(7, 3))
        for k, v in layers.items():
            ax.plot(np.cumsum(v), lw=1.0, label=k)
        ax.set_title("Reward Layer Timeline (cumulative; L0 should dominate)")
        ax.set_xlabel("step"); ax.legend(fontsize=6, ncol=4)
        return self._save(fig, "4_reward_layer_timeline")

    # --- 5. Correlation Heatmap (reward layers vs value) ---
    def correlation_heatmap(self) -> Path:
        layers = self._reward_layers()
        names = list(layers.keys()) + ["value"]
        mat = np.vstack([*layers.values(), self._col("value")])
        with np.errstate(invalid="ignore", divide="ignore"):  # constant layers -> 0 std
            corr = np.nan_to_num(np.corrcoef(mat))
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6)
        ax.set_title("Correlation Heatmap")
        fig.colorbar(im, ax=ax)
        return self._save(fig, "5_correlation_heatmap")

    # --- 6. Failure Atlas (value + cumulative L0 around the run end) ---
    def failure_atlas(self) -> Path:
        layers = self._reward_layers()
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
        a1.plot(self._col("value"), color="tab:red"); a1.set_ylabel("V(s)")
        a1.set_title("Failure Atlas (breach context)")
        a2.plot(np.cumsum(layers.get("L0", np.zeros(len(self.steps)))), color="black")
        a2.set_ylabel("cum L0"); a2.set_xlabel("step")
        return self._save(fig, "6_failure_atlas")

    # --- 7. Pass-Day Atlas ---
    def pass_day_atlas(self) -> Path:
        # COUPLING [C8] -> quantra/diagnostics/telemetry_logger/logger.py: reads risk_context["trailing_dd"]
        # (same key llm_risk_doctor uses); renaming it in StepPacket.risk_context breaks this panel.
        risk = np.array([s["risk_context"].get("trailing_dd", 0.0) for s in self.steps])
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
        a1.plot(self._col("value"), color="tab:green"); a1.set_ylabel("V(s)")
        a1.set_title("Pass-Day Atlas (clean day signature)")
        a2.plot(risk, color="tab:purple"); a2.set_ylabel("trailing DD"); a2.set_xlabel("step")
        return self._save(fig, "7_pass_day_atlas")

    def generate_all(self) -> Dict[str, Path]:
        return {
            "activation_trace": self.activation_trace(),
            "hidden_state_projection": self.hidden_state_projection(),
            "action_value_timeline": self.action_value_timeline(),
            "reward_layer_timeline": self.reward_layer_timeline(),
            "correlation_heatmap": self.correlation_heatmap(),
            "failure_atlas": self.failure_atlas(),
            "pass_day_atlas": self.pass_day_atlas(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M10 — implemented the 7 required visuals.
#   I: Telemetry arrays don't explain anything; the operator + Risk Doctor need the
#      standard visual evidence tied to passing.
#   R: MLP_INTERPRETABILITY_LAYER.md REQUIRED VISUALS (the 7) - each connected to PASSING.
#   A: MLPInterpreter producing all 7 PNGs from a telemetry run (matplotlib Agg + numpy
#      PCA); reward-layer timeline shows L0 dominance, projection colours by action, etc.
#   C: The bot's internal behaviour becomes inspectable evidence, so the Risk Doctor can
#      tell robust pass-behaviour from fragile luck and prescribe fixes that protect the
#      pass rate.
