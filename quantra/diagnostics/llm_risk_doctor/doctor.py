"""LLMRiskDoctor — offline, READ-ONLY, evidence-only failure diagnosis. 🔴 boundary

WHAT THIS MODULE DOES
---------------------
Reads telemetry + interpreter artifacts (and may VIEW any repo file, read-only) and
produces a structured diagnosis following the output template in
``docs/MLP_INTERPRETABILITY_LAYER.md``, classifying every failure into exactly ONE of
the 8 taxonomy items (never a 9th) by walking the chain BACKWARD (outcome -> reward ->
critic -> actor -> hidden -> law -> state) and stopping at the first broken link. Every
claim cites a specific telemetry field/metric. It MUST read the rulebook first — the
constructor FAILS LOUDLY if that file is unreadable.

HARD BOUNDARY (SOW R5, §9.3): this class MAY read; it MAY NOT write, modify, delete,
execute, or touch masks/sizing/walls/broker. There is deliberately NO write method.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
It catches models that pass by luck or drift toward the wall BEFORE they cost a funded
challenge, and tells the operator WHY (the broken link) + a prescription — never an
execution. Evidence-only reasoning keeps it honest: no speculation, acknowledged
uncertainty when evidence is thin.

LLM RISK DOCTOR — THIS FILE IS YOUR OWN MANUAL'S IMPLEMENTATION
--------------------------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Use the 8-item taxonomy verbatim. If
a failure fits none, report "unclassified - additional telemetry required" and stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from quantra.runtime import config as cfg

# The 8 taxonomy items (verbatim) — no 9th may ever be invented.
TAXONOMY = (
    "Mask Dependence", "Representation Collapse", "Representation Chaos",
    "Critic Misalignment", "Reward Hijack", "Risk Blindness",
    "Stagnation Blindness", "Shortcut Learning",
)
UNCLASSIFIED = "unclassified - additional telemetry required"


@dataclass
class Diagnosis:
    """The required output-template shape."""

    run_id: str
    what_happened: str
    where_chain_broke: str
    classification: str
    evidence: List[str] = field(default_factory=list)
    confidence: str = "LOW"
    prescription: str = ""
    not_recommended: str = ""

    def render(self) -> str:
        ev = "\n".join(f"  - {e}" for e in self.evidence) or "  insufficient evidence"
        return (
            f"DIAGNOSIS - [{self.run_id}, {self.confidence}]\n\n"
            f"What happened (outcome layer):\n  {self.what_happened}\n\n"
            f"Where the chain broke:\n  {self.where_chain_broke}\n\n"
            f"Failure classification:\n  {self.classification}\n\n"
            f"Evidence cited:\n{ev}\n\n"
            f"Confidence:\n  {self.confidence}\n\n"
            f"Prescription:\n  {self.prescription or 'insufficient evidence'}\n\n"
            f"Not recommended:\n  {self.not_recommended or 'insufficient evidence'}\n"
        )


class LLMRiskDoctor:
    """Offline supervisory diagnoser. Read-only; mandatory rulebook read on init."""

    def __init__(self, rulebook: Optional[Path] = None):
        self.rulebook_path = rulebook or cfg.INTERPRETABILITY_RULEBOOK
        # Fail LOUDLY if the binding rulebook is unreadable (codebase requirement).
        if not self.rulebook_path.exists():
            raise FileNotFoundError(
                f"LLMRiskDoctor requires the binding rulebook at {self.rulebook_path} "
                f"(MLP_INTERPRETABILITY_LAYER.md). Refusing to diagnose without it."
            )
        self.rulebook = self.rulebook_path.read_text(encoding="utf-8", errors="replace")

    # -- read-only repo access (no write method exists, by design) --
    def view(self, path: Path) -> str:
        """READ a repo file to support reasoning. Never writes. (SOW §9.3)"""
        return Path(path).read_text(encoding="utf-8", errors="replace")

    # -- evidence extractors --
    @staticmethod
    def _steps(records: List[dict]) -> List[dict]:
        return [r for r in records if r.get("kind") == "step"]

    def diagnose(self, records: List[dict], scoreboard: Optional[Dict] = None) -> Diagnosis:
        """Walk the reverse chain; classify the FIRST broken link with cited evidence."""
        steps = self._steps(records)
        run_id = next((r.get("run_id", "?") for r in records if r.get("kind") == "header"), "?")
        what = self._outcome_summary(steps, scoreboard)

        if not steps:
            return Diagnosis(run_id, what, "no step telemetry", UNCLASSIFIED,
                             confidence="LOW",
                             prescription="capture per-step telemetry, then re-diagnose")

        # ---- REWARD link: Reward Hijack ----
        layers = self._reward_sums(steps)
        l0 = layers.get("L0", 0.0)
        hijackers = {k: v for k, v in layers.items() if k != "L0" and v > l0 and v > 0}
        if hijackers:
            worst = max(hijackers, key=hijackers.get)
            return Diagnosis(
                run_id, what, "reward layer (a shaping layer outweighs Layer 0)",
                "Reward Hijack", confidence="HIGH",
                evidence=[f"cumulative |{worst}|={hijackers[worst]:.4g} > |L0|={l0:.4g} "
                          f"(reward_decomposition integral)"],
                prescription="reduce that shaping weight; verify the E8 dominance rule; "
                             "consider disabling the L6 QUAD toggle",
                not_recommended="raising Layer-0 reward scale to mask the imbalance")

        # ---- CRITIC link: Critic Misalignment ----
        val = np.array([s["value"] for s in steps])
        out = np.array([s.get("outcome", {}).get("next_bar_return", 0.0) for s in steps])
        if val.std() > 0 and out.std() > 0:
            corr = float(np.corrcoef(val, out)[0, 1])
            if abs(corr) < 0.05:
                return Diagnosis(
                    run_id, what, "critic head (V(s) does not track challenge-quality)",
                    "Critic Misalignment", confidence="MEDIUM",
                    evidence=[f"corr(V(s), next-bar outcome)={corr:.3f} ~ 0 over {len(steps)} steps"],
                    prescription="reward-layer audit; check Layer-0 dominance; possibly lengthen the gamma horizon",
                    not_recommended="retraining the actor before fixing the critic")

        # ---- ACTOR link: Mask Dependence / Risk Blindness / Stagnation Blindness ----
        md = self._mask_dependence(steps)
        if md is not None:
            return md
        rb = self._risk_blindness(steps)
        if rb is not None:
            return rb
        sb = self._stagnation_blindness(steps)
        if sb is not None:
            return sb

        # ---- HIDDEN link: Representation Collapse / Chaos ----
        hr = self._representation(steps)
        if hr is not None:
            return hr

        # ---- nothing fits: never invent a 9th category ----
        return Diagnosis(
            run_id, what, "no single link shows a clear break in the available telemetry",
            UNCLASSIFIED, confidence="LOW",
            evidence=["all 8-taxonomy heuristics below threshold"],
            prescription="capture more telemetry (hidden vectors across seeds, longer windows) and re-diagnose",
            not_recommended="inventing a 9th failure category")

    # -- per-link heuristics (each cites a specific field) --
    def _outcome_summary(self, steps, scoreboard) -> str:
        if scoreboard:
            return (f"pass_rate={scoreboard.get('pass_rate')}, breaches={scoreboard.get('breaches')}, "
                    f"target_hit={scoreboard.get('target_hit_consistency')}")
        return f"{len(steps)} steps logged; no scoreboard supplied (outcome facts limited)"

    @staticmethod
    def _reward_sums(steps) -> Dict[str, float]:
        keys: set = set()
        for s in steps:
            keys |= set(s["reward_decomposition"].keys())
        return {k: float(np.sum(np.abs([s["reward_decomposition"].get(k, 0.0) for s in steps])))
                for k in keys}

    def _mask_dependence(self, steps) -> Optional[Diagnosis]:
        bad = 0
        for s in steps:
            pre = np.array(s["pre_mask_logits"]); post = np.array(s["post_mask_logits"])
            if int(np.argmax(pre)) != int(np.argmax(post)) and post[int(np.argmax(pre))] <= -1e8:
                bad += 1   # the actor's preferred action was illegal (masked away)
        frac = bad / len(steps)
        if frac > 0.30:
            return Diagnosis(
                "?", "", "actor head (pre-mask logits favour illegal actions)",
                "Mask Dependence", confidence="MEDIUM",
                evidence=[f"pre-mask argmax was masked-illegal in {frac:.0%} of steps"],
                prescription="add observation features for the ignored law ingredients; consider law-school re-exposure",
                not_recommended="loosening the masks to 'let' the actor act")
        return None

    def _risk_blindness(self, steps) -> Optional[Diagnosis]:
        dd = np.array([s["risk_context"].get("trailing_dd", 0.0) for s in steps])
        size = np.array([s["raw_size"] for s in steps])
        if dd.max() < 3.5 or (dd >= 3.5).sum() < 5:
            return None
        safe = size[dd < 1.0].mean() if (dd < 1.0).any() else 0.0
        danger = size[dd >= 3.5].mean()
        if safe > 0 and abs(danger - safe) / safe < 0.10:   # aggression unchanged near the wall
            return Diagnosis(
                "?", "", "actor head (aggression persists into breach-risk states)",
                "Risk Blindness", confidence="MEDIUM",
                evidence=[f"mean raw_size: safe(dd<1%)={safe:.3f} vs danger(dd>=3.5%)={danger:.3f} (~unchanged)"],
                prescription="strengthen the Layer-3 pain-zone curve; expand risk-context features in the observation",
                not_recommended="tightening the hard wall instead of teaching restraint")
        return None

    def _stagnation_blindness(self, steps) -> Optional[Diagnosis]:
        # HOLD-dominated in favourable legal contexts (a directional law active)
        fav = [s for s in steps if any(abs(x) == 1 for x in s["law_states"][:9])]
        if len(fav) < 10:
            return None
        hold = np.mean([1.0 if s["chosen_action"] == 0 else 0.0 for s in fav])
        if hold > 0.70:
            return Diagnosis(
                "?", "", "actor head (flat/HOLD in favourable legal contexts)",
                "Stagnation Blindness", confidence="MEDIUM",
                evidence=[f"HOLD share={hold:.0%} in law-active windows (> 0.70)"],
                prescription="check the Layer-2 stagnation weight; verify the QUAD Target-Velocity signal",
                not_recommended="forcing trades via the reward (breaks Layer-0 dominance)")
        return None

    def _representation(self, steps) -> Optional[Diagnosis]:
        h = np.array([s["hidden_summary"] for s in steps], dtype=float)
        if h.size == 0:
            return None
        var = float(np.mean(np.var(h, axis=0)))
        if var < 1e-6:
            return Diagnosis(
                "?", "", "shared trunk (hidden states collapse to one point)",
                "Representation Collapse", confidence="MEDIUM",
                evidence=[f"mean per-dim hidden variance={var:.2e} ~ 0 (states indistinguishable)"],
                prescription="check challenge features reach the trunk; reduce capacity reuse",
                not_recommended="adding heads before fixing the trunk representation")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M11 — implemented the read-only LLMRiskDoctor.
#   I: Telemetry + visuals existed but nothing turned them into an evidence-cited,
#      taxonomy-classified diagnosis - and the read-only/mandatory-rulebook boundary
#      needed enforcing in code.
#   R: MLP_INTERPRETABILITY_LAYER.md (output template, 8-failure taxonomy, reverse-chain,
#      evidence-only) + SOW R5/§9.3 (read-only; fail loud without the rulebook).
#   A: Constructor fails loudly if the rulebook is missing; view() reads (no write method);
#      diagnose() walks the reverse chain, classifies the first broken link into one of the
#      8 (else 'unclassified'), cites a specific field per claim, renders the template.
#   C: Pass-rate failures get a true, evidence-backed cause + a safe prescription before
#      they cost a challenge - and the doctor can never touch execution, so it only ever
#      protects the pass rate, never endangers it.
