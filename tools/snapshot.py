"""State-vector snapshot guard — Tier 1 of the change-impact tracker.

WHAT THIS DOES
--------------
Pins the policy's observation layout to a committed JSON
(``tests/snapshots/state_vector.json``) and provides the diff + checklist used by
the master-suite test and the pre-commit hook. If anyone changes the state vector
(adds/removes/reorders a feature, flips INCLUDE_RAW_INPUTS, resizes a block), the
snapshot stops matching and the suite fails with a CONCRETE checklist of what else
to update — and an FTMO-framed explanation the LLM Risk Doctor can read.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The observation is the policy's entire world. A silent change to it invalidates
normalization stats, the agent's input dim, telemetry labels, and every existing
checkpoint — any of which can quietly wreck pass-rate between runs. This guard makes
such a change impossible to do *accidentally*: you must consciously re-snapshot and
acknowledge the ripple, which is exactly the discipline that keeps passing consistent.

LLM RISK DOCTOR — HOW TO READ THE SNAPSHOT
------------------------------------------
The committed JSON carries an ``_llm_interpretation`` section describing each block
and what a drift implies for passing. When you triangulate a pass-rate regression,
compare the run's observation layout to this snapshot: if they differ, the world
changed and THAT is the first suspect (re-normalize / re-validate), not the trunk.
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``.

Usage:
    python tools/snapshot.py --check     # exit 1 on drift (CI / pre-commit)
    python tools/snapshot.py --update    # rewrite the snapshot after an intended change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py: state_vector_fingerprint()
# defines STATE_DIM/FEATURE_NAMES/block_widths/include_raw_inputs; diff() below reads those exact
# keys. If schema's fingerprint dict shape changes, diff()/checklist() here must change too.
from quantra.market_pipeline.feature_builder.schema import state_vector_fingerprint  # noqa: E402

# COUPLING [C1] -> tests/snapshots/state_vector.json: this committed snapshot is the pinned
# observation layout; `--update` rewrites it. Any schema.py layout change must be re-snapshotted here.
SNAPSHOT_PATH = REPO_ROOT / "tests" / "snapshots" / "state_vector.json"

# Static, FTMO-framed guidance baked into the snapshot so the LLM Risk Doctor (and
# any human) understands what each block is and what a change implies.
LLM_INTERPRETATION = {
    "purpose": (
        "This snapshot pins the observation the PPO policy sees. Judged against "
        "repeated FTMO passing, not PnL. A drift means the policy's WORLD changed."
    ),
    "blocks": {
        "market": "Normalized/ATR-scaled multi-TF features (law ingredients + time), bounded to +/-10.",
        "market_raw": ("UNNORMALIZED raw SMA + raw CCI levels (operator override). MUST be "
                       "standardized by the agent. If you see Representation Chaos, instability, "
                       "or Shortcut Learning, suspect THIS block first (large magnitude / single-"
                       "feature attribution) before blaming the trunk."),
        "law": "12 law/gate flags. A wrong flag -> check its ingredient in `market` (Term 2).",
        "trade": "Per-slot x5 trade state; the pointer head reads this on CLOSE.",
        "portfolio": "Cross-slot aggregates (net exposure / size / uPnL).",
        "account": ("Equity/buffers + 2 challenge-progress features = the breach-risk picture. "
                    "If this block is zeroed or constant for an episode, the bot is BLIND to "
                    "danger (Representation Collapse) regardless of the trunk."),
    },
    "drift_means": [
        "Verify the PPO agent reads STATE_DIM dynamically (no hardcoded input width).",
        "Regenerate input normalization stats for the NEW/changed features only.",
        "Add/extend telemetry block labels (M9 data contract) for the new features.",
        "Re-run baseline-vs-candidate walk-forward; promote only via the PromotionGate.",
        "Bump REPO_MAP + the IRAC logs of every touched file.",
    ],
}


def build() -> dict:
    """The full snapshot = structural fingerprint + the LLM interpretation block."""
    fp = state_vector_fingerprint()
    fp["_llm_interpretation"] = LLM_INTERPRETATION
    return fp


def load(path: Path = SNAPSHOT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")


def diff(old: dict, new: dict) -> list:
    """Structural deltas between two snapshots (ignores the _llm_interpretation text)."""
    deltas = []
    # COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py: these dict keys
    # (state_dim, include_raw_inputs, block_widths, feature_names) are produced by
    # state_vector_fingerprint(); renaming any key there silently breaks this drift check.
    if old.get("state_dim") != new.get("state_dim"):
        deltas.append(f"state_dim: {old.get('state_dim')} -> {new.get('state_dim')}")
    if old.get("include_raw_inputs") != new.get("include_raw_inputs"):
        deltas.append(f"include_raw_inputs: {old.get('include_raw_inputs')} -> {new.get('include_raw_inputs')}")
    ow, nw = old.get("block_widths", {}), new.get("block_widths", {})
    for k in sorted(set(ow) | set(nw)):
        if ow.get(k) != nw.get(k):
            deltas.append(f"block '{k}' width: {ow.get(k)} -> {nw.get(k)}")
    of, nf = old.get("feature_names", []), new.get("feature_names", [])
    added = [f for f in nf if f not in of]
    removed = [f for f in of if f not in nf]
    if added:
        deltas.append(f"added features ({len(added)}): {added[:12]}{' ...' if len(added) > 12 else ''}")
    if removed:
        deltas.append(f"removed features ({len(removed)}): {removed[:12]}{' ...' if len(removed) > 12 else ''}")
    if not added and not removed and of != nf:
        deltas.append("feature ORDER changed (same set, different order)")
    return deltas


def checklist(deltas: list) -> list:
    """Concrete follow-ups for an intended observation change (FTMO-framed)."""
    if not deltas:
        return []
    return [
        "1. If intended, re-snapshot: `python tools/snapshot.py --update`.",
        "2. Verify PPOAgent (M5) reads STATE_DIM dynamically — no hardcoded input dim.",
        "3. Regenerate input normalization stats for the new/changed features only.",
        "4. Extend TelemetryLogger (M9) block labels so the LLM can name the new features.",
        "5. Re-run baseline-vs-candidate walk-forward; gate on pass-rate via PromotionGate.",
        "6. Append an IRAC entry to schema.py + bump REPO_MAP; run `python tools/impact.py`.",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="State-vector snapshot guard.")
    ap.add_argument("--update", action="store_true", help="rewrite the committed snapshot")
    ap.add_argument("--check", action="store_true", help="exit 1 on drift")
    args = ap.parse_args()

    if args.update:
        write()
        print(f"snapshot written: {SNAPSHOT_PATH.relative_to(REPO_ROOT)}")
        return 0

    if not SNAPSHOT_PATH.exists():
        print("no snapshot yet; run `python tools/snapshot.py --update`", file=sys.stderr)
        return 1
    deltas = diff(load(), build())
    if deltas:
        print("STATE-VECTOR DRIFT DETECTED:", file=sys.stderr)
        for d in deltas:
            print("  - " + d, file=sys.stderr)
        print("\nChecklist (relative to passing FTMO):", file=sys.stderr)
        for c in checklist(deltas):
            print("  " + c, file=sys.stderr)
        return 1 if args.check else 0
    print("state-vector snapshot OK (no drift).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Created the state-vector snapshot guard (Tier 1).
#   I: A change to the observation could ship without re-fitting normalization /
#      agent dim / telemetry, silently degrading pass-rate between runs.
#   R: Operator request for change-impact tracking with LLM-readable, FTMO-framed I/O.
#   A: Committed snapshot (tests/snapshots/state_vector.json) + diff + checklist +
#      an _llm_interpretation block; enforced by the master suite + pre-commit.
#   C: The policy's world can no longer change by accident; intended changes carry a
#      concrete follow-up list — keeping observation, code, and checkpoints in sync,
#      which is what protects a months-long pass streak.
