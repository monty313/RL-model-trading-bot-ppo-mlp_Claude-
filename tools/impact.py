"""Change-impact analyzer — Tier 2 of the change-impact tracker.

WHAT THIS DOES
--------------
Builds an AST import graph of the ``quantra`` package, then for a set of changed
files prints (a) the REVERSE-DEPENDENCY closure — every module that transitively
imports a changed file and may therefore need updates/tests — and (b) a static
"pipeline file -> required follow-ups" checklist. Output is plain text the operator
and the LLM Risk Doctor can read.

    python tools/impact.py path/to/changed.py [more ...]
    python tools/impact.py --staged          # uses `git diff --cached --name-only`

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
"I changed a feature and forgot to update X" is exactly the class of silent bug that
erodes a months-long pass streak. Tracing the blast radius before committing means
the env, agent, telemetry, and tests that depend on a changed file get updated
together — so the bot's world and the code that reasons about it never drift apart.

LLM RISK DOCTOR — HOW TO READ THIS
----------------------------------
Each report names the changed file, what depends on it, and the FTMO-relevant
follow-ups. If a regression appeared right after a pipeline change, run this on that
change to see what *should* have been updated. Rulebook:
``docs/MLP_INTERPRETABILITY_LAYER.md``.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
# COUPLING -> quantra/ package root: PKG names the import namespace; _module_name/_imports_of/
# build_graph only trace modules under this prefix. Renaming the top-level package breaks the graph.
PKG = "quantra"

# COUPLING [C1] -> quantra/market_pipeline/feature_builder/{schema,builder,indicators}.py +
# runtime/config.py + resampler/resampler.py + data_loader/loader.py: these dict keys are
# repo-relative path SUBSTRINGS matched against changed files; if any of those files moves/renames,
# the matching key here must move too or the follow-up checklist silently stops firing.
# Static map: when a file matching the substring changes, these FTMO-framed
# follow-ups apply regardless of the import graph.
PIPELINE_FOLLOWUPS: Dict[str, List[str]] = {
    "feature_builder/schema.py": [
        "Observation layout changed -> run `python tools/snapshot.py --check`.",
        "Verify PPOAgent input dim reads STATE_DIM dynamically.",
        "Regenerate normalization stats for new features; extend telemetry labels.",
        "Re-run baseline-vs-candidate walk-forward (gate on pass-rate).",
    ],
    "feature_builder/builder.py": [
        "Feature math changed -> re-snapshot if dims moved; re-validate no-lookahead test.",
        "Regenerate the memmap feature cache (delete data/features/*).",
    ],
    "feature_builder/indicators.py": [
        "Indicator params changed -> CONFIRM law legal-space is unaffected (SOW-D4 locks).",
        "Re-run law unit tests (M3) once they exist.",
    ],
    "runtime/config.py": [
        "Runtime constants changed -> confirm config.nominal_state_dim == schema.STATE_DIM.",
        "If INCLUDE_RAW_INPUTS flipped, re-snapshot and re-run the raw-block tests.",
    ],
    "resampler/resampler.py": ["Re-run the no-lookahead test (Section C)."],
    "data_loader/loader.py": ["Re-run loader tests (Section B); clear the parquet cache."],
}


def _module_name(path: Path) -> str | None:
    """Map a repo-relative .py path to its dotted module name (quantra.* only)."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return None
    if rel.parts and rel.parts[0] != PKG:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports_of(path: Path, mod: str, is_pkg: bool) -> Set[str]:
    """Internal (quantra.*) modules imported by a file, via AST (no execution).

    Resolves RELATIVE imports (``from .schema import X``, ``from . import y``) using
    the file's own package, so the dependency graph isn't under-reported — without
    this, schema.py would look like a leaf even though builder.py imports it.
    """
    out: Set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return out
    pkg = mod if is_pkg else (mod.rsplit(".", 1)[0] if "." in mod else mod)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.startswith(PKG):
                    out.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:                 # relative import
                base_parts = pkg.split(".")
                ascend = node.level - 1
                if ascend:
                    base_parts = base_parts[:-ascend] if ascend < len(base_parts) else []
                base = ".".join(base_parts)
                if node.module:
                    target = f"{base}.{node.module}" if base else node.module
                    if target.startswith(PKG):
                        out.add(target)
                else:                                          # from . import a, b
                    for n in node.names:
                        target = f"{base}.{n.name}" if base else n.name
                        if target.startswith(PKG):
                            out.add(target)
            elif node.module and node.module.startswith(PKG):  # absolute import
                out.add(node.module)
    return out


def build_graph() -> Dict[str, Set[str]]:
    """module -> set of internal modules it imports."""
    graph: Dict[str, Set[str]] = {}
    for py in (REPO_ROOT / PKG).rglob("*.py"):
        mod = _module_name(py)
        if mod is None:
            continue
        graph[mod] = _imports_of(py, mod, is_pkg=(py.name == "__init__.py"))
    return graph


def reverse_closure(changed_modules: Set[str], graph: Dict[str, Set[str]]) -> Set[str]:
    """All modules that transitively import any changed module."""
    # importers[m] = modules that import m (prefix-aware: importing a package hits children)
    affected: Set[str] = set()
    frontier = set(changed_modules)
    while frontier:
        m = frontier.pop()
        for mod, imps in graph.items():
            if mod in affected or mod in changed_modules:
                continue
            if any(m == i or m.startswith(i + ".") or i.startswith(m + ".") for i in imps):
                affected.add(mod)
                frontier.add(mod)
    return affected


def _staged_files() -> List[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], cwd=REPO_ROOT, text=True
        )
        return [l for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def report(changed: List[str]) -> str:
    graph = build_graph()
    lines = ["=" * 72, "QUANTRA CHANGE-IMPACT REPORT (framed against passing FTMO)", "=" * 72]
    changed_mods = {m for m in (_module_name(REPO_ROOT / c) for c in changed) if m}
    affected = reverse_closure(changed_mods, graph)

    lines.append(f"Changed files ({len(changed)}):")
    for c in changed:
        lines.append(f"  - {c}")
    lines.append("")
    lines.append("Reverse-dependency closure (modules that import the changed code,")
    lines.append("and so may need updates/tests):")
    if affected:
        for m in sorted(affected):
            lines.append(f"  * {m}")
    else:
        lines.append("  (none — leaf change)")
    lines.append("")

    followups: List[str] = []
    for c in changed:
        for key, items in PIPELINE_FOLLOWUPS.items():
            if key in c.replace("\\", "/"):
                followups.extend(items)
    # COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py + tools/snapshot.py:
    # any change under feature_builder/ (or any 'schema' file) implies an observation-layout
    # ripple; this hard-codes that the snapshot guard must be re-run. Keep in sync if the
    # observation-defining module is ever renamed away from 'feature_builder'/'schema'.
    if any("feature_builder" in c or "schema" in c for c in changed):
        followups.append("Always: the master suite Section D + snapshot test must stay green.")
    lines.append("Required follow-ups:")
    if followups:
        for f in dict.fromkeys(followups):  # de-dupe, keep order
            lines.append(f"  [ ] {f}")
    else:
        lines.append("  [ ] Re-run `pytest`; append an IRAC entry to each changed file.")
    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Change-impact analyzer.")
    ap.add_argument("files", nargs="*", help="changed file paths")
    ap.add_argument("--staged", action="store_true", help="use git staged files")
    args = ap.parse_args()
    changed = _staged_files() if args.staged else args.files
    changed = [c for c in changed if c.endswith(".py")]
    if not changed:
        print("no changed .py files to analyze.")
        return 0
    print(report(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Created the AST change-impact analyzer (Tier 2).
#   I: A pipeline change could ripple to env/agent/telemetry/tests undetected.
#   R: Operator request for an AST import-graph + dependency-tracing checklist.
#   A: Built the import graph (resolving RELATIVE imports), reverse-dependency
#      closure, and a static pipeline->follow-ups map; output is FTMO-framed text.
#   C: The blast radius of any change is visible before commit, so dependent code +
#      tests get updated together — the bot's world and its reasoning never drift.
# [2026-06-13] Fixed relative-import resolution.
#   I: `from .schema import X` was skipped, so schema.py looked like a leaf and the
#      graph under-reported dependents — a silent gap in the safety net.
#   R: The graph must reflect TRUE dependencies to be trustworthy.
#   A: Resolved relative imports via the file's package + node.level.
#   C: The dependency map is now correct, so the change-impact checklist can be
#      trusted to catch the follow-ups that keep pass-rate from silently regressing.
