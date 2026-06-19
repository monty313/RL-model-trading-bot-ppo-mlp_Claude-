"""repo_graph — build the live import-dependency graph of the Quantra repo (AST, no execution).

WHAT THIS MODULE DOES
---------------------
Walks every first-party ``*.py`` file (quantra/, barbershop/, scripts/, tests/), parses it with
``ast`` (never imports/executes it), and builds a directed graph: nodes = modules, edges = "imports".
Each node carries its directory group, a neon color, its one-line docstring, and its first
``COUPLING`` note — everything the visual layers need.

WHO CONSUMES IT (both directions):
  -> jarvis_hud.html / 0_QUANTRA_LIVE_COCKPIT.html: load to_d3_json() (write_d3_snapshot) for the
     animated d3 dependency graph.
  -> barbershop/dashboard.py Screen 6 (Repo Map): to_cytoscape_elements() feeds a dash-cytoscape graph;
     click a node -> show its docstring + coupling note.
  -> reads the repo source tree only; depends on NO third-party package for the core build
     (networkx is an OPTIONAL export via to_networkx()).

Pure + offline + dependency-free at its core, so it is safe to run anywhere (CI, Colab, the dashboard).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]          # barbershop/ sits at the repo root
_SCAN_DIRS = ("quantra", "barbershop", "scripts", "tests")
_SKIP = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules", "build", "dist"}

# Color by directory (neon palette; the operator-specified ones are exact). COUPLING -> jarvis_hud.html
# + dashboard Screen 6 legend: these hex values are mirrored in the HUD CSS / cytoscape stylesheet.
_GROUP_COLORS: Dict[str, str] = {
    "quantra.locked_core":     "#e24b4a",   # 🔴 locked core (laws/risk/cost/platform)
    "quantra.learning_system": "#e0a52e",   # 🟠 PPO / reward / trainer / registry
    "quantra.market_pipeline": "#7f77dd",   # 🟣 data → features → law mask
    "quantra.ftmo_passing":    "#1d9e75",   # 🟢 scoreboard / challenge state / validation
    "quantra.diagnostics":     "#5dcaa5",   # 🩵 telemetry / interpreter / risk doctor
    "quantra.env":             "#38bdf8",   # 🔵 the gym environment
    "quantra.live_bridge":     "#3b82f6",   # 🔵 live MT5 bridge
    "quantra.runtime":         "#94a3b8",   # ⚪ runtime config / hardware
    "quantra.constitution":    "#64748b",   # ⚪ mission / safety
    "barbershop":              "#2dd4bf",   # 🩵 the diagnostics dashboard
    "scripts":                 "#22c55e",   # 🟢 scripts
    "tests":                   "#9aa4b2",   # ⚪ tests
}
_DEFAULT_COLOR = "#8b97a8"


@dataclass
class RepoNode:
    module: str                       # dotted module, e.g. "quantra.env.trading_env"
    path: str                         # repo-relative path
    group: str                        # color/group key (e.g. "quantra.env")
    color: str
    doc: str = ""                     # one-line module docstring
    coupling: str = ""                # first COUPLING note found in the file
    loc: int = 0                      # lines of code (node size hint)


@dataclass
class RepoGraph:
    nodes: Dict[str, RepoNode] = field(default_factory=dict)
    edges: List[Tuple[str, str]] = field(default_factory=list)   # (importer, imported), first-party only

    # ---- exporters (the visual layers consume these) ---------------------------
    def to_d3_json(self) -> dict:
        """{"nodes":[{id,group,color,doc,coupling,loc}], "links":[{source,target}]} for d3.js."""
        return {
            "nodes": [{"id": n.module, "group": n.group, "color": n.color, "doc": n.doc,
                       "coupling": n.coupling, "loc": n.loc} for n in self.nodes.values()],
            "links": [{"source": s, "target": t} for s, t in self.edges],
            "groups": {g: c for g, c in _GROUP_COLORS.items()},
        }

    def to_cytoscape_elements(self) -> List[dict]:
        """dash-cytoscape elements (nodes then edges). COUPLING -> dashboard.py Screen 6 stylesheet:
        node data carries color/group/doc/coupling so a click can show the side-panel."""
        els: List[dict] = [{"data": {"id": n.module, "label": n.module.split(".")[-1],
                                     "group": n.group, "color": n.color, "doc": n.doc,
                                     "coupling": n.coupling}} for n in self.nodes.values()]
        els += [{"data": {"source": s, "target": t, "id": f"{s}->{t}"}} for s, t in self.edges]
        return els

    def to_networkx(self):
        """Optional networkx.DiGraph (lazy import; only if the caller wants graph algorithms)."""
        import networkx as nx          # optional dep — not required for build/export
        g = nx.DiGraph()
        for n in self.nodes.values():
            g.add_node(n.module, group=n.group, color=n.color, doc=n.doc)
        g.add_edges_from(self.edges)
        return g

    def write_d3_snapshot(self, path) -> Path:
        """Write to_d3_json() to disk so the self-contained HUD can fetch a REAL, current graph."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_d3_json(), indent=2))
        return p


def _module_name(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _group_of(module: str) -> str:
    best = ""
    for key in _GROUP_COLORS:                # longest-prefix wins (quantra.env before quantra)
        if (module == key or module.startswith(key + ".")) and len(key) > len(best):
            best = key
    return best or module.split(".")[0]


def _first_coupling(src: str) -> str:
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#") and "COUPLING" in s:
            return s.lstrip("# ").strip()[:200]
    return ""


def _imports(tree: ast.AST) -> List[str]:
    out: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:        # absolute import only (first-party resolution)
                out.append(node.module)
    return out


def build_graph(root: Optional[Path] = None) -> RepoGraph:
    """Parse every first-party .py file and return the import RepoGraph. AST-only — nothing is run."""
    root = Path(root or REPO_ROOT)
    files: List[Path] = []
    for d in _SCAN_DIRS:
        base = root / d
        if base.is_dir():
            files += [p for p in base.rglob("*.py") if not (_SKIP & set(p.parts))]

    g = RepoGraph()
    raw_imports: Dict[str, List[str]] = {}
    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(p))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod = _module_name(p, root)
        g.nodes[mod] = RepoNode(
            module=mod, path=str(p.relative_to(root)), group=_group_of(mod),
            color=_GROUP_COLORS.get(_group_of(mod), _DEFAULT_COLOR),
            doc=(ast.get_docstring(tree) or "").strip().split("\n")[0][:160],
            coupling=_first_coupling(src), loc=src.count("\n") + 1)
        raw_imports[mod] = _imports(tree)

    known = set(g.nodes)
    for mod, imps in raw_imports.items():
        for imp in imps:
            target = _resolve(imp, known)
            if target and target != mod:
                g.edges.append((mod, target))
    # de-dup edges, keep order
    seen = set()
    g.edges = [e for e in g.edges if not (e in seen or seen.add(e))]
    return g


def _resolve(imported: str, known: set) -> Optional[str]:
    """Map an imported dotted name to a known module: exact module, its package __init__, or the
    longest known prefix (so `from quantra.env.trading_env import X` and `import quantra.env` both land)."""
    if imported in known:
        return imported
    parts = imported.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in known:
            return cand
    return None


def main() -> None:
    """CLI: write a fresh d3 snapshot the HUD loads. Usage: python -m barbershop.repo_graph [out.json]"""
    import sys
    g = build_graph()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "artifacts" / "repo_graph.json"
    g.write_d3_snapshot(out)
    print(f"repo_graph: {len(g.nodes)} modules, {len(g.edges)} edges -> {out}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# STANDING RULE [2026-06-19, operator] — applies to THIS file and EVERY file going forward: keep
# SHOWING THE WORK. On every edit (1) append a DATED IRAC entry here, and (2) document the cross-file
# RELATIONSHIPS (COUPLING) in both directions so a future reader sees what breaks where, and when.
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-19] C22 — repo import-graph builder (powers the JARVIS HUD + Dash Repo Map).
#   I: The "live dependency graph" features (JARVIS HUD + Barbershop Screen 6) needed a single,
#      accurate source of the repo's module graph — not a hand-maintained edge list that rots.
#   R: Operator brief (Feature 4 "Repo Map" + the JARVIS dependency graph) + the show-the-work rule.
#   A: AST-only build_graph() (no execution) -> RepoGraph with neon group colors, docstrings, and the
#      first COUPLING note per file; exporters to_d3_json()/to_cytoscape_elements()/to_networkx() +
#      write_d3_snapshot() + a `python -m barbershop.repo_graph` CLI. networkx is an OPTIONAL export.
#   C: One legible, always-current picture of how the system fits together — so the operator (and the
#      Risk Doctor) can see the real wiring, and any future coupling change is visible at a glance.
