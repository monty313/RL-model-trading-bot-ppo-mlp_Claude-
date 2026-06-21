"""C22 — repo_graph (import DAG) + ws_broadcaster (telemetry->HUD) + the JARVIS HUD contract.

These power the live dependency graph + the real-time HUD. Pure/offline parts only (no networkx /
websockets / browser needed); the HUD test guards the event-schema contract shared with ws_broadcaster.
"""

from pathlib import Path

from barbershop.repo_graph import build_graph, RepoGraph
from barbershop import ws_broadcaster as wb

_REPO = Path(__file__).resolve().parents[1]


# ─────────────────────────── repo_graph (Feature 4 / HUD graph) ───────────────────────────
def test_repo_graph_builds_real_edges_and_groups():
    g = build_graph()
    assert isinstance(g, RepoGraph) and len(g.nodes) > 40 and len(g.edges) > 40
    # a known coupling must show up as an edge (env imports the reward engine + config)
    outs = {t for s, t in g.edges if s == "quantra.env.trading_env"}
    assert "quantra.learning_system.reward_engine.reward" in outs
    assert "quantra.runtime.config" in outs
    # every node carries a group color + the locked core is red (operator palette)
    assert all(n.color.startswith("#") for n in g.nodes.values())
    lc = next(n for n in g.nodes.values() if n.module.startswith("quantra.locked_core"))
    assert lc.color == "#e24b4a"


def test_repo_graph_exports_d3_and_cytoscape(tmp_path):
    g = build_graph()
    d3 = g.to_d3_json()
    assert set(d3) == {"nodes", "links", "groups"} and d3["nodes"] and d3["links"]
    assert {"id", "group", "color", "doc"} <= set(d3["nodes"][0])
    els = g.to_cytoscape_elements()
    assert any("source" in e["data"] for e in els) and any("label" in e["data"] for e in els)
    out = g.write_d3_snapshot(tmp_path / "g.json")
    assert out.exists() and out.read_text().startswith("{")


# ─────────────────────────── ws_broadcaster (telemetry -> HUD event) ──────────────────────
def test_packet_to_event_maps_step_packet():
    p = {"timestep": 42, "symbol": "EURUSD", "chosen_action": 1,
         "reward_decomposition": {"L0": 0.01, "total": 0.02},
         "risk_context": {"trailing_dd": 2.1}, "active_module": "trainer"}
    e = wb.packet_to_event(p)
    assert e["step"] == 42 and e["module"] == "trainer" and e["reward"] == 0.02
    assert e["drawdown_pct"] == 2.1 and e["action"] == "BUY EURUSD" and e["ftmo"] == "BUILDING"


def test_ftmo_status_thresholds():
    assert wb._ftmo(4.0, 0.0) == "BREACH"
    assert wb._ftmo(1.0, 2.5) == "PASSING"
    assert wb._ftmo(3.2, 1.0) == "AT RISK"
    assert wb._ftmo(1.0, 1.0) == "BUILDING"


def test_iter_events_skips_header_and_garbage():
    import json
    lines = [json.dumps({"kind": "header", "run_id": "r"}),
             json.dumps({"kind": "step", "timestep": 1, "symbol": "US30", "chosen_action": 3,
                         "reward_decomposition": {"total": -0.1}, "risk_context": {"trailing_dd": 3.6}}),
             "not json"]
    evs = list(wb.iter_events_from_lines(lines))
    assert len(evs) == 1 and evs[0]["action"] == "CLOSE US30" and evs[0]["ftmo"] == "AT RISK"


# ─────────────────────────── JARVIS HUD contract (schema must match) ──────────────────────
def test_jarvis_hud_matches_broadcaster_event_schema():
    html = (_REPO / "jarvis_hud.html").read_text()
    assert "ws://localhost:8765" in html                      # same endpoint ws_broadcaster serves
    for key in ("daily_pnl_pct", "drawdown_pct", "win_rate", "ftmo", "module", "reward"):
        assert key in html, f"HUD is missing event key {key}"
    # a synthetic broadcaster event carries the schema the HUD consumes
    ev = next(wb.demo_events(n=1, delay=0))
    assert {"step", "module", "reward", "drawdown_pct", "ftmo", "action"} <= set(ev)
    # every pipeline node the broadcaster can pulse must exist in the HUD graph
    hud_mods = [m for m in ("data_loader", "trading_env", "ppo_agent", "reward_engine", "trainer",
                            "llm_risk_doctor", "live_bridge") if m in html]
    assert len(hud_mods) == 7
