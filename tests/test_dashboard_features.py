"""C25 — the three Barbershop dashboard upgrades: Guide screen, Repo Map, and the SAW->chart overlay.

Pure builder-level checks (no running Dash server / no dash-cytoscape needed). The mock bundle is the
offline data source the dashboard ships with, so these run anywhere.
"""

import plotly.graph_objects as go

from barbershop import dashboard, data, figures


def _bundle():
    return dashboard.load_bundle(source="mock")


# ── Feature 2 — How-to-Use guide screen ──
def test_guide_screen_renders_the_markdown_guide():
    s = str(dashboard.screen_guide(_bundle()))
    assert "Barbershop" in s and "Risk Doctor" in s        # content from BARBERSHOP_GUIDE.md


# ── Feature 4 — Repo Map screen (graceful without dash-cytoscape) ──
def test_repo_map_screen_is_graceful_and_shows_counts():
    s = str(dashboard.screen_repo_map(_bundle()))
    assert "Repo Map" in s and "modules" in s
    assert ("dash-cytoscape" in s or "repo-cyto" in s)     # install note OR the live graph id


# ── Feature 1 — clickable SAW feature overlays on the Day-Replay chart ──
def test_day_replay_overlays_a_clicked_feature():
    b = _bundle()
    traj = b["trajectory"]
    day = int(traj["day_id"].iloc[0])
    names = b.get("feature_names") or data.MOCK_FEATURE_NAMES
    s = str(dashboard.screen_day_replay(b, day, "1m", None, overlay_feature=names[0]))
    assert "Overlaying SAW feature" in s
    # an unknown name degrades to a note, never crashes
    s2 = str(dashboard.screen_day_replay(b, day, "1m", None, overlay_feature="definitely_not_a_feature"))
    assert "no per-bar series" in s2
    # default (no overlay) still renders the replay
    assert "Day" in str(dashboard.screen_day_replay(b, day, "1m", None))


def test_overlay_feature_trace_adds_secondary_axis_line():
    fig = go.Figure()
    figures.overlay_feature_trace(fig, [1, 2, 3], [0.1, 0.2, 0.3], "atr_dev_1m")
    assert any(getattr(t, "yaxis", None) == "y2" for t in fig.data)
    assert fig.layout.yaxis2.overlaying == "y"


def test_app_exposes_guide_and_repo_tabs():
    layout = str(dashboard.make_app(source="mock").layout)
    assert "How to Use" in layout and "Repo Map" in layout
