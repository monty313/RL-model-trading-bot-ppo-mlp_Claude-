# ==========================================================================
# FILE: barbershop/dashboard.py
# PURPOSE: Visual diagnostics dashboard for the Quantra PPO trading bot.
#          Monty uses this AFTER training to understand why the bot acted
#          the way it did and to write generalised reward/penalty rules.
#          This file is READ-ONLY — it cannot change training or the policy.
# ==========================================================================
#
# DEPENDS ON (reads from these files/modules):
#   data/prices_1m.csv          — 1m EURUSD candles
#   data/prices_5m.csv          — 5m EURUSD candles
#   data/prices_30m.csv         — 30m EURUSD candles
#   data/prices_4h.csv          — 4H EURUSD candles
#   logs/trajectory.parquet     — bot action/state/reward log (from telemetry/logger.py)
#   logs/shap_values.parquet    — SHAP attributions per trade step (from telemetry/interpreter.py)
#   models/ppo_actor.pt         — frozen actor weights for SHAP (from ppo/trainer.py)
#   (the REAL pipeline writes artifacts/telemetry/<run_id>.jsonl; barbershop/adapter.py
#    maps it onto the contract above and flags the fields it cannot yet produce.)
#
# DEPENDED ON BY (these read what this file produces):
#   logs/suggested_rules.json   — pattern finder + Doctor export (read by Monty manually)
#   NOTHING ELSE — this is a terminal tool, not a pipeline component
#
# RELATED BLUEPRINT FILES (read these to understand the terms used here):
#   docs/MLP_INTERPRETABILITY_LAYER.md — hidden state, SHAP, action distribution,
#                                        advantage, value estimate
#   docs/STATE_VECTOR.md               — every feature name and group
#   docs/THE_TRADING_CODE.md           — the 9 laws and 3 gates
#   docs/REWARD_DESIGN.md              — reward layers referenced in advantage
#   docs/PPO_ENGINE.md                 — actor/critic architecture
#
# RULE 3 (READ ONLY): this file only ever WRITES under logs/ (the pattern/Doctor
# export). It never writes to models/, ppo/, env/, or quantra/, and never calls
# the live actor — only frozen weights for SHAP.
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. All 5 screens implemented per
#                            Quantra Barbershop spec v1.0 + Risk Doctor chat box.
#   [2026-06-15] [Claude] — Adversarial-review fixes: wired the APPLY/APPROVE/
#                            MODIFY/IGNORE write-path callbacks (were dead);
#                            marker-click now sets day_id so the autopsy opens;
#                            empty-day guard (was IndexError); plateau banner now
#                            demonstrates; live refresh varies by tick; fail-loud
#                            banner when the Doctor is asked with missing telemetry.
#   [2026-06-15] [Claude] — Audit fix: __main__ now auto-opens the browser
#                            (spec Section 0: "your browser will open automatically").
#   [2026-06-16] [Claude] — Real-data integration: load_bundle gained source=
#                            mock|quantra|spec; load_quantra_bundle() maps the latest
#                            artifacts/telemetry/*.jsonl via adapter (cached) with
#                            empty SHAP + flagged placeholders; available_source()
#                            auto-detects a real run; header shows the data source;
#                            __main__ launches on the real run when present.
#   [2026-06-16] [Claude] — BOSS-review fixes: (1) `python barbershop/dashboard.py`
#                            crashed with ModuleNotFoundError (script mode had only
#                            barbershop/ on sys.path) — bootstrap the repo root onto
#                            sys.path; verified the server now serves HTTP 200.
#                            (2) Enter key now sends in the chat (doctor-input.n_submit
#                            wired) per spec "Also sends on Enter key".
#   [2026-06-16] [Claude] — WI-1: bundle carries feature_names (real header names on a
#                            quantra run); group_indicators + losing_trades receive them
#                            so the SAW panel / heatmap / pattern-ATR are correct on real data.
# ==========================================================================

from __future__ import annotations

import sys
from pathlib import Path

# Allow the spec's documented launch `python barbershop/dashboard.py` to work: in
# script mode Python only puts barbershop/ on sys.path, so `from barbershop import ...`
# fails with ModuleNotFoundError. Put the repo root on sys.path. No-op when imported
# normally (as a package) or under pytest (which already sets the path).
if __package__ in (None, ""):                            # running as a bare script
    _ROOT = str(Path(__file__).resolve().parents[1])
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from dash import ALL, MATCH, Dash, Input, Output, State, ctx, dcc, html, no_update

from barbershop import adapter, config, data, doctor_chat, figures, risk_doctor


# ==========================================================================
# DATA LOADING — fail loud on a missing required file (RULE 4), else fall back
# to deterministic mock data so `python barbershop/dashboard.py` runs out of the
# box with no training run.
# ==========================================================================
def available_source() -> str:
    """Pick the best data source available: a real Quantra run if one exists, else mock.

    Reads: artifacts/telemetry/*.jsonl (via adapter.list_real_runs). Returns "quantra"
    when a real training run is present so the dashboard shows YOUR actual output, else
    "mock" so a fresh checkout still runs out of the box.
    """
    try:
        return "quantra" if adapter.list_real_runs() else "mock"
    except Exception:                                    # any IO/quantra issue -> safe default
        return "mock"


def load_bundle(source: str = "mock", use_mock: Optional[bool] = None) -> Dict[str, Any]:
    """Load everything the screens need (trajectory, shap, prices, training wall).

    Reads: depends on `source`:
      "mock"    — deterministic synthetic data (default; runs with no training run);
      "quantra" — the latest REAL run in artifacts/telemetry/*.jsonl, mapped onto the
                  contract by barbershop.adapter (advantage/SHAP shown as not-yet-produced);
      "spec"    — the spec's logs/*.parquet + data/prices_*.csv (fail-loud if missing).
    `use_mock` (legacy) overrides: True -> "mock", False -> "spec".
    Returns a dict bundle. A missing required file raises data.MissingDataFile, which the
    caller turns into a red banner (RULE 4).
    """
    if use_mock is not None:                             # back-compat with the old kwarg
        source = "mock" if use_mock else "spec"
    if source == "quantra":
        return load_quantra_bundle()
    if source == "spec":
        traj = data.load_trajectory()
        shap = data.load_shap()
        prices = {tf: data.load_prices(tf) for tf in config.TIMEFRAMES}
    else:                                                # "mock"
        traj = data.make_mock_trajectory()
        shap = data.make_mock_shap()
        prices = data.make_mock_prices()
    return {"trajectory": traj, "shap": shap, "prices": prices,
            "training_wall": _mock_training_wall(), "source": source,
            "feature_names": data.MOCK_FEATURE_NAMES, "unavailable_fields": []}


# Cache the (expensive) real-run bundle by run-file path so screen navigation
# doesn't re-parse the JSONL + re-resample ~250k price bars on every click.
_QUANTRA_CACHE: Dict[str, Dict[str, Any]] = {}


def load_quantra_bundle() -> Dict[str, Any]:
    """Build a dashboard bundle from the latest REAL Quantra telemetry run.

    Reads: the newest artifacts/telemetry/<run>.jsonl via barbershop.adapter, and the
    real 1m export resampled to each timeframe. Returns the same bundle shape as
    load_bundle, with: SHAP empty (the live pipeline doesn't produce it yet), and
    `unavailable_fields` listing the placeholder columns so the UI can flag them.
    Raises data.MissingDataFile if there is no real run to show.
    """
    runs = adapter.list_real_runs()
    if not runs:
        raise data.MissingDataFile(config.REAL_TELEMETRY_DIR,
                                   "quantra TelemetryLogger (run training first)")
    latest = runs[-1]
    key = str(latest)
    if key in _QUANTRA_CACHE:
        return _QUANTRA_CACHE[key]
    records = adapter.load_real_run(latest)
    traj = adapter.real_to_trajectory(records)
    fnames = adapter.header_feature_names(records) or data.MOCK_FEATURE_NAMES   # real labels
    # Real candles from the 1m export; fall back to mock candles if it isn't present.
    try:
        prices = adapter.resample_prices_from_1m()
    except Exception:
        prices = data.make_mock_prices()
    # Real input-gradient attribution from the producer's sidecar (autopsy RIGHT column),
    # or an empty frame for older runs without it.
    shap = adapter.load_attribution(latest)
    bundle = {"trajectory": traj, "shap": shap, "prices": prices,
              "training_wall": _mock_training_wall(), "source": "quantra",
              "feature_names": fnames, "unavailable_fields": list(adapter.NOT_YET_PRODUCED)}
    _QUANTRA_CACHE[key] = bundle
    return bundle


def _mock_training_wall(n: int = 40, seed: int = 1) -> Dict[str, List[float]]:
    """Synthesise a rising-then-PLATEAUING pass-rate curve for Screen 1 mock mode.

    Reads: nothing. Returns {iterations, pass_rate}. `seed` varies the curve so the
    60s live refresh visibly moves; the last PLATEAU_CHECKPOINTS+1 points are pinned
    flat (within tolerance) so figures.is_plateaued() is True and the plateau banner
    actually demonstrates (the spec lists it; the original curve never triggered it).
    """
    rng = np.random.default_rng(seed)
    iters = list(range(0, n * 500, 500))
    base = 82 * (1 - np.exp(-np.linspace(0, 3, n)))      # rise toward ~82%
    rate = list(np.clip(base + rng.normal(0, 1.5, n), 0, 100))
    flat = float(np.mean(rate[-(config.PLATEAU_CHECKPOINTS + 1):]))   # plateau value
    for i in range(config.PLATEAU_CHECKPOINTS + 1):      # pin the tail flat
        rate[-(i + 1)] = flat
    return {"iterations": iters, "pass_rate": [float(x) for x in rate]}


def error_banner(exc: "data.MissingDataFile") -> html.Div:
    """Build the red FAIL-LOUD banner naming the missing file + its producer (RULE 4).

    Reads: a MissingDataFile exception. Returns an html.Div clearly stating WHICH
    file is missing and WHAT module produces it — never a silent empty chart.
    """
    return html.Div([
        html.B("⚠️ Required data file missing"),
        html.Div(f"File: {exc.path}"),
        html.Div(f"Produced by: {exc.producer}"),
    ], style={"background": config.COLOR_RED, "color": "#fff", "padding": "12px",
              "borderRadius": "8px", "margin": "8px 0", "whiteSpace": "pre-line"})


# ==========================================================================
# SCREEN BUILDERS — each returns a Dash component tree from the data bundle.
# ==========================================================================
def screen_training_wall(bundle: Dict[str, Any]) -> html.Div:
    """Screen 1 — the live training-wall pass-rate chart + status panel + plateau banner."""
    tw = bundle["training_wall"]
    fig = figures.training_wall_figure(tw["iterations"], tw["pass_rate"])
    plateau = figures.is_plateaued(tw["pass_rate"])
    children = [
        html.H3("Screen 1 — Training Wall"),
        dcc.Graph(id="training-wall-graph", figure=fig),
        # Live refresh every 60s (spec). Disabled in mock mode would still tick;
        # the callback simply rebuilds the (mock) curve.
        dcc.Interval(id="training-wall-interval", interval=config.TRAINING_WALL_REFRESH_MS),
        html.Div([
            html.Span(f"Current iteration: {tw['iterations'][-1]:,}  |  "),
            html.Span(f"Best pass rate: {max(tw['pass_rate']):.1f}%  |  "),
            html.Span("Last checkpoint: (mock)"),
        ], style={"padding": "6px"}),
    ]
    if plateau:
        children.append(html.Div(
            "⚠️ Policy plateau detected — go to Day Scoreboard to investigate",
            id="plateau-banner",
            style={"background": config.COLOR_YELLOW, "padding": "10px",
                   "fontWeight": "bold", "borderRadius": "6px"}))
    return html.Div(children)


def screen_scoreboard(bundle: Dict[str, Any]) -> html.Div:
    """Screen 2 — four day cards, sorted worst (DD breached) first; click to replay."""
    cards = data.day_scoreboard(bundle["trajectory"])
    card_divs = [_scoreboard_card(c) for c in cards]
    return html.Div([
        html.H3("Screen 2 — 4-Day Scoreboard"),
        html.Div(card_divs, style={"display": "flex", "gap": "12px"}),
    ])


def _scoreboard_card(card: Dict[str, Any]) -> html.Div:
    """Build one scoreboard card (PASS/FAIL badge, DD status, regime, P&L, trades)."""
    badge = "PASS ✅" if card["passed"] else "FAIL ❌"
    dd_icon = {"Safe": "Safe", "Warning": "Warning", "Breached": "Breached 🚨"}[card["dd_status"]]
    highlight = card["dd_status"] == "Breached"            # worst day -> red highlight
    return html.Div([
        html.H4(f"Day {card['day_id']} — {card['regime']}"),
        html.Div(f"P&L: {card['pnl_pct']:+.2f}%"),
        html.Div(badge),
        html.Div(f"DD Wall: {dd_icon}"),
        html.Div(f"Trades: {card['n_trades']}"),
    ], id={"type": "scoreboard-card", "index": card["day_id"]}, n_clicks=0,
        style={"flex": "1", "padding": "12px", "borderRadius": "8px", "cursor": "pointer",
               "border": f"2px solid {config.COLOR_RED if highlight else config.COLOR_GREY}",
               "background": "rgba(224,83,61,0.08)" if highlight else "transparent"})


def screen_day_replay(bundle: Dict[str, Any], day_id: int, tf: str = "1m",
                      trade_id: Optional[int] = None) -> html.Div:
    """Screen 3 — candlestick replay for one day/TF, with the 1m-only sub-panels.

    Reads: the bundle + selected day/TF. Returns the TF buttons + candlestick
    (with trade overlays) + (on 1m only) the advantage strip + indicator heatmap.
    """
    traj = bundle["trajectory"]
    day_rows = traj[traj["day_id"] == day_id]
    if day_rows.empty:                                   # stale/out-of-range day_id
        return html.Div([html.H3(f"Screen 3 — Day {day_id} Replay"),
                         html.Div(f"No data for day {day_id}.")])
    trades = data.extract_trades(traj, day_id)
    # Centre the window on the selected trade's entry (or the first trade / day start).
    if trade_id is not None and 1 <= trade_id <= len(trades):
        entry = trades[trade_id - 1]["entry_time"]
    elif trades:
        entry = trades[0]["entry_time"]
    else:
        entry = day_rows["timestamp"].iloc[0]
    window = data.timeframe_window(entry, tf)
    # DD breach marker (if this day breached, mark the FIRST breached bar — the
    # collapse, not end-of-day).
    breached_rows = day_rows[day_rows["dd_breached"]]
    breach_time = breached_rows["timestamp"].iloc[0] if not breached_rows.empty else None
    candle = figures.candlestick_figure(bundle["prices"][tf], trades, tf, entry,
                                        window=window, dd_breach_time=breach_time)
    children = [
        html.H3(f"Screen 3 — Day {day_id} Replay"),
        _tf_buttons(tf),
        dcc.Graph(id="replay-candles", figure=candle),
    ]
    # Panels 2 + 3 are 1m-only (spec).
    if tf == "1m":
        adv = data.advantage_series(traj, day_id)
        children.append(dcc.Graph(id="replay-advantage",
                                  figure=figures.advantage_figure(adv, window=window)))
        fnames = bundle.get("feature_names")
        grouped = [data.group_indicators(r, fnames)
                   for _, r in day_rows.sort_values("timestamp").iterrows()]
        ts = list(day_rows.sort_values("timestamp")["timestamp"])
        children.append(dcc.Graph(id="replay-heatmap",
                                  figure=figures.heatmap_figure(grouped, ts)))
    return html.Div(children)


def _tf_buttons(active: str) -> html.Div:
    """Build the [1m][5m][30m][4H] timeframe buttons; the active one is highlighted."""
    return html.Div([
        html.Button(tf, id={"type": "tf-button", "index": tf}, n_clicks=0,
                    style={"marginRight": "6px", "fontWeight": "bold" if tf == active else "normal",
                           "border": f"2px solid {config.COLOR_GOLD if tf == active else config.COLOR_GREY}"})
        for tf in config.TIMEFRAMES
    ], style={"margin": "6px 0"})


def screen_autopsy(bundle: Dict[str, Any], day_id: int, trade_id: int) -> html.Div:
    """Screen 4 — trade autopsy: SAW | chose | caused, side by side (right panel)."""
    traj = bundle["trajectory"]
    trades = data.extract_trades(traj, day_id)
    if not (1 <= trade_id <= len(trades)):
        return html.Div("Select a trade marker to open the autopsy.")
    tr = trades[trade_id - 1]
    # The decision row = the trade's entry row.
    entry_rows = traj[(traj["day_id"] == day_id) & (traj["timestamp"] == tr["entry_time"])]
    row = entry_rows.iloc[0]
    groups = data.group_indicators(row, bundle.get("feature_names"))
    bars = data.action_probability_bars(row)
    masked, legal = data.masked_legal(row)
    # SHAP for this trade step.
    shap_df = bundle["shap"]
    srows = shap_df[(shap_df["day_id"] == day_id) & (shap_df["step"] == int(row["step"]))]
    if len(srows):
        ss = data.shap_sorted(srows.iloc[0], row["action"])
    else:
        ss = {"chosen_action": row["action"], "toward": [], "away": [], "explained": 0.0}

    left = html.Div([html.H4("What the bot SAW"),
                     html.Small(str(tr["entry_time"])),
                     dcc.Graph(figure=figures.state_bars_figure(groups))],
                    style={"flex": "1"})
    middle = html.Div([html.H4("Why it chose this action"),
                       dcc.Graph(figure=figures.action_prob_figure(bars)),
                       html.Div(f"LEGAL: {', '.join(legal)}"),
                       html.Div(f"MASKED: {', '.join(masked) or 'none'}",
                                style={"color": config.COLOR_RED})],
                      style={"flex": "1"})
    right = html.Div([html.H4("What CAUSED that probability"),
                      dcc.Graph(figure=figures.shap_figure(ss))],
                     style={"flex": "1"})
    return html.Div([html.H3(f"Screen 4 — Trade #{trade_id} Autopsy (Day {day_id})"),
                     html.Div([left, middle, right], style={"display": "flex", "gap": "10px"})])


def screen_pattern_finder(bundle: Dict[str, Any]) -> html.Div:
    """Screen 5 — auto-scan losing trades, surface top 3 patterns, apply/ignore/modify."""
    losing = data.losing_trades(bundle["trajectory"])
    # When the trajectory has too few losers to mine, demonstrate on the mock set.
    if len(losing) < 3:
        losing = data.make_mock_losing_trades()
    patterns = data.find_patterns(losing)
    blocks = [_pattern_block(p) for p in patterns]
    # Confirmation line for the [✅ APPLY RULE] export write path.
    status = html.Div(id="pattern-export-status",
                      style={"color": config.COLOR_GREEN, "fontWeight": "bold"})
    return html.Div([html.H3("Screen 5 — Pattern Finder")] + blocks + [status])


def _pattern_block(p: Dict[str, Any]) -> html.Div:
    """Build one pattern card with the plain-English finding + Apply/Ignore/Modify."""
    return html.Div([
        html.H4(f"Pattern {p['rank']} — Found in {p['count']} of {p['total']} losing trades"),
        html.Ul([html.Li(c) for c in p["conditions"]]),
        html.Div(html.B("Suggested rule: ")),
        html.Div(p["suggested_rule"]),
        html.Div([
            html.Button("✅ APPLY RULE", id={"type": "pattern-apply", "index": p["rank"]}, n_clicks=0),
            html.Button("❌ IGNORE", id={"type": "pattern-ignore", "index": p["rank"]}, n_clicks=0),
            html.Button("✏️ MODIFY", id={"type": "pattern-modify", "index": p["rank"]}, n_clicks=0),
        ], style={"marginTop": "6px"}),
        dcc.Input(id={"type": "pattern-text", "index": p["rank"]}, value=p["suggested_rule"],
                  style={"width": "90%", "display": "none"}),
    ], id={"type": "pattern-block", "index": p["rank"]},
        style={"border": f"1px solid {config.COLOR_GREY}", "borderRadius": "8px",
               "padding": "10px", "margin": "8px 0"})


# ==========================================================================
# THE APP.
# ==========================================================================
def make_app(source: Optional[str] = None, use_mock: Optional[bool] = None) -> Dash:
    """Build the Dash app: tab nav over the 5 screens + the Risk Doctor chat box.

    Reads: the data bundle for `source` ("mock" | "quantra" | "spec"). `use_mock`
    (legacy) maps True->"mock", False->"spec". Default source is "mock" so tests +
    a fresh checkout run offline; __main__ launches with available_source() so a real
    training run is shown automatically when present. Returns a configured Dash app
    with all callbacks registered. RULE 3: nothing here writes outside logs/.
    """
    if use_mock is not None:
        source = "mock" if use_mock else "spec"
    source = source or "mock"
    app = Dash(__name__, suppress_callback_exceptions=True)
    init_state = {"screen": 1, "day_id": None, "trade_id": None, "tf": "1m", "source": source}
    src_label = {"mock": "MOCK data (no training run loaded)",
                 "quantra": "REAL Quantra telemetry", "spec": "spec logs/*.parquet"}[source]

    app.layout = html.Div([
        dcc.Store(id="screen-state", data=init_state),
        dcc.Store(id="last-diagnosis", data=None),        # latest (q, text) for [APPROVE]
        html.H2("Quantra Barbershop — training diagnostics"),
        html.Div(f"Data source: {src_label}", style={"color": config.COLOR_GREY,
                                                      "fontSize": "12px", "marginBottom": "6px"}),
        dcc.Tabs(id="screen-tabs", value="s1", children=[
            dcc.Tab(label="1 · Training Wall", value="s1"),
            dcc.Tab(label="2 · Scoreboard", value="s2"),
            dcc.Tab(label="3 · Day Replay", value="s3"),
            dcc.Tab(label="4 · Trade Autopsy", value="s4"),
            dcc.Tab(label="5 · Pattern Finder", value="s5"),
        ]),
        html.Div(id="screen-content"),
        doctor_chat.chat_panel(init_state),
    ], style={"fontFamily": "system-ui, sans-serif", "padding": "12px"})

    _register_callbacks(app)
    return app


def _source_note(bundle: Dict[str, Any]):
    """A small amber note for REAL runs listing the not-yet-produced placeholder fields.

    Reads: the bundle's unavailable_fields. Returns an html.Div (or None for mock) so
    Monty knows which columns are placeholders on a real run and must not be trusted.
    """
    unavailable = bundle.get("unavailable_fields") or []
    if not unavailable:
        return None
    return html.Div(
        "ℹ️ Real run: these fields are not produced by the live pipeline yet and show as "
        "placeholders — " + ", ".join(unavailable),
        style={"background": "rgba(229,188,36,0.18)", "padding": "6px",
               "borderRadius": "6px", "fontSize": "12px", "marginBottom": "6px"})


def _content_for(state: Dict[str, Any]) -> html.Div:
    """Render the active screen, catching a missing-file error into a red banner."""
    try:
        bundle = load_bundle(source=state.get("source", "mock"))
    except data.MissingDataFile as exc:
        return error_banner(exc)                          # RULE 4 — fail loud
    screen = state.get("screen", 1)
    if screen == 1:
        body = screen_training_wall(bundle)
    elif screen == 2:
        body = screen_scoreboard(bundle)
    elif screen == 3:
        body = screen_day_replay(bundle, state.get("day_id") or 1,
                                 state.get("tf", "1m"), state.get("trade_id"))
    elif screen == 4:
        if state.get("day_id") is None or state.get("trade_id") is None:
            body = html.Div("Click a trade marker on Screen 3 to open the autopsy.")
        else:
            body = screen_autopsy(bundle, state["day_id"], state["trade_id"])
    elif screen == 5:
        body = screen_pattern_finder(bundle)
    else:
        body = html.Div("Unknown screen.")
    note = _source_note(bundle)                           # placeholder warning on real runs
    return html.Div([note, body]) if note is not None else body


def _register_callbacks(app: Dash) -> None:
    """Wire every interaction (tab/card/TF/trade nav, pattern apply, Doctor chat)."""

    # --- Navigation: tab change -> screen number -> rebuild content. ---
    @app.callback(Output("screen-state", "data"),
                  Output("screen-content", "children"),
                  Input("screen-tabs", "value"),
                  Input({"type": "scoreboard-card", "index": ALL}, "n_clicks"),
                  Input({"type": "tf-button", "index": ALL}, "n_clicks"),
                  Input("replay-candles", "clickData"),
                  State("screen-state", "data"))
    def navigate(tab, card_clicks, tf_clicks, candle_click, state):
        """Update screen-state from the triggering control and rebuild the content."""
        state = dict(state or {})
        trig = ctx.triggered_id
        if trig == "screen-tabs":
            state["screen"] = int(str(tab)[-1])
        elif isinstance(trig, dict) and trig.get("type") == "scoreboard-card":
            state.update(screen=3, day_id=int(trig["index"]), trade_id=None, tf="1m")
        elif isinstance(trig, dict) and trig.get("type") == "tf-button":
            state["tf"] = trig["index"]; state["screen"] = 3
        elif trig == "replay-candles" and candle_click:
            # A trade marker click carries customdata=trade_id -> open the autopsy.
            pts = candle_click.get("points", [{}])
            cd = pts[0].get("customdata")
            if cd is not None:
                tid = cd[0] if isinstance(cd, list) else cd
                # Persist day_id (default to the replay's day 1) so the Screen-4 guard
                # passes and the autopsy actually opens via the tab path.
                state.update(screen=4, day_id=state.get("day_id") or 1, trade_id=int(tid))
        return state, _content_for(state)

    # --- Screen 1 live refresh (rebuild the training-wall curve). ---
    @app.callback(Output("training-wall-graph", "figure"),
                  Input("training-wall-interval", "n_intervals"),
                  prevent_initial_call=True)
    def refresh_wall(n):
        """Rebuild the training-wall figure on each 60s interval tick (Screen 1 live)."""
        tw = _mock_training_wall(seed=int(n or 0) + 1)    # vary by tick so it visibly moves
        return figures.training_wall_figure(tw["iterations"], tw["pass_rate"])

    # --- Doctor: expand/collapse the chat panel. ---
    @app.callback(Output("doctor-body", "style"),
                  Input("doctor-toggle", "n_clicks"),
                  State("doctor-body", "style"))
    def toggle_doctor(n, style):
        """Expand/collapse the Risk Doctor chat body on each tab click."""
        style = dict(style or {})
        style["display"] = "none" if (n or 0) % 2 == 1 else "block"
        return style

    # --- Doctor: context indicator follows navigation (spec TEST 15). ---
    @app.callback(Output("doctor-context-indicator", "children"),
                  Input("screen-state", "data"))
    def update_context(state):
        """Refresh the 'Doctor is looking at: ...' indicator from screen-state."""
        return doctor_chat.context_indicator_text(state or {})

    # --- Doctor: send a message / full diagnosis -> append to the conversation. ---
    @app.callback(Output("doctor-conversation", "children"),
                  Output("last-diagnosis", "data"),
                  Input("doctor-send", "n_clicks"),
                  Input("doctor-input", "n_submit"),       # Enter key in the input (spec)
                  Input("doctor-full-diagnosis", "n_clicks"),
                  State("doctor-input", "value"),
                  State("doctor-conversation", "children"),
                  State("screen-state", "data"),
                  prevent_initial_call=True)
    def doctor_send(send_n, submit_n, full_n, question, convo, state):
        """Send Monty's question (or a full diagnosis) to the Doctor; append the reply."""
        convo = list(convo or [])
        full = ctx.triggered_id == "doctor-full-diagnosis"
        q = question or ("Give me a full diagnosis of this day." if full else "")
        if not q:
            return convo, no_update
        try:
            bundle = load_bundle(source=(state or {}).get("source", "mock"))
            traj, shap = bundle["trajectory"], bundle["shap"]
            missing = None
        except data.MissingDataFile as exc:
            traj = shap = None
            missing = exc
        convo.append(doctor_chat.render_user_message(q))
        # If the telemetry is missing, fail loud (RULE 4) and do NOT diagnose.
        if missing is not None:
            convo.append(error_banner(missing))
            return convo, no_update
        resp = risk_doctor.ask(q, state or {}, trajectory=traj, shap=shap, full_diagnosis=full)
        if resp.get("offline") or resp.get("manual_missing"):
            convo.append(doctor_chat.render_offline_banner(resp["text"]))
            return convo, no_update
        convo.append(doctor_chat.render_doctor_message(resp, msg_id=len(convo)))
        # Remember this exchange so [APPROVE] can export its prescription. Only real,
        # evidence-backed diagnoses are approvable (not refusals / insufficient-evidence).
        diag = no_update
        if not resp.get("refused") and not resp.get("insufficient_evidence"):
            diag = {"question": q, "text": resp["text"], "state": state}
        return convo, diag

    # --- Doctor: [✅ APPROVE] -> export the prescription to logs/suggested_rules.json. ---
    @app.callback(Output("doctor-approve-status", "children"),
                  Input({"type": "doctor-approve", "index": ALL}, "n_clicks"),
                  State("last-diagnosis", "data"),
                  State("screen-state", "data"),
                  prevent_initial_call=True)
    def approve_doctor(clicks, last, state):
        """Export the latest Doctor prescription to suggested_rules.json (sanctioned write)."""
        if not clicks or not any(clicks) or not last:
            return no_update
        path = risk_doctor.approve_prescription(last["question"], last["text"],
                                                last.get("state") or state or {})
        return f"✅ Prescription exported to {path}"

    # --- Pattern Finder: [✅ APPLY RULE] -> export the (possibly edited) rule. ---
    @app.callback(Output("pattern-export-status", "children"),
                  Input({"type": "pattern-apply", "index": ALL}, "n_clicks"),
                  State({"type": "pattern-text", "index": ALL}, "value"),
                  State("screen-state", "data"),
                  prevent_initial_call=True)
    def apply_pattern(clicks, texts, state):
        """Export the clicked pattern (with any MODIFY edit) to suggested_rules.json."""
        if not clicks or not any(clicks):
            return no_update
        rank = ctx.triggered_id["index"]                  # which pattern's APPLY fired
        bundle = load_bundle(source=(state or {}).get("source", "mock"))
        losing = data.losing_trades(bundle["trajectory"], bundle.get("feature_names"))
        if len(losing) < 3:
            losing = data.make_mock_losing_trades()
        patterns = data.find_patterns(losing)
        match = next((p for p in patterns if p["rank"] == rank), None)
        if match is None:
            return no_update
        # Use the edited text if MODIFY revealed the input and changed it.
        edited = texts[rank - 1] if texts and rank - 1 < len(texts) else None
        entry = {"source": "pattern_finder", **match}
        if edited:
            entry["suggested_rule"] = edited
        path = data.export_rule(entry)
        return f"✅ Rule exported to {path}"

    # --- Pattern Finder: [✏️ MODIFY] reveals the editable rule text. ---
    @app.callback(Output({"type": "pattern-text", "index": MATCH}, "style"),
                  Input({"type": "pattern-modify", "index": MATCH}, "n_clicks"),
                  prevent_initial_call=True)
    def modify_pattern(_n):
        """Reveal the hidden rule-text input so Monty can edit before exporting."""
        return {"width": "90%", "display": "block"}

    # --- Pattern Finder: [❌ IGNORE] hides the pattern card. ---
    @app.callback(Output({"type": "pattern-block", "index": MATCH}, "style"),
                  Input({"type": "pattern-ignore", "index": MATCH}, "n_clicks"),
                  prevent_initial_call=True)
    def ignore_pattern(_n):
        """Hide a dismissed pattern card."""
        return {"display": "none"}


# Entry point: `python barbershop/dashboard.py` -> open http://localhost:8050.
if __name__ == "__main__":          # pragma: no cover (manual launch only)
    import webbrowser
    url = f"http://{config.DASH_HOST}:{config.DASH_PORT}"
    print(f"Quantra Barbershop -> {url}  (Ctrl+C to stop)")
    # Open the browser automatically (spec Section 0: "your browser will open").
    try:
        webbrowser.open(url)
    except Exception:                # headless / no browser -> just print the URL
        pass
    make_app(source=available_source()).run(host=config.DASH_HOST, port=config.DASH_PORT, debug=False)
