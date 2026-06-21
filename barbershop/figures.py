# ==========================================================================
# FILE: barbershop/figures.py
# PURPOSE: Plotly figure builders for the Barbershop dashboard. Pure functions
#          that take already-loaded/transformed data (from barbershop.data) and
#          return go.Figure objects. No Dash import here, so the figures (axis
#          ranges, overlays, vertical lines) are inspectable in unit tests
#          without a running server.
# ==========================================================================
#
# DEPENDS ON: plotly.graph_objects, barbershop.config, barbershop.data.
# PRODUCES:   nothing — returns figures for the Dash layer to display.
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Candlestick + overlays + trade markers,
#                            advantage strip, indicator heatmap, autopsy state
#                            bars, action-probability bars, SHAP bars.
# ==========================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from barbershop import config


def _vline(fig: go.Figure, ts: pd.Timestamp, color: str, text: str) -> None:
    """Add a full-height dashed vertical line + label at time `ts`.

    Reads: nothing. Mutates `fig`. We build the line as an explicit paper-height
    shape (and a separate annotation) using an ISO timestamp string, because
    plotly's add_vline(annotation_text=...) does arithmetic on the x value that
    raises on pandas Timestamps. ISO strings render correctly on a datetime axis.
    """
    xs = pd.Timestamp(ts).isoformat()                     # ISO string -> safe on datetime axis
    fig.add_shape(type="line", x0=xs, x1=xs, xref="x", yref="paper", y0=0, y1=1,
                  line=dict(color=color, dash="dash", width=1.5))
    fig.add_annotation(x=xs, xref="x", yref="paper", y=1.02, text=text,
                       showarrow=False, font=dict(color=color, size=11))


# ==========================================================================
# SCREEN 1 — TRAINING WALL. A single pass-rate line, coloured by trend.
# ==========================================================================
def training_wall_figure(iterations: List[int], pass_rate: List[float]) -> go.Figure:
    """Build the Screen-1 pass-rate line chart with the 80% "consistent pass" line.

    Reads: parallel iteration + pass-rate lists. Returns a go.Figure. The line is
    GREEN when the recent trend rises, YELLOW when flat (+/- tolerance), RED when
    falling — so a glance tells Monty whether the policy is still learning.
    """
    color = _trend_color(pass_rate)                       # trend -> green/yellow/red
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y=pass_rate, mode="lines",
                             line=dict(color=color, width=2), name="pass rate"))
    # The 80% "Consistent Pass Zone" reference line.
    fig.add_hline(y=config.CONSISTENT_PASS_ZONE, line_dash="dash",
                  line_color=config.COLOR_GREY,
                  annotation_text="Consistent Pass Zone", annotation_position="top left")
    fig.update_layout(title="Training Wall — pass rate over training",
                      xaxis_title="training iteration", yaxis_title="pass rate (%)",
                      yaxis_range=[0, 100], template="plotly_white", height=360)
    return fig


def _trend_color(series: List[float]) -> str:
    """Green if the last few points rise, red if they fall, yellow if flat."""
    if len(series) < 2:
        return config.COLOR_GREY
    recent = series[-min(len(series), config.PLATEAU_CHECKPOINTS + 1):]
    delta = recent[-1] - recent[0]                        # net change over the window
    if delta > config.PLATEAU_TOLERANCE_PCT:
        return config.COLOR_GREEN
    if delta < -config.PLATEAU_TOLERANCE_PCT:
        return config.COLOR_RED
    return config.COLOR_YELLOW                            # within tolerance -> plateau


def is_plateaued(pass_rate: List[float]) -> bool:
    """True if pass rate has been flat (within tolerance) for the last N checkpoints."""
    if len(pass_rate) <= config.PLATEAU_CHECKPOINTS:
        return False
    recent = pass_rate[-(config.PLATEAU_CHECKPOINTS + 1):]
    return (max(recent) - min(recent)) <= config.PLATEAU_TOLERANCE_PCT


# ==========================================================================
# SCREEN 3 PANEL 1 — CANDLESTICK with BB/SMA overlays + trade markers.
# ==========================================================================
def candlestick_figure(prices: pd.DataFrame, trades: List[Dict[str, Any]], tf: str,
                       entry_time: pd.Timestamp,
                       window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
                       dd_breach_time: Optional[pd.Timestamp] = None) -> go.Figure:
    """Build the Screen-3 candlestick chart for one timeframe.

    Reads: a price frame (timestamp/open/high/low/close), the day's trades, the
    selected TF, the selected trade's entry_time, and the context window. Returns
    a go.Figure with: real candles, BB20 band fill + BB200 midline + shifted-SMA
    high/low lines, per-trade entry/exit markers (clickable: customdata=trade_id)
    and profit/loss shaded regions, an entry vertical line on 5m/30m/4H, and an
    optional "DD WALL BREACHED" vertical line. The x-axis is pinned to `window`.
    """
    df = prices.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if window is not None:
        df = df[(df["timestamp"] >= window[0]) & (df["timestamp"] <= window[1])]

    fig = go.Figure()
    # The real EURUSD candles.
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="EURUSD"))
    _add_band_overlays(fig, df)                           # BB20 fill, BB200 mid, shifted SMA

    # Trade markers + profit/loss shading.
    for tr in trades:
        _add_trade_overlay(fig, df, tr)

    # Entry vertical line — shown on the higher TFs so Monty can place the trade
    # against the bigger-picture structure (spec Screen 3 + TEST 10).
    if tf != "1m":
        _vline(fig, entry_time, config.COLOR_GOLD, "trade entry")
    # DD wall breach marker, if this day breached.
    if dd_breach_time is not None:
        _vline(fig, dd_breach_time, config.COLOR_RED, "DD WALL BREACHED")

    fig.update_layout(title=f"Day replay — {tf}", xaxis_title="time",
                      yaxis_title="price", template="plotly_white", height=460,
                      xaxis_rangeslider_visible=False)
    if window is not None:
        # Pin the x-axis EXACTLY to the context window (spec TEST 3).
        fig.update_xaxes(range=[window[0], window[1]])
    return fig


def _add_band_overlays(fig: go.Figure, df: pd.DataFrame) -> None:
    """Add BB20 band fill, BB200 midline, and shifted-SMA high/low to a candle fig.

    Reads: the (windowed) price frame. Bands are computed inline (rolling) so the
    overlay renders on any OHLC frame without needing precomputed feature columns.
    """
    if len(df) < 5:
        return                                            # too few bars to band
    c = df["close"]
    bb20_mid = c.rolling(20, min_periods=1).mean()
    bb20_sd = c.rolling(20, min_periods=1).std(ddof=0).fillna(0)
    bb200_mid = c.rolling(200, min_periods=1).mean()
    ssma_hi = df["high"].rolling(4, min_periods=1).mean().shift(4)
    ssma_lo = df["low"].rolling(4, min_periods=1).mean().shift(4)
    x = df["timestamp"]
    # BB20 upper + lower with a transparent blue fill between them.
    fig.add_trace(go.Scatter(x=x, y=bb20_mid + 2 * bb20_sd, mode="lines",
                             line=dict(color="rgba(60,120,216,0.0)"), showlegend=False,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bb20_mid - 2 * bb20_sd, mode="lines",
                             fill="tonexty", fillcolor="rgba(60,120,216,0.12)",
                             line=dict(color="rgba(60,120,216,0.0)"), name="BB20",
                             hoverinfo="skip"))
    # BB200 midline (grey dashed).
    fig.add_trace(go.Scatter(x=x, y=bb200_mid, mode="lines",
                             line=dict(color=config.COLOR_GREY, dash="dash", width=1),
                             name="BB200 mid", hoverinfo="skip"))
    # Shifted SMA on high/low (orange dashed).
    for y, nm in ((ssma_hi, "shifted SMA high"), (ssma_lo, "shifted SMA low")):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 line=dict(color="#E08A1E", dash="dot", width=1),
                                 name=nm, hoverinfo="skip"))


def _add_trade_overlay(fig: go.Figure, df: pd.DataFrame, tr: Dict[str, Any]) -> None:
    """Add one trade's entry/exit markers + profit/loss shading to a candle fig.

    Reads: the windowed price frame + a trade record. The markers carry
    customdata=trade_id so a click opens Screen 4 for that trade.
    """
    entry_p = _price_at(df, tr["entry_time"])
    exit_p = _price_at(df, tr["exit_time"])
    if entry_p is None:
        return                                            # trade is outside this window
    is_long = tr["direction"] == "LONG"
    # Entry triangle: green up for long, red down for short.
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp(tr["entry_time"])], y=[entry_p], mode="markers",
        marker=dict(symbol="triangle-up" if is_long else "triangle-down",
                    size=14, color=config.COLOR_GREEN if is_long else config.COLOR_RED),
        name=f"entry #{tr['trade_id']}", customdata=[tr["trade_id"]],
        hovertemplate=f"Trade #{tr['trade_id']} entry ({tr['direction']})<extra></extra>"))
    # Exit circle.
    if exit_p is not None:
        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(tr["exit_time"])], y=[exit_p], mode="markers",
            marker=dict(symbol="circle", size=11, color=config.COLOR_RED),
            name=f"exit #{tr['trade_id']}", customdata=[tr["trade_id"]],
            hovertemplate=f"Trade #{tr['trade_id']} exit<extra></extra>"))
        # Shade the held region green (profit) or red (loss).
        fig.add_vrect(x0=pd.Timestamp(tr["entry_time"]), x1=pd.Timestamp(tr["exit_time"]),
                      fillcolor=config.COLOR_PROFIT_FILL if tr["profit"] else config.COLOR_LOSS_FILL,
                      line_width=0, layer="below")


def _price_at(df: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
    """Return the close price of the bar nearest to ts (or None if df is empty)."""
    if len(df) == 0:
        return None
    ts = pd.Timestamp(ts)
    idx = (df["timestamp"] - ts).abs().idxmin()           # nearest bar by time
    return float(df.loc[idx, "close"])


# ==========================================================================
# SCREEN 3 PANEL 2 — ADVANTAGE STRIP (1m only).
# ==========================================================================
def advantage_figure(adv: pd.DataFrame,
                     window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> go.Figure:
    """Build the GAE-advantage bar strip, time-aligned to the candle chart.

    Reads: a [timestamp, advantage] frame + the shared context window. Returns a
    go.Figure: green bars above zero (beat the critic), red below (worse than
    expected), a zero line. The x-axis is pinned to the SAME window as Panel 1 so
    bars line up bar-for-bar (spec TEST 4).
    """
    ts = pd.to_datetime(adv["timestamp"], utc=True)
    vals = adv["advantage"].astype(float)
    colors = [config.COLOR_GREEN if v >= 0 else config.COLOR_RED for v in vals]
    fig = go.Figure(go.Bar(x=ts, y=vals, marker_color=colors, name="advantage"))
    fig.add_hline(y=0.0, line_color=config.COLOR_GREY, line_width=1)
    fig.update_layout(title="Advantage (vs Critic expectation)", xaxis_title="time",
                      yaxis_title="GAE advantage", template="plotly_white", height=200,
                      showlegend=False)
    if window is not None:
        fig.update_xaxes(range=[window[0], window[1]])    # identical to candlestick range
    return fig


# ==========================================================================
# SCREEN 3 PANEL 3 — INDICATOR HEATMAP RIBBON (1m only).
# ==========================================================================
# Numeric encoding for the discrete heatmap colourscale.
_COLOR_TO_Z = {config.COLOR_RED: -1.0, config.COLOR_YELLOW: 0.0,
               config.COLOR_GREEN: 1.0, config.COLOR_GREY: float("nan")}
_HEATMAP_SCALE = [[0.0, config.COLOR_RED], [0.5, config.COLOR_YELLOW], [1.0, config.COLOR_GREEN]]


def heatmap_figure(rows: List[Dict[str, Any]], timestamps: List[pd.Timestamp]) -> go.Figure:
    """Build the indicator heatmap ribbon: rows = indicators, cols = candles.

    Reads: per-timestep grouped indicator rows (from data.group_indicators) and the
    aligned timestamps. Returns a go.Heatmap whose cells use the same green/red/
    yellow/grey language as everywhere else. Rows are grouped by the 5 categories.
    """
    row_labels: List[str] = []
    z: List[List[float]] = []
    # `rows` is a list (per timestep) of grouped-indicator structures; flatten the
    # FIRST timestep to get the row order, then fill each column.
    if not rows:
        return go.Figure()
    ordered_cells = [(g["label"], c["name"]) for g in rows[0] for c in g["cells"]]
    for label, name in ordered_cells:
        row_labels.append(f"{label} · {name}")
        series = []
        for step_groups in rows:
            color = _lookup_cell_color(step_groups, name)
            series.append(_COLOR_TO_Z.get(color, float("nan")))
        z.append(series)
    fig = go.Figure(go.Heatmap(
        z=z, x=[pd.Timestamp(t) for t in timestamps], y=row_labels,
        colorscale=_HEATMAP_SCALE, zmid=0, showscale=False))
    fig.update_layout(title="Indicator heatmap", template="plotly_white",
                      height=max(240, 18 * len(row_labels)), xaxis_title="time")
    return fig


def _lookup_cell_color(step_groups: List[Dict[str, Any]], name: str) -> str:
    """Find the colour of indicator `name` within one timestep's grouped structure."""
    for g in step_groups:
        for c in g["cells"]:
            if c["name"] == name:
                return c["color"]
    return config.COLOR_GREY


# ==========================================================================
# SCREEN 4 -> SCREEN 3 — clickable indicator overlay (C25).
# ==========================================================================
def overlay_feature_trace(fig: go.Figure, times: List[Any], values: List[float],
                          name: str) -> go.Figure:
    """C25: overlay ONE SAW feature's per-bar series on a candlestick, on a right-hand y-axis, so
    clicking a feature in Screen 4's 'what the bot SAW' panel shows the indicator the decision rode on.
    COUPLING -> dashboard.screen_day_replay (feeds the day_rows obs_vector series) + the Screen-4
    saw-bars click. Mutates + returns the passed figure."""
    fig.add_trace(go.Scatter(x=list(times), y=list(values), mode="lines",
                             name=f"SAW · {name}", yaxis="y2",
                             line=dict(width=1.8, color="#7F77DD")))
    fig.update_layout(yaxis2=dict(title=name, overlaying="y", side="right", showgrid=False))
    return fig


# ==========================================================================
# SCREEN 4 LEFT — "WHAT THE BOT SAW" state bars.
# ==========================================================================
def state_bars_figure(groups: List[Dict[str, Any]]) -> go.Figure:
    """Build horizontal coloured bars of the full state vector, grouped by category.

    Reads: the grouped-indicator structure for one decision row. Returns a go.Figure
    of horizontal bars: length = |value|, colour = the cell's state colour.
    """
    names: List[str] = []
    vals: List[float] = []
    colors: List[str] = []
    for g in groups:
        for c in g["cells"]:
            names.append(f"{g['label']} · {c['name']}")
            v = c["value"]
            vals.append(abs(float(v)) if isinstance(v, (int, float)) else 1.0)
            colors.append(c["color"])
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
    fig.update_layout(title="What the bot SAW", template="plotly_white",
                      height=max(260, 16 * len(names)), xaxis_title="magnitude",
                      showlegend=False)
    return fig


# ==========================================================================
# SCREEN 4 MIDDLE — action probability bars.
# ==========================================================================
def action_prob_figure(bars: List[Dict[str, Any]]) -> go.Figure:
    """Build the 4 action-probability bars; the chosen action gets a gold border.

    Reads: the action-bar list from data.action_probability_bars. Returns a
    go.Figure of horizontal bars labelled with the action icon + percentage.
    """
    labels = [f"{b['icon']} {b['action']}" for b in bars]
    vals = [b["prob"] * 100.0 for b in bars]
    # Outline the chosen action's bar in gold (spec Screen 4).
    line_widths = [3 if b["chosen"] else 0 for b in bars]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=config.COLOR_GREY,
                    line=dict(color=config.COLOR_GOLD, width=line_widths)),
        text=[f"{v:.0f}%" for v in vals], textposition="outside"))
    fig.update_layout(title="Why it chose this action", xaxis_title="probability (%)",
                      xaxis_range=[0, 100], template="plotly_white", height=240,
                      showlegend=False)
    return fig


# ==========================================================================
# SCREEN 4 RIGHT — SHAP attribution bars.
# ==========================================================================
def shap_figure(shap: Dict[str, Any]) -> go.Figure:
    """Build the SHAP bars: green PUSHED-TOWARD (desc) + red PUSHED-AWAY (desc).

    Reads: the dict from data.shap_sorted. Returns a go.Figure with two coloured
    groups, each sorted largest-first, labelled with indicator name + percentage.
    """
    fig = go.Figure()
    action = shap.get("chosen_action", "")
    if shap["toward"]:
        names = [k for k, _ in shap["toward"]]
        vals = [v for _, v in shap["toward"]]
        fig.add_trace(go.Bar(x=vals, y=names, orientation="h",
                             marker_color=config.COLOR_GREEN,
                             name=f"toward {action}"))
    if shap["away"]:
        names = [k for k, _ in shap["away"]]
        vals = [-v for _, v in shap["away"]]               # negative side for "away"
        fig.add_trace(go.Bar(x=vals, y=names, orientation="h",
                             marker_color=config.COLOR_RED,
                             name=f"away from {action}"))
    # Honest label: this is input x gradient attribution, NOT Shapley values.
    fig.update_layout(
        title=f"What CAUSED it — input×gradient attribution (top {shap.get('explained', 0):.0f}% shown)",
        xaxis_title="attribution (toward + / away −)", template="plotly_white",
        height=300, barmode="overlay")
    return fig
