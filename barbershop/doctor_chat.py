# ==========================================================================
# FILE: barbershop/doctor_chat.py
# PURPOSE: The Risk Doctor chat box — a floating Dash panel docked bottom-right
#          on every Barbershop screen, plus the PURE render helpers that turn a
#          Doctor response into the six-section display (📍🔍🎯✅❌📊). The pure
#          helpers carry the logic the tests assert on; the Dash component is the
#          thin shell around them. The chat box only DISPLAYS the Doctor — it has
#          no execution authority.
# ==========================================================================
#
# DEPENDS ON: dash (html, dcc), barbershop.config, barbershop.risk_doctor.
# PRODUCES:   nothing — returns Dash components for dashboard.py to mount.
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Floating chat panel (collapsed tab,
#                            top bar with Full Diagnosis, conversation area,
#                            input, context indicator) + six-section response
#                            formatter + approve/dismiss buttons.
#   [2026-06-15] [Claude] — Adversarial-review fixes: segment response sections by
#                            unique ICON (robust to paraphrase); added the
#                            doctor-approve-status line for the [APPROVE] write path.
#   [2026-06-16] [Claude] — WI-6: format_sections falls back to section TITLES when a
#                            local model drops the emoji icons, and to a single
#                            'What I see' block when neither is present (reply not lost).
#   [2026-06-19] [C24]    — Surfaced the read-only `/read <path>` slash command in the input
#                            placeholder (risk_doctor.ask intercepts it; no write path).
# ==========================================================================

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from dash import dcc, html

from barbershop import config, risk_doctor


# ==========================================================================
# PURE RENDER HELPERS (unit-tested without a running server).
# ==========================================================================
def format_sections(doctor_text: str) -> List[Dict[str, str]]:
    """Split a Doctor response into the six display sections (spec Section D).

    Reads: the Doctor's raw text. Returns a list of EXACTLY six
    {icon, title, body} dicts in the fixed order (📍 looking at / 🔍 see /
    🎯 means / ✅ do next / ❌ don't / 📊 confidence). Body text is extracted by
    locating each section's title; a section with no match shows
    "insufficient evidence" so the UI never renders a blank header.
    """
    text = doctor_text or ""
    low = text.lower()
    secs = config.DOCTOR_SECTIONS
    # Locate each section by its UNIQUE icon if present, ELSE by its title text
    # (a real local model may paraphrase or drop the emoji). pos_by_i[i] = (start, marker).
    pos_by_i: Dict[int, Tuple[int, str]] = {}
    for i, (icon, title) in enumerate(secs):
        p = text.find(icon)
        if p != -1:
            pos_by_i[i] = (p, icon)
        else:
            tp = low.find(title.lower())
            if tp != -1:
                pos_by_i[i] = (tp, title)
    # Neither icons nor titles anywhere -> put the whole reply in "What I see" (🔍)
    # rather than losing it; the other sections show "insufficient evidence".
    if not pos_by_i:
        return [{"icon": ic, "title": ti,
                 "body": (text.strip() if (j == 1 and text.strip()) else "insufficient evidence")}
                for j, (ic, ti) in enumerate(secs)]
    starts = sorted(p for p, _ in pos_by_i.values())
    sections: List[Dict[str, str]] = []
    for i, (icon, title) in enumerate(secs):
        if i not in pos_by_i:
            sections.append({"icon": icon, "title": title, "body": "insufficient evidence"})
            continue
        start, _marker = pos_by_i[i]
        ends = [p for p in starts if p > start]
        chunk = text[start:(min(ends) if ends else len(text))].strip()
        # Strip a leading "{icon} {title}:" / "{title}:" / "{icon}" header, however it appears.
        if chunk.startswith(icon):
            chunk = chunk[len(icon):].strip()
        if chunk.lower().startswith(title.lower()):
            chunk = chunk[len(title):].strip()
        if chunk.startswith(":"):
            chunk = chunk[1:].strip()
        sections.append({"icon": icon, "title": title, "body": chunk or "insufficient evidence"})
    return sections


def section_icons(doctor_text: str) -> List[str]:
    """Return the six section icons in order (for tests asserting all are present)."""
    return [s["icon"] for s in format_sections(doctor_text)]


def render_doctor_message(resp: Dict[str, Any], msg_id: int = 0) -> html.Div:
    """Build the Dash component for one Doctor message (six sections + buttons).

    Reads: a response dict from risk_doctor.ask. Returns an html.Div: a dark panel
    with the six formatted sections, plus the [✅ APPROVE] / [❌ DISMISS] buttons
    the spec requires under every Doctor response.
    """
    sections = format_sections(resp.get("text", ""))
    body = [html.Div([html.Span(f"{s['icon']} {s['title']}: ",
                                style={"fontWeight": "bold"}),
                      html.Span(s["body"])], style={"marginBottom": "4px"})
            for s in sections]
    buttons = html.Div([
        html.Button("✅ APPROVE — ADD TO SUGGESTED RULES", id={"type": "doctor-approve", "index": msg_id},
                    n_clicks=0, style={"marginRight": "6px"}),
        html.Button("❌ DISMISS", id={"type": "doctor-dismiss", "index": msg_id}, n_clicks=0),
    ], style={"marginTop": "6px"})
    return html.Div(body + [buttons],
                    style={"background": "#1f2430", "color": "#e6e6e6",
                           "padding": "10px", "borderRadius": "8px",
                           "margin": "6px 0", "textAlign": "left"})


def render_user_message(text: str, timestamp: str = "") -> html.Div:
    """Build the Dash component for one of Monty's messages (right-aligned, grey)."""
    return html.Div([html.Div(text), html.Small(timestamp, style={"color": "#888"})],
                    style={"background": "#3a3f4b", "color": "#fff", "padding": "8px",
                           "borderRadius": "8px", "margin": "6px 0 6px 60px",
                           "textAlign": "right"})


def render_offline_banner(message: str) -> html.Div:
    """Build the red error banner shown when the Doctor is offline (RULE 4/6/20)."""
    return html.Div(message, style={"background": config.COLOR_RED, "color": "#fff",
                                    "padding": "10px", "borderRadius": "8px",
                                    "margin": "6px 0", "whiteSpace": "pre-line"})


# ==========================================================================
# THE FLOATING CHAT PANEL (Dash component the dashboard mounts on every screen).
# ==========================================================================
def chat_panel(initial_screen_state: Dict[str, Any]) -> html.Div:
    """Build the bottom-right floating Risk Doctor chat panel.

    Reads: the initial screen state (for the context indicator). Returns the Dash
    component: a collapsible panel with a top bar (label + Full Diagnosis button),
    a scrollable conversation area, an input row, and the context indicator that
    auto-updates as Monty navigates (spec Section E).
    """
    return html.Div(id="doctor-chat-panel", children=[
        # Collapsed tab + expand/collapse toggle.
        html.Button("💬 Risk Doctor", id="doctor-toggle", n_clicks=0,
                    style={"width": "100%", "fontWeight": "bold", "padding": "8px"}),
        html.Div(id="doctor-body", children=[
            # TOP BAR — label + Full Diagnosis trigger.
            html.Div([
                html.Span("💬 Risk Doctor", style={"fontWeight": "bold"}),
                html.Button("📋 Full Diagnosis", id="doctor-full-diagnosis", n_clicks=0,
                            style={"float": "right"}),
            ], style={"padding": "6px", "borderBottom": "1px solid #444"}),
            # CONVERSATION AREA.
            html.Div(id="doctor-conversation", children=[],
                     style={"height": "320px", "overflowY": "auto", "padding": "6px"}),
            # INPUT AREA.
            html.Div([
                dcc.Input(id="doctor-input", type="text",
                          placeholder="Ask the Risk Doctor…  ·  /read <path> to load a file (read-only)",
                          debounce=True, style={"width": "72%"}),
                html.Button("Send", id="doctor-send", n_clicks=0,
                            style={"width": "26%", "marginLeft": "2%"}),
                dcc.Loading(html.Div(id="doctor-spinner"), type="dot"),
            ], style={"padding": "6px"}),
            # CONTEXT INDICATOR — "Doctor is looking at: Screen X — Day Y — Trade Z".
            html.Div(id="doctor-context-indicator",
                     children="Doctor is looking at: " + risk_doctor.context_label(initial_screen_state),
                     style={"fontSize": "11px", "color": "#888", "padding": "0 6px 6px"}),
            # Confirmation line for the [✅ APPROVE] write path.
            html.Div(id="doctor-approve-status",
                     style={"fontSize": "11px", "color": "#1D9E75", "padding": "0 6px 6px"}),
        ], style={"display": "block"}),
    ], style={"position": "fixed", "bottom": "0", "right": "0", "width": "380px",
              "background": "#11151c", "color": "#ddd", "border": "1px solid #444",
              "borderRadius": "8px 8px 0 0", "zIndex": "1000"})


def context_indicator_text(screen_state: Dict[str, Any]) -> str:
    """The 'Doctor is looking at: ...' string for the indicator (spec TEST 15)."""
    return "Doctor is looking at: " + risk_doctor.context_label(screen_state)
