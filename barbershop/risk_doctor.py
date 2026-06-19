# ==========================================================================
# FILE: barbershop/risk_doctor.py
# PURPOSE: The LLM Risk Doctor — Monty's diagnostic partner inside the Barbershop.
#          Reads the Barbershop telemetry + the MLP interpretability manual and
#          answers Monty's questions about why the bot passed or failed, with
#          every claim grounded in a telemetry field. CANNOT change training,
#          rewards, laws, or live execution — it advises, Monty decides.
# ==========================================================================
#
# DEPENDS ON (reads from):
#   logs/trajectory.parquet          — bot action/state/reward log
#   logs/shap_values.parquet         — SHAP attribution values
#   docs/MLP_INTERPRETABILITY_LAYER.md — operating manual (loaded every call, RULE 6)
#   barbershop/config.py             — LLM endpoint + text constants
#   barbershop/data.py               — trajectory tail / pattern finder helpers
#
# PRODUCES (writes to, under logs/ only — RULE 3):
#   logs/doctor_diagnoses.jsonl      — append-only conversation log
#   logs/suggested_rules.json        — prescription export (on approval)
#
# SAFETY (docs/MLP_INTERPRETABILITY_LAYER.md CLAUDE CODE CONTRACT):
#   - No execution authority (RULE 7). A live-trade question is refused.
#   - Manual required (RULE 6). No manual -> the Doctor goes offline, no LLM call.
#   - Context is king (RULE 8). No day selected -> ask Monty to pick one.
#   - Every claim cites a telemetry field; missing evidence -> "insufficient evidence".
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Context-packet assembly, OpenAI-compatible
#                            local LLM call w/ graceful offline, safety rules 6-8,
#                            diagnosis log, fields-cited extraction, prescription
#                            approval/export, full-diagnosis template.
#   [2026-06-15] [Claude] — Adversarial-review anti-fabrication fixes: refuse to
#                            diagnose with empty telemetry (no LLM call); force LOW +
#                            UNVERIFIED when zero fields cited; drop all-NaN
#                            placeholder columns from the packet + warn 'do not cite';
#                            broaden RULE-7 live-trade detection (regex) without
#                            refusing diagnostic questions about past trades.
#   [2026-06-16] [Claude] — WI-5: condense_manual() trims the 32KB manual to a char
#                            budget (DOCTOR_MANUAL_MAX_CHARS) for small local models,
#                            always keeping the safety rules + diagnostic template +
#                            north star; assemble_context_packet uses it for Part 1.
#   [2026-06-19] [C24]    — Multi-provider Risk Doctor + read-only /read. make_client() now
#                            dispatches on config.DOCTOR_PROVIDER (ollama|anthropic|perplexity) and
#                            _completion_text() branches the round-trip (anthropic messages API vs
#                            OpenAI-compat chat.completions); perplexity `sonar` adds live web search.
#                            Added repo_file_loader() + a /read <path> intercept in ask() — READ-ONLY,
#                            repo-root-jailed, no write path. Default stays ollama, so existing tests pass.
# ==========================================================================

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from barbershop import config, data


# ==========================================================================
# OPERATING MANUAL (RULE 6) — must be loaded and sent on EVERY message.
# ==========================================================================
class ManualMissing(Exception):
    """Raised when MLP_INTERPRETABILITY_LAYER.md cannot be found (RULE 6)."""


def load_manual() -> str:
    """Load the MLP interpretability manual text. Raises ManualMissing if absent.

    Reads: docs/MLP_INTERPRETABILITY_LAYER.md (real location, with a root fallback).
    Returns: the full file text — the Doctor's grounding for every definition.
    """
    path = config.mlp_manual_path()
    if not path.exists():
        raise ManualMissing(str(path))
    return path.read_text(encoding="utf-8")


# Sections the condensed manual MUST keep (the Doctor's safety + diagnosis backbone),
# then these high-signal sections fill the remaining budget.
_MANUAL_MUST_KEEP = ("SAFETY RULES", "DIAGNOSTIC OUTPUT TEMPLATE", "NORTH STAR")
_MANUAL_PRIORITY = ("FAILURE TAXONOMY", "FLOW RULES", "CORE DEFINITIONS", "DATA CONTRACT",
                    "TERM ", "WHAT THIS LAYER PROTECTS")


def condense_manual(text: str, max_chars: int) -> str:
    """Trim the manual to ~max_chars while keeping the high-signal sections (RULE 6-safe).

    Reads: the full manual text. Returns it unchanged when already under budget; else a
    condensed version that ALWAYS keeps the safety rules + diagnostic template + north
    star, then fills the remaining budget with the next most useful sections (failure
    taxonomy, flow rules, core definitions, ...), in original order. This keeps the
    Doctor grounded against a small local model's context window without dropping the
    safety backbone.
    """
    if len(text) <= max_chars:
        return text
    # Split into sections at markdown headers (lines starting with '#').
    sections: List[Tuple[str, str]] = []
    header, body = "PREAMBLE", []
    for ln in text.splitlines():
        if ln.startswith("#"):
            sections.append((header, "\n".join(body)))
            header, body = ln, [ln]
        else:
            body.append(ln)
    sections.append((header, "\n".join(body)))

    note = "[Manual condensed for the local model's context — high-signal sections kept.]\n"
    keep_idx, total = set(), len(note)
    # Pass 1: force-keep the must-have sections (safety backbone), regardless of budget.
    for i, (h, b) in enumerate(sections):
        if any(k in h.upper() for k in _MANUAL_MUST_KEEP):
            keep_idx.add(i); total += len(b)
    # Pass 2: add priority sections in original order until the budget is reached.
    for i, (h, b) in enumerate(sections):
        if i in keep_idx:
            continue
        if any(k in h.upper() for k in _MANUAL_PRIORITY) and total + len(b) <= max_chars:
            keep_idx.add(i); total += len(b)
    kept = [sections[i][1] for i in range(len(sections)) if i in keep_idx]
    return note + "\n".join(kept)


# The doc's exact 7-section diagnostic template (Full Diagnosis button, TEST 16).
DIAGNOSTIC_TEMPLATE = (
    "DIAGNOSTIC OUTPUT TEMPLATE — fill every section with evidence or write "
    "'insufficient evidence':\n"
    "  What happened (outcome layer):\n"
    "  Where the chain broke:\n"
    "  Failure classification:\n"
    "  Evidence cited:\n"
    "  Confidence:\n"
    "  Prescription:\n"
    "  Not recommended:\n"
)


# ==========================================================================
# CONTEXT PACKET (spec Section D) — assembled fresh for every message.
# ==========================================================================
def assemble_context_packet(question: str, screen_state: Dict[str, Any],
                            history: List[Tuple[str, str]],
                            trajectory: Optional[pd.DataFrame] = None,
                            shap: Optional[pd.DataFrame] = None,
                            manual_text: Optional[str] = None,
                            full_diagnosis: bool = False) -> Dict[str, str]:
    """Assemble the system prompt + user message sent to the LLM (spec Section D).

    Reads: the manual, the current screen state, the conversation history, and the
    telemetry frames. Returns {'system': <system prompt>, 'user': <question>}.
    The system prompt contains, in order:
      Part 1 — the full MLP_INTERPRETABILITY_LAYER.md (loaded if not supplied),
      Part 2 — the current screen-state snapshot (screen/day/trade + last 10
               trajectory rows for the day + SHAP for the selected trade +
               pattern-finder output on Screen 5),
      Part 3 — the conversation history (capped at the last DOCTOR_HISTORY_LIMIT),
      Part 4 — the 4-line hard-rules reminder (always included).
    """
    manual = manual_text if manual_text is not None else load_manual()
    manual = condense_manual(manual, config.DOCTOR_MANUAL_MAX_CHARS)   # fit the LLM's context

    parts: List[str] = []
    parts.append("=== PART 1: OPERATING MANUAL (MLP_INTERPRETABILITY_LAYER.md) ===\n" + manual)
    if full_diagnosis:
        parts.append("=== FULL DIAGNOSIS REQUESTED ===\n" + DIAGNOSTIC_TEMPLATE)
    parts.append("=== PART 2: CURRENT SCREEN STATE ===\n"
                 + _screen_snapshot(screen_state, trajectory, shap))
    parts.append("=== PART 3: CONVERSATION HISTORY (last "
                 + f"{config.DOCTOR_HISTORY_LIMIT}) ===\n"
                 + _format_history(history))
    parts.append("=== PART 4: HARD RULES ===\n" + config.DOCTOR_HARD_RULES)
    parts.append(
        "Respond in EXACTLY these six sections, each on its own line, plain English:\n"
        + "\n".join(f"{icon} {title}:" for icon, title in config.DOCTOR_SECTIONS))
    return {"system": "\n\n".join(parts), "user": question}


def _screen_snapshot(screen_state: Dict[str, Any], trajectory: Optional[pd.DataFrame],
                     shap: Optional[pd.DataFrame]) -> str:
    """Render the Part-2 screen snapshot (which screen, day, trade + telemetry tails)."""
    screen = screen_state.get("screen")
    day_id = screen_state.get("day_id")
    trade_id = screen_state.get("trade_id")
    lines = [f"Screen: {screen}", f"Day: {day_id}", f"Trade: {trade_id}"]
    # Last 10 trajectory rows for the selected day (the recent decision context).
    if trajectory is not None and day_id is not None:
        g = trajectory[trajectory["day_id"] == day_id].sort_values("timestamp")
        tail = g.tail(config.DOCTOR_TRAJECTORY_TAIL)
        cols = ["step", "action", "action_prob", "advantage", "dd_buffer",
                "pnl_cumulative", "reward"]
        present = [c for c in cols if c in tail.columns]
        # ANTI-FABRICATION — drop placeholder columns that are ALL-NaN on this run
        # (the live pipeline doesn't produce them yet) and tell the Doctor in plain
        # text NOT to cite them, so a NaN can't be passed off as evidence.
        unavailable = [c for c in present
                       if c in config.PLACEHOLDER_FIELDS and tail[c].isna().all()]
        present = [c for c in present if c not in unavailable]
        lines.append("Last trajectory rows for this day:")
        lines.append(tail[present].to_string(index=False))
        if unavailable:
            lines.append("UNAVAILABLE / placeholder fields (do NOT cite as evidence): "
                         + ", ".join(unavailable))
    # SHAP for the selected trade: map the sequential trade_id -> its entry step,
    # then look up the SHAP row by (day, step).
    if (shap is not None and trade_id is not None and trajectory is not None
            and day_id is not None and len(shap)):
        trades = data.extract_trades(trajectory, day_id)
        if 1 <= trade_id <= len(trades):
            entry = trades[trade_id - 1]["entry_time"]
            erow = trajectory[(trajectory["day_id"] == day_id)
                              & (trajectory["timestamp"] == entry)]
            if len(erow):
                step = int(erow.iloc[0]["step"])
                srow = shap[(shap["day_id"] == day_id) & (shap["step"] == step)]
                if len(srow):
                    lines.append("SHAP for selected trade:")
                    lines.append(json.dumps({"toward": srow.iloc[0].get("shap_toward", {}),
                                             "away": srow.iloc[0].get("shap_away", {})}))
    # Pattern Finder output on Screen 5.
    if screen == 5 and trajectory is not None:
        pats = data.find_patterns(data.losing_trades(trajectory))
        lines.append("Pattern Finder output: "
                     + json.dumps([{"rank": p["rank"], "count": p["count"],
                                    "total": p["total"], "rule": p["suggested_rule"]}
                                   for p in pats]))
    return "\n".join(lines)


def _format_history(history: List[Tuple[str, str]]) -> str:
    """Render the last DOCTOR_HISTORY_LIMIT (question, answer) exchanges."""
    recent = history[-config.DOCTOR_HISTORY_LIMIT:] if history else []
    if not recent:
        return "(no prior exchanges)"
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in recent)


def context_label(screen_state: Dict[str, Any]) -> str:
    """Return the "Doctor is looking at: ..." indicator text (spec TEST 15).

    Reads: the screen state. Returns a one-line label naming the screen, day, and
    trade (or "No trade selected") so the Doctor always knows Monty's focus.
    """
    screen = screen_state.get("screen")
    day_id = screen_state.get("day_id")
    trade_id = screen_state.get("trade_id")
    trade_txt = f"Trade {trade_id}" if trade_id is not None else "No trade selected"
    day_txt = f"Day {day_id}" if day_id is not None else "No day selected"
    return f"Screen {screen} — {day_txt} — {trade_txt}"


# ==========================================================================
# SAFETY GUARDS (RULES 7 + 8).
# ==========================================================================
# Present-tense "act NOW" forms the literal trigger list can't enumerate. Tuned to
# catch live-decision asks ("is now a good time to buy?", "should i be long here?")
# WITHOUT refusing diagnostic questions about PAST trades ("why did it go long?").
_LIVE_TRADE_RE = re.compile(
    r"should i (go )?(long|short|buy|sell)\b"
    r"|should i be (long|short)\b"
    r"|\b(buy|sell|enter|long|short|entry)\b[^.?!]{0,25}\b(now|right now|today)\b"
    r"|\bgood (time|entry)\b[^.?!]{0,25}\b(buy|sell|now|right now|today)\b"
    r"|\benter (a )?(long|short)\b"
    r"|close my (live )?position\b",
    re.IGNORECASE,
)


def is_live_trade_question(question: str) -> bool:
    """True if Monty's question asks for a LIVE trade decision (RULE 7 trigger).

    Reads: the question. Returns True for specific live-decision phrases (the
    config trigger list) OR a present-tense "[action] now/here" regex match, so the
    Doctor refuses to act as a signal source while still answering diagnostic
    questions about what the bot DID in training.
    """
    q = question.lower()
    if any(trigger in q for trigger in config.LIVE_TRADE_TRIGGER_WORDS):
        return True
    return bool(_LIVE_TRADE_RE.search(q))


def has_context(screen_state: Dict[str, Any]) -> bool:
    """True if a day (or trade) is selected — RULE 8 needs a focus to answer."""
    return screen_state.get("day_id") is not None or screen_state.get("trade_id") is not None


# ==========================================================================
# THE LLM CALL.
# ==========================================================================
def make_client():
    """Build the LLM client for the configured provider (C24). SDKs imported lazily so the dashboard +
    tests import this module with no server/keys present.
    COUPLING -> config.DOCTOR_PROVIDER + per-provider keys/bases/models:
      ollama (default) / perplexity -> openai.OpenAI (OpenAI-compatible; perplexity adds web search),
      anthropic                     -> anthropic.Anthropic (the messages API)."""
    provider = getattr(config, "DOCTOR_PROVIDER", "ollama")
    if provider == "anthropic":
        from anthropic import Anthropic                    # lazy
        return Anthropic(api_key=config.DOCTOR_ANTHROPIC_KEY or None)
    from openai import OpenAI                              # lazy: ollama + perplexity are OpenAI-compatible
    if provider == "perplexity":
        return OpenAI(base_url=config.DOCTOR_PERPLEXITY_BASE, api_key=config.DOCTOR_PERPLEXITY_KEY)
    return OpenAI(base_url=config.DOCTOR_API_BASE, api_key=config.DOCTOR_API_KEY)


def _completion_text(cli: Any, packet: Dict[str, str]) -> str:
    """One LLM round-trip -> text, dispatched by provider (C24). Anthropic uses messages.create with a
    top-level `system`; OpenAI-compatible providers (ollama/perplexity) use chat.completions.create.
    COUPLING -> assemble_context_packet() packet keys 'system'/'user'."""
    if getattr(config, "DOCTOR_PROVIDER", "ollama") == "anthropic":
        msg = cli.messages.create(
            model=config.DOCTOR_ANTHROPIC_MODEL, max_tokens=config.DOCTOR_MAX_TOKENS,
            system=packet["system"], messages=[{"role": "user", "content": packet["user"]}])
        return msg.content[0].text
    model = (config.DOCTOR_PERPLEXITY_MODEL if getattr(config, "DOCTOR_PROVIDER", "ollama") == "perplexity"
             else config.DOCTOR_MODEL)
    completion = cli.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": packet["system"]},
                  {"role": "user", "content": packet["user"]}],
        max_tokens=config.DOCTOR_MAX_TOKENS, temperature=config.DOCTOR_TEMPERATURE)
    return completion.choices[0].message.content


def repo_file_loader(rel_path: str, max_chars: Optional[int] = None) -> str:
    """READ-ONLY: return a repo file's text for the Doctor's context (the `/read` command). Accepts a
    REPO-RELATIVE path only and refuses anything that resolves outside the repo root — there is NO write
    path here, ever (RULE: the Doctor has no execution/write authority). COUPLING -> config.REPO_ROOT."""
    max_chars = max_chars or getattr(config, "DOCTOR_REPO_READ_MAX_CHARS", 16000)
    root = Path(config.REPO_ROOT).resolve()
    try:
        target = (root / rel_path).resolve()
        target.relative_to(root)                          # raises ValueError if it escapes the repo
    except (ValueError, OSError):
        return f"[/read denied] '{rel_path}' is outside the repo or invalid (read-only, repo paths only)."
    if not target.is_file():
        return f"[/read] file not found: {rel_path}"
    text = target.read_text(encoding="utf-8", errors="replace")
    extra = "" if len(text) <= max_chars else f"\n… [truncated {len(text) - max_chars} chars]"
    return text[:max_chars] + extra


def ask(question: str, screen_state: Dict[str, Any],
        history: Optional[List[Tuple[str, str]]] = None,
        trajectory: Optional[pd.DataFrame] = None,
        shap: Optional[pd.DataFrame] = None,
        client: Any = None, full_diagnosis: bool = False,
        log: bool = True) -> Dict[str, Any]:
    """Answer one of Monty's questions, grounded in the current screen state.

    Reads: telemetry frames + the manual. Optionally calls the LLM (a client may
    be injected for tests). Returns a response dict:
      {text, confidence, fields_cited, offline, refused, manual_missing}
    Applies the safety rules in order: manual present (RULE 6) -> not a live-trade
    question (RULE 7) -> context selected (RULE 8) -> LLM call (graceful offline).
    Every real exchange is appended to logs/doctor_diagnoses.jsonl.
    """
    history = history or []

    # `/read <path>` (C24) — load a repo file into the chat, READ-ONLY. Bypasses the LLM + the context
    # rules (it is a plain file view, not a diagnosis) but never writes. COUPLING -> repo_file_loader.
    if question.strip().startswith("/read "):
        rel = question.strip()[len("/read "):].strip()
        resp = {"text": f"📄 {rel}\n\n{repo_file_loader(rel)}", "confidence": "N/A",
                "fields_cited": [], "offline": False, "refused": False,
                "manual_missing": False, "read_only": True}
        if log:
            _log_exchange(question, {**resp, "text": f"[/read] {rel}"}, screen_state)
        return resp

    # RULE 6 — manual must load, or the Doctor is offline (no LLM call).
    try:
        manual_text = load_manual()
    except ManualMissing:
        return {"text": config.DOCTOR_MANUAL_MISSING, "confidence": "LOW",
                "fields_cited": [], "offline": True, "refused": False,
                "manual_missing": True}

    # RULE 7 — never give a live trade decision.
    if is_live_trade_question(question):
        resp = {"text": config.DOCTOR_REFUSAL_LIVE, "confidence": "HIGH",
                "fields_cited": [], "offline": False, "refused": True,
                "manual_missing": False}
        if log:
            _log_exchange(question, resp, screen_state)
        return resp

    # RULE 8 — need a selected day/trade to anchor the answer.
    if not has_context(screen_state):
        return {"text": config.DOCTOR_NO_CONTEXT, "confidence": "LOW",
                "fields_cited": [], "offline": False, "refused": False,
                "manual_missing": False}

    # ANTI-FABRICATION — a selected day with NO telemetry behind it cannot be
    # diagnosed. Return "insufficient evidence" WITHOUT calling the LLM, so the
    # Doctor never invents a diagnosis when real data is missing/unloaded.
    day_id = screen_state.get("day_id")
    no_data = trajectory is None or len(trajectory) == 0
    if not no_data and day_id is not None:
        no_data = trajectory[trajectory["day_id"] == day_id].empty
    if no_data:
        resp = {"text": config.DOCTOR_NO_EVIDENCE, "confidence": "LOW",
                "fields_cited": [], "offline": False, "refused": False,
                "manual_missing": False, "insufficient_evidence": True}
        if log:
            _log_exchange(question, resp, screen_state)
        return resp

    packet = assemble_context_packet(question, screen_state, history, trajectory,
                                     shap, manual_text=manual_text,
                                     full_diagnosis=full_diagnosis)
    try:
        cli = client or make_client()
        text = _completion_text(cli, packet)              # C24: provider-dispatched (ollama/anthropic/perplexity)
    except Exception:                                     # any connection / server error
        # Graceful offline (spec TEST 20): save the question, tell Monty why.
        offline_resp = {"text": config.DOCTOR_OFFLINE_MESSAGE.format(api_base=config.DOCTOR_API_BASE),
                        "confidence": "LOW", "fields_cited": [], "offline": True,
                        "refused": False, "manual_missing": False}
        if log:
            saved = dict(offline_resp)
            saved["text"] = config.DOCTOR_OFFLINE_RESPONSE      # logged response == "OFFLINE"
            _log_exchange(question, saved, screen_state)
        return offline_resp

    fields_cited = extract_fields_cited(text)
    confidence = extract_confidence(text)
    # ANTI-FABRICATION — an answer that cites ZERO telemetry fields is a hunch, not
    # evidence (RULE 1). Force LOW confidence + a visible warning unless the model
    # already said "insufficient evidence" itself.
    if not fields_cited and "insufficient evidence" not in (text or "").lower():
        text = config.DOCTOR_UNVERIFIED_PREFIX + (text or "")
        confidence = "LOW"
    resp = {"text": text, "confidence": confidence, "fields_cited": fields_cited,
            "offline": False, "refused": False, "manual_missing": False}
    if log:
        _log_exchange(question, resp, screen_state)
    return resp


# ==========================================================================
# RESPONSE PARSING — confidence + evidence (fields cited).
# ==========================================================================
def extract_confidence(text: str) -> str:
    """Extract the stated confidence (HIGH/MEDIUM/LOW) from a Doctor response."""
    up = (text or "").upper()
    # Prefer an explicit "Confidence: X"; else fall back to the first level word.
    for level in ("HIGH", "MEDIUM", "LOW"):
        if f"CONFIDENCE: {level}" in up or f"CONFIDENCE:{level}" in up:
            return level
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in up:
            return level
    return "LOW"


# The vocabulary the Doctor is expected to cite — telemetry columns, metrics, and
# law names. extract_fields_cited scans the response for any of these.
def _known_fields() -> List[str]:
    """The telemetry fields + law names the Doctor may legitimately cite."""
    fields = list(data.required_trajectory_columns())
    fields += ["value", "trailing_dd", "trailing_buffer", "hidden_summary",
               "pre_mask_logits", "post_mask_logits", "action_probs"]
    fields += list(data.MOCK_LAW_NAMES)
    fields += [f"law_state.{n}" for n in data.MOCK_LAW_NAMES]
    return fields


def extract_fields_cited(text: str) -> List[str]:
    """Return the known telemetry fields/laws mentioned in the response (evidence proof).

    Reads: the Doctor's response text. Returns the subset of _known_fields present,
    so the diagnosis log can confirm the answer was grounded in evidence (RULE 1).
    """
    low = (text or "").lower()
    cited: List[str] = []
    for f in _known_fields():
        if f.lower() in low and f not in cited:
            cited.append(f)
    return cited


# ==========================================================================
# DIAGNOSIS LOG (spec Section D) — one JSON object per exchange.
# ==========================================================================
def _log_exchange(question: str, resp: Dict[str, Any],
                  screen_state: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Append one exchange to logs/doctor_diagnoses.jsonl (newline-delimited JSON)."""
    path = Path(path or config.DOCTOR_DIAGNOSES_JSONL)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "screen": screen_state.get("screen"),
        "day_id": screen_state.get("day_id"),
        "trade_id": screen_state.get("trade_id"),
        "monty_question": question,
        "doctor_response": resp.get("text", ""),
        "confidence": resp.get("confidence", "LOW"),
        "fields_cited": resp.get("fields_cited", []),
        "prescription_exported": False,
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return path


def read_diagnoses(path: Optional[Path] = None) -> List[dict]:
    """Read all logged diagnosis entries (one JSON object per line)."""
    path = Path(path or config.DOCTOR_DIAGNOSES_JSONL)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ==========================================================================
# APPROVE PRESCRIPTION (spec Section D) — export "what to do next" to the rules
# file and mark the matching diagnosis-log entry as exported.
# ==========================================================================
def approve_prescription(question: str, doctor_text: str,
                         screen_state: Dict[str, Any]) -> Path:
    """Export the Doctor's prescription to suggested_rules.json + flag the log entry.

    Reads: the Doctor's response. Writes: appends the prescription (the "What to
    do next" / "Prescription" section) to logs/suggested_rules.json (RULE 3 — only
    logs/), and updates the most recent matching diagnosis-log line with
    prescription_exported=true. Returns the suggested_rules.json path.
    """
    prescription = _extract_prescription(doctor_text)
    path = data.export_rule({
        "source": "risk_doctor",
        "screen": screen_state.get("screen"),
        "day_id": screen_state.get("day_id"),
        "trade_id": screen_state.get("trade_id"),
        "question": question,
        "prescription": prescription,
    })
    _mark_exported(question)
    return path


def _extract_prescription(text: str) -> str:
    """Pull the actionable recommendation out of a Doctor response, robustly.

    Reads: the response text. Tries, in order: a line containing 'what to do next' /
    'prescription', then the ✅ section marker (the recommendation section), and falls
    back to a trimmed summary — so the export still works when a local model paraphrases
    the header or drops the colon.
    """
    lines = (text or "").splitlines()
    markers = ("what to do next", "prescription", "✅")    # header phrasings or the icon
    for i, line in enumerate(lines):
        low = line.lower()
        if any(m in (low if m != "✅" else line) for m in markers):
            # Take the rest of this line after the colon, else the next non-empty line.
            after = line.split(":", 1)[1].strip() if ":" in line else line.replace("✅", "").strip()
            # Drop a residual header phrase left on the same line.
            for h in ("what to do next", "prescription"):
                if after.lower().startswith(h):
                    after = after[len(h):].lstrip(": ").strip()
            if after:
                return after
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return nxt.strip()
    return (text or "").strip()[:300]                     # fallback: a trimmed summary


def _mark_exported(question: str, path: Optional[Path] = None) -> None:
    """Set prescription_exported=true on the latest diagnosis-log line for `question`."""
    path = Path(path or config.DOCTOR_DIAGNOSES_JSONL)
    if not path.exists():
        return
    entries = read_diagnoses(path)
    for entry in reversed(entries):                       # latest matching first
        if entry.get("monty_question") == question:
            entry["prescription_exported"] = True
            break
    with open(path, "w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
