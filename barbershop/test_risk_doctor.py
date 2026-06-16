# ==========================================================================
# FILE: barbershop/test_risk_doctor.py
# PURPOSE: Tests 11-20 for the LLM Risk Doctor (continuation spec Section G).
#          All tests MOCK the LLM client (no live API call). They verify the
#          context packet, response format, safety rules (manual required,
#          no execution authority), the diagnosis log, prescription export,
#          history cap, and graceful offline behaviour.
#          Run: pytest barbershop/test_risk_doctor.py
# ==========================================================================
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Tests 11-20: context packet, response
#                            format, missing manual, approve prescription,
#                            context indicator, full diagnosis, no-execution,
#                            history cap, diagnosis log, offline failure.
#   [2026-06-15] [Claude] — Review fixes: de-vacuum TESTs 12/14/16/20 (body
#                            routing, prescription content, all 7 template headers,
#                            drop tautology); added TESTs 21-24 (no-context guard,
#                            empty-telemetry anti-fabrication, zero-evidence
#                            downgrade, RULE-7 paraphrase coverage).
# ==========================================================================

from __future__ import annotations

import json

import pandas as pd
import pytest

from barbershop import config, data, doctor_chat, risk_doctor


# --------------------------------------------------------------------------
# Helpers: a mock OpenAI-compatible client (success + offline variants).
# --------------------------------------------------------------------------
class _MockMessage:
    """A stand-in for one OpenAI choice (exposes .message.content)."""

    def __init__(self, content):
        """Wrap `content` as the .message.content the Doctor reads."""
        self.message = type("M", (), {"content": content})


class _MockCompletions:
    """A stand-in for client.chat.completions: returns canned text or raises."""

    def __init__(self, content, raise_exc=None):
        """Store the canned response text (or an exception to raise) + a call counter."""
        self._content = content
        self._raise = raise_exc
        self.calls = 0

    def create(self, **kwargs):
        """Count the call and return the canned completion (or raise the offline error)."""
        self.calls += 1
        if self._raise:
            raise self._raise
        return type("R", (), {"choices": [_MockMessage(self._content)]})


class MockLLM:
    """Minimal stand-in for openai.OpenAI exposing .chat.completions.create()."""

    def __init__(self, content="ok", raise_exc=None):
        """Build the mock client with a canned response (or an exception to raise)."""
        self._completions = _MockCompletions(content, raise_exc)
        self.chat = type("C", (), {"completions": self._completions})

    @property
    def calls(self):
        """How many times the mock LLM was called (0 == never reached the LLM)."""
        return self._completions.calls


# A correctly-formatted 6-section Doctor response (spec Section D format).
GOOD_RESPONSE = (
    "📍 What I'm looking at: Screen 3, Day 2, the advantage field.\n"
    "🔍 What I see: advantage went negative and dd_buffer fell below 0.25.\n"
    "🎯 What it means for passing: the bot drifted toward the 4% wall.\n"
    "✅ What to do next: increase the OPEN penalty when dd_buffer < 0.25.\n"
    "❌ What NOT to do: do not touch the laws or live execution.\n"
    "📊 Confidence: MEDIUM"
)

SCREEN3 = {"screen": 3, "day_id": 2, "trade_id": 3}


# --------------------------------------------------------------------------
# TEST 11 — Context packet assembly (all four parts present).
# --------------------------------------------------------------------------
def test_context_packet_assembly(mock_trajectory, mock_shap):
    """TEST 11 — Context packet assembly (all four parts present)."""
    packet = risk_doctor.assemble_context_packet(
        "Why did the bot fail Day 2?", SCREEN3, history=[],
        trajectory=mock_trajectory, shap=mock_shap,
        manual_text="UNIQUE_MANUAL_MARKER_TEXT")
    sysp = packet["system"]
    assert "UNIQUE_MANUAL_MARKER_TEXT" in sysp                  # Part 1: manual
    assert "Last trajectory rows for this day" in sysp          # Part 2: day-2 tail
    assert "SHAP for selected trade" in sysp                    # Part 2: SHAP for the trade
    assert "HARD RULES" in sysp and "FTMO passing" in sysp      # Part 4: the 4 hard rules
    assert packet["user"] == "Why did the bot fail Day 2?"      # user message verbatim


# --------------------------------------------------------------------------
# TEST 12 — Response format validation: all six sections render with icons.
# --------------------------------------------------------------------------
def test_response_renders_six_sections():
    """TEST 12 — Response format validation: all six sections render with icons."""
    sections = doctor_chat.format_sections(GOOD_RESPONSE)
    assert len(sections) == 6
    icons = [s["icon"] for s in sections]
    assert icons == [icon for icon, _ in config.DOCTOR_SECTIONS]   # 📍🔍🎯✅❌📊 in order
    # Body text must be routed to the CORRECT section (not just 6 headers present).
    assert "advantage went negative" in sections[1]["body"]    # 🔍 What I see
    assert sections[5]["body"].strip().endswith("MEDIUM")      # 📊 Confidence
    # A response MISSING a section gets the 'insufficient evidence' fallback there.
    partial = "📍 What I'm looking at: Screen 3.\n📊 Confidence: LOW"
    psec = doctor_chat.format_sections(partial)
    assert psec[2]["body"] == "insufficient evidence"          # 🎯 absent -> fallback
    component = doctor_chat.render_doctor_message({"text": GOOD_RESPONSE})
    flat = str(component)
    for icon, _ in config.DOCTOR_SECTIONS:
        assert icon in flat                                     # every icon shown in the UI


# --------------------------------------------------------------------------
# TEST 13 — Missing operating manual: Doctor offline, LLM not called (RULE 6).
# --------------------------------------------------------------------------
def test_missing_manual_takes_doctor_offline(barbershop_tmp, monkeypatch):
    """TEST 13 — Missing operating manual: Doctor offline, LLM not called (RULE 6)."""
    missing = barbershop_tmp / "docs" / "NOPE.md"               # does not exist
    monkeypatch.setattr(config, "mlp_manual_path", lambda: missing)
    client = MockLLM(GOOD_RESPONSE)
    resp = risk_doctor.ask("Why did Day 2 fail?", SCREEN3, client=client)
    assert resp["manual_missing"] is True
    assert resp["text"] == config.DOCTOR_MANUAL_MISSING
    assert client.calls == 0                                    # LLM call NOT attempted


# --------------------------------------------------------------------------
# TEST 14 — Approve prescription -> suggested_rules.json + log flagged exported.
# --------------------------------------------------------------------------
def test_approve_prescription_exports_and_flags(barbershop_tmp, mock_trajectory, mock_shap):
    """TEST 14 — Approve prescription -> suggested_rules.json + log flagged exported."""
    client = MockLLM(GOOD_RESPONSE)
    q = "Why did Day 2 fail?"
    resp = risk_doctor.ask(q, SCREEN3, trajectory=mock_trajectory, shap=mock_shap, client=client)
    out = risk_doctor.approve_prescription(q, resp["text"], SCREEN3)
    assert out.exists()                                        # suggested_rules.json written
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved[-1]["source"] == "risk_doctor"
    # The exported prescription is the ACTUAL 'What to do next' text, not the whole blob.
    assert "increase the OPEN penalty" in saved[-1]["prescription"]
    # The matching diagnosis-log entry is now flagged exported.
    dinodes = risk_doctor.read_diagnoses()
    assert any(d["monty_question"] == q and d["prescription_exported"] for d in dinodes)


# --------------------------------------------------------------------------
# TEST 15 — Context indicator updates on navigation.
# --------------------------------------------------------------------------
def test_context_indicator_updates():
    """TEST 15 — Context indicator updates on navigation."""
    s2 = {"screen": 2, "day_id": 2, "trade_id": None}
    s3 = {"screen": 3, "day_id": 2, "trade_id": 3}
    assert risk_doctor.context_label(s2) == "Screen 2 — Day 2 — No trade selected"
    assert risk_doctor.context_label(s3) == "Screen 3 — Day 2 — Trade 3"
    # The chat indicator prefixes "Doctor is looking at: ".
    assert doctor_chat.context_indicator_text(s3).endswith("Screen 3 — Day 2 — Trade 3")


# --------------------------------------------------------------------------
# TEST 16 — Full diagnosis trigger: system prompt carries the doc template.
# --------------------------------------------------------------------------
def test_full_diagnosis_includes_template(mock_trajectory):
    """TEST 16 — Full diagnosis trigger: system prompt carries the doc template."""
    packet = risk_doctor.assemble_context_packet(
        "full diagnosis", {"screen": 3, "day_id": 2, "trade_id": None}, history=[],
        trajectory=mock_trajectory, manual_text="MANUAL", full_diagnosis=True)
    for header in ("What happened", "Where the chain broke", "Failure classification",
                   "Evidence cited", "Confidence", "Prescription", "Not recommended"):
        assert header in packet["system"]                      # all 7 template sections
    client = MockLLM(GOOD_RESPONSE)
    resp = risk_doctor.ask("full diagnosis", {"screen": 3, "day_id": 2, "trade_id": None},
                           trajectory=mock_trajectory, client=client, full_diagnosis=True)
    assert client.calls == 1 and not resp["offline"]           # a response is produced


# --------------------------------------------------------------------------
# TEST 17 — No execution authority (RULE 7): refuse a live-trade question.
# --------------------------------------------------------------------------
def test_no_execution_authority(barbershop_tmp):
    """TEST 17 — No execution authority (RULE 7): refuse a live-trade question."""
    client = MockLLM(GOOD_RESPONSE)
    resp = risk_doctor.ask("Should I go long on EURUSD now?", SCREEN3, client=client)
    assert resp["refused"] is True
    assert resp["text"] == config.DOCTOR_REFUSAL_LIVE
    assert config.DOCTOR_REFUSAL_LIVE == "I only diagnose training runs, not live positions."
    assert client.calls == 0                                    # no LLM call on a trade ask


# --------------------------------------------------------------------------
# TEST 18 — Conversation history capped at the last 6 exchanges.
# --------------------------------------------------------------------------
def test_history_capped_at_six(mock_trajectory):
    """TEST 18 — Conversation history capped at the last 6 exchanges."""
    history = [(f"QUESTION_{i}", f"ANSWER_{i}") for i in range(10)]
    packet = risk_doctor.assemble_context_packet(
        "new q", SCREEN3, history=history, trajectory=mock_trajectory, manual_text="M")
    sysp = packet["system"]
    # The last 6 exchanges (4..9) are included; earlier ones (0..3) are dropped.
    for i in range(4, 10):
        assert f"QUESTION_{i}" in sysp
    for i in range(0, 4):
        assert f"QUESTION_{i}" not in sysp


# --------------------------------------------------------------------------
# TEST 19 — Diagnosis log append: 3 questions -> 3 valid JSON objects.
# --------------------------------------------------------------------------
def test_diagnosis_log_appends(barbershop_tmp, mock_trajectory, mock_shap):
    """TEST 19 — Diagnosis log append: 3 questions -> 3 valid JSON objects."""
    client = MockLLM(GOOD_RESPONSE)
    for q in ("q one", "q two", "q three"):
        risk_doctor.ask(q, SCREEN3, trajectory=mock_trajectory, shap=mock_shap, client=client)
    lines = config.DOCTOR_DIAGNOSES_JSONL.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3                                      # exactly three exchanges
    for line in lines:
        obj = json.loads(line)
        for key in ("timestamp", "screen", "monty_question", "confidence"):
            assert key in obj


# --------------------------------------------------------------------------
# TEST 20 — Offline LLM graceful failure: message shown, question saved as OFFLINE.
# --------------------------------------------------------------------------
def test_offline_llm_graceful(barbershop_tmp, mock_trajectory):
    """TEST 20 — Offline LLM graceful failure: message shown, question saved as OFFLINE."""
    client = MockLLM(raise_exc=ConnectionError("server down"))
    resp = risk_doctor.ask("Why did Day 2 fail?", SCREEN3, trajectory=mock_trajectory, client=client)
    assert resp["offline"] is True
    assert config.DOCTOR_API_BASE in resp["text"]              # endpoint named in the message
    assert "temporarily offline" in resp["text"]
    # The question is saved with doctor_response == "OFFLINE".
    saved = risk_doctor.read_diagnoses()
    assert saved[-1]["doctor_response"] == config.DOCTOR_OFFLINE_RESPONSE


# --------------------------------------------------------------------------
# TEST 21 — RULE 8: no day/trade selected -> ask Monty to pick one, no LLM call.
# --------------------------------------------------------------------------
def test_no_context_guard(barbershop_tmp):
    """TEST 21 — RULE 8: no day/trade selected -> ask Monty to pick one, no LLM call."""
    client = MockLLM(GOOD_RESPONSE)
    resp = risk_doctor.ask("why?", {"screen": 1, "day_id": None, "trade_id": None}, client=client)
    assert resp["text"] == config.DOCTOR_NO_CONTEXT
    assert client.calls == 0


# --------------------------------------------------------------------------
# TEST 22 — anti-fabrication: a selected day with NO telemetry -> insufficient
# evidence, the LLM is NOT called (the Doctor can't diagnose what it can't see).
# --------------------------------------------------------------------------
def test_empty_telemetry_refuses_to_fabricate(barbershop_tmp):
    """TEST 22 — anti-fabrication: a selected day with NO telemetry -> insufficient"""
    client = MockLLM(GOOD_RESPONSE)
    resp = risk_doctor.ask("Why did Day 2 fail?", SCREEN3, trajectory=None, client=client)
    assert resp.get("insufficient_evidence") is True
    assert "insufficient evidence" in resp["text"].lower()
    assert client.calls == 0                                   # no diagnosis without data


# --------------------------------------------------------------------------
# TEST 23 — anti-fabrication: an answer citing ZERO telemetry fields is forced
# to LOW confidence and flagged UNVERIFIED.
# --------------------------------------------------------------------------
def test_zero_evidence_answer_is_downgraded(barbershop_tmp, mock_trajectory):
    """TEST 23 — anti-fabrication: an answer citing ZERO telemetry fields is forced"""
    client = MockLLM("The model looked fine to me. Confidence: HIGH")   # cites no field
    resp = risk_doctor.ask("Why did Day 2 fail?", SCREEN3, trajectory=mock_trajectory, client=client)
    assert resp["fields_cited"] == []
    assert resp["confidence"] == "LOW"                         # HIGH was overridden
    assert "UNVERIFIED" in resp["text"]


# --------------------------------------------------------------------------
# TEST 24 — RULE 7 holds for PARAPHRASES, but diagnostic questions are answered.
# --------------------------------------------------------------------------
def test_live_trade_paraphrases_refused(barbershop_tmp):
    """TEST 24 — RULE 7 holds for PARAPHRASES, but diagnostic questions are answered."""
    client = MockLLM(GOOD_RESPONSE)
    for q in ("Is now a good time to buy EURUSD?", "should I be long here?",
              "do I enter a long position right now?", "should i short this?",
              "is it a good entry right now?", "buy or sell right now?",
              "should I close my live position?"):
        resp = risk_doctor.ask(q, SCREEN3, client=client)
        assert resp["refused"] is True, f"not refused: {q}"
        assert resp["text"] == config.DOCTOR_REFUSAL_LIVE
    assert client.calls == 0                                   # never reached the LLM
    # A diagnostic question about a PAST trade is NOT refused.
    assert risk_doctor.is_live_trade_question("why did the bot go long on day 2?") is False


# --------------------------------------------------------------------------
# WI-5 — the manual is condensed to fit a local model's context, but the safety
# backbone (rules + diagnostic template + north star) is always kept (RULE 6-safe).
# --------------------------------------------------------------------------
def test_manual_condensed_keeps_safety_backbone_under_budget():
    """WI-5 — manual condensed below budget yet keeps safety rules + diagnostic template."""
    full = risk_doctor.load_manual()                       # the real ~32KB manual
    assert len(full) > config.DOCTOR_MANUAL_MAX_CHARS       # it does exceed the budget
    cond = risk_doctor.condense_manual(full, config.DOCTOR_MANUAL_MAX_CHARS)
    assert len(cond) < len(full)                           # actually shortened
    assert "SAFETY RULES" in cond                          # the safety backbone is kept
    assert "DIAGNOSTIC OUTPUT TEMPLATE" in cond
    assert "NORTH STAR" in cond
    # The assembled system prompt uses the condensed manual (Part 1 no longer 32KB).
    pk = risk_doctor.assemble_context_packet(
        "q", {"screen": 3, "day_id": 2, "trade_id": None}, history=[])
    assert "[Manual condensed" in pk["system"]
    assert len(pk["system"]) < len(full)                   # much smaller than the full manual
    # A short manual (e.g. the test marker) passes through unchanged.
    assert risk_doctor.condense_manual("tiny manual", config.DOCTOR_MANUAL_MAX_CHARS) == "tiny manual"
