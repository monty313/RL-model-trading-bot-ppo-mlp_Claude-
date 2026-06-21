# ==========================================================================
# FILE: barbershop/config.py
# PURPOSE: Single source of constants for the Quantra Barbershop diagnostics
#          dashboard and the LLM Risk Doctor. Holds every file path, threshold,
#          timeframe window, indicator grouping, colour rule, and the local-LLM
#          endpoint settings. Nothing here trains, trades, or mutates the policy.
# ==========================================================================
#
# DEPENDS ON (reads from):
#   docs/MLP_INTERPRETABILITY_LAYER.md  — the Risk Doctor's operating manual
#                                         (resolved at its REAL repo location)
#
# PRODUCES (writes to): nothing — this is a constants module.
#
# RECONCILIATION NOTE (Barbershop spec v1.0 vs the real Quantra repo):
#   The spec was written against an idealised file layout
#   (data/prices_5m.csv, logs/trajectory.parquet, models/ppo_actor.pt,
#    root-level *.md). The real repo differs:
#     - price bars:   data/raw/EURUSD_M1.csv (1m only; 5m/30m/4H are RESAMPLED
#                     in-memory by quantra.market_pipeline.resampler — there are
#                     no prices_5m.csv files on disk).
#     - telemetry:    artifacts/telemetry/<run_id>.jsonl (JSONL, not parquet),
#                     written by quantra/diagnostics/telemetry_logger/logger.py.
#                     It has NO per-step GAE advantage and NO SHAP values.
#     - checkpoints:  artifacts/checkpoints/brain.pt (not models/ppo_actor.pt).
#     - docs:         docs/*.md (not repo root).
#   To keep this dashboard self-consistent AND honest, every path below is a
#   configurable constant. The defaults follow the SPEC paths (data/, logs/,
#   models/) so a fresh checkout + mock data "just works"; barbershop/adapter.py
#   maps the REAL artifacts/telemetry JSONL onto the spec contract and marks the
#   fields the live pipeline does not yet produce (advantage, SHAP).
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Constants for Barbershop v1.0 +
#                            Risk Doctor continuation v1.0. Paths reconciled
#                            against the real Quantra repo (see note above).
#   [2026-06-15] [Claude] — Adversarial-review fixes: added PLACEHOLDER_FIELDS
#                            (anti-fabrication), DOCTOR_NO_EVIDENCE +
#                            DOCTOR_UNVERIFIED_PREFIX, and made
#                            LIVE_TRADE_TRIGGER_WORDS specific (no false refusals).
#   [2026-06-19] [C24]    — Added DOCTOR_PROVIDER (ollama|anthropic|perplexity) + per-provider keys/
#                            models (env-sourced, never hardcoded) + DOCTOR_REPO_READ_MAX_CHARS for the
#                            /read reader. Default stays ollama (offline-first); risk_doctor reads these.
# ==========================================================================

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# ROOTS. barbershop/ lives directly under the repo root, so parents[1] of this
# file is the repo root. Everything else is resolved relative to it so a Colab
# run and a local run behave identically.
# --------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]   # .../final rl model 6_13
BARBERSHOP_DIR: Path = REPO_ROOT / "barbershop"
DATA_DIR: Path = REPO_ROOT / "data"
LOGS_DIR: Path = REPO_ROOT / "logs"        # spec location for telemetry + exports
MODELS_DIR: Path = REPO_ROOT / "models"    # spec location for the frozen actor
DOCS_DIR: Path = REPO_ROOT / "docs"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"   # the REAL pipeline output root

# --------------------------------------------------------------------------
# PRICE BARS. The spec names one CSV per timeframe. The real repo only ships the
# 1m export and resamples the rest, so the adapter can synthesise the higher-TF
# CSVs on demand; these paths are where the dashboard LOOKS for them.
# --------------------------------------------------------------------------
PRICE_CSVS: dict[str, Path] = {
    "1m": DATA_DIR / "prices_1m.csv",
    "5m": DATA_DIR / "prices_5m.csv",
    "30m": DATA_DIR / "prices_30m.csv",
    "4H": DATA_DIR / "prices_4h.csv",
}

# --------------------------------------------------------------------------
# TELEMETRY + EXPORT FILES (spec paths under logs/). The dashboard READS the
# first three and WRITES the last two (RULE 3: it may only ever write to logs/).
# --------------------------------------------------------------------------
TRAJECTORY_PARQUET: Path = LOGS_DIR / "trajectory.parquet"        # produced by telemetry/logger.py
SHAP_PARQUET: Path = LOGS_DIR / "shap_values.parquet"            # produced by telemetry/interpreter.py
MLP_TELEMETRY_PARQUET: Path = LOGS_DIR / "mlp_telemetry.parquet"  # hidden states / head outputs
PPO_ACTOR_PT: Path = MODELS_DIR / "ppo_actor.pt"                 # frozen actor weights (SHAP only)
SUGGESTED_RULES_JSON: Path = LOGS_DIR / "suggested_rules.json"    # WRITE: pattern + prescription export
DOCTOR_DIAGNOSES_JSONL: Path = LOGS_DIR / "doctor_diagnoses.jsonl"  # WRITE: append-only chat log

# The REAL telemetry directory the adapter reads JSONL runs from.
REAL_TELEMETRY_DIR: Path = ARTIFACTS_DIR / "telemetry"


def mlp_manual_path() -> Path:
    """Locate MLP_INTERPRETABILITY_LAYER.md — the Risk Doctor's operating manual.

    Reads: nothing (path resolution only).
    Returns: the first existing of docs/<file> (the REAL repo location) or
             <repo root>/<file> (the spec's stated location); if neither
             exists, returns the docs/ path so the caller's FAIL-LOUD check
             (RULE 6) names a sensible location.
    """
    name = "MLP_INTERPRETABILITY_LAYER.md"
    for candidate in (DOCS_DIR / name, REPO_ROOT / name):   # real first, then spec
        if candidate.exists():
            return candidate
    return DOCS_DIR / name


# Other blueprint files the Doctor may quote (resolved at the real docs/ home).
BLUEPRINT_FILES: dict[str, Path] = {
    "state_vector": DOCS_DIR / "STATE_VECTOR.md",
    "trading_code": DOCS_DIR / "THE_TRADING_CODE.md",
    "reward_design": DOCS_DIR / "REWARD_DESIGN.md",
    "ppo_engine": DOCS_DIR / "PPO_ENGINE.md",
}

# --------------------------------------------------------------------------
# DASH SERVER.
# --------------------------------------------------------------------------
DASH_HOST: str = "127.0.0.1"
DASH_PORT: int = 8050
TRAINING_WALL_REFRESH_MS: int = 60_000   # Screen 1 live refresh = 60 s (spec)

# --------------------------------------------------------------------------
# LOCAL LLM (Risk Doctor). OpenAI-compatible endpoint so Monty can point it at
# Ollama / LM Studio / vLLM. Defaults = Ollama. (The wider repo uses Anthropic
# for cloud calls; the Barbershop Doctor is deliberately a LOCAL, offline-first
# tool per the continuation spec.)
# --------------------------------------------------------------------------
DOCTOR_API_BASE: str = "http://localhost:11434/v1"   # default = Ollama
DOCTOR_MODEL: str = "llama3"                          # default model id
DOCTOR_API_KEY: str = "ollama"                        # placeholder key (local servers ignore it)

# --- Provider selection (C24, 2026-06-19). Default stays the local, offline-first Ollama path, so the
# existing behaviour + tests are unchanged. COUPLING -> risk_doctor.make_client()/_completion_text():
# DOCTOR_PROVIDER picks which client/endpoint is built. Keys come from ENV — NEVER hardcoded/committed.
import os  # noqa: E402  (only for the provider env-var reads below)
DOCTOR_PROVIDER: str = os.environ.get("DOCTOR_PROVIDER", "ollama")     # ollama | anthropic | perplexity
# Anthropic (Claude) — uses the `anthropic` SDK directly (messages API), NOT the OpenAI-compat shim.
DOCTOR_ANTHROPIC_KEY: str = os.environ.get("DOCTOR_ANTHROPIC_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
DOCTOR_ANTHROPIC_MODEL: str = os.environ.get("DOCTOR_ANTHROPIC_MODEL", "claude-sonnet-4-5")
# Perplexity — OpenAI-compatible endpoint; `sonar` models have live WEB SEARCH built in (no extra setup).
DOCTOR_PERPLEXITY_KEY: str = os.environ.get("DOCTOR_PERPLEXITY_KEY", os.environ.get("PERPLEXITY_API_KEY", ""))
DOCTOR_PERPLEXITY_BASE: str = "https://api.perplexity.ai"
DOCTOR_PERPLEXITY_MODEL: str = os.environ.get("DOCTOR_PERPLEXITY_MODEL", "sonar")
# /read <path> repo reader — READ-ONLY, capped, never escapes the repo root (no write path exists).
DOCTOR_REPO_READ_MAX_CHARS: int = 16000
DOCTOR_MAX_TOKENS: int = 800                          # keep responses short
DOCTOR_TEMPERATURE: float = 0.2                       # low — diagnosis, not creativity
DOCTOR_HISTORY_LIMIT: int = 6                         # last N exchanges in the system prompt
DOCTOR_TRAJECTORY_TAIL: int = 10                      # last N trajectory rows in the context packet
# The MLP manual is ~32 KB (~8k tokens). Sending it whole every call overflows a small
# local model's context (default llama3 is ~8k). Condense it to this many chars when it
# exceeds the budget, keeping the high-signal sections (safety rules, diagnostic template,
# failure taxonomy, north star, core definitions). RECOMMENDATION: point DOCTOR_MODEL at a
# long-context local model (e.g. llama3.1:8b-instruct-128k, qwen2.5, mistral-nemo) and raise
# this if you want the full manual every call.
DOCTOR_MANUAL_MAX_CHARS: int = 12000

# --------------------------------------------------------------------------
# CHALLENGE / DIAGNOSTIC THRESHOLDS. These mirror the FTMO geometry the bot is
# trained against (2.5% target, 4% trailing wall) and the heatmap colour rules.
# --------------------------------------------------------------------------
DAILY_TARGET_PCT: float = 2.5            # Phase-A profit target
DD_WALL_PCT: float = 4.0                 # trailing drawdown wall
CONSISTENT_PASS_ZONE: float = 80.0       # Screen 1 dashed line (% pass rate)
PASS_RATE_WINDOW: int = 1000             # rolling window of steps for pass-rate
DD_BUFFER_RED_THRESHOLD: float = 0.25    # DD buffer below this -> RED (danger)
DD_BUFFER_YELLOW_THRESHOLD: float = 0.50  # below this (but >= red) -> YELLOW
ATR_HIGH_MULT: float = 1.3               # ATR > 1.3x daily avg -> "high volatility"
PLATEAU_TOLERANCE_PCT: float = 2.0       # +/- this over checkpoints = "flat" (yellow)
PLATEAU_CHECKPOINTS: int = 3             # flat for this many checkpoints -> banner

# --------------------------------------------------------------------------
# ACTION SPACE. The dashboard contract orders probabilities [long, short, hold,
# close] (spec Section 4). The live engine uses ints {HOLD:0, OPEN_LONG:1,
# OPEN_SHORT:2, CLOSE:3} — adapter.py maps between the two.
# --------------------------------------------------------------------------
# Contract constants live in barbershop/contract.py (the single source of truth); they
# are re-exported here so existing config.* references keep working. PLACEHOLDER_FIELDS:
# fields the live pipeline may not produce — flagged so a NaN is never cited as evidence.
from barbershop.contract import (  # noqa: E402  (re-export of the data contract)
    ACTIONS, ENGINE_ACTION_INTS, PLACEHOLDER_FIELDS,
)

# Per-action icons (UI only — stays in config; keyed by the contract ACTIONS).
ACTION_ICONS: dict[str, str] = {
    "OPEN_LONG": "⬆️",   # up arrow
    "OPEN_SHORT": "⬇️",  # down arrow
    "HOLD": "⬜",              # white square
    "CLOSE": "\U0001f534",         # red circle
}

# --------------------------------------------------------------------------
# TIMEFRAME CONTEXT WINDOWS (Screen 3). Minutes of context to show on EACH side
# of the selected trade's entry time, per the spec's per-TF rules.
# --------------------------------------------------------------------------
TF_CONTEXT_MINUTES: dict[str, int] = {
    "1m": 30,            # +/- 30 minutes
    "5m": 120,           # +/- 2 hours
    "30m": 720,          # +/- 12 hours
    "4H": 5 * 24 * 60,   # +/- 5 days
}
TIMEFRAMES: list[str] = ["1m", "5m", "30m", "4H"]
PANELS_1M_ONLY: tuple[str, str] = ("advantage_strip", "indicator_heatmap")  # hidden on higher TFs

# --------------------------------------------------------------------------
# HEATMAP / BAR COLOURS (Screen 3 ribbon + Screen 4 bars). One palette so the
# "what the bot saw" colour language is identical everywhere.
# --------------------------------------------------------------------------
COLOR_GREEN: str = "#1D9E75"    # bullish / favourable / active / pushed-toward
COLOR_RED: str = "#E0533D"      # bearish / dangerous / blocked / pushed-away
COLOR_YELLOW: str = "#E5BC24"   # neutral / borderline
COLOR_GREY: str = "#7A7F87"     # not applicable
COLOR_GOLD: str = "#E8B339"     # chosen-action highlight border
COLOR_PROFIT_FILL: str = "rgba(29,158,117,0.18)"   # green trade shade
COLOR_LOSS_FILL: str = "rgba(224,83,61,0.18)"      # red trade shade

# --------------------------------------------------------------------------
# INDICATOR GROUPS (Screen 3 heatmap rows + Screen 4 left column). The 5
# categories from STATE_VECTOR.md, with the human labels the spec asks for.
# Concrete feature names are resolved at runtime from the mock/real observation
# so this list stays robust to schema growth (e.g. the new training-wheel feats).
# --------------------------------------------------------------------------
INDICATOR_GROUPS: list[tuple[str, str]] = [
    ("challenge_health", "\U0001f3e6 Challenge Health"),   # DD buffer, daily progress, trades left
    ("laws_gates", "⚖️ Laws & Gates"),           # each law/gate active/blocked
    ("market_structure", "\U0001f4ca Market Structure"),   # BB, SMA, price vs bands
    ("momentum", "⚡ Momentum"),                       # CCI variants on all TFs
    ("volatility", "\U0001f321️ Volatility"),         # ATR values
]

# --------------------------------------------------------------------------
# REGIME LABELS (Screen 2 card labels) — the vocabulary the mock generator and
# the adapter both use so the scoreboard text is stable.
# --------------------------------------------------------------------------
REGIME_LABELS: list[str] = ["Trending", "Choppy", "News Day", "Slow Day"]

# --------------------------------------------------------------------------
# RISK DOCTOR TEXT CONSTANTS. The hard-rule reminder + canned phrases must be
# byte-stable because the tests assert on them.
# --------------------------------------------------------------------------
DOCTOR_HARD_RULES: str = (
    "HARD RULES - YOU MUST FOLLOW THESE EVERY RESPONSE:\n"
    "1. Every claim must cite a specific telemetry field or metric.\n"
    "2. If evidence is missing, say 'insufficient evidence' and stop.\n"
    "3. Never suggest touching execution, laws, or live trading.\n"
    "4. Always state your confidence: HIGH / MEDIUM / LOW.\n"
    "5. The mission is FTMO passing - not raw PnL."
)
DOCTOR_REFUSAL_LIVE: str = "I only diagnose training runs, not live positions."
DOCTOR_OFFLINE_RESPONSE: str = "OFFLINE"
DOCTOR_OFFLINE_MESSAGE: str = (
    "⚠️ Risk Doctor temporarily offline.\n"
    "Check that your local LLM server is running at {api_base}.\n"
    "Your question has been saved and will be answered when reconnected."
)
DOCTOR_MANUAL_MISSING: str = (
    "Risk Doctor offline - MLP_INTERPRETABILITY_LAYER.md not found.\n"
    "This file is required for the Doctor to operate correctly."
)
DOCTOR_NO_CONTEXT: str = "Please select a day or trade first so I can see what you're looking at."
DOCTOR_NO_EVIDENCE: str = (
    "Insufficient evidence - no telemetry is loaded for the selected day, so I cannot "
    "diagnose it without guessing. Load a training run (logs/trajectory.parquet) first."
)
# Prepended to any LLM answer that cites zero telemetry fields (anti-fabrication).
DOCTOR_UNVERIFIED_PREFIX: str = (
    "⚠️ UNVERIFIED - this answer cited no telemetry field, so treat it as a hunch, "
    "not evidence. (Confidence forced to LOW.)\n\n"
)

# The six display sections the chat box renders (icon, heading). The Doctor's
# free text is parsed/segmented into these; tests assert all six icons appear.
DOCTOR_SECTIONS: list[tuple[str, str]] = [
    ("\U0001f4cd", "What I'm looking at"),     # 📍
    ("\U0001f50d", "What I see"),               # 🔍
    ("\U0001f3af", "What it means for passing"),  # 🎯
    ("✅", "What to do next"),              # ✅
    ("❌", "What NOT to do"),               # ❌
    ("\U0001f4ca", "Confidence"),               # 📊
]

# Phrases that, if present in Monty's question, trip RULE 7 (no execution authority).
# Deliberately specific (e.g. "should i go long", NOT bare "go long") so DIAGNOSTIC
# questions about PAST trades ("why did the bot go long on day 2?") are NOT refused;
# is_live_trade_question() adds a regex for present-tense "[action] now/here" forms.
LIVE_TRADE_TRIGGER_WORDS: tuple[str, ...] = (
    "should i go long", "should i go short", "should i buy", "should i sell",
    "should i be long", "should i be short", "go long now", "go short now",
    "enter now", "open a trade now", "place a trade", "take this trade now",
    "good time to buy", "good time to sell", "close my live position",
)
