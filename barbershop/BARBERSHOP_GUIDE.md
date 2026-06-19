# 💈 Barbershop — How to Use This Dashboard

> *"Get the haircut before going to school."* The Barbershop is the **read-only** cockpit where you
> watch the policy try to pass FTMO-style challenges, see exactly what broke and why, make **one**
> educated change via the `OVERRIDES` dict, and run again. It never trades, never edits training, and
> never writes to your policy — it only *shows* and *explains*.

This guide lives in `barbershop/BARBERSHOP_GUIDE.md` and renders inside the dashboard on **Screen 0
(How to Use)** / the floating **`[?]`** button. Keep it open while you learn the loop.

---

## 0. The mission (read this first)

The bot has exactly one job: **pass the challenge** — hit **+2.5%/day** without breaching the
**−4% trailing wall**, day after day. Every screen, score, and color here is in service of that one
question. PnL is *not* the scoreboard; **pass-rate → breaches → consistency → drawdown** is.

---

## 1. Data-source modes — demo vs real

The dashboard auto-detects its source (shown top-left as `Barbershop data source:`):

| Mode | When | What you see |
|---|---|---|
| **`mock`** | no telemetry on disk yet | labelled **DEMO** curves/cards so the UI is explorable |
| **`real`** | a run exists under `artifacts/telemetry/<run_id>.jsonl` | the actual per-step/-day data from your run |

A **DEMO** label on a chart means *placeholder, not a real result*. Run the Barbershop notebook
(`colab/Quantra_Barbershop.ipynb`) to produce real telemetry, then refresh.

---

## 2. The screens

### Screen 1 · Training Wall
The macro view: **pass-rate over training** plus the current run's headline FTMO numbers. The curve is
a labelled *demo curve* until a real pass-rate series is logged (a known gap). Auto-refreshes on a
`dcc.Interval`.

### Screen 2 · Scoreboard
One **card per challenge day**. Green = passed (target hit, no breach), red = failed, a wall icon =
breached the −4% trailing wall. **Click any card → jumps to Screen 3 (Day Replay)** for that day.

### Screen 3 · Day Replay
Candlestick replay of one day with the bot's **trades drawn on the chart** (entries/exits). Use the
**timeframe buttons** (1m/5m/…) to zoom. **Click any trade marker → jumps to Screen 4 (Trade Autopsy)**
for that trade.

### Screen 4 · Trade Autopsy
*Why* the bot did what it did, for one trade:
- **LEFT — what the bot SAW**: the input features (the locked observation). **Click any feature name
  to overlay its values on the Screen 3 chart** so you can see the indicator the decision rode on.
- **RIGHT — attribution**: how much each input pushed the action, via **input×gradient** (an honest
  approximation, *not* true SHAP). Read it as *relative* influence: bigger bar = stronger pull. The
  feature names are the exact labels from `quantra/market_pipeline/feature_builder/schema.py`.

### Screen 5 · Pattern Finder
Mines recurring setups behind failures/passes and proposes a **plain-English rule** for each. Buttons:
- **APPLY** — copies the suggested rule into your next-run `OVERRIDES` (it does **not** edit code; you
  still launch the run). **MODIFY** — edit the rule text first. **IGNORE** — dismiss it.

### Screen 6 · Repo Map *(new)*
A live **import-dependency graph** of the whole repo (`barbershop/repo_graph.py`, built from `ast`).
Nodes are colored by package — 🔴 `locked_core`, 🟠 `learning_system`, 🟣 `market_pipeline`,
🟢 `ftmo_passing`, 🩵 `diagnostics`/`barbershop`, 🔵 `env`/`live_bridge`. **Click a node** to see that
file's docstring + its first `COUPLING` note, so you can trace how a change ripples across files.

---

## 3. The Risk Doctor (the `?`-panel chat)

A read-only LLM that **diagnoses** using telemetry **evidence only** — it never guesses, never trades,
and has **no write access** to anything.

**What it CAN do:** read the loaded run's telemetry, classify a failure into the fixed taxonomy
(reward hijack, critic miscalibration, etc.), cite the per-layer reward decomposition, and — with the
repo reader — read any source file you point it at.

**What it CANNOT do:** change training, rewards, the policy, or execution; invent diagnoses without
telemetry; or act on its own. It's an advisor, not an operator.

**Repo reader:** type **`/read <path>`** (e.g. `/read quantra/env/trading_env.py`) to pull a file into
its context — **read-only, always**. **Backend** is configurable in `barbershop/config.py`
(`DOCTOR_PROVIDER`): a local model (Ollama/LM-Studio), Anthropic's Claude, or Perplexity `sonar` (live
web search). Keys live in env vars, never in code.

---

## 4. The JARVIS HUD + going LIVE

`jarvis_hud.html` (repo root) is a **self-contained** holographic cockpit: an animated dataflow graph
of the pipeline, live FTMO stats, and a scrolling telemetry log. Open it in any browser.

- **By itself** it runs a built-in **SIM** (amber pill) so it's alive immediately.
- **To go LIVE** (green pill), stream real telemetry into it:
  ```bash
  pip install websockets
  python -m barbershop.ws_broadcaster            # follows the newest artifacts/telemetry run
  # …or, with no trainer running:
  python -m barbershop.ws_broadcaster --demo
  ```
  The HUD connects to `ws://localhost:8765`; every training step then pulses the matching node and
  scrolls the log. Pipeline node ids mirror the SOW milestones (`data_loader → … → trainer → …`).

---

## 5. The change guardrail (so you never lose a policy)

When you shape the policy, stay inside the rails. Some changes are **resume-safe** (an old checkpoint
still loads); a few **force a fresh retrain**. The dashboard + `artifacts/policy_registry/README.md`
§4 hold the full **Compatibility Map**, but the short version:

| ✅ Safe to change (resume-safe) | ⚠️ Forces a fresh start |
|---|---|
| `training_phase`, `training_wheels` | the **observation width** `STATE_DIM` (schema / raw-inputs toggle) |
| challenge numbers (target / risk / perm-dd / failed-day) | the reward **layer arrangement** (add/remove/rename a layer) |
| **all reward weights + term math** (RewardConfig, C16/C17) | the **law set** (`LAW_NAMES`) |

Shape with the left column freely. Touch the right column only on purpose — the registry will refuse a
mismatched resume with a plain-English `CompatibilityError` (and never overwrites the old checkpoint).

---

## 6. The loop, in one breath

**Pick a window + OVERRIDES (Cell 3) → run the passes (Cell 5, real env metrics) → save the Policy
Card + read the Leaderboard (Cell 6) → diagnose on Screens 1–6 + the Risk Doctor → change ONE knob →
run again.** Repeat until a policy passes most days, then resume/promote it to Full Training.

---

*Update log (IRAC) — [2026-06-19] C22: created the in-dashboard guide. **I:** every feature needed a
single in-app reference (screens, navigation, Risk Doctor limits, data modes, the new Repo Map + JARVIS
HUD, the compatibility guardrail). **R:** operator brief (Feature 2 help screen) + the show-the-work
rule. **A:** wrote this guide; `dashboard.py` renders it on Screen 0 / the `[?]` button. **C:** the
operator always has the full workflow at hand, so the diagnose-and-shape loop stays fast and correct.*
