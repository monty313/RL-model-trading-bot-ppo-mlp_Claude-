# 🛰️ QUANTRA · J.A.R.V.I.S. HUD — Full Operating Guide

A cinematic, Iron-Man-style real-time cockpit for the RL trading system. This guide covers **every
piece, every mode, and every knob** — from "double-click to watch it run" to "stream live training
telemetry into a holographic graph."

There are **three parts** and they're designed to work alone or together:

| File | What it is | Needs |
|---|---|---|
| **`jarvis_hud.html`** | the HUD itself — a single self-contained web page | a browser (nothing else) |
| **`barbershop/ws_broadcaster.py`** | streams real telemetry → the HUD over a WebSocket | Python + `websockets` |
| **`barbershop/repo_graph.py`** | builds the real import-dependency graph of the repo | Python (stdlib) |

---

## 1. ⚡ Quick start (10 seconds, no setup)

1. Open **`jarvis_hud.html`** in any modern browser (double-click it, or drag it into a tab).
2. Watch the **boot sequence** (`INITIALIZING QUANTRA SYSTEMS…`), then the HUD powers on with a sweep.
3. The pill top-right reads **`SIM`** (amber) — it's running a built-in **simulator**, so every panel is
   alive immediately with plausible fake data. **That's expected with no live feed.**

You now see: an animated pipeline graph, three hexagon vitals gauges, a scrolling telemetry log, a
live clock, and the rotating arc-reactor. **Nothing is real yet** — that's the next section.

---

## 2. 🔴 Going LIVE (stream real training telemetry)

The HUD listens on **`ws://localhost:8765`**. The broadcaster reads the telemetry your training writes
and pushes each step to the HUD. Three steps:

```bash
# 1) one-time: the live server needs the websockets lib
pip install websockets

# 2) produce telemetry — run anything that writes artifacts/telemetry/<run>.jsonl, e.g. the
#    acceptance chain or a Barbershop/Train run. (Skip this and use --demo below to test the pipe.)
python -m quantra.acceptance            # or your training entry point

# 3) start the bridge (in a second terminal) — it follows the NEWEST telemetry file:
python -m barbershop.ws_broadcaster
```

Now **open / refresh `jarvis_hud.html`**: the pill flips to **`LIVE`** (green), and every training step
**pulses the matching node**, moves the gauges, and scrolls a real log line.

**No trainer running but want to test the live path?**
```bash
python -m barbershop.ws_broadcaster --demo   # synthetic stream over the real WebSocket
```

> The HUD auto-reconnects: if the broadcaster stops, it drops back to **SIM** and keeps retrying the
> socket every 4 s, flipping to **LIVE** again the moment the broadcaster returns.

---

## 3. 🧭 Reading the HUD (every panel)

**Header** — the rotating **arc-reactor** emblem + `QUANTRA` wordmark; a live **clock**; and three pills:
`SIM/LIVE` (data source), `FTMO …` (current challenge status, color-coded), and the clock.

**Center — System Dataflow graph** (d3.js force layout, left→right by pipeline stage):
- **Nodes** = pipeline modules, colored by tier (🟣 market pipeline, 🔵 env, 🟠 learning system,
  🟢 diagnostics, blue live-bridge). **Hover** a node → a tooltip shows its **SOW milestone + role**
  (e.g. *M6 · RewardEngine — L0–L6 + QUAD · E8 dominance*).
- **Edges** carry **comet dots** flowing source→target (the data path). When a step touches a module,
  that node **brightens, grows, and emits an expanding pulse-ring**, and its edges' comets speed up.
- A slow **radar sweep** rotates behind the graph. Click **`↻ PULSE ALL`** to fire every node at once.

**Right — Vitals**:
- **EPISODE REWARD** — big readout, green when ≥0, red when negative (eased, not jumpy).
- Three **circle-in-hexagon arc gauges**:
  - **WIN RATE** (cyan arc, 0–100%),
  - **PnL /2.5** — daily PnL toward the **+2.5% target**; turns green once it hits target,
  - **DD /4.0** — drawdown toward the **−4% wall**; ramps **amber at 3%, red at 3.5%+**.
- **FTMO STATUS** badge + step counter: `BUILDING → PASSING` (target hit), `AT RISK` (deep drawdown),
  `BREACH` (wall hit) — the badge pulses in its status color.

**Bottom — Telemetry Stream**: color-coded log lines reveal with a **typewriter** effect, newest on
top, auto-trimmed. Green = good step, amber = warning (near the wall), red = breach.

**Ambient**: drifting particle field, CRT scanlines, holographic flicker, a faint arc-reactor watermark
at 1 RPM, breathing corner brackets, and a cursor targeting-reticle.

---

## 4. 📡 The event schema (the contract)

The HUD and the broadcaster meet on **one JSON message shape**. If you write your own producer, emit
this (every field optional except keep it JSON; the HUD holds the previous value for anything missing):

```json
{ "t":"step", "step": 5821, "module": "trainer",
  "reward": 0.34, "daily_pnl_pct": 1.82, "drawdown_pct": 1.2,
  "win_rate": 57.0, "ftmo": "BUILDING", "action": "BUY EURUSD 0.02" }
```

- **`module`** must be one of the pipeline node ids to pulse a node: `data_loader, feature_builder,
  law_mask_engine, trading_env, ppo_agent, rollout_buffer, reward_engine, curriculum_manager, trainer,
  telemetry_logger, mlp_interpreter, llm_risk_doctor, live_bridge`.
- **`ftmo`** ∈ `PASSING | BUILDING | AT RISK | BREACH`.

`ws_broadcaster.packet_to_event()` is the single place that maps a `telemetry_logger` **StepPacket** to
this schema (reward from `reward_decomposition.total`, drawdown from `risk_context.trailing_dd`, action
from `chosen_action`+`symbol`, node from an optional `active_module`). It's pure + unit-tested.

---

## 5. 🗺️ The dependency graph (repo_graph.py)

The HUD's center graph is the *curated pipeline* (clean + cinematic). The **accurate, full** import
graph of the whole repo comes from `barbershop/repo_graph.py` (AST-only — it never runs your code):

```bash
python -m barbershop.repo_graph                    # writes artifacts/repo_graph.json
python -m barbershop.repo_graph my_graph.json      # …or to a path you choose
```

That JSON (`{nodes:[{id,group,color,doc,coupling,loc}], links:[{source,target}], groups:{…}}`) powers
the future **Barbershop Screen 6 (Repo Map)** and can be loaded into any d3/cytoscape view. Colors:
🔴 `locked_core`, 🟠 `learning_system`, 🟣 `market_pipeline`, 🟢 `ftmo_passing`, 🩵 `diagnostics`/
`barbershop`, 🔵 `env`/`live_bridge`, ⚪ `runtime`/`scripts`/`tests`. Each node carries its **docstring**
and its **first `COUPLING` note** — so you can trace what depends on what.

---

## 6. 🎨 Customizing the HUD

Everything is in `jarvis_hud.html` (one file, no build step):

| Want to… | Where |
|---|---|
| change the WebSocket address | `connect()` → `new WebSocket("ws://localhost:8765")` |
| recolor the theme | the `:root` CSS vars (`--cyan`, `--gold`, `--green`, `--bg`, …) |
| add/rename a pipeline node | the `TIER` / `ROLE` / `COLOR` / `LAYER` maps + `LINKS` near the top of the script |
| change sim cadence / values | `simStep()` + `setInterval(simStep, 900)` |
| slow/speed the comet flow | the `frame()` loop (`ph + 0.005…`) |
| change boot lines | the `BOOT` array near the bottom |

The HUD only ever **reads** the socket — it has no control authority over training or execution.

---

## 7. 🧯 Troubleshooting

- **Pill stuck on `SIM`** → the broadcaster isn't up, or it's on a different host/port. Start
  `python -m barbershop.ws_broadcaster` (or `--demo`) and confirm it printed `ws://localhost:8765`.
- **`ws_broadcaster: No telemetry…`** → nothing under `artifacts/telemetry/`. Run a producer first, or
  use `--demo`.
- **`needs websockets`** → `pip install websockets`.
- **Graph/log frozen, no errors** → you're in SIM and the tab was backgrounded; browsers throttle
  timers in background tabs. Bring it to the foreground.
- **Fonts look plain** → Orbitron/Rajdhani load from Google Fonts; if you're fully offline the HUD
  falls back to a system mono/sans — everything still works.
- **Opening via `file://` and the socket won't connect** → that's fine; `ws://localhost` is allowed
  from `file://`. If a corporate proxy blocks localhost WS, serve the file:
  `python -m http.server` then open `http://localhost:8000/jarvis_hud.html`.

---

## 8. 🏗️ Architecture (one glance)

```
 Trainer / acceptance ──▶ telemetry_logger ──▶ artifacts/telemetry/<run>.jsonl
                                                          │ tail -f
                                          barbershop/ws_broadcaster.py
                                          packet_to_event()  (pure, tested)
                                                          │ ws://localhost:8765
                                                          ▼
                                                  jarvis_hud.html
                            applyEvent() → pulse node · ease gauges · scroll log
                            (no socket → built-in SIMULATOR, amber "SIM" pill)

 barbershop/repo_graph.py ──▶ artifacts/repo_graph.json ──▶ (Barbershop Screen 6 · any d3/cytoscape)
```

---

## 9. 🔌 Make the trainer pulse exact nodes (optional, advanced)

By default the broadcaster tags every step as `module:"trainer"`. To light up the *real* active stage,
have your training loop add an **`active_module`** field to the telemetry packet it writes (one of the
node ids in §4). `packet_to_event()` already reads it — no HUD change needed. *(Wiring `active_module`
into `quantra/learning_system/trainer/trainer.py` is the remaining Screen-6 task.)*

---

*That's the whole system. Open `jarvis_hud.html`, run `ws_broadcaster.py`, and watch QUANTRA learn to
pass FTMO in real time — Stark-style.* 🛰️

*Update log (IRAC) — [2026-06-19] C24: created the JARVIS operating guide. **I:** the HUD + broadcaster
+ repo graph needed one detailed, end-to-end usage reference. **R:** operator request ("add very
detailed instructions on how to use all of the JARVIS stuff"). **A:** wrote this guide (quick start,
going live, every panel, the event schema, the graph builder, customization, troubleshooting,
architecture). **C:** anyone can run the live cockpit and read it correctly in minutes.*
