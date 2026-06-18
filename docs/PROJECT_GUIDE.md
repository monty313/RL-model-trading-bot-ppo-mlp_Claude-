# Quantra — Complete Project Guide (Barbershop, Training, Trading)

> **Purpose of this file.** This is the single source-of-truth manual for the Quantra
> project, written so an assistant (e.g. a Perplexity Space) can answer *"how do I use
> this project?"* — covering the Barbershop diagnostics dashboard, training, backtesting,
> live trading, and every feature, with exact file names, locations, and commands.
> Last updated 2026-06-18.

---

## 1. What Quantra is

Quantra is a **reinforcement-learning trading bot** whose goal is to **pass FTMO-style
prop-firm challenges repeatedly**: reach a daily profit **target (default +2.5%)** without
ever touching a **trailing drawdown wall (default −4%)**, on real MT5 forex/metal/index bars.

- **Algorithm:** PPO (Proximal Policy Optimization) — an actor-critic with a clipped
  objective and GAE advantages.
- **Policy I/O:** the agent sees a fixed observation vector (**STATE_DIM = 203**) and emits
  4 heads — direction `{HOLD, OPEN_LONG, OPEN_SHORT, CLOSE}`, a Beta-distributed size, a
  slot pointer (for closing), and a value estimate.
- **Safety spine:** 9 "laws" + 3 "gates" + operator "training-wheel" masks forbid illegal/
  counter-trend/illiquid trades **before** the policy acts (logit −1e9), in both training
  and live. A RiskManager guarantees total open risk never exceeds the remaining buffer.
  - **About the ADF (stationarity) gate:** it is **not just a blocker to remove**. Its
    purpose is to teach the bot to trade **both stationary AND non-stationary** market
    conditions by controlling *when* the gate is enforced vs relaxed. Loosening its
    p-value threshold is a **diagnostic** move (let more bars through, watch the policy,
    then decide if the *calibration* needs a permanent change) — the gate stays in the
    architecture; only its calibration changes. (Stated again wherever the gate appears.)
- **Barbershop:** a separate, **read-only** Dash dashboard (+ an LLM "Risk Doctor") used
  **after** training to understand *why* the bot did what it did and to write better
  reward/penalty rules.

**Two operating modes (entered freely, NOT sequential):**
- **Barbershop mode** — *"get the haircut before going to school."* Fast, operator-driven
  diagnose-and-shape: pick a small window of challenge days, run the policy fast, watch
  what breaks (above all the **gate block rate**), make one educated `OVERRIDES` edit,
  repeat. Can happen **before, during, or after** full training (§4.10).
- **Full Training mode** — long walk-forward runs + curriculum phases (Law School → Setup
  Recognition → Full Market) that *generalize* the shaped policy across regimes (§4.6,
  `colab/Quantra_Train.ipynb`). Uses the Barbershop-shaped policy as starting weights.

**Two repos — know which one you're in (full protocol in §4.13):**
- **ACTIVE WORK** (all edits + commits happen here): `https://github.com/monty313/RL-model-trading-bot-ppo-mlp_Claude-`
- **FALLBACK / SAFE RESTORE** (NEVER edited — clean emergency revert only): `https://github.com/monty313/final-rl-model-6_13`

**Known gaps (honest — stated in every overview; full detail in §7):** (1) the gates block
~98.7% of trade opportunities on real EURUSD — the **#1 active blocker**, a *calibration*
issue not a bug; (2) no real trained model yet (synthetic-trained, doesn't transfer);
(3) the sim models ONE wall, real FTMO has TWO; (4) Barbershop Screen 1 is a labelled demo
curve until a real pass-rate series is logged; (5) trade-autopsy attribution is
input×gradient, not true SHAP.

---

## 2. Directory tree (the real layout)

```
final rl model 6_13/
├── README.md                      # repo intro
├── REPO_MAP.md                    # high-level module map
├── INSTRUCTIONS.md                # live work queue (pending/agreed work)
├── COUPLINGS.md                   # cross-file coupling map (C1–C9) — read before refactors
├── CHANGE_IMPACT.md               # change-impact tracker notes
├── pyproject.toml                 # pkg config + pytest (testpaths=["tests"])
├── requirements.txt               # deps (torch, dash, openai, pandas, …)
├── 0_QUANTRA_LIVE_COCKPIT.html    # static live cockpit mockup
│
├── docs/                          # the binding blueprint + this guide
│   ├── PROJECT_GUIDE.md           # ← THIS FILE (how to use everything)
│   ├── 00_START_HERE.md
│   ├── SCOPE_OF_WORK.md           # SOW (the spec)
│   ├── SOW_2_BUILD_SPEC.md        # binding build spec
│   ├── THE_TRADING_CODE.md        # the 9 laws + 3 gates (exact params/TFs)
│   ├── STATE_VECTOR.md            # every observation feature + group
│   ├── REWARD_DESIGN.md           # the layered reward (L0–L6 + QUAD)
│   ├── PPO_ENGINE.md              # actor/critic architecture + locked dials
│   ├── MLP_INTERPRETABILITY_LAYER.md  # the Risk Doctor's operating manual (terms, taxonomy)
│   ├── OPEN_QUESTIONS.md
│   └── architecture/quantra_map.html  # static architecture map
│
├── quantra/                       # the bot (training + live)
│   ├── runtime/
│   │   ├── config.py              # 🔧 ALL runtime knobs (challenge, symbols, paths, toggles)
│   │   ├── device.py              # CPU/GPU selection (RepresentativePolicy)
│   │   ├── optimizer.py           # hardware auto-optimizer (plan())
│   │   ├── autoscale.py           # CPU env scaling
│   │   ├── throughput_benchmark.py
│   │   ├── utilization_monitor.py
│   │   └── __main__.py            # `python -m quantra.runtime` (startup/benchmark)
│   ├── market_pipeline/           # bars → features → legal action space
│   │   ├── data_loader/loader.py          # parse MT5 CSV / Drive (load_symbol, parse_mt5_csv)
│   │   ├── resampler/resampler.py          # 1m → 5m/30m/4H, lookahead-safe as-of merge
│   │   ├── feature_builder/
│   │   │   ├── indicators.py               # BB, CCI, ATR, shifted-SMA, ADX, ADF, training-wheel params
│   │   │   ├── schema.py                   # StateVectorSchema (STATE_DIM=203, block layout)
│   │   │   ├── builder.py                   # FeatureBuilder + offline precompute (memmap cache)
│   │   │   └── RAW_INPUTS.md                # note on the raw-input block
│   │   └── law_mask_engine/engine.py       # law states → action mask (incl. training wheels)
│   ├── locked_core/               # 🔴 locked physics/laws (change needs sign-off)
│   │   ├── laws/laws.py                     # the 9 directional laws + 3 gates
│   │   ├── risk_manager/risk.py             # raw_size→lots, no-overshoot (B5), margin ceiling
│   │   ├── cost_layer/costs.py              # spread + slippage + $5 RT/lot (forex) costs
│   │   └── platform_adapter/adapters.py     # MT5 / sim broker adapter interface
│   ├── ftmo_passing/
│   │   ├── challenge_state.py               # account: equity, peak, trailing wall, breach, target, daily reset
│   │   └── validation/
│   │       ├── walk_forward.py              # 7-seed walk-forward harness
│   │       └── scoreboard.py                # pass-rate scoreboard
│   ├── env/trading_env.py          # the RL env (sequential multi-symbol, PnL, breach, masks)
│   ├── learning_system/            # the PPO trainer + reward
│   │   ├── ppo_agent/
│   │   │   ├── agent.py                     # ActorCritic (3×256) + PPOAgent (act / evaluate)
│   │   │   └── loss.py                      # PPO clipped loss (ratio, value, entropy, KL)
│   │   ├── trainer/
│   │   │   ├── gae.py                       # GAE advantage + returns (γ=0.997, λ=0.97 locked)
│   │   │   ├── trainer.py                   # the PPO loop (collect → GAE → K-epoch update → checkpoint)
│   │   │   └── scheduler.py                 # aggression scheduler (G8 miss-rate)
│   │   ├── rollout_buffer/buffer.py         # on-policy 10-field transition store
│   │   ├── reward_engine/reward.py          # layered reward L0–L6 + QUAD bonus
│   │   ├── curriculum_manager/curriculum.py # law-school stages + 1m feature mask
│   │   └── hpo/hpo.py                       # Optuna HPO (non-locked dials only)
│   ├── live_bridge/                # live/demo MT5 loop
│   │   ├── live_runner.py                   # CLI live runner
│   │   ├── live_session.py                  # the live decision loop (mirrors training masks)
│   │   ├── execution_adapter.py             # broker order execution
│   │   └── manual_halt.py                   # ManualHalt kill-switch
│   ├── diagnostics/                # telemetry + interpretability
│   │   ├── telemetry_logger/logger.py       # versioned JSONL StepPacket per decision
│   │   ├── mlp_interpreter/interpreter.py   # the M10 visuals (matplotlib)
│   │   ├── llm_risk_doctor/doctor.py        # the in-pipeline (Anthropic) risk doctor
│   │   └── failure_atlas/atlas.py
│   ├── constitution/               # mission + safety boundaries (markdown)
│   └── acceptance.py               # acceptance gate
│
├── barbershop/                    # 💈 the read-only diagnostics dashboard + Risk Doctor
│   ├── dashboard.py               # the Dash app (5 screens) — `python barbershop/dashboard.py`
│   ├── data.py                    # pure data layer (mock gen, loaders, transforms)
│   ├── figures.py                 # Plotly figure builders
│   ├── adapter.py                 # REAL telemetry (artifacts/telemetry/*.jsonl) → dashboard contract
│   ├── contract.py                # single source of the data contract (columns, actions)
│   ├── risk_doctor.py             # the LOCAL-LLM Risk Doctor brain (OpenAI-compatible)
│   ├── doctor_chat.py             # the chat-box UI + 6-section response renderer
│   ├── config.py                  # dashboard paths, thresholds, DOCTOR_* LLM settings
│   ├── conftest.py                # test fixtures (mock data)
│   ├── test_dashboard.py          # tests 1–10 + extras
│   ├── test_risk_doctor.py        # tests 11–20 + extras
│   └── REMEDIATION_PLAN.md        # the 8 fixes applied after the mentor review (all done)
│
├── scripts/                       # runnable entry points
│   ├── real_backtest.py           # honest train-on-train / test-on-held-out backtest on real bars
│   └── emit_real_telemetry.py     # PRODUCER: run the policy on real bars → telemetry for the Barbershop
│
├── tests/
│   ├── conftest.py                # shared synthetic-data fixtures
│   └── test_ftmo_master_suite.py  # the master suite (Sections A–T)
│
├── tools/
│   ├── snapshot.py                # state-vector snapshot guard (--check / --update)
│   └── impact.py                  # change-impact graph
│
├── colab/
│   ├── Quantra_Train.ipynb        # FULL TRAINING mode notebook (7-seed walk-forward)
│   └── Quantra_Barbershop.ipynb   # BARBERSHOP mode fast loop (8 cells; §4.10 / §4.14)
│
├── data/        (gitignored)      # raw MT5 CSVs (data/raw/EURUSD_M1.csv, EURUSD_recent.csv, …)
├── artifacts/   (gitignored*)     # checkpoints + telemetry (artifacts/checkpoints, artifacts/telemetry)
│   └── policy_registry/           # one folder per policy (manifest/performance/compat — §4.11)
│       └── README.md              # *the ONLY committed file under artifacts/ (registry guide)
└── logs/        (gitignored)      # Barbershop runtime exports (suggested_rules.json, doctor_diagnoses.jsonl)
```

---

## 3. The end-to-end workflow (mental model)

```
 1. DATA      data/raw/EURUSD_M1.csv  (real MT5 1m bars)
                  │  market_pipeline.data_loader.load_symbol
 2. FEATURES  resampler → feature_builder → 203-feature observation (cached memmap)
                  │  market_pipeline.feature_builder
 3. LAWS/MASK laws.py → law_mask_engine → legal action set (laws+gates+training wheels)
                  │
 4. ENV       env/trading_env.py  (PnL, costs, margin, breach wall, daily reset)
                  │
 5. TRAIN     learning_system.trainer.Trainer  (PPO: collect → GAE → clipped update)
                  │  reward_engine supplies the layered reward
 6. CHECKPOINT artifacts/checkpoints/<name>.pt   (+ walk-forward validation)
                  │
 7. TELEMETRY diagnostics.telemetry_logger  → artifacts/telemetry/<run>.jsonl
                  │  (scripts/emit_real_telemetry.py is the producer)
 8. BARBERSHOP barbershop/dashboard.py  → diagnose WHY (5 screens + Risk Doctor)
                  │  → logs/suggested_rules.json  (rules YOU approve to feed back into training)
 9. LIVE/DEMO live_bridge.live_runner  (DEMO first; same masks as training)
```

Plain English: real bars → features → a legal action space → the env simulates trading
with real costs and the FTMO wall → PPO trains a policy against the layered reward →
checkpoints are validated → a run emits telemetry → the Barbershop lets you see and
diagnose every decision → you approve reward/penalty rule changes → retrain → eventually
go live on a demo account.

---

## 4. Features and how to use them (commands)

All commands run from the repo root. Python 3.10, deps in `requirements.txt`
(`pip install -r requirements.txt`).

### 4.1 Run the test suite (verify the substrate is sound)
```
pytest tests/ barbershop/         # full suite (155 tests + barbershop) — should be green
pytest barbershop/                # just the 37 Barbershop tests
python tools/snapshot.py --check  # verify the observation layout hasn't drifted
```

### 4.2 Honest backtest on real bars (train on a slice, test on held-out)
`scripts/real_backtest.py` loads real MT5 bars, trains the PPO brain on the first 70%,
then runs the **deterministic** policy on the held-out tail and prints an
MT5-Strategy-Tester-style report (ground-truth net = real account change).
```
python scripts/real_backtest.py --symbol EURUSD --path data/raw/EURUSD_recent.csv \
    --updates 40 --target 2.5 --risk 4.0
```
Flags: `--updates` (PPO updates; 0 = untrained baseline), `--train_frac` (default 0.7),
`--target` / `--risk` (daily % target / trailing %). Writes an equity-curve PNG to `data/`.

> **`--path` now defaults to `None` (fix, 2026-06-18).** Omitting `--path` makes
> `load_symbol` resolve the bars itself — Parquet cache → Drive mount → `gdown`
> auto-download by registered Drive file ID — instead of raising `FileNotFoundError` on a
> clean checkout. So on a fresh machine with no local CSV you can just run
> `python scripts/real_backtest.py --symbol EURUSD --updates 40`. (Previously `--path`
> defaulted to a hardcoded CSV and was always wrapped in `Path(...)`, which blocked the
> very first real run — the gateway to a real FTMO pass. See the IRAC in the script header.)

### 4.3 Produce real telemetry for the Barbershop (the PRODUCER)
`scripts/emit_real_telemetry.py` runs the deterministic policy over a held-out slice of
real bars and logs a real `StepPacket` per bar (+ per-day packets, GAE advantage, and an
input-gradient attribution sidecar) to `artifacts/telemetry/<run>.jsonl`.
```
python scripts/emit_real_telemetry.py --symbol EURUSD --path data/raw/EURUSD_recent.csv --days 4
```
Flags: `--days` (real days to record), `--checkpoint` (a trained brain to load; falls back
to a fresh policy if the width doesn't match), `--run_id`. Output:
`artifacts/telemetry/<run_id>.jsonl` + `<run_id>_attribution.jsonl`.

### 4.4 Launch the Barbershop dashboard
```
python barbershop/dashboard.py     # opens http://localhost:8050 (browser auto-opens)
```
It **auto-detects** the data source: if a real run exists in `artifacts/telemetry/`, it
shows that ("Data source: REAL Quantra telemetry"); otherwise it runs on deterministic
**mock** data so a fresh checkout still works. See §5 for the screens.

### 4.5 Use the Risk Doctor (in the dashboard)
The Risk Doctor is a **local** LLM (OpenAI-compatible) you chat with in the bottom-right of
every screen. Configure it in `barbershop/config.py`:
- `DOCTOR_API_BASE` (default `http://localhost:11434/v1` = Ollama),
- `DOCTOR_MODEL` (default `llama3`; **recommend a long-context model** — the operating
  manual is ~32 KB), `DOCTOR_API_KEY`, `DOCTOR_MAX_TOKENS`.
Start a local server (e.g. `ollama run llama3`) then ask questions like *"why did the bot
fail Day 2?"*. With no server it shows a graceful "offline" message (your question is saved).

### 4.6 Train a brain
- **Locally / programmatically:** build a `TradingEnv` and run
  `quantra.learning_system.trainer.Trainer(env).train(n_updates)`; checkpoint with
  `Trainer.checkpoint(name)` → `artifacts/checkpoints/<name>.pt`.
- **Colab walk-forward (recommended for a real brain):** `colab/Quantra_Train.ipynb`
  runs the 7-seed walk-forward and promotes checkpoints by pass-rate.
- The training **reward** comes from `learning_system/reward_engine/reward.py` (layered,
  Layer-0-dominant). Locked PPO dials: γ=0.997, λ=0.97 (`trainer/gae.py`), clip/entropy
  ranges (`trainer/scheduler.py`).

### 4.7 Training wheels (operator counter-trend blocks)
Two semi-permanent masks block opening *against* a strong 30m+4H trend (CCI 5/15 SMA20-sh0;
BB 10/100 dev0.5). Toggle with `quantra.runtime.config.TRAINING_WHEELS` (default **ON**).
They're observable features (`tw_cci_block`, `tw_bb_block`) AND enforced masks; removable
by flipping the flag. (Same masks run in training and live.)

### 4.8 Per-day challenge inputs (target / risk / leverage / mode)
Use `quantra.runtime.config.make_challenge(...)` to build a validated challenge:
```
make_challenge(daily_target_pct=2.5, daily_risk_pct=4.0, ftmo_mode=True,
               leverage=100.0, stop_for_day=False, account_size=10_000.0)
```
- `ftmo_mode=True` → the 2-phase challenge (auto-flat at target → fresh tight Phase-B wall).
- `ftmo_mode=False` → a single trailing stop that runs indefinitely (side-account mode);
  `stop_for_day=True` banks and stops at target.
- `leverage` ∈ {50,100,200,500,1000,2000}; margin is the real physical cap.
Inject per day with `env.reset(challenge=make_challenge(...))`.

### 4.9 Live / demo trading
`quantra/live_bridge/` runs the live loop on MT5 (Windows). **DEMO account first.** The
`ManualHalt` kill-switch and breach-auto-flat are armed. The live session uses the **same**
laws/gates/training-wheel masks as training (discipline transfers). A one-command demo
launcher is on the queue (see `INSTRUCTIONS.md`).

### 4.10 The Barbershop Fast Loop (`colab/Quantra_Barbershop.ipynb`)

> **Status:** 🟡 new system — the notebook is a runnable **skeleton**. Loop plumbing (live
> output, checkpoint-on-interrupt, Policy Registry write with auto-naming, ngrok dashboard)
> is real; the one integration point — `barbershop_run_day()` — ships as a clearly-labelled
> `DEMO_MODE` stub (placeholder metrics) until it is wired to `TradingEnv`.

**What it is.** Barbershop *mode* (§1) made operational: *"get the haircut before going to
school."* You pick a small window of FTMO challenge days, run the policy fast, watch what
breaks, make **one** educated edit (via `OVERRIDES`, §4.12), and repeat. It can run
**before, during, or after** Full Training (§4.6) — the two modes are not sequential.

**Inputs (top of the notebook, Cell 3):**
- `POLICY_NAME` — operator label for the save (the *final* name is auto-generated, §4.11).
- `START_DATE` — where in the data the challenge window starts (e.g. `"2023-03-01"`).
- `N_DAYS` — consecutive FTMO challenge days to train on (e.g. `8`).
- `N_PASSES` — how many times the loop repeats over those `N_DAYS`.
- `CHECKPOINT_INTERVAL` — save weights + telemetry + registry every N passes.
- `RESUME_FROM` — checkpoint path to continue from, or `None` to start fresh.

**Live output (printed during training, after every pass — not at the end):**
```
Pass 3/20
  Day 1: PASS   | +2.3% | DD -1.1% | 5 trades | Gate blocks: 43%
  Day 2: FAIL   | -1.8% | DD -4.0% BREACHED | 2 trades | Gate blocks: 97%
  Day 3: PASS   | +2.6% | DD -0.8% | 7 trades | Gate blocks: 31%
  ...
  Summary: 5/8 passed | Avg P&L: +0.9% | Avg DD: -1.9% | Avg gate block rate: 61%
```
The **gate block rate per day is the #1 diagnostic signal** — it tells you whether the
policy is being *prevented* from trading (the gate-lockout gap, §7). A healthy shaping run
shows it falling as you relax gates and the policy learns to pass.

**Stop / resume.** Hit Colab's stop button any time: the loop **catches the interrupt and
saves a clean checkpoint before dying** — you never lose weights on stop. Setting
`RESUME_FROM` to a checkpoint continues from that exact point; the Policy Registry (§4.11)
tracks what was already done.

**Visualization (dual mode).** (1) **Inline** charts in the notebook (Cell 7) — Screen 1
wall + Screen 3 day replay rendered via `barbershop/figures.py`
(`training_wall_figure`, `candlestick_figure`). (2) **ngrok tunnel** (Cell 8) auto-starts
the full Dash app (`barbershop.dashboard.make_app`) in a background thread so you can open
all **5 screens + the Risk Doctor** in your browser. Telemetry is auto-emitted to
`artifacts/telemetry/<run>.jsonl` after every checkpoint so the Barbershop always has fresh
data (canonical real-bar producer: `scripts/emit_real_telemetry.py`).

**ADF gate reminder.** When you loosen `adf_p_value_threshold` in `OVERRIDES`, you are using
the stationarity gate as a **diagnostic** — letting more non-stationary bars through to see
what the policy does — *not* deleting it. Its purpose is to teach the bot to trade **both
stationary AND non-stationary** regimes; the gate stays, only its calibration changes (§4.12, §7).

### 4.11 The Policy Registry (`artifacts/policy_registry/<policy_name>/`)

Every policy gets a saved **identity** so you can answer: *"What is this policy's
perspective on how to pass the FTMO challenge?"* — what config produced it, what it saw, and
how well it passed. Written by the Barbershop notebook (Cell 6); read by you, the dashboard,
and the Risk Doctor. **Three files per policy** (full reader's guide:
`artifacts/policy_registry/README.md`):

1. **`manifest.json`** — auto-generated at save time, **never hand-written**: `policy_name`
   (auto, see below), `auto_name_basis` (the diff in plain tokens), `created`, `base_policy`
   (what it resumed from, or `null`), `data_window` (`start` + `n_days`),
   `n_passes_completed`, `state_dim`, `training_wheels`, `overrides_applied` (the full
   `OVERRIDES` dict), `compatibility_signature`.
2. **`performance.json`** — updated after every pass: `pass_history` (per pass:
   `days_passed`/`days_failed`/`avg_pnl`/`avg_dd`/`breach_count`/`avg_gate_block_rate`),
   `best_pass`, `overall_pass_rate`.
3. **`compatibility.sig`** — a hash of **state_dim + reward-layer shape + law-parameter
   fingerprint**, checked on `RESUME_FROM` (§4.12).

**Auto-name generation (NEVER hardcoded).** The folder name is **derived from the
`OVERRIDES` diff vs the baseline config** — never typed by hand or invented by an assistant.
Each meaningful change becomes a short token; tokens join with `-` behind a `v<N>` that
increments from `base_policy`. *Example:* loosen the ADF gate 2× **and** double the drawdown
penalty, resumed from `v1-gate-test` → **`v2-adf2x-ddpenalty2x`**. (A gate token like
`adf2x` means "shaped with the ADF gate relaxed 2×", **not** "the gate was removed" — §4.10.)
Monty may rename a policy afterward, but the auto-name is always generated first and recorded.

Registry contents are **git-ignored** (large + run-specific); only the `README.md` is
committed. Persist real checkpoints/manifests to **Google Drive** (auto-saved every
`CHECKPOINT_INTERVAL`). See §4.13 for why the repo is not your backup.

### 4.12 Runtime Override System (the `OVERRIDES` dict)

After a Barbershop run you decide what to change — and you make that change **inside the
notebook**, not by editing source files and not by asking an LLM to patch code every time. A
runtime `OVERRIDES` dict (Cell 3) is **injected into the env/training loop at launch without
touching `quantra/runtime/config.py` or `laws.py`.** The exact dict used is saved to the
Policy Registry (§4.11) for every run, so you always know which configuration produced which
result. Tuneable knobs:

| Knob | Default | What it does (relative to passing) |
|---|---|---|
| `adf_p_value_threshold` | `0.05` | Loosen the **stationarity** gate (let more non-stationary bars through). |
| `atr_min_multiplier` | `1.0` | Loosen the **ATR-liquidity** gate. |
| `spread_max_pips` | `2.0` | Tighten/loosen the **spread** gate. |
| `reward_l1_pnl_weight` | `1.0` | Layer-0 net-PnL weight — **NEVER set to 0** (see below). |
| `reward_l2_drawdown_penalty` | `1.0`→`2.0` | Drawdown-penalty multiplier (the L3 pain-zone ramp). |
| `reward_l3_hold_penalty` | `0.0`→`0.5` | Penalty for holding too long (anti-stagnation). |
| `reward_l4_gate_bonus` | `0.0`→`0.1` | Small bonus for **not** being blocked by the gate. |
| `training_wheels` | `True` | Enforce (`True`) / remove (`False`) the counter-trend OPEN masks. |

**ADF note (again, by rule):** loosening `adf_p_value_threshold` is a **diagnostic tool** to
unblock training and observe the policy with more freedom — then you interpret the result to
decide whether the *calibration* needs a permanent change. The gate stays in the
architecture; you are tuning **both stationary AND non-stationary** trading, not disabling
stationarity filtering.

**What triggers a `CompatibilityError`.** If an override changes **`STATE_DIM`** or the
**reward shape** in a way that breaks an existing checkpoint's assumptions, resuming raises a
**`CompatibilityError`** with a plain-English reason for *why* the policy must start fresh —
and it **saves the old checkpoint first**. Old policies are **never deleted automatically**.
Safe to resume across: tuning gate thresholds, tuning reward *weights* (so long as Layer-0
dominance holds — `reward_l1_pnl_weight` ≠ 0), toggling `training_wheels`.

> 🔴 **Locked-parameter guard.** `reward_l1_pnl_weight = 0` would break **Layer-0 dominance**,
> a locked invariant (§6). Likewise γ/λ, the 9 laws / 3 gates params, the action-mask logic,
> and `STATE_DIM` are locked: changing them is a **proposed amendment requiring Monty's
> sign-off**, not a routine override.

### 4.13 Repo Safety Protocol (primary vs fallback)

The project lives in **two GitHub repos**, and you must always know which one you're in:

| Role | Repo | Rule |
|---|---|---|
| **ACTIVE WORK** | `github.com/monty313/RL-model-trading-bot-ppo-mlp_Claude-` | All code changes, new files, and commits happen here. Every session works here. |
| **FALLBACK / SAFE RESTORE** | `github.com/monty313/final-rl-model-6_13` | **Never edited.** It is the last-known-clean snapshot you revert to if the working repo breaks beyond repair. |

**Protocol.**
1. Do all work in the **active** repo (this one).
2. **Before any major code change, push the current working state** so the fallback can be
   re-synced from a known-good commit — the fallback must stay a clean restore point.
3. If the active repo breaks beyond repair: clone **`final-rl-model-6_13`** to restore, then
   re-apply only the verified-good work on top.
4. **Never delete or overwrite the fallback repo**, and never confuse the two — this protocol
   exists so no session (human or LLM) ever edits the wrong one.

### 4.14 Colab GPU Setup (80% target · cache-once · cell order)

Monty runs **Colab Pro**; training should use **~80% of whatever GPU/CPU instance is
assigned**. The hardware auto-optimizer (`quantra/runtime/optimizer.py`, `plan()` /
`print_report()`) already does this — the Barbershop notebook **calls it at startup**
(Cell 1, via `from quantra.runtime import plan, print_report, UtilizationMonitor`). It races
CPU vs GPU on the real four-head workload, picks the faster device (preferring CPU on a
near-tie to save GPU hours), and sizes parallelism to ~80%.

**Critical performance rule — cache once, then live on GPU.** The data pipeline (CSV parse →
feature build → memmap cache) runs **once at startup** and the cache is reused forever; every
second of CPU setup is a second you're not seeing results. The notebook's **8-cell order**
enforces this:

| Cell | Does | Skips if… |
|---|---|---|
| 1 | Clone active repo · mount Drive · install deps · race hardware (~80%) | deps sentinel exists |
| 2 | Load data + build features → memmap cache | cache exists |
| 3 | Set `INPUTS` + `OVERRIDES` (**operator edits here**) | — |
| 4 | Load or init policy (resume or fresh) + compatibility check | — |
| 5 | **Training loop** (GPU-bound · live output · clean stop-on-interrupt) | — |
| 6 | Auto-emit telemetry → write Policy Registry entry (auto-named) | — |
| 7 | Inline Barbershop charts (`figures.py`) | — |
| 8 | Start ngrok tunnel → full dashboard | — |

Checkpoints auto-save to **Drive** every `CHECKPOINT_INTERVAL` passes, so a Colab disconnect
resumes from the last Drive checkpoint (not from zero).

---

## 5. The Barbershop — 5 screens + Risk Doctor

A read-only post-training diagnostic tool. It **never** writes outside `logs/` and never
changes training, rewards, or the policy.

1. **Screen 1 — Training Wall.** Pass-rate over training iterations; green rising / yellow
   flat / red falling; an 80% "Consistent Pass Zone" line; a plateau banner. *(On a real
   run with no logged pass-rate series, it shows an honest "Demo curve" label.)*
2. **Screen 2 — 4-Day Scoreboard.** One card per training day (regime, P&L%, PASS/FAIL,
   DD status Safe/Warning/Breached, trades), sorted worst-first. Click a card → Screen 3.
3. **Screen 3 — Day Replay.** Candlesticks + BB/SMA overlays + clickable trade markers +
   profit/loss shading + a "DD WALL BREACHED" line. Timeframe buttons `[1m][5m][30m][4H]`
   set the context window. On **1m only**: an **advantage strip** (real GAE) and an
   **indicator heatmap** (real feature names). Click a trade → Screen 4.
4. **Screen 4 — Trade Autopsy.** Three columns: LEFT = what the bot SAW (state bars),
   MIDDLE = action probabilities (chosen = gold border) + masked/legal label, RIGHT =
   **input×gradient attribution** (real, labelled *not Shapley*). Panels grey out honestly
   if a run lacks the data.
5. **Screen 5 — Pattern Finder.** Auto-scans losing trades, surfaces the top patterns in
   plain English, with **APPLY** (export to `logs/suggested_rules.json`) / IGNORE / MODIFY.
6. **Risk Doctor (chat box, all screens).** A local LLM grounded in
   `docs/MLP_INTERPRETABILITY_LAYER.md` (loaded every call, condensed to fit context). It
   answers in 6 sections (📍 looking at / 🔍 see / 🎯 means for passing / ✅ do next /
   ❌ don't / 📊 confidence), refuses live-trade questions, never fabricates (says
   "insufficient evidence"), logs to `logs/doctor_diagnoses.jsonl`, and can export an
   approved prescription to `logs/suggested_rules.json`. The **Full Diagnosis** button
   runs the doc's diagnostic template on the selected day.

---

## 6. Key configuration (where the knobs live)

`quantra/runtime/config.py`:
- `SYMBOLS` = EURUSD, XAUUSD, GBPUSD, US30; per-symbol `POINT_SIZE`, `CONTRACT_SIZE`,
  `SLIPPAGE_POINTS`, `ASSET_CLASS`, Drive file IDs.
- `ChallengeConfig` defaults + `make_challenge(...)`; `FTMO_ON_BOUNDS` / `FTMO_OFF_BOUNDS`.
- `TRAINING_WHEELS` (default True), `INCLUDE_RAW_INPUTS` (default True; STATE_DIM 203/185).
- `RiskConfig` (stop_atr_mult, lot_step, min/max lot, per-trade risk frac),
  `CostConfig` ($5 RT/lot forex), `HardwareConfig` (≈80% util, CPU-first).
- Paths: `DATA_DIR`, `ARTIFACT_DIR`, `CHECKPOINT_DIR`, `TELEMETRY_DIR`, `REPORT_DIR`.

`barbershop/config.py`: dashboard paths, `DOCTOR_API_BASE/MODEL/KEY/MAX_TOKENS`,
`DOCTOR_MANUAL_MAX_CHARS`, thresholds (target/wall/DD-buffer colours), timeframe windows.

🔴 **Locked (need sign-off to change):** γ/λ in `gae.py`; the 9 laws/3 gates params in
`indicators.py`/`laws.py`; Layer-0 dominance + pain ramp + QUAD ceiling in `reward.py`;
the action-mask logic in `engine.py`; `STATE_DIM`/schema (re-pin snapshot via
`tools/snapshot.py --update`).

---

## 7. Honest current state (what works, what's a gap)

**Verified correct (audited this project):**
- The RL learning math — actor/critic, GAE, PPO clip objective, rollout buffer, the
  training loop, and the layered reward — is mathematically correct (read + numerical
  proofs + independent audit, 0 defects).
- The env account physics — trailing wall, breach latch, target, daily reset, costs,
  no-overshoot sizing, margin, PnL decomposition — is arithmetically correct.
- The Barbershop: 37 tests pass; real telemetry flows end-to-end; honesty guards in place.

**Known gaps / honest caveats (state these in every overview):**
1. **GATE LOCKOUT — the #1 active work item.** The gates (chiefly the *stationarity* gate,
   open ~5.6%, + ATR-liquidity) block **~98.7%** of trade opportunities on real EURUSD. This
   is the binding blocker to passing — a **calibration** issue, not an arithmetic bug. Note
   the ADF (stationarity) gate's purpose is to teach the bot to trade **both stationary AND
   non-stationary** regimes by controlling when it is enforced vs relaxed; the fix is to
   recalibrate it (use `OVERRIDES` diagnostically, §4.12), **not** to delete it. Watch the
   per-day **gate block rate** in the Barbershop Fast Loop (§4.10) as the primary signal.
2. **No real trained model yet.** The policy has only been trained on **synthetic data**; it
   does not transfer to real bars (≈9 trades / 5,565 bars). A real **Barbershop run** is the
   first step (then Full Training walk-forward).
3. **One wall, not two.** The sim models a single daily-re-anchored trailing wall; real FTMO
   has **TWO** limits (max daily loss from day-start AND a permanent max-overall drawdown). A
   sim pass does **not** yet guarantee a live-legal pass.
4. **Screen 1 demo curve.** Barbershop **Screen 1** shows a labelled **demo curve** until the
   trainer logs a real pass-rate series.
5. **input×gradient, not SHAP.** The trade-autopsy attribution (Screen 4) is
   **input×gradient**, honestly labelled — not true Shapley values.

---

## 8. Glossary (the terms the assistant should know)

- **Actor / policy:** the network that outputs the action probability distribution.
- **Critic / value V(s):** estimates expected return from a state.
- **Trajectory:** the (state, action, reward) sequence collected per rollout.
- **Rewards-to-go:** discounted sum of future rewards (γ=0.997).
- **Advantage A = RTG − V(s):** how much better an action was than the critic expected
  (positive → reinforce, negative → discourage).
- **GAE:** Generalized Advantage Estimation (λ=0.97) — smooths the advantage.
- **Probability ratio r = π_new/π_old; PPO clip (ε):** caps each update to a small,
  stable step.
- **Laws (9) / Gates (3):** the legal-action spine — directional laws ban the wrong
  direction; gates (ATR-liquidity, spread, stationarity) ban new opens in bad conditions.
- **ADF / stationarity gate:** the gate that bans new opens when bars look non-stationary.
  Its purpose is **not** to be removed — it teaches the bot to trade **both stationary AND
  non-stationary** regimes by controlling *when* it is enforced vs relaxed. Loosening its
  p-value threshold is a **diagnostic** (unblock + observe), after which you decide if the
  **calibration** needs a permanent change. The gate stays; only calibration moves (§4.12, §7).
- **Training wheels:** operator counter-trend OPEN blocks (CCI/BB on 30m+4H).
- **`OVERRIDES`:** the runtime override dict (§4.12) injected at launch — tune gates/reward
  weights/training-wheels without editing `config.py` or `laws.py`; saved per run to the
  Policy Registry.
- **Policy Registry:** `artifacts/policy_registry/<name>/` — a policy's saved identity
  (auto-named manifest + performance + compatibility signature, §4.11).
- **Barbershop mode / Full Training mode:** the two operating modes (§1) — fast
  diagnose-and-shape vs long walk-forward generalization; entered freely, not sequential.
- **Gate block rate:** the per-day fraction of bars where the gates blocked a new open —
  the **#1 diagnostic** in the Barbershop Fast Loop (§4.10).
- **The wall / breach:** the trailing drawdown limit; touching it = breach = challenge failed.
- **Target:** the daily profit goal (+2.5%).
- **Barbershop:** the read-only diagnostics dashboard. **Risk Doctor:** the LLM that
  explains the telemetry, grounded in `docs/MLP_INTERPRETABILITY_LAYER.md`.
- **Telemetry:** the per-decision JSONL log (`artifacts/telemetry/<run>.jsonl`) the
  Barbershop reads.

---

*For deeper specs see `docs/` (THE_TRADING_CODE, STATE_VECTOR, REWARD_DESIGN, PPO_ENGINE,
MLP_INTERPRETABILITY_LAYER). For the Barbershop fixes log see `barbershop/REMEDIATION_PLAN.md`.
For cross-file couplings before any refactor see `COUPLINGS.md`.*

---

## Update Log (IRAC)

- **[2026-06-18]** Added the Barbershop fast-loop system to the guide (§4.10–§4.14), rewrote
  §1/§7 for the two-modes + repo-safety + five-known-gaps model, and applied the
  `real_backtest.py --path` fix note (§4.2).
  - **I:** The operator needed a fast, in-notebook diagnose-and-shape loop (Barbershop mode)
    with a saved policy identity, a no-code-edit override system, an unambiguous primary/
    fallback repo rule, and an honest Colab-performance pattern — none of which were
    documented, and the repo roles in the brief contradicted the operator's correction.
  - **R:** Operator brief Sections 1–11 + the corrected repo roles (ACTIVE =
    `RL-model-trading-bot-ppo-mlp_Claude-`, FALLBACK = `final-rl-model-6_13`) + standing
    rules (FTMO framing, ADF-purpose note everywhere, 5 gaps in every overview, auto-names
    from the `OVERRIDES` diff, no invented files).
  - **A:** Wrote §4.10 (Fast Loop), §4.11 (Policy Registry + auto-naming), §4.12 (Runtime
    Override System + `CompatibilityError`), §4.13 (Repo Safety Protocol), §4.14 (Colab GPU
    Setup); split §7 into 5 gaps with gate-lockout as #1; added the ADF stationary/
    non-stationary note wherever the gate appears; created
    `colab/Quantra_Barbershop.ipynb` (8-cell skeleton) and
    `artifacts/policy_registry/README.md`; fixed `scripts/real_backtest.py` `--path`.
  - **C:** Monty and any LLM can now run, save, resume, and reason about a Barbershop policy
    against the single mission — repeatedly passing the FTMO challenge — and always know
    which repo is safe to edit. New systems are clearly marked as a runnable skeleton, not
    claimed as finished, preserving the project's honesty contract.
