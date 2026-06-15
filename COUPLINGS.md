# COUPLINGS — change-one-place-must-change-another map

This is the authoritative list of **cross-file / cross-location couplings**: values,
shapes, names, and orderings that are *defined in one place but depended on in
others*. If you change the definition, you MUST update every consumer or the bot
breaks (often silently). Each shared symbol carries an inline `# COUPLING:` comment
at its definition site pointing here.

Much of cluster **C1** is also *enforced automatically* by the change-impact tracker
([`CHANGE_IMPACT.md`](CHANGE_IMPACT.md)): the snapshot guard fails the test suite if
the observation layout drifts. The rest is enforced by the single master suite
(`pytest`) — run it after any change below.

---

## C1 — Observation layout (the biggest coupling)
**Defined in** `quantra/market_pipeline/feature_builder/schema.py` (`STATE_DIM`,
`FEATURE_NAMES`, `PRECOMPUTED_NAMES`, `block_spans`, `RAW_FEATURE_NAMES`,
`EXPECTED_WIDTHS`).
**If you add/remove/reorder a feature or change a block width, also update:**
- `config.py` → `RuntimeConfig.nominal_state_dim` (must equal `STATE_DIM`).
- `tests/snapshots/state_vector.json` → `python tools/snapshot.py --update` (guard fails otherwise).
- `builder.py` → `_compute_tf_features` must emit exactly the schema names, in `PRECOMPUTED_NAMES` order; clip excludes `RAW_FEATURE_NAMES`.
- `laws.py` → `_IDX` (column index by name) + every law's feature names.
- `env/trading_env.py` → `_COL` (column index by name, used by the reward proxy).
- `learning_system/trainer/scheduler.py` → `_COL` (G8 reads `ssma_align_*`).
- `live_bridge/live_session.py` → rebuilds the obs identically.
- `learning_system/curriculum_manager/curriculum.py` → `_EARLY_MASK_1M` names + `feature_mask` (size `STATE_DIM`).
- `diagnostics/telemetry_logger` → logs `FEATURE_NAMES` (grouped block names, the data contract).
- The PPO agent reads `STATE_DIM` dynamically (no hardcode) — verify on any change.
**Why:** the observation is the policy's entire world; a silent shape change invalidates
normalization, the agent input dim, telemetry labels, and every checkpoint.

## C2 — Direction action encoding {HOLD=0, OPEN_LONG=1, OPEN_SHORT=2, CLOSE=3}, `N_DIR_ACTIONS=4`
**Defined in** `quantra/market_pipeline/law_mask_engine/engine.py`.
**Consumers:** `ppo_agent/agent.py` (direction head = 4 logits + OPEN/CLOSE gating),
`ppo_agent/device.py`-equivalent `runtime/device.py` (`RepresentativePolicy` mirrors 4),
`env/trading_env.py` (`_apply_action`), `live_bridge/live_session.py`, `live_runner.py`.
**Why:** the integer meaning of each action is assumed everywhere; reorder it and the
agent opens when it means to close.

## C3 — `N_SLOTS = 5` (trade slots per symbol)
**Defined in** `quantra/market_pipeline/feature_builder/schema.py`.
**Consumers:** schema `trade` block (7 × 5 = 35), `ppo_agent/agent.py` (pointer head = 5)
+ `runtime/device.py` (`RepresentativePolicy` pointer = 5), `env/trading_env.py`,
`live_bridge/execution_adapter.py`, `live_bridge/live_session.py`,
`law_mask_engine/engine.py` (`build_pointer_mask`).
**Why:** the pointer head width, the trade-block width, and the slot arrays must all agree.

## C4 — Law/gate block order (the 12 names)
**Defined in** `quantra/locked_core/laws/laws.py` (`LAW_NAMES`) **and mirrored in**
`schema.py::_law_names` (must be identical order).
**Consumers:** `compute_law_states` returns columns in this order; `law_mask_engine`
indexes gates as the LAST 3 (`_GATE_IDX`); `curriculum_manager` references law names.
**Why:** the mask engine slices `law_states[:9]` (directional) and `[9:]` (gates) by position.

## C5 — Per-symbol broker/cost/risk dicts (must cover every symbol in `SYMBOLS`)
**Defined in** `quantra/runtime/config.py` (`SYMBOLS`, `ASSET_CLASS`, `POINT_SIZE`,
`CONTRACT_SIZE`, `SLIPPAGE_POINTS`, `DRIVE_FILE_IDS`, `DRIVE_FILENAMES`).
**Consumers:** `cost_layer/costs.py`, `risk_manager/risk.py`, `env/trading_env.py`,
`ftmo_passing/challenge_state.py`, `live_bridge/live_session.py`, `data_loader`.
**Why:** a symbol added to `SYMBOLS` without an entry in each dict → wrong sizing/cost or a KeyError.

## C6 — Locked PPO patience dials γ=0.997, λ=0.97
**Defined in** `quantra/learning_system/trainer/gae.py` (`GAMMA`, `LAMBDA`).
**Consumers:** the trainer; the HPO (`learning_system/hpo`) must NEVER tune them.
**Why:** 🔴 hand-locked "math of patience"; off-limits to Optuna (SOW G2/G6).

## C7 — CCI feature names (raw) `cci{p}_{tf}` + `cci{p}_sma_{tf}` [2026-06-13 operator: raw]
**Defined in** `schema.py::_market_names` (+ `RAW_FEATURE_NAMES`).
**Consumers:** `builder.py` (emits them), `laws.py` (ST2/T2/PB2 read value vs sma + ±100),
`curriculum.py` (`_EARLY_MASK_1M` masks `cci{p}_1m`).
**Why:** rename in one place and the laws read a missing column (KeyError) or mask nothing.

## C8 — Telemetry data contract / schema version
**Defined in** `quantra/diagnostics/telemetry_logger`.
**Consumers:** `mlp_interpreter` (reads the packets to build the 7 visuals),
`llm_risk_doctor` (reads the per-step fields). Field renames must propagate to both.
**Why:** the interpreter + Risk Doctor reason from exact field names (the data contract).

## C9 — Training-wheel feature names + block flags `tw_*` [2026-06-15 operator]
**Defined in** `indicators.py` (`WHEEL_CCI_PERIODS/WHEEL_CCI_SMA/WHEEL_CCI_SHIFT`,
`WHEEL_BB_FAST/SLOW/DEV`, `WHEEL_TFS`) + `schema.py::_market_names` (the `tw_cci{5,15}*`,
`tw_bb{10,100}_{up,lo}_*` ingredients + the `tw_cci_block` / `tw_bb_block` flags) +
`config.TRAINING_WHEELS` (the on/off toggle).
**Consumers:** `builder.py` (emits the ingredients + computes the 2 flags),
`law_mask_engine.build_direction_mask` (`_apply_training_wheels` bans counter-trend opens
from the flags), `env/trading_env.py::_wheel_states` + `live_bridge/live_session.py`
(read the flags by name and pass them to the mask — train/live parity).
**Why:** these are a SEPARATE operator override (semi-permanent counter-trend "training
wheels"), isolated from the locked 9 laws — they DO read 4H by explicit operator decision,
unlike the laws. Rename a `tw_*` name in one place and the mask reads a missing column or
bans nothing; flip `TRAINING_WHEELS` and enforcement lifts everywhere at once. The flags
are observation features too (the "acts"), so a rename also shifts the pinned STATE_DIM.

---

## Update Log (IRAC) — standing rule since 2026-06-13
- **[2026-06-13]** Created the cross-file coupling map.
  - **I:** Shared values/shapes/orderings were depended on across many files with only
    scattered awareness; a change in one place could silently break another.
  - **R:** Operator request to identify cross-location couplings + the change-impact discipline.
  - **A:** Enumerated 8 coupling clusters with definition sites + consumers + why; added inline
    `# COUPLING:` notes at the definition sites pointing here.
  - **C:** Any future edit to a shared constant now has a visible blast-radius checklist, so the
    bot's interdependent pieces stay in sync — preventing the silent bugs that erode a pass streak.
