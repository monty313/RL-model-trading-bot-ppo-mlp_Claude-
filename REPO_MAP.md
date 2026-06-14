# REPO_MAP — SOW J1 tier tree ↔ importable Python packages

The SOW Section 3 tree uses **digit-prefixed directory names** (`00_constitution/`,
`04_learning_system/`, …). Python identifiers cannot begin with a digit, and the
SOW's own Section 12 invocation (`python -m quantra.live_runner`) requires valid,
importable names. So each SOW tier is exposed under an **importable alias** while
its tier number is preserved here and as a `# SOW tier:` tag in every package's
`__init__.py`. **No architecture changed — only directory names became valid
identifiers** (approved deviation, plan sign-off 2026-06-13, per SOW R5).

| SOW tier (Section 3)        | Importable package                         | Contents |
|-----------------------------|--------------------------------------------|----------|
| `00_constitution/`          | `quantra.constitution`                     | mission, constitutional_rules, safety_boundaries |
| `01_locked_core/` 🔴        | `quantra.locked_core`                      | `laws`, `risk_manager`, `cost_layer`, `platform_adapter` |
| `02_ftmo_passing/`          | `quantra.ftmo_passing`                     | scoreboard, challenge_state, episode_rule, `validation` |
| `03_market_pipeline/`       | `quantra.market_pipeline`                  | `data_loader`, `resampler`, `feature_builder`, `law_mask_engine` |
| `04_learning_system/`       | `quantra.learning_system`                  | `ppo_agent`, `rollout_buffer`, `reward_engine`, `trainer`, `curriculum_manager`, `hpo` |
| `05_diagnostics/`           | `quantra.diagnostics`                      | `telemetry_logger`, `mlp_interpreter`, `llm_risk_doctor`, `failure_atlas` |
| `06_live_bridge/`           | `quantra.live_bridge`                      | live_runner, execution_adapter, manual_halt |
| `99_docs/`                  | `docs/`                                     | the 10 blueprint files (binding) |
| — (cross-cutting) [C]       | `quantra.runtime`                          | hardware optimizer + runtime config |
| — (the env)                 | `quantra.env`                              | the gym environment (Env module) |

## Where the binding docs live
`docs/` holds the in-tree copy of the blueprint (SOW §3 `99_docs/`):

```
docs/
├── 00_START_HERE.md
├── OPEN_QUESTIONS.md
├── THE_TRADING_CODE.md
├── STATE_VECTOR.md
├── REWARD_DESIGN.md
├── PPO_ENGINE.md
├── MLP_INTERPRETABILITY_LAYER.md   ← the LLM Risk Doctor's binding rulebook
├── SCOPE_OF_WORK.md
├── SOW_2_BUILD_SPEC.md
└── wtf_are_you_talking_about.md
```

## Build status (milestones, SOW §13–14)
- [x] **M0** — repo skeleton, docs in-tree, runtime hardware optimizer, Colab notebook
- [x] **M1** — data pipeline (MT5 loader + lookahead-safe resampler)
- [x] **M2** — FeatureBuilder + canonical state vector (offline precompute + memmap).
  **176 scalars** = 146 normalized + a 30-feature operator-added RAW SMA/CCI block
  (`market_raw`, toggle `config.INCLUDE_RAW_INPUTS`; risks/safeguards in
  [`feature_builder/RAW_INPUTS.md`](quantra/market_pipeline/feature_builder/RAW_INPUTS.md))
- [x] **Change-impact tracker** — snapshot guard + AST analyzer ([`CHANGE_IMPACT.md`](CHANGE_IMPACT.md))
- [x] **M3** — LawMask: 9 laws + 3 gates, two enforcement modes, −1e9 masking
  (`quantra/locked_core/laws/` + `quantra/market_pipeline/law_mask_engine/`); state
  vector → **179** (added 3 gate ingredients: spread×2, ADF).
- [x] **M4** — RiskManager + CostLayer + Env (4 symbols true-sequential, 5 slots,
  shared account, real FTMO costs, Phase-A wall) + **B5 no-overshoot invariant**
  (`quantra/locked_core/{risk_manager,cost_layer}/`, `quantra/ftmo_passing/challenge_state.py`,
  `quantra/env/trading_env.py`).
- [ ] M5 PPOAgent · M6 RewardEngine · M7 curriculum+episode · M8 trainer · M9 telemetry ·
  M10 interpreter · M11 risk doctor · M12 validation · M13 HPO · M14 live bridge · M15 acceptance

**Tests:** one master suite — `tests/test_ftmo_master_suite.py` (run `pytest`). All future
tests append there. **Every file carries an IRAC update log** — see
[`quantra/constitution/update_rules.md`](quantra/constitution/update_rules.md).


---

## Update Log (IRAC) — standing rule since 2026-06-13
*Every change appends a dated IRAC entry; the **Conclusion** always states why it
makes the bot pass FTMO more consistently. Rule: [quantra/constitution/update_rules.md](quantra/constitution/update_rules.md).*

- **[2026-06-13]** Tier map + build status reflect M1 done and the new rules.
  - **I:** The map showed only M0 done and didn't reference the IRAC/master-suite rules.
  - **R:** Operator IRAC rule (2026-06-13).
  - **A:** Marked M0+M1 complete; noted the single master suite and the IRAC update-log rule.
  - **C:** An accurate map makes future milestones land in the right place, all aligned to passing.
- **[2026-06-13]** M2 complete — FeatureBuilder + 146-scalar state vector.
  - **I:** The map needed to reflect M2 (perception layer) done.
  - **R:** SOW §13 implementation order; STATE_VECTOR.md schema.
  - **A:** Marked M2 complete; the canonical schema lives in `quantra/market_pipeline/feature_builder/schema.py`.
  - **C:** Contributors and the LLM can see the observation layer is locked + verified, so M3 (laws) builds on a faithful, asserted world — keeping the path to consistent passing on track.
- **[2026-06-13]** Operator raw-input block (state vector 146→176) + change-impact tracker.
  - **I:** Operator wants raw SMA/CCI added; and a guard so future obs changes can't silently degrade pass-rate.
  - **R:** Operator directives; raw block overrides the no-raw-price rule (RAW_INPUTS.md); tracker per CHANGE_IMPACT.md.
  - **A:** Added `market_raw` (30, unclipped, flagged, toggleable); built snapshot guard + AST analyzer + pre-commit; 28 master-suite tests green.
  - **C:** The policy gains the requested signal with a clean off-ramp, and the observation can't change by accident — both protect consistent passing.
- **[2026-06-13]** M3 complete — LawMask (9 laws + 3 gates) + 3 gate ingredients (dim 179).
  - **I:** The bot had no spine defining legal directions, and 2 gates lacked observable ingredients.
  - **R:** THE_TRADING_CODE.md (exact laws/params) + SOW C5 (−1e9) + law-ingredient coverage rule.
  - **A:** Built `laws.py` (12 states) + `law_mask_engine` (masks, both modes); added spread/ADF ingredients; re-pinned snapshot to 179. 40 tests green.
  - **C:** Breach-bound directions are now mechanically blocked before the policy acts — the foundation of not breaching, hence of consistent passing.
- **[2026-06-13]** M4 complete — Env + RiskManager + CostLayer + B5 invariant.
  - **I:** The challenge physics needed to exist and be proven overshoot-proof + faithful.
  - **R:** SOW B5/H3/§10.5/§2.4/§2.7.
  - **A:** Built RiskManager (round-down, no overshoot), CostLayer ($5 RT forex-only), ChallengeState (buffer+wall), TradingEnv (4-symbol true-sequential, 5 slots, shared account); 12 tests incl. 2000-case sizing fuzz + B5. 52 tests green.
  - **C:** The bot trains on faithful physics where collective overshoot is impossible by construction — so learned behaviour transfers to passing real challenges.
