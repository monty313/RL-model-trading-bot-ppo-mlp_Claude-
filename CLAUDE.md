# CLAUDE.md — read me first (auto-loaded each session)

This file is auto-read by Claude Code at the start of every session. It exists so a NEW
session can pick up our momentum without the previous chat's memory. **For the live state
of the current work, read [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) — that is
the running thread.** This file is the durable orientation.

## What this project is
One universal **PPO** policy (tiny 3×256 MLP, ~207-feature observation) trained to
**repeatedly PASS FTMO-style challenges** — **+2.5% daily target without breaching the
4% trailing daily drawdown** — *not* to maximize PnL. An episode is N trading days on ONE
continuous account (the balance carries forward; a day that hits the wall is locked out
until midnight; the episode ends only when the days run out or the account is blown).

## Operator priority order (drives every reward/design decision)
1. **PASS the challenge consistently.**
2. **Make +2.5% of initial balance WITHOUT breaching the trailing DD.**
3. Everything else.

## Working branch
`claude/focused-faraday-if1ue7` — develop, commit, and push here (the Colab notebooks
clone this branch). Do not push elsewhere without explicit permission.

## How to run
- **Tests:** `python -m pytest tests/` (currently **196 passed, 1 skipped**). Run before every commit.
- **Train:** `colab/Quantra_Train.ipynb` — Cell 5 (hyperparameters) → 5a (build env) → 5b
  (the loop; streams ONLY the per-day scoreboard). Real EURUSD bars come from the user's
  Google Drive; if Drive isn't mounted it falls back to LOUDLY-labelled synthetic (smoke only).
- **Evaluate / Barbershop:** `colab/Quantra_Barbershop.ipynb` (real OOS per-day scoreboard).

## Conventions (do not violate without operator sign-off)
- **🔴 LOCKED** items (γ=0.997 / λ=0.97, the law set, masks, sizing, the 4% wall, E8
  Layer-0 dominance) change ONLY with the operator's explicit approval. Propose, don't apply.
- **Notebook = the operator control panel.** Experiment knobs live in the notebook's
  HYPERPARAMETERS cell and flow through `OVERRIDES` (which auto-names + reproduces the
  policy). **`quantra/runtime/config.py` defaults are the blessed, test-asserted baseline
  — leave them alone** unless promoting a proven change. Every experiment must be one-line
  revertible.
- **IRAC standing rule:** every code edit appends a dated I/R/A/C entry to that file's
  UPDATE LOG, and documents the cross-file COUPLING it depends on (both directions). Docs
  track code.
- Commit messages: clear + descriptive. Only commit/push when the work is complete and
  tests pass.

## Where things live
- Blueprint / design docs: `docs/` (start at `docs/00_START_HERE.md`; reward math in
  `docs/REWARD_DESIGN.md`; laws in `docs/THE_TRADING_CODE.md`).
- Reward engine: `quantra/learning_system/reward_engine/reward.py`
- PPO trainer / GAE / loss: `quantra/learning_system/trainer/` + `quantra/learning_system/ppo_agent/loss.py`
- Env (physics, masks, wall, exits): `quantra/env/trading_env.py`
- Action masks / laws: `quantra/market_pipeline/law_mask_engine/engine.py`, `quantra/locked_core/laws/laws.py`
- Per-day scoreboard used in training + eval: `quantra/learning_system/barbershop_runner.py`
- Cross-file coupling index: `COUPLINGS.md`
