# RL-model-trading-bot-ppo-mlp_Claude- — IRAC handoff (2026-10-17)

## Issue
Quantra: single PPO 3x256 MLP to repeatedly pass FTMO challenges, with LLM 'Risk Doctor' (barbershop) and Colab training.
Central question: No trained policy with pass-rate evidence; stopped mid-UI/doc work.

## Rule
docs/SOW_2_BUILD_SPEC.md (binding), docs/MLP_INTERPRETABILITY_LAYER.md; scoreboard: pass rate > breaches > target consistency > max DD. Doctrine: scalping-only; 5m CCI 30/100 burst entry; 1R=0.25% equity; max 4 concurrent; daily floor -3R; pyramid winners OK, add-to-losers banned.

## Application
47 commits, last d68e850 2026-06-16 (docs). WI-4..WI-8 (ab642af..1704de1) were UI/Doctor/contract work; Screen-1 'demo vs real' = mock data still present. quantra.runtime CPU/GPU racer. 6 TODO/FIXME. Open PR #2 (Barbershop docs). Data on Google Drive (not verifiable).

## Conclusion
Status: abandoned. Stale since 2026-06-16. Blockers: mock data in UI, no training result, superseded by Camillion.
Next: 1) Close/merge PR #2. 2) Archive in favor of Camillion_Claude_RL_model.
