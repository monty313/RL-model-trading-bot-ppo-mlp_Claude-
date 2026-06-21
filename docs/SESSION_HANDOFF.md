# SESSION HANDOFF — live work thread (continue our momentum)

**Updated: 2026-06-21 · Branch: `claude/focused-faraday-if1ue7`**

A new Claude Code session starts fresh (it clones the repo, it does NOT inherit the prior
chat). This file is the running thread so the next session continues exactly where we left
off. Read `CLAUDE.md` first for durable orientation; this is the current state + next steps.

---

## ▶️ START THE NEXT SESSION — copy-paste this as your first message
> Open a new Claude Code session on this repo, branch `claude/focused-faraday-if1ue7`, then paste:

```
Read CLAUDE.md and docs/SESSION_HANDOFF.md in full, then give me a 5-line recap of where
we are and what's next — don't change anything yet. Context: a PPO bot to repeatedly PASS
FTMO-style challenges (+2.5%/day without breaching the 4% trailing DD). Priority: (1) pass
consistently, (2) +2.5% without breaching trailing DD, (3) everything else. We just added
return normalization and are about to run training on real EURUSD to check the whipsaw is
gone and breaches trend down. I'll paste scoreboards as they stream; help me read them and
decide the next lever (next candidate: the drawdown-pain weight for green-day peak-giveback
breaches). Respect 🔴-locked items, keep every change one-line revertible in the notebook.
```

---

## The goal (north star)
A PPO policy that **repeatedly PASSES** FTMO-style challenges: **+2.5%/day without breaching
the 4% trailing daily drawdown**, day after day, on one continuous account. Priority order:
**(1) pass consistently → (2) +2.5% without breaching trailing DD → (3) everything else.**

## The arc so far (what we found → did → why) — newest last
1. **Bot was converging to "always HOLD" (did nothing).** Root cause: the notebook had the
   **CCI-regime open-gate ON** (`CCI_REGIME_GATE=True`). On real EURUSD the gate's "all four
   1m+4H CCIs agree" condition is true on <1% of bars, so it masked ~99% of opens → the
   policy only ever saw HOLD → collapsed to HOLD (ent≈0, miss≈0, 0 trades/day).
   **Fix:** gate OFF (commit `5162f14`). The CCI regime stays in the *observation* so the
   policy can learn it as a soft feature, never a hard gate. **Verdict: experiment failed,
   keep OFF.**
2. **Bot then traded again** (entropy alive, trades on every day) but **breached every day**
   and won ~17% — and the health trace whipsawed (value_loss → ~1.0, entropy crash/recover
   ~upd 1750, kl climbing to 0.2, clip to 0.45).
3. **Diagnosed the whipsaw = reward SCALE.** A perfect +2.5% day's net-PnL reward sums to only
   **0.025**, but the end-of-day EVENT rewards (`failed_day_penalty`, `fast_pass_bonus`) were
   **5.0** → a once-a-day spike ~200× the entire day's PnL reward. The (unnormalized) value
   function couldn't track that spike. **Fix step 1:** dropped both events 5.0 → **0.5**
   (commit `53e8023`).
4. **Operator tried PnL-dominant scaling** (`net_pnl_weight` 1.0 → 10000, "perfect day reads
   250"). Discovered the trainer **normalizes advantages per minibatch**, so the absolute
   number is cosmetic — only term RATIOS reach the policy — and the value loss is the part
   that blows up.
5. **Added return normalization (the real fix, commit `82d86e0`).** `TrainConfig.normalize_rewards`
   (VecNormalize-style: divide rewards by a running std of the discounted return; preserves all
   term ratios so E8 holds). Verified: with `net_pnl_weight=10000`, max value_loss **1896 → 0.06**
   (~32000× smaller, finite). With normalization on, absolute scale is irrelevant, so **`net_pnl_weight`
   went back to 1.0** — keeping PnL=10000 would have drowned the anti-breach pain + pass/fail
   signals 500:1 (the very "without breaching" signal of priority #2). Now the reward reads in
   priority order: **passing stamps (0.5) ≥ anti-breach pain ≥ PnL-toward-2.5% ≫ shaping.**
6. **Training cell now streams ONLY the per-day scoreboard** (commit `9165ff0`), live, no
   health/checkpoint chatter.

## Current experiment config (all in the notebook HYPERPARAMETERS cell; config.py defaults untouched)
| Knob | Value | Why | Revert |
|---|---|---|---|
| `CCI_REGIME_GATE` | `False` | gate experiment failed (always-HOLD) | n/a (keep off) |
| `FAILED_DAY_PENALTY` | `0.5` | passing signal, was 5.0 (caused whipsaw) | set 5.0 |
| `FAST_PASS_BONUS` | `0.5` | passing signal, was 5.0 | set 5.0 |
| `NET_PNL_WEIGHT` | `1.0` | absolute scale moot under normalization; keep anti-breach audible | raise it |
| `NORMALIZE_REWARDS` | `True` | stable value loss at any reward scale | set False |
| `MIN_AGGRESSION` | `0.35` | exploration floor so masks don't freeze the policy | set 0.0 |
| `RISK_PER_TRADE` | `0.005` | 5 slots × 0.5% = 2.5% < 4% wall (overshoot guard) | — |
| `HARD_STOP_FRAC` | `0.005` | hard per-trade stop, cut losers small | 0.0 |

## Open question we're answering next
Does normalization **kill the whipsaw** AND do **breaches trend down / a first PASS appear**?
We had (pre-normalization): breaches 8/8 → 6/8 over 1000→2000 updates, win ~17% flat, 0 passes.

## NEXT STEPS (do these)
1. Run `colab/Quantra_Train.ipynb` on **real EURUSD** (confirm Cell 5a prints `[data] REAL bars`,
   not synthetic). Cell 5b streams the per-day scoreboard only.
2. **Watch `value_loss` behavior** (now via stability — it should stay small/steady, no
   blow-up to ~1.0, no entropy crash). The whipsaw should be gone.
3. **Watch the scoreboard trend:** breaches falling (priority #1/#2 working) and eventually a
   day that both survives AND hits +2.5% = the first PASS.
4. If breaches stay high specifically on **green-day peak-giveback** (the days-4&6 pattern:
   positive pnl, dd under 4% from open, but BREACH because the wall trails the intraday PEAK),
   the next lever is the **`DRAWDOWN_PAIN_WEIGHT`** (the "without breaching trailing DD" term) —
   raise it so protecting the peak is learned. Don't touch locked items.

## 🧭 BASELINES / ROLLBACK LADDER (named recoverable states on `claude/focused-faraday-if1ue7`)
Reverting CODE is easy (Git); training CONTINUITY is not — any policy trained under a given
`STATE_DIM` only resumes on that same observation contract. So we keep named anchors:

| Commit | Observation | What it is | Trainable baseline? |
|---|---|---|---|
| **`d796b42`** | **207-dim** | **last PRE-change baseline** — no expert engine, no trade_state. "Go back to the old learner" = this. | ✅ the canonical 207-dim baseline |
| `c4f3d5c` | 207-dim | expert engine added but **UNWIRED** (pure module; obs unchanged) | ✅ also 207-dim trainable |
| **`286adc9`** | **215-dim** | **current integration point** — `trade_state` block wired (+8). | ✅ new 215-dim contract |

⚠️ **`STATE_DIM` 207 → 215 at `286adc9`** → the registry will NOT resume any 207-dim checkpoint
(compatibility signature changed). The next Colab run is a **fresh retrain** on the 215-dim obs.
Treat any artifacts trained from `286adc9` on as belonging to the 215-dim regime, never mixed
with old 207-dim policies.

## DONE / IN-FLIGHT / PARKED
- **`trade_state` observation block — DONE + WIRED (commit `286adc9`).** 8 account-level
  discipline scalars the operator asked the policy to see: `daily_realized_pnl_pct`,
  `daily_drawdown_pct`, `trades_today`, `consecutive_losses`, `consecutive_wins`,
  `position_open`, `risk_budget_remaining`, `time_since_last_trade`. Env-filled (action-dependent),
  appended after `account`; masks/sizing/wall/reward untouched. Tests green (217 passed, 1 skipped).
- **Expert Signal Layer (Idea B) — Phase 1 BUILT + TESTED, NOT WIRED (commit `c4f3d5c`).**
  See [`docs/EXPERT_SIGNAL_DESIGN.md`](EXPERT_SIGNAL_DESIGN.md). Pure engine in
  `quantra/market_pipeline/expert_signal/` (soft features: `regime_bias`, `confidence`,
  `trend_strength`, `volatility_ok`, `session_ok`, soft `do_not_trade`, derived `expert_long/short`)
  over STRAT-001/002/004/006; reuses `compute_law_states`; 18 tests. 🔴 `do_not_trade` stays a
  FEATURE never a mask (the CCI-gate lesson). **Deliberately NOT wired** — wiring adds +8 →
  ANOTHER `STATE_DIM` change / fresh-start, so batch it intentionally with the next fresh-start
  plan (operator decision). Pause here to first test whether the trade_state features help.
- **Behavioral Cloning (Idea A) — PARKED.** Auto-generate `(obs, direction)` demos by
  replaying the rules through the env, warm-start the direction head before PPO (class-weight
  to avoid always-HOLD), behind a `BC_WARMSTART_EPOCHS` knob. Bigger lever for the
  cold-start "no common sense" problem; revisit after the expert-feature layer.

## Known mechanics worth remembering
- **The 4% wall is TRAILING from the intraday peak** (`peak_equity − 4%×account`, peak resets
  each midnight). So a green day can still BREACH by giving back >4% from its high (this is why
  days 4 & 6 breached while net-positive). The scoreboard's `dd` column slightly under-samples
  the true peak-to-trough vs the per-bar wall check — a reporting nuance, not the cause.
- **Advantages are normalized per minibatch** (`ppo_agent/loss.py`) → only reward RATIOS reach
  the policy; **value loss is NOT** advantage-normalized → that's what `normalize_rewards` fixes.
- **QUAD daily bonus** (in `reward.py`) is defined but NOT wired into training — dead code today.

## Files touched this thread
`colab/Quantra_Train.ipynb`, `quantra/learning_system/trainer/trainer.py`,
`quantra/learning_system/policy_registry/registry.py`, `quantra/runtime/config.py`,
`quantra/env/trading_env.py`, `tests/test_ftmo_master_suite.py`. Full reasoning is in each
file's IRAC UPDATE LOG and the git history (`git log --oneline`).
