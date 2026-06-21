# EXPERT SIGNAL LAYER — design doc (Idea B)

**Status:** PROPOSED (design only — no code written yet). **Date:** 2026-06-21.
**Decision owner:** operator. **Scope locked this round:** STRAT-001…006 (market-based).

> One-line: turn the operator's rule-based strategy portfolio (the
> `Example trading strategies portfolio v.2` PDF) into a small block of compact,
> machine-readable **observation features** — a *soft expert read* the PPO policy
> can learn to use — **without** changing actions, masks, sizing, or the reward.

---

## 1. Why this, and why as OBSERVATION features (not actions, not a gate)

The operator can express an edge as rules and show it on a backtest. There are three
doors to inject that into the bot:

| Door | Mechanism | Verdict |
|---|---|---|
| Action (behavioral cloning) | copy the rule's trades into the policy weights | parked (Idea A) — bigger lever, more machinery, separate project |
| **Observation (this doc)** | **expose the rule's read as input features** | **CHOSEN** — low-risk, additive, complements BC later |
| Hard mask/gate | force-block actions by rule | ❌ **rejected** — this is the `CCI_REGIME_GATE` that caused the always-HOLD collapse (see SESSION_HANDOFF arc #1) |

Key truth carried over from the comparison: **features give the bot eyes, not hands.**
Adding `regime_bias` etc. does not change the RL difficulty — the policy still must
*learn* (via reward) when to act on the read. Improvement is expected to be **gradual**,
and the **reward still owns survival** (the STRAT rules are risk-blind — none of them know
about the 4% trailing wall).

This door is also the one the operator's own prior conclusion already endorsed: the CCI
regime "stays in the *observation* so the policy can learn it as a soft feature, never a
hard gate."

---

## 2. The big finding — we are ~80% there already

The observation **already contains a baby ExpertSignalGenerator**: the `law` block
(`feature_builder/schema.py:179-185`) is **9 directional laws + 3 market-condition
signals**, each −1/0/+1:

```
law_super_trend_bb, law_super_trend_cci, law_super_trend_ssma,
law_trend_bb,       law_trend_cci,       law_trend_ssma,
law_pullback_bb,    law_pullback_cci,    law_pullback_ssma,
market_volatility_obs, market_spread_obs, market_stationarity_obs
```

Mapping to the PDF:

| PDF strategy | Already encoded as | New work needed |
|---|---|---|
| **STRAT-001** BB regime (BB200 HTF / BB20 LTF) | `law_*_bb` (super_trend / trend / pullback) + `boll_*` columns | aggregate into a single bias/score |
| **STRAT-002** dual CCI(30/100) | `law_*_cci` + `cci30/cci100_{tf}` + `cci_sync_{tf}` | aggregate; add ±100 "surge" level |
| **STRAT-003** triple CCI(14/100/900)+SMA(20) | **partial** — matrix has CCI **10/30/100**, not 14/900 | ⚠️ gap: either approximate with 10/30/100 or add periods in feature_builder (precompute) |
| **STRAT-004** SMA stack (50 / 4 / 4-shift) | `law_*_ssma` + `ssma_align_{tf}` + `raw_sma*` | aggregate |
| **STRAT-006** ADX/ATR filter (do-not-trade / great-movement) | `market_volatility_obs` + `adx5/adx15_{tf}` + `atr_level/ref/dev_{tf}` | derive `volatility_ok` / `do_not_trade` explicitly |

**Therefore the engine is mostly a VOTING AGGREGATOR over signals the bot already
computes**, producing a few decisive summary features, plus a small number of genuinely
new ones (`expert_confidence`, `trend_strength`, `session_ok`, soft `do_not_trade`). This
is *less new indicator math than it looks* — and that is good: less surface area, fewer
ways to break the locked law/mask semantics.

---

## 3. The output schema (first cut — 8 features)

All bounded → they live in the **normalized/clipped** path, **NOT** `RAW_FEATURE_NAMES`
(no standardization needed).

| Feature | Range | Meaning | Source STRATs |
|---|---|---|---|
| `expert_long`   | {0,1} | net long bias AND tradeable | 001/002/004 votes ∧ 006 filter |
| `expert_short`  | {0,1} | net short bias AND tradeable | 001/002/004 votes ∧ 006 filter |
| `expert_confidence` | [0,1] | agreement strength across votes | all |
| `regime_bias`   | {−1,0,+1} | net directional bias | 001/002/004 |
| `trend_strength`| [0,1] | dual-TF alignment depth | 006 + dual-TF checks |
| `volatility_ok` | {0,1} | "OK to trade" (movement present) | 006 (ADX>SMA ∧ ATR>SMA) |
| `session_ok`    | [0,1] | session-quality weight (London/NY hi) | new (time features exist: `time_sin/cos_hour`) |
| `do_not_trade`  | {0,1} | **SOFT** chop/avoid flag (observation only) | 006 + neutral-regime |

🔴 **`do_not_trade` is a feature, never a mask.** It must not touch
`law_mask_engine/engine.py`. (Re-run-the-CCI-gate hazard.)

Deferred to a later round (need external data): `news_risk` (ForexFactory red folder),
COT bias (STRAT-010), opening-bell range (STRAT-009, needs session-anchored H/L).

---

## 4. Architecture & integration points (grounded in the code)

### 4.1 The engine — a pure, precomputed feature function
- **New module:** `quantra/market_pipeline/expert_signal/engine.py`.
- **Pure functions, one per STRAT family** (`_bb_regime_vote`, `_dual_cci_vote`,
  `_sma_stack_vote`, `_adx_atr_filter`), each returning `{long_score, short_score}` (or
  `{allow_trade, trend_strength}` for the filter), then an aggregator → the 8 features.
- **Reads precomputed columns BY NAME** (`cci30_{tf}`, `boll_bb200_mid_{tf}`,
  `adx15_{tf}`, `atr_dev_{tf}`, …) from the feature row. **Do NOT** use the PDF-paste's
  per-step `df.index.asof` pandas lookups — that breaks the precompute design and is slow.
- **Action-independent** (depends only on market bars) ⇒ it belongs in the **precomputed
  path**, computed once per bar in the FeatureBuilder, exactly like `market`/`market_raw`.

### 4.2 Schema wiring
- Add an `"expert"` block to `_BLOCK_BUILDERS` (`schema.py:228`) with `_expert_names()`,
  **appended after `market_raw`** so no existing feature index shifts.
- Add it to `_PRECOMPUTED_BLOCKS` (it's action-independent).
- Update `EXPECTED_WIDTHS` (`schema.py:324`), `config.nominal_state_dim`, and regenerate
  the committed snapshot via `tools/snapshot.py --update`.
- It is **bounded** ⇒ stays out of `RAW_FEATURE_NAMES`.

### 4.3 Config / control panel (operator convention)
- Defaults (toggles + thresholds) in `quantra/runtime/config.py` (the blessed baseline).
- Every knob mirrored into the notebook HYPERPARAMETERS → `OVERRIDES`, **one-line
  revertible**: `EXPERT_SIGNALS=True`, per-STRAT on/off, `ADX_MIN`, `CONF_THRESHOLD`,
  session windows.

### 4.4 What we DO NOT touch
- 🔴 `law_mask_engine/engine.py`, the 9 locked laws, masks, sizing, the 4% wall,
  γ/λ, the reward term math. This layer is **purely additive perception**.

---

## 5. The unavoidable cost (operator-acknowledged)

Adding any feature changes `STATE_DIM` (207 → 207+K). Per `schema.py:282-287`, that
changes **every saved policy's compatibility signature**, so the registry **refuses to
resume old checkpoints** → **a fresh retrain is required.** This is a one-time cost, paid
once; all subsequent shaping (toggles, thresholds) is revertible without another reset.
**Operator has accepted this** for this project.

---

## 6. Test & telemetry plan
- Unit tests per vote function on synthetic feature rows (each regime → expected vote).
- Aggregator tests (vote tally → `regime_bias`/`confidence`/`expert_long`).
- Update the master-suite block-width assertions (they SHOULD fail until updated — that's
  the guardrail). Run `python -m pytest tests/` green before any commit.
- Telemetry/interpreter auto-label from `FEATURE_NAMES`, so per-expert-feature neuron
  attribution comes **for free** — the read-out for "does the policy actually use these?"

---

## 7. Build order (when greenlit)
1. `expert_signal/engine.py` + config dataclass + unit tests (pure, no wiring) → tests green.
2. FeatureBuilder emits the block (precomputed) + schema `"expert"` block + width asserts
   + snapshot regenerate → master suite green.
3. Notebook HYPERPARAMETERS toggles (`EXPERT_SIGNALS`, per-STRAT) → `OVERRIDES`.
4. Fresh train run; read scoreboard + interpreter attribution; decide next lever.
- IRAC entry + COUPLINGS note on every edited file (standing rule).

## 8. Open questions to resolve before/at build
- **STRAT-003 trinity:** approximate with CCI 10/30/100, or add 14/100/900 to the
  feature_builder precompute? (Adds width; same fresh-start already paid.)
- **HTF/LTF mapping:** PDF says H1 bias / M5 entry; the matrix carries 1m/5m/30m/4H. Pick
  LTF=5m, HTF=30m (or 4H, which is observation-only by lock) per strategy.
- **`session_ok` source:** derive from existing `time_sin_hour/time_cos_hour`, or add an
  explicit session-window feature?

---

## UPDATE LOG (IRAC)
- **[2026-06-21] Created (design only).**
  - **I:** Operator wants to inject the STRAT portfolio as a soft expert read; needs a
    grounded, low-risk plan that respects locked items.
  - **R:** Observation-features door (not mask/BC); additive; locked-core untouched;
    operator convention (config defaults + notebook OVERRIDES, one-line revertible).
  - **A:** Scoped STRAT-001…006, defined an 8-feature block, identified that the `law`
    block already encodes most of it, mapped integration points in `schema.py` /
    FeatureBuilder, flagged the `STATE_DIM` fresh-start cost (accepted).
  - **C:** Gives the policy a compact expert read it can learn to weight, without risking
    the always-HOLD gate failure or disturbing the survival reward — eyes now, hands
    (BC) and the reward later. Helps consistent passing by pre-digesting the operator's
    regime/entry/no-trade knowledge into decisive features.
