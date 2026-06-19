# Policy Registry — How to Read a Policy's Identity

> **Rulebook (for the Risk Doctor / any LLM):** `docs/MLP_INTERPRETABILITY_LAYER.md`.
> **Master manual:** `docs/PROJECT_GUIDE.md` §4.11 (Policy Registry), §4.10 (Barbershop
> Fast Loop), §4.12 (Runtime Override System).
>
> Everything in this folder exists to answer one question for a given policy:
> **"What is this policy's perspective on how to pass the FTMO challenge?"** — i.e. what
> configuration (`training_phase`, `training_wheels`, the challenge numbers) produced it, what
> data it saw, and how well it actually passed (+2.5%/day without breaching the −4% trailing wall).

This README is the only file in `artifacts/` that is committed to git. The actual
registry contents (`manifest.json`, `performance.json`, `compatibility.sig`, and policy
weights) are run-specific and stay **git-ignored** — they live on disk / Google Drive,
not in the repo.

---

## Where things live

```
artifacts/policy_registry/<policy_name>/
├── manifest.json        # WHO this policy is (auto-generated at save time — never hand-written)
├── performance.json     # HOW it did (updated after every pass over the N_DAYS window)
└── compatibility.sig    # WHETHER you can resume it (state_dim + reward shape + law fingerprint)
```

A policy is created/updated by the Barbershop Fast Loop notebook
(`colab/Quantra_Barbershop.ipynb`, Cell 6) and read back by the Barbershop dashboard
and by you, the operator.

---

## 1. The auto-generated name (NEVER hand-typed)

The folder name (`<policy_name>`) is **derived from the `OVERRIDES` diff vs the baseline
config** — it is never written by hand and never invented by an assistant. The rule:

- Start from the baseline config (`quantra/runtime/config.py`, `laws.py`).
- Diff the run's `OVERRIDES` dict against it.
- Encode each meaningful change as a short token, then join with `-`.

**Example.** If Monty set `training_phase = "constrained"` and turned `training_wheels`
off, resumed from `v1-baseline`, the auto-name is:

```
v2-constrained-wheelsoff
```

The leading `v<N>` increments from the `base_policy` it resumed from. Monty *may* rename
a policy after the fact, but the **auto-name is always generated first** and recorded in
`manifest.json` under `auto_name_basis` so you can always reconstruct what changed.

> **The 3 market-condition signals are no longer tunable gates (2026-06-18 redesign).**
> Volatility, spread, and stationarity became **observation-only** signals: the bot SEES them
> and learns to trade both stationary AND non-stationary conditions itself. There is no
> `adf_p_value_threshold` to encode in a name anymore — the enforcement knob is
> `training_phase` (`free` = observation-only; `constrained` = the stationarity signal
> re-enforces). So a token like `constrained` means "this policy was shaped with the
> stationarity signal re-enforcing", not "a gate threshold was tuned".

---

## 2. Reading `manifest.json` (WHO the policy is)

```jsonc
{
  "policy_name": "v2-constrained-wheelsoff",   // auto-generated from the OVERRIDES diff
  "auto_name_basis": {                          // the diff, in plain tokens
    "changes":     ["training_phase=constrained", "training_wheels=OFF"],
    "wheel_state": "OFF"
  },
  "created": "2026-06-18T14:32:00",
  "base_policy": "v1-baseline",              // what it was RESUME_FROM'd (or null = fresh)
  "data_window": {"start": "2023-03-01", "n_days": 8},
  "n_passes_completed": 40,
  "state_dim": 207,                          // 🔴 locked dimension — see §6 of PROJECT_GUIDE
  "training_wheels": false,
  "training_phase": "constrained",
  "overrides_applied": { /* ...the full OVERRIDES dict used for this run... */ },
  "compatibility_signature": "sha256:abc123..."
}
```

Read it top-to-bottom as the policy's "birth certificate":
- **`overrides_applied`** is the exact knob set that shaped this policy. This is the
  ground truth for *why* it behaves the way it does.
- **`base_policy`** tells you its lineage (what weights it started from).
- **`data_window`** tells you which FTMO challenge days it was shaped on.
- **`state_dim` / `compatibility_signature`** decide whether you can resume it (§4).

---

## 3. Reading `performance.json` (HOW it did)

```jsonc
{
  "pass_history": [
    {"pass": 1,  "days_passed": 3, "days_failed": 5, "avg_pnl": -0.4,
     "avg_dd": -2.1, "breach_count": 2, "avg_gate_block_rate": 0.97},
    {"pass": 2,  "...": "..."}
  ],
  "best_pass": {"pass": 12, "days_passed": 6, "...": "..."},
  "overall_pass_rate": 0.52
}
```

What to look at, in order of importance to passing:
1. **`days_passed` / `overall_pass_rate`** — the real scoreboard (target hit + no breach),
   not raw PnL.
2. **`breach_count`** — any breach of the −4% trailing wall = a failed challenge day.
   A policy that passes by luck but breaches often is not a real pass.
3. **`avg_dd` / trade count** — in `PHASE_FREE` the 3 market-condition signals are
   observation-only (they never block), so a LOW trade count means the *policy* is choosing
   not to trade (shape it via reward), NOT a gate lockout. (`avg_gate_block_rate` is retained
   in the schema for continuity; it is ~0 in `PHASE_FREE` and >0 only in `PHASE_CONSTRAINED`.)
4. **`best_pass`** — the single best pass over the window, useful for choosing a
   checkpoint to promote into Full Training mode.

---

## 4. Checking compatibility before you resume (`compatibility.sig`)

`compatibility.sig` is a hash of **state_dim + reward-layer shape + law-parameter
fingerprint**. When you set `RESUME_FROM` in the Barbershop notebook, the loop checks the
saved signature against the *current* config + `OVERRIDES`:

- **Match** → resume from that exact checkpoint, continue the pass history.
- **Mismatch** → the system raises a **`CompatibilityError`** with a plain-English reason
  (e.g. "STATE_DIM changed 207 → 189 because INCLUDE_RAW_INPUTS was toggled; the old
  network's input layer no longer fits"), then offers to **start fresh or abort**. It
  **saves the old checkpoint first and never deletes it** — old policies are only ever
  superseded, never overwritten automatically.

What changes the signature (and therefore forces a fresh start):
- Anything that changes **`STATE_DIM`** / the observation schema (🔴 locked — needs sign-off).
- Anything that changes the **reward layer shape** (the number/arrangement of layers, not
  just a weight multiplier).
- Anything that changes a **locked law parameter fingerprint**.

What does **not** change the signature (safe to resume across):
- Toggling `training_phase` (`free` ↔ `constrained`) — enforcement only, not the input shape.
- Toggling `training_wheels` (observable + enforced, but not part of the input shape).
- Changing the challenge numbers (`daily_target_pct`, `daily_risk_pct`, `permanent_dd_pct`).
- (Operator-tunable reward weights are PLANNED, not yet wired — `reward.py` uses internal
  constants today.)

---

## 5. Loading an old policy (resume, inspect, or promote)

1. **Find it:** browse `artifacts/policy_registry/` and read each `manifest.json`
   (`overrides_applied` + `auto_name_basis` tell you its perspective; `performance.json`
   tells you how well it passed).
2. **Resume it in Barbershop mode:** in `colab/Quantra_Barbershop.ipynb`, set
   `RESUME_FROM = "artifacts/policy_registry/<policy_name>/<checkpoint>.pt"` (Cell 3),
   keep or adjust `OVERRIDES`, and run. The loop validates `compatibility.sig` first (§4).
3. **Promote it to Full Training mode:** use the policy's checkpoint as the starting
   weights for the longer walk-forward run (`colab/Quantra_Train.ipynb`). Barbershop
   *shapes*; Full Training *generalizes* across regimes — you move between the two freely.
4. **Diagnose it in the Barbershop dashboard:** launch `barbershop/dashboard.py` (or the
   ngrok tunnel from the Barbershop notebook). It auto-detects the latest telemetry under
   `artifacts/telemetry/` and renders the 5 screens + Risk Doctor.

---

## 6. Repo safety (so you never lose a policy)

> **ACTIVE WORK repo** (all edits/commits happen here):
> `https://github.com/monty313/RL-model-trading-bot-ppo-mlp_Claude-`
> **FALLBACK / SAFE RESTORE** (never edited — emergency revert only):
> `https://github.com/monty313/final-rl-model-6_13`

Registry contents are git-ignored, so they are **not** protected by the repo. Persist
checkpoints + manifests to **Google Drive** (the Barbershop notebook auto-saves to Drive
every `CHECKPOINT_INTERVAL` passes). Before any major code change in the active repo, push
the working state so the fallback repo stays a clean restore point. See `PROJECT_GUIDE.md`
§4.13 (Repo Safety Protocol).

---

## Known gaps (always state these honestly)

A registry entry's numbers are only as honest as the simulation behind them. The current,
acknowledged gaps:

1. **No real trained model yet (#1 active work item).** Policies so far were trained on
   synthetic data and do not transfer to real bars. A real Barbershop run is the first step.
   *(The former #1 — the ~98.7% gate lockout — is architecturally **fixed**: the 3 gates became
   phase-gated observations (2026-06-18), so `PHASE_FREE` trades freely; unproven on real bars
   until a real run.)*
2. **One wall, not two.** The sim models one daily trailing wall; real FTMO has TWO (daily
   loss from day-start AND permanent max drawdown). The −10% max is an **observation** (C12
   `dist_to_perm_dd`), not enforced in training. A sim pass is not a guaranteed live pass.
3. **Screen 1 demo curve.** Barbershop Screen 1 shows a labelled demo curve until a real
   pass-rate series is logged.
4. **input×gradient, not SHAP.** Trade-autopsy attribution is input×gradient, not true
   Shapley values.

---

## Update Log (IRAC)

- **[2026-06-18]** Created the policy-registry README + a `.gitignore` carve-out so it (and
  only it) is committed.
  - **I:** Every trained policy needs a saved, human-readable identity (what config made
    it, how it passed, whether it can be resumed), but `artifacts/` is git-ignored, so
    there was no committed place to document how to read a registry entry.
  - **R:** Operator brief Section 5 (Policy Registry) + Section 10-C (this file) + the
    standing rule that auto-names come from the `OVERRIDES` diff and the 5 known gaps are
    stated in every high-level overview.
  - **A:** Wrote this plain-English guide (manifest/performance/compatibility, auto-naming,
    resume/promote flow) and added `!artifacts/policy_registry/README.md` to `.gitignore`
    while keeping real registry data ignored.
  - **C:** Anyone (operator or LLM) can now read a policy's "perspective on passing the
    FTMO challenge" and safely decide whether to resume, promote, or start fresh —
    without risking the large run-specific artifacts being committed.

- **[2026-06-18]** Synced the registry guide to the gates→observations redesign + C12.
  - **I:** The guide described tunable gate thresholds, `gate_changes`/`reward_changes` in
    auto_name_basis, STATE_DIM=203, and "gate lockout" as the #1 gap — all now false.
  - **R:** Operator/Perplexity redesign (2026-06-18) + the honesty rule (docs == code).
  - **A:** Updated the auto-name example (`v2-constrained-wheelsoff`), the manifest sample
    (auto_name_basis `changes`/`wheel_state`, state_dim 207, `training_phase`), the performance
    diagnostics (pass-rate/breach first; `avg_gate_block_rate` is ~0 in PHASE_FREE), the
    compatibility + safe-to-resume lists (phase/wheels/challenge knobs), and the known gaps.
  - **C:** Anyone reading a policy's identity now sees the real knobs and the true "learns
    market conditions in PHASE_FREE" story — no obsolete gate-threshold framing.
