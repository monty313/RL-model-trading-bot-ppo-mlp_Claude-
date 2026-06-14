# Quantra — Deep Reinforcement Learning in Trading (PPO + MLP)

> **One universal PPO policy built to *repeatedly pass FTMO-style challenges*
> (2.5% daily target / 4% trailing wall) — not to maximize PnL.**
> Raw PnL is a sanity check. The scoreboard is: **pass rate → breach count →
> target-hit consistency → max-DD path.**

Built from the binding spec in [`docs/SOW_2_BUILD_SPEC.md`](docs/SOW_2_BUILD_SPEC.md)
and the 9 blueprint docs alongside it. The LLM Risk Doctor's binding rulebook is
[`docs/MLP_INTERPRETABILITY_LAYER.md`](docs/MLP_INTERPRETABILITY_LAYER.md).

---

## Why this trains fast and cheap (read this before renting a GPU)

The locked network is a **tiny 3×256 MLP with ~145 inputs**. For PPO on a net this
small, the bottleneck is **environment + feature simulation (CPU work)**, *not* the
GPU. A paid Colab GPU will usually sit near-idle while CPU env-stepping bottlenecks
the run — pure wasted money.

So Quantra ships a **hardware auto-optimizer** (`quantra.runtime`) that:

1. **Races CPU vs GPU** on a representative four-head workload at startup and picks
   the genuinely faster device.
2. **Prefers CPU** unless a GPU beats it by ≥1.30× (configurable) — a near-tie does
   not justify a paid accelerator.
3. **Auto-scales** vectorised parallel environments to drive **~80% utilisation** of
   the chosen device (leaving headroom so the Colab kernel stays alive).
4. **Measures and reports** real CPU/GPU utilisation so you can *see* the 80%.

### Try it in one command
```bash
pip install -r requirements.txt
python -m quantra.runtime
```
You'll get a throughput race table, the device decision, the parallel-env count,
and the measured utilisation. On most machines (and free Colab) it will choose CPU
and tell you the GPU isn't worth its cost for this model.

---

## Run on Google Colab

1. Open [`colab/Quantra_Train.ipynb`](colab/Quantra_Train.ipynb) in Colab.
2. **Pick a CPU (or "None"/High-RAM) runtime** — the optimizer will confirm a GPU
   isn't needed. Only switch to GPU if the race table actually shows ≥1.30×.
3. Run the cells: clone → install extras → mount Drive (price data lives in your
   `rl-trading-data` folder) → race hardware → smoke-train → utilisation + scoreboard.

The price data (4 symbols, MT5 1m, ~5 yr) is pulled from your Google Drive — by
Drive mount, or by file ID via `gdown` as a fallback. IDs are registered in
[`quantra/runtime/config.py`](quantra/runtime/config.py).

---

## Architecture at a glance (all locked, see `docs/`)

| Piece | Locked design |
|---|---|
| Algorithm | PPO, on-policy, **γ=0.997, λ=0.97** (never tuned) |
| Network | shared **3×256 MLP** trunk, **4 heads**: direction · Beta size · pointer (5 slots) · value |
| Action space | {HOLD, OPEN_LONG, OPEN_SHORT, CLOSE}; **5 trade slots/symbol**; CLOSE routed by the pointer head |
| Symbols | EURUSD, XAUUSD, GBPUSD, US30 — stepped **true-sequentially** each 1m bar over **one shared account** |
| Laws | **9 laws + 3 gates** as **masks** (logit −1e9 on forbidden), never reward terms |
| Reward | layered **L0–L6 + QUAD bonus**; **Layer 0 (net PnL) always dominates** (E8) |
| Episode | two-phase: 4% trailing → at +2.5% auto-flat all → fresh 1% trailing |
| Validation | walk-forward **12mo/2mo/1mo, 7 seeds**; promote on ≥3 seeds + no worse breach |
| Diagnostics | telemetry → 7 visuals → **read-only LLM Risk Doctor** (8-failure taxonomy) |

## Repository layout
See [`REPO_MAP.md`](REPO_MAP.md). The SOW's numbered tiers (`00_…`–`06_…`, `99_docs`)
map to importable packages under `quantra/` (digit-prefixed names aren't valid Python
identifiers).

## Build status
**✅ M0→M15 COMPLETE.** The full SOW (`docs/SOW_2_BUILD_SPEC.md` Section 13) is built:
data pipeline · ~179-scalar feature/state vector · 9 laws + 3 gates (−1e9 masks) ·
4-symbol true-sequential env with the B5 no-overshoot invariant · four-head PPO agent ·
layered reward (Layer-0 dominance) · curriculum + two-phase episode rule · trainer
(GAE + aggression scheduler + G8) · versioned telemetry · the 7 interpreter visuals ·
read-only LLM Risk Doctor · scoreboard + walk-forward + promotion gate · Optuna HPO ·
MT5 live bridge + kill switches · end-to-end acceptance. Run `pytest` (97 tests) and
`python -m quantra.acceptance` to see the whole chain.

Next (operator): point it at the real MT5 bars in Colab and run the 7-seed walk-forward
to establish a pass rate.

**Tests:** one master suite — `tests/test_ftmo_master_suite.py` (just run `pytest`).
All future tests append there. **Every file carries an IRAC update log** (Issue ·
Rule · Application · Conclusion-why-it-helps-pass-FTMO); see
[`quantra/constitution/update_rules.md`](quantra/constitution/update_rules.md).

## ⚠️ Disclaimer
Trading is risky. This is research software for passing simulated prop-firm
challenges; it is not financial advice. Live deployment places real orders only when
*you* run the live bridge, and always behind the manual-halt and breach-auto-flat
kill switches.


---

## Update Log (IRAC) — standing rule since 2026-06-13
*Every change appends a dated IRAC entry; the **Conclusion** always states why it
makes the bot pass FTMO more consistently. Rule: [quantra/constitution/update_rules.md](quantra/constitution/update_rules.md).*

- **[2026-06-13]** Front door updated for M0/M1 + the new standing rules.
  - **I:** The README had no change-history and didn't surface the IRAC + master-suite rules.
  - **R:** Operator IRAC rule (2026-06-13) + SOW R2-R4.
  - **A:** Documented M0/M1 status, the hardware optimizer, and pointed to the standing rules.
  - **C:** A current, honest front door keeps contributors aligned to repeated FTMO passing and prevents drift.
