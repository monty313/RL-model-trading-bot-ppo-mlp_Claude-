# INSTRUCTIONS — Pending / Next Steps

A running log of agreed-but-not-yet-built work, so nothing is lost between sessions.
(The binding spec is still `docs/SOW_2_BUILD_SPEC.md`; this file is the live work queue.)

---

## 1. OPEN — CCI multi-SMA-on-CCI refactor (awaiting operator sign-off) 🔴
**Current state:** each CCI (periods 10/30/100, on 1m/5m/30m/4H) is compared to exactly
**one** smoothed reference: `cci_dev = (CCI − SMA(CCI, 2) shifted 4) / 100`. Only one
SMA length (2), one shift (4). No standalone SMA-on-CCI line, no explicit zero / ±100
flags, no multi-SMA "continuous-trend" state.

**Operator intent:** use *multiple* shifted SMA-on-CCI references to define **continuous-
trend states** — "CCI moving farther from zero without breaking" — comparing CCI behaviour
at different magnitudes and across timeframes.

**Proposed smallest refactor (observation-only; laws/locked `cci_dev` UNCHANGED):**
per (TF, CCI period) add shifted-SMA-on-CCI at lengths **{TO CONFIRM, e.g. 2 / 5 / 10}**
(all shift 4) + a stacked **trend-state flag** (`+1` if CCI > SMA_short > SMA_mid > SMA_long
and CCI>0; `-1` mirror; `0` otherwise) + explicit `cci_sign` and `cci_extreme` (±100) flags.
Routes through `schema.py` → snapshot guard → `builder.py` (same pattern as the raw-input
block). 🔴 CCI params are LOCKED → **needs explicit approval + the chosen SMA-length set
before implementing.**

## 2. APPROVED — DEFERRED: MT5 demo launcher (build AFTER #1 is resolved)
Operator approved ("yes"). Add a **one-command** `live_bridge` launcher tying
checkpoint + MT5 login + `LiveSession` + `MT5BarFeed` into a push-button **DEMO** run:
- entry e.g. `python -m quantra.live_bridge.demo_launcher --login <id> --server <srv> --checkpoint <path> --symbols EURUSD,XAUUSD,GBPUSD,US30`
- keep ManualHalt + breach-auto-flat armed; **DEMO account first**, never funded on first run.
- Do this once the CCI/normalization question (#1) is cleared.

## 3. STANDING — before any live/funded run (operator tasks, not code)
- Train a real-data brain via the 7-seed walk-forward in Colab → a promoted checkpoint
  (there is no trained model yet, only synthetic-data runs).
- Validate the MT5 live loop on a DEMO account (terminal-only calls are source-verified only).

---

## Update Log (IRAC) — standing rule since 2026-06-13
- **[2026-06-13]** Created the pending-work instructions file.
  - **I:** Two agreed items (CCI multi-SMA refactor; MT5 demo launcher) were at risk of being lost between sessions.
  - **R:** Operator request ("put that in instructions file") + the master-suite/IRAC discipline.
  - **A:** Recorded the open CCI proposal (needs sign-off + SMA lengths) and the approved-but-deferred MT5 demo launcher.
  - **C:** The next steps toward a trained, demo-validated, MT5-ready passer are queued and auditable — nothing slips, which keeps the path to consistent passing on track.
