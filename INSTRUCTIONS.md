# INSTRUCTIONS — Pending / Next Steps

A running log of agreed-but-not-yet-built work, so nothing is lost between sessions.
(The binding spec is still `docs/SOW_2_BUILD_SPEC.md`; this file is the live work queue.)

---

## 1. DONE — CCI kept RAW (operator final decision 2026-06-13, commit 389c35d) ✅
Operator decided: do NOT normalize CCI. The observation now exposes the **raw CCI value**
`cci{p}_{tf}` and the **raw shifted-forward SMA** `cci{p}_sma_{tf}` (period 2, shift 4) —
no `/100`, no `(CCI−SMA)/100`. The applied SMA stays period 2 / shift 4 exactly; the laws
read raw value vs raw SMA (legal space identical, Section F tests verify). STATE_DIM 179→167
(the duplicate `raw_cci` block was removed). Snapshot re-pinned; 101 tests green.

## 1b. DONE — cross-file coupling docs (commit pending) ✅
Authoritative map in `COUPLINGS.md` (8 clusters) + inline `# COUPLING:` notes at the
definition sites. Enforced by the change-impact tracker + the master suite.

## 2. DONE — Bollinger: keep BOTH normalized + raw (operator decision 2026-06-15) ✅
Kept the ATR-normalized distance `boll_{band}_{tf}=(close−band)/ATR14` AND added 18 RAW
band-level features `boll_{band}_raw_{tf}` (BB20/BB200 mid/up/lo on 5m/30m/4H, unclipped,
in RAW_FEATURE_NAMES). Additive, observation-only — laws still read the normalized sign so
the legal space is unchanged. STATE_DIM 167→185 (market 92→110). Snapshot re-pinned; 101 tests green.

## 3. APPROVED — NEXT: MT5 demo launcher (now unblocked)
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
