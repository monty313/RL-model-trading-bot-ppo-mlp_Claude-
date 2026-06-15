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

## 2b. DONE — Training wheels: semi-permanent counter-trend block masks (operator 2026-06-15) ✅
Added two INDEPENDENT, toggleable (`config.TRAINING_WHEELS`, default ON) counter-trend
OPEN-block masks, isolated from the locked 9 laws:
- **CCI wheel:** block SELL when CCI 5 AND 15 (applied SMA 20, shift 0) are BOTH above
  their SMA on BOTH 4H and 30m; block BUY when both below on both TFs.
- **BB wheel:** block SELL when price is above the upper band of BB10 AND BB100 (dev 0.5)
  on BOTH 4H and 30m; block BUY when price is below both lower bands on both TFs.
The ingredients + the two block flags (`tw_cci_block`, `tw_bb_block`) are in the observation
("acts"); the flags are enforced as masks ("laws"). STATE_DIM 185→203 (market 110→128);
snapshot re-pinned; full suite green (Section TW added). Same masks run train+live (parity).
**FLAGGED (operator override):** these wheels READ 4H to block, which the locked laws never
do (4H is observation-only for them) — kept as a separate, removable override per operator
direction. They also serve the "train faster relative to passing" goal: blocking
counter-trend opens removes the biggest source of avoidable breaches, so episodes aren't
wasted and the policy converges toward passing in fewer updates. See `COUPLINGS.md` [C9].

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
- **[2026-06-15]** Built the training-wheel counter-trend block masks (item 2b).
  - **I:** Operator wanted semi-permanent counter-trend "training wheels" (CCI 5/15 SMA20-sh0
    + BB 10/100 dev0.5 on 4h+30m) to stop the bot opening against a strong trend, plus the
    signals visible in the observation, plus a way to train faster relative to passing.
  - **R:** Operator directive 2026-06-15 (sells specified, buys = mirror, confirmed); must
    stay isolated from the locked 9 laws + be removable; same masks train+live.
  - **A:** New WHEEL_* indicator params; 18 observation features (16 ingredients + 2 block
    flags) appended to `market`; `build_direction_mask` gained training_wheels/wheel_states;
    env + live read the flags; `config.TRAINING_WHEELS` toggle (default ON). STATE_DIM
    185→203, snapshot re-pinned, Section TW added, full suite green.
  - **C:** The bot can no longer open into a strong 30m+4H trend, removing the biggest source
    of avoidable breaches — fewer wasted episodes per window = faster convergence to a
    consistently-passing brain — with the locked laws untouched and the wheels removable.
    FLAG: the wheels read 4H (operator override of the laws' 4H-observation-only rule).
