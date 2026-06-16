# Barbershop Remediation Plan

**Created:** 2026-06-16 · **Owner:** Claude (for Monty) · **Status:** IN PROGRESS

This plan fixes every issue raised in the 2026-06-16 mentor review of `barbershop/`.
The driving principle is the one the review ended on: **stop showing placeholder data as
if it were real.** Every item either makes a panel show *real* data or makes it *honestly
say it has none* — never fabricate. Work the items top-to-bottom; check each box only when
its acceptance criteria pass and the change is committed.

Run `pytest barbershop/` after every item — it must stay green. The real-run producer is
`scripts/emit_real_telemetry.py`; the real telemetry lands in `artifacts/telemetry/` (gitignored).

---

## WI-1 — Real feature names on real data  🔴 highest (correctness bug)
**Problem.** `group_indicators` defaults to `MOCK_FEATURE_NAMES` (9 names); the dashboard
calls it with no names, so on a real 203-feature run the "What the bot SAW" panel + the
heatmap label the first 9 indices with mock names. The telemetry header *has* the real
`feature_names`, but the adapter drops them. `losing_trades._atr_value` has the same defect.
**Fix.** Surface `feature_names` from the telemetry header through the adapter → the bundle →
`group_indicators(..., feature_names=...)` at both call sites; thread it into the pattern
finder's ATR lookup too. Mock bundle carries `MOCK_FEATURE_NAMES`.
**Acceptance.** On a real run, `group_indicators` uses the real schema names and categorises
the real features; a test asserts a real obs maps to >9 named cells with correct categories.
**Status:** [x] DONE 2026-06-16 — real run now yields 218 correctly-categorised cells
(203 features + 12 laws + 3 health) vs 9 mislabelled before. Test added. *(Noted follow-up:
156 market-structure rows make the heatmap tall — a future UX grouping/collapse item, not a
correctness issue.)*

## WI-2 — Real GAE advantage on real runs  🔴
**Problem.** `advantage` is never produced; the Panel-2 strip is empty on real runs.
**Fix.** Compute GAE per day in `emit_real_telemetry.py` over the real rollout (locked
γ/λ from the trainer) and log it per step; the adapter reads it. Advantage becomes REAL.
**Acceptance.** A re-emitted real run has non-NaN advantage with both signs; the advantage
strip renders; `advantage` drops out of the "not produced" list for that run.
**Status:** [x] DONE 2026-06-16 — re-emitted run: advantage non-NaN for all 5,565 steps,
range −1.17…1.27, both signs. Producer computes per-day GAE (locked γ/λ); adapter reads
`outcome.advantage`. *(The "not produced" banner cleanup is WI-4.)*

## WI-3 — Real attribution for the autopsy RIGHT column  🟠
**Problem.** SHAP is random mock numbers; empty on real runs. The marquee feature is a stub.
**Fix.** In the producer, compute **input×gradient attribution** of the chosen-action logit
w.r.t. the observation for each trade step (a legitimate, fast saliency — *not* Shapley), write
a `<run>_attribution.jsonl` sidecar; the adapter loads it into the SHAP contract columns.
**Label it honestly** in the UI ("input-gradient attribution, not Shapley").
**Acceptance.** A real run's autopsy RIGHT column shows real, signed, sorted attributions
grouped toward/away by real feature name; a test covers the sidecar load + grouping.
**Status:** [x] DONE 2026-06-16 — producer writes 16 trade-step attributions keyed on REAL
feature names (cci100_sma_4H, …); adapter.load_attribution loads them; figure labelled
"input×gradient attribution (not Shapley)". Caught+fixed a bug: list_real_runs was picking
up the sidecar as a run. Unit test added.

## WI-4 — Dynamic placeholder detection + honest UI grey-out  🟠
**Problem.** `unavailable_fields` is a static list; panels render empty/fake bars instead of
saying "no data."
**Fix.** Compute `unavailable_fields` dynamically (a contract field is unavailable if it's
all-NaN/empty in the mapped frame). The screens grey out + label any panel whose backing
field is unavailable, instead of rendering an empty/fabricated figure.
**Acceptance.** A run missing advantage/SHAP shows greyed, labelled panels (not empty bars);
a run that HAS them shows real panels. Tested both ways.
**Status:** [ ]

## WI-5 — Risk Doctor context budget  🟠
**Problem.** The full 32 KB (~8k-token) manual is sent on every message; against the default
`llama3` that likely overflows the context before the question lands. The Doctor has never
run against a live LLM.
**Fix.** Add `DOCTOR_MANUAL_MAX_CHARS` + a budget-aware manual loader that keeps the
high-signal sections (North Star, Core Definitions, Safety Rules, Diagnostic Template,
Failure Taxonomy) when the manual exceeds budget; keep RULE-6's "manual present or offline"
intact. Document a long-context model recommendation in `config`.
**Acceptance.** The assembled system prompt stays under a configurable char budget; a test
asserts the trimmed manual still contains the safety rules + diagnostic template headers.
**Status:** [ ]

## WI-6 — Robust Doctor response parsing  🟡
**Problem.** Section split keys off exact emoji icons; prescription extraction greps one
phrase. A real model that paraphrases or drops an icon → garbled sections / wrong export.
**Fix.** `format_sections` falls back to matching section *titles* when icons are absent, and
to a single "What I see" block when neither is found; `_extract_prescription` gains a
robust fallback. Tests cover a no-icon and a paraphrased response.
**Acceptance.** A response with titles-but-no-icons parses into the right sections; a blob
with neither degrades gracefully; prescription export still finds the recommendation.
**Status:** [ ]

## WI-7 — Single data-contract source of truth  🟡
**Problem.** The contract shape lives in three places (mock generators, adapter, validators);
this drift already caused two review bugs.
**Fix.** Add `barbershop/contract.py` owning the column lists, action order, engine-int map,
and placeholder fields; refactor `data`, `adapter`, `config` to import from it. Behaviour
identical; tests stay green.
**Acceptance.** One module defines the contract; `data`/`adapter` reference it; full suite green.
**Status:** [ ]

## WI-8 — Perf + Screen-1 honesty  🟡
**Problem.** The mock bundle is rebuilt on every render; Screen 1's training wall is synthetic
even on a real run and its "live refresh" reseeds noise.
**Fix.** Memoise the mock bundle; on a real run, read a real pass-rate series if one exists
(`artifacts/telemetry/<run>_passrate.json`), else label Screen 1 honestly as a demo curve
(no fabricated "live" claim on real data).
**Acceptance.** Mock bundle built once; Screen 1 on a real run is labelled demo (or shows a
real series when present); a test covers the label/real-series branch.
**Status:** [ ]

---

## Checklist (mirror of the above — tick on commit)
- [x] WI-1 Real feature names on real data
- [x] WI-2 Real GAE advantage
- [x] WI-3 Real input-gradient attribution
- [ ] WI-4 Dynamic placeholder detection + grey-out
- [ ] WI-5 Risk Doctor context budget
- [ ] WI-6 Robust Doctor parsing
- [ ] WI-7 Single contract source
- [ ] WI-8 Perf + Screen-1 honesty

## Update log
- [2026-06-16] [Claude] — Plan created from the mentor review. 8 work items, ordered by severity.
