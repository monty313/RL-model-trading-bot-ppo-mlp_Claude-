# MLP INTERPRETABILITY LAYER
### The diagnostics contract for seeing what the shared MLP trunk learns, why it acts, and whether that internal behavior supports repeated FTMO-style challenge passing.

**Status:** 🟡 Add-on blueprint — pairs with G7 telemetry, I4 LLM Risk Doctor, J2 modules (TelemetryLogger, MLPInterpreter, LLMRiskDoctor)
**Origin key:** [M] = Monty | [P] = Perplexity adopted | [C] = Claude bridge
**Repo:** https://github.com/monty313/Qunatra_Deep-Reinforcement-Learning-in-Trading.git

---

## HOW TO READ THIS FILE

This document has TWO audiences. It speaks to both, in the same words, on purpose.

1. **Claude Code** — to build the interpretability infrastructure (TelemetryLogger, MLPInterpreter, the LLM Risk Doctor's input contract) as first-class permanent infrastructure.
2. **The LLM Risk Doctor itself** — at inference time, this file IS its operating manual. The definitions, rules, and the failure taxonomy below are what it uses to reason. Every term it might invoke is defined here, with the rule for how to apply it to passing the FTMO challenge.

Where the file says "the LLM must…" or "the Risk Doctor must…" — those are instructions to the live LLM at diagnosis time. Where it says "Claude Code must…" — those are build-spec requirements.

---

## NORTH STAR (READ FIRST — THE LLM ANCHORS EVERY DIAGNOSIS TO THIS)

The bot is NOT being trained to maximize PnL. It is being trained to **repeatedly pass FTMO-style constraints over time** — hit DAILY_TARGET, avoid the 4% trailing wall, do it consistently across many windows and seeds.

**The single question the LLM must answer for every diagnosis:**

> Did the MLP's internal behavior on this run help the bot pass FTMO consistently, or did it damage that ability — even if PnL looked good?

PnL is diagnostic only. Pass rate, breach count, target-hit consistency, max DD path are the scoreboard. The LLM judges everything against passing, not against profit.

---

## WHAT THIS LAYER PROTECTS AGAINST

A model can look "good" on PnL while being **bad for the mission** if it:
- Drifts close to the drawdown wall too often
- Wins through fragile behavior (one seed, one regime)
- Learns shortcuts that don't generalize
- Has a healthy actor but a broken critic (or vice versa) and you can't tell which

Without this layer, the LLM gives shallow diagnoses:
- ❌ "performance dropped"
- ❌ "drawdown increased"
- ❌ "reward may need tuning"

With this layer, the LLM gives deep diagnoses:
- ✅ "the hidden representation collapsed in breach-risk states — the bot stopped distinguishing danger from normal trade context after window 3"
- ✅ "the actor became overconfident in weak legal contexts — premium and weak legal setups produced near-identical action distributions"
- ✅ "the critic stopped recognizing stagnation early enough — V(s) stayed flat through three consecutive stagnant bars before correcting"

That gap is the whole point of this layer.

---

## NON-NEGOTIABLE SAFETY RULES (apply to the LLM Risk Doctor at inference)

The interpretability layer is **diagnostics and supervision only**.

| The LLM Risk Doctor MAY read | The LLM Risk Doctor MAY NOT touch |
|---|---|
| Observations, law states, hidden states, head outputs, rewards, risk telemetry, outcomes | Execution commands, action legality / masks, size feasibility, hard walls, broker instructions |

It may produce **diagnoses and prescriptions** (curriculum changes, reward review, retraining focus, operator alerts). It must NEVER trigger a live action. Hard boundary. Period.

**Two more inference-time rules for the LLM:**

1. **Evidence-only reasoning.** Every claim must cite specific telemetry fields, visuals, or metric values. No "the model probably…" — only "the model X, as shown by Y." If the evidence isn't there, say so. Speculation without evidence is forbidden.
2. **Acknowledged uncertainty.** When evidence is ambiguous, say *"diagnosis uncertain — need [specific additional telemetry]"* and stop. Confident wrong answers are worse than honest "I don't know."

---

# CORE DEFINITIONS

*Every term: full definition → why it matters for FTMO passing → attributes → methods → the two most connected terms → connection rules → what HEALTHY looks like → what UNHEALTHY looks like → the diagnosis rule the LLM applies.*

---

## TERM 1 — STATE VECTOR [M+P]

**Definition.** The normalized observation the policy receives at each 1m decision step. The bot's complete visible world at that moment: market structure (1m/5m/30m/4H), trade slots (×5 per symbol), shared account block, challenge-progress features, and the law/gate ingredients + flags.

**Why it matters for passing.** If the state vector doesn't faithfully describe the FTMO-relevant situation, the MLP cannot form a useful internal belief about challenge state. Garbage in → black-box nonsense out.

**Attributes.**
- Multi-timeframe market features (1m / 5m / 30m / 4H)
- Law ingredients + law/gate flags (every law's underlying inputs PLUS its 0/1 active flag)
- Trade-state block: per-slot ×5 (direction, uPnL, age, entry distance/ATR, MFE/MAE, momentum flag, occupied flag)
- Account block: equity vs initial, equity SMA baseline, daily-loss buffer, trailing-loss buffer, daily challenge progress, overall challenge progress, breach-risk proximity
- Normalized, platform-blind encoding (ATR units, returns, z-scores — no raw prices)

**Methods.** Present the environment in a machine-readable form. Preserve every law ingredient as a separate observation. Carry challenge context so the policy can learn challenge-aware behavior.

**Two most connected terms.** Hidden State · Law Context

**Connection rules.**
- The hidden state is a transformation of the state vector. If state vector is incomplete, the hidden state is untrustworthy.
- Law context must appear inside the state vector — the bot must learn behavior inside legal structure, not outside it.

**Healthy.** Distinguishes the challenge-critical situations: safe trend continuation · legal pullback opportunity · stagnation inside favorable context · breach-risk approach · post-target management state.

**Unhealthy.** Premium and weak setups produce near-identical observations · breach-risk states look like normal trading · challenge progress isn't visible.

**Diagnosis rule the LLM applies.** If hidden state shows representation problems (Term 3), first check whether the state vector itself is missing or distorting a challenge-critical signal before blaming the trunk.

---

## TERM 2 — LAW CONTEXT [M+P]

**Definition.** The set of directional permissions and gates that define what is legal at the current step. Laws run BEFORE PPO action selection and constrain the legal action set via masking (logit = −1e9 on forbidden actions).

**Why it matters for passing.** The project rule is absolute: laws are NEVER reward terms. They define the legal space before reward applies. The bot must learn inside legal structure. A clean law mask saves the bot from directional stupidity that would breach the FTMO wall.

**Attributes.**
- Active law family + state (9 core laws: 3 super-trend, 3 trend, 3 pullback)
- Gate states (ATR Liquidity, Spread Filter, Stationarity Regime — 3 misc gates)
- Enforcement mode: **live ban mode** (bans forbidden directions; everything else legal) OR **law-school permission mode** (permits ONLY when the stage's law is active)
- Legal action set after masking (what the actor head can actually sample from)

**Methods.** Ban forbidden directions. Define permission windows during curriculum stages. Separate "bad action" prevention from reward shaping.

**Two most connected terms.** Action Distribution · State Vector

**Connection rules.**
- Action distribution MUST always be interpreted in the context of what was legal. An action with 0.95 probability means nothing if it was the only legal option.
- State vector MUST include the ingredients + flags representing law context.

**Healthy.** Premium setups are clearly legal · low-quality/dangerous directions are masked · the model is evaluated on what it does INSIDE the legal space.

**Unhealthy.** Bot repeatedly favors illegal actions in pre-mask logits (mask-dependence) · law context features barely move when laws activate/deactivate (the bot isn't using them).

**Diagnosis rule.** Before blaming the actor for a bad decision, the LLM MUST verify what was legal. "Bad" actions taken under restrictive masks often reflect that the masks themselves were too narrow, not that the actor was broken.

---

## TERM 3 — HIDDEN STATE [P]

**Definition.** The internal activation pattern produced by the shared MLP trunk (3×256) after reading the state vector. The compressed internal representation of "what the bot thinks is going on right now."

**Why it matters for passing.** This is the closest practical thing to the bot's internal thought process. It's where market, trade, account, and challenge information are combined before producing actions and value estimates. If the hidden state can't tell breach-risk from normal trading, the bot will breach.

**Attributes.**
- Per-layer activation values (post-activation summaries; full vectors at sampled steps)
- Activation magnitude + sparsity (fraction of neurons firing)
- Hidden vector geometry across similar states (do similar challenge situations cluster?)
- Stability across seeds + windows (does the same regime produce similar hidden states across runs?)

**Methods.** Compress many inputs into a decision-ready representation. Separate meaningful regimes internally. Carry information useful to BOTH actor and critic heads.

**Two most connected terms.** Action Distribution · Value Estimate

**Connection rules.**
- Direction + size outputs must be explainable as downstream consequences of the hidden state.
- The critic's V(s) must also be explainable as a downstream consequence of the SAME hidden representation. Both heads see the same trunk — divergent failures often start here.

**Healthy.** Clusters similar challenge situations together · separates safe from dangerous account states · becomes distinctive near breach-risk · produces repeatable internal patterns for premium legal setups across seeds.

**Unhealthy. Two named failure modes (the LLM uses these labels):**
- **Representation Collapse** — different situations compress into indistinguishable hidden states. The bot can no longer tell them apart.
- **Representation Chaos** — very similar situations map to wildly different hidden states. The bot's internal world is unstable.

**Diagnosis rule.** When a pass-rate failure appears, the LLM checks hidden-state quality FIRST. If hidden states for breach-risk look identical to safe states, the actor and critic were doomed regardless of training quality.

---

## TERM 4 — ACTION DISTRIBUTION [P]

**Definition.** The output probability structure of the actor heads. In this architecture, that's THREE action heads (per the locked four-head design):
1. **Direction head** — categorical over {HOLD, OPEN_LONG, OPEN_SHORT, CLOSE}, masked per position state
2. **Size head** — Beta distribution over raw_size ∈ [0, 1]
3. **Pointer head** — categorical over the 5 trade slots, only used on CLOSE

(The 4th head is the value/critic — covered under Term 5.)

**Why it matters for passing.** Passing the challenge depends on whether the model takes the right kind of action at the right time, with the right confidence and restraint. Overconfidence near the wall = breach. Restraint in premium legal states = missed target.

**Attributes.**
- Unmasked logits (what the actor "wanted" to do)
- Masked logits (what laws allowed)
- Final action probabilities (post-mask, sampled or argmax in live)
- Chosen action + alternates
- raw_size output (Beta sample in training, Beta mean live)
- Post-RiskManager feasible size (RiskManager may shrink raw_size below daily-risk-buffer constraints)
- Pointer head output on CLOSE (which slot, with what confidence)

**Methods.** Score legal actions. Express uncertainty. Translate internal belief into trade behavior.

**Two most connected terms.** Law Context · Risk Context

**Connection rules.**
- An action distribution is MEANINGLESS without knowing what was legal at that step (Term 2).
- A size output is INCOMPLETE without knowing what RiskManager feasibility allowed afterward (Term 6).

**Healthy.** High confidence in premium legal setups · restraint in weak or noisy setups · smaller or no positions near danger · clean exits when risk worsens or edge fades.

**Unhealthy. Named failure modes:**
- **Mask Dependence** — pre-mask logits repeatedly favor illegal actions. The actor "wanted" to do the wrong thing; it just got lucky the law saved it.
- **Risk Blindness** — high-aggression action distributions persist into breach-risk states.
- **Stagnation Blindness** — flat/HOLD-dominated distributions in favorable legal contexts.

**Diagnosis rule.** Always compare action distribution against TWO contexts side by side: what was legal, and what risk context permitted. A flagged "bad action" is only diagnostic if it was both legal AND risk-acceptable.

---

## TERM 5 — VALUE ESTIMATE [P]

**Definition.** The critic head's forecast V(s) of expected future return from the current state under the current policy. In this project, it powers GAE (γ=0.997, λ=0.97) and the "reason-to-hold" signal — patience under challenge constraints.

**Why it matters for passing.** The critic is how the model learns patience, continuation, and delayed gratification — instead of reacting bar-to-bar. A broken critic causes the bot to exit winners early or hold losers too long, both fatal for consistent passing.

**Attributes.**
- Current V(s)
- Advantage estimate Â (GAE)
- TD error (one-step prediction miss)
- Temporal consistency (does V(s) move smoothly or thrash?)
- Correlation with realized outcomes (does high V(s) actually predict good outcomes?)
- Correlation with reward layers (which reward layer is the critic "listening to"?)

**Methods.** Estimate future usefulness of current state. Support hold/continue decisions. Shape learning through advantage signals.

**Two most connected terms.** Hidden State · Reward Decomposition

**Connection rules.**
- Value comes from the hidden state, so value failures often trace back to poor internal representations (Term 3).
- Value must align better with **Layer 0 outcome + challenge safety** than with decorative shaping alone (Term 7 E8 dominance rule).

**Healthy.** Higher value in calm legal progress states · weakening value near stagnation or pain-zone approach · useful hold signals during productive in-trade continuation · poor states forecast as poor BEFORE the breach actually lands.

**Unhealthy. Named failure mode:**
- **Critic Misalignment** — V(s) and advantage fail to reflect true challenge-quality states. The critic is paying attention to the wrong thing.

**Diagnosis rule.** When the actor looks reasonable but pass rate still degrades, the LLM examines the critic next. "Right actions, wrong timing" almost always means the critic is broken, not the actor.

---

## TERM 6 — RISK CONTEXT [M+P]

**Definition.** The current account position relative to hard FTMO constraints: daily-loss proximity, 4% trailing wall proximity, target status, remaining daily-risk buffer.

**Why it matters for passing.** This project is not allowed to ignore challenge physics. Risk context is what makes an "otherwise profitable" action UNACCEPTABLE in challenge terms.

**Attributes.**
- Daily drawdown proximity (% of daily loss limit consumed)
- Trailing drawdown proximity (distance to the 4% wall)
- Remaining daily risk buffer (what RiskManager has left to spend)
- Day PnL vs DAILY_TARGET
- Post-target state (after the +2.5% auto-flat, the 1% Phase B trailing applies)
- Breach-risk category (the only HARD-CODED macro category; others are soft-learned)

**Methods.** Inform position restraint. Change what "good behavior" means near the wall. Reweight reward context through Layer 5 breach-risk handling.

**Two most connected terms.** Action Distribution · Reward Decomposition

**Connection rules.**
- Action size and willingness to act MUST adapt to risk context.
- Reward shaping near danger must preserve Layer 0 dominance but strongly favor protection (Layer 3 pain-zone is exponential 3.5%→4.0%).

**Healthy.** Lower aggression near daily wall · cleaner exits under worsening pain-zone pressure · preservation of pass-day status · no reckless "one more trade" behavior near failure thresholds.

**Unhealthy.** Action distributions identical at 0% drawdown and 3.8% drawdown · size head outputs the same Beta mean regardless of buffer remaining.

**Diagnosis rule.** For ANY breach episode, the LLM walks backward from the breach moment: when did risk context become dangerous? When did the action distribution shift? The gap between those two moments is the bot's "danger blindness window" — the most actionable diagnostic.

---

## TERM 7 — REWARD DECOMPOSITION [M+P]

**Definition.** The split of total reward into its defined layers: Layer 0 net PnL · Layer 1 momentum · Layer 2 stagnation · Layer 3 pain zone · Layer 4 target progress · Layer 5 category weighting · Layer 6 daily bonus (E7 streak + E9 QUAD bonus).

**Why it matters for passing.** The project rule is absolute (Layer 7/E8): **the bot must never win the reward game while losing the trading game.** Layer 0 (net PnL after costs) must dominate. Decomposition is how we PROVE that's still true at any point in training.

**Attributes.** Per-layer contribution to total reward, episode-aggregated AND per-step. Plus: the QUAD bonus signals (Drawdown Efficiency, Law Productivity, Target Velocity payable; TD-stability qualifier-only).

**Methods.** Reveal what the training signal is ACTUALLY paying for. Diagnose shaping dominance. Connect critic behavior to reward truth.

**Two most connected terms.** Value Estimate · Outcome

**Connection rules.**
- Value should learn from the real reward structure, not a fake simplified view of it.
- Outcome analysis must check whether reward layers were aligned with real challenge passing.

**Healthy.** Layer 0 dominates · shaping layers stay secondary · pain-zone pressure visible near risk limits · momentum and stagnation help timing without hijacking the objective · QUAD bonus stays under its 95% ceiling.

**Unhealthy. Named failure mode:**
- **Reward Hijack** — shaping layers (1, 2, 4, or QUAD components) influence behavior MORE than Layer 0. The bot is gaming the shaper, not learning to trade.

**Diagnosis rule.** When the bot's behavior looks coherent but pass rate is bad, the LLM checks reward decomposition. If any single shaping layer's cumulative contribution exceeds Layer 0 over a window, that's a reward hijack — and the bot's strategy is optimizing for the wrong thing.

---

## TERM 8 — OUTCOME [P]

**Definition.** What actually happened after the decision: short-horizon price behavior, trade result, account result, FTMO challenge result.

**Why it matters for passing.** Interpretability is useless if it never reconnects internal belief to real consequences. Outcome is the truth that grades the chain.

**Attributes.**
- Next-bar / next-window price move
- Trade PnL + trade quality
- Contribution to day target
- Contribution to breach
- Pass/fail episode label
- Validation score impact (the scoreboard hierarchy: pass rate → breach count → target-hit consistency → max DD path)

**Methods.** Close the loop between internal state and real-world result. Support post-hoc diagnosis. Evaluate whether the model's internal beliefs were actually useful.

**Two most connected terms.** Action Distribution · Reward Decomposition

**Connection rules.**
- Actions create outcomes.
- Reward should correlate with outcomes in challenge-relevant ways, not in incidentally-profitable ways.

**Healthy.** Repeated target attainment without breach · low unnecessary heat · fewer law-adjacent mistakes · better pass-rate consistency across windows and seeds.

**Unhealthy. Named failure mode:**
- **Shortcut Learning** — the network keys on incidental state cues (date, symbol, time of session) instead of durable market/challenge structure. Performance collapses when the cues change.

**Diagnosis rule.** The LLM correlates outcome with hidden-state geometry (Term 3): if pass-day outcomes share a hidden-state cluster, the bot has learned a real pattern. If pass-day outcomes are scattered across hidden states, the bot is passing by luck — and won't keep passing.

---

## TERM 9 — TELEMETRY LOGGER [P]

**Definition.** The module that records the structured evidence needed to reconstruct what happened at input, hidden, output, reward, and risk levels.

**Why it matters.** No telemetry → no interpretability. If it wasn't logged, it cannot be diagnosed later. The LLM cannot reason from evidence that doesn't exist.

**Attributes.** Schema version · Run ID, seed, window ID, episode ID · Per-step packet · Per-trade summaries · Per-day summaries · Artifact index.

**Methods.** Capture raw facts. Preserve replayability for diagnostics. Feed both MLPInterpreter and LLMRiskDoctor.

**Two most connected terms.** MLP Interpreter · LLM Risk Doctor

**Connection rules.** Interpreter computes analyses from logger output. Risk Doctor consumes interpreter results PLUS raw summaries.

**Healthy.** Challenge-relevant events are never missing · breach paths are reconstructable · target-hit days are reconstructable · same schema works across train/validate/deterioration review.

**Diagnosis rule for the LLM.** Before producing any diagnosis, the LLM checks the telemetry version, completeness, and whether the events being analyzed were actually fully logged. A diagnosis based on partial telemetry must say so out loud.

---

## TERM 10 — MLP INTERPRETER [P]

**Definition.** The analysis module that converts raw telemetry into readable diagnostics: plots, metrics, comparisons, failure atlases.

**Why it matters.** Raw arrays don't explain anything. This module turns hidden mechanics into operator-usable evidence — and into the structured visuals the LLM reasons from.

**Attributes.** Plot generators · Projection tools (PCA, UMAP, t-SNE) · Correlation analyzers · Attribution analyzers · Episode trace builders · Failure atlas builders · Pass-Day atlas builders.

**Methods.** Transform hidden vectors into visual structure. Link internal states to action/value/reward/outcomes. Produce standard reports for every major run.

**Two most connected terms.** Hidden State · LLM Risk Doctor

**Connection rules.** Reads hidden states + output traces. Produces the interpretable package the Risk Doctor uses.

**Healthy.** Makes it easy to answer: what the bot thought · why it acted · where it got confused · why passes succeeded · why breaches happened.

---

## TERM 11 — LLM RISK DOCTOR [M+P]

**Definition.** Offline supervisory diagnosis module that reads telemetry + interpretability artifacts and produces **structured failure analysis + training prescriptions**.

**Why it matters.** The LLM must reason from real internal evidence, not vague scoreboard summaries. This entire file exists to give it that evidence.

**Attributes.** Input contract · Diagnosis schema · Failure taxonomy (below) · Prescription schema · Confidence + evidence fields · HARD BOUNDARY: no execution authority.

**Methods.** Diagnose actor, critic, reward, and regime problems. Detect likely failure modes. Suggest curriculum / telemetry / reward / restriction reviews.

**Two most connected terms.** Telemetry Logger · MLP Interpreter

**Connection rules.**
- It must NEVER invent explanations unsupported by telemetry.
- Every diagnosis must cite evidence from logged facts and derived visuals.

**Healthy Risk Doctor behavior.** Identifies true causes of pass-rate deterioration · separates law, actor, critic, and reward issues · does not overreact to raw PnL noise · stays focused on challenge consistency.

---

# REQUIRED DATA CONTRACT [C]

Every step packet MUST include, at minimum:

- **IDs:** run, seed, window, episode, timestep, symbol
- **Time:** timestamp, bar index
- **Observation:** full normalized state vector + grouped feature block names
- **Law:** active laws/gates and enforcement mode, legal actions before sampling
- **Policy outputs:** pre-mask logits, post-mask logits, action probabilities, chosen action, **pointer-head output (when CLOSE)**, raw_size, feasible size after RiskManager
- **Critic:** V(s) output
- **Trunk:** hidden layer vectors or compressed summaries (per-layer)
- **Reward:** decomposition by every layer + QUAD bonus signal states
- **Risk:** risk context snapshot (all Term 6 attributes)
- **Outcome:** short-horizon outcome labels, trade lifecycle link if in position

**Schema MUST be versioned.** No future refactor may remove these hooks without explicit approval from Monty.

---

# REQUIRED VISUALS

Every significant run must produce these standard artifacts.

### 1. Activation Trace
Neuron activity over time around important events. **Healthy:** coherent shifts before decisions, not random bursts.

### 2. Hidden-State Projection
PCA first; UMAP/t-SNE optional. Color by action, law state, breach-risk, pass/fail, symbol. **Healthy:** meaningful clustering by challenge-relevant regime.

### 3. Action/Value Timeline
Action probabilities, chosen action, raw size, feasible size, V(s) over time. **Healthy:** confidence and value rise in premium legal states and soften near danger.

### 4. Reward Layer Timeline
All reward layers + QUAD components through episode/trade. **Healthy:** Layer 0 dominates; shaping layers modulate but never overpower.

### 5. Correlation Heatmap
Input blocks ↔ hidden neurons ↔ outputs ↔ reward layers. **Healthy:** interpretable structure, not noise or single-feature domination.

### 6. Failure Atlas
Standardized multi-panel view for breach episodes, stagnation episodes, law-adjacent failures. **Healthy:** failures become classifiable.

### 7. Pass-Day Atlas
Standardized multi-panel view for clean pass days. **Healthy:** passing behavior shows stable recurring internal signatures.

---

# FAILURE TAXONOMY — THE LLM CLASSIFIES EVERY FAILURE INTO ONE OF THESE

For each, the LLM must cite the SPECIFIC evidence field/visual that justifies the classification.

| Failure | Definition | Required evidence | Typical prescription |
|---|---|---|---|
| **Mask Dependence** | Pre-mask logits repeatedly favor illegal actions | Logit distribution showing illegal actions dominant pre-mask across N episodes | Add observation features for the law ingredients the bot is ignoring; consider law-school re-exposure |
| **Representation Collapse** | Different challenge situations compress to indistinguishable hidden states | PCA/UMAP showing breach-risk and safe states overlapping | Reduce trunk capacity reuse; check whether challenge features are reaching the trunk |
| **Representation Chaos** | Very similar situations map to unstable hidden states | High intra-cluster variance for same-regime states across seeds | Lower learning rate; check rollout/minibatch ratio; check entropy not too high |
| **Critic Misalignment** | V(s) + advantage fail to reflect true challenge quality | Low correlation between V(s) and pass/fail outcomes; TD error fails to shrink near danger | Reward layer audit; check Layer 0 dominance; possibly lengthen γ horizon |
| **Reward Hijack** | A shaping layer's cumulative contribution > Layer 0 over the window | Per-layer reward integral showing shaping > Layer 0 | Reduce shaping weights; verify E8 dominance rule; potentially disable Layer 6 QUAD toggle |
| **Risk Blindness** | Aggression persists into breach-risk states | Action distribution unchanged between safe and 3.5%+ drawdown buckets | Strengthen Layer 3 pain-zone curve; expand risk context features in observation |
| **Stagnation Blindness** | Flat/HOLD dominates in favorable legal contexts | HOLD prob > 0.7 in confirmed premium law-active windows | Check Layer 2 stagnation weight; verify Target Velocity signal in QUAD bonus |
| **Shortcut Learning** | Bot keys on incidental cues (date, symbol, session) | Pass-day outcomes scatter across hidden states; performance drops on held-out symbols/dates | Walk-forward re-validation with stratified holdout; audit observation for accidental leakage |

**Rule for the LLM:** If a failure doesn't fit any of these eight, the LLM does NOT invent a ninth. It reports "unclassified pattern — additional telemetry required" and stops.

---

# FLOW RULES — THE CHAIN THE LLM REASONS ALONG

1. **State Vector** describes market + trade + account + challenge + law context
2. **Law Context** defines what actions are legal
3. **Shared MLP Trunk → Hidden State** forms the internal belief
4. **Actor Heads** (direction, size, pointer) convert belief into trade preferences
5. **Critic Head** converts belief into future-value estimate
6. **RiskManager + Hard Walls** convert raw intentions into executable challenge-safe behavior
7. **Reward System** trains the policy with Layer 0 dominant challenge-aware feedback
8. **Outcome** reveals what actually happened
9. **TelemetryLogger** records the full chain
10. **MLPInterpreter** turns the chain into readable diagnostics
11. **LLMRiskDoctor** turns diagnostics into structured diagnosis + prescription

**When diagnosing a failure, the LLM walks this chain in REVERSE — outcome → reward → critic → actor → hidden state → law → state vector — and stops at the first link where evidence shows the break.**

---

# THE LLM'S DIAGNOSTIC OUTPUT TEMPLATE

Every diagnosis the Risk Doctor produces must follow this shape:

```
DIAGNOSIS — [run_id, window_id, severity]

What happened (outcome layer):
  [pass rate / breach count / target-hit metrics — pure facts, no interpretation]

Where the chain broke:
  [the FIRST link in the reverse-walk where evidence shows the problem]

Failure classification:
  [one of the 8 taxonomy items — or "unclassified, more telemetry needed"]

Evidence cited:
  - [specific telemetry field, visual, or metric — never vague]
  - [...]

Confidence:
  [HIGH / MEDIUM / LOW — LOW means say "uncertain" out loud]

Prescription:
  [curriculum / reward / telemetry / restriction recommendation — NOT execution]

Not recommended:
  [actions the operator might be tempted to take that this evidence does NOT support]
```

If any section can't be filled with evidence, the LLM writes `"insufficient evidence"` in that section. It does NOT speculate to fill the blank.

---

# CLAUDE CODE CONTRACT [C]

Claude Code must implement this file as a permanent add-on spec connected to:
- TelemetryLogger
- MLPInterpreter
- LLMRiskDoctor
- G7 telemetry hooks
- I4 Risk Doctor input contract

Claude Code MUST preserve:
- Same terminology used here (the LLM relies on these exact term names)
- Same schema names in the data contract
- Same safety boundary (no execution authority)
- Same challenge-passing purpose (not generic ML accuracy)
- Same Layer 0 dominance rule
- Same law-before-reward rule

Claude Code MUST NOT:
- Reduce this to generic "explainability"
- Use visuals that are pretty but unconnected to passing
- Allow the Risk Doctor to infer diagnoses without telemetry evidence
- Attach any execution authority to the LLM Risk Doctor

## STANDING RULE — SHOW THE WORK [2026-06-19, operator]

Applies to **every file in this project, present and future** (the per-file "UPDATE LOG (IRAC)"
headers point here as the rulebook). On **every** edit, Claude Code must:

1. **Append a dated IRAC entry** to that file's update log — `[YYYY-MM-DD] <tag> — <summary>` then
   **I**ssue / **R**ule / **A**pplication / **C**onclusion (Conclusion = why it helps the bot pass
   FTMO more consistently, with no new bug or inefficiency).
2. **Document the cross-file RELATIONSHIPS** the change depends on, inline, with a `COUPLING ->`
   comment that names the **other file(s)** and the **exact attribute / field / dict-key** relied on,
   in **both directions** (what this file reads from there, and what there must keep stable for this
   file). State what breaks if the coupled name changes.
3. **Date the re-pointed logic** — tag changed math/branches with the edit date (e.g. `# C17
   [2026-06-19] ...`) so a future reader sees *what* connects to *what*, *what breaks where*, and
   *when it changed*.

This is how the project stays legible and the LLM Risk Doctor can reconstruct the chronological
"why". New files start with the same IRAC header + this rulebook reference and follow the same rule.

---

# DONE CRITERIA [C]

This add-on is "done" only when the codebase produces, for any important run:
1. A reconstructable decision trace (telemetry replays the chain)
2. Hidden-state diagnostics (the 7 required visuals)
3. Action/value/reward timelines
4. Pass-day AND failure atlases
5. Evidence-backed LLM diagnosis following the output template
6. All of it framed around repeated FTMO-style challenge passing — never raw PnL

If any of these is missing, the layer is incomplete.

---

*This file is binding for both build (Claude Code) and inference (LLM Risk Doctor). Same words, two readers, one purpose: turn the PPO MLP from a black box into an inspectable machine, so the LLM can give accurate, evidence-based diagnoses tied to passing FTMO consistently.*
