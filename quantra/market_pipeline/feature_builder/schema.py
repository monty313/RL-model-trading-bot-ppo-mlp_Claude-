"""StateVectorSchema — the canonical, ordered layout of the observation vector.

WHAT THIS MODULE DOES
---------------------
Defines EXACTLY which scalars the policy sees, in what order, grouped into named
blocks, so the total width is fixed and asserted everywhere. This is the contract
the FeatureBuilder fills, the env assembles, the PPO trunk consumes, and the
TelemetryLogger labels (the data contract requires "grouped feature block names").

Block widths (default INCLUDE_RAW_INPUTS=True -> total 203):
    market     128   market+time + gate ingredients + RAW CCI + RAW Bollinger + training wheels
    market_raw  18   RAW price-SMA inputs (operator-directed; precomputed)
    law         12   9 laws + 3 gates (filled by LawMask, M3)
    trade x5    35   7 features x 5 slots (env, M4)
    portfolio    3   aggregates across slots (env, M4)
    account      7   equity/buffers + 2 challenge-progress (env, M4)
    ------------------
    TOTAL      203   (185 when INCLUDE_RAW_INPUTS=False)

CCI is kept RAW (operator decision 2026-06-13): no /100, no normalized deviation —
the raw CCI value + its raw shifted-forward SMA(2, shift4) are exposed so the policy
compares CCI to its location 4 bars ago. Those CCI columns are in RAW_FEATURE_NAMES
(unclipped). The first TWO blocks (market + market_raw = 128) are action-independent
and are the FeatureBuilder's precomputed output (``PRECOMPUTED_NAMES`` / ``PRECOMPUTED_DIM``).

OPERATOR OVERRIDE [2026-06-13]
-----------------------------
``market_raw`` holds RAW indicator levels (SMA period 1 shift 0-3, SMA 30/50 on
5m/30m/4H; raw CCI 10/30/100 on 1m/5m/30m/4H). This is an operator-directed addition
that intentionally departs from STATE_VECTOR.md's "never feed raw prices" encoding
rule. Risks + the required safeguard (input standardization in the agent) are written
in ``RAW_INPUTS.md`` and flagged via ``RAW_FEATURE_NAMES`` so the M5 agent standardizes
them. Toggle with ``quantra.runtime.config.INCLUDE_RAW_INPUTS``.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
A frozen, named, asserted observation guarantees the bot always sees the full
FTMO-relevant picture and that telemetry can map any neuron to its driving feature
(MLP_INTERPRETABILITY_LAYER Term 1) — the precondition for diagnosable, repeatable
passing. The raw block is delineated + flagged so the LLM can tell whether a
pass-rate problem traces to an unnormalized input (shortcut learning / instability)
versus the policy itself.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Use ``block_slice(name)`` /
``FEATURE_NAMES`` to know which observation indices belong to which block. The
``market_raw`` indices are UNNORMALIZED — if you see Representation Chaos or
Shortcut Learning, check whether a raw feature is dominating (large magnitude,
single-feature attribution) before blaming the trunk; the prescription is usually
"standardize the raw block / re-fit input stats", not a trunk change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

# COUPLING [C1] -> quantra/runtime/config.py: INCLUDE_RAW_INPUTS toggles whether the
# market_raw block exists, which changes STATE_DIM (167 vs 149); config.nominal_state_dim
# and the committed state_vector.json snapshot must be regenerated when this flag flips.
from quantra.runtime.config import INCLUDE_RAW_INPUTS

# ---------------------------------------------------------------------------
# Timeframe groupings per the law/observation specs. 4H is ALWAYS observation-only.
# ---------------------------------------------------------------------------
BOLL_TFS = ["5m", "30m", "4H"]
CCI_TFS = ["1m", "5m", "30m", "4H"]
ATR_TFS = ["1m", "5m", "30m", "4H"]
SSMA_TFS = ["1m", "5m", "30m", "4H"]
Z_TFS = ["1m", "5m", "30m", "4H"]
ADX_TFS = ["1m", "30m", "4H"]

_BB_BANDS = ["bb20_mid", "bb20_up", "bb20_lo", "bb200_mid", "bb200_up", "bb200_lo"]
_CCI_PERIODS = [10, 30, 100]

# TRAINING-WHEEL feature set [operator decision 2026-06-15]. Ingredients for the two
# semi-permanent counter-trend OPEN-block masks (CCI-based + BB-based), computed on
# 30m+4H. ADDITIVE + observation-only as ingredients; the derived block flags
# (tw_cci_block / tw_bb_block) are what the mask engine enforces when
# config.TRAINING_WHEELS is on. 4H is used here by operator override of the locked
# "4H observation-only" rule — kept isolated from the locked 9 laws. COUPLING [C9]
# -> feature_builder/indicators.py (WHEEL_* params), feature_builder/builder.py
# (emits these), law_mask_engine/engine.py (reads the flags by name via env._COL).
_WHEEL_TFS = ["30m", "4H"]

# Raw-input groupings (operator override). Raw SMA on the trend TFs; raw CCI on the
# full CCI observation TF set.
# COUPLING [C1] -> feature_builder/builder.py: RAW_SMA_TFS is imported there to gate which
# timeframes emit raw_sma* columns; the names produced must match _market_raw_names() below.
RAW_SMA_TFS = ["5m", "30m", "4H"]
RAW_CCI_TFS = ["1m", "5m", "30m", "4H"]


def _market_names() -> List[str]:
    """The 89 NORMALIZED market+time feature names (bounded, ATR-scaled, clipped)."""
    names: List[str] = []
    for tf in BOLL_TFS:
        for band in _BB_BANDS:
            names.append(f"boll_{band}_{tf}")          # NORMALIZED: (close - band) / ATR14
    # RAW Bollinger band levels [operator decision 2026-06-15: keep BOTH the normalized
    # ATR-scaled distance AND the unnormalized raw band PRICE level]. UNCLIPPED (price
    # levels) -> in RAW_FEATURE_NAMES; the M5 agent standardizes them.
    # COUPLING [C1] -> builder._compute_tf_features (emits these), RAW_FEATURE_NAMES (below).
    for tf in BOLL_TFS:
        for band in _BB_BANDS:
            names.append(f"boll_{band}_raw_{tf}")        # RAW band price level
    # CCI kept RAW [operator decision 2026-06-13]: expose the raw CCI value AND its
    # raw shifted-forward SMA (period 2, shift 4) so the policy compares current CCI
    # against the CCI's location 4 bars ago — NOT a normalized deviation. No /100.
    # COUPLING: these exact names are read by quantra.locked_core.laws.laws (CCI laws),
    # listed in builder._compute_tf_features, masked in curriculum._EARLY_MASK_1M (1m),
    # and flagged RAW below (RAW_FEATURE_NAMES, unclipped). Rename here => change all.
    for tf in CCI_TFS:
        for p in _CCI_PERIODS:
            names.append(f"cci{p}_{tf}")          # RAW CCI value (~[-300, 300], unclipped)
            names.append(f"cci{p}_sma_{tf}")       # RAW shifted-forward SMA(CCI, 2, sh4)
        names.append(f"cci_sync_{tf}")             # +1/0/-1 flag (raw CCI vs its SMA)
    names.append("cci_pullback_5m")                # +1/0/-1 flag
    for tf in ATR_TFS:
        names += [f"atr_level_{tf}", f"atr_ref_{tf}", f"atr_dev_{tf}"]
    for tf in SSMA_TFS:
        names += [f"ssma_high_dist_{tf}", f"ssma_low_dist_{tf}", f"ssma_align_{tf}"]
    for tf in Z_TFS:
        names += [f"z10_{tf}", f"z100_{tf}"]
    for tf in ADX_TFS:
        names += [f"adx5_{tf}", f"adx15_{tf}"]
    names += ["candle_return_1m", "candle_range_1m", "candle_uwick_1m", "candle_lwick_1m"]
    names += ["time_sin_hour", "time_cos_hour", "time_dow"]
    # Gate ingredients [M3]: Spread Filter (spread vs ATR + vs candle range) and
    # Stationarity Regime Gate (rolling Dickey-Fuller stat). Observable per the
    # law-ingredient coverage rule so the bot sees why a gate opened/closed.
    names += ["spread_atr_1m", "spread_range_ratio_1m", "adf_stat_1m"]
    # TRAINING-WHEEL ingredients + block flags [operator 2026-06-15]. APPENDED at the
    # END of the market block so every pre-existing feature index is unchanged. 8 RAW
    # CCI(5,15) value+SMA(20,sh0) on 30m/4H (in RAW_FEATURE_NAMES, unclipped) + 8
    # NORMALIZED BB(10,100,dev0.5) upper/lower distances (close-band)/ATR on 30m/4H
    # (clipped) + 2 aggregate three-way block flags. The flags are what the mask
    # engine reads; the ingredients make them interpretable (the "acts" the operator
    # wants observable before they become enforced "laws").
    for tf in _WHEEL_TFS:
        for p in (5, 15):
            names.append(f"tw_cci{p}_{tf}")        # RAW CCI value (unclipped)
            names.append(f"tw_cci{p}_sma_{tf}")     # RAW applied SMA(20, shift 0)
        for p in (10, 100):
            names.append(f"tw_bb{p}_up_{tf}")       # (close - upper band) / ATR14
            names.append(f"tw_bb{p}_lo_{tf}")       # (close - lower band) / ATR14
    # +1 = uptrend context (blocks OPEN_SHORT) / -1 = downtrend (blocks OPEN_LONG).
    names += ["tw_cci_block", "tw_bb_block"]
    return names


def _market_raw_names() -> List[str]:
    """RAW price-SMA inputs (operator override). UNNORMALIZED level values.

    18 raw SMA: sma1 shift 0-3, sma30, sma50 on 5m/30m/4H.
    (Raw CCI moved into the main `market` CCI block when CCI was un-normalized
    [2026-06-13] — the old `raw_cci{p}` here was then a duplicate, so it was removed.)
    """
    if not INCLUDE_RAW_INPUTS:
        return []
    names: List[str] = []
    for tf in RAW_SMA_TFS:
        for k in (0, 1, 2, 3):
            names.append(f"raw_sma1_sh{k}_{tf}")     # SMA period 1 = price, shifted k
        names.append(f"raw_sma30_{tf}")
        names.append(f"raw_sma50_{tf}")
    return names


# COUPLING [C4] -> quantra/locked_core/laws/laws.py (LAW_NAMES) + law_mask_engine/engine.py
# (_OBS_IDX/_LAW_IDX, [:9] dir-slice): this is the canonical 12-name law block — 9
# directional laws THEN 3 market-condition observation signals (formerly "gates"). The 3
# signals are OBSERVATION-ONLY by default (config.TRAINING_PHASE == PHASE_FREE); only the
# stationarity signal re-enforces in PHASE_CONSTRAINED. laws.LAW_NAMES must mirror this
# exactly; the mask engine hardcodes offset 9 for the signals. Reorder/rename => break both.
def _law_names() -> List[str]:
    return [
        "law_super_trend_bb", "law_super_trend_cci", "law_super_trend_ssma",
        "law_trend_bb", "law_trend_cci", "law_trend_ssma",
        "law_pullback_bb", "law_pullback_cci", "law_pullback_ssma",
        "market_volatility_obs", "market_spread_obs", "market_stationarity_obs",
    ]


# COUPLING [C3 in COUPLINGS.md]: N_SLOTS is the source of truth for trade slots. It
# sets the trade block width (7 x N_SLOTS = 35), the agent pointer head width, the
# RepresentativePolicy pointer logits, and the env/live slot arrays. Change => update
# ppo_agent, runtime.device, env.trading_env, execution_adapter, live_session, mask engine.
N_SLOTS = 5
# COUPLING [C1/C3] -> quantra/env/trading_env.py (_COL) + learning_system/trainer/scheduler.py:
# these 7 per-slot feature names AND their order define each slot's sub-vector that the
# env fills (slot{s}_{f}) and the trainer/scheduler reads by name. Reorder/rename => fix _COL.
_SLOT_FEATURES = ["dir", "upnl", "age", "entry_dist", "mfe", "mae", "occupied"]  # 7 -> trade block


def _trade_names() -> List[str]:
    return [f"slot{s}_{f}" for s in range(N_SLOTS) for f in _SLOT_FEATURES]


# COUPLING [C1] -> quantra/env/trading_env.py: these 3 portfolio aggregate names + order
# are filled by the env per step (the "portfolio" sub-vector of assemble_state); change => fix env.
def _portfolio_names() -> List[str]:
    return ["port_net_exposure", "port_net_size", "port_total_upnl"]


# COUPLING [C1] -> quantra/ftmo_passing/challenge_state.py (account_block()) + env/trading_env.py:
# the order of these 8 account names IS the order ChallengeState.account_block() must emit
# and the env appends to the obs. Add/reorder => fix account_block() + EXPECTED_WIDTHS["account"].
# CHANGED: 2026-06-18 | Added acct_dist_to_perm_dd (C12 — 8th account scalar, appended at END)
# WHY: bot's observation now includes runway to the permanent max-overall-loss wall (survival)
# AFFECTS: challenge_state.account_block() (emits it), EXPECTED_WIDTHS["account"] 7->8, STATE_DIM 206->207
def _account_names() -> List[str]:
    return [
        "acct_equity_norm", "acct_equity_dev", "acct_equity_slope",
        "acct_trailing_buffer", "acct_daily_buffer",
        "acct_day_progress", "acct_overall_progress", "acct_dist_to_perm_dd",
    ]


# Ordered blocks: (name, builder). Order IS the observation order. market + market_raw
# are the precomputed (action-independent) blocks.
# COUPLING [C1] -> quantra/env/trading_env.py (assemble_state block order) + feature_builder/builder.py:
# this tuple order sets the concatenation order in builder.assemble_state and the env's
# block offsets; _PRECOMPUTED_BLOCKS below selects which blocks builder precomputes.
_BLOCK_BUILDERS = [
    ("market", _market_names),
    ("market_raw", _market_raw_names),
    ("law", _law_names),
    ("trade", _trade_names),
    ("portfolio", _portfolio_names),
    ("account", _account_names),
]
_PRECOMPUTED_BLOCKS = ("market", "market_raw")


@dataclass(frozen=True)
class StateVectorSchema:
    """Frozen, validated layout of the observation vector."""

    blocks: Dict[str, List[str]]
    feature_names: List[str]
    block_spans: Dict[str, Tuple[int, int]]

    @property
    def dim(self) -> int:
        return len(self.feature_names)

    def block_slice(self, name: str) -> slice:
        s, e = self.block_spans[name]
        return slice(s, e)

    def index_of(self, feature: str) -> int:
        return self.feature_names.index(feature)


def build_schema() -> StateVectorSchema:
    """Assemble + validate the canonical schema. Raises if names aren't unique."""
    blocks: Dict[str, List[str]] = {}
    feature_names: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for name, fn in _BLOCK_BUILDERS:
        block = fn()
        start = len(feature_names)
        feature_names.extend(block)
        spans[name] = (start, len(feature_names))
        blocks[name] = block
    if len(set(feature_names)) != len(feature_names):
        dupes = sorted({n for n in feature_names if feature_names.count(n) > 1})
        raise ValueError(f"duplicate feature names in schema: {dupes}")
    return StateVectorSchema(blocks=blocks, feature_names=feature_names, block_spans=spans)


# Singleton + public constants.
SCHEMA = build_schema()
# COUPLING [C1] -> quantra/runtime/config.py (nominal_state_dim must == STATE_DIM),
# quantra/ppo_agent/agent.py (reads STATE_DIM for trunk input), tests/snapshots/state_vector.json
# (re-pin via tools/snapshot.py --update). FEATURE_NAMES order is logged by the telemetry
# logger + indexed by env._COL/laws._IDX. Change STATE_DIM/order => update all of these.
# ⚠️ COMPATIBILITY [C18+] -> quantra/learning_system/policy_registry/registry.py
# (compatibility_signature): STATE_DIM is the FIRST input to a policy's compatibility hash. Changing
# it (e.g. toggling INCLUDE_RAW_INPUTS, or adding/removing any feature block) changes EVERY saved
# policy's signature, so the registry refuses to RESUME old checkpoints (CompatibilityError — the old
# net's input layer no longer fits) and they must be RETRAINED fresh. This is the #1 "can't go back to
# an old policy" hazard the operator flagged: if you change the dim here, expect a forced fresh start.
STATE_DIM = SCHEMA.dim                                   # 207 (raw on) / 189 (raw off)
FEATURE_NAMES = SCHEMA.feature_names

# The precomputed (action-independent) feature set = market + market_raw, in order.
# COUPLING: this ORDER is the FeatureBuilder's output column order AND the index map
# used by laws._IDX, env._COL, and the live session. Reorder here => reorder there.
PRECOMPUTED_NAMES = SCHEMA.blocks["market"] + SCHEMA.blocks["market_raw"]
PRECOMPUTED_DIM = len(PRECOMPUTED_NAMES)                 # 146 (raw on) / 128 (raw off)

# Backwards-compatible aliases (the normalized block only).
MARKET_NAMES = SCHEMA.blocks["market"]
MARKET_DIM = len(MARKET_NAMES)

# RAW features bypass the FeatureBuilder ±CLIP and must be standardized by the M5
# agent's input layer (they are unbounded price/CCI levels). The LLM reads this set
# to know which observation indices are unnormalized.
# COUPLING: builder.build_market_matrix clips ONLY columns NOT in this set; the M5
# agent must standardize every name in here. CCI raw features live in the `market`
# block but are listed here too (they are raw, ~[-300,300]).
_RAW_CCI_NAMES = ([f"cci{p}_{tf}" for tf in CCI_TFS for p in _CCI_PERIODS]
                  + [f"cci{p}_sma_{tf}" for tf in CCI_TFS for p in _CCI_PERIODS])
# RAW Bollinger band levels [operator 2026-06-15] are price levels -> also unclipped.
_RAW_BOLL_NAMES = [f"boll_{band}_raw_{tf}" for tf in BOLL_TFS for band in _BB_BANDS]
# Training-wheel RAW CCI values + their SMAs [operator 2026-06-15] are unbounded
# (~[-300,300]) -> unclipped + standardized by the M5 agent, like the other CCI block.
_RAW_WHEEL_CCI_NAMES = ([f"tw_cci{p}_{tf}" for tf in _WHEEL_TFS for p in (5, 15)]
                        + [f"tw_cci{p}_sma_{tf}" for tf in _WHEEL_TFS for p in (5, 15)])
RAW_FEATURE_NAMES = (set(SCHEMA.blocks["market_raw"]) | set(_RAW_CCI_NAMES)
                     | set(_RAW_BOLL_NAMES) | set(_RAW_WHEEL_CCI_NAMES))

# Canonical block widths — asserted by the master suite so a refactor can't silently
# drop a challenge-critical feature. COUPLING: must equal the real block_spans; the
# master suite (Section D) asserts both, and config.nominal_state_dim must equal STATE_DIM.
EXPECTED_WIDTHS = {
    "market": 131,  # 128 + 3 new 5m-ATR feats (atr_level/ref/dev_5m for market_volatility_obs)
    "market_raw": 18 if INCLUDE_RAW_INPUTS else 0,   # raw price-SMA only (raw CCI moved to `market`)
    "law": 12, "trade": 35, "portfolio": 3, "account": 8,  # account 7->8: +acct_dist_to_perm_dd (C12, 2026-06-18)
}


def state_vector_fingerprint() -> dict:
    """Structural fingerprint of the observation — the change-impact snapshot source.

    The change-impact tracker (tests snapshot guard + tools/impact.py) diffs this
    against a committed JSON. A drift here means the policy's WORLD changed shape —
    which ripples to the agent input dim, normalization, telemetry labels, and any
    checkpoint. Catching that automatically is how we stop a silent observation
    change from quietly degrading FTMO pass-rate between runs.
    """
    blocks = {n: [s, e] for n, (s, e) in SCHEMA.block_spans.items()}
    # COUPLING [C1] -> tools/snapshot.py + tests/snapshots/state_vector.json: these exact
    # dict keys (schema_version, state_dim, block_widths, feature_names, sha256, ...) are
    # the fingerprint the snapshot guard diffs. Add/rename a key or bump schema_version =>
    # regenerate the committed JSON via tools/snapshot.py --update.
    payload = {
        "schema_version": 1,
        "include_raw_inputs": INCLUDE_RAW_INPUTS,
        "state_dim": STATE_DIM,
        "precomputed_dim": PRECOMPUTED_DIM,
        "block_widths": {n: e - s for n, (s, e) in SCHEMA.block_spans.items()},
        "blocks": blocks,
        "raw_feature_names": sorted(RAW_FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps([payload["state_dim"], payload["feature_names"]], sort_keys=True).encode()
    ).hexdigest()
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). Rulebook:
#   docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M2 — canonical state-vector schema created.
#   I: The bot's perception had no single, asserted layout; a later refactor could
#      drop/reorder a challenge-critical feature and silently blind the policy to
#      breach-risk — a top cause of inconsistent passing.
#   R: STATE_VECTOR.md block counts (~145) + the telemetry data contract.
#   A: Defined 146 scalars in 5 ordered blocks with unique-name validation.
#   C: A frozen, named, asserted observation means the bot always sees the full FTMO
#      picture and telemetry can map any neuron to its feature.
# [2026-06-13] Operator override — added RAW SMA + RAW CCI block (market_raw).
#   I: The operator wants raw SMA (sma1 sh0-3, sma30/50 on 5m/30m/4H) and raw CCI
#      (10/30/100 on 1m/5m/30m/4H) added ALONGSIDE the normalized features, to give
#      the policy un-transformed level signals. This departs from the no-raw-price rule.
#   R: Operator directive (2026-06-13) overrides STATE_VECTOR.md encoding for this
#      block; raw inputs flagged so the M5 agent standardizes them (RAW_INPUTS.md).
#   A: Added the 30-feature `market_raw` block (gated by config.INCLUDE_RAW_INPUTS),
#      RAW_FEATURE_NAMES (bypass clip + mark for standardization), PRECOMPUTED_NAMES
#      (market+market_raw=119); STATE_DIM 146->176. Snapshot guard + impact tool track drift.
#   C: The policy gains the requested raw signals while the raw block stays isolated,
#      flagged, and toggleable — so we can ablate raw-vs-normalized on the scoreboard
#      and, if raw inputs hurt pass-rate or stability, disable them without touching
#      the rest of the perception. Net: more signal to try, with a clean off-ramp that
#      protects consistent passing.
# [2026-06-13] M3 — added gate ingredients (spread x2, adf) to the market block.
#   I: The Spread Filter + Stationarity gates were in the law block but their
#      INGREDIENTS (spread vs ATR/range, ADF stat) weren't observable — violating the
#      law-ingredient coverage rule, so the bot couldn't see WHY a gate opened/closed.
#   R: Law-ingredient coverage rule [M] + F4 (rolling ADF 100-bar p<0.05).
#   A: Added spread_atr_1m, spread_range_ratio_1m, adf_stat_1m to `market` (89->92);
#      STATE_DIM 176->179. Snapshot guard flags the drift (re-pinned via --update).
#   C: Every gate's ingredients are now observable, so the bot learns gate-aware
#      behaviour (trade only in live/stationary regimes) instead of treating gates as
#      opaque on/off — fewer dead-market/illiquid trades that erode the pass rate.
# [2026-06-13] Operator decision — CCI kept RAW (no normalization).
#   I: CCI was exposed normalized (cci_norm=CCI/100, cci_dev=(CCI-SMA)/100). Operator
#      wants RAW CCI + the RAW shifted-forward SMA (a 'where was CCI 4 bars ago'
#      comparison, not a normalized deviation). The market_raw raw_cci then duplicated it.
#   R: Operator override (locked-item approval per R5); applied SMA stays period 2, shift 4.
#   A: Replaced cci{p}_norm/cci{p}_dev with cci{p}/cci{p}_sma (RAW, in RAW_FEATURE_NAMES,
#      unclipped); removed the duplicate raw_cci from market_raw (30->18); STATE_DIM 179->167.
#      Sign-identical so the laws' legal space is UNCHANGED (laws read raw value vs raw SMA).
#   C: The policy sees CCI's true magnitude + its 4-bar-ago smoothed location in raw form
#      (the operator's intended trend signal) with law semantics intact.
# [2026-06-15] Operator decision — Bollinger: keep BOTH normalized + raw.
#   I: Bollinger was only ATR-normalized distance (close-band)/ATR; operator wants the
#      unnormalized data kept too.
#   R: Operator override (additive, observation-only; laws still read the normalized
#      distance sign, so the legal space is unchanged).
#   A: Added 18 RAW band-level features boll_{band}_raw_{tf} (BB20/BB200 mid/up/lo on
#      5m/30m/4H) in RAW_FEATURE_NAMES (unclipped); market 92->110, STATE_DIM 167->185.
#   C: The policy sees both the ATR-relative band distance AND the raw band positions/width
#      (volatility in price terms) - more signal, with the masks/legal space untouched.
# [2026-06-15] Operator decision — training-wheel ingredients + block flags.
#   I: The operator wants semi-permanent counter-trend OPEN-block "training wheels"
#      (CCI 5/15 SMA20-sh0 + BB 10/100 dev0.5 on 30m+4H). Their ingredients did not exist
#      in the observation, and the operator wants them visible ("acts") AND enforced ("laws").
#   R: Operator decision 2026-06-15; additive + APPENDED at the end of `market` (no existing
#      index shifts); 4H used by explicit operator override of the 4H-observation-only rule,
#      isolated from the locked 9 laws; removable via config.TRAINING_WHEELS.
#   A: Appended 16 ingredients (8 RAW tw_cci{5,15}(_sma) + 8 normalized tw_bb{10,100}_{up,lo})
#      + 2 three-way flags (tw_cci_block, tw_bb_block). RAW CCI values added to RAW_FEATURE_NAMES;
#      market 110->128, STATE_DIM 185->203. Snapshot guard flags the drift (re-pinned --update).
#   C: The flags become hard counter-trend masks (engine), so the bot can't open into the
#      wheels' uptrend/downtrend - fewer breach-bound opens, faster convergence to passing -
#      while the locked laws stay untouched and the wheels can be removed later.
