"""Quantra runtime configuration — the single source of runtime knobs.

WHAT THIS MODULE DOES
---------------------
Holds the runtime-memory inputs and infrastructure constants Quantra needs at
launch: the FTMO challenge defaults, the four traded symbols, the Google-Drive
file IDs for the price data, filesystem paths, and the hardware-utilisation
targets used by the rest of ``quantra.runtime``.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Per SOW Section 1.4 the policy is *account-size-blind* and takes its targets as
**pure runtime memory** (no saved config file dictating strategy). This module is
that runtime memory: ``daily_target_pct`` (2.5) and ``daily_risk_pct`` (4.0) are
the very constraints the bot is being trained to respect on every window. Keeping
them here — never hardcoded inside laws, rewards, or the env — is what lets the
same brain be pointed at a different target/loss pair without retraining, which is
the whole point of "repeatedly pass FTMO-style challenges over time".

The hardware section exists so training cycles are cheap and fast: we want to
*finish* walk-forward windows, not burn paid GPU hours on a tiny MLP. Faster,
cheaper iterations = more seeds, more windows, a better-validated pass rate.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Binding rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. If you (the Risk Doctor)
see a diagnosis that depends on the daily target / loss / account size, read those
values FROM the telemetry packet's run-config block, NOT from this file — at
inference the operator may have passed different runtime values. This file only
provides the *defaults* (SOW-A3) and the immutable infrastructure constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------------------
# Filesystem layout. Everything Quantra writes (data cache, checkpoints,
# telemetry, reports) hangs off the repo root so a Colab run and a local run
# behave identically.
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"            # original MT5 CSV exports
PARQUET_DIR: Path = DATA_DIR / "parquet"    # cleaned, UTC-indexed bars
FEATURE_CACHE_DIR: Path = DATA_DIR / "features"  # memmapped precomputed features
ARTIFACT_DIR: Path = REPO_ROOT / "artifacts"
CHECKPOINT_DIR: Path = ARTIFACT_DIR / "checkpoints"
TELEMETRY_DIR: Path = ARTIFACT_DIR / "telemetry"
REPORT_DIR: Path = ARTIFACT_DIR / "reports"
DOCS_DIR: Path = REPO_ROOT / "docs"

# The interpretability rulebook the LLM Risk Doctor MUST be able to read
# (MLP_INTERPRETABILITY_LAYER.md §"the codebase should fail loudly if the LLM is
# invoked without that file being accessible"). Referenced here so a single
# constant locates it for every module.
INTERPRETABILITY_RULEBOOK: Path = DOCS_DIR / "MLP_INTERPRETABILITY_LAYER.md"


# ---------------------------------------------------------------------------
# Symbols + their Google-Drive price-data file IDs (folder `rl-trading-data`,
# owner mmayes313@gmail.com). The data_loader (M1) resolves bars in this order:
# explicit local path -> Colab Drive mount -> gdown by these IDs.
# ---------------------------------------------------------------------------
# COUPLING [C5 in COUPLINGS.md]: every symbol in SYMBOLS must have an entry in each
# per-symbol dict below (ASSET_CLASS, POINT_SIZE, CONTRACT_SIZE, SLIPPAGE_POINTS,
# DRIVE_FILE_IDS, DRIVE_FILENAMES). Add a symbol => add it to ALL of them, or the
# cost/risk/loader layers KeyError or mis-size. Consumers: cost_layer, risk_manager,
# env, challenge_state, live_session, data_loader.
SYMBOLS: List[str] = ["EURUSD", "XAUUSD", "GBPUSD", "US30"]  # SOW §12.1 order

# Asset class drives the cost model (SOW §10.5): forex pays $5/RT/lot; metals &
# indices pay no per-trade commission (spread + slippage only).
# COUPLING [C5] -> cost_layer/costs.py: costs.py looks up ASSET_CLASS[symbol] vs
# CostConfig.commissioned_classes ("forex") to decide commission; a new class string
# here that isn't in commissioned_classes silently pays no commission.
ASSET_CLASS: Dict[str, str] = {
    "EURUSD": "forex",
    "GBPUSD": "forex",
    "XAUUSD": "metal",
    "US30": "index",
}

# Point size = price value of 1 MT5 "point" per symbol. MT5 exports the <SPREAD>
# column in POINTS, so the Spread Filter law (spread vs candle range, both in price)
# needs this to convert. Broker-dependent; conservative defaults for 5-digit forex,
# 2-digit gold, 1.0 index. Used by the FeatureBuilder's spread features + the
# Spread Filter gate so the bot learns under realistic execution friction (SOW-H2).
# COUPLING [C5] -> feature_builder/builder.py + locked_core/laws/laws.py: builder uses
# POINT_SIZE[symbol] to convert MT5 <SPREAD> points->price for the spread feature; the
# Spread Filter law reads that feature. Missing key => KeyError on that symbol.
POINT_SIZE: Dict[str, float] = {
    "EURUSD": 1e-5,
    "GBPUSD": 1e-5,
    "XAUUSD": 1e-2,
    "US30": 1.0,
}
DEFAULT_POINT_SIZE: float = 1e-5

# Contract size = account-currency (USD) value of a 1.0-lot, 1.0-price move. All 4
# symbols are USD-quoted, so PnL_USD = price_change * CONTRACT_SIZE * lots (no FX
# conversion). Used by CostLayer + RiskManager (M4) to translate price <-> dollars
# so the bot's risk is measured in the same units as the FTMO wall.
# COUPLING [C5] -> risk_manager/risk.py + cost_layer/costs.py + env/trading_env.py:
# they index CONTRACT_SIZE[symbol] for lot-sizing and PnL; add a SYMBOL => add a key
# here or risk/cost/PnL KeyError for that symbol.
CONTRACT_SIZE: Dict[str, float] = {
    "EURUSD": 100_000.0,   # 1 lot = 100k EUR; 0.0001 move = $10
    "GBPUSD": 100_000.0,
    "XAUUSD": 100.0,       # 1 lot = 100 oz; $1 move = $100
    "US30": 1.0,           # 1 lot = $1 / index point
}

# Per-symbol fixed slippage in POINTS, applied adversely on every fill (SOW §10.5).
# Conservative defaults; never zero (no costless world, SOW C8).
# COUPLING [C5] -> cost_layer/costs.py + live_bridge/live_session.py: costs.py reads
# SLIPPAGE_POINTS[symbol] (x POINT_SIZE) for fill adversity; key must exist per SYMBOL.
SLIPPAGE_POINTS: Dict[str, float] = {
    "EURUSD": 5.0,
    "GBPUSD": 5.0,
    "XAUUSD": 20.0,
    "US30": 10.0,
}


# COUPLING [C5] -> cost_layer/costs.py: costs.py reads these exact field names
# (commission_per_lot_rt_forex, commissioned_classes) and matches commissioned_classes
# against ASSET_CLASS values above. Rename a field or change the class string => costs break.
@dataclass(frozen=True)
class CostConfig:
    """Real FTMO costs (SOW §10.5). $5 RT/lot on FOREX only; metals/indices pay no
    per-trade commission (spread + slippage only). No costless world ever (C8)."""

    commission_per_lot_rt_forex: float = 5.0   # round-trip $/lot, forex only
    # Asset classes that pay the per-trade commission. Metals/indices excluded.
    commissioned_classes: tuple = ("forex",)


# COUPLING [C5] -> risk_manager/risk.py: risk.py reads these exact field names
# (stop_atr_mult, lot_step, min_lot, max_lot, max_per_trade_risk_frac) to size/round lots;
# renaming any breaks lot-sizing. HPO may tune values but must not rename fields.
@dataclass(frozen=True)
class RiskConfig:
    """RiskManager dials (NON-sacred; tunable). The invariant they enforce — total
    open-slot risk never exceeds the remaining daily-risk buffer — is what makes the
    4% wall hard to ever reach, which is the mechanical core of not breaching."""

    stop_atr_mult: float = 1.5       # reference stop = stop_atr_mult * ATR(price)
    lot_step: float = 0.01           # broker rounding granularity
    min_lot: float = 0.01
    max_lot: float = 50.0
    max_per_trade_risk_frac: float = 0.01  # per-trade cap = 1% of account size

# COUPLING [C5] -> data_loader/loader.py: loader.py resolves each SYMBOL's bars via
# DRIVE_FILE_IDS[symbol] (gdown) and DRIVE_FILENAMES[symbol] (Drive-mount lookup); a
# SYMBOL missing from either dict cannot be loaded.
DRIVE_FILE_IDS: Dict[str, str] = {
    "EURUSD": "1tsR789vdRYE4rwDAE-hreWF2zN1slcjH",
    "GBPUSD": "1503qJQxjLwA2O0zIiakiOM_68Ucb-ypm",
    "XAUUSD": "1bzICq5oXh3z5PIwrovDa8xUHwRV-d8Gz",
    "US30": "1YF3vr4gBAm-4PZPLGCG0plWPDSaXH14h",
}

# Original Drive filenames (for Colab Drive-mount resolution, where we look the
# file up by name under the mounted `rl-trading-data` folder rather than by ID).
DRIVE_FILENAMES: Dict[str, str] = {
    "EURUSD": "EURUSD_M1_202101131130_202605270000_2020_2026.csv",
    "GBPUSD": "GBPUSD_M1_202101131952_202605270000.csv",
    "XAUUSD": "XAUUSD_M1_202009230753_202605262259.csv",
    "US30": "US30_M1_202007231046_202605262359.csv",
}
DRIVE_FOLDER_NAME: str = "rl-trading-data"
DRIVE_FOLDER_ID: str = "1azEnCfwQjxPkBemmv9mxY3GyVAMcjF-3"


# ---------------------------------------------------------------------------
# Observation toggle [operator override, 2026-06-13]. When True the state vector
# includes the RAW SMA + RAW CCI block (`market_raw`), departing from the
# no-raw-price encoding rule. The M5 agent MUST standardize these (see
# feature_builder/RAW_INPUTS.md). Flip to False to ablate raw-vs-normalized; the
# schema + STATE_DIM follow automatically. Read by feature_builder.schema.
# ---------------------------------------------------------------------------
# COUPLING [C1] -> feature_builder/schema.py: schema reads INCLUDE_RAW_INPUTS to add/drop
# the market_raw block, which sets STATE_DIM (167 on / 149 off). Flipping this changes the
# observation width everywhere; nominal_state_dim below mirrors the same branch.
INCLUDE_RAW_INPUTS: bool = True


# ---------------------------------------------------------------------------
# TRAINING WHEELS [operator decision 2026-06-15]. Semi-permanent counter-trend
# OPEN-block masks (CCI 5/15 SMA20-sh0 + BB 10/100 dev0.5 on 30m+4H). When True the
# env/live session forbid opening AGAINST the wheels' trend context. These are
# "training wheels": ON by default so the bot can't open into a strong 30m+4H trend
# (fewer breach-bound trades => faster convergence to passing), removable by flipping
# this to False once the policy is disciplined. The wheel INGREDIENTS + flags are in
# the observation regardless (operator wants them visible); this flag only gates
# ENFORCEMENT. COUPLING -> env/trading_env.py + live_bridge/live_session.py default
# their `training_wheels` arg to this; law_mask_engine.build_direction_mask applies it.
# ---------------------------------------------------------------------------
TRAINING_WHEELS: bool = True


# COUPLING [C5] -> ftmo_passing/challenge_state.py + reward_engine/reward.py +
# env/trading_env.py + live_bridge/live_session.py: they read these exact field names
# (daily_target_pct, daily_risk_pct, pain_zone_start_pct, hard_wall_pct, ...) to set the
# target/wall thresholds; renaming a field silently changes what "passing" means.
@dataclass(frozen=True)
class ChallengeConfig:
    """Runtime FTMO inputs (SOW §1.4, §2.6, §2.7). Defaults per SOW-A3.

    Why frozen: these define *what passing means* on a given run. They must be
    captured verbatim into telemetry so the Risk Doctor judges behaviour against
    the exact target/wall that were in force — never against assumed defaults.
    """

    daily_target_pct: float = 2.5     # Phase-A profit that triggers auto-flat (ftmo_mode)
    daily_risk_pct: float = 4.0       # the trailing stop-loss (account-level wall). INPUT.
    phase_b_trailing_pct: float = 1.0  # fresh trailing wall after target (ftmo_mode only, §2.6)
    pain_zone_start_pct: float = 3.5  # exponential reward Layer 3 begins here
    hard_wall_pct: float = 4.0        # force-flatten + lockout (SOW §2.7)
    ftmo_mode: bool = True            # ON: 2-phase, auto-flat at target. OFF: target is the AIM
                                      # (drives progress + success-%) but not a forced stop — runs on.
    stop_for_day: bool = False        # OFF-mode toggle: bank + STOP when the target is hit (else run on).
    ftmo_account_size: float = 10_000.0  # reference scaling only; policy is blind
    leverage: float = 100.0          # account leverage (1:100). Margin = notional / leverage. INPUT.


# Operator input bounds [decision 2026-06-15]. ftmo_mode ON keeps challenge-safe ranges;
# OFF unlocks the wide side-account envelope (target up to 100%, trailing risk up to 40%).
# The bot is trained scale-invariant (% space), so any (target, risk) in-range maps onto
# the SAME normalized policy — no per-combo retraining. COUPLING -> make_challenge() clamps
# operator input into these; live_runner.py / live_session.py build configs through it.
FTMO_ON_BOUNDS = {"target": (0.25, 10.0), "risk": (1.0, 10.0)}
FTMO_OFF_BOUNDS = {"target": (1.0, 100.0), "risk": (1.0, 40.0)}
LEVERAGE_CHOICES = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0)


def make_challenge(daily_target_pct: float = 2.5, daily_risk_pct: float = 4.0, *,
                   ftmo_mode: bool = True, leverage: float = 100.0,
                   stop_for_day: bool = False,
                   phase_b_trailing_pct: float = 1.0, account_size: float = 10_000.0,
                   pain_zone_start_pct: float | None = None,
                   hard_wall_pct: float | None = None) -> "ChallengeConfig":
    """Build a VALIDATED ChallengeConfig — the operator entry point for per-day inputs.

    Clamps target/risk into the active mode's bounds, and pins the breach wall + pain
    zone to the trailing-risk INPUT (so the reward/breach geometry follows the operator's
    chosen stop, not a stale 3.5/4.0). ChallengeConfig stays frozen; per-day changes mean
    constructing a fresh one (env.reset(challenge=...) / a new live session)."""
    b = FTMO_ON_BOUNDS if ftmo_mode else FTMO_OFF_BOUNDS
    t = float(min(max(daily_target_pct, b["target"][0]), b["target"][1]))
    r = float(min(max(daily_risk_pct, b["risk"][0]), b["risk"][1]))
    hw = r if hard_wall_pct is None else float(hard_wall_pct)        # wall == trailing stop input
    pz = (0.875 * hw) if pain_zone_start_pct is None else float(pain_zone_start_pct)  # 7/8 of wall
    return ChallengeConfig(
        daily_target_pct=t, daily_risk_pct=r, phase_b_trailing_pct=phase_b_trailing_pct,
        pain_zone_start_pct=pz, hard_wall_pct=hw, ftmo_mode=ftmo_mode,
        stop_for_day=stop_for_day, ftmo_account_size=account_size,
        leverage=max(1.0, float(leverage)))


@dataclass(frozen=True)
class HardwareConfig:
    """Targets for the auto-optimizer (``quantra.runtime.optimizer``).

    The user's instruction: use ~80% of whichever device (CPU or GPU) trains
    fastest, prefer CPU, never waste paid GPU hours on a 3x256 MLP. These knobs
    encode that policy; ``optimizer.plan()`` reads them.
    """

    utilization_target: float = 0.80   # fraction of the chosen device to drive
    prefer_cpu: bool = True            # tie-break / near-tie goes to CPU (cheaper)
    # A GPU must beat CPU throughput by at least this factor to be worth its cost,
    # otherwise we stay on CPU and tell the operator to drop the GPU runtime.
    gpu_speedup_required: float = 1.30
    benchmark_seconds: float = 2.5     # wall-time budget for the startup race
    min_envs: int = 1
    max_envs: int = 256                # safety ceiling on vectorised worlds
    # Leave at least this many logical cores free so Colab stays responsive and
    # the kernel isn't OOM/again-killed.
    reserved_cores: int = 1


@dataclass
class RuntimeConfig:
    """Top-level runtime memory assembled at launch and logged to telemetry."""

    challenge: ChallengeConfig = field(default_factory=ChallengeConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    symbols: List[str] = field(default_factory=lambda: list(SYMBOLS))
    seed: int = 0

    # Nominal state-vector width for the *startup* benchmark only. Mirrors
    # quantra.market_pipeline.feature_builder.schema.STATE_DIM (176 with raw inputs
    # on; 146 off) without importing it (avoids an import cycle); the master suite
    # asserts they match. We never let this nominal value leak into training shapes.
    # COUPLING: must equal feature_builder.schema.STATE_DIM (asserted by the master
    # suite). 203 with raw inputs on (raw CCI + raw price-SMA + training wheels), 185 off.
    nominal_state_dim: int = field(default_factory=lambda: 203 if INCLUDE_RAW_INPUTS else 185)

    # COUPLING [C8] -> diagnostics/telemetry_logger/logger.py + llm_risk_doctor/doctor.py:
    # this dict becomes telemetry's run-config block; the Risk Doctor reads target/loss
    # from that block (these keys), NOT from this file. Key shape here is the data contract.
    def to_dict(self) -> dict:
        """Flatten for telemetry's run-config block (M9 data contract)."""
        d = asdict(self)
        d["repo_root"] = str(REPO_ROOT)
        return d


def ensure_dirs() -> None:
    """Create all writable directories. Safe to call repeatedly (idempotent).

    Called once at launch so the very first Colab run has somewhere to put the
    parquet cache, checkpoints, and telemetry without manual mkdir.
    """
    for d in (DATA_DIR, RAW_DIR, PARQUET_DIR, FEATURE_CACHE_DIR,
              ARTIFACT_DIR, CHECKPOINT_DIR, TELEMETRY_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def in_colab() -> bool:
    """True when running inside a Google Colab kernel.

    Used by the data_loader to prefer a Drive mount and by the optimizer to size
    the utilisation monitor's logging cadence.
    """
    return "google.colab" in os.sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Runtime config documented + pinned by the master suite.
#   I: FTMO target/loss defaults, Drive file IDs and paths had no change-log or test pin.
#   R: SOW-A3 (2.5%/4.0% configurable, never hardcoded strategy) + the new IRAC rule.
#   A: Annotated every constant for its FTMO role; defaults now asserted in the master suite (Section A).
#   C: The bot always trains/judges against the real 2.5%/4% walls, so a 'pass' is reproducible and meaningful.
# [2026-06-13] nominal_state_dim 145 -> 146 to match the M2 schema.
#   I: The benchmark's nominal width (145) no longer matched the real observation
#      width once M2 fixed the canonical layout at 146.
#   R: STATE_VECTOR.md (~145) realised as schema.STATE_DIM = 146; keep config import-cycle-free.
#   A: Set nominal_state_dim = 146; the master suite now asserts config == schema.
#   C: The hardware race times the true observation width, so the device/cost choice
#      reflects the real workload — no wasted GPU spend, faster path to a validated pass rate.
# [2026-06-13] nominal_state_dim -> 179/149 (M3 gate ingredients) + added POINT_SIZE.
#   I: M3 added 3 gate-ingredient features (176->179); and the Spread Filter needs a
#      points->price conversion the config didn't provide.
#   R: Law-ingredient coverage (spread, ADF) + SOW-H2 realistic spread costs.
#   A: Bumped nominal_state_dim to 179 (raw on)/149 (off); added per-symbol POINT_SIZE.
#   C: config stays in lockstep with the schema and the bot sees real spread friction,
#      so it learns to avoid illiquid/dead-market trades that quietly erode pass-rate.
# [2026-06-13] M4 — added contract specs + CostConfig + RiskConfig + slippage.
#   I: The env/RiskManager/CostLayer need per-symbol contract sizes, real FTMO costs,
#      and risk dials to size and cost trades in account dollars.
#   R: SOW §10.5 ($5 RT/lot forex; metals/indices no commission; spread+slippage) +
#      H3 (raw_size -> lots vs remaining buffer) + C8 (no costless world).
#   A: Added CONTRACT_SIZE, SLIPPAGE_POINTS, CostConfig ($5 RT forex-only), RiskConfig
#      (stop_atr_mult, lot_step, min/max lot, per-trade risk cap).
#   C: Trades are sized + costed in the same dollars as the 4% wall, so the
#      no-overshoot invariant is enforceable and the learned edge survives real fees —
#      both prerequisites for passing rather than looking profitable.
# [2026-06-13] nominal_state_dim -> 167 (CCI un-normalized).
#   I: CCI raw decision removed the duplicate raw_cci block (179->167); the benchmark
#      width must track the schema.
#   R: COUPLING - config.nominal_state_dim must equal schema.STATE_DIM (master-suite asserted).
#   A: 167 (raw inputs on) / 149 (off).
#   C: The hardware race times the true observation width, so the device/cost choice stays honest.
# [2026-06-15] nominal_state_dim -> 185 (raw Bollinger band levels added).
#   I: Operator kept BOTH normalized + raw Bollinger; market 92->110, STATE_DIM 167->185.
#   R: COUPLING [C1] - nominal_state_dim must equal schema.STATE_DIM (master-suite asserted).
#   A: 185 (raw inputs on) / 167 (off).
#   C: The benchmark + device choice stay aligned with the real observation width.
# [2026-06-15] Per-day inputs: leverage + make_challenge() + mode bounds.
#   I: Daily target/trailing-stop weren't per-day settable, ftmo_mode was a dead flag,
#      and there was no leverage/margin input (operator uses 1:50..1:2000 accounts).
#   R: Operator decision 2026-06-15 (adjustable target/risk/leverage per day; ftmo OFF
#      unlocks the wide side-account envelope; % space so one brain scales by proportion).
#   A: Added ChallengeConfig.leverage; FTMO_ON/OFF_BOUNDS + LEVERAGE_CHOICES; make_challenge()
#      clamps target/risk to the mode bounds and pins wall+pain-zone to the trailing input.
#   C: The operator can dial each account's target/stop/leverage and the bot trades the same
#      normalized policy against it - consistent passing on the priority, safe scale when off.
# [2026-06-15c] Training wheels flag + nominal_state_dim -> 203.
#   I: Operator added semi-permanent counter-trend OPEN-block masks ("training wheels")
#      with their own ingredients/flags; nothing toggled enforcement and the benchmark
#      width no longer matched the schema (185->203).
#   R: Operator decision 2026-06-15; COUPLING [C1] nominal_state_dim == schema.STATE_DIM
#      (master-suite asserted); wheels must be removable.
#   A: Added TRAINING_WHEELS (default True); nominal_state_dim 203 (raw on)/185 (off).
#   C: The bot can't open against a strong 30m+4H trend while learning (fewer breaches =
#      faster path to a passing policy), the wheels come off with one flag, and the
#      hardware race still times the true observation width.
# [2026-06-15b] Added stop_for_day toggle (OFF mode).
#   I: OFF still needs a target (the aim) but the operator may want to bank+stop at it.
#   R: Operator correction 2026-06-15.
#   A: ChallengeConfig.stop_for_day + make_challenge passthrough.
#   C: OFF can either run past the target (default) or bank a side-account day on demand.
