"""CurriculumManager — law-school stages (trend -> reversion -> stationarity+ATR).

WHAT THIS MODULE DOES
---------------------
Drives the full-chart, law-gated curriculum (SOW §7): a sequence of stages, each of
which (a) puts the env in LAW-SCHOOL permission mode with the stage's required law
context, (b) optionally masks 1m timing features so the bot learns structure first,
and (c) sets the stationarity gate mode. Graduation advances the stage. After the
final stage the env flips to LIVE-ban mode.

Stages (SOW §7.1):
  1. trend            -> super-trend + trend laws are the permission context
  2. reversion        -> pull-back laws are the permission context
  3. stationarity_atr -> any law context, but ONLY when ADF-stationary + ATR gate open

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Structure-first learning (trade only inside the stage's law context, 1m timing masked
early) builds a bot that respects the laws by HABIT before it is trusted with fine 1m
entries. Fewer law-adjacent mistakes -> fewer breaches -> a higher, more consistent
pass rate. The curriculum shapes *how* the policy learns the legal space.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Mask Dependence often shows up when
a stage advances too early. ``current_stage()`` + the env's mask_mode tell you which
permission context was active; correlate that with law-adjacent failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# COUPLING [C4] -> locked_core/laws/laws.py: LAW_NAMES order/strings drive _trend_laws/
# _pullback_laws (substring match) and the LAW_NAMES[:9] directional slice below; if laws.py
# renames a law or reorders the 12 names, the stage required_laws change silently.
from quantra.locked_core.laws.laws import LAW_NAMES
# COUPLING [C1] -> market_pipeline/feature_builder/schema.py: STATE_DIM sizes feature_mask;
# PRECOMPUTED_NAMES order resolves the _EARLY_MASK_1M indices (guarded but order-sensitive).
from quantra.market_pipeline.feature_builder.schema import PRECOMPUTED_NAMES, STATE_DIM
# COUPLING -> market_pipeline/law_mask_engine/engine.py: MODE_LIVE/MODE_SCHOOL are the
# enforcement-mode constants the env (trading_env.py) expects in law_school_config().
from quantra.market_pipeline.law_mask_engine.engine import MODE_LIVE, MODE_SCHOOL

# 1m TIMING features masked in early stages (structure-first). We KEEP the law-binding
# 1m ingredients (shifted-SMA + ATR + the gate ingredients) so the laws still activate;
# we mask the pure observation/timing features the slower structure already implies.
# COUPLING: each name MUST exist in schema.PRECOMPUTED_NAMES or it silently no-ops.
# CCI 1m names changed to raw (cci{p}_1m) when CCI was un-normalized [2026-06-13];
# CCI 1m is observation-only (no law binds 1m CCI), so masking it early is safe.
_EARLY_MASK_1M = [
    "candle_return_1m", "candle_range_1m", "candle_uwick_1m", "candle_lwick_1m",
    "z10_1m", "z100_1m", "adx5_1m", "adx15_1m",
    "cci10_1m", "cci30_1m", "cci100_1m",
]


@dataclass
class Stage:
    """One curriculum stage's law-school configuration."""

    name: str
    required_laws: List[str]
    stationarity_mode: Optional[str] = None   # "A"/"B"/None (gate disabled)
    mask_1m: bool = True                       # mask 1m timing features (structure-first)


def _trend_laws() -> List[str]:
    return [n for n in LAW_NAMES if "super_trend" in n or n.startswith("law_trend")]


def _pullback_laws() -> List[str]:
    return [n for n in LAW_NAMES if "pullback" in n]


DEFAULT_STAGES: List[Stage] = [
    Stage("trend", _trend_laws(), stationarity_mode=None, mask_1m=True),
    Stage("reversion", _pullback_laws(), stationarity_mode=None, mask_1m=True),
    # COUPLING [C4] -> locked_core/laws/laws.py: LAW_NAMES[:9] assumes the first 9 names are
    # the DIRECTIONAL laws and [9:] the 3 gates (mirrored by law_mask_engine [:9]/[9:]).
    Stage("stationarity_atr", LAW_NAMES[:9], stationarity_mode="A", mask_1m=False),
]


@dataclass
class CurriculumManager:
    """Holds the stage list + a cursor; configures the env and advances on graduation."""

    stages: List[Stage] = field(default_factory=lambda: list(DEFAULT_STAGES))
    idx: int = 0
    graduated: bool = False   # True after the final stage -> LIVE mode

    def current_stage(self) -> Optional[Stage]:
        return None if self.graduated else self.stages[self.idx]

    def law_school_config(self) -> dict:
        """Env kwargs for the current stage: live-ban after graduation, else school."""
        # COUPLING -> env/trading_env.py: these dict KEYS (mask_mode, required_laws,
        # stationarity_mode) are spread as TradingEnv constructor kwargs of the same name.
        st = self.current_stage()
        if st is None:
            return {"mask_mode": MODE_LIVE, "required_laws": None, "stationarity_mode": "A"}
        return {"mask_mode": MODE_SCHOOL, "required_laws": st.required_laws,
                "stationarity_mode": st.stationarity_mode or "off"}

    def feature_mask(self) -> np.ndarray:
        """A (STATE_DIM,) multiplicative mask: 0 on masked 1m timing features, else 1.

        The trainer multiplies observations by this so early stages can't lean on 1m
        timing — forcing structure-first learning (fewer fragile micro-entries that
        breach). After mask_1m stages it is all ones.
        """
        # COUPLING [C1] -> trainer/trainer.py: returns a (STATE_DIM,) array the trainer
        # element-wise multiplies against each obs; its length MUST equal the obs width.
        mask = np.ones(STATE_DIM, dtype=np.float32)
        st = self.current_stage()
        if st is not None and st.mask_1m:
            for name in _EARLY_MASK_1M:
                if name in PRECOMPUTED_NAMES:
                    mask[PRECOMPUTED_NAMES.index(name)] = 0.0
        return mask

    def graduate(self) -> None:
        """Advance to the next stage; after the last, flip to LIVE (graduated)."""
        if self.graduated:
            return
        if self.idx >= len(self.stages) - 1:
            self.graduated = True
        else:
            self.idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M7 — implemented the CurriculumManager.
#   I: Nothing staged training to teach structure-first inside law context (law-school
#      permission mode) before trusting the bot with fine 1m timing.
#   R: SOW §7 (full-chart, law-gated curriculum; trend -> reversion -> stationarity+ATR;
#      1m masking early; graduation -> live ban mode).
#   A: Stage configs (required_laws, stationarity_mode, mask_1m), env law_school_config,
#      feature_mask zeroing 1m timing features (keeping law-binding ingredients), graduate().
#   C: The bot learns to respect the laws by habit before fine timing, so it makes fewer
#      law-adjacent mistakes -> fewer breaches -> a higher, steadier pass rate.
# [2026-06-13] _EARLY_MASK_1M CCI names follow the CCI-raw rename.
#   I: The 1m mask listed cci{p}_norm_1m; those names were removed when CCI went raw,
#      so the mask would silently no-op (guarded by `if name in PRECOMPUTED_NAMES`).
#   R: COUPLING with schema feature names (mask entries must be real columns).
#   A: Renamed to cci{p}_1m (raw CCI 1m; observation-only, no law binds 1m CCI).
#   C: Structure-first masking still zeros 1m CCI timing early, so the curriculum keeps
#      teaching law-respect before fine timing - unchanged effect on the pass rate.
