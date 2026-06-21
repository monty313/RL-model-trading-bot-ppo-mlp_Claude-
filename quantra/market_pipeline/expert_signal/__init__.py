"""quantra.market_pipeline.expert_signal — the SOFT expert-signal observation layer.

Distils the operator's rule-based STRAT portfolio into a small block of bounded,
observation-only features (see engine.py + docs/EXPERT_SIGNAL_DESIGN.md). Phase 1:
the pure engine only — not yet wired into the schema/observation.
"""

from .engine import (
    DEFAULT_EXPERT_CONFIG,
    EXPERT_DIM,
    EXPERT_NAMES,
    ExpertSignalConfig,
    compute_expert_signals,
    expert_signals_dict,
)

__all__ = [
    "DEFAULT_EXPERT_CONFIG",
    "EXPERT_DIM",
    "EXPERT_NAMES",
    "ExpertSignalConfig",
    "compute_expert_signals",
    "expert_signals_dict",
]
