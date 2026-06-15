"""HPO — Optuna over NON-SACRED dials only. 🔴 sacred-guard

WHAT THIS MODULE DOES
---------------------
Hyperparameter search (Optuna) restricted to the non-sacred dials, with a hard guard
that REFUSES to tune the hand-locked items (G6): gamma, lambda, and the aggression-
scheduler logic + its G2 ranges. Objectives maximize the pass-rate scoreboard (never
PnL). The search space + the sacred guard are testable without Optuna; ``run_study``
imports Optuna lazily.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The "math of patience" (gamma/lambda) and the exploration schedule are what make the
bot learn to pass; letting a search chase short-term reward could tune them away. The
guard makes that impossible, so HPO can improve the *tunable* stability dials while the
patience that underpins passing stays locked.

🔴 LOCKED (off-limits to Optuna): gamma, lambda, the aggression scheduler logic + the
G2 ranges. The guard enforces this in code.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. If a tuned run regresses, the cause
is a non-sacred dial (value_coef, grad_clip, batch sizes); gamma/lambda/scheduler are
guaranteed unchanged.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable

# 🔴 SACRED — Optuna may NEVER tune these (G2/G6 hand-locks).
# COUPLING [C6] -> trainer/gae.py + trainer/scheduler.py: this guard exists because
# gae.py hand-locks GAMMA/LAMBDA and scheduler.py hand-locks AggressionRanges (entropy/
# clip/lr/epochs) + the scheduler logic; if a new locked dial is added there, add its
# name here or HPO could tune it.
SACRED_DIALS = frozenset({
    "gamma", "lambda", "scheduler_logic",
    "entropy_range", "clip_range", "lr_range", "epochs_range",
})

# Non-sacred search space: (low, high) for floats, [choices] for categoricals.
# COUPLING -> trainer/trainer.py: these key NAMES (value_coef, minibatch, rollout_size)
# echo TrainConfig fields the objective callback maps tuned params onto; keep the names
# aligned with TrainConfig or the wiring silently drops a dial.
DEFAULT_SEARCH_SPACE: Dict[str, object] = {
    "value_coef": (0.3, 0.8),         # critic loss weight
    "grad_clip_norm": (0.3, 1.0),     # gradient clipping
    "minibatch": [32, 64, 128],       # later-phase minibatch
    "rollout_size": [512, 1024, 2048],
}


def validate_not_sacred(names: Iterable[str]) -> None:
    """Raise if any requested tuning name is a hand-locked (sacred) dial."""
    bad = sorted(set(names) & SACRED_DIALS)
    if bad:
        raise ValueError(
            f"HPO may NOT tune hand-locked dials {bad} (gamma/lambda/scheduler are "
            f"locked per SOW G2/G6). Remove them from the search space."
        )


def suggest(trial, space: Dict[str, object] = None) -> Dict[str, object]:
    """Suggest non-sacred params from an Optuna trial (or a duck-typed stub).

    ``trial`` must expose suggest_float / suggest_int / suggest_categorical (the
    Optuna API). The sacred guard runs first, so no locked dial can ever be sampled.
    """
    space = space or DEFAULT_SEARCH_SPACE
    validate_not_sacred(space.keys())
    out: Dict[str, object] = {}
    for name, spec in space.items():
        if isinstance(spec, (list, tuple)) and not (len(spec) == 2 and all(isinstance(x, float) for x in spec)):
            out[name] = trial.suggest_categorical(name, list(spec))
        else:
            lo, hi = spec
            out[name] = trial.suggest_float(name, float(lo), float(hi))
    return out


def run_study(objective: Callable[[Dict[str, object]], float],
              n_trials: int = 20, direction: str = "maximize",
              space: Dict[str, object] = None, seed: int = 0):
    """Optuna study over the non-sacred space. ``objective(params)`` returns the
    pass-rate to MAXIMIZE (never PnL). Optuna imported lazily."""
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover
        raise ImportError("HPO requires optuna: `pip install optuna`.") from exc

    space = space or DEFAULT_SEARCH_SPACE
    validate_not_sacred(space.keys())
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _obj(trial):
        return objective(suggest(trial, space))

    study = optuna.create_study(direction=direction,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_obj, n_trials=n_trials)
    return study


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M13 — implemented Optuna HPO with the sacred guard.
#   I: Tuning was needed for the stability dials, but a naive search could tune away the
#      patience (gamma/lambda) + schedule that make the bot pass.
#   R: SOW G6 (Optuna on non-sacred dials only; gamma/lambda/scheduler hand-locked).
#   A: SACRED_DIALS guard (validate_not_sacred raises), non-sacred default space,
#      suggest() + run_study() maximizing the pass-rate scoreboard (never PnL).
#   C: HPO can improve the tunable stability dials while the patience underpinning
#      passing stays locked - so tuning helps the pass rate and can never sabotage it.
