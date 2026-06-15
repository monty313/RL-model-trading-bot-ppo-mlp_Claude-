"""`python -m quantra.runtime` — race devices and print the hardware plan.

WHAT THIS DOES
--------------
A zero-argument smoke entry point: build the runtime config, run the CPU-vs-GPU
throughput race, size parallelism to ~80%, and print the decision. This is the
first thing to run in Colab (and the M0 acceptance check) to confirm the box is
configured to train cheaply.

HOW IT SERVES FTMO PASSING
--------------------------
Gives the operator an immediate, honest read on cost/speed before committing to a
training run — so paid GPU time is never wasted on a tiny MLP. Rulebook for the
Risk Doctor: ``docs/MLP_INTERPRETABILITY_LAYER.md`` (infrastructure, not learning).
"""

from __future__ import annotations

from .config import RuntimeConfig, ensure_dirs, in_colab
from .optimizer import plan, print_report
from .utilization_monitor import UtilizationMonitor


def main() -> None:
    ensure_dirs()
    cfg = RuntimeConfig()
    print(f"Environment: {'Google Colab' if in_colab() else 'local'}")
    print(f"Symbols    : {', '.join(cfg.symbols)}")
    # COUPLING [C5] -> config.py ChallengeConfig: reads field names daily_target_pct,
    # daily_risk_pct, ftmo_mode; renaming any in config.py breaks this banner.
    print(
        f"Challenge  : target {cfg.challenge.daily_target_pct}% / "
        f"risk {cfg.challenge.daily_risk_pct}% trailing | ftmo_mode={cfg.challenge.ftmo_mode}"
    )

    # Measure utilisation across the benchmark itself so the report reflects a
    # real load, not an idle sample.
    # COUPLING [C1] -> config.py: passes cfg.nominal_state_dim as the benchmark width; that
    # value must equal feature_builder.schema.STATE_DIM or the race times the wrong shape.
    with UtilizationMonitor(interval=0.25) as mon:
        hw = plan(state_dim=cfg.nominal_state_dim, hw=cfg.hardware)
    util = mon.stop()

    print_report(hw)
    print(f"Utilisation during race: {util.render()}")
    print(
        "\nNote: the locked net is a 3x256 MLP (~145 inputs). If the GPU row above "
        "is not >= 1.30x the CPU row, a CPU/free Colab runtime is the cheaper, "
        "faster choice - exactly what the optimizer selected."
    )


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Instant cost/speed read for the operator.
#   I: The operator needs an honest read on cost/speed BEFORE committing paid GPU hours.
#   R: Infrastructure entry point; never touches learning.
#   A: `python -m quantra.runtime` prints the plan + measured utilisation; ASCII-safe console output.
#   C: Prevents wasted GPU spend so budget goes to the validation that establishes the pass rate.
