"""The hardware auto-optimizer — one call that picks device + parallelism.

WHAT THIS MODULE DOES
---------------------
``plan()`` races CPU vs GPU on a representative workload, applies the user's
"prefer CPU / don't waste GPU money" policy, sizes vectorised-env parallelism to
~80% of the winner, pins thread counts, and returns a ``HardwarePlan`` that the
trainer consumes. ``print_report()`` renders the decision for the operator.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
This is the module that fulfils the operator's instruction: train as fast and
cheap as possible by using ~80% of whichever device is genuinely faster. Faster,
cheaper rollouts -> more seeds and walk-forward windows validated -> a more
trustworthy FTMO pass rate, which is the only scoreboard that matters.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. The HardwarePlan is pure
infrastructure. It changes how *fast* training runs, never *what* it learns. When
diagnosing pass/breach behaviour, ignore device/n_envs — they are not in the
causal chain (State Vector -> Law -> Hidden State -> Heads -> Risk -> Reward ->
Outcome). Treat this only as context for runtime/cost questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .autoscale import ScalePlan, apply_thread_limits, plan_cpu_scale, plan_gpu_scale
from .config import HardwareConfig
from .device import available_devices
from .throughput_benchmark import BenchResult, race_devices


# COUPLING -> __main__.py + learning_system/trainer/trainer.py: print_report (in this file)
# and the trainer read these field names (device, scale, bench, decision_note, n_envs).
# Renaming a field breaks the report and the trainer's hardware consumption.
@dataclass(frozen=True)
class HardwarePlan:
    """The resolved runtime compute decision."""

    device: str            # torch device string, e.g. "cpu" or "cuda:0"
    device_kind: str       # "cpu" | "cuda" | "mps"
    device_label: str
    scale: ScalePlan
    bench: List[BenchResult]
    decision_note: str

    @property
    def n_envs(self) -> int:
        return self.scale.n_envs


def _pick_device(bench: List[BenchResult], hw: HardwareConfig):
    """Apply the prefer-CPU policy to the benchmark results.

    Rule (encodes the user's money instruction): a GPU is selected ONLY if it
    beats CPU throughput by at least ``gpu_speedup_required`` (default 1.30x).
    Otherwise we stay on CPU — even if the GPU is marginally faster — because a
    near-tie does not justify a paid accelerator on this tiny network.
    """
    # COUPLING -> throughput_benchmark.py BenchResult (.kind/.steps_per_sec/.label) +
    # config.py HardwareConfig (.gpu_speedup_required/.prefer_cpu). Field renames in either
    # break this device-pick policy. Kind strings "cpu"/"cuda"/"mps" mirror device.py.
    by_kind = {b.kind: b for b in bench}
    cpu = by_kind.get("cpu")
    cpu_sps = cpu.steps_per_sec if cpu else 0.0

    # Best non-CPU accelerator, if any.
    accel = max(
        (b for b in bench if b.kind in ("cuda", "mps") and b.steps_per_sec > 0),
        key=lambda b: b.steps_per_sec,
        default=None,
    )

    if accel is None:
        return "cpu", cpu.label if cpu else "CPU", "no accelerator available -> CPU"

    if cpu_sps <= 0:
        return accel.kind, accel.label, "CPU benchmark unavailable -> accelerator"

    speedup = accel.steps_per_sec / cpu_sps
    if speedup >= hw.gpu_speedup_required:
        return (
            accel.kind,
            accel.label,
            f"{accel.label} is {speedup:.2f}x CPU (>= {hw.gpu_speedup_required:.2f}x) -> worth the cost",
        )
    if hw.prefer_cpu:
        return (
            "cpu",
            cpu.label,
            f"{accel.label} only {speedup:.2f}x CPU (< {hw.gpu_speedup_required:.2f}x) -> "
            f"stay on CPU, drop the GPU runtime to save money",
        )
    return accel.kind, accel.label, f"{accel.label} {speedup:.2f}x CPU"


def plan(
    state_dim: int,
    hw: Optional[HardwareConfig] = None,
    benchmark_seconds: Optional[float] = None,
) -> HardwarePlan:
    """Race devices, choose one, size parallelism, pin threads. Returns the plan.

    ``state_dim`` should be the real feature width once M2 exists; at M0 the
    nominal 145 is fine for selecting hardware.
    """
    hw = hw or HardwareConfig()
    secs = benchmark_seconds if benchmark_seconds is not None else hw.benchmark_seconds

    bench = race_devices(state_dim=state_dim, seconds=secs, devices=available_devices())
    kind, label, note = _pick_device(bench, hw)

    if kind == "cpu":
        scale = plan_cpu_scale(hw)
    else:
        scale = plan_gpu_scale(hw)
    apply_thread_limits(scale)

    device = "cuda:0" if kind == "cuda" else kind
    return HardwarePlan(
        device=device,
        device_kind=kind,
        device_label=label,
        scale=scale,
        bench=bench,
        decision_note=note,
    )


def print_report(p: HardwarePlan) -> None:
    """Print the device decision + parallelism so the operator can see the spend."""
    print("=" * 68)
    print("QUANTRA HARDWARE PLAN")
    print("=" * 68)
    print("Throughput race (env-steps/sec, higher is better):")
    for b in p.bench:
        flag = "  <-- chosen" if b.label == p.device_label else ""
        extra = f"  [{b.note}]" if b.note else ""
        print(f"  {b.label:32s} {b.steps_per_sec:12,.0f}{flag}{extra}")
    print("-" * 68)
    print(f"Decision : {p.decision_note}")
    print(f"Device   : {p.device}  ({p.device_label})")
    print(f"Scale    : {p.scale.rationale}")
    print(f"Envs     : {p.n_envs} parallel worlds | torch threads: {p.scale.torch_threads}")
    print("=" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] One call: race -> prefer-CPU -> scale -> report.
#   I: Choosing a device blindly either wastes GPU money or starves throughput.
#   R: Prefer CPU unless a GPU is >=1.30x faster; ~80% autoscale; infra is outside the learning causal chain.
#   A: race_devices -> _pick_device policy -> autoscale -> thread pin -> print_report.
#   C: The cheapest fast substrate -> more seeds/windows -> a more trustworthy FTMO pass rate.
