"""Startup throughput race: measure real steps/sec per device, pick the winner.

WHAT THIS MODULE DOES
---------------------
Runs a short, representative PPO-style workload (batched forward over a vectorised
rollout + a backward/optimizer step on the four-head 3x256 net) on each available
device for a fixed wall-time budget, and reports throughput in
*environment-steps per second*. The optimizer uses these numbers to choose the
device honestly instead of assuming a GPU is faster.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
For a ~145-input 3x256 MLP the GPU's compute advantage is usually swamped by
host<->device transfer overhead and the env step (numpy) cost. Measuring this up
front prevents the classic waste: renting a Colab GPU that then sits ~5% utilised
while CPU env-stepping bottlenecks the run. Cheaper, faster iterations -> more
seeds and windows validated per dollar -> a more trustworthy pass rate.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. The benchmark result is an
infrastructure fact (steps/sec), never a learning signal. If a run is slow, this
file's numbers tell you whether the bottleneck is device or env — but slowness is
never a *failure-taxonomy* item. Do not classify throughput as Risk Blindness etc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import torch

# COUPLING -> device.py: RepresentativePolicy.forward returns (direction, size, pointer,
# value) in that order; _bench_one below unpacks it as (dlog, size, plog, v). Keep the
# head order in device.py and the unpack here in sync.
from .device import DeviceInfo, RepresentativePolicy, available_devices, resolve_device


# COUPLING -> optimizer.py: optimizer._pick_device + print_report read these exact field
# names (kind, label, steps_per_sec, note). Rename a field => the device-pick/report break.
@dataclass(frozen=True)
class BenchResult:
    """Per-device benchmark outcome, ready for the optimizer + the run report."""

    kind: str
    label: str
    steps_per_sec: float
    batch: int
    note: str = ""


def _bench_one(
    dev: DeviceInfo,
    state_dim: int,
    batch: int,
    seconds: float,
) -> BenchResult:
    """Time one device on a representative rollout+update loop.

    The workload mimics PPO's inner loop: a batched policy forward (the vectorised
    env produces ``batch`` observations per step), a cheap surrogate loss, and a
    backward+step. We count how many environment-steps we can push through within
    ``seconds`` of wall time, which is exactly the quantity training is bottlenecked
    on.
    """
    torch_dev = resolve_device(dev.kind)
    try:
        model = RepresentativePolicy(state_dim).to(torch_dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Static input buffer reused each step (training reuses a rollout tensor).
        x = torch.randn(batch, state_dim, device=torch_dev)

        # Warm-up so lazy CUDA context / cuDNN autotune isn't charged to the race.
        for _ in range(3):
            dlog, size, plog, v = model(x)
            loss = dlog.pow(2).mean() + size.pow(2).mean() + plog.pow(2).mean() + v.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if dev.kind == "cuda":
            torch.cuda.synchronize()

        steps = 0
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < seconds:
            dlog, size, plog, v = model(x)
            loss = dlog.pow(2).mean() + size.pow(2).mean() + plog.pow(2).mean() + v.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        if dev.kind == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        sps = (steps * batch) / elapsed if elapsed > 0 else 0.0
        return BenchResult(dev.kind, dev.label, sps, batch)
    except Exception as exc:  # pragma: no cover - defensive: a bad GPU shouldn't crash launch
        return BenchResult(dev.kind, dev.label, 0.0, batch, note=f"benchmark failed: {exc}")


def race_devices(
    state_dim: int,
    batch: int = 2048,
    seconds: float = 2.5,
    devices: Optional[List[DeviceInfo]] = None,
) -> List[BenchResult]:
    """Benchmark every available device and return results (CPU first).

    ``batch`` approximates one vectorised rollout slice; 2048 is large enough to
    give a GPU its best shot while staying realistic for this net.
    """
    devices = devices or available_devices()
    return [_bench_one(d, state_dim, batch, seconds) for d in devices]


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Measure CPU-vs-GPU instead of assuming.
#   I: Assuming a GPU is faster wastes paid hours on a ~145-input 3x256 MLP.
#   R: Operator cost mandate; measure-don't-assume; infra fact, never a learning signal.
#   A: Time env-steps/sec per device over a fixed budget; a failing GPU degrades to 0, never crashes launch.
#   C: Cheap, fast iteration -> more walk-forward windows validated per dollar -> a trustworthy pass rate.
