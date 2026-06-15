"""Device discovery + a representative policy for the startup throughput race.

WHAT THIS MODULE DOES
---------------------
Enumerates the compute devices actually available (CPU always; CUDA / Apple MPS
when present) and builds a *representative* PPO network matching the locked
architecture (~145 inputs, 3x256 shared trunk, four heads) so the benchmark can
measure real rollout+update throughput per device before a single training step
is taken for real.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The mission needs many walk-forward windows x 7 seeds. Throughput is the budget.
Choosing the device empirically — rather than blindly grabbing a GPU — means we
spend the cheapest fast option, run more validation, and don't pay for idle CUDA
on a tiny MLP. The representative model deliberately mirrors PPO_ENGINE.md's
four-head 3x256 design so the race reflects the real workload, not a toy.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. This module never touches
policy weights used in training; it only constructs a throwaway clone for timing.
A device choice has NO bearing on what the bot learned — do not attribute any
pass/breach behaviour to CPU-vs-GPU selection. It is purely an infrastructure
decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DeviceInfo:
    """A candidate compute device and a human-readable label for reports."""

    kind: str          # "cpu" | "cuda" | "mps"
    torch_device: str  # e.g. "cpu", "cuda:0"
    label: str         # e.g. "CPU (16 logical cores)", "NVIDIA T4"


def available_devices() -> List[DeviceInfo]:
    """List every device we could train on, CPU first (the preferred default)."""
    import os

    devices: List[DeviceInfo] = [
        DeviceInfo("cpu", "cpu", f"CPU ({os.cpu_count()} logical cores)")
    ]
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        devices.append(DeviceInfo("cuda", "cuda:0", f"GPU ({name})"))
    # Apple Silicon — relevant for local dev on Macs, never on Colab.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.append(DeviceInfo("mps", "mps", "GPU (Apple MPS)"))
    return devices


class RepresentativePolicy(nn.Module):
    """A stand-in for the real PPOAgent (M5) used ONLY by the benchmark.

    Mirrors the locked shape so timing is honest:
      * shared 3x256 MLP trunk (PPO_ENGINE.md G5),
      * direction head: 4 logits {HOLD, OPEN_LONG, OPEN_SHORT, CLOSE},
      * size head: 2 params (Beta alpha/beta) (G3),
      * pointer head: 5 logits over trade slots (B2),
      * value head: scalar V(s).
    The real agent will subclass/replace this; the benchmark only needs forward
    cost to be representative, so weights are random and never saved.
    """

    # COUPLING [C2/C3 in COUPLINGS.md]: these head widths MUST mirror the real
    # ppo_agent.ActorCritic (direction 4, Beta size 2, pointer N_SLOTS=5). If the
    # action space changes, update both — else the benchmark times the wrong shape.
    DIRECTION_ACTIONS = 4
    SIZE_PARAMS = 2
    N_SLOTS = 5

    def __init__(self, state_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        d = state_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.direction_head = nn.Linear(hidden, self.DIRECTION_ACTIONS)
        self.size_head = nn.Linear(hidden, self.SIZE_PARAMS)
        self.pointer_head = nn.Linear(hidden, self.N_SLOTS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        # COUPLING -> throughput_benchmark.py: _bench_one unpacks this 4-tuple in order
        # (dlog, size, plog, v). Reorder/extend these heads => benchmark loss line breaks.
        return (
            self.direction_head(h),
            self.size_head(h),
            self.pointer_head(h),
            self.value_head(h),
        )


def resolve_device(kind: str) -> torch.device:
    """Map a DeviceInfo.kind to a concrete ``torch.device``."""
    return torch.device("cuda:0" if kind == "cuda" else kind)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Representative net locked to the four-head 3x256 shape.
#   I: The device race could mis-time the policy if its benchmark net didn't match the locked architecture.
#   R: PPO_ENGINE four-head (direction/Beta-size/pointer/value) 3x256 lock + the cost mandate.
#   A: Mirrored the locked heads in RepresentativePolicy; the shape is pinned by the master suite.
#   C: An honest race picks the genuinely faster device, so training time/cost isn't wasted -> more validated seeds.
