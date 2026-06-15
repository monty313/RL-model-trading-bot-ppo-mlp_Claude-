"""Background utilisation sampler — proves we actually hit the ~80% target.

WHAT THIS MODULE DOES
---------------------
Spawns a lightweight background thread that periodically samples CPU% (via
``psutil``) and, when present, GPU% + GPU memory (via ``pynvml``). It exposes a
running summary (mean / peak utilisation) that the optimizer prints in the run
report and the trainer can fold into telemetry.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
"Optimise for 80% usage" is only credible if we *measure* it. This monitor closes
the loop: if the chosen device sits at 15%, the report says so and the operator
can stop paying for it. Confirmed-efficient runs mean the walk-forward validation
that selects for pass-rate completes on time and on budget.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. Utilisation is an infrastructure
health metric. Low GPU utilisation explains COST/SPEED, never a trading failure.
Keep it out of the 8-failure taxonomy.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

# Optional deps — degrade gracefully so a bare environment still imports/runs.
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:  # NVIDIA Management Library bindings (nvidia-ml-py / pynvml)
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None


# COUPLING -> __main__.py: __main__ calls UtilizationMonitor as a context manager then
# prints summary.render(); the trainer/telemetry may also fold these fields in. The field
# names + render() string shape are the consumed contract.
@dataclass
class UtilizationSummary:
    """Aggregated utilisation over a monitored window."""

    samples: int = 0
    cpu_mean: float = 0.0
    cpu_peak: float = 0.0
    gpu_mean: float = 0.0
    gpu_peak: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    gpu_available: bool = False
    _cpu: List[float] = field(default_factory=list, repr=False)
    _gpu: List[float] = field(default_factory=list, repr=False)

    def render(self) -> str:
        """One-line human summary for the launch report."""
        if self.gpu_available:
            return (
                f"CPU {self.cpu_mean:4.0f}% mean / {self.cpu_peak:4.0f}% peak | "
                f"GPU {self.gpu_mean:4.0f}% mean / {self.gpu_peak:4.0f}% peak "
                f"({self.gpu_mem_peak_mb:.0f} MB peak)"
            )
        return f"CPU {self.cpu_mean:4.0f}% mean / {self.cpu_peak:4.0f}% peak (no GPU)"


class UtilizationMonitor:
    """Start/stop sampler. Use as a context manager around the work to measure."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._summary = UtilizationSummary()
        self._nvml_handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._summary.gpu_available = True
            except Exception:  # pragma: no cover
                self._nvml_handle = None

    def _sample_loop(self) -> None:
        # Prime psutil so the first cpu_percent() isn't a meaningless 0.0.
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            if psutil is not None:
                self._summary._cpu.append(psutil.cpu_percent(interval=None))
            if self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self._summary._gpu.append(float(util.gpu))
                    self._summary.gpu_mem_peak_mb = max(
                        self._summary.gpu_mem_peak_mb, mem.used / 1e6
                    )
                except Exception:  # pragma: no cover
                    pass
            self._stop.wait(self.interval)

    def start(self) -> "UtilizationMonitor":
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> UtilizationSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        s = self._summary
        if s._cpu:
            s.cpu_mean = sum(s._cpu) / len(s._cpu)
            s.cpu_peak = max(s._cpu)
        if s._gpu:
            s.gpu_mean = sum(s._gpu) / len(s._gpu)
            s.gpu_peak = max(s._gpu)
        s.samples = max(len(s._cpu), len(s._gpu))
        return s

    def __enter__(self) -> "UtilizationMonitor":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Prove the 80% target is met.
#   I: 'Optimise for 80% usage' is not credible unless it is measured.
#   R: Infrastructure health metric; explicitly OUTSIDE the 8-failure taxonomy.
#   A: Background CPU/GPU sampler with graceful degradation (psutil/pynvml optional); reported at launch.
#   C: Confirmed-efficient runs mean validation completes on time/budget; low GPU% explains cost, never a breach.
