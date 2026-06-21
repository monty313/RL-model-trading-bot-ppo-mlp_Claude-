"""Issue-3 (Phase C) — smoke tests for the walk-forward GLUE (scripts/walk_forward_eval.py).

These prove the glue wires the LOCKED harness end-to-end WITHOUT re-running heavy training over years
of data:
  * test_runner_drives_windows... uses WalkForwardRunner.run over a real multi-window index with a
    deterministic STUB eval_fn -> proves the harness loops windows x seeds into a Scoreboard + a
    PromotionGate decision.
  * test_real_eval_fn_returns_valid_runresult calls the REAL eval_fn (make_eval_fn) once on a small
    synthetic SymbolData -> proves it actually trains + deterministic-evals + returns a valid RunResult.
Together: multiple-window scoreboard/promotion output AND real RunResult production, kept light + deterministic.
"""

import numpy as np
import pandas as pd

from quantra.runtime import config as cfg
from quantra.env.trading_env import SymbolData
from quantra.market_pipeline.feature_builder import PRECOMPUTED_DIM, PRECOMPUTED_NAMES
from quantra.learning_system.trainer.trainer import TrainConfig
from quantra.ftmo_passing.validation import (
    PromotionGate, RunResult, WalkForwardRunner, Window, generate_windows)
from scripts.walk_forward_eval import _slice, make_eval_fn


def _gate_matrix(n):
    m = np.zeros((n, PRECOMPUTED_DIM), dtype=np.float32)
    for name, val in [("atr_dev_1m", 0.1), ("atr_dev_30m", 0.1),
                      ("spread_range_ratio_1m", 0.3), ("adf_stat_1m", -3.5)]:
        m[:, PRECOMPUTED_NAMES.index(name)] = val
    return m


def test_runner_drives_windows_into_scoreboard_and_promotion():
    """WalkForwardRunner.run over a real multi-window index + a deterministic stub eval_fn yields a
    Scoreboard (N windows x seeds) and a PromotionGate (bool, reason). No training — harness wiring only."""
    idx = pd.date_range("2021-01-01", periods=20 * 30, freq="D")   # ~20 months -> several 12/2/1 windows
    windows = generate_windows(idx)
    assert len(windows) >= 2                                       # the locked protocol made >1 window

    def stub_eval(window: Window, seed: int) -> RunResult:         # even seeds "pass", seed 1 breaches
        return RunResult(passed=(seed % 2 == 0), breached=(seed == 1),
                         target_hit=(seed % 2 == 0), max_drawdown=0.02, pnl=10.0)

    scoreboard, seed_pass = WalkForwardRunner(n_seeds=3).run(idx, stub_eval)
    assert scoreboard.n == len(windows) * 3
    assert 0.0 <= scoreboard.pass_rate <= 1.0
    assert set(scoreboard.summary()) >= {"pass_rate", "breaches", "n"}
    assert len(seed_pass) == 3
    ok, reason = PromotionGate().promote(scoreboard, baseline=None, seed_pass_counts=seed_pass)
    assert isinstance(ok, bool) and isinstance(reason, str)


def test_real_eval_fn_returns_valid_runresult():
    """The REAL eval_fn (make_eval_fn) trains a fresh seeded brain on the window's train span and
    deterministic-evals on its test span, returning a well-formed RunResult — on cheap synthetic bars."""
    DAY = 8                                   # bars/day (cheap); 12 synthetic days
    n = 12 * DAY
    idx = pd.date_range("2021-01-01", periods=n, freq="3h")
    sd = SymbolData(_gate_matrix(n), close=np.full(n, 1.20), atr=np.full(n, 1e-3),
                    spread=np.full(n, 2e-5), valid_from=0,
                    dates=(np.arange(n) // DAY).astype(np.int64))
    challenge = cfg.make_challenge(daily_target_pct=2.5, daily_risk_pct=4.0)
    eval_fn = make_eval_fn(sd, idx, "EURUSD", challenge,
                           {"daily_target_pct": 2.5, "daily_risk_pct": 4.0},
                           updates=1, train_cfg=TrainConfig(rollout_size=32, minibatch=16))
    window = Window(idx[0], idx[8 * DAY], idx[8 * DAY], idx[n - 1])   # ~8d train, ~4d test

    r = eval_fn(window, seed=0)
    assert isinstance(r, RunResult)
    assert isinstance(r.passed, bool) and isinstance(r.breached, bool) and isinstance(r.target_hit, bool)
    assert isinstance(r.max_drawdown, float) and r.max_drawdown >= 0.0
    assert isinstance(r.pnl, float)


def test_slice_windows_symboldata_without_rebuild():
    n = 40
    sd = SymbolData(_gate_matrix(n), close=np.arange(n, dtype=float), atr=np.full(n, 1e-3),
                    spread=np.full(n, 2e-5), valid_from=0, dates=np.arange(n, dtype=np.int64))
    s = _slice(sd, 10, 25)
    assert len(s.close) == 15 and s.matrix.shape == (15, PRECOMPUTED_DIM)
    assert s.dates is not None and s.dates[0] == 10 and s.valid_from == 0
