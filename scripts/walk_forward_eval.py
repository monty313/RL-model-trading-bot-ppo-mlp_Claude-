"""scripts/walk_forward_eval.py — run the LOCKED walk-forward protocol on REAL bars.

WHAT THIS IS
------------
Glue that WIRES the existing, locked walk-forward harness
(``quantra/ftmo_passing/validation/walk_forward.py``: ``generate_windows`` + ``WalkForwardRunner`` +
``PromotionGate``, with ``Scoreboard`` from ``scoreboard.py``) to a REAL ``eval_fn``. Per
(window, seed) it:
  1. slices the cached SymbolData to the window's 12-month TRAIN span and 2-month TEST span,
  2. trains a FRESH PPOAgent on the train span (seeded — that's the per-seed robustness the
     protocol checks),
  3. runs the DETERMINISTIC policy on the held-out test span (via ``barbershop_runner.run_pass``),
  4. returns a ``RunResult`` (passed / breached / target_hit / max_drawdown / pnl).
Then it prints the ``Scoreboard`` summary + the ``PromotionGate`` decision.

This is ADDITIVE GLUE ONLY — the locked 12/2/1-month, 7-seed protocol and the I3 promotion
conditions in ``walk_forward.py`` / ``scoreboard.py`` are NOT modified. The ``eval_fn`` that
``WalkForwardRunner.run`` always required (the file said "the e2e wiring lands in M15") is supplied
here against real data.

WHEN TO USE WHICH OOS TOOL
--------------------------
  * Quick SINGLE-SPLIT out-of-sample today: ``scripts/real_backtest.py`` (train on a slice, test on
    the held-out tail — one window). Operational and fast.
  * Full ROLLING walk-forward (12mo train / 2mo test / 1mo step / N seeds): THIS script. Needs ~14+
    months of bars to produce even one window; a multi-year history to be meaningful.

Usage:
    python scripts/walk_forward_eval.py --symbol EURUSD --updates 200
    python scripts/walk_forward_eval.py --symbol EURUSD --path data/raw/EURUSD.csv --seeds 7 --updates 400

Rulebook (for the Risk Doctor / any LLM): docs/MLP_INTERPRETABILITY_LAYER.md.
COUPLING -> quantra/ftmo_passing/validation/* (locked protocol; this only supplies eval_fn),
env/trading_env.py (TradingEnv + prepare_symbol_data + SymbolData), learning_system/trainer
(Trainer), learning_system/barbershop_runner (run_pass — deterministic per-day eval + canonical
account), market_pipeline/data_loader (load_symbol).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantra.runtime import config as cfg                                       # noqa: E402
from quantra.market_pipeline.data_loader import load_symbol                     # noqa: E402
from quantra.env.trading_env import TradingEnv, prepare_symbol_data, SymbolData  # noqa: E402
from quantra.learning_system.ppo_agent.agent import PPOAgent                    # noqa: E402
from quantra.learning_system.trainer.trainer import Trainer, TrainConfig        # noqa: E402
from quantra.learning_system.barbershop_runner import run_pass                  # noqa: E402
from quantra.ftmo_passing.validation import (                                   # noqa: E402
    PromotionGate, RunResult, Scoreboard, WalkForwardRunner, Window, generate_windows)


def _slice(sd: SymbolData, lo: int, hi: int) -> SymbolData:
    """Window a cached SymbolData to bars [lo:hi] WITHOUT a feature rebuild (rows align 1:1 with the
    source df, same trick as barbershop_runner.slice_symbol_data / real_backtest's split)."""
    cut = lambda a: None if a is None else a[lo:hi]
    return SymbolData(matrix=sd.matrix[lo:hi], close=sd.close[lo:hi], atr=sd.atr[lo:hi],
                      spread=sd.spread[lo:hi], valid_from=0, dates=cut(sd.dates))


def make_eval_fn(sd: SymbolData, index, symbol: str, challenge, overrides: dict, *,
                 updates: int, train_cfg: Optional[TrainConfig] = None
                 ) -> Callable[[Window, int], RunResult]:
    """Build the ``eval_fn(window, seed) -> RunResult`` that ``WalkForwardRunner.run`` calls.

    `index` is the SOURCE bar DatetimeIndex (the same one passed to generate_windows / run); window
    timestamps are mapped back to integer bar positions with searchsorted so the cached SymbolData is
    sliced without re-running the feature build. A FRESH PPOAgent is trained per (window, seed)."""
    tcfg = train_cfg or TrainConfig()

    def _pos(ts) -> int:
        return int(index.searchsorted(pd.Timestamp(ts)))

    def eval_fn(window: Window, seed: int) -> RunResult:
        train_sd = _slice(sd, _pos(window.train_start), _pos(window.train_end))
        test_sd = _slice(sd, _pos(window.test_start), _pos(window.test_end))

        # 1) train a fresh brain on the TRAIN span (seeded -> per-seed robustness the protocol checks)
        agent = PPOAgent()
        tenv = TradingEnv({symbol: train_sd}, challenge=challenge)
        Trainer(tenv, agent=agent,
                train_cfg=TrainConfig(rollout_size=tcfg.rollout_size, minibatch=tcfg.minibatch,
                                      value_coef=tcfg.value_coef, seed=seed)).train(updates)

        # 2) DETERMINISTIC eval on the held-out TEST span -> real per-day rows + the canonical account.
        #    n_days huge -> run_pass runs to end-of-test-data, splitting at the CE(S)T day boundaries.
        rows, account = run_pass(agent, {symbol: test_sd}, overrides, n_days=10_000,
                                 deterministic=True, return_account=True)

        # 3) map to a RunResult per scoreboard.py's contract: passed == hit the daily target AND never
        #    breached the trailing wall over the window. (A per-day "passed" row already means that day
        #    hit target without breaching, so target_hit == any passed day.)
        breached = bool(account.breached or any(r["breached"] for r in rows))
        target_hit = any(r["passed"] for r in rows)
        passed = bool(target_hit and not breached)
        max_dd = max((abs(r["dd_pct"]) for r in rows), default=0.0) / 100.0   # fraction; lower better
        pnl = float(account.equity - account.account_size)                    # diagnostic only
        return RunResult(passed=passed, breached=breached, target_hit=target_hit,
                         max_drawdown=max_dd, pnl=pnl)

    return eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description="Rolling walk-forward OOS validation on real bars.")
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--path", default=None,
                    help="explicit CSV path; omit to use load_symbol's cache/Drive/gdown fallback")
    ap.add_argument("--updates", type=int, default=200, help="PPO updates per (window, seed)")
    ap.add_argument("--seeds", type=int, default=7, help="seeds per window (locked protocol = 7)")
    ap.add_argument("--target", type=float, default=2.5)
    ap.add_argument("--risk", type=float, default=4.0)
    a = ap.parse_args()

    t0 = time.time()
    df, rep = load_symbol(a.symbol, path=Path(a.path) if a.path else None)
    print(f"[load] {len(df):,} bars  {df.index.min()} -> {df.index.max()}  source={rep.source}  "
          f"({time.time()-t0:.1f}s)")
    sd = prepare_symbol_data(df, symbol=a.symbol)

    windows = generate_windows(df.index)   # locked 12mo train / 2mo test / 1mo step
    if not windows:
        raise SystemExit("Not enough data for even one 12mo-train / 2mo-test window. "
                         "Use scripts/real_backtest.py for a single-split OOS run instead.")
    print(f"[walk-forward] {len(windows)} window(s) x {a.seeds} seed(s) — locked 12/2/1-month protocol")

    challenge = cfg.make_challenge(daily_target_pct=a.target, daily_risk_pct=a.risk)
    overrides = {"daily_target_pct": a.target, "daily_risk_pct": a.risk}
    eval_fn = make_eval_fn(sd, df.index, a.symbol, challenge, overrides, updates=a.updates)

    scoreboard, seed_pass = WalkForwardRunner(n_seeds=a.seeds).run(df.index, eval_fn)
    print("[scoreboard]", scoreboard.summary())
    ok, reason = PromotionGate().promote(scoreboard, baseline=None, seed_pass_counts=seed_pass)
    print(f"[promotion] promote={ok} :: {reason}")
    print(f"[seeds] per-seed window-pass counts: {seed_pass}")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-19] Issue-3 (Phase C) — wired the locked walk-forward harness to a real eval_fn.
#   I: validation/walk_forward.py had the protocol (generate_windows + WalkForwardRunner +
#      PromotionGate) but no concrete eval_fn ("e2e wiring lands in M15"), so there was no way to run
#      a real rolling out-of-sample walk-forward — only the single-split scripts/real_backtest.py.
#   R: Operator Phase C (make walk-forward operational; ADD glue, do NOT modify the locked protocol).
#   A: New scripts/walk_forward_eval.py: make_eval_fn() slices the cached SymbolData to each window's
#      train/test spans (searchsorted on the source index, no feature rebuild), trains a fresh seeded
#      PPOAgent on the train span, deterministic-evals on the test span via run_pass (real per-day rows
#      + canonical account), and returns a RunResult per scoreboard.py's contract. main() runs the full
#      WalkForwardRunner -> Scoreboard -> PromotionGate path on real bars. No locked file changed.
#   C: The 12/2/1-month, N-seed walk-forward is now operational on real data, so a brain can be judged
#      on rolling OOS windows (not one lucky split) before promotion — the real test of repeatable passing.
