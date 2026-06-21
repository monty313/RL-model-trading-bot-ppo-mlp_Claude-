"""Tests for the env-filled `trade_state` observation block (operator request 2026-06-21).

Covers: the schema contract (8 names, appended after `account`, STATE_DIM 215), assemble_state
placement, the block math/normalization, reset() init, the win/loss streak update in
_record_close, and an end-to-end env run (obs width 215, block finite + bounded, open bumps the
counters, position_open toggles) + the midnight reset of trades_today.

Envs are built from a tiny all-zeros precomputed matrix with valid_from=0 (the existing master-suite
pattern) so opens are legal (all laws 0) and reset() doesn't wait out the ~6000-bar 30m-BB200 warmup.
"""

from __future__ import annotations

import numpy as np

from quantra.env.trading_env import SymbolData, TradingEnv
from quantra.market_pipeline.feature_builder.schema import (
    EXPECTED_WIDTHS,
    PRECOMPUTED_DIM,
    SCHEMA,
    STATE_DIM,
)

HOLD, OPEN_LONG, CLOSE = 0, 1, 3   # law_mask_engine action ids
_TS = SCHEMA.block_spans["trade_state"]            # (start, end) of the new block
_NAMES = SCHEMA.blocks["trade_state"]
_TI = {n: i for i, n in enumerate(_NAMES)}         # index within the block


def _sd(T=120, close=None, dates=None, atr=0.001):
    # EURUSD-like price/ATR so the RiskManager can actually size a min lot into the buffer
    # (atr=1 on price 100 is a giant stop -> "below min lot"); all-zeros matrix keeps opens legal.
    mat = np.zeros((T, PRECOMPUTED_DIM), dtype=np.float32)
    close = np.full(T, 1.2) if close is None else np.asarray(close, dtype=float)
    # spread is in PRICE (≈1 pip); a large value blows the account on the open cost
    return SymbolData(mat, close=close, atr=np.full(T, atr),
                      spread=np.full(T, 1e-4), valid_from=0, dates=dates)


def _env(**kw):
    return TradingEnv({"EURUSD": _sd(**kw)})


# --------------------------------------------------------------------------- contract
def test_schema_block_contract():
    assert STATE_DIM == 215
    assert EXPECTED_WIDTHS["trade_state"] == 8
    assert _NAMES == [
        "daily_realized_pnl_pct", "daily_drawdown_pct", "trades_today",
        "consecutive_losses", "consecutive_wins", "position_open",
        "risk_budget_remaining", "time_since_last_trade",
    ]
    # appended at the very END, right after `account` (no pre-existing index shifted)
    assert _TS[1] == STATE_DIM
    assert SCHEMA.block_spans["account"][1] == _TS[0]


def test_assemble_state_places_trade_state_last():
    from quantra.market_pipeline.feature_builder.builder import assemble_state

    ts = np.arange(1, 9, dtype=np.float32)          # 8 distinct values
    state = assemble_state(np.zeros(PRECOMPUTED_DIM, dtype=np.float32), trade_state=ts)
    assert state.shape == (STATE_DIM,)
    np.testing.assert_array_equal(state[_TS[0]:_TS[1]], ts)
    # omitted -> zero-filled (live_session / warmup contract)
    state0 = assemble_state(np.zeros(PRECOMPUTED_DIM, dtype=np.float32))
    assert np.all(state0[_TS[0]:_TS[1]] == 0.0)


# ------------------------------------------------------------------------- block math
def test_block_math_and_normalization():
    env = _env()
    env.reset()
    blk = env._trade_state_block()
    assert blk.shape == (8,)
    assert blk[_TI["daily_realized_pnl_pct"]] == 0.0     # fresh reset: balance == day_start_balance
    assert blk[_TI["daily_drawdown_pct"]] == 0.0         # equity == peak
    assert blk[_TI["position_open"]] == 0.0              # nothing open
    assert blk[_TI["risk_budget_remaining"]] > 0.0       # the daily trailing band (% of account)

    env._trades_today = 10
    env._consec_losses = 5
    env._consec_wins = 3
    env._bars_since_trade = 30
    blk = env._trade_state_block()
    assert blk[_TI["trades_today"]] == 10 / 20.0
    assert blk[_TI["consecutive_losses"]] == 5 / 5.0
    assert blk[_TI["consecutive_wins"]] == 3 / 5.0
    assert blk[_TI["time_since_last_trade"]] == 30 / 60.0
    assert np.isfinite(blk).all() and (np.abs(blk) <= 10.0).all()


def test_position_open_flag():
    env = _env()
    env.reset()
    assert env._trade_state_block()[_TI["position_open"]] == 0.0
    env.slots["EURUSD"][0].occupied = True
    assert env._trade_state_block()[_TI["position_open"]] == 1.0


def test_reset_initializes_counters():
    env = _env()
    env._trades_today = 99
    env._consec_wins = 7
    env._consec_losses = 4
    env._bars_since_trade = 500
    env.reset()
    assert env._trades_today == 0
    assert env._consec_wins == 0
    assert env._consec_losses == 0
    assert env._bars_since_trade == 0
    assert env._day_start_balance == env.account.balance


# ------------------------------------------------------------------------- streaks
def test_record_close_updates_streaks():
    env = _env()
    env.reset()
    for _ in range(3):
        env._record_close(realized_gross=1.0, net=1.0, age=10, ever_in_profit=True)
    assert env._consec_wins == 3 and env._consec_losses == 0
    env._record_close(realized_gross=-1.0, net=-1.0, age=10, ever_in_profit=True)
    assert env._consec_losses == 1 and env._consec_wins == 0
    env._record_close(realized_gross=0.0, net=0.0, age=10, ever_in_profit=False)   # flat: no change
    assert env._consec_losses == 1 and env._consec_wins == 0


# ------------------------------------------------------------------------- integration
def test_env_obs_width_and_block_bounds_over_episode():
    # a gentle wave so some closes win and some lose (variety through the streak counters)
    T = 400
    wave = 1.2 + 0.002 * np.sin(np.arange(T) / 7.0)
    env = TradingEnv({"EURUSD": _sd(T=T, close=wave)})
    obs = env.reset()
    assert obs.shape == (STATE_DIM,)
    saw_open = saw_flat = saw_executed_open = False
    prev_trades = env._trades_today
    for _ in range(T):
        if env.done:
            break
        any_open = any(sl.occupied for sl in env.slots["EURUSD"])
        action = (OPEN_LONG, 0.5, 0) if not any_open else (CLOSE, 0.0, 0)
        obs, _r, done, info = env.step(action)
        if str(info.get("executed", "")).startswith("OPEN"):
            saw_executed_open = True
            assert env._trades_today >= prev_trades + 1     # the open was counted
            # the open reset the idle timer to 0; the within-step bar advance then makes it 1
            assert env._bars_since_trade <= 1
            prev_trades = env._trades_today
        if done:
            break
        assert obs.shape == (STATE_DIM,)
        ts = obs[_TS[0]:_TS[1]]
        assert np.isfinite(ts).all() and (np.abs(ts) <= 10.0).all()
        saw_open = saw_open or any(sl.occupied for sl in env.slots["EURUSD"])
        saw_flat = saw_flat or not any(sl.occupied for sl in env.slots["EURUSD"])
    assert saw_executed_open and env._trades_today > 0       # opens really happened + were counted
    assert saw_open and saw_flat                            # position_open toggled both ways


def test_trades_today_resets_at_midnight():
    T = 120
    dates = np.array([0] * 60 + [1] * 60)                    # one calendar-day boundary at t=60
    env = TradingEnv({"EURUSD": _sd(T=T, dates=dates)})
    env.reset()
    env._trades_today = 9
    start_days = env._days_elapsed
    while env.t + 1 < env.T and env._days_elapsed == start_days:
        env._advance_bar()
    assert env._days_elapsed > start_days, "no day boundary crossed"
    assert env._trades_today == 0                            # reset at midnight
    assert env._day_start_balance == env.account.balance
