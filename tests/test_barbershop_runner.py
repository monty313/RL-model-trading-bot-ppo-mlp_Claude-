"""C21 — barbershop_runner: real per-day metrics from a TradingEnv + policy.

Confirms run_pass() runs a deterministic policy over an N_DAYS continuous-account episode and returns
one REAL scoreboard row per day in the schema the Barbershop loop + Policy Registry consume. Uses a
cheap synthetic multi-day SymbolData (no feature build) so the test is fast and offline.
"""

import numpy as np

from quantra.runtime import config as cfg
from quantra.env.trading_env import SymbolData
from quantra.market_pipeline.feature_builder import PRECOMPUTED_DIM, PRECOMPUTED_NAMES
from quantra.learning_system.ppo_agent.agent import PPOAgent
from quantra.learning_system.barbershop_runner import build_env, run_pass, slice_symbol_data

_ROW_KEYS = {"day", "passed", "pnl_pct", "dd_pct", "breached", "trades",
             "wins", "closes", "win_rate", "gate_block_rate"}


def _multiday_data(days: int = 3, bpd: int = 30, atr: float = 1e-4):
    """A (T, PRECOMPUTED_DIM) gate-open matrix + integer day ids, spanning days+1 full days so the
    episode_days boundary (not end-of-data) cleanly ends the run. Mirrors the suite's _open_gate_matrix."""
    T = (days + 1) * bpd
    m = np.zeros((T, PRECOMPUTED_DIM), dtype=np.float32)
    for name, val in [("atr_dev_1m", 0.1), ("atr_dev_30m", 0.1),
                      ("spread_range_ratio_1m", 0.3), ("adf_stat_1m", -3.5)]:
        m[:, PRECOMPUTED_NAMES.index(name)] = val
    close = (1.20 + 1e-4 * np.sin(np.arange(T) / 3.0)).astype(float)
    dates = (np.arange(T) // bpd).astype(np.int64)        # calendar-day id per bar -> daily boundaries
    return {"EURUSD": SymbolData(matrix=m, close=close, atr=np.full(T, atr),
                                 spread=np.full(T, 2e-5), valid_from=0, dates=dates)}


def test_run_pass_returns_one_real_row_per_day():
    data = _multiday_data(days=3, bpd=30)
    rows = run_pass(PPOAgent(), data, overrides={"training_phase": "free"}, n_days=3)
    assert len(rows) == 3
    for i, r in enumerate(rows, 1):
        assert set(r) == _ROW_KEYS
        assert r["day"] == i
        assert isinstance(r["passed"], bool) and isinstance(r["breached"], bool)
        assert r["dd_pct"] <= 0.0                         # drawdown reported as a non-positive %
        assert r["trades"] >= 0 and 0.0 <= r["gate_block_rate"] <= 1.0
        # win rate is a real percentage over the day's discretionary closes (0..100), wins <= closes
        assert 0 <= r["wins"] <= r["closes"] and 0.0 <= r["win_rate"] <= 100.0


def test_run_pass_rows_feed_passrecord_schema():
    """The rows drop straight into the registry PassRecord.from_dict via summarize_pass keys."""
    from quantra.learning_system.policy_registry import PassRecord
    rows = run_pass(PPOAgent(), _multiday_data(days=2, bpd=25), overrides=None, n_days=2)
    avg = lambda k: round(sum(r[k] for r in rows) / len(rows), 3)
    summary = {"pass": 1, "days_passed": sum(r["passed"] for r in rows),
               "days_failed": sum(not r["passed"] for r in rows),
               "avg_pnl": avg("pnl_pct"), "avg_dd": avg("dd_pct"),
               "breach_count": sum(r["breached"] for r in rows),
               "avg_gate_block_rate": avg("gate_block_rate")}
    rec = PassRecord.from_dict(summary)                   # must not raise
    assert rec.days_passed + rec.days_failed == 2


def test_slice_symbol_data_windows_without_rebuild():
    sd = _multiday_data(days=3, bpd=30)["EURUSD"]
    sliced = slice_symbol_data(sd, 30)
    assert len(sliced.close) == len(sd.close) - 30
    assert sliced.matrix.shape == (len(sd.close) - 30, sd.matrix.shape[1])
    assert sliced.dates is not None and sliced.dates[0] == sd.dates[30]
    assert slice_symbol_data(sd, 0).close.shape == sd.close.shape   # index 0 == no-op window


def test_build_env_applies_overrides_and_phase():
    orig_phase = cfg.TRAINING_PHASE
    try:
        env = build_env(_multiday_data(days=1, bpd=20),
                        {"daily_target_pct": 3.0, "daily_risk_pct": 5.0, "training_wheels": False,
                         "training_phase": "constrained", "daily_progress_weight": 2e-3}, n_days=1)
        assert env.challenge_cfg.daily_target_pct == 3.0 and env.challenge_cfg.daily_risk_pct == 5.0
        assert env.training_wheels is False
        assert env.reward_cfg.daily_progress_weight == 2e-3
        assert cfg.TRAINING_PHASE == cfg.PHASE_CONSTRAINED     # global phase override applied
    finally:
        cfg.TRAINING_PHASE = orig_phase                        # don't leak global state to other tests


def test_build_env_applies_per_trade_risk_override():
    """A max_per_trade_risk_frac override caps position sizing in the built env (the structural lever
    for stopping breaches); absent, the env keeps the default 1%/trade RiskConfig."""
    default_env = build_env(_multiday_data(days=1, bpd=20), {"daily_target_pct": 2.5}, n_days=1)
    assert default_env.risk_cfg.max_per_trade_risk_frac == 0.01          # default untouched

    capped_env = build_env(_multiday_data(days=1, bpd=20),
                           {"daily_target_pct": 2.5, "max_per_trade_risk_frac": 0.0025}, n_days=1)
    assert capped_env.risk_cfg.max_per_trade_risk_frac == 0.0025          # override honored


def test_build_env_applies_cci_regime_gate_override():
    """The temporary CCI-regime open-gate is OFF by default (repo unchanged) and turns on only when the
    OVERRIDES ask for it — a reversible experiment knob, never altering the locked-core masks."""
    default_env = build_env(_multiday_data(days=1, bpd=20), {"daily_target_pct": 2.5}, n_days=1)
    assert default_env.cci_regime_gate == cfg.CCI_REGIME_GATE and default_env.cci_regime_gate is False

    gated_env = build_env(_multiday_data(days=1, bpd=20),
                          {"daily_target_pct": 2.5, "cci_regime_gate": True}, n_days=1)
    assert gated_env.cci_regime_gate is True                             # override honored


def test_run_pass_return_account_returns_canonical_challenge_state():
    """C14/Fix 5: return_account=True hands back the CANONICAL end-of-run ChallengeState (so the
    Barbershop card reads the LIVE consecutive_loss_days, not a value recomputed from the day flags),
    while the DEFAULT call is unchanged (a plain list of rows for every existing caller/test)."""
    from quantra.ftmo_passing import ChallengeState
    data = _multiday_data(days=3, bpd=30)                      # tiny moves -> the policy can't hit +2.5%

    rows_only = run_pass(PPOAgent(), data, overrides=None, n_days=3)   # default contract
    assert isinstance(rows_only, list) and len(rows_only) == 3

    rows, account = run_pass(PPOAgent(), data, overrides=None, n_days=3, return_account=True)
    assert isinstance(rows, list) and len(rows) == 3
    assert isinstance(account, ChallengeState)                 # the real runtime object, not a copy
    assert isinstance(account.consecutive_loss_days, int)
    # every finalized day fails (can't reach +2.5% on this data), so the live streak is real (>=1) and
    # never exceeds the days that hit a midnight boundary.
    assert 1 <= account.consecutive_loss_days <= 3


def test_checkpoint_payload_roundtrip_restores_weights():
    """C14/Fix 5 (A2 notebook-side save/load): the {state_dict, state_dim} payload the Barbershop loop
    writes via torch.save reloads into a fresh PPOAgent EXACTLY, and the state_dim guard is real.
    (No PPOAgent.save/load is added — agent.py stays untouched; this proves the notebook pattern.)"""
    import torch
    from quantra.market_pipeline.feature_builder import STATE_DIM

    a = PPOAgent()
    payload = {"state_dict": a.net.state_dict(), "state_dim": STATE_DIM}   # exactly what the notebook saves
    b = PPOAgent()                                                          # different random init
    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.net.parameters(), b.net.parameters()))
    assert int(payload["state_dim"]) == STATE_DIM                          # the guard Cell 4 checks before load
    b.net.load_state_dict(payload["state_dict"])                           # the load Cell 4 performs
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.net.parameters(), b.net.parameters()))
