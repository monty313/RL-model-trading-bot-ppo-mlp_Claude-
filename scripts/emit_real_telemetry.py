"""Emit a REAL telemetry run from REAL EURUSD bars -> artifacts/telemetry/<run>.jsonl.

This is the producer the Barbershop dashboard's "quantra" source reads. It runs the
DETERMINISTIC policy over a held-out slice of real MT5 bars and logs a real StepPacket
per bar (+ a per-day packet) via the canonical TelemetryLogger, populating the
risk-context fields the barbershop adapter maps onto the dashboard contract.

It does NOT train and does NOT change the policy — it just records what the (loaded or
fresh) brain does on real prices, so the dashboard shows real data instead of mock.

Usage:
    python scripts/emit_real_telemetry.py --symbol EURUSD --path data/raw/EURUSD_recent.csv --days 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantra.runtime import config as cfg                                       # noqa: E402
from quantra.market_pipeline.data_loader import load_symbol                     # noqa: E402
from quantra.env.trading_env import TradingEnv, prepare_symbol_data, SymbolData  # noqa: E402
from quantra.learning_system.ppo_agent.agent import PPOAgent                    # noqa: E402
from quantra.market_pipeline.law_mask_engine.engine import build_pointer_mask   # noqa: E402
from quantra.market_pipeline.feature_builder.schema import STATE_DIM            # noqa: E402
from quantra.diagnostics.telemetry_logger.logger import TelemetryLogger, StepPacket  # noqa: E402


def _regime_label(day_returns: np.ndarray) -> str:
    """Heuristically label a day's regime from its 1m returns (honest, simple)."""
    if len(day_returns) < 2:
        return "Slow Day"
    net = abs(float(np.nansum(day_returns)))            # net drift over the day
    vol = float(np.nanstd(day_returns))                 # choppiness
    if net > 4 * vol and net > 0:
        return "Trending"
    if vol > 0 and net < vol:
        return "Choppy"
    if vol == 0:
        return "Slow Day"
    return "News Day" if vol > 2 * (np.nanmedian(np.abs(day_returns)) + 1e-12) else "Slow Day"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--path", default="data/raw/EURUSD_recent.csv")
    ap.add_argument("--days", type=int, default=4)            # how many real days to record
    ap.add_argument("--train_frac", type=float, default=0.7)  # use the held-out tail
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/acceptance_brain.pt")
    ap.add_argument("--run_id", default="real_eurusd")
    a = ap.parse_args()

    t0 = time.time()
    df, rep = load_symbol(a.symbol, path=Path(a.path))
    print(f"[load] {len(df):,} REAL bars  {df.index.min()} -> {df.index.max()}  ({time.time()-t0:.1f}s)")
    sd = prepare_symbol_data(df, symbol=a.symbol)
    T, vf = len(sd.close), sd.valid_from
    split = vf + int((T - vf) * a.train_frac)
    test = SymbolData(sd.matrix[split:], sd.close[split:], sd.atr[split:],
                      sd.spread[split:], valid_from=0,
                      dates=(sd.dates[split:] if sd.dates is not None else None))
    test_index = df.index[split:]                            # real timestamps for the slice
    print(f"[slice] held-out test bars={len(test.close):,}  recording up to {a.days} real days")

    challenge = cfg.make_challenge()
    agent = PPOAgent()
    # Use a trained brain if its width matches the current schema; else a fresh policy.
    ckpt = Path(a.checkpoint)
    if ckpt.exists():
        try:
            blob = torch.load(ckpt, map_location="cpu")
            if int(blob.get("state_dim", -1)) == STATE_DIM:
                agent.net.load_state_dict(blob["state_dict"])
                print(f"[brain] loaded {ckpt.name} (state_dim={STATE_DIM})")
            else:
                print(f"[brain] {ckpt.name} state_dim={blob.get('state_dim')} != {STATE_DIM} -> fresh policy")
        except Exception as e:                              # corrupt / incompatible -> fresh
            print(f"[brain] could not load {ckpt.name} ({e}) -> fresh policy")
    else:
        print("[brain] no checkpoint -> fresh (untrained) policy")

    env = TradingEnv({a.symbol: test}, challenge=challenge)
    log = TelemetryLogger(run_id=a.run_id, out_dir=cfg.TELEMETRY_DIR)

    obs = env.reset()
    sym = a.symbol
    episode_id = 0                                          # 0-based day index (adapter -> day_id+1)
    cur_date = test.dates[env.t] if test.dates is not None else 0
    day_start_t = env.t
    day_rets: list = []
    done = False
    step_in_day = 0
    total_steps = 0

    def close_day(eid: int, rets: list):
        """Emit the per-day packet (regime + pass/breach) the scoreboard reads."""
        passed = bool(env.account.target_hit and not env.account.breached)
        log.log_day({"episode_id": eid, "day_id": eid + 1,
                     "regime": _regime_label(np.asarray(rets, float)),
                     "pass_result": passed, "dd_breached": bool(env.account.breached),
                     "day_pnl_pct": float(env.account.day_pnl / env.account.account_size * 100.0)})

    while not done:
        row = env.data[sym].matrix[env.t]
        dm = torch.as_tensor(env.direction_mask(sym), dtype=torch.float32)
        occ = [s.occupied for s in env.slots[sym]]
        pm = torch.as_tensor(build_pointer_mask(occ), dtype=torch.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # Real policy outputs: masked action distribution + value + chosen (argmax) action.
        with torch.no_grad():                                # inference only -> no autograd graph
            ddist, _sdist, _pdist, value, dlog = agent._dists(
                obs_t.unsqueeze(0), dm.unsqueeze(0), pm.unsqueeze(0))
            probs = ddist.probs[0].tolist()
            a_dir, a_size, a_ptr, _v = agent.act_deterministic(obs_t, dm, pm)
        a_dir = int(a_dir[0]); a_size_f = float(a_size[0]); a_ptr_i = int(a_ptr[0])
        legal = [i for i in range(4) if float(dm[i]) > -1e8]

        ts = str(test_index[env.t]) if env.t < len(test_index) else ""
        date_id = test.dates[env.t] if test.dates is not None else 0
        prev_close = float(env.data[sym].close[env.t - 1]) if env.t > 0 else float(env.data[sym].close[env.t])
        cur_close = float(env.data[sym].close[env.t])
        day_rets.append((cur_close - prev_close) / max(prev_close, 1e-9))

        obs2, reward, done, info = env.step((a_dir, a_size_f, a_ptr_i))

        # Risk context: the exact keys barbershop.adapter reads off a real run.
        # dd_buffer is the FRACTION of the trailing-DD allowance still remaining (1.0 =
        # full cushion, 0 = at the wall) — NOT remaining$/account, which would cap at the
        # ~4% band and always read as "warning". Match the dashboard's 0..1 semantics.
        acct = env.account
        dd_band = (challenge.daily_risk_pct / 100.0) * acct.account_size   # the DD allowance ($)
        risk_context = {
            "trailing_buffer": float(min(1.0, max(0.0, acct.remaining_buffer / max(dd_band, 1e-9)))),
            "daily_pnl": float(acct.day_pnl / acct.account_size * 100.0),
            "position_open": bool(env._n_open(sym) > 0),
            "position_dir": ("LONG" if env._position(sym) > 0 else
                             "SHORT" if env._position(sym) < 0 else "NONE"),
            "open_upnl": float(env._total_unrealized()),
            "breached": bool(acct.breached),
            "equity": float(acct.equity),
        }
        packet = StepPacket(
            run_id=a.run_id, seed=0, window_id="w0", episode_id=episode_id,
            timestep=step_in_day, symbol=sym, timestamp=ts, bar_index=int(env.t),
            observation=[float(x) for x in obs], law_states=[float(x) for x in env._law_states(sym)],
            enforcement_mode="live", legal_actions=legal,
            pre_mask_logits=[float(x) for x in dlog[0].tolist()],
            post_mask_logits=[float(x) for x in (dlog[0] + dm).tolist()],
            action_probs=[float(p) for p in probs], chosen_action=a_dir,
            pointer_output=(a_ptr_i if a_dir == 3 else None),
            raw_size=a_size_f, feasible_size=float(info.get("lots", 0.0)),
            value=float(value[0]), hidden_summary=[],
            reward_decomposition={"total": float(reward)}, quad_signals={},
            risk_context=risk_context, outcome={"realized": float(info.get("realized", 0.0))})
        log.log_step(packet)
        total_steps += 1
        step_in_day += 1

        # Day rollover (real calendar day change) -> close the day, start the next.
        if not done:
            obs = obs2
            new_date = test.dates[env.t] if test.dates is not None else 0
            if new_date != date_id:
                close_day(episode_id, day_rets)
                episode_id += 1
                day_rets = []
                step_in_day = 0
                if episode_id >= a.days:                    # recorded enough real days
                    break

    close_day(episode_id, day_rets)                         # final (partial) day
    path = log.flush()
    size_mb = path.stat().st_size / 1e6
    print(f"[emit] {total_steps:,} real steps across {episode_id + 1} day(s) -> {path}  ({size_mb:.1f} MB)")
    print(f"[done] {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
