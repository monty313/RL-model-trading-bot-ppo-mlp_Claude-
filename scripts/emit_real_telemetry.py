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
from quantra.market_pipeline.feature_builder.schema import STATE_DIM, FEATURE_NAMES  # noqa: E402
from quantra.learning_system.trainer.gae import compute_gae                    # noqa: E402
from quantra.diagnostics.telemetry_logger.logger import TelemetryLogger, StepPacket  # noqa: E402


_ACT = {0: "HOLD", 1: "OPEN_LONG", 2: "OPEN_SHORT", 3: "CLOSE"}   # engine int -> action str


def compute_attribution(agent, obs, chosen_dir: int, top_k: int = 20):
    """Input x gradient attribution of the chosen direction logit w.r.t. the observation.

    Reads: the agent net + the obs that produced the decision. Returns (toward, away)
    dicts {feature_name: magnitude} for the top_k features by |attribution|, split by
    sign (toward = pushed the chosen logit up, away = pushed it down). This is a
    legitimate, fast saliency — NOT Shapley values — and is labelled as such in the UI.
    """
    obs_t = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
    dlog, _slog, _plog, _value = agent.net(obs_t.unsqueeze(0))
    agent.net.zero_grad(set_to_none=True)
    dlog[0, chosen_dir].backward()                          # d(chosen logit)/d(obs)
    attr = (obs_t.grad.detach().numpy() * np.asarray(obs, dtype=float))   # input x gradient
    order = np.argsort(-np.abs(attr))[:top_k]               # strongest contributors first
    names = FEATURE_NAMES
    toward = {names[i]: float(attr[i]) for i in order if attr[i] > 0 and i < len(names)}
    away = {names[i]: float(-attr[i]) for i in order if attr[i] < 0 and i < len(names)}
    return toward, away


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
    day_steps: list = []        # (packet, reward, value) buffered to compute GAE at day close
    attr_records: list = []     # input-gradient attribution for trade steps (autopsy RIGHT col)
    done = False
    step_in_day = 0
    total_steps = 0

    def flush_day(eid: int, steps: list, rets: list):
        """Compute REAL per-day GAE advantage, attach it to each step, then log the day.

        Advantage is a rollout-level quantity (locked gamma=0.997/lambda=0.97), so we
        buffer the day's steps and compute it here, treating the day end as a terminal
        segment. Each step's advantage goes into its outcome dict; the adapter reads it.
        """
        if steps:
            rewards = torch.tensor([s[1] for s in steps], dtype=torch.float32)
            values = torch.tensor([s[2] for s in steps], dtype=torch.float32)
            dones = torch.zeros(len(steps), dtype=torch.float32); dones[-1] = 1.0  # day end terminal
            adv, _ret = compute_gae(rewards, values, dones, 0.0)   # locked gamma/lambda
            for (pkt, _r, _v), a_val in zip(steps, adv.tolist()):
                pkt.outcome["advantage"] = float(a_val)            # REAL GAE advantage
                log.log_step(pkt)
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
        day_steps.append((packet, float(reward), float(value[0])))   # advantage filled at day close
        # Real attribution for trade steps (entry/exit) -> the autopsy RIGHT column.
        if a_dir in (1, 2, 3):
            toward, away = compute_attribution(agent, obs, a_dir)
            attr_records.append({"timestamp": ts, "day_id": episode_id + 1, "step": step_in_day,
                                 "chosen_action": _ACT[a_dir], "shap_toward": toward, "shap_away": away})
        total_steps += 1
        step_in_day += 1

        # Day rollover (real calendar day change) -> close the day, start the next. The day-id basis
        # is test.dates, built by prepare_symbol_data at 00:00 CE(S)T (cfg.CHALLENGE_TZ, Europe/Prague)
        # [2026-06-19], so this rollover fires at the SAME CE(S)T boundary as the env's daily reset.
        if not done:
            obs = obs2
            new_date = test.dates[env.t] if test.dates is not None else 0
            if new_date != date_id:
                flush_day(episode_id, day_steps, day_rets)
                episode_id += 1
                day_steps = []
                day_rets = []
                step_in_day = 0
                if episode_id >= a.days:                    # recorded enough real days
                    break

    flush_day(episode_id, day_steps, day_rets)              # final (partial) day
    path = log.flush()
    size_mb = path.stat().st_size / 1e6
    print(f"[emit] {total_steps:,} real steps across {episode_id + 1} day(s) -> {path}  ({size_mb:.1f} MB)")
    # Sidecar: the input-gradient attribution for trade steps (autopsy RIGHT column).
    if attr_records:
        import json
        attr_path = path.with_name(path.stem + "_attribution.jsonl")
        with open(attr_path, "w", encoding="utf-8") as fh:
            for rec in attr_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"[attribution] {len(attr_records)} trade-step attributions -> {attr_path}")
    print(f"[done] {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
