"""Real-data backtest on REAL MT5 bars -> MT5-Strategy-Tester-style report.

HONEST by construction: it loads real bars, (optionally) trains the PPO brain on a
train slice, then runs the DETERMINISTIC policy on a held-out test slice and reports the
REAL metrics of whatever was trained — no synthetic data, no cherry-picking.

Usage:
    python scripts/real_backtest.py --symbol EURUSD --path data/raw/EURUSD_recent.csv \
        --updates 40 --target 2.5 --risk 4.0
    # ...or omit --path to let load_symbol resolve the bars itself (Parquet cache ->
    # Drive mount -> gdown auto-download by registered Drive file ID):
    python scripts/real_backtest.py --symbol EURUSD --updates 40

Rulebook (for the Risk Doctor / any LLM reading this): docs/MLP_INTERPRETABILITY_LAYER.md.
This script only MEASURES a policy on real held-out bars; it is not in the
State -> Law -> Hidden -> Heads -> Risk -> Reward -> Outcome causal chain.

Update Log (IRAC):
- [2026-06-18] Fix: --path now defaults to None and is passed through as
  ``Path(a.path) if a.path else None``.
  - I: With --path defaulting to a hardcoded CSV ("data/raw/EURUSD_recent.csv")
       and always wrapped in Path(...), omitting --path on a clean checkout (no
       local CSV) raised FileNotFoundError instead of using load_symbol's
       Parquet-cache / Drive / gdown auto-download fallback (loader.py treats
       path=None as "resolve the bars for me"). That blocks the very first real
       Barbershop/backtest run, which is the gateway to a real FTMO pass.
  - R: load_symbol(symbol, path: Optional[Path] = None) only triggers the
       cache/Drive/gdown resolution chain when path IS None. Operator brief
       Section 9 mandates the guard so omitting --path triggers the fallback.
  - A: Set default=None on --path and pass ``Path(a.path) if a.path else None``
       so an explicit path still works and an omitted path triggers the loader's
       auto-download.
  - C: A clean checkout can now run the real backtest with no local data file,
       which is the first step toward training a policy that actually passes the
       FTMO challenge on real bars.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# HONEST-OUTPUT FIX: the report contains em-dashes / box glyphs. On Windows the console
# defaults to cp1252 and mangles them (— -> "ﾄ"), which makes an "honest report" look
# corrupt. Force UTF-8 on stdout/stderr so the report renders exactly as written.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # py3.7+; no-op if already utf-8
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantra.runtime import config as cfg                                      # noqa: E402
from quantra.market_pipeline.data_loader import load_symbol                    # noqa: E402
from quantra.env.trading_env import TradingEnv, prepare_symbol_data, SymbolData  # noqa: E402
from quantra.learning_system.ppo_agent.agent import PPOAgent                   # noqa: E402
from quantra.learning_system.trainer.trainer import Trainer, TrainConfig       # noqa: E402
from quantra.market_pipeline.law_mask_engine.engine import build_pointer_mask  # noqa: E402


def compute_stats(trades, equity, account_size):
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t < 0]
    gp = float(sum(wins)); gl = float(sum(losses)); closed_net = gp + gl
    pf = (gp / abs(gl)) if gl < 0 else (float("inf") if gp > 0 else 0.0)
    n = len(trades)
    win = 100.0 * len(wins) / n if n else 0.0
    ep = closed_net / n if n else 0.0
    eq = np.asarray(equity, float) if len(equity) else np.asarray([account_size], float)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    final = float(eq[-1])
    # GROUND TRUTH net = total account change (incl. any breach force-flatten loss), NOT just
    # the agent's discretionary closes. forced = the gap between the two (breach/flatten P&L).
    net_equity = final - account_size
    return dict(net=net_equity, closed_net=closed_net, forced=net_equity - closed_net,
                gp=gp, gl=gl, pf=pf, n=n, win=win, ep=ep,
                maxdd=float(dd.max()), maxdd_pct=float((dd / np.maximum(peak, 1e-9)).max() * 100.0),
                final=final)


def backtest(symbol, env, agent):
    """Deterministic backtest: returns (closed-trade PnLs, equity curve, breached, daily)."""
    obs = env.reset()
    trades, equity = [], []
    pass_days, breach_days, day_ids = 0, 0, set()
    done = False
    while not done:
        dm = torch.as_tensor(env.direction_mask(symbol), dtype=torch.float32)
        occ = [s.occupied for s in env.slots[symbol]]
        pm = torch.as_tensor(build_pointer_mask(occ), dtype=torch.float32)
        a_dir, a_size, a_ptr, _ = agent.act_deterministic(
            torch.as_tensor(obs, dtype=torch.float32), dm, pm)
        obs2, _r, done, info = env.step((int(a_dir[0]), float(a_size[0]), int(a_ptr[0])))
        if info.get("executed") == "CLOSE":
            trades.append(float(info.get("realized", 0.0)))
        equity.append(env.account.equity)
        if env.account.target_hit:
            pass_days = max(pass_days, 1)             # at least one day hit target before stop
        if not done:
            obs = obs2
    breached = bool(env.account.breached)
    return trades, equity, breached, pass_days


def report_str(symbol, bars, challenge, s, breached, span):
    survived = "SURVIVED (no breach)" if not breached else "BREACHED the 4% wall — challenge FAILED"
    net_pct = s["net"] / challenge.ftmo_account_size * 100.0
    return "\n".join([
        f"================ QUANTRA BACKTEST — {symbol} (REAL MT5 bars) ================",
        f" period (test slice)   {span}",
        f" bars tested           {bars:>16,}",
        f" config                target {challenge.daily_target_pct}%  trailing {challenge.daily_risk_pct}%  "
        f"leverage 1:{int(challenge.leverage)}",
        f" account size          {challenge.ftmo_account_size:>16,.2f}",
        f" Total Net Profit      {s['net']:>16,.2f}   ({net_pct:+.2f}% of account)  <- GROUND TRUTH",
        f"   closed-trade P&L    {s['closed_net']:>16,.2f}   (agent's own CLOSE decisions)",
        f"   breach/flatten P&L  {s['forced']:>16,.2f}   (forced liquidation on breach)",
        f" Gross Profit          {s['gp']:>16,.2f}",
        f" Gross Loss            {s['gl']:>16,.2f}",
        f" Profit Factor         {s['pf']:>16.2f}   (closed trades only)",
        f" Total Trades          {s['n']:>16,}",
        f" Win rate %            {s['win']:>16.2f}",
        f" Expected Payoff       {s['ep']:>16,.2f}",
        f" Max Equity Drawdown   {s['maxdd']:>16,.2f}   ({s['maxdd_pct']:.2f}%)",
        f" Final Equity          {s['final']:>16,.2f}",
        f" Outcome               {survived}",
        "=" * 74,
    ])


def render_png(equity, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(equity, color="#1D9E75", lw=1.0)
    ax.axhline(equity[0] if len(equity) else 0, color="#888", lw=0.8, ls="--")
    ax.set_title(title); ax.set_xlabel("test bar"); ax.set_ylabel("equity")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    # default=None so an OMITTED --path triggers load_symbol's cache/Drive/gdown
    # fallback instead of FileNotFoundError on a clean checkout. See IRAC in the
    # module docstring (operator brief Section 9).
    ap.add_argument("--path", default=None,
                    help="explicit CSV path; omit to use the Parquet cache / Drive "
                         "mount / gdown auto-download fallback in load_symbol")
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--target", type=float, default=2.5)
    ap.add_argument("--risk", type=float, default=4.0)
    a = ap.parse_args()

    t0 = time.time()
    df, rep = load_symbol(a.symbol, path=Path(a.path) if a.path else None)
    print(f"[load] {len(df):,} bars  {df.index.min()} -> {df.index.max()}  "
          f"spread={rep.had_spread}  ({time.time()-t0:.1f}s)")
    sd = prepare_symbol_data(df, symbol=a.symbol)
    T, vf = len(sd.close), sd.valid_from
    print(f"[features] T={T:,}  valid_from(warmup)={vf:,}  usable={T-vf:,}")
    split = vf + int((T - vf) * a.train_frac)

    def sl(arr, lo, hi):
        return None if arr is None else arr[lo:hi]
    train = SymbolData(sd.matrix[:split], sd.close[:split], sd.atr[:split],
                       sd.spread[:split], valid_from=vf, dates=sl(sd.dates, 0, split))
    test = SymbolData(sd.matrix[split:], sd.close[split:], sd.atr[split:],
                      sd.spread[split:], valid_from=0, dates=sl(sd.dates, split, T))
    challenge = cfg.make_challenge(daily_target_pct=a.target, daily_risk_pct=a.risk)
    agent = PPOAgent()

    env = TradingEnv({a.symbol: test}, challenge=challenge)
    bt = backtest(a.symbol, env, agent)
    s0 = compute_stats(bt[0], bt[1], challenge.ftmo_account_size)
    print(f"[untrained baseline] trades={s0['n']} net={s0['net']:.2f} maxDD%={s0['maxdd_pct']:.2f}")

    if a.updates > 0:
        tenv = TradingEnv({a.symbol: train}, challenge=challenge)
        tr = Trainer(tenv, agent=agent, train_cfg=TrainConfig(seed=0))
        t1 = time.time()
        hist = tr.train(a.updates)
        print(f"[train] {a.updates} updates in {time.time()-t1:.1f}s  "
              f"last: loss={hist[-1].get('loss',0):.4f} value_loss={hist[-1].get('value_loss',0):.4f} "
              f"miss_rate={hist[-1].get('miss_rate',0):.2f}")

    env = TradingEnv({a.symbol: test}, challenge=challenge)
    trades, equity, breached, _ = backtest(a.symbol, env, agent)
    s = compute_stats(trades, equity, challenge.ftmo_account_size)
    span = f"{df.index[split]} -> {df.index[-1]}"
    print(report_str(a.symbol, len(test.close), challenge, s, breached, span))
    png = render_png(equity, f"data/backtest_{a.symbol}_equity.png",
                     f"Quantra {a.symbol} — deterministic backtest on real bars")
    print(f"[equity curve] {png}")


if __name__ == "__main__":
    main()
