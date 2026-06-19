"""ws_broadcaster — stream Quantra telemetry to the JARVIS HUD over a WebSocket.

WHAT THIS DOES
--------------
Bridges quantra.diagnostics.telemetry_logger (the per-step JSONL the trainer writes) to
``jarvis_hud.html`` so every training step lights up the HUD live. It tails the newest
``artifacts/telemetry/<run_id>.jsonl``, maps each StepPacket to the HUD event schema, and broadcasts
it to every connected browser at ws://localhost:8765.

EVENT SCHEMA (the CONTRACT shared with jarvis_hud.html applyEvent()):
    {"t":"step","step":int,"module":str,"reward":float,"daily_pnl_pct":float|None,
     "drawdown_pct":float,"win_rate":float|None,"ftmo":"PASSING|BUILDING|AT RISK|BREACH",
     "action":"BUY EURUSD","symbol":str}

COUPLING (both directions):
  -> quantra/diagnostics/telemetry_logger/logger.py: reads StepPacket keys by name
     (timestep, symbol, chosen_action, reward_decomposition, risk_context["trailing_dd"], and the
     OPTIONAL "active_module" the trainer can add for node-pulsing). Renaming those breaks the map.
  -> jarvis_hud.html: emits exactly the event schema above; changing a key here means changing it there.
  -> runtime/config.py: defaults the telemetry dir to cfg.TELEMETRY_DIR.

Core mapping (packet_to_event / iter_events_from_lines) is PURE + dependency-free (tested). Only the
live server needs `websockets` (pip install websockets); run `python -m barbershop.ws_broadcaster --demo`
to stream synthetic events with no trainer running.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

HOST, PORT = "localhost", 8765
_ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE"}


def _ftmo(dd_pct: float, pnl_pct: Optional[float]) -> str:
    if dd_pct >= 4.0:
        return "BREACH"
    if pnl_pct is not None and pnl_pct >= 2.5:
        return "PASSING"
    if dd_pct >= 3.0:
        return "AT RISK"
    return "BUILDING"


def packet_to_event(p: Dict, *, step: Optional[int] = None) -> Dict:
    """Map ONE telemetry StepPacket dict -> a HUD event. Pure (no I/O); the single place the two
    schemas meet. Missing fields (daily_pnl_pct / win_rate live in the env, not the StepPacket) pass
    through as None and the HUD simply holds the prior value."""
    rd = p.get("reward_decomposition") or {}
    reward = rd.get("total", rd.get("L0", 0.0))
    rc = p.get("risk_context") or {}
    dd = abs(float(rc.get("trailing_dd", rc.get("drawdown_pct", 0.0))))
    pnl = p.get("daily_pnl_pct")
    sym = p.get("symbol", "?")
    action = _ACTIONS.get(p.get("chosen_action"), "HOLD")
    module = p.get("active_module") or "trainer"   # trainer can add active_module to pulse a node
    return {"t": "step", "step": int(step if step is not None else p.get("timestep", 0)),
            "module": module, "reward": round(float(reward), 3), "drawdown_pct": round(dd, 2),
            "daily_pnl_pct": (round(float(pnl), 3) if pnl is not None else None),
            "win_rate": p.get("win_rate"), "ftmo": _ftmo(dd, pnl),
            "action": f"{action} {sym}", "symbol": sym}


def iter_events_from_lines(lines: Iterable[str]) -> Iterator[Dict]:
    """Turn raw telemetry JSONL lines into HUD events (skips headers + non-step rows). Pure/offline —
    used by the live tailer and by the tests."""
    n = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("kind") == "header":
            continue
        n += 1
        yield packet_to_event(rec, step=rec.get("timestep", n))


def latest_telemetry(telemetry_dir: Optional[Path] = None) -> Optional[Path]:
    """Newest <run_id>.jsonl under the telemetry dir (the run the HUD should follow)."""
    if telemetry_dir is None:
        from quantra.runtime import config as cfg
        telemetry_dir = cfg.TELEMETRY_DIR
    files = sorted(Path(telemetry_dir).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def tail_events(path: Path, poll: float = 0.4) -> Iterator[Dict]:
    """Follow a telemetry file (tail -f) yielding HUD events as new lines land. Blocks forever."""
    path = Path(path)
    with path.open("r") as f:
        for ev in iter_events_from_lines(f):    # replay what's already there
            yield ev
        while True:                              # then stream new lines
            line = f.readline()
            if line:
                yield from iter_events_from_lines([line])
            else:
                time.sleep(poll)


def demo_events(n: int = 10_000, delay: float = 0.95) -> Iterator[Dict]:
    """Synthetic events so the live path can be exercised with no trainer running (--demo)."""
    import random
    rng = random.Random(7)
    mods = ["data_loader", "feature_builder", "law_mask_engine", "trading_env", "ppo_agent",
            "rollout_buffer", "reward_engine", "curriculum_manager", "trainer", "telemetry_logger",
            "mlp_interpreter", "llm_risk_doctor", "live_bridge"]
    syms = ["EURUSD", "GBPUSD", "XAUUSD", "US30"]
    dd, pnl, win, step = 0.0, 0.0, 52.0, 4800
    for _ in range(n):
        step += 1
        dd = max(0.0, min(4.3, dd + (rng.random() - 0.55) * 0.22))
        pnl = max(-3.0, min(4.2, pnl + (rng.random() - 0.45) * 0.18))
        win = max(38.0, min(72.0, win + (rng.random() - 0.5) * 1.4))
        reward = round((rng.random() - 0.42) * 0.9, 2)
        sym = rng.choice(syms)
        yield {"t": "step", "step": step, "module": rng.choice(mods), "reward": reward,
               "drawdown_pct": round(dd, 2), "daily_pnl_pct": round(pnl, 2), "win_rate": round(win, 1),
               "ftmo": _ftmo(dd, pnl), "action": f"{rng.choice(['BUY','SELL','CLOSE','HOLD'])} {sym}",
               "symbol": sym}
        time.sleep(delay)


def serve(events: Iterator[Dict], host: str = HOST, port: int = PORT) -> None:
    """Broadcast an event stream to every connected HUD over a WebSocket. Needs `websockets`."""
    try:
        import asyncio
        import websockets
    except ImportError:
        raise SystemExit("ws_broadcaster needs `websockets`:  pip install websockets")

    clients: set = set()

    async def handler(ws):
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)

    async def pump():
        loop = asyncio.get_event_loop()
        while True:
            ev = await loop.run_in_executor(None, lambda: next(events, None))
            if ev is None:
                break
            if clients:
                msg = json.dumps(ev)
                await asyncio.gather(*[c.send(msg) for c in list(clients)], return_exceptions=True)

    async def run():
        async with websockets.serve(handler, host, port):
            print(f"ws_broadcaster: ws://{host}:{port} — open jarvis_hud.html (LIVE pill turns green)")
            await pump()

    asyncio.run(run())


def main() -> None:
    import sys
    if "--demo" in sys.argv:
        print("ws_broadcaster: DEMO stream (no trainer needed).")
        serve(demo_events())
        return
    path = latest_telemetry()
    if path is None:
        raise SystemExit("No telemetry under artifacts/telemetry/. Run a trainer, or use --demo.")
    print(f"ws_broadcaster: following {path}")
    serve(tail_events(path))


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# STANDING RULE [2026-06-19, operator]: every edit appends a DATED IRAC entry + documents the
# cross-file COUPLING in both directions (what breaks where, and when).
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-19] C22 — telemetry -> WebSocket bridge for the JARVIS HUD.
#   I: jarvis_hud.html could only show simulated data; there was no path from real training telemetry
#      to the live HUD.
#   R: Operator brief ("make it actually live" — telemetry_logger -> WebSocket -> JARVIS dashboard).
#   A: packet_to_event() (pure StepPacket->HUD-event map) + iter_events_from_lines/tail_events (follow
#      the newest artifacts/telemetry run) + a websockets server + a --demo synthetic stream. Core map
#      is dependency-free + tested; only the live server needs `websockets`.
#   C: Every training step can light up the HUD in real time, so the operator watches the system learn
#      to pass FTMO live — turning telemetry into an at-a-glance, trustable signal.
