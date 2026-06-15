"""LiveSession — the live 1m-bar loop that drives the policy on a real feed. 🔴

WHAT THIS MODULE DOES
---------------------
The missing live loop (the M14 CLI said "no feed wired"; this wires it). Each time a
new 1m bar CLOSES it: pulls the trailing bar window per symbol, rebuilds the 179-dim
observation (M2 features + M3 law states + a LIVE portfolio/account mirror), computes
the masks, runs the policy DETERMINISTICALLY, sizes via the RiskManager, and executes
via the ExecutionAdapter — honoring the manual halt + 4% breach auto-flat. Two feeds:
  * MT5BarFeed     — pulls closed bars from a live MT5 terminal (operator machine).
  * ReplayBarFeed  — replays a precomputed frame (the offline test substrate).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
A trained brain is worthless without a faithful live loop. This rebuilds, bar-by-bar,
the SAME observation/masks/slots the env used in training, so the learned pass-behaviour
actually reproduces live — behind the hard kill switches that keep a bad session
non-fatal. Bars are read from index 1 (the last CLOSED bar), never the forming bar, so
there is no live lookahead.

🔴 LOCKED: deterministic inference, RiskManager clip, kill switches, no-lookahead feed,
diagnostics isolation (no telemetry import here).

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. OFF-LIMITS at runtime — you read only
checkpointed telemetry, never the live session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# COUPLING -> quantra/env/trading_env.py: reuses the Slot dataclass; its fields
# (occupied, direction, entry_price, lots, risk_per_lot, age, mfe, mae) + upnl() are read below to mirror env.
from quantra.env.trading_env import Slot
# COUPLING [C5] -> quantra/ftmo_passing/challenge_state.py: uses ChallengeState.account_block()
# (its block ORDER must match schema account names, C1) + realize/mark_to_market/equity/wall_equity.
from quantra.ftmo_passing.challenge_state import ChallengeState
from quantra.learning_system.ppo_agent.agent import PPOAgent
from quantra.live_bridge.execution_adapter import ExecutionAdapter
from quantra.live_bridge.manual_halt import ManualHalt
from quantra.locked_core.risk_manager.risk import RiskManager
# COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py: rebuilds the obs from
# STATE_DIM/assemble_state; block ORDER here must equal schema/env, else live obs != trained obs.
from quantra.market_pipeline.feature_builder import (
    PRECOMPUTED_NAMES,
    STATE_DIM,
    assemble_state,
    build_market_matrix,
)
from quantra.market_pipeline.feature_builder import indicators as _ind
# COUPLING [C3] -> quantra/market_pipeline/feature_builder/schema.py: N_SLOTS must equal
# execution_adapter.N_SLOTS + env slot count + ppo pointer width (trade block 7*5=35).
from quantra.market_pipeline.feature_builder.schema import N_SLOTS
from quantra.locked_core.laws.laws import compute_law_states
# COUPLING [C2] -> quantra/market_pipeline/law_mask_engine/engine.py: direction ints + mask
# builders; the {OPEN_LONG=1,OPEN_SHORT=2,CLOSE=3} mapping must match agent head + env + live_runner.
from quantra.market_pipeline.law_mask_engine.engine import (
    CLOSE,
    OPEN_LONG,
    OPEN_SHORT,
    build_direction_mask,
    build_pointer_mask,
)
from quantra.runtime import config as cfg


# --------------------------------------------------------------------------- feeds
class BarFeed:
    def latest(self, symbol: str) -> pd.DataFrame: raise NotImplementedError
    def advance(self) -> bool: return False


@dataclass
class ReplayBarFeed(BarFeed):
    """Replays precomputed frames bar-by-bar (offline test substrate)."""

    frames: Dict[str, pd.DataFrame]
    cursor: int = 0
    window: int = 6600          # trailing bars handed to the feature builder

    def __post_init__(self):
        self._len = min(len(f) for f in self.frames.values())
        if self.cursor == 0:
            self.cursor = min(self._len - 1, self.window)

    def latest(self, symbol: str) -> pd.DataFrame:
        lo = max(0, self.cursor - self.window)
        return self.frames[symbol].iloc[lo:self.cursor + 1]

    def advance(self) -> bool:
        if self.cursor + 1 >= self._len:
            return False
        self.cursor += 1
        return True


@dataclass
class MT5BarFeed(BarFeed):
    """Pulls the last CLOSED bars from a live MT5 terminal via the adapter."""

    adapter: object
    window: int = 600

    def latest(self, symbol: str) -> pd.DataFrame:  # pragma: no cover - live terminal
        return self.adapter.recent_bars(symbol, self.window)


# ----------------------------------------------------------------- live portfolio
@dataclass
class LivePortfolio:
    """Live mirror of the per-symbol 5 slots + shared account, for obs assembly."""

    symbols: List[str]
    challenge: cfg.ChallengeConfig = field(default_factory=cfg.ChallengeConfig)
    max_lot: float = 50.0   # mirrors RiskConfig.max_lot so port_net_size obs matches the env [fix]

    def __post_init__(self):
        self.account = ChallengeState(self.challenge.ftmo_account_size, self.challenge)
        self.slots: Dict[str, List[Slot]] = {s: [Slot() for _ in range(N_SLOTS)] for s in self.symbols}
        self.price: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self.atr: Dict[str, float] = {s: 1e-9 for s in self.symbols}

    # COUPLING [C5] -> quantra/runtime/config.py: CONTRACT_SIZE is keyed by config.SYMBOLS
    # (same per-symbol dict env/cost_layer use); a symbol missing here silently falls back to 1.0.
    def _contract(self, sym: str) -> float:
        return cfg.CONTRACT_SIZE.get(sym, 1.0)

    def n_open(self, sym: str) -> int:
        return sum(1 for sl in self.slots[sym] if sl.occupied)

    def position(self, sym: str) -> int:
        dirs = {sl.direction for sl in self.slots[sym] if sl.occupied}
        return 1 if dirs == {1} else -1 if dirs == {-1} else 0

    def committed_risk(self) -> float:
        return sum(sl.lots * sl.risk_per_lot
                   for s in self.symbols for sl in self.slots[s] if sl.occupied)

    def used_margin(self) -> float:
        """Broker margin tied up by all open slots (notional / leverage). Mirrors the
        env's _used_margin so LIVE sizing matches training [2026-06-15]. COUPLING ->
        quantra/locked_core/risk_manager/risk.py size(leverage=, free_margin=)."""
        lev = max(1.0, self.challenge.leverage)
        return sum(sl.lots * self._contract(s) * self.price[s] / lev
                   for s in self.symbols for sl in self.slots[s] if sl.occupied)

    def mark(self, sym: str, price: float, atr: float) -> None:
        self.price[sym] = price
        self.atr[sym] = max(atr, 1e-9)
        for sl in self.slots[sym]:
            if sl.occupied:
                sl.age += 1
                cur = sl.upnl(price, self._contract(sym))
                sl.mfe = max(sl.mfe, cur); sl.mae = min(sl.mae, cur)
        tot = sum(sl.upnl(self.price[s], self._contract(s))
                  for s in self.symbols for sl in self.slots[s])
        self.account.mark_to_market(tot)

    def open(self, sym: str, slot_idx: int, direction: int, entry: float, lots: float, rpl: float) -> None:
        sl = self.slots[sym][slot_idx]
        sl.occupied = True; sl.direction = direction; sl.entry_price = entry
        sl.lots = lots; sl.risk_per_lot = rpl; sl.age = 0; sl.mfe = sl.mae = 0.0

    def close(self, sym: str, slot_idx: int) -> None:
        sl = self.slots[sym][slot_idx]
        if sl.occupied:
            self.account.realize(sl.upnl(self.price[sym], self._contract(sym)))
        self.slots[sym][slot_idx] = Slot()

    # obs blocks (same layout as the env)
    # COUPLING [C1][C3] -> quantra/market_pipeline/feature_builder/schema.py (trade block 7*N_SLOTS=35)
    # + quantra/env/trading_env.py: the per-slot 7-field order below (direction, upnl, age, entry-dist,
    # mfe, mae, occupied) must match schema's trade-block names + the env's trade_block exactly.
    def trade_block(self, sym: str) -> np.ndarray:
        acct, c, atr, con = self.account.account_size, self.price[sym], self.atr[sym], self._contract(sym)
        out = np.zeros(N_SLOTS * 7, dtype=np.float32)
        for i, sl in enumerate(self.slots[sym]):
            if not sl.occupied:
                continue
            b = i * 7
            out[b:b + 7] = [sl.direction, sl.upnl(c, con) / acct, min(sl.age / 1000.0, 10.0),
                            (c - sl.entry_price) / atr, sl.mfe / acct, sl.mae / acct, 1.0]
        return out

    # COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py + quantra/env/trading_env.py:
    # this 3-element portfolio block order must match schema's portfolio-block names + the env's.
    def portfolio_block(self, sym: str) -> np.ndarray:
        c, con = self.price[sym], self._contract(sym)
        opn = [sl for sl in self.slots[sym] if sl.occupied]
        return np.array([sum(s.direction for s in opn) / N_SLOTS,
                         sum(s.lots for s in opn) / max(self.max_lot, 1e-9),  # == env's /max_lot [fix]
                         sum(s.upnl(c, con) for s in opn) / self.account.account_size], dtype=np.float32)


# ------------------------------------------------------------------- live session
class LiveSession:
    """The live loop: feed -> obs -> deterministic policy -> sized order -> broker."""

    def __init__(self, agent: PPOAgent, feed: BarFeed, execution: ExecutionAdapter,
                 risk: RiskManager, symbols: List[str], halt: Optional[ManualHalt] = None,
                 challenge: Optional[cfg.ChallengeConfig] = None,
                 point_sizes: Optional[Dict[str, float]] = None,
                 training_wheels: Optional[bool] = None):
        self.agent = agent
        # Training wheels [operator 2026-06-15]: same counter-trend OPEN blocks as the env,
        # so the discipline learned in training transfers to live. Defaults to the global
        # config flag. COUPLING [C9] -> reads the precomputed tw_cci_block/tw_bb_block flags.
        self.training_wheels = cfg.TRAINING_WHEELS if training_wheels is None else bool(training_wheels)
        self._wheel_idx = (PRECOMPUTED_NAMES.index("tw_cci_block"),
                           PRECOMPUTED_NAMES.index("tw_bb_block"))
        self.feed = feed
        self.exec = execution
        self.risk = risk
        self.symbols = symbols
        self.halt = halt or ManualHalt()
        self.portfolio = LivePortfolio(symbols, challenge or cfg.ChallengeConfig(),
                                       max_lot=risk.cfg.max_lot)
        # COUPLING [C5] -> quantra/runtime/config.py: POINT_SIZE/DEFAULT_POINT_SIZE per-symbol dict
        # (same one build_market_matrix expects); these point sizes must match training to reproduce features.
        self.point_sizes = point_sizes or {s: cfg.POINT_SIZE.get(s, cfg.DEFAULT_POINT_SIZE) for s in symbols}

    def _observe(self, sym: str):
        """Build (obs, price, atr) for the latest closed bar of ``sym``."""
        bars = self.feed.latest(sym)
        mm = build_market_matrix(bars, point_size=self.point_sizes[sym])
        market_row = mm.matrix[-1]
        price = float(bars["close"].iloc[-1])
        # COUPLING -> quantra/market_pipeline/feature_builder/indicators.py: ATR_PERIOD is a locked param;
        # must equal the period used in training's feature pipeline or the entry-dist feature drifts.
        atr = float(_ind.atr(bars["high"], bars["low"], bars["close"], _ind.ATR_PERIOD).iloc[-1] or 1e-9)
        self.portfolio.mark(sym, price, atr)
        law = compute_law_states(market_row)
        # COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py + env/trading_env.py:
        # assemble_state's block order (market, law, trade, portfolio, account) + account_block() names
        # must match schema; mismatch => live obs misaligned vs the trained STATE_DIM vector.
        obs = assemble_state(market_row, law_flags=law,
                             trade=self.portfolio.trade_block(sym),
                             portfolio=self.portfolio.portfolio_block(sym),
                             account=self.portfolio.account.account_block())
        wheel = np.array([market_row[self._wheel_idx[0]], market_row[self._wheel_idx[1]]],
                         dtype=np.float32)
        return obs, price, atr, law, wheel

    def step_symbol(self, sym: str) -> dict:
        """Process one symbol at the latest bar: decide + execute deterministically."""
        if self.halt.is_halted:
            return {"action": "HALTED", "symbol": sym}
        obs, price, atr, law, wheel = self._observe(sym)
        dm = build_direction_mask(law, self.portfolio.position(sym), self.portfolio.n_open(sym),
                                  training_wheels=self.training_wheels,
                                  wheel_states=wheel if self.training_wheels else None)
        pm = build_pointer_mask([sl.occupied for sl in self.portfolio.slots[sym]])
        # COUPLING -> quantra/learning_system/ppo_agent/agent.py: unpacks act_deterministic's
        # 4-tuple (a_dir, a_size, a_ptr, value) positionally (same as live_runner.py); reorder there -> wrong here.
        a_dir, a_size, a_ptr, _ = self.agent.act_deterministic(
            torch.as_tensor(obs, dtype=torch.float32),
            torch.as_tensor(dm, dtype=torch.float32),
            torch.as_tensor(pm, dtype=torch.float32))
        a_dir, a_size, a_ptr = int(a_dir[0]), float(a_size[0]), int(a_ptr[0])

        if a_dir in (OPEN_LONG, OPEN_SHORT):
            # COUPLING [C5] -> quantra/ftmo_passing/challenge_state.py: reads ChallengeState.remaining_buffer.
            budget = self.portfolio.account.remaining_buffer - self.portfolio.committed_risk()
            # COUPLING -> quantra/locked_core/risk_manager/risk.py: reads SizeResult .feasible/.reason/.lots/
            # .risk_per_lot (same fields live_runner.py uses); rename any -> break OPEN sizing here.
            # Margin ceiling + mode-aware cap [2026-06-15] — IDENTICAL to the env so the
            # learned sizing reproduces live (ftmo OFF: no per-trade cap; margin binds).
            sr = self.risk.size(
                sym, a_size, atr, budget,
                apply_per_trade_cap=self.portfolio.challenge.ftmo_mode,
                price=price, contract=self.portfolio._contract(sym),
                leverage=self.portfolio.challenge.leverage,
                free_margin=self.portfolio.account.equity - self.portfolio.used_margin(),
            )
            if not sr.feasible:
                return {"action": "OPEN_SKIPPED", "symbol": sym, "reason": sr.reason}
            side = 1 if a_dir == OPEN_LONG else -1
            slot = self.exec.open(sym, side, sr.lots, price)
            if slot is not None:
                self.portfolio.open(sym, slot, side, price, sr.lots, sr.risk_per_lot)
            return {"action": "OPEN", "symbol": sym, "slot": slot, "lots": sr.lots}
        if a_dir == CLOSE:
            ok = self.exec.close(sym, a_ptr, price)
            if ok:
                self.portfolio.close(sym, a_ptr)
            return {"action": "CLOSE", "symbol": sym, "slot": a_ptr, "ok": ok}
        return {"action": "HOLD", "symbol": sym}

    def _check_breach(self, price_map: Dict[str, float]) -> bool:
        # COUPLING [C5] -> quantra/ftmo_passing/challenge_state.py: reads .equity + .wall_equity
        # (the 4% wall anchor); rename either attr there -> breach detection silently breaks.
        acct = self.portfolio.account
        if acct.equity <= acct.wall_equity:
            self.exec.close_all()
            for s in self.symbols:
                for i in range(N_SLOTS):
                    self.portfolio.slots[s][i] = Slot()
            # COUPLING -> quantra/live_bridge/manual_halt.py: latches via ManualHalt._halted directly
            # (same private attr live_runner.breach_autoflat sets); keep that attr name stable.
            self.halt._halted = True
            return True
        return False

    def run_steps(self, n: int) -> List[dict]:
        """Process up to n new bars across all symbols (true-sequential per bar)."""
        infos: List[dict] = []
        for _ in range(n):
            for sym in self.symbols:           # true-sequential within the bar
                infos.append(self.step_symbol(sym))
            if self._check_breach(self.portfolio.price):
                infos.append({"action": "BREACH_AUTOFLAT"})
                break
            if not self.feed.advance():
                break
        return infos


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M14b — wired the live 1m-bar loop (MT5 + replay feeds).
#   I: M14 left "no live feed wired"; a trained brain couldn't actually trade on MT5.
#   R: SOW §2.10/§10 (live determinism + kill switches) + C4 (1m decisions) + no-lookahead.
#   A: ReplayBarFeed/MT5BarFeed; LivePortfolio mirrors slots+account for obs; LiveSession
#      rebuilds the 179-dim obs from the trailing CLOSED-bar window, masks, decides
#      deterministically, sizes vs the live buffer, executes, and breach-auto-flats.
#   C: The learned pass-behaviour reproduces live bar-by-bar with the same obs/masks/slots,
#      behind hard kill switches - the bridge from a trained brain to a banked live pass.
# [2026-06-15] Live sizing mirrors training (margin + mode-aware cap).
#   I: live OPEN sized vs the buffer only - it ignored margin + the ftmo_mode cap, so live
#      lots could diverge from what the policy trained against.
#   R: Operator decision 2026-06-15 (margin model; train/live parity).
#   A: LivePortfolio.used_margin(); step_symbol passes apply_per_trade_cap=ftmo_mode +
#      price/contract/leverage/free_margin into risk.size - identical to the env.
#   C: The learned sizing reproduces live to the lot, so demo/funded behaviour matches
#      training - no live surprise that breaks a pass.
# [2026-06-15c] port_net_size normalized by max_lot (was hardcoded /50).
#   I: (audit) live divided net lots by literal 50.0 while the env divides by RiskConfig.max_lot
#      - a latent obs divergence if max_lot is ever overridden.
#   R: Logic audit 2026-06-15 (train obs == live obs).
#   A: LivePortfolio.max_lot (set from risk.cfg.max_lot); portfolio_block divides by it.
#   C: The port_net_size obs scalar matches the env for any max_lot, preserving train/live
#      parity so the learned behaviour reproduces live. (Live obs-warmup + LiveRunner are queued.)
# [2026-06-15d] Training wheels mirrored live (train/live mask parity).
#   I: The env gained operator counter-trend OPEN-block "training wheels"; live must apply
#      the IDENTICAL mask or the live legal space would differ from what the bot trained under.
#   R: Operator decision 2026-06-15; the same masks run in train + live (discipline transfers).
#   A: __init__ takes training_wheels (defaults cfg.TRAINING_WHEELS) + caches the
#      tw_cci_block/tw_bb_block column indices; _observe returns the 2 flags from market_row;
#      step_symbol passes them to build_direction_mask.
#   C: The live policy is forbidden the same counter-trend opens as in training, so the
#      breach-avoidance discipline transfers to the real challenge - and lifts identically
#      everywhere when config.TRAINING_WHEELS is flipped off.
