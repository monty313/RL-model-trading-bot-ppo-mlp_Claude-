"""TradingEnv — the FTMO challenge made steppable. 🔴 physics

WHAT THIS MODULE DOES
---------------------
The real-chart environment that brings M1-M3 together (SOW J2 Env contract):
  * 4 symbols stepped TRUE-SEQUENTIALLY each 1m bar (SOW B5) — one decision per
    symbol-step; the bar advances only after all 4 have acted.
  * 5 trade slots PER symbol; OPEN fills the next free slot (masked at 5), CLOSE is
    routed to the pointer-selected slot (SOW B2).
  * ONE shared account block (ChallengeState) read by every symbol — and updated
    within the bar, so symbol k sees the buffer already consumed by symbols 0..k-1.
    This is what guarantees the 4 symbols cannot collectively overshoot the daily-
    risk buffer in one bar (the B5 invariant).
  * Real FTMO costs on every fill (CostLayer); sizing via the RiskManager against the
    live remaining buffer; the LawMask enforces the -1e9 action mask each step.
  * Phase-A 4% trailing wall -> force-flatten all + lockout on breach. (The +2.5%
    auto-flat -> Phase B two-phase rule is M7; hooks are in ChallengeState.)

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
Training against faithful challenge physics — true-sequential shared risk, real
costs, hard wall, lawful action set — is what makes the learned behaviour transfer
to passing real challenges. If the env's physics are wrong, every downstream metric
lies. The reward returned here is the Layer-0 net-PnL proxy; the full layered reward
(L0-L6 + QUAD) wraps it in M6.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. The observation handed to the
policy is assembled here (precomputed market+raw 122 · law 12 · trade 35 · portfolio
3 · account 7 = 179). For a breach, the env's force-flatten + ChallengeState.breached
mark the moment; correlate it with the action distribution to find the danger-
blindness window. The env NEVER lets a masked action execute — a "bad action" in
telemetry was legal here, so blame the actor's intent, not the env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# COUPLING [C5] -> quantra/ftmo_passing/challenge_state.py: env builds ChallengeState and
# reads its attrs/methods (equity, peak_equity, remaining_buffer, breached, should_autoflat,
# mark_to_market, realize, charge, enter_phase_b, account_block). Their names are the contract.
from quantra.ftmo_passing.challenge_state import ChallengeState
from quantra.locked_core.cost_layer.costs import CostLayer
from quantra.locked_core.laws.laws import compute_law_states
from quantra.locked_core.risk_manager.risk import RiskManager
from quantra.learning_system.reward_engine.reward import RewardContext, RewardEngine
# COUPLING [C1] -> quantra/market_pipeline/feature_builder/schema.py: assemble_state expects
# the block order (precomputed market+raw · law · trade · portfolio · account); PRECOMPUTED_DIM
# fixes SymbolData.matrix width. A schema STATE_DIM/block change must be mirrored in _obs() below.
from quantra.market_pipeline.feature_builder import PRECOMPUTED_DIM, assemble_state
# COUPLING [C3] -> quantra/market_pipeline/feature_builder/schema.py: N_SLOTS=5 governs slot
# count + the 7*N_SLOTS trade block here; also schema._market_names order via PRECOMPUTED_NAMES (C1).
from quantra.market_pipeline.feature_builder.schema import N_SLOTS, PRECOMPUTED_NAMES

# Feature column index by name (reward proxies read a couple of market features).
# COUPLING [C1 in COUPLINGS.md]: depends on schema.PRECOMPUTED_NAMES ORDER (same map
# laws._IDX + scheduler._COL build). A feature reorder invalidates these lookups.
_COL = {name: i for i, name in enumerate(PRECOMPUTED_NAMES)}
# COUPLING [C2] -> quantra/market_pipeline/law_mask_engine/engine.py: direction action ints
# {HOLD=0,OPEN_LONG=1,OPEN_SHORT=2,CLOSE=3} are defined there and indexed in _apply_action; they
# must match ppo_agent/agent.py's direction head + live_bridge/live_session.py. Reorder -> wrong trades.
from quantra.market_pipeline.law_mask_engine.engine import (
    CLOSE,
    HOLD,
    MODE_LIVE,
    OPEN_LONG,
    OPEN_SHORT,
    build_direction_mask,
    build_pointer_mask,
)
from quantra.runtime import config as cfg
from quantra.runtime.config import ChallengeConfig


@dataclass
class SymbolData:
    """Per-symbol arrays the env steps over (all aligned to a shared 1m index)."""

    matrix: np.ndarray    # (T, PRECOMPUTED_DIM) precomputed features
    close: np.ndarray     # (T,) close price (execution)
    atr: np.ndarray       # (T,) ATR in price (sizing / entry distance)
    spread: np.ndarray    # (T,) spread in price (cost)
    valid_from: int = 0


@dataclass
class Slot:
    """One of the 5 trade slots per symbol."""

    occupied: bool = False
    direction: int = 0        # +1 long, -1 short
    entry_price: float = 0.0
    lots: float = 0.0
    risk_per_lot: float = 0.0  # USD/lot to its reference stop (for buffer accounting)
    age: int = 0
    mfe: float = 0.0          # max favourable uPnL (USD)
    mae: float = 0.0          # max adverse uPnL (USD)

    def upnl(self, close: float, contract: float) -> float:
        if not self.occupied:
            return 0.0
        return (close - self.entry_price) * self.direction * self.lots * contract


class TradingEnv:
    """Sequential multi-symbol env. One step = one symbol's decision at one bar."""

    def __init__(
        self,
        data: Dict[str, SymbolData],
        challenge: Optional[ChallengeConfig] = None,
        mask_mode: str = MODE_LIVE,
        required_laws: Optional[Sequence[str]] = None,
        stationarity_mode: str = "A",
        risk_cfg: Optional[cfg.RiskConfig] = None,
        cost_cfg: Optional[cfg.CostConfig] = None,
    ):
        self.symbols: List[str] = list(data.keys())            # fixed processing order
        self.data = data
        self.challenge_cfg = challenge or ChallengeConfig()
        self.mask_mode = mask_mode
        self.required_laws = list(required_laws) if required_laws else None
        self.stationarity_mode = stationarity_mode

        lengths = {len(d.matrix) for d in data.values()}
        assert len(lengths) == 1, "all symbols must share one aligned index/length"
        self.T = lengths.pop()
        self.start = max(d.valid_from for d in data.values())  # after every warmup

        self.risk = RiskManager(self.challenge_cfg.ftmo_account_size, risk_cfg)
        self.cost = CostLayer(cost_cfg)
        self.reward_engine = RewardEngine(self.challenge_cfg)   # M6 layered reward
        self.slots: Dict[str, List[Slot]] = {}
        self.account: ChallengeState = None  # set in reset()
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> np.ndarray:
        self.t = self.start
        self.cursor = 0  # which symbol acts next within the bar
        self.done = False
        self.account = ChallengeState(self.challenge_cfg.ftmo_account_size, self.challenge_cfg)
        self.slots = {s: [Slot() for _ in range(N_SLOTS)] for s in self.symbols}
        self._mark_to_market()
        self._prev_equity = self.account.equity
        return self._obs()

    # ------------------------------------------------------------------ helpers
    def _contract(self, sym: str) -> float:
        # COUPLING [C5] -> quantra/runtime/config.py: CONTRACT_SIZE is a per-symbol dict keyed
        # by config.SYMBOLS; cost_layer/costs.py + live_bridge/live_session.py read the same dict.
        return cfg.CONTRACT_SIZE.get(sym, 1.0)

    def _open_slots(self, sym: str) -> List[Slot]:
        return [sl for sl in self.slots[sym] if sl.occupied]

    def _n_open(self, sym: str) -> int:
        return sum(1 for sl in self.slots[sym] if sl.occupied)

    def _position(self, sym: str) -> int:
        """Net position sign for the symbol (slots are single-direction by mask)."""
        dirs = {sl.direction for sl in self._open_slots(sym)}
        if dirs == {1}:
            return 1
        if dirs == {-1}:
            return -1
        return 0  # flat (or — defensively — mixed, which the mask prevents)

    def _total_unrealized(self) -> float:
        tot = 0.0
        for sym in self.symbols:
            c = self.data[sym].close[self.t]
            con = self._contract(sym)
            for sl in self.slots[sym]:
                tot += sl.upnl(c, con)
        return tot

    def _committed_risk(self) -> float:
        """Sum of every open slot's committed risk (USD) — drives B5 buffer math."""
        return sum(sl.lots * sl.risk_per_lot
                   for sym in self.symbols for sl in self.slots[sym] if sl.occupied)

    def _mark_to_market(self) -> None:
        self.account.mark_to_market(self._total_unrealized())

    # ------------------------------------------------------------------ observation
    def _trade_block(self, sym: str) -> np.ndarray:
        """7 features x 5 slots = 35, schema order, normalized."""
        acct = self.account.account_size
        c = self.data[sym].close[self.t]
        atr = max(self.data[sym].atr[self.t], 1e-12)
        con = self._contract(sym)
        # COUPLING [C3/C1] -> quantra/market_pipeline/feature_builder/schema.py: trade block is
        # 7 features x N_SLOTS (schema 7*5=35); the per-slot field order written below (direction,
        # upnl, age, entry-dist, mfe, mae, occupied) must match schema's trade-block names.
        out = np.zeros(N_SLOTS * 7, dtype=np.float32)
        for i, sl in enumerate(self.slots[sym]):
            if not sl.occupied:
                continue
            base = i * 7
            out[base + 0] = sl.direction
            out[base + 1] = sl.upnl(c, con) / acct                 # uPnL normalized
            out[base + 2] = min(sl.age / 1000.0, 10.0)             # holding age
            out[base + 3] = (c - sl.entry_price) / atr             # entry distance (ATR)
            out[base + 4] = sl.mfe / acct
            out[base + 5] = sl.mae / acct
            out[base + 6] = 1.0                                    # occupied
        return out

    def _portfolio_block(self, sym: str) -> np.ndarray:
        """3 aggregates across the CURRENT symbol's slots (cross-symbol = account)."""
        c = self.data[sym].close[self.t]
        con = self._contract(sym)
        open_slots = self._open_slots(sym)
        net_exposure = sum(sl.direction for sl in open_slots) / N_SLOTS
        net_size = sum(sl.lots for sl in open_slots) / max(self.risk.cfg.max_lot, 1e-9)
        total_upnl = sum(sl.upnl(c, con) for sl in open_slots) / self.account.account_size
        return np.array([net_exposure, net_size, total_upnl], dtype=np.float32)

    def _law_states(self, sym: str) -> np.ndarray:
        return compute_law_states(self.data[sym].matrix[self.t])

    def direction_mask(self, sym: str) -> np.ndarray:
        return build_direction_mask(
            self._law_states(sym), self._position(sym), self._n_open(sym),
            self.mask_mode, self.required_laws, self.stationarity_mode,
        )

    def _obs(self) -> np.ndarray:
        sym = self.symbols[self.cursor]
        # COUPLING [C1] -> quantra/market_pipeline/feature_builder/__init__.py (assemble_state):
        # block kwargs (law_flags · trade · portfolio · account) + their widths must match schema
        # block_spans; the account block order mirrors challenge_state.account_block() / schema.
        return assemble_state(
            self.data[sym].matrix[self.t],
            law_flags=self._law_states(sym),
            trade=self._trade_block(sym),
            portfolio=self._portfolio_block(sym),
            account=self.account.account_block(),
        )

    # ------------------------------------------------------------------ execution
    def _apply_action(self, sym: str, direction: int, raw_size: float, pointer: int) -> dict:
        """Execute one symbol's action under the mask. Returns an info dict."""
        info = {"executed": "HOLD", "lots": 0.0, "cost": 0.0, "realized": 0.0,
                "size_reason": "", "coerced": False}
        dmask = self.direction_mask(sym)
        if dmask[direction] <= -1e8:          # forbidden -> coerce to HOLD (defensive)
            info["coerced"] = True
            direction = HOLD
        c = self.data[sym].close[self.t]
        atr = self.data[sym].atr[self.t]
        spread = self.data[sym].spread[self.t]
        con = self._contract(sym)

        if direction in (OPEN_LONG, OPEN_SHORT):
            free = next((sl for sl in self.slots[sym] if not sl.occupied), None)
            if free is None:
                return info  # all 5 full (mask should have blocked this)
            # Buffer available to THIS open = remaining buffer minus all committed risk
            # (incl. slots opened by prior symbols THIS bar -> true-sequential B5).
            available = self.account.remaining_buffer - self._committed_risk()
            sr = self.risk.size(sym, raw_size, atr, available)
            info["size_reason"] = sr.reason
            if not sr.feasible:
                return info
            d = 1 if direction == OPEN_LONG else -1
            free.occupied = True
            free.direction = d
            free.entry_price = c
            free.lots = sr.lots
            free.risk_per_lot = sr.risk_per_lot
            free.age = 0
            free.mfe = free.mae = 0.0
            oc = self.cost.open_cost(sym, sr.lots, spread)
            self.account.charge(oc.total)
            info.update(executed=("OPEN_LONG" if d == 1 else "OPEN_SHORT"),
                        lots=sr.lots, cost=oc.total)

        elif direction == CLOSE:
            occ = [i for i, sl in enumerate(self.slots[sym]) if sl.occupied]
            if not occ:
                return info
            idx = pointer if pointer in occ else occ[0]   # forced/snap to an open slot
            sl = self.slots[sym][idx]
            realized = sl.upnl(c, con)
            cc = self.cost.close_cost(sym, sl.lots)
            self.account.realize(realized)
            self.account.charge(cc.total)
            info.update(executed="CLOSE", lots=sl.lots, cost=cc.total, realized=realized)
            self.slots[sym][idx] = Slot()  # free it

        return info

    def _advance_bar(self) -> None:
        """Move to t+1: age slots, update MFE/MAE, daily reset on date change."""
        new_t = self.t + 1
        # daily reset (SOW §10.3) when the calendar day changes.
        # (index dates are carried on the matrices' producer; here we approximate via
        # a per-symbol date array if present — else skip; M7 wires the precise reset.)
        self.t = new_t
        for sym in self.symbols:
            c = self.data[sym].close[self.t]
            con = self._contract(sym)
            for sl in self.slots[sym]:
                if sl.occupied:
                    sl.age += 1
                    cur = sl.upnl(c, con)
                    sl.mfe = max(sl.mfe, cur)
                    sl.mae = min(sl.mae, cur)

    def _force_flatten(self) -> None:
        """Breach / lockout: realize every open slot at current price + close cost."""
        for sym in self.symbols:
            c = self.data[sym].close[self.t]
            con = self._contract(sym)
            for i, sl in enumerate(self.slots[sym]):
                if sl.occupied:
                    self.account.realize(sl.upnl(c, con))
                    self.account.charge(self.cost.close_cost(sym, sl.lots).total)
                    self.slots[sym][i] = Slot()

    def _reward(self, sym: str) -> float:
        """Build the RewardContext for the acting symbol and return the layered reward.

        L0 = equity delta after costs / account (dominant). Momentum is a proxy
        (in-position + CCI-sync agrees with trade dir + ATR alive); stagnation is left
        False in M4/M6 (the 3x5m-bar tracker is a documented later refinement). The
        full QUAD daily bonus is added at day boundaries by M7.
        """
        acct = self.account.account_size
        l0 = (self.account.equity - self._prev_equity) / acct
        pos = self._position(sym)
        in_pos = pos != 0
        row = self.data[sym].matrix[self.t]
        # COUPLING [C1/C7] -> quantra/market_pipeline/feature_builder/schema.py + builder.py:
        # "cci_sync_5m" / "atr_dev_1m" must exist in PRECOMPUTED_NAMES (builder emits them in that
        # order). Rename a feature there and these _COL lookups raise KeyError.
        cci = float(row[_COL["cci_sync_5m"]])
        atr_alive = float(row[_COL["atr_dev_1m"]]) > 0.0
        momentum = in_pos and (cci * pos > 0) and atr_alive
        dd_pct = max(0.0, (self.account.peak_equity - self.account.equity) / acct * 100.0)
        ctx = RewardContext(
            net_pnl_delta=l0, in_position=in_pos, momentum_aligned=momentum,
            stagnation=False, drawdown_pct=dd_pct,
            # COUPLING [C1] -> quantra/ftmo_passing/challenge_state.py: index [5] is day_progress
            # in account_block()'s fixed 7-scalar order; reorder the account block and this picks
            # the wrong scalar. RewardContext field names couple to reward_engine/reward.py.
            day_progress=float(self.account.account_block()[5]),
            breach_risk=dd_pct >= self.challenge_cfg.pain_zone_start_pct,
        )
        return self.reward_engine.reward(ctx)

    # ------------------------------------------------------------------ step
    def step(self, action) -> tuple:
        """One symbol-step. action = (direction:int, raw_size:float, pointer:int).

        Returns (obs, reward, done, info). reward is the Layer-0 net-PnL proxy
        (equity delta over the step); the full layered reward is M6.
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        # COUPLING [C2/C3] -> quantra/ppo_agent/agent.py + live_bridge/live_session.py: the action
        # tuple (direction int [C2], raw_size float, pointer slot int [0..N_SLOTS) [C3]) is the
        # policy->env contract; the agent's 4 heads must emit exactly this order/typing.
        direction, raw_size, pointer = int(action[0]), float(action[1]), int(action[2])
        sym = self.symbols[self.cursor]

        info = self._apply_action(sym, direction, raw_size, pointer)
        self._mark_to_market()  # equity reflects this symbol's cost/realize at bar t

        # Hard wall: breach -> force-flatten all + lockout + end episode. Else the
        # two-phase rule: at +2.5% day net, auto-flat ALL and switch to the Phase-B
        # 1% trailing wall (the episode continues — this is a WIN checkpoint, SOW §2.6).
        if self.account.breached:
            self._force_flatten()
            self._mark_to_market()
            self.done = True
        elif self.account.should_autoflat:
            self._force_flatten()
            self.account.enter_phase_b()
            self._mark_to_market()

        # Advance the within-bar cursor; after the last symbol, advance the bar.
        if not self.done:
            if self.cursor < len(self.symbols) - 1:
                self.cursor += 1
            else:
                self.cursor = 0
                if self.t + 1 >= self.T:
                    self.done = True
                else:
                    self._advance_bar()
                    self._mark_to_market()
                    if self.account.breached:
                        self._force_flatten()
                        self._mark_to_market()
                        self.done = True
                    elif self.account.should_autoflat:
                        self._force_flatten()
                        self.account.enter_phase_b()
                        self._mark_to_market()

        reward = self._reward(sym)                                # M6 layered reward
        self._prev_equity = self.account.equity
        info.update(symbol=sym, equity=self.account.equity,
                    remaining_buffer=self.account.remaining_buffer,
                    breached=self.account.breached)
        obs = None if self.done else self._obs()
        return obs, reward, self.done, info


def prepare_symbol_data(df_1m, symbol: str = "EURUSD", point_size: Optional[float] = None) -> SymbolData:
    """Build a SymbolData (features + execution arrays) from one symbol's 1m bars.

    Reuses the M2 FeatureBuilder for the precomputed matrix and the M2 indicators for
    the execution ATR, so what the bot SEES and what it TRADES on come from the same
    lookahead-safe source — no train/execute mismatch that would fake a pass.
    """
    from quantra.market_pipeline.feature_builder import indicators as ind
    from quantra.market_pipeline.feature_builder.builder import build_market_matrix

    # COUPLING [C5] -> quantra/runtime/config.py: POINT_SIZE per-symbol dict + DEFAULT_POINT_SIZE
    # scalar; same keys (config.SYMBOLS) used by data_loader/loader.py + cost_layer/costs.py.
    ps = point_size if point_size is not None else cfg.POINT_SIZE.get(symbol, cfg.DEFAULT_POINT_SIZE)
    mm = build_market_matrix(df_1m, point_size=ps)
    close = df_1m["close"].to_numpy(dtype=np.float64)
    atr = ind.atr(df_1m["high"], df_1m["low"], df_1m["close"], ind.ATR_PERIOD).fillna(0.0).to_numpy()
    spread = (df_1m["spread"].astype(float) * ps).to_numpy()
    return SymbolData(matrix=mm.matrix, close=close, atr=atr, spread=spread, valid_from=mm.valid_from)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M4 — implemented the sequential 4-symbol env.
#   I: Features/laws/risk/costs existed in isolation; nothing stepped them as the
#      actual FTMO challenge (shared account, 5 slots, true-sequential risk, wall).
#   R: SOW B5 (sequential loop, shared account, true-sequential within-bar), B2/B3
#      (5 slots, pointer CLOSE, next-free OPEN, masked at 5), §10.5 costs, §2.7 wall.
#   A: TradingEnv: per-symbol-step decisions; opens sized against the live buffer
#      MINUS all committed risk (so symbol k sees prior symbols' opens); costs on every
#      fill; -1e9 mask enforced; Phase-A wall force-flattens all; 179-dim obs assembled.
#   C: The bot now trains on faithful challenge physics where collective overshoot is
#      impossible by construction — so the behaviour it learns is the behaviour that
#      passes real challenges, not a simulation artifact.
# [2026-06-13] M6 — env reward now uses the layered RewardEngine.
#   I: step() returned a raw Layer-0 proxy; the policy needs the full layered reward.
#   R: REWARD_DESIGN.md (L0 dominant + shaping) wired via RewardContext.
#   A: Added self.reward_engine + _reward(sym) building the context (L0 equity delta,
#      momentum proxy, daily drawdown for pain zone, day progress, breach-risk).
#   C: Training now optimizes the real objective with Layer-0 dominance, so PPO is
#      pulled toward net progress inside the legal/risk-safe space - i.e. toward passing.
