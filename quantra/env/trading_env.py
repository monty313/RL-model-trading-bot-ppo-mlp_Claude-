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
policy is assembled here (precomputed market+raw 149 · law 12 · trade 35 · portfolio
3 · account 8 = 207). For a breach, the env's force-flatten + ChallengeState.breached
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
    dates: Optional[np.ndarray] = None  # (T,) integer calendar-day id per bar -> daily reset [fix]


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
        training_wheels: Optional[bool] = None,
        episode_days: Optional[int] = None,
        reward_cfg: Optional[cfg.RewardConfig] = None,
    ):
        self.symbols: List[str] = list(data.keys())            # fixed processing order
        self.data = data
        self.challenge_cfg = challenge or ChallengeConfig()
        self.mask_mode = mask_mode
        self.required_laws = list(required_laws) if required_laws else None
        self.stationarity_mode = stationarity_mode
        # C10 [operator 2026-06-19]: an episode spans this many TRADING DAYS on ONE continuous
        # account (balance carries forward; a daily breach locks out the rest of that day and
        # resets at midnight — it does NOT end the episode). None = run to end-of-data (the
        # single-window behaviour synthetic tests rely on). The notebooks pass TRAINING_DAYS /
        # EVAL_DAYS. The episode also ends if equity falls to cfg.ACCOUNT_FLOOR_EQUITY (blown).
        self.episode_days = episode_days
        # Training wheels [operator 2026-06-15]: semi-permanent counter-trend OPEN blocks.
        # Default follows config.TRAINING_WHEELS so a global flip removes them everywhere.
        self.training_wheels = cfg.TRAINING_WHEELS if training_wheels is None else bool(training_wheels)

        lengths = {len(d.matrix) for d in data.values()}
        assert len(lengths) == 1, "all symbols must share one aligned index/length"
        self.T = lengths.pop()
        self.start = max(d.valid_from for d in data.values())  # after every warmup

        self.risk_cfg = risk_cfg or cfg.RiskConfig()
        # C16 [2026-06-19]: operator-tunable reward weights (plain-English RewardConfig). Threaded
        # into the RewardEngine here + on reset(challenge=) so a run's objective is explicit + saved.
        self.reward_cfg = reward_cfg or cfg.RewardConfig()
        self.risk = RiskManager(self.challenge_cfg.ftmo_account_size, self.risk_cfg)
        self.cost = CostLayer(cost_cfg)
        self.reward_engine = RewardEngine(self.challenge_cfg, self.reward_cfg)   # M6 layered reward
        self.slots: Dict[str, List[Slot]] = {}
        self.account: ChallengeState = None  # set in reset()
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, challenge: Optional[ChallengeConfig] = None) -> np.ndarray:
        # Per-day injection [2026-06-15]: pass a fresh ChallengeConfig to start this
        # episode/day on operator-chosen target/risk/leverage/mode WITHOUT rebuilding the
        # env. Rebuilds RiskManager (account size) + RewardEngine (pain band) so the whole
        # stack tracks the new config. COUPLING -> runtime/config.make_challenge() builds it.
        if challenge is not None:
            self.challenge_cfg = challenge
            self.risk = RiskManager(self.challenge_cfg.ftmo_account_size, self.risk_cfg)
            self.reward_engine = RewardEngine(self.challenge_cfg, self.reward_cfg)
        self.t = self.start
        self.cursor = 0  # which symbol acts next within the bar
        self.done = False
        self._days_elapsed = 0            # C10: completed calendar-day boundaries this episode
        self._pending_day_penalty = 0.0   # C11: failed-day penalty for the day that just ended
        # C17: signed, account-normalized trade-quality accrued by this step's closes; consumed in
        # _reward() -> RewardContext.trade_close_quality -> reward_engine/reward.py decompose() L4.
        self._pending_trade_quality = 0.0
        self.account = ChallengeState(self.challenge_cfg.ftmo_account_size, self.challenge_cfg)
        self.slots = {s: [Slot() for _ in range(N_SLOTS)] for s in self.symbols}
        # PER-SYMBOL reward attribution [2026-06-15 fix]: each symbol's L0 reflects ONLY its
        # own positions' PnL, not the whole-portfolio bar move (which used to land entirely on
        # the last symbol's step). _sym_realized = closed PnL net of costs; contrib_prev = the
        # last graded contribution. COUPLING -> _reward()/_apply_action()/_force_flatten().
        self._sym_realized = {s: 0.0 for s in self.symbols}
        self._sym_contrib_prev = {s: 0.0 for s in self.symbols}
        self._decision_t = self.t
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

    def _sym_unrealized(self, sym: str) -> float:
        c = self.data[sym].close[self.t]
        con = self._contract(sym)
        return sum(sl.upnl(c, con) for sl in self.slots[sym] if sl.occupied)

    def _sym_contribution(self, sym: str) -> float:
        """This symbol's OWN PnL = its realized-net (closed PnL − costs) + its open slots'
        unrealized at the current bar. Sum over symbols == equity − account_size, so this is
        a true decomposition. Used for per-symbol L0 attribution [2026-06-15 credit-assignment
        fix]: the reward for holding a EURUSD winner lands on the EURUSD step, not on US30."""
        return self._sym_realized[sym] + self._sym_unrealized(sym)

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

    def _wheel_states(self, sym: str) -> np.ndarray:
        """(tw_cci_block, tw_bb_block) for the current bar — precomputed market flags.
        COUPLING [C9] -> feature_builder/builder.py emits these; engine consumes them."""
        row = self.data[sym].matrix[self.t]
        return np.array([row[_COL["tw_cci_block"]], row[_COL["tw_bb_block"]]], dtype=np.float32)

    def direction_mask(self, sym: str) -> np.ndarray:
        m = build_direction_mask(
            self._law_states(sym), self._position(sym), self._n_open(sym),
            self.mask_mode, self.required_laws, self.stationarity_mode,
            training_wheels=self.training_wheels,
            wheel_states=self._wheel_states(sym) if self.training_wheels else None,
        )
        # C10 [2026-06-19]: while the day is locked out (a daily-wall breach OR an OFF-mode
        # stop-for-day at target), forbid NEW opens for the rest of that day — the account is flat
        # and stays flat until midnight reset_day() lifts the lockout. CLOSE/HOLD stay legal. The
        # agent SEES the block in its mask (no wasted probability); _apply_action coerces as a backstop.
        if self.account is not None and self.account.locked_out:
            m = m.copy()
            m[OPEN_LONG] = -1e9
            m[OPEN_SHORT] = -1e9
        return m

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
            # Margin ceiling (1:leverage) + mode-aware per-trade cap [2026-06-15]:
            # ftmo_mode ON keeps the 1%-of-account per-trade cap; OFF removes it so
            # confidence scales the whole budget and MARGIN is the real physical ceiling.
            sr = self.risk.size(
                sym, raw_size, atr, available,
                apply_per_trade_cap=self.challenge_cfg.ftmo_mode,
                price=c, contract=con,
                leverage=self.challenge_cfg.leverage,
                free_margin=self.account.equity - self._used_margin(),
            )
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
            self._sym_realized[sym] -= oc.total          # per-symbol attribution
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
            self._sym_realized[sym] += realized - cc.total   # per-symbol attribution
            # C17: feed L4 trade-quality (read sl.mfe BEFORE the slot is freed -> reward_engine L4).
            self._record_close_quality(realized, sl.mfe > 0.0)
            info.update(executed="CLOSE", lots=sl.lots, cost=cc.total, realized=realized)
            self.slots[sym][idx] = Slot()  # free it

        return info

    def _advance_bar(self) -> None:
        """Move to t+1: age slots, update MFE/MAE, and on a calendar-day change grade the day that
        just ended (C11 failed-day penalty), reset the day (C10), and end the episode once the
        configured number of trading days is exhausted."""
        new_t = self.t + 1
        self.t = new_t
        # Daily reset (SOW §10.3) on a calendar-day change: re-anchor the day + fresh Phase-A
        # target/wall, so each day is its own challenge [2026-06-15 fix: reset_day now actually
        # fires]. Needs SymbolData.dates; synthetic data without dates keeps single-episode semantics.
        d = self.data[self.symbols[0]].dates
        if d is not None and d[self.t] != d[self.t - 1]:
            # C11 [2026-06-19]: grade the FINISHED day BEFORE reset_day re-anchors. The penalty is
            # stashed and added to this step's reward in step(). The account (balance/equity) carries
            # forward across the boundary — only the day anchors + breach/lockout flags reset (C10).
            self._pending_day_penalty += self._failed_day_penalty()
            self.account.reset_day()
            self._days_elapsed += 1
            # C10: an episode is episode_days TRADING DAYS; end it once they are exhausted.
            if self.episode_days is not None and self._days_elapsed >= self.episode_days:
                self.done = True
        for sym in self.symbols:
            c = self.data[sym].close[self.t]
            con = self._contract(sym)
            for sl in self.slots[sym]:
                if sl.occupied:
                    sl.age += 1
                    cur = sl.upnl(c, con)
                    sl.mfe = max(sl.mfe, cur)
                    sl.mae = min(sl.mae, cur)

    def _used_margin(self) -> float:
        """Broker margin currently tied up by all open slots (notional / leverage)
        [2026-06-15]. free_margin = equity − used_margin feeds the RiskManager's margin
        ceiling, so OPEN is blocked when the 1:leverage account can't carry more size.
        COUPLING -> locked_core/risk_manager/risk.py size(leverage=, free_margin=)."""
        lev = max(1.0, self.challenge_cfg.leverage)
        total = 0.0
        for sym in self.symbols:
            c = self.data[sym].close[self.t]
            con = self._contract(sym)
            for sl in self.slots[sym]:
                if sl.occupied:
                    total += (sl.lots * con * c) / lev
        return total

    def _on_target_autoflat(self) -> None:
        """Day target hit -> flatten all. ftmo_mode ON: enter the tighter Phase-B wall and keep
        trading (banks the pass, SOW §2.6). ftmo OFF + stop_for_day: bank the day and LOCK OUT the
        rest of that day (C10: the DAY stops, not the episode — it resumes next midnight). OFF
        without stop_for_day never reaches here (should_autoflat stays False so it runs PAST the
        target) [operator decision 2026-06-15: OFF keeps target as aim]."""
        self._force_flatten()
        if self.challenge_cfg.ftmo_mode:
            self.account.enter_phase_b()
        else:
            # CHANGED: 2026-06-19 (C10) | was self.done = True. A stop-for-day now LOCKS OUT the day
            # so the multi-day episode continues; reset_day() lifts the lockout at midnight.
            self.account.locked_out = True
        self._mark_to_market()

    def _force_flatten(self) -> None:
        """Breach / lockout: realize every open slot at current price + close cost."""
        for sym in self.symbols:
            c = self.data[sym].close[self.t]
            con = self._contract(sym)
            for i, sl in enumerate(self.slots[sym]):
                if sl.occupied:
                    pnl = sl.upnl(c, con)
                    cost = self.cost.close_cost(sym, sl.lots).total
                    self.account.realize(pnl)
                    self.account.charge(cost)
                    self._sym_realized[sym] += pnl - cost     # per-symbol attribution
                    # C17: a forced close still feeds L4 trade-quality (read sl.mfe before freeing)
                    # — round-tripping a winner into a breach/lockout is exactly a "gave back a winner".
                    self._record_close_quality(pnl, sl.mfe > 0.0)
                    self.slots[sym][i] = Slot()

    def _failed_day_penalty(self) -> float:
        """C11 [2026-06-19]: the end-of-day penalty for the day that just finished. A LARGE,
        PROPORTIONAL hit — reward = -failed_day_penalty * day_shortfall_fraction — applied at the
        midnight boundary. 0 when the day's target was reached; grows the further below target the
        day ended; worst for days that hit the wall (largest shortfall). Big by design so it is not
        averaged away over a day's bars: it makes CONSISTENCY (pass most days), not mere survival,
        the objective. COUPLING -> challenge_state.day_shortfall_fraction + config.failed_day_penalty."""
        w = float(getattr(self.challenge_cfg, "failed_day_penalty", 0.0))
        if w <= 0.0:
            return 0.0
        return -w * self.account.day_shortfall_fraction

    def _account_blown(self) -> bool:
        """C10: the episode ends if the continuous account is blown — equity at/below the floor
        (cfg.ACCOUNT_FLOOR_EQUITY, default 0.0). A daily breach only LOCKS OUT the day; only a blown
        account (or exhausting episode_days / end-of-data) ends the multi-day episode."""
        return self.account.equity <= cfg.ACCOUNT_FLOOR_EQUITY

    def _record_close_quality(self, realized_gross: float, ever_in_profit: bool) -> None:
        """C17 [2026-06-19]: accrue ONE closed trade's trade-quality whisper for the current step.
        Operator sign rules: a profitable close rewards (+); a losing close that was EVER in profit
        penalizes (-, "gave back a winner"); a losing close that was never in profit contributes 0
        (it just didn't work — no penalty). `realized_gross` is the trade's uPnL at the close price
        (gross, pre close-cost) — the SAME quantity used for `ever_in_profit` via Slot.mfe>0, so the
        winner/loser test and the in-profit test are on one consistent basis (close costs already
        live in L0). Normalized by account_size so it shares L0's units (account fraction) and can't
        break the E8 Layer-0 dominance proof.
        COUPLING -> learning_system/reward_engine/reward.py: this sum becomes
        RewardContext.trade_close_quality, scaled in decompose() as L4 = trade_quality_weight * sum.
        Call sites: _apply_action() CLOSE branch + _force_flatten() (every realization)."""
        if realized_gross > 0.0:
            q = realized_gross
        elif realized_gross < 0.0 and ever_in_profit:
            q = -abs(realized_gross)
        else:
            return  # losing trade that was never in profit (or exactly flat) -> no L4 signal
        self._pending_trade_quality += q / self.account.account_size

    def _consume_trade_quality(self) -> float:
        """Return the trade-quality accrued this step and reset the accumulator (consumed once per
        step by _reward, so each L4 contribution is counted on exactly the step its trade closed)."""
        q = self._pending_trade_quality
        self._pending_trade_quality = 0.0
        return q

    def _reward(self, sym: str) -> float:
        """Build the RewardContext for the acting symbol and return the layered reward.

        L0 = equity delta after costs / account (dominant). Momentum (L1) is a proxy
        (in-position + CCI-sync agrees with trade dir + ATR alive). L2 daily-progress and L4
        trade-quality are the C17 [2026-06-19] re-pointed terms — fed here from ChallengeState
        (day_pnl / daily_target_equity) and from this step's accrued close-quality. The full QUAD
        daily bonus is added at day boundaries by M7.
        """
        acct = self.account.account_size
        # PER-SYMBOL L0 [2026-06-15 fix]: reward this symbol's OWN PnL change since its last
        # step, NOT the whole-portfolio equity delta (which mis-credited the last symbol).
        contrib = self._sym_contribution(sym)
        l0 = (contrib - self._sym_contrib_prev[sym]) / acct
        self._sym_contrib_prev[sym] = contrib
        pos = self._position(sym)
        in_pos = pos != 0
        # Momentum read at the DECISION bar (not the advanced bar) so all 4 symbols are graded
        # symmetrically on the bar the action was actually decided on [2026-06-15 fix].
        row = self.data[sym].matrix[self._decision_t]
        # COUPLING [C1/C7] -> quantra/market_pipeline/feature_builder/schema.py + builder.py:
        # "cci_sync_5m" / "atr_dev_1m" must exist in PRECOMPUTED_NAMES (builder emits them in that
        # order). Rename a feature there and these _COL lookups raise KeyError.
        cci = float(row[_COL["cci_sync_5m"]])
        atr_alive = float(row[_COL["atr_dev_1m"]]) > 0.0
        momentum = in_pos and (cci * pos > 0) and atr_alive
        dd_pct = max(0.0, (self.account.peak_equity - self.account.equity) / acct * 100.0)
        ctx = RewardContext(
            net_pnl_delta=l0, in_position=in_pos, momentum_aligned=momentum,
            drawdown_pct=dd_pct,
            breach_risk=dd_pct >= self.challenge_cfg.pain_zone_start_pct,
            # C17 [2026-06-19] re-pointed reward inputs. COUPLING -> quantra/ftmo_passing/
            # challenge_state.py: day_pnl (= equity - day_start_equity) and daily_target_equity
            # (= day_start_equity*(1+daily_target_pct/100)) are READ here by those property names —
            # rename either there and L2 daily-progress breaks. COUPLING -> learning_system/
            # reward_engine/reward.py: these keyword names must match RewardContext's fields, and
            # decompose() turns them into L2 = daily_progress_weight*max(0,day_pnl)/day_target_equity.
            day_pnl=self.account.day_pnl,
            day_target_equity=self.account.daily_target_equity,
            # trade_close_quality is the per-step sum the env accrued while CLOSING trades this step
            # (_record_close_quality, called from _apply_action's CLOSE branch + _force_flatten);
            # consume-and-reset it here so it lands on exactly this step's L4 trade-quality term.
            trade_close_quality=self._consume_trade_quality(),
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
        self._decision_t = self.t          # bar the action is decided on (pre any advance)

        info = self._apply_action(sym, direction, raw_size, pointer)
        self._mark_to_market()  # equity reflects this symbol's cost/realize at bar t

        # C10 [2026-06-19]: a daily-wall breach FORCE-FLATTENS all + LOCKS OUT the rest of the day
        # (account.locked_out was set in mark_to_market) but does NOT end the episode — the day
        # resets at midnight and the account carries forward. Else the +target rule: ON enters
        # Phase B (banks the pass, episode continues); OFF+stop_for_day locks out the day (both via
        # _on_target_autoflat). Only a BLOWN account ends the episode here.
        if self.account.breached:
            self._force_flatten()
            self._mark_to_market()
        elif self.account.should_autoflat:
            self._on_target_autoflat()
        if self._account_blown():
            self.done = True

        # Advance the within-bar cursor; after the last symbol, advance the bar.
        if not self.done:
            if self.cursor < len(self.symbols) - 1:
                self.cursor += 1
            else:
                self.cursor = 0
                if self.t + 1 >= self.T:
                    self.done = True
                else:
                    self._advance_bar()   # may end the episode at episode_days + stash a C11 penalty
                    self._mark_to_market()
                    if self.account.breached:
                        self._force_flatten()
                        self._mark_to_market()
                    elif self.account.should_autoflat:
                        self._on_target_autoflat()
                    if self._account_blown():
                        self.done = True

        # C11 [2026-06-19]: add the failed-day penalty for any day that ended on this step (stashed
        # by _advance_bar at the midnight boundary) on top of the per-step layered reward.
        day_penalty = self._pending_day_penalty
        self._pending_day_penalty = 0.0
        reward = self._reward(sym) + day_penalty
        self._prev_equity = self.account.equity
        info.update(symbol=sym, equity=self.account.equity,
                    remaining_buffer=self.account.remaining_buffer,
                    breached=self.account.breached, locked_out=self.account.locked_out,
                    day_penalty=day_penalty, days_elapsed=self._days_elapsed)
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
    # Calendar-day id per bar (midnight-normalized) so the env can fire the daily reset on a
    # day change [2026-06-15 fix]. Non-datetime index (synthetic tests) -> None (single episode).
    try:
        dates = df_1m.index.normalize().asi8.copy()
    except Exception:
        dates = None
    return SymbolData(matrix=mm.matrix, close=close, atr=atr, spread=spread,
                      valid_from=mm.valid_from, dates=dates)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# STANDING RULE [2026-06-19, operator] — applies to THIS file and EVERY file going forward: keep
# SHOWING THE WORK. On every edit (1) append a DATED IRAC entry here, and (2) in the code comments
# DOCUMENT the cross-file RELATIONSHIPS the change depends on (the COUPLING) — name the other file(s)
# and the exact attr/field/key relied on, in BOTH directions — and date the re-pointed logic, so any
# future reader/editor can see what connects to what, and what breaks where, and when it changed.
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M4 — implemented the sequential 4-symbol env.
#   I: Features/laws/risk/costs existed in isolation; nothing stepped them as the
#      actual FTMO challenge (shared account, 5 slots, true-sequential risk, wall).
#   R: SOW B5 (sequential loop, shared account, true-sequential within-bar), B2/B3
#      (5 slots, pointer CLOSE, next-free OPEN, masked at 5), §10.5 costs, §2.7 wall.
#   A: TradingEnv: per-symbol-step decisions; opens sized against the live buffer
#      MINUS all committed risk (so symbol k sees prior symbols' opens); costs on every
#      fill; -1e9 mask enforced; Phase-A wall force-flattens all; full-width obs assembled.
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
# [2026-06-15] Per-day challenge injection + margin-aware sizing.
#   I: reset() couldn't take a fresh per-day config; OPEN ignored margin; no free-margin calc.
#   R: Operator decision 2026-06-15 (adjustable per-day inputs; leverage/margin model).
#   A: store self.risk_cfg; reset(challenge=) rebuilds RiskManager+RewardEngine; _used_margin()
#      feeds free_margin; OPEN passes apply_per_trade_cap=ftmo_mode + price/contract/leverage.
#   C: A day can start on operator-chosen target/stop/leverage/mode and the bot sizes against
#      the real margin ceiling - faithful challenge physics for both modes, no overshoot.
# [2026-06-15b] _on_target_autoflat: mode-correct target handling.
#   I: both auto-flat sites hard-coded enter_phase_b; OFF + stop_for_day needs to bank+stop.
#   R: Operator correction 2026-06-15 (OFF keeps the target; stop_for_day banks the day).
#   A: Routed both target-hit sites through _on_target_autoflat() - ON enters Phase B; OFF
#      with stop_for_day flattens + ends the day; OFF default never triggers (runs on).
#   C: The target behaves the operator's way in every mode, so side-account days bank
#      cleanly while the FTMO pass still locks behind the tighter wall.
# [2026-06-15c] Logic-audit fixes: per-symbol L0 attribution + momentum timing + daily reset.
#   I: (audit) the whole-bar portfolio PnL was credited to the LAST symbol's step (dominant
#      L0 mis-attribution); the last symbol's momentum was graded at t+1; reset_day never fired.
#   R: Logic audit 2026-06-15 (verified bugs) — fix WITHOUT breaking B5 no-overshoot.
#   A: _sym_realized/_sym_contribution give each symbol its OWN PnL delta as L0 (exact
#      decomposition: Σ contributions == equity−account_size); _decision_t grades momentum at
#      the decision bar; _advance_bar fires account.reset_day() on a SymbolData.dates change.
#   C: The dominant learning signal now reaches the asset that actually held the position, and
#      each day is its own fresh 2.5%/4% challenge — the credit assignment the bot needs to LEARN
#      to pass. Demonstrated: 15/15 synthetic challenge-days banked +2.5%, 0 breached (worst DD 2.86%).
# [2026-06-15d] Training wheels — counter-trend OPEN blocks wired into direction_mask.
#   I: Operator wants semi-permanent "training wheels" that forbid opening against a strong
#      30m+4H trend (CCI 5/15 + BB 10/100 context) so the bot stops wasting episodes on
#      breach-bound counter-trend trades.
#   R: Operator decision 2026-06-15; same masks must run in train + live (discipline transfers);
#      isolated from the locked 9 laws; removable via config.TRAINING_WHEELS.
#   A: Added self.training_wheels (defaults to cfg.TRAINING_WHEELS) + _wheel_states(sym) reading
#      the precomputed tw_cci_block/tw_bb_block flags; direction_mask passes them to
#      build_direction_mask. live_session does the same for parity.
#   C: The policy can no longer open into the wheels' trend, so fewer breaches per window =>
#      more banked days per seed => faster, cheaper convergence to a consistently-passing brain.
# [2026-06-19] C10/C11 — multi-day episode (one continuous account) + the failed-day penalty.
#   I: An episode was a single window that ENDED on the first breach; there was no notion of "N
#      trading days on one account", a breached day couldn't reopen, and nothing punished failing
#      a day's target — so the bot could learn to merely survive, not to pass consistently.
#   R: Operator spec 2026-06-19 (C10: episode = N days on ONE continuous account; breach = lockout
#      + reset_day, NOT done; ends at N days or a blown account. C11: large proportional EOD penalty
#      for missing the daily target. C13 dropped.).
#   A: TradingEnv(episode_days=) + _days_elapsed; breach now force-flattens + locks out the day
#      (no done) — only _account_blown()/episode_days/end-of-data end the episode; direction_mask
#      forbids opens while locked_out; _on_target_autoflat OFF+stop_for_day locks out the day (was
#      done); _advance_bar grades the finished day via _failed_day_penalty() (∝ day_shortfall_fraction)
#      BEFORE reset_day, counts days, and ends at episode_days; the penalty is added to that step's
#      reward and surfaced in info (day_penalty/locked_out/days_elapsed).
#   C: The bot now trains on a faithful many-day account where a bad day costs real carried-forward
#      equity AND a big reward hit, and a breached day reopens fresh next morning — so it learns the
#      CONSISTENCY (pass most days, survive the account) that repeatedly passing FTMO actually requires.
# [2026-06-19] C17 — feed the re-pointed daily-progress + trade-quality reward terms.
#   I: After C16 the reward weights had plain-English names but proxy math; the operator re-pointed
#      two terms to LITERALLY measure their names, which the env must now supply inputs for.
#   R: Operator spec 2026-06-19 (C17: daily-progress = +reward growing with how far into profit the
#      day is; trade-quality = +on a winning close, − on giving back a once-profitable trade, 0 on a
#      never-profitable loser; re-verify E8; nothing else changes).
#   A: _reward() now passes day_pnl + day_target_equity (read from ChallengeState.day_pnl /
#      .daily_target_equity — COUPLING -> ftmo_passing/challenge_state.py) and trade_close_quality
#      into RewardContext (COUPLING -> learning_system/reward_engine/reward.py decompose L2/L4).
#      Added _record_close_quality() (Slot.mfe>0 = "ever in profit"; account-normalized; the operator
#      sign rules) called from the CLOSE branch of _apply_action + from _force_flatten, accrued into
#      _pending_trade_quality and consumed once per step by _consume_trade_quality(). Dropped the dead
#      stagnation/day_progress ctx args.
#   C: The reward the bot actually receives now rewards getting closer to each day's target and BANKING
#      winners (and discourages round-tripping them into losses/breaches) — directly shaping the
#      consistent, profit-locking behaviour that passing FTMO repeatedly requires, with L0 still dominant.
