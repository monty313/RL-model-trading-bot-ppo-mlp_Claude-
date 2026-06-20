"""barbershop_runner — drive a TradingEnv with a policy for ONE pass and return per-day metrics.

WHAT THIS MODULE DOES
---------------------
The Barbershop fast loop (colab/Quantra_Barbershop.ipynb, Cell 5) needs REAL per-day scoreboard
rows — pass/fail, day P&L %, worst drawdown, breach, trade count — to feed the Policy Registry's
performance.json + the Leaderboard. This module is the single integration point that was a DEMO
stub: it builds a TradingEnv from the run's OVERRIDES, runs the policy deterministically over the
N_DAYS window (C10: ONE continuous account, the balance carries forward day to day), and snapshots
each finished day.

WHY PER-PASS, NOT PER-DAY: C10 makes the account continuous across the window, so a day cannot be
simulated in isolation — run_pass() runs the whole episode (episode_days=N_DAYS) and splits it at
the env's calendar-day boundaries.

COUPLING (both directions):
  -> env/trading_env.py: builds TradingEnv(challenge, reward_cfg, training_wheels, episode_days);
     steps it with the action tuple (direction_int, size_float, pointer_int); reads info["executed"/
     "coerced"/"breached"/"locked_out"/"days_elapsed"] + account.equity/peak_equity/account_size.
     Renaming those env outputs/attrs breaks the per-day snapshot here.
  -> ppo_agent/agent.py: agent.act_deterministic(obs, dir_mask, ptr_mask) -> (a_dir,a_size,a_ptr,value)
     (the same 4-tuple acceptance.py/live_session use, in that order).
  -> market_pipeline/law_mask_engine/engine.py: build_pointer_mask([slot.occupied,...]).
  -> runtime/config.py: make_challenge()/RewardConfig() translate the OVERRIDES; training_phase is a
     GLOBAL (engine reads cfg.TRAINING_PHASE) so build_env() sets it from the override.
  <- colab/Quantra_Barbershop.ipynb Cell 5: calls run_pass() per pass; the rows feed Cell 6
     (PassRecord -> performance.json -> Leaderboard) and print_pass_table.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from quantra.runtime import config as cfg
from quantra.env.trading_env import TradingEnv, SymbolData
from quantra.market_pipeline.law_mask_engine.engine import build_pointer_mask

# COUPLING -> runtime/config.RewardConfig field names: only these weight knobs are forwarded from an
# OVERRIDES dict into RewardConfig (failed_day_penalty is a challenge knob, handled by make_challenge).
_REWARD_KEYS = ("net_pnl_weight", "step_pnl_weight", "daily_progress_weight",
                "drawdown_pain_weight", "drawdown_pain_steepness", "trade_quality_weight")
_TRADE_ACTIONS = ("OPEN_LONG", "OPEN_SHORT", "CLOSE")


def build_env(data: Dict[str, "object"], overrides: Optional[dict], n_days: int) -> TradingEnv:
    """Construct a TradingEnv that realizes the OVERRIDES for an N_DAYS continuous-account episode.
    Mirrors what config.build_overrides_dict() records, so a saved policy's manifest and the env it
    actually ran are the same configuration. NOTE: training_phase is a GLOBAL knob (the law-mask
    engine reads cfg.TRAINING_PHASE), so it is set here from the override — a documented side effect."""
    ov = overrides or {}
    challenge = cfg.make_challenge(
        daily_target_pct=ov.get("daily_target_pct", 2.5),
        daily_risk_pct=ov.get("daily_risk_pct", 4.0),
        ftmo_mode=ov.get("ftmo_mode", True),
        stop_for_day=ov.get("stop_for_day", False),
        permanent_dd_pct=ov.get("permanent_dd_pct", 10.0),
        failed_day_penalty=ov.get("failed_day_penalty", 5.0))
    reward_cfg = cfg.RewardConfig(**{k: ov[k] for k in _REWARD_KEYS if k in ov})
    # Per-trade risk cap (sizing) [2026-06-19]: optional override. None -> default RiskConfig (1%/trade).
    # Lower this to shrink position size so a day physically can't lose more than ~N_SLOTS*frac before the
    # wall — the structural lever for stopping breaches (reward shaping can't, by E8). COUPLING ->
    # runtime/config.py RiskConfig.max_per_trade_risk_frac + env/trading_env.py (risk_cfg -> RiskManager).
    risk_frac = ov.get("max_per_trade_risk_frac")
    risk_cfg = cfg.RiskConfig(max_per_trade_risk_frac=float(risk_frac)) if risk_frac is not None else None
    phase = ov.get("training_phase")
    if phase is not None:   # COUPLING -> law_mask_engine reads cfg.TRAINING_PHASE (a module global)
        cfg.TRAINING_PHASE = cfg.PHASE_CONSTRAINED if str(phase) == "constrained" else cfg.PHASE_FREE
    return TradingEnv(data, challenge=challenge, reward_cfg=reward_cfg, risk_cfg=risk_cfg,
                      training_wheels=ov.get("training_wheels", cfg.TRAINING_WHEELS),
                      cci_regime_gate=ov.get("cci_regime_gate", cfg.CCI_REGIME_GATE),
                      episode_days=n_days)


def slice_symbol_data(sd: SymbolData, start_index: int) -> SymbolData:
    """Return a SymbolData beginning at start_index — lets the Barbershop honor its START_DATE input
    WITHOUT re-running the (slow) feature build: the matrix/close/atr/spread/dates rows align 1:1 with
    the source 1m df, so the notebook maps START_DATE -> a bar index (df.index.searchsorted) and slices
    the already-built SymbolData here. COUPLING -> env/trading_env.py SymbolData fields."""
    i = max(0, int(start_index))
    return SymbolData(matrix=sd.matrix[i:], close=sd.close[i:], atr=sd.atr[i:], spread=sd.spread[i:],
                      valid_from=max(0, sd.valid_from - i),
                      dates=None if sd.dates is None else sd.dates[i:])


class _DayAccum:
    """Accumulates one calendar day's stats as the env steps through it (the env re-anchors at the
    midnight boundary, so the finished day must be measured live, not read back afterwards)."""

    def __init__(self, open_equity: float):
        self.open_equity = open_equity
        self.peak = open_equity
        self.max_dd_pct = 0.0
        self.trades = 0
        self.coerced = 0
        self.steps = 0
        self.breached = False
        # RULE 2 [operator 2026-06-20]: win rate is measured over EVERY trade realized in the day — a
        # discretionary CLOSE, a breach/target force-flatten, OR the end-of-day flatten (all positions
        # are closed by EOD, so every trade gets a verdict). A "win" is a close whose realized PnL is
        # positive NET of the close cost. The env surfaces both per step as info["n_closed"]/["n_wins"]
        # (env._step_closed_nets), so this no longer undercounts the trades the bot didn't close itself.
        self.closes = 0
        self.wins = 0

    def update(self, env: TradingEnv, info: dict) -> None:
        acct = env.account
        self.peak = max(self.peak, acct.equity)
        self.max_dd_pct = max(self.max_dd_pct,
                              (self.peak - acct.equity) / acct.account_size * 100.0)
        if info.get("executed", "HOLD") in _TRADE_ACTIONS:
            self.trades += 1
        self.closes += int(info.get("n_closed", 0))               # RULE 2: ALL realized closes this step
        self.wins += int(info.get("n_wins", 0))                   # ... that won NET of the close cost
        if info.get("coerced"):
            self.coerced += 1
        if info.get("breached") or info.get("locked_out"):
            self.breached = True
        self.steps += 1

    def finalize(self, day_idx: int, end_equity: float, target_pct: float) -> dict:
        pnl_pct = ((end_equity - self.open_equity) / self.open_equity * 100.0) if self.open_equity else 0.0
        # "passed" == hit the day's +target at SOME point (peak) AND never breached the wall — the same
        # condition the env/scoreboard use (peak reaching target == ChallengeState.target_hit latch).
        target_reached = self.peak >= self.open_equity * (1.0 + target_pct / 100.0)
        block_rate = (self.coerced / self.steps) if self.steps else 0.0
        # win_rate = % of discretionary closes that won (NET of cost); 0.0 when the day closed no trades.
        win_rate = (100.0 * self.wins / self.closes) if self.closes else 0.0
        return {"day": day_idx, "passed": bool(target_reached and not self.breached),
                "pnl_pct": round(pnl_pct, 2), "dd_pct": round(-self.max_dd_pct, 2),
                "breached": bool(self.breached), "trades": self.trades,
                "wins": self.wins, "closes": self.closes, "win_rate": round(win_rate, 1),
                "gate_block_rate": round(block_rate, 3)}


def run_pass(agent, data: Dict[str, "object"], overrides: Optional[dict], n_days: int,
             *, deterministic: bool = True, max_steps: Optional[int] = None,
             return_account: bool = False):
    """Run ONE Barbershop pass = N_DAYS on one continuous account; return a per-day scoreboard list
    whose dict schema matches the loop's print_pass_table/summarize_pass + registry PassRecord:
    {"day","passed","pnl_pct","dd_pct","breached","trades","wins","closes","win_rate","gate_block_rate"}
    (wins/closes/win_rate = winning DISCRETIONARY closes, count, and % won NET of cost — see _DayAccum).

    `agent` is a PPOAgent (untrained until M8 wires training; the metrics are REAL either way — they
    are what THIS policy actually did on the env, no placeholders). `deterministic` uses the live
    argmax policy (the Barbershop diagnoses a fixed policy).

    return_account [C14, 2026-06-19]: default False keeps the historical contract (returns `rows`
    only — every existing caller/test is unaffected). When True, returns `(rows, account)` where
    `account` is the CANONICAL end-of-run ChallengeState (the env is otherwise discarded here). The
    Barbershop notebook (Cell 6) reads `account.consecutive_loss_days` from it for the Policy Card —
    the live current streak, NOT a value recomputed from the day flags. COUPLING -> ftmo_passing/
    challenge_state.py (ChallengeState.consecutive_loss_days) + policy_registry build_card()."""
    env = build_env(data, overrides, n_days)
    obs = env.reset()
    ov = overrides or {}
    target_pct = float(ov.get("daily_target_pct", env.challenge_cfg.daily_target_pct))
    cap = max_steps if max_steps is not None else sum(len(d.close) for d in data.values()) + 8

    rows: List[dict] = []
    day = _DayAccum(env.account.equity)
    elapsed = 0
    done = False
    steps = 0
    while not done and steps < cap:
        sym = env.symbols[env.cursor]
        dm = env.direction_mask(sym)
        pm = build_pointer_mask([s.occupied for s in env.slots[sym]])
        ot, dt, pt = (torch.as_tensor(obs, dtype=torch.float32),
                      torch.as_tensor(dm, dtype=torch.float32),
                      torch.as_tensor(pm, dtype=torch.float32))
        if deterministic:
            a_dir, a_size, a_ptr, _ = agent.act_deterministic(ot, dt, pt)
        else:
            st = agent.act(ot, dt, pt)
            a_dir, a_size, a_ptr = st.a_direction, st.a_size, st.a_pointer
        obs, _reward, done, info = env.step((int(a_dir[0]), float(a_size[0]), int(a_ptr[0])))
        day.update(env, info)
        if info["days_elapsed"] > elapsed:        # a calendar day just ended (env already reset_day'd)
            rows.append(day.finalize(len(rows) + 1, env.account.equity, target_pct))
            elapsed = info["days_elapsed"]
            day = _DayAccum(env.account.equity)   # new day opens at the carried-forward equity
        steps += 1
    # finalize a trailing in-progress day (episode ended mid-day: blown account / end-of-data)
    if day.steps > 0 and len(rows) < n_days:
        rows.append(day.finalize(len(rows) + 1, env.account.equity, target_pct))
    if return_account:                      # C14: hand back the canonical end-of-run ChallengeState
        return rows, env.account            # (env is otherwise discarded) for the Policy Card streak
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# STANDING RULE [2026-06-19, operator] — applies to THIS file and EVERY file going forward: keep
# SHOWING THE WORK. On every edit (1) append a DATED IRAC entry here, and (2) in the code comments
# DOCUMENT the cross-file RELATIONSHIPS the change depends on (the COUPLING) — name the other file(s)
# and the exact attr/field/key relied on, in BOTH directions — and date the re-pointed logic, so any
# future reader/editor can see what connects to what, and what breaks where, and when it changed.
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-19] C21 — wire the Barbershop loop to a real TradingEnv (replace the DEMO stub).
#   I: barbershop_run_day() was a labelled placeholder, so the Policy Registry + Leaderboard recorded
#      fake per-day metrics — you couldn't tell which config actually passes.
#   R: Operator directive 2026-06-19 ("wire barbershop_run_day() to TradingEnv — the last piece")
#      + C10 (one continuous account over N_DAYS) + the deterministic-eval rule (SOW §2.10).
#   A: build_env() translates an OVERRIDES dict into a TradingEnv (make_challenge + RewardConfig +
#      training_wheels + episode_days; training_phase -> the global cfg.TRAINING_PHASE). run_pass()
#      steps the deterministic policy through the whole episode and splits it at the env's day
#      boundaries (info["days_elapsed"]) into REAL per-day rows; _DayAccum measures each day live
#      (P&L %, worst drawdown, breach, trades, gate-block rate). Rows match PassRecord's schema.
#   C: The Leaderboard now ranks policies on what they ACTUALLY did on the env — so the operator can
#      trust which configuration passes best and resume/promote it toward a consistent FTMO pass.
# [2026-06-19] C14/Fix 5 — run_pass can return the canonical end-of-run account (opt-in).
#   I: run_pass() built the env internally and discarded it, so the notebook could not reach the
#      canonical ChallengeState.consecutive_loss_days (Fix 4's runtime counter) to put on the Policy Card.
#   R: Operator audit Fix 5 (approved plan): expose the account WITHOUT breaking the existing return
#      contract; the card's consecutive_loss_days MUST be the live end-of-run value, not a recompute.
#   A: Added return_account=False (default unchanged -> rows only; all existing callers/tests intact).
#      When True, run_pass returns (rows, env.account) so the Barbershop loop reads
#      account.consecutive_loss_days and passes it to build_card(consecutive_loss_days=...). No other
#      behaviour changed. COUPLING -> ftmo_passing/challenge_state.py + policy_registry/registry.py.
#   C: The Policy Card now carries the policy's TRUE back-to-back loss streak from the run that produced
#      it — an honest, at-a-glance consistency signal, with zero disruption to existing eval paths.
# [2026-06-19] Risk-cap override — build_env honors max_per_trade_risk_frac from OVERRIDES.
#   I: Iterations showed the policy breaches the -4% wall almost every day; reward shaping (E8-tiny) and
#      20x more training did NOT help (worse: 19/20 breached). The breach is STRUCTURAL — 1%/trade x up
#      to 5 slots can exceed the 4% wall — but per-trade risk was not an OVERRIDE, so train (notebook)
#      and eval (run_pass->build_env) couldn't be capped consistently.
#   R: Operator option A — make per-trade risk a first-class, consistent override; ADD only (no locked
#      protocol touched). The invariant (total open risk <= remaining buffer) is unchanged; this just
#      shrinks the per-trade cap so a day can't lose more than ~N_SLOTS*frac before the wall.
#   A: build_env() now reads ov["max_per_trade_risk_frac"]; if present it builds RiskConfig with that cap
#      and passes risk_cfg to TradingEnv (None -> default 1%). So the same OVERRIDES value caps BOTH the
#      eval here and the notebook training env -> train/eval stay consistent. COUPLING -> runtime/config.py
#      RiskConfig + env/trading_env.py (risk_cfg -> RiskManager).
#   C: The operator can now cap position size from one knob, so a bad day physically can't blow past the
#      wall — the structural fix that keeps the policy ALIVE long enough for training to search for an edge.
# [2026-06-20] Per-day WIN RATE on the scoreboard + the cci_regime_gate override (reporting only).
#   I: The day-by-day scoreboard showed pnl/dd/trades but not how OFTEN the bot's trades won, so the
#      operator couldn't see whether a regime gate (or any change) improves trade selection vs just sizing.
#   R: Operator request 2026-06-20 — surface "% of trades won". Reporting-layer only (no env/locked-core
#      change): each CLOSE realizes one round-trip trade; a win is realized PnL NET of the close cost > 0.
#   A: _DayAccum tracks closes/wins from info["executed"]=="CLOSE" + (info["realized"]-info["cost"])>0 and
#      finalize() adds wins/closes/win_rate to the per-day row; run_pass schema doc updated. build_env also
#      honors the temporary OVERRIDES["cci_regime_gate"] (defaults to cfg.CCI_REGIME_GATE). COUPLING ->
#      env/trading_env.py (_apply_action CLOSE info keys realized/cost; cci_regime_gate) + tests _ROW_KEYS.
#   C: Every eval/gauge day now reports the policy's win rate alongside PnL, so the operator can judge
#      whether a guardrail improves the QUALITY of trades (win rate) — not only whether it survives the wall.
# [2026-06-20] RULE 2 — win rate counts EVERY realized close (all trades close by EOD).
#   I: The discretionary-only win rate showed 0/0 on days the bot opened but never closed itself (it rode
#      positions to the wall), so the column was blind exactly when it mattered.
#   R: Operator rule 2026-06-20: every trade must be CLOSED BY END OF DAY to count; count all realized
#      closes (discretionary + breach/target/EOD flatten). The env now flattens at EOD and surfaces every
#      realized close per step.
#   A: _DayAccum.update now sums info["n_closed"]/info["n_wins"] (env._step_closed_nets) instead of only
#      info["executed"]=="CLOSE"; closes == round-trip trades, wins == positive net of cost. COUPLING ->
#      env/trading_env.py (step() surfaces n_closed/n_wins; _advance_bar EOD _force_flatten).
#   C: The per-day win rate is now complete and honest over ALL of a day's trades — no 0/0 blind spots —
#      so the operator can actually read whether the policy's trades are winning.
