# ==========================================================================
# FILE: barbershop/data.py
# PURPOSE: The pure (Dash-free) data layer for the Barbershop dashboard.
#          Generates SYNTHETIC MOCK telemetry/price data (so the dashboard and
#          its tests run with no real training run), loads real files with
#          FAIL-LOUD missing-file errors (RULE 4), and provides every transform
#          the screens need: scoreboard sorting, timeframe windows, heatmap
#          colours, advantage alignment, trade autopsy grouping, SHAP sorting,
#          and the Pattern Finder. No Dash import here -> unit-testable.
# ==========================================================================
#
# DEPENDS ON: barbershop.config (paths, thresholds, colours, groups).
# PRODUCES:   logs/suggested_rules.json (export_rule) — the ONLY write target,
#             which is under logs/ as RULE 3 requires.
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Mock generators, fail-loud loaders,
#                            and all Screen 1-5 transforms (scoreboard, TF
#                            windows, heatmap colours, advantage, autopsy, SHAP,
#                            pattern finder, rule export).
#   [2026-06-15] [Claude] — Adversarial-review fixes: bank realised trade_pnl on
#                            the CLOSE bar (was lost), real SHAP explained-ratio,
#                            enrich losing_trades with atr_ratio + adv_neg_within_3
#                            so all pattern predicates work on real runs, mark
#                            dd_breached only on collapse bars, +__init__ docstring.
#   [2026-06-16] [Claude] — WI-1: losing_trades takes feature_names so the ATR lookup
#                            uses the REAL observation, not the 9-name mock fallback.
#   [2026-06-16] [Claude] — WI-4: detect_unavailable() reports which contract fields are
#                            genuinely absent (all-NaN/empty) so the UI greys them honestly.
# ==========================================================================

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from barbershop import config


# ==========================================================================
# FAIL-LOUD LOADING (RULE 4) — a missing required file raises a typed error
# naming WHICH file is missing and WHAT module produces it, so the dashboard
# can render a clear red banner instead of an empty chart.
# ==========================================================================
class MissingDataFile(Exception):
    """Raised when a required input file is absent. Carries the path + producer."""

    def __init__(self, path: Path, producer: str):
        """Store the missing path + its producer and build the fail-loud message."""
        self.path = Path(path)            # the file we looked for
        self.producer = producer          # the module/step that creates it
        super().__init__(
            f"Required file missing: {self.path}  —  produced by: {producer}"
        )


# Which module produces each file (shown in the fail-loud banner).
PRODUCERS: Dict[str, str] = {
    "trajectory.parquet": "quantra/diagnostics/telemetry_logger/logger.py (TelemetryLogger)",
    "shap_values.parquet": "quantra/diagnostics/mlp_interpreter/interpreter.py (MLPInterpreter)",
    "mlp_telemetry.parquet": "quantra/diagnostics/telemetry_logger/logger.py (TelemetryLogger)",
    "ppo_actor.pt": "quantra/learning_system/trainer/trainer.py (PPO training loop)",
    "prices_1m.csv": "data loader / MT5 export (data/raw/EURUSD_M1.csv)",
    "prices_5m.csv": "barbershop/adapter.py resample of the 1m export",
    "prices_30m.csv": "barbershop/adapter.py resample of the 1m export",
    "prices_4h.csv": "barbershop/adapter.py resample of the 1m export",
}


def _producer_for(path: Path) -> str:
    """Return the human producer string for a path (by filename)."""
    return PRODUCERS.get(Path(path).name, "unknown producer")


# ==========================================================================
# DATA CONTRACT — the columns the dashboard expects (spec Section 4).
# ==========================================================================
def required_trajectory_columns() -> List[str]:
    """Return the list of columns trajectory.parquet must contain (spec Section 4)."""
    return [
        "timestamp", "day_id", "step", "action", "action_prob", "all_probs",
        "masked_actions", "advantage", "value_estimate", "reward",
        "pnl_cumulative", "dd_buffer", "trade_open", "trade_direction",
        "trade_pnl", "law_state", "obs_vector", "regime", "pass_result",
        "dd_breached",
    ]


def validate_trajectory_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of REQUIRED columns missing from df (empty list == valid)."""
    return [c for c in required_trajectory_columns() if c not in df.columns]


def required_shap_columns() -> List[str]:
    """Columns shap_values.parquet must contain (spec Section 4)."""
    return ["timestamp", "day_id", "step", "chosen_action", "shap_toward", "shap_away"]


def detect_unavailable(trajectory: pd.DataFrame, shap: Optional[pd.DataFrame]) -> List[str]:
    """Return the contract fields that are GENUINELY absent on this run (all-NaN/empty).

    Reads: the mapped trajectory + shap frame. Returns the subset of
    {advantage, shap, regime} that has no real data, so the UI can grey those panels
    out HONESTLY (rather than rendering empty or fabricated bars). A field present on
    a run drops off the list automatically — no static assumptions.
    """
    out: List[str] = []
    if "advantage" in trajectory.columns and trajectory["advantage"].isna().all():
        out.append("advantage")
    if shap is None or len(shap) == 0:
        out.append("shap")
    if "regime" in trajectory.columns and (trajectory["regime"].astype(str) == "unlabelled").all():
        out.append("regime")
    return out


# A small, representative observation schema for mock data, named like the real
# Quantra schema (boll_*, cci*, atr_*) so group_indicators can categorise it.
MOCK_FEATURE_NAMES: List[str] = [
    "boll_bb20_up_5m", "boll_bb200_mid_30m", "ssma_align_5m",   # market structure
    "cci10_5m", "cci30_30m", "cci100_4H", "tw_cci_block",       # momentum
    "atr_level_1m", "atr_dev_30m",                              # volatility
]

# The 12 law/gate names (mirrors quantra.locked_core.laws.LAW_NAMES order) used
# for the mock law_state dict + the Laws & Gates heatmap rows.
MOCK_LAW_NAMES: List[str] = [
    "law_super_trend_bb", "law_super_trend_cci", "law_super_trend_ssma",
    "law_trend_bb", "law_trend_cci", "law_trend_ssma",
    "law_pullback_bb", "law_pullback_cci", "law_pullback_ssma",
    "gate_atr_liquidity", "gate_spread", "gate_stationarity",
]


# ==========================================================================
# MOCK GENERATORS — deterministic synthetic data so the dashboard + tests run
# with no real training run (spec Section 5: "use SYNTHETIC MOCK DATA").
# ==========================================================================
def make_mock_trajectory(seed: int = 0, days: int = 4) -> pd.DataFrame:
    """Build a deterministic 4-day mock trajectory DataFrame (spec data contract).

    Reads: nothing. Returns: a DataFrame with EVERY Section-4 column, covering
    `days` training days. Day outcomes are fixed so tests are stable:
      day 1 = PASS,  day 2 = FAIL + DD BREACHED,  day 3 = PASS,  day 4 = FAIL.
    Each day has a handful of steps including 2-3 trades (OPEN then CLOSE).
    """
    rng = np.random.default_rng(seed)
    # Fixed per-day outcomes: (regime, pass_result, dd_breached, day_pnl_pct).
    day_plan = [
        ("Trending", True, False, 2.8),    # day 1: passed
        ("News Day", False, True, -4.2),   # day 2: failed AND breached the wall
        ("Choppy", True, False, 2.6),      # day 3: passed
        ("Slow Day", False, False, 0.4),   # day 4: failed (never hit target, no breach)
    ][:days]

    rows: List[Dict[str, Any]] = []
    base_day = pd.Timestamp("2024-03-11 08:00:00", tz="UTC")   # a Monday
    steps_per_day = 14            # small but enough for context windows
    for di, (regime, passed, breached, day_pnl) in enumerate(day_plan, start=1):
        day_start = base_day + pd.Timedelta(days=di - 1)
        pnl = 0.0                 # running daily P&L in % of account
        dd_buffer = 1.0           # full trailing-DD buffer at day open
        # Decide which steps are trade opens/closes (deterministic per day).
        open_steps = {3, 8} if di != 2 else {2, 6, 10}   # the breach day trades more
        trade_open = False
        trade_dir = "NONE"
        pending_realized = 0.0    # the active trade's realised P&L, banked on CLOSE
        for s in range(steps_per_day):
            ts = day_start + pd.Timedelta(minutes=s * 7)   # 7-min spacing
            # Drift P&L toward the planned day result; the breach day dives late.
            pnl += day_pnl / steps_per_day + rng.normal(0, 0.05)
            collapse = breached and s >= steps_per_day - 4   # the late-day wall-hit bars
            if collapse:
                pnl -= 1.1         # the late-day collapse that hits the wall
            dd_buffer = float(np.clip(1.0 + min(pnl, 0.0) / config.DD_WALL_PCT, 0.0, 1.0))
            # Action selection: open on open_steps, close 2 steps later, else hold.
            # trade_pnl convention: 0 while flat / on the OPEN bar; the running
            # unrealised P&L while holding; and the REALISED result on the CLOSE bar
            # (so extract_trades + losing_trades, which read the CLOSE row, are correct).
            action = "HOLD"
            row_trade_pnl = 0.0
            if s in open_steps and not trade_open:
                action = "OPEN_LONG" if rng.random() > 0.5 else "OPEN_SHORT"
                trade_open = True
                trade_dir = "LONG" if action == "OPEN_LONG" else "SHORT"
                # Decide this trade's realised outcome now (negative-biased on breach days).
                pending_realized = float(rng.normal(-10 if breached else 7, 12))
            elif trade_open and (s - 2) in open_steps:
                action = "CLOSE"
                trade_open = False
                trade_dir = "NONE"
                row_trade_pnl = pending_realized          # bank the realised P&L on CLOSE
            elif trade_open:
                row_trade_pnl = float(pending_realized * 0.5)  # partial unrealised while held
            # Probabilities over [long, short, hold, close]; chosen action peaks.
            probs = _probs_for(action, rng)
            masked = _masked_for(action, trade_open)
            # Advantage: negative just before the breach day's losing closes.
            adv = float(rng.normal(0.1, 0.3))
            if breached and s >= steps_per_day - 5:
                adv = float(-abs(rng.normal(0.6, 0.2)))   # critic-beating turns sour
            rows.append({
                "timestamp": ts,
                "day_id": di,
                "step": s,
                "action": action,
                "action_prob": float(max(probs)),
                "all_probs": [float(p) for p in probs],
                "masked_actions": masked,
                "advantage": adv,
                "value_estimate": float(rng.normal(0.2, 0.2)),
                "reward": float(rng.normal(0.05, 0.2)),
                "pnl_cumulative": float(pnl),
                "dd_buffer": dd_buffer,
                "trade_open": bool(trade_open),
                "trade_direction": trade_dir,
                "trade_pnl": row_trade_pnl,
                "law_state": _mock_law_state(rng),
                "obs_vector": [float(x) for x in rng.normal(0, 1, len(MOCK_FEATURE_NAMES))],
                "regime": regime,
                "pass_result": bool(passed),
                # dd_breached marks ONLY the collapse bars (where the buffer hits the
                # wall), so the dashboard's breach line lands on the collapse, not
                # end-of-day. day_scoreboard still flags the day via .any().
                "dd_breached": bool(collapse),
            })
    return pd.DataFrame(rows)


def _probs_for(action: str, rng: np.random.Generator) -> np.ndarray:
    """Return a 4-vector of action probs [long, short, hold, close] peaking on `action`."""
    base = np.abs(rng.normal(0.1, 0.05, 4))        # small noise floor
    idx = config.ACTIONS.index(action)             # which action won
    base[idx] += 1.2                               # give the chosen action the mass
    return base / base.sum()                       # normalise to a distribution


def _masked_for(action: str, trade_open: bool) -> List[str]:
    """Which actions were illegal this step (mock mask consistent with position)."""
    if trade_open:
        return ["OPEN_LONG", "OPEN_SHORT"]   # already in a trade -> can't open another
    return ["CLOSE"]                          # flat -> nothing to close


def _mock_law_state(rng: np.random.Generator) -> Dict[str, str]:
    """A mock {law_name: 'ACTIVE'|'BLOCKED'} dict over the 12 laws/gates."""
    return {name: ("ACTIVE" if rng.random() > 0.5 else "BLOCKED") for name in MOCK_LAW_NAMES}


def make_mock_shap(seed: int = 0) -> pd.DataFrame:
    """Build mock SHAP attributions, one row per trade step in the mock trajectory.

    Reads: nothing. Returns: a DataFrame with the spec SHAP columns. `shap_toward`
    and `shap_away` are {indicator_name: value} dicts (values >= 0; larger = stronger).
    """
    traj = make_mock_trajectory(seed=seed)
    trades = traj[traj["action"].isin(["OPEN_LONG", "OPEN_SHORT", "CLOSE"])]
    rng = np.random.default_rng(seed + 7)
    rows: List[Dict[str, Any]] = []
    for _, r in trades.iterrows():
        toward = {n: float(abs(rng.normal(0.3, 0.15))) for n in MOCK_FEATURE_NAMES[:5]}
        away = {n: float(abs(rng.normal(0.2, 0.1))) for n in MOCK_FEATURE_NAMES[5:]}
        rows.append({
            "timestamp": r["timestamp"],
            "day_id": int(r["day_id"]),
            "step": int(r["step"]),
            "chosen_action": r["action"],
            "shap_toward": toward,
            "shap_away": away,
        })
    return pd.DataFrame(rows)


def make_mock_prices(seed: int = 0) -> Dict[str, pd.DataFrame]:
    """Build mock OHLC price frames for each timeframe, spanning all 4 mock days.

    Reads: nothing. Returns: {tf: DataFrame[timestamp, open, high, low, close, volume]}.
    Bars are a gentle random walk so candlesticks render and overlays have room.
    """
    out: Dict[str, pd.DataFrame] = {}
    start = pd.Timestamp("2024-03-08 00:00:00", tz="UTC")    # a few days before day 1
    end = pd.Timestamp("2024-03-16 00:00:00", tz="UTC")      # a few days after day 4
    freq_map = {"1m": "1min", "5m": "5min", "30m": "30min", "4H": "4h"}
    for tf, freq in freq_map.items():
        idx = pd.date_range(start, end, freq=freq, tz="UTC")
        rng = np.random.default_rng(seed + len(tf))          # per-TF determinism
        close = 1.10 + np.cumsum(rng.normal(0, 0.0004, len(idx)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        wig = np.abs(rng.normal(0, 0.0003, len(idx)))
        out[tf] = pd.DataFrame({
            "timestamp": idx,
            "open": open_,
            "high": np.maximum(open_, close) + wig,
            "low": np.minimum(open_, close) - wig,
            "close": close,
            "volume": rng.integers(10, 200, len(idx)).astype(float),
        })
    return out


def make_mock_losing_trades(n: int = 12, n_with_pattern: int = 8, seed: int = 0) -> pd.DataFrame:
    """Build mock losing trades for the Pattern Finder test (spec TEST 8).

    Reads: nothing. Returns a DataFrame of `n` losing trades where exactly
    `n_with_pattern` share the condition (dd_buffer < 0.25 AND atr_ratio > 1.3).
    The remaining trades have neither, so the combined condition has support
    exactly `n_with_pattern`/`n`.
    """
    rng = np.random.default_rng(seed + 3)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        has = i < n_with_pattern                       # first n_with_pattern carry the pattern
        dd = float(rng.uniform(0.05, 0.24)) if has else float(rng.uniform(0.40, 0.95))
        atr_ratio = float(rng.uniform(1.35, 2.0)) if has else float(rng.uniform(0.6, 1.1))
        rows.append({
            "trade_id": i + 1,
            "day_id": (i % 4) + 1,
            "dd_buffer": dd,
            "atr_ratio": atr_ratio,                    # ATR / daily-average ATR
            "advantage": float(-abs(rng.normal(0.4, 0.2))),   # losing -> advantage < 0
            "trade_pnl": float(-abs(rng.normal(20, 10))),     # closed at a loss
            "adv_neg_within_3": bool(rng.random() > 0.5),
        })
    return pd.DataFrame(rows)


# ==========================================================================
# PERSISTENCE HELPERS (tests write mock data then load it via the real loaders).
# ==========================================================================
def save_trajectory(df: pd.DataFrame, path: Path) -> Path:
    """Write a trajectory DataFrame to parquet (dict/list columns are JSON-encoded)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    enc = df.copy()
    for col in ("all_probs", "masked_actions", "law_state", "obs_vector"):
        if col in enc.columns:
            enc[col] = enc[col].apply(json.dumps)       # parquet-safe object columns
    enc.to_parquet(path)
    return path


def save_shap(df: pd.DataFrame, path: Path) -> Path:
    """Write a SHAP DataFrame to parquet (dict columns JSON-encoded)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    enc = df.copy()
    for col in ("shap_toward", "shap_away"):
        if col in enc.columns:
            enc[col] = enc[col].apply(json.dumps)
    enc.to_parquet(path)
    return path


def save_prices(prices: Dict[str, pd.DataFrame], data_dir: Path) -> Dict[str, Path]:
    """Write per-TF price frames to the spec CSV names under data_dir."""
    data_dir = Path(data_dir); data_dir.mkdir(parents=True, exist_ok=True)
    name = {"1m": "prices_1m.csv", "5m": "prices_5m.csv",
            "30m": "prices_30m.csv", "4H": "prices_4h.csv"}
    out = {}
    for tf, df in prices.items():
        p = data_dir / name[tf]
        df.to_csv(p, index=False)
        out[tf] = p
    return out


def load_trajectory(path: Optional[Path] = None) -> pd.DataFrame:
    """Load trajectory.parquet, decoding JSON columns. FAIL LOUD if missing (RULE 4)."""
    path = Path(path or config.TRAJECTORY_PARQUET)
    if not path.exists():
        raise MissingDataFile(path, _producer_for(path))
    df = pd.read_parquet(path)
    for col in ("all_probs", "masked_actions", "law_state", "obs_vector"):
        if col in df.columns and df[col].map(lambda v: isinstance(v, str)).any():
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)   # contract: UTC
    return df


def load_shap(path: Optional[Path] = None) -> pd.DataFrame:
    """Load shap_values.parquet, decoding JSON columns. FAIL LOUD if missing."""
    path = Path(path or config.SHAP_PARQUET)
    if not path.exists():
        raise MissingDataFile(path, _producer_for(path))
    df = pd.read_parquet(path)
    for col in ("shap_toward", "shap_away"):
        if col in df.columns and df[col].map(lambda v: isinstance(v, str)).any():
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_prices(tf: str, path: Optional[Path] = None) -> pd.DataFrame:
    """Load one timeframe's price CSV. FAIL LOUD if missing (RULE 4)."""
    path = Path(path or config.PRICE_CSVS[tf])
    if not path.exists():
        raise MissingDataFile(path, _producer_for(path))
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ==========================================================================
# SCREEN 2 — 4-DAY SCOREBOARD. Build per-day cards, sorted worst -> best
# (DD-breached first), per the spec.
# ==========================================================================
def day_scoreboard(trajectory: pd.DataFrame) -> List[Dict[str, Any]]:
    """Summarise each training day into a scoreboard card, worst day first.

    Reads: a trajectory DataFrame. Returns: a list of card dicts with keys
    day_id, regime, pnl_pct, passed, dd_status, n_trades — ordered so the most
    severe day (failed + DD breached) is first (spec Screen 2).
    """
    cards: List[Dict[str, Any]] = []
    for day_id, g in trajectory.groupby("day_id"):
        breached = bool(g["dd_breached"].any())
        passed = bool(g["pass_result"].iloc[0])
        # DD status from the worst (lowest) buffer touched that day.
        min_buffer = float(g["dd_buffer"].min())
        if breached:
            dd_status = "Breached"
        elif min_buffer < config.DD_BUFFER_RED_THRESHOLD:
            dd_status = "Warning"
        else:
            dd_status = "Safe"
        n_trades = int((g["action"] == "CLOSE").sum())   # one close == one completed trade
        cards.append({
            "day_id": int(day_id),
            "regime": str(g["regime"].iloc[0]),
            "pnl_pct": float(g["pnl_cumulative"].iloc[-1]),   # end-of-day running P&L
            "passed": passed,
            "dd_status": dd_status,
            "n_trades": n_trades,
        })
    # Severity sort: breached first, then failed, then by lowest P&L (worst first).
    severity = {"Breached": 0, "Warning": 1, "Safe": 2}
    cards.sort(key=lambda c: (severity[c["dd_status"]], c["passed"], c["pnl_pct"]))
    return cards


# ==========================================================================
# SCREEN 3 — TIMEFRAME WINDOWS. Exact context window per TF, centred on the
# trade entry time (spec TEST 3 / TEST 10).
# ==========================================================================
def timeframe_window(entry_time: pd.Timestamp, tf: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) = entry_time +/- the TF's context window.

    Reads: config.TF_CONTEXT_MINUTES. Returns: an exact (start, end) tuple so the
    candlestick x-axis can be pinned to it. Raises KeyError on an unknown tf.
    """
    entry_time = pd.Timestamp(entry_time)
    half = pd.Timedelta(minutes=config.TF_CONTEXT_MINUTES[tf])   # +/- half-window
    return entry_time - half, entry_time + half


# ==========================================================================
# SCREEN 3 PANEL 3 + SCREEN 4 LEFT — HEATMAP / STATE COLOURS. One colour rule
# per indicator category (spec TEST 5).
# ==========================================================================
def cell_color(category: str, name: str, value: Any, *,
               daily_atr_avg: Optional[float] = None) -> str:
    """Return the heatmap/bar colour for one indicator cell.

    Reads: config thresholds + colours. Returns a hex colour:
      GREEN  = bullish / favourable / active
      RED    = bearish / dangerous / blocked
      YELLOW = neutral / borderline
      GREY   = not applicable / missing
    Category-specific rules mirror the spec's colour language.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return config.COLOR_GREY                       # not applicable

    if category == "challenge_health":
        # DD buffer: lower = more dangerous (closer to the wall).
        if name == "dd_buffer":
            v = float(value)
            if v < config.DD_BUFFER_RED_THRESHOLD:
                return config.COLOR_RED                # < 25% buffer -> danger
            if v < config.DD_BUFFER_YELLOW_THRESHOLD:
                return config.COLOR_YELLOW             # 25-50% -> borderline
            return config.COLOR_GREEN                  # healthy cushion
        # Other health metrics: positive favourable, negative adverse.
        return _sign_color(float(value))

    if category == "laws_gates":
        s = str(value).upper()
        if s in ("ACTIVE", "1", "1.0", "+1"):
            return config.COLOR_GREEN                  # law/gate active -> favourable
        if s in ("BLOCKED", "-1", "-1.0"):
            return config.COLOR_RED                    # blocked -> direction forbidden
        return config.COLOR_YELLOW                     # inactive / neutral

    if category == "volatility":
        # ATR: only DANGEROUS when far above the day's average; otherwise neutral.
        v = float(value)
        if daily_atr_avg and daily_atr_avg > 0 and v > config.ATR_HIGH_MULT * daily_atr_avg:
            return config.COLOR_RED                    # volatility spike -> danger
        return config.COLOR_YELLOW                     # normal ATR -> neutral

    # market_structure / momentum and any other numeric: sign carries direction.
    try:
        return _sign_color(float(value))
    except (TypeError, ValueError):
        return config.COLOR_GREY


def _sign_color(v: float) -> str:
    """Green for positive, red for negative, yellow for ~zero."""
    if v > 1e-9:
        return config.COLOR_GREEN
    if v < -1e-9:
        return config.COLOR_RED
    return config.COLOR_YELLOW


def group_indicators(row: pd.Series,
                     feature_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Group one trajectory row's indicators into the 5 categories with colours.

    Reads: config.INDICATOR_GROUPS + the row's obs_vector/law_state/health fields.
    Returns: a list of {key, label, cells:[{name, value, color}]} in category order
    (used by the Screen-3 heatmap rows and the Screen-4 "what the bot saw" column).
    """
    names = feature_names or MOCK_FEATURE_NAMES
    obs = list(row.get("obs_vector", []))
    by_name = {n: (obs[i] if i < len(obs) else float("nan")) for i, n in enumerate(names)}

    groups: List[Dict[str, Any]] = []
    for key, label in config.INDICATOR_GROUPS:
        cells: List[Dict[str, Any]] = []
        if key == "challenge_health":
            health = [
                ("dd_buffer", float(row.get("dd_buffer", float("nan")))),
                ("day_progress", float(row.get("pnl_cumulative", 0.0))),
                ("trade_pnl", float(row.get("trade_pnl", 0.0))),
            ]
            for nm, val in health:
                cells.append({"name": nm, "value": val,
                              "color": cell_color(key, nm, val)})
        elif key == "laws_gates":
            law_state = row.get("law_state", {}) or {}
            for nm, val in law_state.items():
                cells.append({"name": nm, "value": val,
                              "color": cell_color(key, nm, val)})
        else:
            # market_structure (boll/ssma), momentum (cci), volatility (atr).
            for nm, val in by_name.items():
                if _category_of(nm) != key:
                    continue
                cells.append({"name": nm, "value": val,
                              "color": cell_color(key, nm, val)})
        groups.append({"key": key, "label": label, "cells": cells})
    return groups


def _category_of(feature_name: str) -> str:
    """Map a feature name to one of the 5 categories by its Quantra-schema prefix."""
    n = feature_name.lower()
    if n.startswith("boll_") or n.startswith("ssma_"):
        return "market_structure"
    if n.startswith("cci") or n.startswith("tw_cci"):
        return "momentum"
    if n.startswith("atr"):
        return "volatility"
    return "market_structure"        # default bucket for unrecognised market feats


# ==========================================================================
# SCREEN 3 PANEL 2 — ADVANTAGE STRIP. Time-aligned GAE advantage per 1m step.
# ==========================================================================
def extract_trades(trajectory: pd.DataFrame, day_id: int) -> List[Dict[str, Any]]:
    """Pair OPEN/CLOSE rows into trade records for the Screen-3 candle overlays.

    Reads: a trajectory + day id. Returns a list of
      {trade_id, entry_time, exit_time, direction, pnl, profit(bool)}.
    A trade spans from an OPEN row to the next CLOSE row; profit is taken from
    the CLOSE row's trade_pnl (>= 0 -> green shade, < 0 -> red shade).
    """
    g = trajectory[trajectory["day_id"] == day_id].sort_values("timestamp").reset_index(drop=True)
    trades: List[Dict[str, Any]] = []
    open_row: Optional[pd.Series] = None
    tid = 0
    for _, r in g.iterrows():
        if r["action"] in ("OPEN_LONG", "OPEN_SHORT") and open_row is None:
            open_row = r                                   # remember the entry
        elif r["action"] == "CLOSE" and open_row is not None:
            tid += 1
            pnl = float(r.get("trade_pnl", 0.0))
            trades.append({
                "trade_id": tid,
                "entry_time": open_row["timestamp"],
                "exit_time": r["timestamp"],
                "direction": "LONG" if open_row["action"] == "OPEN_LONG" else "SHORT",
                "pnl": pnl,
                "profit": pnl >= 0,
            })
            open_row = None
    return trades


def advantage_series(trajectory: pd.DataFrame, day_id: int) -> pd.DataFrame:
    """Return a day's [timestamp, advantage] frame, sorted by time.

    Reads: a trajectory DataFrame + a day id. Returns a 2-column frame whose
    timestamps are EXACTLY the candlestick step timestamps for that day, so the
    advantage strip lines up bar-for-bar with Panel 1 (spec TEST 4).
    """
    g = trajectory[trajectory["day_id"] == day_id].sort_values("timestamp")
    return g[["timestamp", "advantage"]].reset_index(drop=True)


# ==========================================================================
# SCREEN 4 — TRADE AUTOPSY. Action probabilities, masked/legal split, SHAP sort.
# ==========================================================================
def action_probability_bars(row: pd.Series) -> List[Dict[str, Any]]:
    """Return the 4 action-probability bars for the autopsy MIDDLE column.

    Reads: the row's all_probs + action. Returns a list of
    {action, prob, chosen, icon} in [long, short, hold, close] order; the bar
    for the chosen action has chosen=True (gold border in the UI).
    """
    probs = list(row.get("all_probs", [0, 0, 0, 0]))
    chosen = row.get("action", "HOLD")
    bars = []
    for i, act in enumerate(config.ACTIONS):
        bars.append({
            "action": act,
            "prob": float(probs[i]) if i < len(probs) else 0.0,
            "chosen": act == chosen,
            "icon": config.ACTION_ICONS[act],
        })
    return bars


def masked_legal(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Return (masked_actions, legal_actions) for the autopsy mask label.

    Reads: the row's masked_actions. Returns the two lists so the UI can warn
    about mask dependence (a high prob on the only legal action means nothing).
    """
    masked = list(row.get("masked_actions", []))
    legal = [a for a in config.ACTIONS if a not in masked]
    return masked, legal


def shap_sorted(shap_row: pd.Series, chosen_action: str,
                top_k: int = 5) -> Dict[str, Any]:
    """Sort SHAP contributors into toward/away groups for the autopsy RIGHT column.

    Reads: a shap row's shap_toward / shap_away dicts. Returns
    {toward:[(name,val)], away:[(name,val)], explained: float} with each list
    sorted largest-first and truncated to top_k (spec TEST 7).
    """
    toward_all = sorted(dict(shap_row.get("shap_toward", {})).items(),
                        key=lambda kv: kv[1], reverse=True)
    away_all = sorted(dict(shap_row.get("shap_away", {})).items(),
                      key=lambda kv: kv[1], reverse=True)
    toward, away = toward_all[:top_k], away_all[:top_k]
    # REAL explained fraction: how much of the TOTAL attribution magnitude is carried
    # by the shown top-k bars (so truncating to top_k genuinely lowers the percentage).
    total = sum(v for _, v in toward_all) + sum(v for _, v in away_all)
    shown = sum(v for _, v in toward) + sum(v for _, v in away)
    explained = 0.0 if total <= 0 else 100.0 * shown / total
    return {"chosen_action": chosen_action, "toward": toward, "away": away,
            "explained": explained}


# ==========================================================================
# SCREEN 5 — PATTERN FINDER. Scan losing trades, surface the top shared
# conditions, write plain-English rules (spec TEST 8).
# ==========================================================================
# Each candidate condition: (key, plain-English description, predicate, specificity).
# `specificity` breaks ties so a 2-factor pattern outranks a 1-factor one at
# equal support (the more specific rule is the more useful prescription).
_PATTERN_CONDITIONS = [
    ("dd_low_and_atr_high",
     "DD Wall buffer was below 25% AND ATR was above 1.3x the day's average",
     lambda r: r.get("dd_buffer", 1.0) < config.DD_BUFFER_RED_THRESHOLD
     and r.get("atr_ratio", 0.0) > config.ATR_HIGH_MULT, 2),
    ("dd_low",
     "DD Wall buffer was below 25%",
     lambda r: r.get("dd_buffer", 1.0) < config.DD_BUFFER_RED_THRESHOLD, 1),
    ("atr_high",
     "ATR was above 1.3x the day's average",
     lambda r: r.get("atr_ratio", 0.0) > config.ATR_HIGH_MULT, 1),
    ("adv_neg",
     "Advantage went negative within 3 bars of entry",
     lambda r: bool(r.get("adv_neg_within_3", False)), 1),
]

# Plain-English suggested prescription per condition key.
_PATTERN_RULES = {
    "dd_low_and_atr_high":
        "Increase the penalty for OPEN when DD buffer < 25% AND ATR > 1.3x the daily average.",
    "dd_low":
        "Increase the penalty for OPEN when DD buffer < 25%.",
    "atr_high":
        "Increase the penalty for OPEN when ATR > 1.3x the daily average.",
    "adv_neg":
        "Add a reward shaping term that discourages entries whose advantage turns negative within 3 bars.",
}


def losing_trades(trajectory: pd.DataFrame,
                  feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Extract losing trades (advantage < 0 AND trade closed at a loss), ENRICHED.

    Reads: a trajectory DataFrame + the run's real `feature_names` (so the ATR lookup
    indexes the REAL observation, not the 9-name mock fallback). Returns the CLOSE
    rows that lost money while the critic was beaten (advantage < 0), each enriched
    with the derived predicate columns the Pattern Finder needs so ALL conditions
    work on REAL trajectories (not just the mock-loser fixture):
      - atr_ratio        = the bar's ATR feature / that day's mean ATR feature
      - adv_neg_within_3 = did advantage go negative within 3 steps after entry
    Missing ingredients degrade gracefully (atr_ratio defaults to 1.0).
    """
    names = feature_names or MOCK_FEATURE_NAMES
    closes = trajectory[(trajectory["action"] == "CLOSE")
                        & (trajectory["advantage"] < 0)
                        & (trajectory["trade_pnl"] < 0)].copy()
    if closes.empty:
        return closes.reset_index(drop=True)

    # Locate an ATR-like feature in the REAL observation to build atr_ratio.
    atr_idx = next((i for i, n in enumerate(names) if n.startswith("atr")), None)

    def _atr_value(row) -> float:
        """Return |the row's ATR observation feature| (NaN if no ATR feature exists)."""
        obs = list(row.get("obs_vector", []))
        return abs(float(obs[atr_idx])) if (atr_idx is not None and atr_idx < len(obs)) else float("nan")

    # Per-day mean ATR feature (the denominator for atr_ratio).
    day_atr_mean: Dict[int, float] = {}
    for day_id, g in trajectory.groupby("day_id"):
        vals = [_atr_value(r) for _, r in g.iterrows()]
        vals = [v for v in vals if v == v]                # drop NaN
        day_atr_mean[int(day_id)] = (sum(vals) / len(vals)) if vals else 1.0

    atr_ratios, adv_flags = [], []
    for _, close in closes.iterrows():
        day_id = int(close["day_id"])
        mean = day_atr_mean.get(day_id, 1.0) or 1.0
        a = _atr_value(close)
        atr_ratios.append((a / mean) if (a == a and mean) else 1.0)
        # advantage in the 3 steps AFTER this trade's entry (its matching OPEN).
        adv_flags.append(_advantage_dipped_after_entry(trajectory, close))
    closes["atr_ratio"] = atr_ratios
    closes["adv_neg_within_3"] = adv_flags
    return closes.reset_index(drop=True)


def _advantage_dipped_after_entry(trajectory: pd.DataFrame, close_row: pd.Series) -> bool:
    """True if advantage went negative within 3 steps after this trade's entry step."""
    day = trajectory[trajectory["day_id"] == close_row["day_id"]].sort_values("step")
    trades = extract_trades(trajectory, int(close_row["day_id"]))
    # Match this CLOSE to its trade by exit timestamp, then read its entry step.
    match = next((t for t in trades if t["exit_time"] == close_row["timestamp"]), None)
    if match is None:
        return False
    entry_rows = day[day["timestamp"] == match["entry_time"]]
    if entry_rows.empty:
        return False
    entry_step = int(entry_rows.iloc[0]["step"])
    window = day[(day["step"] > entry_step) & (day["step"] <= entry_step + 3)]
    return bool((window["advantage"] < 0).any())


def find_patterns(losing: pd.DataFrame, max_patterns: int = 3) -> List[Dict[str, Any]]:
    """Find the top shared conditions among losing trades (spec Screen 5 / TEST 8).

    Reads: a losing-trades DataFrame (needs dd_buffer/atr_ratio/adv_neg_within_3).
    Returns: up to `max_patterns` pattern dicts, each
      {rank, count, total, key, conditions:[...], plain_english, suggested_rule}
    ranked by support (count) then specificity. count/total let the UI say
    "Found in 8 of 12 losing trades".
    """
    total = len(losing)
    scored = []
    for key, desc, pred, spec in _PATTERN_CONDITIONS:
        count = int(sum(bool(pred(r)) for _, r in losing.iterrows()))   # support
        if count == 0:
            continue
        scored.append({"key": key, "count": count, "total": total,
                       "conditions": desc.split(" AND "), "plain_english": desc,
                       "suggested_rule": _PATTERN_RULES[key], "_spec": spec})
    # Rank by support desc, then specificity desc (more factors win ties).
    scored.sort(key=lambda p: (p["count"], p["_spec"]), reverse=True)
    out = []
    for rank, p in enumerate(scored[:max_patterns], start=1):
        p["rank"] = rank
        p.pop("_spec", None)
        out.append(p)
    return out


# ==========================================================================
# EXPORT — write an approved pattern/prescription to logs/suggested_rules.json.
# This is the ONLY write the dashboard performs, and it is under logs/ (RULE 3).
# ==========================================================================
def export_rule(entry: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Append one rule entry to suggested_rules.json (creating it as a JSON array).

    Reads: the existing file if present. Writes: the updated array under logs/.
    Returns: the file path. Used by Screen-5 [APPLY RULE] and the Doctor's
    [APPROVE] button — both stamp a UTC timestamp + a `source`.
    """
    path = Path(path or config.SUGGESTED_RULES_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = [existing]
        except json.JSONDecodeError:
            existing = []                       # corrupt file -> start fresh array
    entry = dict(entry)
    entry.setdefault("exported_at", datetime.now(timezone.utc).isoformat())
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path
