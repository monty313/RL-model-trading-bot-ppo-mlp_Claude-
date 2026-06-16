# ==========================================================================
# FILE: barbershop/adapter.py
# PURPOSE: Honest bridge between the REAL Quantra telemetry and the Barbershop
#          spec's data contract. The spec (Section 4) assumes idealised
#          trajectory.parquet / shap_values.parquet files; the live pipeline
#          instead writes artifacts/telemetry/<run_id>.jsonl (richer, JSONL,
#          with NO per-step GAE advantage and NO SHAP). This module maps what
#          IS produced onto the contract and marks — loudly and explicitly —
#          the fields the live pipeline does not yet emit, so the dashboard
#          never PRETENDS to show data it doesn't have.
# ==========================================================================
#
# DEPENDS ON:
#   artifacts/telemetry/<run_id>.jsonl  — real TelemetryLogger output
#   quantra.diagnostics.telemetry_logger.logger (loader, imported lazily)
#   quantra.locked_core.laws (LAW_NAMES, for the law_state mapping)
#
# PRODUCES (optional helper): data/prices_{1m,5m,30m,4h}.csv resampled from the
#   real 1m export, so Screen 3 can render real candles.
#
# WHAT IS NOT AVAILABLE FROM LIVE TELEMETRY (filled with NaN / empty + flagged):
#   - advantage      : GAE advantage is computed during PPO update, not logged
#                      per step. Mapped to NaN. (Add to StepPacket to enable.)
#   - SHAP values    : the MLPInterpreter emits visuals, not SHAP attributions.
#                      No shap_values file is produced yet. (Needs a SHAP pass.)
#   - regime label   : not in StepPacket; defaulted to "unlabelled".
#   - pass_result /  : derived here from the per-day 'day' packets if present,
#     dd_breached      else inferred from risk_context; flagged when inferred.
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Maps artifacts/telemetry JSONL onto the
#                            spec trajectory contract; flags advantage/SHAP/regime
#                            as not-yet-produced; 1m->multi-TF price resampler.
#   [2026-06-15] [Claude] — Adversarial-review fixes: key day packets in the same
#                            1-based id space the steps look up (regime/pass_result
#                            were silently lost); resample is in-memory by default
#                            (RULE 3 — no write outside logs/ unless opted in).
#   [2026-06-16] [Claude] — WI-1: header_feature_names() surfaces the real schema
#                            feature names so the dashboard labels the obs correctly.
#   [2026-06-16] [Claude] — WI-2: advantage now read from outcome.advantage (the
#                            producer logs REAL per-day GAE); NaN only when truly absent.
#   [2026-06-16] [Claude] — WI-3: load_attribution() reads the input-gradient sidecar
#                            into the SHAP columns; list_real_runs() excludes the
#                            *_attribution.jsonl sidecar so it's never read as a run.
# ==========================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from barbershop import config

# Fields the live telemetry does NOT yet produce — the dashboard + Risk Doctor
# surface this list so Monty knows exactly what is real vs placeholder on a real
# run (single source of truth in config so both modules agree).
NOT_YET_PRODUCED: List[str] = list(config.PLACEHOLDER_FIELDS)


def list_real_runs(telemetry_dir: Optional[Path] = None) -> List[Path]:
    """List real telemetry runs (artifacts/telemetry/*.jsonl), newest last.

    Reads: the telemetry directory. Returns: sorted list of .jsonl run files
    (empty if the directory doesn't exist — the caller falls back to mock data).
    """
    d = Path(telemetry_dir or config.REAL_TELEMETRY_DIR)
    if not d.exists():
        return []
    # Exclude sidecars (e.g. <run>_attribution.jsonl) so a sidecar is never mistaken
    # for a run — otherwise runs[-1] would point at the sidecar, not the run.
    return sorted(p for p in d.glob("*.jsonl") if not p.stem.endswith("_attribution"))


def load_attribution(run_path: Path) -> pd.DataFrame:
    """Load the input-gradient attribution sidecar for a real run (autopsy SHAP columns).

    Reads: <run>_attribution.jsonl next to the run file (written by the producer for each
    trade step). Returns a DataFrame in the SHAP contract shape
    (timestamp, day_id, step, chosen_action, shap_toward, shap_away), or an empty frame
    of those columns if no sidecar exists (older runs). The values are input x gradient
    saliencies — a legitimate attribution, NOT Shapley values (the UI says so).
    """
    import json
    p = Path(run_path)
    side = p.with_name(p.stem + "_attribution.jsonl")
    cols = ["timestamp", "day_id", "step", "chosen_action", "shap_toward", "shap_away"]
    if not side.exists():
        return pd.DataFrame(columns=cols)
    rows = [json.loads(ln) for ln in side.read_text(encoding="utf-8").splitlines() if ln.strip()]
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def header_feature_names(records: List[dict]) -> Optional[List[str]]:
    """Return the run's real observation feature names from the telemetry header.

    Reads: the loaded JSONL records. Returns the header's `feature_names` list (the
    real STATE_DIM-wide schema names) so the dashboard can label the observation
    correctly on a real run, or None if no header carries them.
    """
    for r in records:
        if r.get("kind") == "header" and r.get("feature_names"):
            return list(r["feature_names"])
    return None


def load_real_run(path: Path) -> List[dict]:
    """Load a real telemetry JSONL run as a list of record dicts.

    Reads: a <run_id>.jsonl file. Returns the parsed records (header + step +
    trade + day packets). Uses the canonical TelemetryLogger loader when quantra
    is importable, else a plain JSONL parse so this works standalone.
    """
    path = Path(path)
    try:
        from quantra.diagnostics.telemetry_logger.logger import TelemetryLogger
        return TelemetryLogger.load(path)
    except Exception:                                     # quantra not importable / other
        import json
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def _law_names() -> List[str]:
    """Return the 12 law/gate names (from quantra if available, else the mock list)."""
    try:
        from quantra.locked_core.laws import LAW_NAMES
        return list(LAW_NAMES)
    except Exception:
        from barbershop.data import MOCK_LAW_NAMES
        return list(MOCK_LAW_NAMES)


def real_to_trajectory(records: List[dict]) -> pd.DataFrame:
    """Map real telemetry records onto the spec trajectory contract (best effort).

    Reads: the loaded JSONL records. Returns a DataFrame with EVERY spec Section-4
    column. Fields the live pipeline does not emit are filled with NaN/defaults
    and listed in NOT_YET_PRODUCED so the UI can flag them. The mapping is:
      chosen_action(int) -> action(str)        via config.ENGINE_ACTION_INTS
      action_probs       -> all_probs          (reordered to [long,short,hold,close])
      legal_actions      -> masked_actions      (complement of legal)
      value              -> value_estimate
      risk_context       -> dd_buffer (trailing), pnl_cumulative
      reward_decomposition-> reward (sum)
      law_states(list)   -> law_state(dict)     via LAW_NAMES
      observation        -> obs_vector
    """
    steps = [r for r in records if r.get("kind") == "step"]
    # Key the day packets in the SAME 1-based id space the steps look up below
    # (steps use day_id = episode_id + 1). A packet may carry an explicit 1-based
    # day_id OR a 0-based episode_id; normalise both to the 1-based key so the
    # lookup never silently misses (which would blank regime/pass_result).
    days: Dict[int, dict] = {}
    for r in records:
        if r.get("kind") == "day":
            key = int(r["day_id"]) if "day_id" in r else int(r.get("episode_id", -1)) + 1
            days[key] = r
    law_names = _law_names()
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(steps):
        chosen_int = int(r.get("chosen_action", 0))
        action = config.ENGINE_ACTION_INTS.get(chosen_int, "HOLD")
        # Engine probs are indexed [HOLD,LONG,SHORT,CLOSE]; reorder to the contract
        # order [LONG,SHORT,HOLD,CLOSE].
        engine_probs = list(r.get("action_probs", [0, 0, 0, 0]))
        all_probs = _reorder_probs(engine_probs)
        legal = r.get("legal_actions", [])
        masked = _masked_from_legal(legal)
        risk = r.get("risk_context", {}) or {}
        law_states = r.get("law_states", [])
        law_state = {law_names[j]: ("ACTIVE" if (j < len(law_states) and law_states[j] != 0)
                                    else "BLOCKED")
                     for j in range(min(len(law_names), len(law_states)))}
        day_id = int(r.get("episode_id", 0)) + 1
        day_packet = days.get(day_id, {})
        rows.append({
            "timestamp": pd.to_datetime(r.get("timestamp"), utc=True, errors="coerce"),
            "day_id": day_id,
            "step": int(r.get("timestep", i)),
            "action": action,
            "action_prob": float(all_probs[config.ACTIONS.index(action)]) if all_probs else 0.0,
            "all_probs": all_probs,
            "masked_actions": masked,
            # REAL GAE advantage when the producer logged it (outcome.advantage); else NaN.
            "advantage": float((r.get("outcome", {}) or {}).get("advantage", float("nan"))),
            "value_estimate": float(r.get("value", float("nan"))),
            "reward": float(sum((r.get("reward_decomposition", {}) or {}).values())),
            "pnl_cumulative": float(risk.get("daily_pnl", risk.get("pnl", float("nan")))),
            "dd_buffer": float(risk.get("trailing_buffer", risk.get("trailing_dd", float("nan")))),
            "trade_open": bool(risk.get("position_open", False)),
            "trade_direction": str(risk.get("position_dir", "NONE")),
            "trade_pnl": float(risk.get("open_upnl", 0.0)),
            "law_state": law_state,
            "obs_vector": list(r.get("observation", [])),
            "regime": str(day_packet.get("regime", "unlabelled")),   # not in StepPacket
            "pass_result": bool(day_packet.get("pass_result", False)),
            "dd_breached": bool(day_packet.get("dd_breached", risk.get("breached", False))),
        })
    return pd.DataFrame(rows)


def _reorder_probs(engine_probs: List[float]) -> List[float]:
    """Reorder engine [HOLD,LONG,SHORT,CLOSE] probs to contract [LONG,SHORT,HOLD,CLOSE]."""
    if len(engine_probs) != 4:
        return [0.0, 0.0, 1.0, 0.0]                       # safe default: all HOLD
    hold, long_, short, close = engine_probs
    return [float(long_), float(short), float(hold), float(close)]


def _masked_from_legal(legal: List[Any]) -> List[str]:
    """Convert a legal-action list (ints or names) to the masked (illegal) names."""
    legal_names = set()
    for a in legal:
        if isinstance(a, int):
            legal_names.add(config.ENGINE_ACTION_INTS.get(a, ""))
        else:
            legal_names.add(str(a))
    return [a for a in config.ACTIONS if a not in legal_names]


def resample_prices_from_1m(src_1m_csv: Optional[Path] = None,
                            out_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Resample the real 1m MT5 export into per-TF OHLC frames for Screen 3.

    Reads: the 1m export (default data/raw/EURUSD_M1.csv) via the quantra loader +
    resampler when available, else a plain pandas OHLC resample. Returns a dict
    {tf: DataFrame} IN MEMORY by default — RULE 3 forbids this read-only tool from
    writing outside logs/. To cache on disk, pass an explicit out_dir (choose a
    logs/ subdir); the frames are then also written there as prices_{tf}.csv.
    """
    # Load 1m bars.
    try:
        from quantra.market_pipeline.data_loader import load_symbol
        df1, _ = load_symbol("EURUSD",
                             path=Path(src_1m_csv) if src_1m_csv else None)
    except Exception:
        src = Path(src_1m_csv or (config.DATA_DIR / "raw" / "EURUSD_M1.csv"))
        df1 = pd.read_csv(src)
        df1["timestamp"] = pd.to_datetime(df1.get("timestamp", df1.iloc[:, 0]), utc=True)
        df1 = df1.set_index("timestamp")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    freq = {"1m": "1min", "5m": "5min", "30m": "30min", "4H": "4h"}
    name = {"1m": "prices_1m.csv", "5m": "prices_5m.csv",
            "30m": "prices_30m.csv", "4H": "prices_4h.csv"}
    out: Dict[str, pd.DataFrame] = {}
    for tf, fr in freq.items():
        res = df1.resample(fr).agg(agg).dropna().reset_index()
        res = res.rename(columns={res.columns[0]: "timestamp"})
        out[tf] = res
        if out_dir is not None:                           # explicit opt-in disk cache only
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            res.to_csv(Path(out_dir) / name[tf], index=False)
    return out
