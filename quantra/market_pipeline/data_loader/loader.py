"""MT5 1m bar loader — Drive / gdown / local CSV -> clean UTC bars + Parquet cache.

WHAT THIS MODULE DOES
---------------------
Resolves and parses the raw MT5 1-minute history exports for each symbol into a
clean, UTC-indexed OHLCV(+spread) DataFrame, and caches the result to Parquet so
later runs skip the (slow) CSV parse. Source resolution order:
    1. an explicit local path / a CSV already in ``data/raw``
    2. a Colab Google-Drive mount (``/content/drive/MyDrive/rl-trading-data``)
    3. ``gdown`` download by the registered Drive file ID (config.DRIVE_FILE_IDS)

The parser is format-tolerant: MT5 exports vary (tab vs comma vs semicolon; angle-
bracket headers like ``<DATE> <TIME> <OPEN>...``; combined or split date/time), so
we sniff the delimiter and map columns by normalized name rather than assuming one
layout.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The bot can only learn to pass the challenge it is actually shown. Faithful bars —
correct timestamps, real per-bar spread, no silent reordering — are the ground
truth the laws, costs, and risk walls all read. The real ``<SPREAD>`` column (when
present) feeds the Spread Filter law and the cost layer with true execution
friction, so a learned edge survives live costs. Parquet caching keeps iteration
fast and cheap (more windows/seeds validated per dollar).

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. If a diagnosis suspects
"Shortcut Learning" tied to a symbol/date, confirm the loader didn't introduce a
gap or duplicate timestamps first (data artifact != learned shortcut). The
``load_report`` returned alongside the frame records row counts, date span, and
dropped-duplicate counts so you can rule that out from evidence.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from quantra.runtime import config as cfg

# Canonical output columns. Every loaded symbol frame has exactly these, indexed
# by a tz-naive UTC DatetimeIndex named "time" (bar OPEN time, MT5 convention).
# COUPLING [C5] -> resampler/resampler.py (_AGG keys), feature_builder/builder.py, locked_core/cost_layer/costs.py, env/trading_env.py:
# these exact column names/order ("spread","tick_volume",...) are aggregated/read downstream; rename or reorder breaks all of them.
OHLCV_COLUMNS = ["open", "high", "low", "close", "tick_volume", "spread"]

# Synonyms we accept for each canonical field (normalized: lowercased, stripped of
# angle brackets / spaces / underscores). Order doesn't matter.
_COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "date": ["date", "dater", "gmt"],
    "time": ["time", "timestamp", "datetime"],
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c", "price"],
    # MT5 exports carry BOTH <TICKVOL> (tick volume) and <VOL> (real volume).
    # Keep them distinct so they don't collide onto one canonical column.
    "tick_volume": ["tickvol", "tickvolume", "tickv"],
    "real_volume": ["vol", "volume", "realvolume", "realvol"],
    "spread": ["spread", "spr"],
}


# COUPLING [C8] -> diagnostics/telemetry_logger/logger.py: these LoadReport field
# names/order are recorded into telemetry/run logs the LLM Risk Doctor reads; renaming a field breaks that contract.
@dataclass
class LoadReport:
    """Provenance of a loaded symbol frame — recorded into telemetry/run logs."""

    symbol: str
    source: str            # "local" | "drive_mount" | "gdown"
    raw_path: str
    rows: int
    start: str
    end: str
    dropped_duplicates: int
    delimiter: str
    had_spread: bool
    notes: List[str] = field(default_factory=list)


def _norm(name: str) -> str:
    return name.strip().lower().replace("<", "").replace(">", "").replace(" ", "").replace("_", "")


def _sniff_delimiter(sample: str) -> str:
    """Best-effort delimiter detection for an MT5 export sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        return dialect.delimiter
    except csv.Error:
        # Fall back to whichever candidate appears most on the first data line.
        first = sample.splitlines()[0] if sample else ""
        return max(["\t", ",", ";"], key=first.count)


def _map_columns(raw_cols: List[str]) -> Dict[str, str]:
    """Map raw header names -> canonical field names via the synonym table.

    Guarantees uniqueness: if two raw columns would map to the same canonical
    field, only the first wins and the second is left un-renamed (ignored). This
    prevents duplicate canonical columns (e.g. a stray second volume column) that
    would otherwise make ``df[col]`` a 2-D frame and corrupt parsing.
    """
    mapping: Dict[str, str] = {}
    used: set[str] = set()
    for raw in raw_cols:
        n = _norm(raw)
        for canon, syns in _COLUMN_SYNONYMS.items():
            if n in syns and canon not in used:
                mapping[raw] = canon
                used.add(canon)
                break
    return mapping


def parse_mt5_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse one MT5 export into a clean UTC-indexed OHLCV+spread frame.

    Returns ``(df, meta)`` where meta carries the detected delimiter, duplicate
    count, and whether a real spread column was present. Lookahead-safe by
    construction: rows are sorted by time and exact-duplicate timestamps dropped
    (keeping the first), so downstream resampling/feature math sees a clean,
    monotonic 1m series — a prerequisite for not learning a leakage-driven, false
    edge that would breach live.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(8192)
    delimiter = _sniff_delimiter(head)

    # Detect whether the first row is a header (non-numeric tokens) or data.
    first_line = head.splitlines()[0] if head else ""
    first_tokens = first_line.split(delimiter)
    has_header = any(_norm(t) in sum(_COLUMN_SYNONYMS.values(), []) for t in first_tokens)

    if has_header:
        df = pd.read_csv(path, sep=delimiter, dtype=str, engine="python")
        colmap = _map_columns(list(df.columns))
        df = df.rename(columns=colmap)
    else:
        # Headerless: assume the canonical MT5 order date,time,o,h,l,c,tickvol,vol,spread
        names = ["date", "time", "open", "high", "low", "close",
                 "tick_volume", "real_volume", "spread"]
        df = pd.read_csv(path, sep=delimiter, dtype=str, header=None,
                         names=names[: len(first_tokens)], engine="python")

    # --- build the timestamp ---
    if "date" in df.columns and "time" in df.columns:
        ts = df["date"].str.strip() + " " + df["time"].str.strip()
    elif "time" in df.columns:
        ts = df["time"].str.strip()
    elif "date" in df.columns:
        ts = df["date"].str.strip()
    else:
        raise ValueError(f"{path.name}: could not find a date/time column among {list(df.columns)}")

    # MT5 dates are usually YYYY.MM.DD; normalize separators so pandas infers cleanly.
    ts = ts.str.replace(".", "-", regex=False)
    index = pd.to_datetime(ts, errors="coerce", format="mixed")

    # Build numeric columns on df's integer index FIRST, then attach the parsed
    # datetime index positionally. Assigning a RangeIndex Series into a
    # DatetimeIndex frame would align-by-index and silently produce all-NaN.
    out = pd.DataFrame(index=df.index)
    for col in ["open", "high", "low", "close", "tick_volume", "spread"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = 0.0  # spread/tick_volume may be absent in some exports
    out.index = index  # positional attach (same length as df)

    out = out[~out.index.isna()]
    out.index.name = "time"
    out = out[out["close"].notna()]
    out = out.sort_index()
    before = len(out)
    out = out[~out.index.duplicated(keep="first")]
    dropped = before - len(out)

    had_spread = bool((out["spread"] != 0).any())
    # COUPLING (file-local) -> load_symbol below: these meta dict keys
    # ("delimiter","dropped_duplicates","had_spread") and the (df, meta) tuple order are unpacked there into LoadReport.
    meta = {"delimiter": delimiter, "dropped_duplicates": dropped, "had_spread": had_spread}
    return out[OHLCV_COLUMNS], meta


def _resolve_raw_path(symbol: str, explicit: Optional[Path]) -> tuple[Path, str]:
    """Find the raw CSV for ``symbol``; download via gdown if necessary.

    Returns ``(path, source_label)``. This is the single place that knows about
    Drive — keeping the parser pure and testable on local fixtures.
    """
    cfg.ensure_dirs()
    if explicit is not None:
        return Path(explicit), "local"

    # COUPLING [C5] -> runtime/config.py: DRIVE_FILENAMES/DRIVE_FILE_IDS/DRIVE_FOLDER_NAME
    # are per-symbol keyed by config.SYMBOLS; adding a symbol there requires entries in all those dicts or resolution fails here.
    filename = cfg.DRIVE_FILENAMES.get(symbol, f"{symbol}_M1.csv")

    # 1. already in data/raw, or any {symbol}_M1_*.csv there
    local = cfg.RAW_DIR / filename
    if local.exists():
        return local, "local"
    matches = sorted(cfg.RAW_DIR.glob(f"{symbol}_M1*.csv"))
    if matches:
        return matches[0], "local"

    # 2. Colab Drive mount
    mount = Path("/content/drive/MyDrive") / cfg.DRIVE_FOLDER_NAME / filename
    if mount.exists():
        return mount, "drive_mount"

    # 3. gdown by registered file ID
    file_id = cfg.DRIVE_FILE_IDS.get(symbol)
    if not file_id:
        raise FileNotFoundError(
            f"No raw CSV for {symbol} in {cfg.RAW_DIR}, no Drive mount, and no Drive "
            f"file ID registered. Provide a local path or mount Drive."
        )
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "gdown is required to fetch price data by Drive ID. "
            "`pip install gdown`, or mount Drive / pass a local path."
        ) from exc
    dest = cfg.RAW_DIR / filename
    gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest, "gdown"


# COUPLING -> feature_builder/builder.py (df_1m, _ = load_symbol(symbol)) + load_all below:
# the (df, LoadReport) return-tuple order is unpacked by those callers; flipping it silently swaps frame for report.
def load_symbol(
    symbol: str,
    path: Optional[Path] = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load one symbol's clean 1m bars (cached as Parquet) + a provenance report.

    Caching: the first call parses the CSV and writes ``data/parquet/{symbol}_1m.parquet``;
    subsequent calls read the Parquet (seconds vs minutes), which is the cheap-iteration
    behaviour that lets us run many walk-forward windows affordably.
    """
    cfg.ensure_dirs()
    cache = cfg.PARQUET_DIR / f"{symbol}_1m.parquet"
    if use_cache and path is None and cache.exists():
        df = pd.read_parquet(cache)
        rep = LoadReport(
            symbol=symbol, source="cache", raw_path=str(cache), rows=len(df),
            start=str(df.index.min()), end=str(df.index.max()),
            dropped_duplicates=0, delimiter="-", had_spread=bool((df["spread"] != 0).any()),
            notes=["loaded from Parquet cache"],
        )
        return df, rep

    raw_path, source = _resolve_raw_path(symbol, path)
    df, meta = parse_mt5_csv(raw_path)
    if use_cache and path is None:
        df.to_parquet(cache)
    rep = LoadReport(
        symbol=symbol, source=source, raw_path=str(raw_path), rows=len(df),
        start=str(df.index.min()), end=str(df.index.max()),
        dropped_duplicates=meta["dropped_duplicates"], delimiter=meta["delimiter"],
        had_spread=meta["had_spread"],
    )
    return df, rep


def load_all(symbols: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load every configured symbol. Used by the env/feature precompute (M2/M4)."""
    # COUPLING [C5] -> runtime/config.py: SYMBOLS is the canonical per-symbol list; the
    # cost/risk/env per-symbol dicts must stay keyed by the same set or load_all yields symbols those layers can't price.
    symbols = symbols or cfg.SYMBOLS
    return {s: load_symbol(s)[0] for s in symbols}


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13.
# Every change to this file APPENDS a dated IRAC entry below (newest last):
#   I (Issue) / R (Rule) / A (Application) / C (Conclusion -> why this makes the
#   bot pass FTMO MORE CONSISTENTLY, with no bug or inefficiency). The LLM Risk
#   Doctor reads this log to reconstruct the chronological 'why' when
#   triangulating a pass-rate regression. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] Format-tolerant, lookahead-free MT5 parse + cache.
#   I: MT5 exports vary; a wrong parse (dup stamps, NaN cols, index-misalignment) corrupts the bot's world.
#   R: Faithful, lookahead-free bars; real <SPREAD> feeds the Spread Filter law + the cost layer.
#   A: Delimiter/column sniff, unique-column guard, positional datetime attach, dedupe, Parquet cache; Section B tests.
#   C: The FTMO situation the bot trains on is faithful, so a learned pass-behaviour is real and transfers live.
