# ==========================================================================
# FILE: barbershop/test_dashboard.py
# PURPOSE: Tests 1-10 for the Barbershop dashboard (spec Section 5). All tests
#          use SYNTHETIC MOCK DATA (no real training run). They exercise the
#          pure data/figure logic behind every screen so the suite runs fast and
#          offline. Run: pytest barbershop/test_dashboard.py
# ==========================================================================
#
# UPDATE LOG — every change to this file must be logged here:
#   [2026-06-15] [Claude] — First build. Tests 1-10: data loading, scoreboard,
#                            timeframe windows, advantage alignment, heatmap
#                            colours, autopsy panels, SHAP, pattern finder,
#                            missing-file banner, higher-TF vertical line.
#   [2026-06-15] [Claude] — Review fixes: de-vacuum TESTs 4/5/7/8 (exact window,
#                            ATR danger branch, real SHAP truncation, specificity
#                            tiebreak); added plateau-banner + write-path-wiring tests.
#   [2026-06-16] [Claude] — Added test_quantra_source_loads_real_run: the real
#                            Quantra-telemetry path is detected + mapped + flagged.
# ==========================================================================

from __future__ import annotations

import json

import pandas as pd
import pytest

from barbershop import config, data, dashboard, figures


# --------------------------------------------------------------------------
# TEST 1 — Data loading: trajectory + price CSVs load, columns present, UTC.
# --------------------------------------------------------------------------
def test_mock_bundle_memoised_and_screen1_honest():
    """WI-8 — mock bundle built once; Screen 1 labels demo vs real honestly."""
    dashboard._MOCK_BUNDLE = None                          # reset the cache for this test
    b1 = dashboard.load_bundle(source="mock")
    b2 = dashboard.load_bundle(source="mock")
    assert b1 is b2                                        # memoised — not rebuilt every call
    assert b1["training_wall_real"] is False
    # A demo (non-real) curve shows the honest demo note, not a bare "live" claim.
    s = str(dashboard.screen_training_wall(b1))
    assert "Demo curve" in s
    # A bundle WITH a real pass-rate series is labelled real (no demo note).
    real_b = dict(b1); real_b["training_wall_real"] = True
    assert "real training run" in str(dashboard.screen_training_wall(real_b))


def test_adapter_loads_passrate_sidecar(tmp_path):
    """WI-8 — a real pass-rate sidecar loads; its absence returns None (-> demo curve)."""
    from barbershop import adapter
    run = tmp_path / "r.jsonl"
    run.write_text("{}", encoding="utf-8")
    assert adapter.load_passrate(run) is None             # no sidecar -> demo
    (tmp_path / "r_passrate.json").write_text(
        json.dumps({"iterations": [0, 1], "pass_rate": [10, 20]}), encoding="utf-8")
    pr = adapter.load_passrate(run)
    assert pr["iterations"] == [0, 1] and pr["pass_rate"] == [10.0, 20.0]


def test_contract_is_single_source_of_truth():
    """WI-7 — config + data reference the one contract module (no drift between them)."""
    from barbershop import contract
    assert config.ACTIONS is contract.ACTIONS
    assert config.ENGINE_ACTION_INTS is contract.ENGINE_ACTION_INTS
    assert config.PLACEHOLDER_FIELDS is contract.PLACEHOLDER_FIELDS
    assert data.required_trajectory_columns() == contract.TRAJECTORY_COLUMNS
    assert data.required_shap_columns() == contract.SHAP_COLUMNS


def test_data_loading_and_columns_and_utc(barbershop_tmp, mock_trajectory, mock_prices):
    """TEST 1 — Data loading: trajectory + price CSVs load, columns present, UTC."""
    traj_path = data.save_trajectory(mock_trajectory, config.DATA_DIR / "trajectory.parquet")
    price_paths = data.save_prices(mock_prices, config.DATA_DIR)

    loaded = data.load_trajectory(traj_path)                  # loads without error
    assert data.validate_trajectory_columns(loaded) == []     # every contract column present
    assert str(loaded["timestamp"].dt.tz) == "UTC"            # timestamps parse as UTC

    for tf, p in price_paths.items():
        px = data.load_prices(tf, p)
        assert {"open", "high", "low", "close"}.issubset(px.columns)
        assert str(px["timestamp"].dt.tz) == "UTC"


# --------------------------------------------------------------------------
# TEST 2 — Day scoreboard: 4 cards, correct PASS/FAIL, worst (breached) first.
# --------------------------------------------------------------------------
def test_day_scoreboard_renders_and_sorts_worst_first(mock_trajectory):
    """TEST 2 — Day scoreboard: 4 cards, correct PASS/FAIL, worst (breached) first."""
    cards = data.day_scoreboard(mock_trajectory)
    assert len(cards) == 4                                     # one card per training day
    assert cards[0]["dd_status"] == "Breached"                # worst day sorted first
    # The breached day is a FAIL; at least one PASS exists among the four.
    assert cards[0]["passed"] is False
    assert any(c["passed"] for c in cards)
    # Severity order is non-increasing (breached -> warning -> safe).
    sev = {"Breached": 0, "Warning": 1, "Safe": 2}
    order = [sev[c["dd_status"]] for c in cards]
    assert order == sorted(order)


# --------------------------------------------------------------------------
# TEST 3 — Timeframe switching: entry +/- exact window per TF.
# --------------------------------------------------------------------------
def test_timeframe_windows_are_exact():
    """TEST 3 — Timeframe switching: entry +/- exact window per TF."""
    entry = pd.Timestamp("2024-03-14 09:42", tz="UTC")
    expect = {"1m": 30, "5m": 120, "30m": 720, "4H": 5 * 24 * 60}   # minutes each side
    for tf, minutes in expect.items():
        start, end = data.timeframe_window(entry, tf)
        assert end - entry == pd.Timedelta(minutes=minutes)
        assert entry - start == pd.Timedelta(minutes=minutes)


# --------------------------------------------------------------------------
# TEST 4 — Advantage strip alignment: Panel 2 x-axis matches Panel 1 exactly.
# --------------------------------------------------------------------------
def test_advantage_strip_aligns_with_candles(mock_trajectory, mock_prices):
    """TEST 4 — Advantage strip alignment: Panel 2 x-axis matches Panel 1 exactly."""
    day_id = 2
    trades = data.extract_trades(mock_trajectory, day_id)
    entry = trades[0]["entry_time"]
    window = data.timeframe_window(entry, "1m")
    candle = figures.candlestick_figure(mock_prices["1m"], trades, "1m", entry, window=window)
    adv = data.advantage_series(mock_trajectory, day_id)
    adv_fig = figures.advantage_figure(adv, window=window)
    # Each panel is pinned to the EXACT expected window (not just equal to each other).
    assert [pd.Timestamp(x) for x in candle.layout.xaxis.range] == [window[0], window[1]]
    assert [pd.Timestamp(x) for x in adv_fig.layout.xaxis.range] == [window[0], window[1]]
    # Every advantage timestamp inside the window exists in the 1m candle series
    # (and there IS at least one, so the subset check isn't vacuously true).
    candle_times = set(pd.to_datetime(mock_prices["1m"]["timestamp"], utc=True))
    in_window = adv[(adv["timestamp"] >= window[0]) & (adv["timestamp"] <= window[1])]
    assert len(in_window) > 0
    assert set(in_window["timestamp"]).issubset(candle_times)


# --------------------------------------------------------------------------
# TEST 5 — Indicator heatmap colours.
# --------------------------------------------------------------------------
def test_heatmap_colour_assignment():
    """TEST 5 — Indicator heatmap colours."""
    assert data.cell_color("challenge_health", "dd_buffer", 0.15) == config.COLOR_RED
    assert data.cell_color("volatility", "atr_level_1m", 0.0003) == config.COLOR_YELLOW
    assert data.cell_color("laws_gates", "law_super_trend_bb", "ACTIVE") == config.COLOR_GREEN
    # Exercise the volatility DANGER branch the heatmap exists to flag: a >1.3x ATR
    # spike vs the day average is RED; just below threshold stays YELLOW.
    assert data.cell_color("volatility", "atr_level_1m", 0.0010, daily_atr_avg=0.0005) == config.COLOR_RED
    assert data.cell_color("volatility", "atr_level_1m", 0.0006, daily_atr_avg=0.0005) == config.COLOR_YELLOW
    # DD buffer just below the green threshold is YELLOW (borderline), healthy is GREEN.
    assert data.cell_color("challenge_health", "dd_buffer", 0.40) == config.COLOR_YELLOW
    assert data.cell_color("challenge_health", "dd_buffer", 0.80) == config.COLOR_GREEN


# --------------------------------------------------------------------------
# TEST 6 — Trade autopsy panels.
# --------------------------------------------------------------------------
def test_trade_autopsy_panels(mock_trajectory):
    """TEST 6 — Trade autopsy panels."""
    day_id = 2
    trades = data.extract_trades(mock_trajectory, day_id)
    assert len(trades) >= 3                                    # trade ③ exists on this day
    tr = trades[2]                                            # the third trade
    row = mock_trajectory[(mock_trajectory["day_id"] == day_id)
                          & (mock_trajectory["timestamp"] == tr["entry_time"])].iloc[0]
    # LEFT — 5 indicator groups.
    groups = data.group_indicators(row)
    assert len(groups) == 5
    # MIDDLE — 4 probability bars summing to ~1, exactly one chosen (gold border).
    bars = data.action_probability_bars(row)
    assert len(bars) == 4
    assert abs(sum(b["prob"] for b in bars) - 1.0) < 1e-6
    assert sum(1 for b in bars if b["chosen"]) == 1
    # Masked actions are available to label.
    masked, legal = data.masked_legal(row)
    assert set(masked).issubset(set(config.ACTIONS))
    assert set(masked).isdisjoint(set(legal))


# --------------------------------------------------------------------------
# TEST 7 — SHAP panel: toward green desc, away red desc, explained variance.
# --------------------------------------------------------------------------
def test_shap_panel_sorted_and_grouped():
    """TEST 7 — SHAP panel: toward green desc, away red desc, explained variance."""
    shap_row = pd.Series({
        "shap_toward": {"cci10_5m": 0.5, "boll_bb20_up_5m": 0.3, "ssma_align_5m": 0.1},
        "shap_away": {"atr_level_1m": 0.4, "tw_cci_block": 0.2},
    })
    s = data.shap_sorted(shap_row, "OPEN_SHORT")               # top_k=5 -> all shown
    assert len(s["toward"]) == 3 and len(s["away"]) == 2
    toward_vals = [v for _, v in s["toward"]]
    away_vals = [v for _, v in s["away"]]
    assert toward_vals == sorted(toward_vals, reverse=True)    # toward sorted descending
    assert away_vals == sorted(away_vals, reverse=True)        # away sorted descending
    # explained is a REAL fraction: all 5 contributors shown -> 100%; truncating to
    # top_k=2 must genuinely LOWER it below 100% (not a hardcoded stub).
    assert abs(s["explained"] - 100.0) < 1e-6
    s2 = data.shap_sorted(shap_row, "OPEN_SHORT", top_k=2)
    assert 0 < s2["explained"] < 100.0
    fig = figures.shap_figure(s)                               # renders without error
    assert len(fig.data) == 2                                  # toward + away groups


# --------------------------------------------------------------------------
# TEST 8 — Pattern finder: 8/12 shared condition -> Pattern 1; APPLY exports JSON.
# --------------------------------------------------------------------------
def test_pattern_finder_detects_and_exports(barbershop_tmp):
    """TEST 8 — Pattern finder: 8/12 shared condition -> Pattern 1; APPLY exports JSON."""
    losing = data.make_mock_losing_trades(n=12, n_with_pattern=8)
    patterns = data.find_patterns(losing)
    assert patterns[0]["count"] == 8 and patterns[0]["total"] == 12
    # The specificity tiebreak must rank the 2-factor pattern first (most useful),
    # not a 1-factor pattern that ties on count.
    assert patterns[0]["key"] == "dd_low_and_atr_high"
    assert len(patterns[0]["conditions"]) == 2
    assert "DD" in patterns[0]["suggested_rule"] and "ATR" in patterns[0]["suggested_rule"]
    # [APPLY RULE] exports to logs/suggested_rules.json.
    out = data.export_rule({"source": "pattern_finder", **patterns[0]})
    assert out.exists()
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(saved, list) and saved[-1]["count"] == 8


# --------------------------------------------------------------------------
# TEST 9 — Missing file error: red banner names the file + producer, no crash.
# --------------------------------------------------------------------------
def test_missing_file_shows_fail_loud_banner(barbershop_tmp):
    """TEST 9 — Missing file error: red banner names the file + producer, no crash."""
    missing = config.DATA_DIR / "trajectory.parquet"          # never created
    with pytest.raises(data.MissingDataFile) as exc_info:
        data.load_trajectory(missing)
    banner = dashboard.error_banner(exc_info.value)
    text = str(banner)                                        # flatten the component tree
    assert "trajectory.parquet" in text                       # names WHICH file
    assert "TelemetryLogger" in text                          # names WHAT produces it


# --------------------------------------------------------------------------
# TEST 10 — Higher-TF vertical entry line at the exact entry timestamp.
# --------------------------------------------------------------------------
def test_higher_tf_vertical_entry_line(mock_trajectory, mock_prices):
    """TEST 10 — Higher-TF vertical entry line at the exact entry timestamp."""
    day_id = 2
    trades = data.extract_trades(mock_trajectory, day_id)
    entry = trades[0]["entry_time"]
    for tf in ("5m", "30m", "4H"):
        window = data.timeframe_window(entry, tf)
        fig = figures.candlestick_figure(mock_prices[tf], trades, tf, entry, window=window)
        line_shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert line_shapes, f"no entry vline on {tf}"
        # The line sits at the exact entry timestamp.
        assert pd.Timestamp(line_shapes[0].x0) == entry
    # On 1m there is no entry vline (spec: higher-TFs only).
    win1 = data.timeframe_window(entry, "1m")
    fig1 = figures.candlestick_figure(mock_prices["1m"], trades, "1m", entry, window=win1)
    assert not [s for s in fig1.layout.shapes if s.type == "line"]


# --------------------------------------------------------------------------
# EXTRA — Screen 1 plateau banner actually renders on a flat curve.
# --------------------------------------------------------------------------
def test_plateau_banner_renders_on_flat_curve():
    """EXTRA — Screen 1 plateau banner actually renders on a flat curve."""
    assert figures.is_plateaued([81.0, 81.2, 80.9, 81.1]) is True     # flat -> plateau
    assert figures.is_plateaued([10.0, 40.0, 70.0, 95.0]) is False    # rising -> not
    # The shipped mock curve is flattened at the tail, so the banner is present.
    bundle = dashboard.load_bundle(use_mock=True)
    screen = dashboard.screen_training_wall(bundle)
    assert "plateau-banner" in str(screen)
    assert "plateau detected" in str(screen).lower()


# --------------------------------------------------------------------------
# EXTRA — the sanctioned write-path callbacks (APPLY / APPROVE) are actually wired.
# (The review found these buttons had NO callbacks — the only write path was dead.)
# --------------------------------------------------------------------------
def test_write_path_callbacks_are_registered():
    """EXTRA — the sanctioned write-path callbacks (APPLY / APPROVE) are actually wired."""
    app = dashboard.make_app(use_mock=True)
    keys = " ".join(app.callback_map.keys())
    assert "pattern-export-status" in keys      # [APPLY RULE] -> export_rule is wired
    assert "doctor-approve-status" in keys       # [APPROVE] -> approve_prescription is wired


def test_enter_key_sends_to_doctor():
    """EXTRA — the chat input's Enter key (n_submit) is wired to the send callback (spec)."""
    app = dashboard.make_app(use_mock=True)
    # Find a callback that takes doctor-input.n_submit as an Input (Enter-to-send).
    found = False
    for spec in app.callback_map.values():
        for inp in spec.get("inputs", []):
            if inp.get("id") == "doctor-input" and inp.get("property") == "n_submit":
                found = True
    assert found, "Enter key (doctor-input.n_submit) is not wired to send"


# --------------------------------------------------------------------------
# EXTRA — the REAL-data path: a Quantra telemetry JSONL run is auto-detected,
# mapped onto the contract by the adapter, and its not-yet-produced fields flagged.
# --------------------------------------------------------------------------
def test_quantra_source_loads_real_run(tmp_path, monkeypatch):
    """EXTRA — real Quantra telemetry is detected, mapped (regime/pass), placeholders flagged."""
    tdir = tmp_path / "telemetry"; tdir.mkdir()
    run = tdir / "run1.jsonl"
    records = [
        {"kind": "header", "schema_version": "1.0.0"},
        {"kind": "day", "episode_id": 0, "regime": "Trending",
         "pass_result": True, "dd_breached": False},
        {"kind": "step", "episode_id": 0, "timestep": 0, "timestamp": "2024-03-11T08:00:00Z",
         "chosen_action": 1, "action_probs": [0.1, 0.7, 0.1, 0.1], "legal_actions": [0, 1, 2],
         "value": 0.2, "observation": [0.0, 0.1, -0.2],
         "law_states": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
         "risk_context": {"trailing_buffer": 0.8, "daily_pnl": 1.2},
         "reward_decomposition": {"l0": 0.1}},
    ]
    run.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    # Point the adapter at our synthetic run dir; avoid reading the real 1m export.
    monkeypatch.setattr(config, "REAL_TELEMETRY_DIR", tdir)
    monkeypatch.setattr(dashboard.adapter, "resample_prices_from_1m",
                        lambda *a, **k: data.make_mock_prices())
    dashboard._QUANTRA_CACHE.clear()
    # Auto-detection prefers the real run.
    assert dashboard.available_source() == "quantra"
    bundle = dashboard.load_bundle(source="quantra")
    assert bundle["source"] == "quantra"
    # The day packet's regime + pass_result were mapped onto the step row (the bug the
    # review caught — episode_id 0-based vs day_id 1-based — stays fixed here).
    assert list(bundle["trajectory"]["regime"]) == ["Trending"]
    assert bool(bundle["trajectory"]["pass_result"].iloc[0]) is True
    # SHAP isn't produced by the live pipeline yet -> empty, and the placeholders are flagged.
    assert bundle["shap"].empty
    assert "advantage" in bundle["unavailable_fields"]


# --------------------------------------------------------------------------
# WI-1 — group_indicators uses the REAL feature names (not the 9-name mock set)
# so the "What the bot SAW" panel + heatmap are correct on real 203-feature data.
# --------------------------------------------------------------------------
def test_group_indicators_uses_real_feature_names():
    """WI-1 — real feature names label + categorise the obs correctly (not 9 mock names)."""
    feature_names = ["boll_bb20_up_5m", "boll_bb200_mid_30m", "ssma_align_5m",
                     "cci10_5m", "cci30_30m", "cci100_4H", "tw_cci_block",
                     "atr_level_1m", "atr_dev_30m", "z10_1m", "adx5_1m"]
    row = pd.Series({"obs_vector": [0.5, -0.3, 0.0, 1.2, -0.4, 0.1, 1.0, 0.6, -0.2, 0.3, 0.7],
                     "law_state": {"market_volatility_obs": "ACTIVE"}, "dd_buffer": 0.8,
                     "pnl_cumulative": 1.0, "trade_pnl": 0.0})
    groups = data.group_indicators(row, feature_names)
    by_key = {g["key"]: [c["name"] for c in g["cells"]] for g in groups}
    assert "cci10_5m" in by_key["momentum"]                 # cci -> momentum
    assert "boll_bb20_up_5m" in by_key["market_structure"]  # boll -> market structure
    assert "atr_level_1m" in by_key["volatility"]           # atr -> volatility
    # All 11 real market features are represented (not truncated to the 9 mock names).
    total = sum(len(by_key[k]) for k in ("market_structure", "momentum", "volatility"))
    assert total == 11
    # The mock fallback still works when no names are supplied.
    row2 = pd.Series({"obs_vector": [0.0] * len(data.MOCK_FEATURE_NAMES),
                      "law_state": {}, "dd_buffer": 0.8, "pnl_cumulative": 0.0, "trade_pnl": 0.0})
    assert data.group_indicators(row2)                      # doesn't crash


# --------------------------------------------------------------------------
# WI-2 — the adapter reads REAL GAE advantage that the producer logged in
# outcome.advantage (so the advantage strip is real, not empty, on real runs).
# --------------------------------------------------------------------------
def test_adapter_reads_real_advantage_from_outcome():
    """WI-2 — adapter maps outcome.advantage onto the contract advantage column."""
    from barbershop import adapter
    base = dict(chosen_action=0, action_probs=[1, 0, 0, 0], legal_actions=[0], value=0.1,
                observation=[0.0], law_states=[0] * 12, risk_context={}, reward_decomposition={})
    records = [
        {"kind": "header"},
        {"kind": "step", "episode_id": 0, "timestep": 0, "timestamp": "2024-03-11T08:00:00Z",
         "outcome": {"advantage": 0.42}, **base},
        {"kind": "step", "episode_id": 0, "timestep": 1, "timestamp": "2024-03-11T08:01:00Z",
         "outcome": {"advantage": -0.17}, **base},
    ]
    df = adapter.real_to_trajectory(records)
    assert list(df["advantage"]) == [0.42, -0.17]           # real, both signs, not NaN
    # And a step with no logged advantage stays NaN (honest absence).
    df2 = adapter.real_to_trajectory([{"kind": "step", "episode_id": 0, "timestep": 0, **base}])
    assert df2["advantage"].isna().all()


# --------------------------------------------------------------------------
# WI-3 — the input-gradient attribution sidecar loads into the SHAP contract
# columns (so the autopsy RIGHT column is real on real runs, not empty/fake).
# --------------------------------------------------------------------------
def test_adapter_loads_attribution_sidecar(tmp_path):
    """WI-3 — attribution sidecar -> SHAP contract columns; missing sidecar -> empty frame."""
    from barbershop import adapter
    run = tmp_path / "run1.jsonl"
    run.write_text('{"kind":"header"}\n', encoding="utf-8")
    side = tmp_path / "run1_attribution.jsonl"
    side.write_text(json.dumps({"timestamp": "2024-03-11T08:00:00Z", "day_id": 1, "step": 3,
                                "chosen_action": "OPEN_SHORT",
                                "shap_toward": {"cci10_5m": 0.5},
                                "shap_away": {"atr_level_1m": 0.2}}) + "\n", encoding="utf-8")
    df = adapter.load_attribution(run)
    assert len(df) == 1 and df.iloc[0]["chosen_action"] == "OPEN_SHORT"
    assert df.iloc[0]["shap_toward"] == {"cci10_5m": 0.5}
    # No sidecar -> empty frame with the right columns (older runs degrade cleanly).
    empty = adapter.load_attribution(tmp_path / "nope.jsonl")
    assert empty.empty and "shap_toward" in empty.columns


# --------------------------------------------------------------------------
# WI-4 — unavailable_fields is computed DYNAMICALLY so panels grey out honestly
# only when the data is genuinely absent (present fields drop off the list).
# --------------------------------------------------------------------------
def test_detect_unavailable_is_dynamic():
    """WI-4 — present advantage/shap/regime -> nothing flagged; absent -> all flagged."""
    import numpy as np
    have = pd.DataFrame({"advantage": [0.1, -0.2], "regime": ["Trending", "Trending"]})
    assert data.detect_unavailable(have, pd.DataFrame({"shap_toward": [{}]})) == []
    miss = pd.DataFrame({"advantage": [np.nan, np.nan], "regime": ["unlabelled", "unlabelled"]})
    assert set(data.detect_unavailable(miss, pd.DataFrame())) == {"advantage", "shap", "regime"}
    # The greyed panel says "not available" instead of rendering empty/fake bars.
    panel = str(dashboard._unavailable_panel("Advantage strip", "no GAE logged"))
    assert "not available" in panel
