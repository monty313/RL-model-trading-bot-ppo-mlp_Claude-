"""C18 — Policy Registry (Policy Card + Leaderboard) tests.

Pins the committed contract in artifacts/policy_registry/README.md: auto-naming from the OVERRIDES
diff, the manifest/performance JSON schema, the compatibility resume-gate, and the pass-quality
Leaderboard ranking. Uses tmp_path so nothing touches the real (git-ignored) registry on disk.
"""

import json

import pytest

from quantra.runtime import config as cfg
from quantra.learning_system.trainer.trainer import TrainConfig
from quantra.learning_system.policy_registry.registry import (
    CompatibilityError, Leaderboard, PassRecord, PolicyCard, auto_name, build_card,
    check_compatibility, compatibility_signature, default_law_fingerprint, default_reward_layer_keys,
)


# ───────────────────────── C19: the OVERRIDES dict (config.build_overrides_dict) ──────────────────
def test_overrides_empty_when_everything_is_default():
    """(1) No knob differs from baseline -> empty OVERRIDES -> the baseline name."""
    ov = cfg.build_overrides_dict(challenge=cfg.ChallengeConfig(), reward=cfg.RewardConfig(),
                                  train=TrainConfig(), training_phase=cfg.TRAINING_PHASE,
                                  training_wheels=cfg.TRAINING_WHEELS)
    assert ov == {}
    assert auto_name(ov)[0] == "v1-baseline"


def test_overrides_single_knob_reflected_in_name():
    """(2) One changed knob -> exactly that key in OVERRIDES -> it shows in the name."""
    ov = cfg.build_overrides_dict(challenge=cfg.ChallengeConfig(daily_target_pct=3.0))
    assert ov == {"daily_target_pct": 3.0}
    assert auto_name(ov)[0] == "v1-tgt3"


def test_overrides_multiple_knobs_all_appear_in_name():
    """(3) Several changes across challenge/reward/train/phase/wheels -> all appear in the name."""
    ov = cfg.build_overrides_dict(
        challenge=cfg.ChallengeConfig(daily_target_pct=3.0),
        reward=cfg.RewardConfig(daily_progress_weight=2e-3),
        train=TrainConfig(seed=7),
        training_phase=cfg.PHASE_CONSTRAINED, training_wheels=False)
    assert ov == {"daily_target_pct": 3.0, "daily_progress_weight": 2e-3, "seed": 7,
                  "training_phase": "constrained", "training_wheels": False}
    name, basis = auto_name(ov)
    for tok in ("tgt3", "dailyprog0.002", "seed7", "constrained", "wheelsoff"):
        assert tok in name
    assert basis["wheel_state"] == "OFF"


def test_overrides_failed_day_penalty_not_double_counted():
    """failed_day_penalty lives in BOTH ChallengeConfig (authoritative) + RewardConfig (mirror);
    a change recorded once, from the challenge."""
    ov = cfg.build_overrides_dict(challenge=cfg.ChallengeConfig(failed_day_penalty=8.0),
                                  reward=cfg.RewardConfig(failed_day_penalty=8.0))
    assert ov == {"failed_day_penalty": 8.0}


def test_auto_name_baseline_is_v1_baseline():
    name, basis = auto_name({})
    assert name == "v1-baseline"
    assert basis["changes"] == [] and basis["wheel_state"] == "ON"


def test_auto_name_matches_readme_example():
    # README §1/§2 worked example: phase=constrained + wheels off, resumed from v1 -> v2-...
    name, basis = auto_name({"training_phase": "constrained", "training_wheels": False},
                            base_policy="v1-baseline")
    assert name == "v2-constrained-wheelsoff"
    assert basis["changes"] == ["training_phase=constrained", "training_wheels=OFF"]
    assert basis["wheel_state"] == "OFF"


def test_auto_name_unchanged_knobs_emit_no_token():
    name, basis = auto_name({"daily_target_pct": 2.5, "training_wheels": True})  # == baseline
    assert name == "v1-baseline" and basis["changes"] == []


def test_auto_name_numeric_token_and_version_increment():
    name, basis = auto_name({"daily_target_pct": 3.0}, base_policy="v3-foo")
    assert name == "v4-tgt3"                       # v3 -> v4; 3.0 -> "3"
    assert basis["changes"] == ["daily_target_pct=3"]


def test_compat_signature_ignores_weights_but_reacts_to_dim_and_layers():
    keys = ("L0", "L1", "L2", "L3", "L4", "L5_mult")
    base = compatibility_signature(207, keys, "lawfp")
    assert base.startswith("sha256:")
    assert compatibility_signature(207, keys, "lawfp") == base       # deterministic; weights not in sig
    assert compatibility_signature(189, keys, "lawfp") != base       # STATE_DIM change -> fresh start
    assert compatibility_signature(207, keys[:-1], "lawfp") != base  # layer arrangement change
    check_compatibility(base, base)                                   # match -> no raise
    with pytest.raises(CompatibilityError):
        check_compatibility(base, compatibility_signature(189, keys, "lawfp"), detail="207->189")


def test_default_fingerprints_track_live_modules():
    keys = default_reward_layer_keys()                # COUPLING -> reward.py decompose
    assert keys[0] == "L0" and "L5_mult" in keys and "total" not in keys
    assert keys == default_reward_layer_keys()        # stable
    assert len(default_law_fingerprint()) == 16       # COUPLING -> laws.LAW_NAMES


def test_policy_card_save_load_and_readme_schema(tmp_path):
    card = build_card(overrides={"training_phase": "constrained", "training_wheels": False},
                      state_dim=207, data_window={"start": "2023-03-01", "n_days": 8},
                      base_policy="v1-baseline")
    card.record_pass(PassRecord(1, days_passed=3, days_failed=5, avg_pnl=-0.4, avg_dd=-2.1, breach_count=2))
    card.record_pass(PassRecord(2, days_passed=6, days_failed=2, avg_pnl=0.8, avg_dd=-1.2, breach_count=0))
    d = card.save(root=tmp_path)

    man = json.loads((d / "manifest.json").read_text())
    perf = json.loads((d / "performance.json").read_text())
    assert (d / "compatibility.sig").read_text().startswith("sha256:")
    # manifest schema (README §2)
    assert man["policy_name"] == "v2-constrained-wheelsoff" and man["state_dim"] == 207
    assert man["training_phase"] == "constrained" and man["training_wheels"] is False
    assert man["base_policy"] == "v1-baseline" and man["n_passes_completed"] == 2
    assert man["auto_name_basis"]["wheel_state"] == "OFF"
    # performance schema (README §3) — 'pass' key, best_pass, overall_pass_rate
    assert set(perf) == {"pass_history", "best_pass", "overall_pass_rate"}
    assert perf["pass_history"][0]["pass"] == 1
    assert perf["best_pass"]["days_passed"] == 6
    assert abs(perf["overall_pass_rate"] - (3 + 6) / (3 + 5 + 6 + 2)) < 1e-12

    back = PolicyCard.load("v2-constrained-wheelsoff", root=tmp_path)
    assert back.n_passes_completed == 2 and back.best_pass()["days_passed"] == 6
    assert back.compatibility_signature == card.compatibility_signature


def test_leaderboard_ranks_by_pass_quality(tmp_path):
    good = build_card(overrides={"training_phase": "constrained"}, state_dim=207,
                      data_window={"start": "2023-01-01", "n_days": 8})
    good.record_pass(PassRecord(1, 7, 1, 1.0, -1.0, breach_count=0)); good.save(root=tmp_path)
    bad = build_card(overrides={"training_wheels": False}, state_dim=207,
                     data_window={"start": "2023-01-01", "n_days": 8})
    bad.record_pass(PassRecord(1, 2, 6, -1.0, -3.0, breach_count=3)); bad.save(root=tmp_path)

    board = Leaderboard.from_dir(root=tmp_path)
    rows = board.rows()
    assert len(rows) == 2
    assert rows[0].policy_name == good.policy_name              # 0.875 pass-rate beats 0.25
    assert rows[0].best_days_passed == 7 and rows[1].total_breaches == 3
    assert "pass%" in board.render()


def test_leaderboard_empty_and_skips_bad_folders(tmp_path):
    assert "empty" in Leaderboard.from_dir(root=tmp_path).render()
    junk = tmp_path / "not-a-policy"; junk.mkdir()
    (junk / "manifest.json").write_text("{ broken json")
    assert Leaderboard.from_dir(root=tmp_path).rows() == []     # corrupt manifest skipped, no crash
