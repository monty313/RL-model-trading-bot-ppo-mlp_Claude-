"""PolicyCard + Leaderboard — the Policy Registry's read/write engine.

WHAT THIS MODULE DOES
---------------------
Gives every trained policy a saved, human-readable IDENTITY and a way to rank them:
  * PolicyCard  -> writes/reads the 3 files per policy (manifest.json / performance.json /
                   compatibility.sig) under artifacts/policy_registry/<policy_name>/.
  * auto_name() -> derives <policy_name> from the run's OVERRIDES diff vs the baseline config
                   (NEVER hand-typed — README §1: e.g. "v2-constrained-wheelsoff").
  * compatibility_signature()/check_compatibility() -> the resume gate (README §4): a hash of
                   state_dim + reward-layer SHAPE + law fingerprint; mismatch -> CompatibilityError.
  * Leaderboard -> ranks every registry entry by passing quality (the real scoreboard: pass-rate
                   first, then breaches — README §3), for the operator / dashboard.

This is the exact contract documented in artifacts/policy_registry/README.md (the only committed
file under artifacts/). The registry CONTENTS are git-ignored + run-specific; this code is the
machinery the Barbershop notebook (Cell 6) and the trainer call to produce them.

🔴 The auto-name is ALWAYS generated from the OVERRIDES diff — never invented by hand or by an LLM.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from quantra.runtime import config as cfg

# COUPLING -> locked_core/laws/laws.py (LAW_NAMES) + learning_system/reward_engine/reward.py
# (RewardEngine/RewardContext). Imported lazily in the default_* helpers below so importing the
# registry never drags in heavy modules unless a caller actually wants the project defaults.

MANIFEST_FILE = "manifest.json"
PERFORMANCE_FILE = "performance.json"
COMPAT_FILE = "compatibility.sig"


# ───────────────────────────── auto-naming (README §1) ──────────────────────────────
# Short, stable name tokens per knob, in the ORDER they appear in a policy name. COUPLING ->
# runtime/config.py (ChallengeConfig/RewardConfig field names) + PROJECT_GUIDE §4.12 OVERRIDES: a
# key here must match the OVERRIDES key the trainer injects, or the diff silently ignores it.
_NAME_ORDER: Tuple[str, ...] = (
    "training_phase", "training_wheels", "daily_target_pct", "daily_risk_pct", "permanent_dd_pct",
    "ftmo_mode", "stop_for_day", "net_pnl_weight", "step_pnl_weight", "daily_progress_weight",
    "drawdown_pain_weight", "drawdown_pain_steepness", "trade_quality_weight", "failed_day_penalty",
    # TrainConfig knobs (config.build_overrides_dict diffs these too) — kept LAST so the "perspective"
    # knobs (phase/wheels/challenge/reward) lead the name and optimization details trail it.
    "rollout_size", "minibatch", "value_coef", "g8_lookahead", "seed",
)
_SHORT: Dict[str, str] = {
    "training_wheels": "wheels", "daily_target_pct": "tgt", "daily_risk_pct": "risk",
    "permanent_dd_pct": "permdd", "ftmo_mode": "ftmo", "stop_for_day": "stopday",
    "net_pnl_weight": "netpnl", "step_pnl_weight": "steppnl", "daily_progress_weight": "dailyprog",
    "drawdown_pain_weight": "pain", "drawdown_pain_steepness": "paink", "trade_quality_weight": "tradeq",
    "failed_day_penalty": "failday",
    "rollout_size": "roll", "minibatch": "mb", "value_coef": "vcoef", "g8_lookahead": "g8look",
    "seed": "seed",
}


def baseline_overrides() -> Dict[str, object]:
    """The project's DEFAULT knob set — the thing a run's OVERRIDES is diffed against to auto-name.
    COUPLING -> runtime/config.py: pulls the live defaults (ChallengeConfig + RewardConfig + the
    TRAINING_* module defaults) so the baseline always tracks the real config, not a stale copy."""
    ch = cfg.ChallengeConfig()
    rw = cfg.RewardConfig()
    return {
        "training_phase": "free" if cfg.TRAINING_PHASE == cfg.PHASE_FREE else "constrained",
        "training_wheels": bool(cfg.TRAINING_WHEELS),
        "daily_target_pct": ch.daily_target_pct, "daily_risk_pct": ch.daily_risk_pct,
        "permanent_dd_pct": ch.permanent_dd_pct, "ftmo_mode": ch.ftmo_mode,
        "stop_for_day": ch.stop_for_day,
        "net_pnl_weight": rw.net_pnl_weight, "step_pnl_weight": rw.step_pnl_weight,
        "daily_progress_weight": rw.daily_progress_weight, "drawdown_pain_weight": rw.drawdown_pain_weight,
        "drawdown_pain_steepness": rw.drawdown_pain_steepness, "trade_quality_weight": rw.trade_quality_weight,
        "failed_day_penalty": rw.failed_day_penalty,
    }


def _numfmt(v: float) -> str:
    return f"{v:g}"   # 3.0 -> "3", 2.5 -> "2.5", 0.002 -> "0.002"


def _token_for(key: str, value: object) -> Tuple[str, str]:
    """Return (name_token, plain_change) for a single changed knob. README §1 token style."""
    if key == "training_phase":                       # value IS the token (e.g. "constrained")
        return str(value), f"training_phase={value}"
    if isinstance(value, bool):                       # wheels/ftmo/stopday -> "<short>off"/"<short>on"
        short = _SHORT.get(key, key)
        return f"{short}{'on' if value else 'off'}", f"{key}={'ON' if value else 'OFF'}"
    short = _SHORT.get(key, re.sub(r'[^a-z0-9]', '', key.lower()))
    return f"{short}{_numfmt(value)}", f"{key}={_numfmt(value)}"   # numeric


def _next_version(base_policy: Optional[str]) -> int:
    """v<N> increments from the base_policy it resumed from; fresh (None) starts at v1 (README §1)."""
    if base_policy:
        m = re.match(r"v(\d+)", str(base_policy))
        if m:
            return int(m.group(1)) + 1
    return 1


def auto_name(overrides: Dict[str, object], *, baseline: Optional[Dict[str, object]] = None,
              base_policy: Optional[str] = None) -> Tuple[str, Dict[str, object]]:
    """Derive (policy_name, auto_name_basis) from the OVERRIDES diff vs the baseline (README §1).
    NEVER hand-typed. Deterministic: tokens are emitted in _NAME_ORDER. No changes -> 'v<N>-baseline'."""
    base = baseline if baseline is not None else baseline_overrides()
    tokens: List[str] = []
    changes: List[str] = []
    keys = list(_NAME_ORDER) + sorted(k for k in overrides if k not in _NAME_ORDER)
    for key in keys:
        if key not in overrides:
            continue
        value = overrides[key]
        if key in base and value == base[key]:
            continue                                  # unchanged vs baseline -> no token
        tok, chg = _token_for(key, value)
        tokens.append(tok)
        changes.append(chg)
    n = _next_version(base_policy)
    name = f"v{n}-" + "-".join(tokens) if tokens else f"v{n}-baseline"
    wheels_eff = bool(overrides.get("training_wheels", base.get("training_wheels", True)))
    basis = {"changes": changes, "wheel_state": "OFF" if not wheels_eff else "ON"}
    return name, basis


# ─────────────────────── compatibility signature (README §4) ─────────────────────────
# 📍 COMPATIBILITY MAP (the "change-one-file, fix-the-others" web — full table in
# artifacts/policy_registry/README.md §4). The signature has THREE inputs, each owned by one file;
# the two default_* helpers below read the LIVE values so the signature tracks code automatically:
#   1. state_dim            <- market_pipeline/feature_builder/schema.py STATE_DIM   (caller passes it)
#   2. reward layer keys    <- learning_system/reward_engine/reward.py decompose()   (default_reward_layer_keys)
#   3. law fingerprint      <- locked_core/laws/laws.py LAW_NAMES                     (default_law_fingerprint)
# Change any one -> EVERY saved policy's signature changes -> RESUME raises CompatibilityError and the
# policy must be RETRAINED.
# ✅ SAFE TO CHANGE without a fresh start (NOT in the signature): training_phase, training_wheels, the
#    challenge numbers, and ALL reward weights (C16) + term math (C17) — same dim + L-keys + laws, so an
#    old policy still resumes. Shape the policy with those; only touch the three inputs above on purpose.
class CompatibilityError(RuntimeError):
    """Raised on RESUME_FROM when the saved signature != the current config+OVERRIDES (README §4)."""


def default_reward_layer_keys() -> Tuple[str, ...]:
    """The reward LAYER arrangement = the 'L*' keys of decompose() (NOT weights/derived totals).
    COUPLING -> learning_system/reward_engine/reward.py: derived live from RewardEngine().decompose()
    so adding/removing a layer auto-changes the signature, while a C16/C17 weight/math re-point (same
    L0..L5_mult keys) keeps it stable -> old policies stay resume-safe across those edits."""
    from quantra.learning_system.reward_engine.reward import RewardContext, RewardEngine
    keys = RewardEngine().decompose(RewardContext(net_pnl_delta=0.0)).keys()
    return tuple(k for k in keys if k.startswith("L"))


def default_law_fingerprint() -> str:
    """Fingerprint the law set. COUPLING -> locked_core/laws/laws.py: hashes LAW_NAMES (the 9
    directional laws + 3 market signals); reorder/rename there -> signature changes -> fresh start."""
    from quantra.locked_core.laws.laws import LAW_NAMES
    return hashlib.sha256(json.dumps(list(LAW_NAMES)).encode()).hexdigest()[:16]


def compatibility_signature(state_dim: int, reward_layer_keys: Sequence[str],
                            law_fingerprint: str) -> str:
    """sha256 over state_dim + reward-layer shape + law fingerprint (README §4). 'Shape' = the layer
    KEYS, not the weights — so operator-tunable weights (C16) and re-pointed math (C17) do NOT change
    it, but changing STATE_DIM or the layer arrangement does."""
    payload = json.dumps({"state_dim": int(state_dim),
                          "reward_layer_keys": list(reward_layer_keys),
                          "law_fingerprint": law_fingerprint}, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def check_compatibility(saved_sig: str, current_sig: str, *, detail: str = "") -> None:
    """Pass silently when signatures match; else raise CompatibilityError with a plain-English reason
    (README §4: 'STATE_DIM changed 207 -> 189 ...'). The caller saves the old checkpoint first and
    never overwrites it — old policies are only ever superseded."""
    if saved_sig != current_sig:
        why = f" ({detail})" if detail else ""
        raise CompatibilityError(
            f"incompatible policy: saved {saved_sig} != current {current_sig}{why}. The old network's "
            f"shape no longer fits — start fresh (the old checkpoint is kept, never overwritten).")


# ──────────────────────────── performance records (README §3) ────────────────────────
@dataclass(frozen=True)
class PassRecord:
    """One pass over the N_DAYS window. Keys match performance.json (README §3); 'pass' is the
    1-based pass index (a reserved word, so the field is pass_n and serializes to 'pass')."""

    pass_n: int
    days_passed: int
    days_failed: int
    avg_pnl: float
    avg_dd: float
    breach_count: int
    avg_gate_block_rate: float = 0.0   # ~0 in PHASE_FREE; >0 only in PHASE_CONSTRAINED (README §3)

    def to_dict(self) -> Dict[str, object]:
        return {"pass": self.pass_n, "days_passed": self.days_passed, "days_failed": self.days_failed,
                "avg_pnl": self.avg_pnl, "avg_dd": self.avg_dd, "breach_count": self.breach_count,
                "avg_gate_block_rate": self.avg_gate_block_rate}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "PassRecord":
        return cls(pass_n=int(d["pass"]), days_passed=int(d["days_passed"]),
                   days_failed=int(d["days_failed"]), avg_pnl=float(d["avg_pnl"]),
                   avg_dd=float(d["avg_dd"]), breach_count=int(d["breach_count"]),
                   avg_gate_block_rate=float(d.get("avg_gate_block_rate", 0.0)))


# ─────────────────────────────── the Policy Card ─────────────────────────────────────
@dataclass
class PolicyCard:
    """WHO + HOW a policy is. Writes/reads manifest.json + performance.json + compatibility.sig under
    cfg.POLICY_REGISTRY_DIR/<policy_name>/ (README §2-§4). Build one with build_card(), record_pass()
    after each pass, save() to disk; load() to read back."""

    policy_name: str
    auto_name_basis: Dict[str, object]
    created: str
    base_policy: Optional[str]
    data_window: Dict[str, object]
    state_dim: int
    training_wheels: bool
    training_phase: str
    overrides_applied: Dict[str, object]
    compatibility_signature: str
    n_passes_completed: int = 0
    # C14 [2026-06-19]: back-to-back failed-day streak (audit Fix 4). The runtime SOURCE is
    # ftmo_passing/challenge_state.py ChallengeState.consecutive_loss_days; this is the value surfaced
    # onto the manifest (set by the caller — live-run population wired in a later step). 0 = none yet.
    consecutive_loss_days: int = 0
    pass_history: List[PassRecord] = field(default_factory=list)

    # ---- mutation -----------------------------------------------------------------
    def record_pass(self, rec: PassRecord) -> None:
        """Append one pass's result and bump the counter (called by the trainer/Barbershop Cell 6)."""
        self.pass_history.append(rec)
        self.n_passes_completed = len(self.pass_history)

    # ---- derived performance (README §3) ------------------------------------------
    def overall_pass_rate(self) -> float:
        """Real scoreboard: total days passed / total days attempted across all passes (0..1)."""
        tot = sum(r.days_passed + r.days_failed for r in self.pass_history)
        return sum(r.days_passed for r in self.pass_history) / tot if tot else 0.0

    def best_pass(self) -> Optional[Dict[str, object]]:
        """The single strongest pass — most days passed, ties broken by fewest breaches (README §3)."""
        if not self.pass_history:
            return None
        best = max(self.pass_history, key=lambda r: (r.days_passed, -r.breach_count))
        return best.to_dict()

    # ---- serialization (the 3 files) ----------------------------------------------
    def manifest(self) -> Dict[str, object]:
        return {"policy_name": self.policy_name, "auto_name_basis": self.auto_name_basis,
                "created": self.created, "base_policy": self.base_policy,
                "data_window": self.data_window, "n_passes_completed": self.n_passes_completed,
                "state_dim": self.state_dim, "training_wheels": self.training_wheels,
                "training_phase": self.training_phase, "overrides_applied": self.overrides_applied,
                "consecutive_loss_days": self.consecutive_loss_days,   # C14 back-to-back loss streak
                "compatibility_signature": self.compatibility_signature}

    def performance(self) -> Dict[str, object]:
        return {"pass_history": [r.to_dict() for r in self.pass_history],
                "best_pass": self.best_pass(), "overall_pass_rate": self.overall_pass_rate()}

    def dir(self, root: Optional[Path] = None) -> Path:
        return (root or cfg.POLICY_REGISTRY_DIR) / self.policy_name

    def save(self, root: Optional[Path] = None) -> Path:
        """Write the 3 files atomically-ish under <root>/<policy_name>/ and return that folder.
        COUPLING -> artifacts/policy_registry/README.md: file names + JSON keys must match the guide."""
        d = self.dir(root)
        d.mkdir(parents=True, exist_ok=True)
        (d / MANIFEST_FILE).write_text(json.dumps(self.manifest(), indent=2))
        (d / PERFORMANCE_FILE).write_text(json.dumps(self.performance(), indent=2))
        (d / COMPAT_FILE).write_text(self.compatibility_signature)
        return d

    @classmethod
    def load(cls, policy_name: str, root: Optional[Path] = None) -> "PolicyCard":
        """Read a saved policy's identity back (used by the dashboard, Risk Doctor, resume flow)."""
        d = (root or cfg.POLICY_REGISTRY_DIR) / policy_name
        man = json.loads((d / MANIFEST_FILE).read_text())
        perf = json.loads((d / PERFORMANCE_FILE).read_text()) if (d / PERFORMANCE_FILE).exists() else {}
        card = cls(policy_name=man["policy_name"], auto_name_basis=man.get("auto_name_basis", {}),
                   created=man["created"], base_policy=man.get("base_policy"),
                   data_window=man.get("data_window", {}), state_dim=int(man["state_dim"]),
                   training_wheels=bool(man["training_wheels"]), training_phase=man["training_phase"],
                   overrides_applied=man.get("overrides_applied", {}),
                   compatibility_signature=man["compatibility_signature"],
                   n_passes_completed=int(man.get("n_passes_completed", 0)),
                   consecutive_loss_days=int(man.get("consecutive_loss_days", 0)))   # C14
        card.pass_history = [PassRecord.from_dict(r) for r in perf.get("pass_history", [])]
        return card


def build_card(*, overrides: Dict[str, object], state_dim: int, data_window: Dict[str, object],
               base_policy: Optional[str] = None, baseline: Optional[Dict[str, object]] = None,
               reward_layer_keys: Optional[Sequence[str]] = None,
               law_fingerprint: Optional[str] = None, created: Optional[str] = None,
               consecutive_loss_days: int = 0) -> PolicyCard:
    """One call the Barbershop notebook (Cell 6) / trainer makes to mint a policy's identity: auto-name
    from the OVERRIDES diff, compute the compatibility signature, and fill the manifest. Reward-layer
    keys + law fingerprint default to the live project values (resume-safe across C16/C17)."""
    base = baseline if baseline is not None else baseline_overrides()
    name, basis = auto_name(overrides, baseline=base, base_policy=base_policy)
    sig = compatibility_signature(
        state_dim,
        reward_layer_keys if reward_layer_keys is not None else default_reward_layer_keys(),
        law_fingerprint if law_fingerprint is not None else default_law_fingerprint())
    phase = str(overrides.get("training_phase", base.get("training_phase", "free")))
    wheels = bool(overrides.get("training_wheels", base.get("training_wheels", True)))
    return PolicyCard(
        policy_name=name, auto_name_basis=basis,
        created=created or datetime.now().replace(microsecond=0).isoformat(),
        base_policy=base_policy, data_window=data_window, state_dim=int(state_dim),
        training_wheels=wheels, training_phase=phase, overrides_applied=dict(overrides),
        consecutive_loss_days=int(consecutive_loss_days),   # C14 (audit Fix 4); caller sets the live value
        compatibility_signature=sig)


# ───────────────────────────────── Leaderboard ───────────────────────────────────────
@dataclass(frozen=True)
class LeaderboardRow:
    """One policy's line on the board (the passing-quality summary, README §3 scoreboard order)."""

    policy_name: str
    overall_pass_rate: float
    best_days_passed: int
    total_breaches: int
    n_passes: int
    created: str


class Leaderboard:
    """Ranks every saved policy by how well it PASSES (not raw PnL): pass-rate desc, then best
    days-passed desc, then fewest total breaches, then most recent. For the operator + dashboard."""

    def __init__(self, rows: List[LeaderboardRow]):
        self._rows = rows

    @classmethod
    def from_dir(cls, root: Optional[Path] = None) -> "Leaderboard":
        """Scan cfg.POLICY_REGISTRY_DIR for policy folders (anything with a manifest.json) and rank
        them. Skips folders missing/!corrupt manifests so a half-written run can't crash the board."""
        base = root or cfg.POLICY_REGISTRY_DIR
        rows: List[LeaderboardRow] = []
        if base.exists():
            for d in sorted(p for p in base.iterdir() if p.is_dir()):
                if not (d / MANIFEST_FILE).exists():
                    continue
                try:
                    card = PolicyCard.load(d.name, root=base)
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
                bp = card.best_pass()
                rows.append(LeaderboardRow(
                    policy_name=card.policy_name, overall_pass_rate=card.overall_pass_rate(),
                    best_days_passed=int(bp["days_passed"]) if bp else 0,
                    total_breaches=sum(r.breach_count for r in card.pass_history),
                    n_passes=card.n_passes_completed, created=card.created))
        rows.sort(key=lambda r: (-r.overall_pass_rate, -r.best_days_passed, r.total_breaches,
                                 _neg_iso(r.created)))
        return cls(rows)

    def rows(self) -> List[LeaderboardRow]:
        return list(self._rows)

    def top(self, n: int = 1) -> List[LeaderboardRow]:
        return self._rows[:n]

    def render(self) -> str:
        """A plain-text board the operator can read in a notebook/terminal."""
        if not self._rows:
            return "Policy Leaderboard: (empty — no saved policies yet)"
        head = f"{'#':>2}  {'policy':<32} {'pass%':>6} {'best_days':>9} {'breaches':>8} {'passes':>6}"
        lines = [head, "-" * len(head)]
        for i, r in enumerate(self._rows, 1):
            lines.append(f"{i:>2}  {r.policy_name:<32} {r.overall_pass_rate*100:>5.1f}% "
                         f"{r.best_days_passed:>9} {r.total_breaches:>8} {r.n_passes:>6}")
        return "\n".join(lines)


def _neg_iso(created: str) -> str:
    """Sort key helper: newer ISO timestamps should rank first within a tie. Invert lexicographically
    by mapping each char to its complement so a plain ascending sort puts the latest date on top."""
    return "".join(chr(0x10FFFF - ord(c)) if ord(c) < 0x10FFFF else c for c in created)


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# STANDING RULE [2026-06-19, operator] — applies to THIS file and EVERY file going forward: keep
# SHOWING THE WORK. On every edit (1) append a DATED IRAC entry here, and (2) in the code comments
# DOCUMENT the cross-file RELATIONSHIPS the change depends on (the COUPLING) — name the other file(s)
# and the exact attr/field/key relied on, in BOTH directions — and date the re-pointed logic, so any
# future reader/editor can see what connects to what, and what breaks where, and when it changed.
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-19] C18 — implemented the Policy Registry (Policy Card + Leaderboard).
#   I: artifacts/policy_registry/README.md documented a per-policy identity (manifest/performance/
#      compatibility) + auto-naming + a resume gate, but NO code produced or read it, and there was
#      no way to rank policies by how well they actually pass.
#   R: README.md (the committed contract) + PROJECT_GUIDE §4.11/§4.12; operator brief "Policy Card +
#      Leaderboard"; the standing show-the-work + Layer-0/scoreboard rules.
#   A: PolicyCard (writes/reads the 3 files), auto_name() (OVERRIDES-diff-vs-baseline -> v<N>-tokens,
#      never hand-typed), compatibility_signature()/check_compatibility() (state_dim + reward-LAYER
#      keys + LAW_NAMES hash; weights/math re-points stay resume-safe), and Leaderboard (ranks by
#      pass-rate -> best-days -> fewest-breaches). COUPLING: cfg.POLICY_REGISTRY_DIR, RewardEngine
#      decompose keys, laws.LAW_NAMES, ChallengeConfig/RewardConfig defaults (baseline_overrides).
#   C: Every trained policy now gets a saved, honest IDENTITY and the operator can SEE which config
#      passes best and safely resume/promote it — turning scattered runs into a comparable scoreboard
#      that drives toward a consistently-passing champion.
# [2026-06-19] C19 — name tokens for the TrainConfig knobs + the compatibility map.
#   I: config.build_overrides_dict() now also diffs TrainConfig, but _NAME_ORDER/_SHORT had no tokens
#      for those keys (they would slug into ugly names); and the dim/shaping compatibility web — the
#      "change one file, lose your old policies" hazard — wasn't mapped in one place.
#   R: Operator spec 2026-06-19 (OVERRIDES includes TrainConfig defaults) + operator directive to
#      document the dim + shaping coupling across files so future edits don't silently break resume.
#   A: Added clean tokens (roll/mb/vcoef/g8look/seed) kept LAST in _NAME_ORDER (perspective knobs lead);
#      added the 📍 COMPATIBILITY MAP comment by compatibility_signature() and ⚠️ COMPATIBILITY notes at
#      the THREE owner files (schema.STATE_DIM, reward.decompose L-keys, laws.LAW_NAMES) + the README §4
#      map table. default_* helpers still read live values so the signature can't drift from code.
#   C: Policy names stay readable across every knob, and any future editor who touches the dim/shaping
#      sees exactly which other files move with it and that old policies will need a fresh start — so we
#      never silently lose the ability to resume a past policy or to keep training.
# [2026-06-19] C14 — consecutive_loss_days surfaced onto the Policy Card manifest (audit Fix 4).
#   I: The card had no "back-to-back loss count" — the clearest consistency-failure signal — so the
#      audit's Section-3 risk trio (daily DD limit, dist to -10% wall, back-to-back losses) was incomplete.
#   R: Operator audit Fix 4: surface the streak on the MANIFEST (not performance.json for now), minimal
#      scope; live-run population deferred to Fix 5.
#   A: Added PolicyCard.consecutive_loss_days (default 0), emitted in manifest(), restored in load(), and
#      an explicit build_card(consecutive_loss_days=0) param so the caller sets it (no silent stub). The
#      runtime SOURCE is ChallengeState.consecutive_loss_days (ftmo_passing/challenge_state.py, finalized
#      in reset_day()); wiring the live value from a Barbershop run lands in Fix 5. daily_risk_pct + the
#      -10% permanent_dd_pct already live on the manifest via overrides_applied, so this completes the trio.
#   C: A policy's worst back-to-back miss streak is now part of its saved identity, so the operator can
#      compare consistency across policies at a glance — the metric that separates a repeatable FTMO
#      passer from a streaky one.
