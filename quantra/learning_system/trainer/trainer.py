"""Trainer — the PPO loop: collect rollout -> GAE -> minibatch updates -> schedule. 🔴

WHAT THIS MODULE DOES
---------------------
Drives the multi-symbol env with the PPOAgent to collect on-policy rollouts, computes
GAE (locked gamma/lambda), runs K epochs of minibatch PPO updates (ppo_loss), advances
the aggression scheduler from the G8 missed-opportunity rate, and checkpoints the brain.
Wires the CurriculumManager (law-school stage config + 1m feature mask).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
This is where everything becomes a policy that passes: the M4 physics + M3 masks + M6
reward + M5 agent are turned, by repeated patient PPO steps, into behaviour that hits
the target without breaching. Checkpointing every brain lets the validation pipeline
(M12) keep only those that improve the pass-rate scoreboard.

🔴 LOCKED: gamma/lambda (in GAE), the aggression ranges + scheduler logic.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. The per-update diagnostics
(approx_kl, clip_frac, value_loss, miss_rate) are the training-health trace. A KL that
explodes = unstable step (Representation Chaos risk); a miss_rate stuck high =
Stagnation Blindness. Read them along the reverse chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from quantra.learning_system.curriculum_manager.curriculum import CurriculumManager
from quantra.learning_system.ppo_agent.agent import PPOAgent
from quantra.learning_system.ppo_agent.loss import ppo_loss
from quantra.learning_system.rollout_buffer.buffer import RolloutBuffer
# GAMMA is imported alongside compute_gae for the reward-normalizer's discounted-return accumulator
# (it must use the SAME locked discount the GAE backup uses, or the running scale would be inconsistent).
from quantra.learning_system.trainer.gae import compute_gae, GAMMA
from quantra.learning_system.trainer.scheduler import AggressionScheduler, missed_opportunity
# COUPLING [C1] -> market_pipeline/feature_builder/schema.py: STATE_DIM sizes the agent
# input, the buffer rows, and the zero next_obs / checkpoint metadata below.
from quantra.market_pipeline.feature_builder import STATE_DIM
# COUPLING [C3] -> market_pipeline/law_mask_engine/engine.py: build_pointer_mask returns
# an N_SLOTS-wide mask; its length must match the agent pointer head + buffer ptr_mask.
from quantra.market_pipeline.law_mask_engine.engine import build_pointer_mask
from quantra.runtime import config as cfg


@dataclass
class TrainConfig:
    rollout_size: int = 512        # G4 early law school
    minibatch: int = 64            # G4 early law school
    value_coef: float = 0.5
    g8_lookahead: int = 30         # bars ahead to score a missed opportunity
    seed: int = 0
    # [2026-06-21] VecNormalize-style RETURN normalization. Default False = current behaviour (the
    # baseline + tests are byte-identical when off). When True, the Trainer divides rewards by a running
    # std of the discounted return before GAE, so returns stay ~O(1) and the (unnormalized) value loss
    # is stable regardless of the operator's reward weights — required to run a large net_pnl_weight (or
    # any far-from-1 weight) without the value head blowing up. Preserves all RELATIVE term ratios (it is
    # a single positive scalar division of the TOTAL reward), so E8 Layer-0 dominance is untouched.
    # COUPLING -> registry._NAME_ORDER/_SHORT ("normalize_rewards"/"rewnorm") + colab notebook TRAIN_CFG.
    normalize_rewards: bool = False


class RunningMeanStd:
    """Chan/parallel running mean+variance for a scalar stream (the discounted RETURN).

    Used only by the optional reward normalizer (TrainConfig.normalize_rewards). It tracks the std of
    the discounted return so rewards can be scaled to ~O(1) no matter how large the operator's reward
    weights make them — which is what keeps GAE + the (unnormalized) value loss numerically stable.
    Pure numpy; carries no torch/graph state. var starts at 1.0 so the very first (pre-warmup) division
    is a no-op-ish ~/1 rather than /0."""

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = float(eps)

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        b_mean = float(x.mean()); b_var = float(x.var()); b_count = float(x.size)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta * delta * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var)) + 1e-8


class Trainer:
    """One training run over a single multi-symbol env (a walk-forward train window)."""

    def __init__(self, env, agent: Optional[PPOAgent] = None,
                 train_cfg: Optional[TrainConfig] = None,
                 scheduler: Optional[AggressionScheduler] = None,
                 curriculum: Optional[CurriculumManager] = None,
                 device: str = "cpu"):
        self.env = env
        self.cfg = train_cfg or TrainConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.agent = agent or PPOAgent(state_dim=STATE_DIM, device=device)
        self.scheduler = scheduler or AggressionScheduler()
        self.curriculum = curriculum or CurriculumManager()
        self.device = device
        self.opt = torch.optim.Adam(self.agent.net.parameters(), lr=self.scheduler.values().lr)
        self.buffer = RolloutBuffer(self.cfg.rollout_size, STATE_DIM, device=device)
        self._feature_mask = torch.as_tensor(self.curriculum.feature_mask())
        self.history: List[Dict[str, float]] = []
        # Reward-normalizer state (used only when cfg.normalize_rewards): running std of the discounted
        # return + the cross-rollout discounted accumulator. Persist across updates so the scale estimate
        # keeps improving over the run (reset of the accumulator happens only on episode `done`).
        self._ret_rms = RunningMeanStd()
        self._ret_acc = 0.0
        self._obs = self._apply_mask(self.env.reset())

    # ----- helpers -----
    def _apply_mask(self, obs: np.ndarray) -> torch.Tensor:
        """Multiply the observation by the curriculum's 1m-timing feature mask."""
        return torch.as_tensor(obs, dtype=torch.float32) * self._feature_mask

    def _masks(self):
        # COUPLING -> env/trading_env.py: reads env.symbols/.cursor/.slots[sym] (each slot's
        # .occupied) and calls env.direction_mask(sym); renaming any of these on the env
        # breaks rollout collection. slot.occupied feeds build_pointer_mask (C3).
        sym = self.env.symbols[self.env.cursor]
        dm = torch.as_tensor(self.env.direction_mask(sym), dtype=torch.float32)
        occ = [s.occupied for s in self.env.slots[sym]]
        pm = torch.as_tensor(build_pointer_mask(occ), dtype=torch.float32)
        return sym, dm, pm

    # ----- rollout collection -----
    def collect_rollout(self) -> Dict[str, float]:
        """Fill the buffer with rollout_size on-policy symbol-steps; track G8 miss-rate."""
        self.buffer.clear()
        misses, flats = 0, 0
        while not self.buffer.is_full:
            sym, dm, pm = self._masks()
            t_now = self.env.t
            was_flat = self.env._position(sym) == 0
            step = self.agent.act(self._obs, dm, pm)
            # COUPLING -> ppo_agent/agent.py: reads AgentStep field names a_direction/
            # a_size/a_pointer (and log_prob/value/done below) — must match that dataclass.
            a_dir = int(step.a_direction[0]); a_size = float(step.a_size[0]); a_ptr = int(step.a_pointer[0])
            # COUPLING -> env/trading_env.py: env.step takes the (a_dir, a_size, a_ptr) tuple
            # and returns (next_obs, reward, done, info) in this order — keep both contracts.
            nxt, reward, done, _info = self.env.step((a_dir, a_size, a_ptr))

            # G8 missed-opportunity scoring (training-only; lookahead allowed here).
            if was_flat:
                flats += 1
                # COUPLING -> env/trading_env.py + data_loader/loader.py: env.data[sym]
                # exposes .close/.atr arrays and a PRECOMPUTED_NAMES-ordered .matrix row;
                # data.matrix[t] is passed straight into scheduler.missed_opportunity (C1).
                data = self.env.data[sym]
                k = t_now + self.cfg.g8_lookahead
                if k < len(data.close) and data.atr[t_now] > 0:
                    move_atr = (data.close[k] - data.close[t_now]) / data.atr[t_now]
                    if missed_opportunity(data.matrix[t_now], True, float(move_atr)):
                        misses += 1

            next_obs = self._apply_mask(nxt) if not done else torch.zeros(STATE_DIM)
            # COUPLING -> rollout_buffer/buffer.py: positional arg order must match
            # RolloutBuffer.add(obs, a_dir, a_size, a_ptr, reward, next_obs, logp, value,
            # done, dir_mask, ptr_mask).
            self.buffer.add(self._obs, a_dir, a_size, a_ptr, reward, next_obs,
                            float(step.log_prob[0]), float(step.value[0]), float(done), dm, pm)
            self._obs = next_obs if not done else self._apply_mask(self.env.reset())

        miss_rate = misses / max(1, flats)
        return {"miss_rate": miss_rate, "flat_steps": flats}

    # ----- reward normalization (optional) -----
    def _normalize_rewards(self, reward: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """VecNormalize-style: divide rewards by the running std of the DISCOUNTED RETURN (no mean
        shift, so reward signs/structure AND the E8 inter-layer ratios are preserved — it is one
        positive scalar division of the TOTAL reward). The running std is updated with THIS rollout's
        discounted returns BEFORE it scales them, so even the first rollout is normalized by its own
        scale (no cold-start spike). Result is clamped to +/-10 (standard) against rare outliers.
        Keeps returns ~O(1) so compute_gae + the unnormalized value loss stay stable at any reward
        weight. COUPLING -> trainer.gae.GAMMA (same locked discount) + compute_gae (consumes this)."""
        r = reward.detach().cpu().numpy().astype(np.float64)
        d = done.detach().cpu().numpy()
        disc = np.empty_like(r)
        acc = self._ret_acc
        for i in range(r.size):                 # discounted-return accumulator, reset on episode done
            acc = acc * GAMMA + r[i]
            disc[i] = acc
            if d[i] > 0.5:
                acc = 0.0
        self._ret_acc = acc
        self._ret_rms.update(disc)              # update the scale estimate with this rollout first...
        return torch.clamp(reward / self._ret_rms.std, -10.0, 10.0)   # ...then scale by it

    # ----- update -----
    def update(self, dials) -> Dict[str, float]:
        b = self.buffer.get()
        with torch.no_grad():
            # net(...)[3] indexes the VALUE head — COUPLING -> ppo_agent/agent.py: relies on
            # ActorCritic.forward returning (dir, size, ptr, value) with value at index 3.
            last_value = 0.0 if float(b["done"][-1]) > 0.5 else float(self.agent.net(self._obs.unsqueeze(0))[3])
        # Optional VecNormalize-style return scaling (TrainConfig.normalize_rewards). Keeps returns
        # ~O(1) so the value loss stays stable at any reward weight; OFF -> raw rewards (unchanged).
        rewards = self._normalize_rewards(b["reward"], b["done"]) if self.cfg.normalize_rewards else b["reward"]
        # COUPLING -> trainer/gae.py: positional args (rewards, values, dones, last_value);
        # b keys come from RolloutBuffer.get(). compute_gae returns (adv, ret) in this order.
        adv, ret = compute_gae(rewards, b["value_old"], b["done"], last_value)

        for g in self.opt.param_groups:
            g["lr"] = dials.lr
        n = len(b["obs"])
        idx = np.arange(n)
        last = {}
        for _ in range(dials.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.cfg.minibatch):
                mb = idx[s:s + self.cfg.minibatch]
                batch = {k: v[mb] for k, v in b.items()}
                loss, diag = ppo_loss(self.agent, batch, adv[mb], ret[mb],
                                      clip_eps=dials.clip_eps, value_coef=self.cfg.value_coef,
                                      entropy_coef=dials.entropy_coef)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.net.parameters(), 0.5)
                self.opt.step()
                last = diag
        return last

    # ----- top-level loop -----
    def train(self, n_updates: int) -> List[Dict[str, float]]:
        for _ in range(n_updates):
            roll = self.collect_rollout()
            self.scheduler.update(roll["miss_rate"])
            dials = self.scheduler.values()
            diag = self.update(dials)
            diag.update(roll, aggression=self.scheduler.aggression, lr=dials.lr,
                        entropy_coef=dials.entropy_coef, epochs=dials.epochs)
            self.history.append(diag)
        return self.history

    def checkpoint(self, name: str = "brain") -> Path:
        """Save the brain (SOW §8.4: every brain checkpointed, benchmarked, never lost)."""
        # COUPLING -> runtime/config.py: depends on cfg.ensure_dirs() + cfg.CHECKPOINT_DIR;
        # renaming either there breaks checkpointing.
        cfg.ensure_dirs()
        path = cfg.CHECKPOINT_DIR / f"{name}.pt"
        torch.save({"state_dict": self.agent.net.state_dict(),
                    "history": self.history, "state_dim": STATE_DIM}, path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M8 — implemented the PPO Trainer.
#   I: The pieces (env, agent, reward, buffer, GAE, scheduler, curriculum) existed but
#      nothing ran the on-policy loop that turns them into a trained, checkpointed brain.
#   R: SOW §2.1/2.8 (on-policy PPO), G2/G4 (ranges, 512/64), G8 (scheduler input), §8.4
#      (checkpoint every brain).
#   A: collect_rollout (agent steps env, fills buffer, scores G8 miss-rate) -> GAE ->
#      K-epoch minibatch ppo_loss with grad clip -> scheduler.update -> checkpoint;
#      curriculum 1m feature mask applied to observations.
#   C: Repeated patient PPO steps under the M4 physics + masks turn exploration into a
#      brain that hits target without breaching - and every brain is saved for the
#      promotion gate, so only pass-rate improvements survive.
# [2026-06-21] Optional VecNormalize-style RETURN normalization (TrainConfig.normalize_rewards).
#   I: The operator wants the reward terms scaled so PnL is a meaningful number AND the run is stable.
#      Advantages are already per-minibatch normalized (loss.py), but the VALUE loss is NOT — so a large
#      reward scale (e.g. a big net_pnl_weight, or the 5.0 day-end events) balloons value_loss and
#      whipsaws the policy (observed: value_loss -> ~1.0, entropy crash/recover ~upd 1750/2750).
#   R: Operator decision 2026-06-21 ("MAKE IT NORMALIZED", priority: pass consistently > +2.5% w/o
#      breaching trailing DD > everything else). Standard fix: scale rewards by the running std of the
#      discounted return so returns stay ~O(1) at any weight; preserve all RELATIVE ratios so E8 holds.
#   A: Added RunningMeanStd + Trainer._normalize_rewards (discounted-return accumulator -> RMS -> divide
#      -> clamp +/-10), gated by TrainConfig.normalize_rewards (default False = unchanged baseline; the
#      notebook turns it ON). GAMMA imported from gae so the accumulator uses the SAME locked discount.
#      registry._NAME_ORDER/_SHORT gained "normalize_rewards"/"rewnorm" so a normalized run is named +
#      reproduced. gamma/lambda, the GAE math, masks, sizing, wall, and the reward layer keys are all
#      UNCHANGED. With normalization on, the absolute reward scale is irrelevant to learning (only term
#      ratios are), so net_pnl_weight returns to 1.0 and the anti-breach pain/event signals stay audible.
#   C: The run is numerically stable at any operator weighting, so the policy can be driven toward
#      "+2.5% without breaching the trailing DD, consistently" without the value head detonating.
