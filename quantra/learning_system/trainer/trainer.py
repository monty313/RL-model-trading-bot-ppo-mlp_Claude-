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
from quantra.learning_system.trainer.gae import compute_gae
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

    # ----- update -----
    def update(self, dials) -> Dict[str, float]:
        b = self.buffer.get()
        with torch.no_grad():
            # net(...)[3] indexes the VALUE head — COUPLING -> ppo_agent/agent.py: relies on
            # ActorCritic.forward returning (dir, size, ptr, value) with value at index 3.
            last_value = 0.0 if float(b["done"][-1]) > 0.5 else float(self.agent.net(self._obs.unsqueeze(0))[3])
        # COUPLING -> trainer/gae.py: positional args (rewards, values, dones, last_value);
        # b keys come from RolloutBuffer.get(). compute_gae returns (adv, ret) in this order.
        adv, ret = compute_gae(b["reward"], b["value_old"], b["done"], last_value)

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
