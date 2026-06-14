"""RolloutBuffer — on-policy storage of the locked 10-field transition. 🔴

WHAT THIS MODULE DOES
---------------------
Preallocated, fixed-capacity storage for one PPO rollout (no replay buffer — PPO is
strictly on-policy, SOW §2.1). Each symbol-step stores the locked 10-field transition
(SOW §2.9): ``(s, a_direction, a_size, a_pointer, reward, s', logp_old, V_old, done,
masks)`` where ``masks`` = the direction mask + the pointer/slot mask and ``logp_old``
is the SUM of the three action-head log-probs.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
PPO's on-policy update needs exactly these fields to recompute the ratio and
advantage correctly. Replay would contaminate the gradient with stale, off-policy
data and quietly break the patient, law-bounded policy we're training to pass. Storing
the masks lets the update re-mask identically, so the importance ratio is computed in
the same legal space the action was sampled from.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. ``logp`` here is the SUM across the
3 active action heads (size only on OPEN, pointer only on CLOSE). If a diagnosis needs
to know what was legal at a step, read the stored ``dir_mask`` / ``ptr_mask`` — they
are the ground truth of the legal space, not a reconstruction.
"""

from __future__ import annotations

import torch

# The 10 locked fields (SOW §2.9). 'masks' is materialised as dir_mask + ptr_mask.
FIELDS = ("obs", "a_direction", "a_size", "a_pointer", "reward",
          "next_obs", "logp_old", "value_old", "done", "masks")


class RolloutBuffer:
    """Fixed-capacity, single-trajectory PPO buffer. Cleared after every update."""

    def __init__(self, capacity: int, state_dim: int, n_dir: int = 4,
                 n_slots: int = 5, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = device
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.obs = z(capacity, state_dim)
        self.next_obs = z(capacity, state_dim)
        self.a_direction = z(capacity, dt=torch.long)
        self.a_size = z(capacity)
        self.a_pointer = z(capacity, dt=torch.long)
        self.reward = z(capacity)
        self.done = z(capacity)
        self.logp_old = z(capacity)
        self.value_old = z(capacity)
        self.dir_mask = z(capacity, n_dir)
        self.ptr_mask = z(capacity, n_slots)
        self.ptr = 0

    def __len__(self) -> int:
        return self.ptr

    @property
    def is_full(self) -> bool:
        return self.ptr >= self.capacity

    def add(self, obs, a_direction, a_size, a_pointer, reward, next_obs,
            logp_old, value_old, done, dir_mask, ptr_mask) -> None:
        """Store one symbol-step transition (all 10 locked fields)."""
        if self.is_full:
            raise RuntimeError("RolloutBuffer is full; collect() then clear() before adding.")
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32)
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.a_direction[i] = int(a_direction)
        self.a_size[i] = float(a_size)
        self.a_pointer[i] = int(a_pointer)
        self.reward[i] = float(reward)
        self.done[i] = float(done)
        self.logp_old[i] = float(logp_old)
        self.value_old[i] = float(value_old)
        self.dir_mask[i] = torch.as_tensor(dir_mask, dtype=torch.float32)
        self.ptr_mask[i] = torch.as_tensor(ptr_mask, dtype=torch.float32)
        self.ptr += 1

    def get(self) -> dict:
        """Return the filled slice as tensors (for GAE + minibatch updates, M8)."""
        n = self.ptr
        return {
            "obs": self.obs[:n], "next_obs": self.next_obs[:n],
            "a_direction": self.a_direction[:n], "a_size": self.a_size[:n],
            "a_pointer": self.a_pointer[:n], "reward": self.reward[:n],
            "done": self.done[:n], "logp_old": self.logp_old[:n],
            "value_old": self.value_old[:n], "dir_mask": self.dir_mask[:n],
            "ptr_mask": self.ptr_mask[:n],
        }

    def clear(self) -> None:
        """Discard the rollout (on-policy: never reused). Resets the write pointer."""
        self.ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M5 — implemented the on-policy RolloutBuffer.
#   I: PPO needs the locked 10-field transition stored per step to compute the ratio
#      and advantage; a replay buffer would corrupt the on-policy gradient.
#   R: SOW §2.1 (on-policy, no replay) + §2.9 (the 10 fields, masks + summed log-prob).
#   A: Fixed-capacity tensors for all 10 fields incl. dir_mask + ptr_mask; add/get/clear;
#      cleared after every update (never reused).
#   C: The patient, law-bounded policy is updated on its OWN fresh experience in the
#      same legal space it acted in — which is what keeps PPO's improvement honest and
#      pointed at passing.
