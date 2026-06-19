"""PPOAgent — the four-head actor-critic on the shared 3x256 MLP trunk. 🔴

WHAT THIS MODULE DOES
---------------------
The locked policy network (PPO_ENGINE.md): a shared 3x256 Tanh MLP trunk feeding
FOUR heads —
  1. Direction head : categorical over {HOLD, OPEN_LONG, OPEN_SHORT, CLOSE}, masked
  2. Size head      : Beta(alpha, beta) over raw_size in [0,1]  (G3 locked)
  3. Pointer head   : categorical over the 5 trade slots, masked (used on CLOSE)
  4. Value head     : V(s) for GAE
It samples actions (training) or acts deterministically (live: argmax direction ·
Beta-mean size · argmax pointer), and exposes the SUMMED three-head log-prob and
summed entropy with the locked gating rule: the size head contributes only when an
OPEN fires, the pointer head only when CLOSE fires (masked heads -> 0 logp/entropy).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
This is the brain that turns the 207-dim FTMO observation into a legal, sized,
slot-aware action. The −1e9 masks are applied to the logits HERE, so the policy can
literally never sample a breach-bound or illegal action. The shared trunk lets the
critic's patience (high-γ value) and the actor's restraint come from the same belief,
which is how the bot learns to hold winners and avoid the wall — i.e. to pass.

🔴 LOCKED: four-head architecture, 3x256 trunk, Beta size head. Summed-log-prob and
the OPEN/CLOSE gating are the loss contract.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md`` (Term 4 Action Distribution). Always
read the action probs against the mask: a 0.95 direction prob is meaningless if it
was the only legal option. Mask Dependence = pre-mask (unmasked) logits repeatedly
favor a forbidden action — compare ``unmasked_logits`` vs the mask. The Beta mean is
the live size; if it ignores risk context (same mean at 0% and 3.8% DD) that's Risk
Blindness. Never modify the masks/sizing here — read only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Beta, Categorical

# COUPLING [C1] -> market_pipeline/feature_builder/schema.py: STATE_DIM is the trunk
# input width; if schema's FEATURE_NAMES/EXPECTED_WIDTHS change, this network's input
# layer changes too (and runtime/config.nominal_state_dim must match).
from quantra.market_pipeline.feature_builder import STATE_DIM
# COUPLING [C2/C3] -> market_pipeline/law_mask_engine/engine.py: the direction action
# ints (HOLD/OPEN_LONG/OPEN_SHORT/CLOSE), N_DIR_ACTIONS and N_SLOTS are defined there;
# the gating (_gates) and head widths below MUST stay in lockstep with that file +
# runtime/device.py (RepresentativePolicy mirrors these head sizes).
from quantra.market_pipeline.law_mask_engine.engine import (
    CLOSE,
    N_DIR_ACTIONS,
    N_SLOTS,
    OPEN_LONG,
    OPEN_SHORT,
)

_SIZE_EPS = 1e-6  # clamp Beta samples off {0,1} so log_prob is finite


def _orthogonal(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def _masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Categorical over additive-masked logits ({0, -1e9}). Fully-masked rows fall
    back to uniform so the distribution is valid (their log-prob is never used —
    e.g. the pointer when CLOSE is itself illegal)."""
    masked = logits + mask
    all_masked = (mask <= -1e8).all(dim=-1, keepdim=True)
    masked = torch.where(all_masked, torch.zeros_like(masked), masked)
    return Categorical(logits=masked)


@dataclass
class AgentStep:
    # COUPLING -> learning_system/trainer/trainer.py: collect_rollout reads these field
    # NAMES (step.a_direction/a_size/a_pointer/log_prob/value); renaming/reordering here
    # breaks the buffer.add() call there. Fields also map to RolloutBuffer FIELDS.
    """One forward pass result (training): action + bookkeeping for the buffer."""

    a_direction: torch.Tensor
    a_size: torch.Tensor
    a_pointer: torch.Tensor
    log_prob: torch.Tensor       # SUM of the active heads' log-probs
    value: torch.Tensor
    entropy: torch.Tensor        # SUM of the active heads' entropies


class ActorCritic(nn.Module):
    """Shared 3x256 Tanh trunk + four heads (PPO_ENGINE.md G5)."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers, d = [], state_dim
        for _ in range(depth):
            layers += [_orthogonal(nn.Linear(d, hidden), gain=2 ** 0.5), nn.Tanh()]
            d = hidden
        self.trunk = nn.Sequential(*layers)
        # COUPLING [C2/C3 in COUPLINGS.md]: head widths == the action space. direction
        # = N_DIR_ACTIONS (law_mask_engine), pointer = N_SLOTS (schema). runtime.device
        # .RepresentativePolicy must mirror these. state_dim defaults to STATE_DIM [C1].
        self.direction_head = _orthogonal(nn.Linear(hidden, N_DIR_ACTIONS), gain=0.01)
        self.size_head = _orthogonal(nn.Linear(hidden, 2), gain=0.01)   # Beta a,b params
        self.pointer_head = _orthogonal(nn.Linear(hidden, N_SLOTS), gain=0.01)
        self.value_head = _orthogonal(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return (self.direction_head(h), self.size_head(h),
                self.pointer_head(h), self.value_head(h).squeeze(-1))

    @staticmethod
    def _beta(size_params: torch.Tensor) -> Beta:
        """Beta with alpha,beta >= 1 (softplus+1) -> unimodal, stable (no U-shape)."""
        a = 1.0 + torch.nn.functional.softplus(size_params[..., 0])
        b = 1.0 + torch.nn.functional.softplus(size_params[..., 1])
        return Beta(a, b)


class PPOAgent:
    """Wraps ActorCritic with act / deterministic-act / evaluate for PPO + live."""

    def __init__(self, state_dim: int = STATE_DIM, device: str = "cpu"):
        self.net = ActorCritic(state_dim).to(device)
        self.device = device

    # -- helpers --
    @staticmethod
    def _gates(a_dir: torch.Tensor):
        """Boolean gates: size active on OPEN, pointer active on CLOSE (the lock)."""
        open_gate = (a_dir == OPEN_LONG) | (a_dir == OPEN_SHORT)
        close_gate = a_dir == CLOSE
        return open_gate.float(), close_gate.float()

    def _dists(self, x: torch.Tensor, dir_mask: torch.Tensor, ptr_mask: torch.Tensor):
        dlog, slog, plog, value = self.net(x)
        return (_masked_categorical(dlog, dir_mask), self.net._beta(slog),
                _masked_categorical(plog, ptr_mask), value, dlog)

    # -- training: sample all heads --
    @torch.no_grad()
    def act(self, obs: torch.Tensor, dir_mask: torch.Tensor, ptr_mask: torch.Tensor) -> AgentStep:
        """Stochastic action for rollout collection. obs/masks are (B, ·) or (·,).

        Runs under no_grad: logp_old and V_old are stored as CONSTANTS; the gradient
        path is rebuilt by ``evaluate_actions`` during the PPO update (correct on-policy
        semantics + no spurious autograd graph through the buffer)."""
        x, dm, pm = _b(obs), _b(dir_mask), _b(ptr_mask)
        ddist, sdist, pdist, value, _ = self._dists(x, dm, pm)
        a_dir = ddist.sample()
        a_size = sdist.sample().clamp(_SIZE_EPS, 1 - _SIZE_EPS)
        a_ptr = pdist.sample()
        og, cg = self._gates(a_dir)
        logp = ddist.log_prob(a_dir) + og * sdist.log_prob(a_size) + cg * pdist.log_prob(a_ptr)
        ent = ddist.entropy() + og * sdist.entropy() + cg * pdist.entropy()
        return AgentStep(a_dir, a_size, a_ptr, logp, value, ent)

    @torch.no_grad()
    def act_deterministic(self, obs, dir_mask, ptr_mask):
        """Live action (SOW §2.10): argmax direction · Beta-mean size · argmax pointer."""
        x, dm, pm = _b(obs), _b(dir_mask), _b(ptr_mask)
        dlog, slog, plog, value = self.net(x)
        a_dir = (dlog + dm).argmax(dim=-1)
        beta = self.net._beta(slog)
        a_size = (beta.concentration1 / (beta.concentration1 + beta.concentration0))
        a_ptr = (plog + pm).argmax(dim=-1)
        # COUPLING -> live_bridge/live_session.py + live_runner.py + acceptance.py: this
        # 4-tuple order (a_dir, a_size, a_ptr, value) is unpacked positionally there.
        return a_dir, a_size, a_ptr, value

    def evaluate_actions(self, obs, dir_mask, ptr_mask, a_dir, a_size, a_ptr):
        """Recompute summed log-prob, summed entropy, V(s) for stored actions (PPO update).

        Gating uses the STORED a_dir (what was actually taken), so masked heads
        contribute exactly zero — the locked loss contract.
        """
        ddist, sdist, pdist, value, dlog = self._dists(_b(obs), _b(dir_mask), _b(ptr_mask))
        a_size = a_size.clamp(_SIZE_EPS, 1 - _SIZE_EPS)
        og, cg = self._gates(a_dir)
        logp = ddist.log_prob(a_dir) + og * sdist.log_prob(a_size) + cg * pdist.log_prob(a_ptr)
        ent = ddist.entropy() + og * sdist.entropy() + cg * pdist.entropy()
        # COUPLING -> learning_system/ppo_agent/loss.py: ppo_loss unpacks this 3-tuple as
        # (new_logp, entropy, values) in that exact order — keep them aligned.
        return logp, ent, value


def _b(t: torch.Tensor) -> torch.Tensor:
    """Ensure a leading batch dim (act() is called per symbol-step with a 1-D obs)."""
    t = t if torch.is_tensor(t) else torch.as_tensor(t, dtype=torch.float32)
    return t.unsqueeze(0) if t.dim() == 1 else t


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M5 — implemented the four-head PPO actor-critic.
#   I: Nothing turned the FTMO observation into a legal, sized, slot-aware
#      action, nor produced the summed log-prob the PPO loss requires.
#   R: PPO_ENGINE.md (3x256 trunk; 4 heads; Beta size G3; summed 3-head log-prob;
#      size active on OPEN / pointer on CLOSE; -1e9 masks on logits; live = argmax/mean).
#   A: ActorCritic (shared trunk + 4 heads, orthogonal init) + PPOAgent.act /
#      act_deterministic / evaluate_actions with the OPEN/CLOSE gating and masked-
#      categorical fallback so masked heads contribute exactly 0 logp/entropy.
#   C: The policy can never sample an illegal/breach-bound action (masks on the
#      logits) and learns patience from the shared critic — the brain that, trained
#      under the M4 physics, learns to hit target without touching the wall.
