"""PPO loss — L = L_clip − c1·L_value + c2·entropy, on the summed 3-head log-prob. 🔴

WHAT THIS MODULE DOES
---------------------
Computes the locked PPO objective (PPO_ENGINE.md, SOW §2.8) for a minibatch. The
policy ratio uses the SUMMED log-prob across the three action heads (direction · Beta
size · pointer), recomputed by ``PPOAgent.evaluate_actions`` with the OPEN/CLOSE gating
so masked heads contribute zero. Advantages are normalized per minibatch. Returns the
scalar loss plus diagnostics (approx-KL, clip fraction, components) for telemetry.

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
The clip keeps each update a small, trust-region step so the patient, law-bounded
policy improves without lurching into a worse regime (and breaching). c1·value fits
the critic that supplies the "reason to hold" (high-γ patience); c2·entropy keeps
enough exploration to find premium legal setups. Getting this objective exactly right
is what turns the M4 physics + the masks into a policy that passes consistently.

🔴 The summed-log-prob ratio and the gating are the loss contract; γ/λ live in the
GAE (M8) and are hand-locked.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md``. ``approx_kl`` spiking or
``clip_frac`` near 1 means the step was too big (instability -> Representation Chaos
risk). A value loss that won't shrink near danger is Critic Misalignment. These
diagnostics are emitted here for exactly that reverse-chain reasoning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def ppo_loss(
    agent,
    batch: Dict[str, torch.Tensor],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.25,
    value_coef: float = 0.5,
    entropy_coef: float = 0.03,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return (loss_to_minimize, diagnostics). Defaults sit inside the law-school
    aggression ranges (clip 0.25-0.35, entropy 0.03-0.08); the trainer (M8) sets the
    live values from the missed-opportunity scheduler."""
    # COUPLING -> ppo_agent/agent.py: arg ORDER here matches PPOAgent.evaluate_actions
    # (obs, dir_mask, ptr_mask, a_dir, a_size, a_ptr) and the 3-tuple it returns.
    # COUPLING -> rollout_buffer/buffer.py: these batch KEYS (obs/dir_mask/ptr_mask/
    # a_direction/a_size/a_pointer/logp_old) are exactly the dict keys RolloutBuffer.get()
    # emits; renaming a field there silently KeyErrors here.
    new_logp, entropy, values = agent.evaluate_actions(
        batch["obs"], batch["dir_mask"], batch["ptr_mask"],
        batch["a_direction"], batch["a_size"], batch["a_pointer"],
    )
    logp_old = batch["logp_old"]

    # Per-minibatch advantage normalization (standard PPO; stabilises the step).
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratio = torch.exp(new_logp - logp_old)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_obj = torch.min(surr1, surr2).mean()                 # maximize

    value_loss = 0.5 * (values - returns).pow(2).mean()         # minimize
    entropy_bonus = entropy.mean()                              # maximize

    # SOW §2.8 objective to MAXIMIZE: L_clip − c1·L_value + c2·entropy.
    # Loss to MINIMIZE is its negation.
    loss = -(policy_obj - value_coef * value_loss + entropy_coef * entropy_bonus)

    with torch.no_grad():
        diag = {
            "loss": float(loss),
            "policy_obj": float(policy_obj),
            "value_loss": float(value_loss),
            "entropy": float(entropy_bonus),
            # Schulman k3 KL estimator: E[(r-1) - log r] — always >= 0, lower variance than
            # the naive (logp_old - new_logp) mean [2026-06-15: diagnostic robustness].
            "approx_kl": float(((ratio - 1.0) - (new_logp - logp_old)).mean()),
            "clip_frac": float(((ratio - 1.0).abs() > clip_eps).float().mean()),
            "ratio_mean": float(ratio.mean()),
        }
    return loss, diag


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M5 — implemented the PPO loss with summed 3-head log-prob.
#   I: The four-head policy needed its exact PPO objective (clipped surrogate on the
#      summed log-prob + value + entropy) before any training could run.
#   R: SOW §2.8 (L = L_clip − c1·L_value + c2·entropy; ratio uses the summed 3-head
#      log-prob; masked heads contribute 0) + the law-school aggression ranges.
#   A: ppo_loss() via agent.evaluate_actions (gated summed log-prob), clipped surrogate,
#      per-minibatch advantage norm, value + entropy terms, KL/clip diagnostics.
#   C: Each update is a correct, small trust-region step on the legal policy, so the
#      patient law-bounded behaviour improves toward passing without lurching into a
#      breach-prone regime.
# [2026-06-15] approx_kl -> Schulman k3 estimator (diagnostic robustness).
#   I: approx_kl used the naive (logp_old - new_logp).mean(), which can go negative and is
#      higher-variance — a noisier training-health signal for the Risk Doctor's KL reads.
#   R: Canonical PPO (Schulman): KL ≈ E[(r-1) - log r], always >= 0, lower variance. Loss math unchanged.
#   A: approx_kl = ((ratio-1) - (new_logp - logp_old)).mean(); policy/value/entropy terms untouched.
#   C: A trustworthy KL trace makes instability (Representation Chaos) easier to catch early,
#      protecting the patient policy — and thus the pass rate. (Full actor/critic audit pending.)
