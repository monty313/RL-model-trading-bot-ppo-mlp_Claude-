"""GAE — Generalized Advantage Estimation with the LOCKED gamma=0.997, lambda=0.97. 🔴

WHAT THIS MODULE DOES
---------------------
Computes advantages + value targets from a rollout (rewards, V(s), dones, bootstrap
V(s_T)) using GAE with the hand-locked discount/trace. gamma=0.997 and lambda=0.97 are
the "math of patience" (REWARD_DESIGN.md / G2) and are NEVER tuned (off-limits to HPO).

HOW IT SERVES REPEATED FTMO-STYLE PASSING
-----------------------------------------
High gamma + high lambda is what teaches the critic (and thus the actor) to value the
distant payoff of holding a good trade and respecting the wall, rather than reacting
bar-to-bar. That patience is the temperament that hits the daily target without
breaching — i.e. that passes consistently.

🔴 LOCKED: gamma=0.997, lambda=0.97. Changing them needs Monty's approval.

LLM RISK DOCTOR — HOW TO THINK ABOUT THIS FILE
----------------------------------------------
Rulebook: ``docs/MLP_INTERPRETABILITY_LAYER.md`` (Term 5 Value Estimate). Advantages
that don't go negative as the bot approaches the wall are a Critic Misalignment signal
— the GAE is correct; the value head learned the wrong thing.
"""

from __future__ import annotations

import torch

GAMMA = 0.997   # 🔴 locked
LAMBDA = 0.97   # 🔴 locked


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                last_value: float, gamma: float = GAMMA, lam: float = LAMBDA):
    """Return (advantages, returns) for a rollout. ``last_value`` bootstraps V(s_T).

    done[t]=1 cuts the backup at episode boundaries (no value leaks across a breach /
    end-of-data), so each episode's patience math is self-contained.
    """
    t_n = rewards.shape[0]
    adv = torch.zeros(t_n, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(t_n)):
        next_value = last_value if t == t_n - 1 else float(values[t + 1])
        nonterminal = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_value * nonterminal - float(values[t])
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE LOG (IRAC) - standing rule since 2026-06-13. I/R/A/C; Conclusion is always
# why this helps the bot pass FTMO consistently. Rulebook: docs/MLP_INTERPRETABILITY_LAYER.md
# ─────────────────────────────────────────────────────────────────────────────
# [2026-06-13] M8 — implemented GAE with the locked gamma/lambda.
#   I: PPO needs advantages + value targets, computed with the patience discount.
#   R: G2 (gamma=0.997, lambda=0.97 hand-locked, off-limits to HPO) + episode-boundary cuts.
#   A: compute_gae backward pass with done-masking; gamma/lambda as locked defaults.
#   C: The critic learns long-horizon value so the bot holds winners and respects the
#      wall - the patience that turns exploration into consistent passing.
