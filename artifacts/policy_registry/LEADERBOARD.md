# Policy Leaderboard

> **Rulebook (for the Risk Doctor / any LLM):** `docs/MLP_INTERPRETABILITY_LAYER.md`.
> **Master manual:** `docs/PROJECT_GUIDE.md` §4.11 (Policy Registry).

This is a **manually maintained placeholder**. It ranks saved policies by how well they
actually pass the FTMO challenge (+2.5%/day without breaching the −4% trailing wall) so the
operator can see, at a glance, which policy to **resume** or **promote**. Today the rows are
`TBD` stubs — the live ranking is produced dynamically by `Leaderboard.from_dir().render()`
(`quantra/learning_system/policy_registry/registry.py`) from each policy's `performance.json`;
this file will be **auto-generated from that same source in a later session**.

| Rank | Policy Name | Date | Win Rate | Daily PnL | FTMO Phase | Notes |
|------|-------------|------|----------|-----------|------------|-------|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD | TBD |

This file is intentionally static for now. Auto-generation will be wired in a later session.
