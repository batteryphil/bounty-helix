# Bounty-Helix

An autonomous AI agent built on the Helix-AGI architecture, repurposed for open-source opportunity discovery and execution.

## Mission

Helix autonomously discovers paid GitHub issues, open-source bounties, grants, and sponsored tasks — then plans, implements, and submits pull requests to earn passive income through legitimate open-source contributions.

## What It Does

1. **Discovers** paid issues on GitHub, CodeTriage, Polar.sh, IssueHunt, Open Collective
2. **Scores** opportunities by reward, complexity, and fit
3. **Plans** implementation before touching any code
4. **Executes** fixes, runs tests, generates PRs
5. **Learns** which repos/projects yield the best ROI over time

## Architecture

Built on Helix-AGI:
- **Hermes-3 8B** (GPU) — primary reasoning and code generation
- **Qwen 0.5B CPU Coprocessor** — context compression for large repos/docs
- **LoRA Self-Trainer** — parametric self-improvement from completed tasks
- **Mission Control Dashboard** — real-time monitoring at `localhost:5050`

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the agent
python core/pulse_loop.py

# Start the dashboard
python dashboard/dashboard.py
```

## Constraints

- Only pursues **legitimate open-source contributions**
- Never violates repository rules, platform ToS, or licensing
- All PRs are human-reviewable before submission
