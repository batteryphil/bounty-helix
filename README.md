# 🎯 Bounty-Helix

An autonomous AI agent that hunts, solves, and submits paid open-source bounties on IssueHunt — fully automated, runs on your local GPU.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![GPU](https://img.shields.io/badge/GPU-RTX%203060%2B-green) ![License](https://img.shields.io/badge/license-MIT-brightgreen)

---

## What It Does

Bounty-Helix runs 24/7 on your machine and:

1. **Searches IssueHunt** for open funded GitHub issues
2. **Ranks opportunities** by effort, reward, and fit
3. **Plans and implements** fixes autonomously using Hermes-3 LLM
4. **Writes solutions** to the `solutions/` directory
5. **Submits pull requests** with your IssueHunt attribution so payouts route to you

---

## Requirements

| Requirement | Notes |
|---|---|
| GPU | NVIDIA RTX 3060 (12GB VRAM) or better |
| RAM | 16GB+ recommended |
| Storage | ~20GB free (model weights) |
| Python | 3.10 or 3.12 |
| CUDA | 11.8+ |
| Git | 2.x |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/batteryphil/bounty-helix.git
cd bounty-helix
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes peft
pip install flask requests python-dotenv numpy
```

> **Note:** `bitsandbytes` enables 4-bit NF4 quantization — required to fit the 8B model in 12GB VRAM.

### 4. Download the model

The agent uses **NousResearch/Hermes-3-Llama-3.1-8B** (~16GB download, stored in HF cache).

```bash
# One-time download (requires HuggingFace account)
pip install huggingface_hub
huggingface-cli login
huggingface-cli download NousResearch/Hermes-3-Llama-3.1-8B
```

Or set a custom cache path if your system drive is small:

```bash
export HF_HOME=/data/hf_cache
huggingface-cli download NousResearch/Hermes-3-Llama-3.1-8B
```

### 5. Create your `.env` file

```bash
cp .env.example .env
nano .env
```

Fill in your credentials:

```env
# GitHub Personal Access Token
# Create at: https://github.com/settings/tokens
# Scopes needed: repo, read:user
GITHUB_TOKEN=ghp_your_token_here
GITHUB_USER=your_github_username

# IssueHunt account (for bounty payout attribution)
# Sign up at: https://issuehunt.io
ISSUEHUNT_USERNAME=your_issuehunt_username
ISSUEHUNT_PROFILE=https://issuehunt.io/u/your_issuehunt_username

# HuggingFace token (for model download)
HF_TOKEN=hf_your_token_here
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

### 6. Run setup

```bash
python setup.py
```

This initializes the data directory, creates required files, and verifies your environment.

### 7. Start the agent

```bash
# Start both the agent and dashboard
bash start.sh
```

Or start manually:

```bash
# Terminal 1 — Agent
python main.py

# Terminal 2 — Dashboard (optional, monitor at http://localhost:5050)
python dashboard/dashboard.py
```

---

## How Payouts Work

1. The agent finds an IssueHunt-funded issue and creates a fix
2. It submits a PR with your IssueHunt username in the description:
   > IssueHunt contributor: @your_username (https://issuehunt.io/u/your_username)
3. The maintainer reviews and merges your PR
4. IssueHunt detects the merge and **automatically releases the bounty** to your account
5. Withdraw from [issuehunt.io](https://issuehunt.io) to PayPal or bank

> **Important:** You must have a verified IssueHunt account with a payout method configured before submitting PRs.

---

## Monitoring

Open the dashboard at **http://localhost:5050** to see:

- Live agent THINK blocks (what it's reasoning about)
- Tool calls (what it's doing — searching, reading issues, writing code)
- Opportunities database
- Solutions in progress

---

## Solutions Directory

The agent posts all work here:

```
solutions/
├── active/      ← Currently being solved
├── submitted/   ← PR sent, awaiting review
├── accepted/    ← Merged and paid 💰
└── rejected/    ← Closed (lessons learned inside)
```

Each folder contains:
- `PLAN.md` — analysis and approach
- `PATCH.diff` — the code fix
- `PR_DESCRIPTION.md` — the pull request body
- `LESSONS.md` — outcome notes

---

## Configuration

Key settings in `llm/providers/hermes_tool_provider.py`:

| Setting | Default | Notes |
|---|---|---|
| `think_budget` | 200 tokens | Reasoning depth per pulse |
| `act_budget` | 1024 tokens | Code generation budget |
| `token_budget` | 2048 tokens | Full user task budget |
| `MAX_TOOL_LOOPS` | 5 | Max tool calls per pulse |

Pulse intervals in `core/pulse_loop.py`:

| State | Interval | Trigger |
|---|---|---|
| ACTIVE | 10s | User message received |
| REGULAR | 30s | Autonomous work mode |
| RESTING | 60s | Idle / waiting |

---

## Troubleshooting

**Agent says "awaiting direction" instead of hunting bounties**
→ Wipe session state: `python3 -c "import os; [open(f,'w').write('') for f in ['data/curiosity_knowledge.jsonl','data/experience_tuples.jsonl']]"` then restart.

**GitHub search returns 401**
→ Your `GITHUB_TOKEN` isn't loading. Check `.env` exists and `GITHUB_TOKEN` is set correctly.

**Out of VRAM (OOM error)**
→ The model needs ~5.6GB VRAM in 4-bit mode. Close other GPU processes first.

**Model download fails**
→ Run `huggingface-cli login` and ensure your HF token has read access.

---

## Architecture

```
main.py                  ← Entry point, wires all components
core/
  pulse_loop.py          ← Main agent loop (THINK → ACT cycle)
  preconscious.py        ← Belief/memory injection before each pulse
llm/providers/
  hermes_tool_provider.py← Hermes-3 inference + tool calling
tools/
  issuehunt.py           ← IssueHunt bounty search
  github_api.py          ← GitHub read/write operations
  tool_executor.py       ← Dispatches tool calls
dashboard/
  dashboard.py           ← Web UI at localhost:5050
data/
  opportunities.json     ← Discovered bounties database
solutions/               ← All agent work product
```

---

## License

MIT — use freely, contribute back.

---

*Built on [Helix-AGI](https://github.com/batteryphil/bounty-helix) • Powered by Hermes-3-Llama-3.1-8B*
