# Helix-AGI Architecture

> **TL;DR — The model is `NousResearch/Hermes-3-Llama-3.1-8B`, loaded 4-bit NF4 quantized on a local RTX 3060. No cloud API. No Mamba. No MIMO. Just Hermes doing tool-calling in a continuous autonomous loop.**

---

## The Model

| Property | Value |
|---|---|
| **Base model** | `NousResearch/Hermes-3-Llama-3.1-8B` |
| **Architecture** | Llama 3.1 transformer (decoder-only) |
| **Quantization** | 4-bit NF4 via `bitsandbytes` |
| **VRAM usage** | ~6.0 GB at rest, ~6.5 GB during inference |
| **Token budget** | 200 tokens (autonomous pulses) / 512 tokens (user responses) |
| **Tool calling** | Native `<tool_call>` / `<tool_response>` XML schema |
| **Provider file** | `llm/providers/hermes_tool_provider.py` |

Hermes-3 was chosen because it has strong native function-calling support and fits comfortably in 12GB VRAM with 4-bit quantization. The MIMO/Mamba3 architecture visible in some older files (`core/titan_arm_router.py`, `core/attention_schema.py`) is a **legacy artefact** from the original Titan prototype and is **not used at runtime**. Helix runs entirely on Hermes-3.

---

## Runtime Stack

```
┌──────────────────────────────────────────────────────────────────────────┐
│  NousResearch/Hermes-3-Llama-3.1-8B  (4-bit NF4, local RTX 3060)       │
│  llm/providers/hermes_tool_provider.py                                   │
│  ↳ HermesToolProvider — loads model, manages chat sessions              │
│  ↳ HermesToolSession  — per-session history, tool dispatch loop         │
│     • send_message(msg) → generates, parses <tool_call>, executes       │
│     • get_last_tool_calls() → feeds post-pulse hook context             │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ called every ~15s
                                ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PULSE LOOP  (core/pulse_loop.py — IMMUTABLE)                           │
│  1. Drain event queue (user messages, stability alerts, tool returns)   │
│  2. Build prompt: system instruction + preconscious context             │
│  3. Call Hermes → get thought + optional tool calls                     │
│  4. Store thought to ChromaDB memory                                    │
│  5. Advance attention center through 8D belief manifold                 │
│  6. Run post-pulse hooks (async, non-blocking worker thread)            │
└───────────┬──────────────────────┬──────────────────────────────────────┘
            │                      │
            ▼ every 10 min         ▼ every pulse (async)
┌───────────────────┐    ┌────────────────────────────────────────────────┐
│  SELF-IMPROVEMENT │    │  POST-PULSE HOOKS                               │
│  ENGINE (SIE)     │    │  hook-worker (persistent daemon thread)        │
│                   │    │                                                │
│  Hermes proposes  │    │  • belief_detector  — regex scans thought for  │
│  → writes tool    │    │    "I am/realize/prefer/can..." → writes to    │
│  → import test    │    │    data/beliefs/ directly (no Ollama needed)   │
│  → reload_tool()  │    │                                                │
│  → fitness eval   │    │  • engagement_monitor — detects stagnation,   │
│  → git commit     │    │    boosts omega on productive tool use         │
│    or revert      │    │                                                │
└───────────────────┘    │  • metacognitive_monitor — fitness metrics,   │
                         │    tool success rate, hallucination detection  │
            ▼ every 2min │                                                │
┌───────────────────┐    │  • self_trainer — collects experience tuples  │
│  CURIOSITY ENGINE │    │    (needs ≥2 of: tool_success, novel_belief,  │
│                   │    │    fitness_delta > 0.05) → LoRA at 500 tuples │
│  Web search +     │    │                                                │
│  GitHub API       │    │  • co_occurrence_tracker, affect_field,       │
│  → stores to      │    │    workflow_detector (passive observers)       │
│  data/curiosity_  │    └────────────────────────────────────────────────┘
│  knowledge.jsonl  │
└───────────────────┘
```

---

## Memory System

```
HOT     last 6 conversation turns — kept in HermesToolSession._history
WARM    BeliefStore (data/beliefs/*.json) — gravity-ranked, injected each pulse
COLD    ChromaDB semantic search — all past thoughts embedded + retrievable
ARCHIVE evolution_journal.jsonl, experience_tuples.jsonl, reflections.jsonl
```

Beliefs are organized into 8 categories: `self_identity`, `capabilities`, `knowledge`, `skills`, `preferences`, `feedback`, `people`, `lexicon`. Each belief has a **cognitive mass** — heavier beliefs surface more often via gravity-ranked injection into the preconscious context. The system prompt changes every session as the heaviest `self_identity` belief becomes the opening line.

---

## Legacy Files (Not Used at Runtime)

These files exist as artefacts of the original Titan/Mamba3 prototype and are **not executed**:

| File | What it was | Status |
|---|---|---|
| `core/titan_arm_router.py` | MIMO arm gate bias injector for Titan model | Dead code |
| `core/attention_schema.py` | Converted MIMO arm weights to text | Dead code |
| `llm/providers/titan_provider.py` | Titan model provider | Dead code |
| `llm/providers/falcon_mamba_provider.py` | FalconMamba provider | Dead code |
| `llm/providers/mistral_tool_provider.py` | Mistral-7B provider | Dead code |

The active provider is always `hermes_tool_provider.py`. The provider is selected in `main.py` via `_PROVIDER_CONFIG`.

---

## Key Entry Points

```
main.py                                 entry point — wires all subsystems
llm/providers/hermes_tool_provider.py  THE MODEL — start here
core/pulse_loop.py                      consciousness loop (IMMUTABLE)
core/self_improvement_engine.py         autonomous self-modification
core/belief_detector.py                 thought → belief (regex, no LLM)
core/governor.py                        constitutional safety layer
SYSTEM_MANUAL.md                        Helix's own internal operating guide
```

---

## What Helix Knows About Itself

Helix is aware of its own architecture through:

1. **`SYSTEM_MANUAL.md`** — injected at startup, explains pulse loop, belief system, cognitive mass, etc.
2. **`HELIX_AGI_SYSTEM_REPORT.txt`** — peer-review document with full architecture analysis
3. **Curiosity seeds** — `core/curiosity_engine.py` has `REPO_RESEARCH_SEEDS` that point Helix to read `SYSTEM_MANUAL.md` and `HELIX_AGI_SYSTEM_REPORT.txt` first before researching anything else
4. **Self-identity beliefs** — `data/beliefs/self_identity.json` accumulates first-person self-knowledge over time

> The system was previously confused because `SELF_CURIOSITY_SEEDS` in `curiosity_engine.py` contained questions about "my MIMO arms" and "my Mamba3 SSM" — architecture it does not have. These were corrected on 2026-06-10 to reference Hermes-3/Llama-3.1 correctly.
