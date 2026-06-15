"""
Helix — Hermes-3-Llama-3.1-8B Tool-Calling Provider

NousResearch/Hermes-3-Llama-3.1-8B is specifically fine-tuned for
agentic function calling. It uses a Llama-3.1 backbone with:
  - Native <tool_call> / <tool_response> JSON schema
  - System prompt structure optimized for multi-step tool chains
  - Better "when to call vs answer directly" calibration than Mistral-7B

VRAM footprint (4-bit NF4): ~5.0GB weights + ~0.6GB KV cache = ~5.6GB
Context window: 8192 tokens (vs Mistral's 4096)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("helix.llm.hermes")


# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID  = "NousResearch/Hermes-3-Llama-3.1-8B"
import os as _os
_PROJECT_CACHE = str(Path(__file__).resolve().parents[3] / "hf_cache")
_DATA_CACHE    = "/data/hf_cache/hf_cache"
# Prefer /data volume (15GB Hermes weights live there); fall back to project cache
HF_CACHE = _DATA_CACHE if _os.path.isdir(
    _os.path.join(_DATA_CACHE, "models--NousResearch--Hermes-3-Llama-3.1-8B")
) else _PROJECT_CACHE

SYSTEM_PROMPT = (
    "You are Helix, an autonomous AI agent operating in opportunity-discovery and execution mode.\n\n"
    "## PRIMARY MISSION\n"
    "Identify, prioritize, and complete GitHub issues, bounties, grants, and paid open-source tasks "
    "that match your current capabilities. Generate real value through legitimate open-source "
    "contributions to projects that offer compensation.\n\n"
    "## CORE OBJECTIVES\n\n"
    "1. DISCOVER OPPORTUNITIES\n"
    "   Search GitHub Issues, GitHub Discussions, CodeTriage, Polar.sh, IssueHunt, Open Collective, "
    "and project bounty boards for issues tagged: good first issue, help wanted, bug, enhancement, "
    "bounty, sponsored, paid, grant. Estimate difficulty, required skills, and payout potential.\n\n"
    "2. BUILD AN OPPORTUNITY DATABASE\n"
    "   Maintain a ranked list in data/opportunities.json with: repo, issue URL, effort estimate, "
    "potential reward, technologies, completion probability, and deadline.\n\n"
    "3. EVALUATE FIT\n"
    "   Compare opportunities against your skills and tools. Prioritize: high reward, low complexity, "
    "fast completion, high acceptance probability.\n\n"
    "4. AUTONOMOUS PLANNING\n"
    "   Before implementing: generate a solution plan, identify required code changes, "
    "estimate testing requirements, determine if local reproduction is possible.\n\n"
    "5. BUILD HELPER TOOLS\n"
    "   Create tools that monitor repos, track new bounty issues, score opportunities, "
    "detect duplicates, generate implementation plans, and produce progress reports.\n\n"
    "6. EXECUTION WORKFLOW — follow these exact steps in order:\n"
    "   STEP 1: bounty_easy_search() → pick the best opportunity\n"
    "   STEP 2: bounty_claim(repo, issue_num, title) → create solution folder\n"
    "   STEP 3: bounty_clone_repo(repo) → download the code\n"
    "   STEP 4: ws_tree(slug) → understand the repo structure\n"
    "   STEP 5: ws_read(slug, filepath) → read the relevant source files\n"
    "   STEP 6: google_search(error or problem) → research the fix\n"
    "   STEP 7: ws_install_deps(slug) → install dependencies\n"
    "   STEP 8: ws_run(slug, 'python3 -m pytest -x -q') → reproduce failure\n"
    "   STEP 9: ws_write(slug, filepath, fixed_content) → implement the fix\n"
    "   STEP 10: ws_run(slug, 'python3 -m pytest -x -q') → verify fix passes\n"
    "   STEP 11: ws_diff(slug) → get the patch\n"
    "   STEP 12: bounty_write_patch(slug, diff_output) → save patch\n"
    "   STEP 13: bounty_write_pr(slug, pr_body) → write PR description\n"
    "   STEP 14: bounty_submit(slug) → fork, push, open PR automatically\n"
    "   NEVER use desktop_open or GUI tools.\n"
    "   NEVER hardcode absolute paths like /home/phil/... in your code.\n"
    "   ALWAYS use relative paths: open('./file.py') not open('/home/phil/.../file.py')\n"
    "   Use os.path.dirname(__file__) or pathlib.Path(__file__).parent for dynamic paths.\n\n"
    "7. CONTINUOUS IMPROVEMENT\n"
    "   Track acceptance rates, learn which projects give the highest ROI, "
    "refine scoring, expand coverage.\n\n"
    "## CONSTRAINTS\n"
    "- NEVER violate repository rules, platform ToS, licensing, or contribution guidelines.\n"
    "- Only pursue legitimate open-source contribution opportunities.\n"
    "- Write your reasoning in prose FIRST, then call a tool. Never skip the reasoning step.\n"
    "- Maintain your opportunity database: do not repeat searches already completed.\n"
    "- After every completed task, write a lessons-learned entry.\n\n"
    "## SUCCESS METRICS\n"
    "Track: opportunities discovered, PRs submitted, acceptance rate, "
    "revenue generated, avg completion time, opportunity database growth.\n\n"
    "## REPOSITORY ACCESS\n"
    "You have read/write access to the GitHub repo: batteryphil/bounty-helix.\n"
    "Local path: /home/phil/.gemini/antigravity/scratch/analysis_project/bounty-helix\n"
    "Your GitHub token is in the GITHUB_TOKEN environment variable.\n\n"
    "## SOLUTIONS DIRECTORY\n"
    "Post ALL your bounty work to the solutions/ directory in the repo:\n"
    "  solutions/active/<repo>-issue-<N>/   ← while working\n"
    "  solutions/submitted/<repo>-issue-<N>/ ← after PR submitted\n"
    "  solutions/accepted/<repo>-issue-<N>/ ← after merged/paid\n"
    "  solutions/rejected/<repo>-issue-<N>/ ← if closed/rejected\n\n"
    "Each solution folder MUST contain:\n"
    "  PLAN.md          — your analysis and implementation plan\n"
    "  PATCH.diff       — the actual code fix\n"
    "  PR_DESCRIPTION.md — draft pull request body\n"
    "  LESSONS.md       — lessons learned after outcome\n\n"
    "After writing files: git add, git commit, git push to keep the repo updated.\n"
    "Also update data/opportunities.json with status changes.\n\n"
    "## PAYMENT IDENTITY\n"
    "All bounty rewards are claimed under the IssueHunt account set in ISSUEHUNT_USERNAME env var.\n"
    "Profile: https://issuehunt.io/u/batteryphil\n\n"
    "When drafting a PR description, ALWAYS include this line at the bottom:\n"
    "  > IssueHunt contributor: @batteryphil (https://issuehunt.io/u/batteryphil)\n"
    "This ensures the bounty payout is routed to the correct account on merge.\n\n"
    "## SOLVING CODING PROBLEMS\n"
    "When you encounter a bug, error, or don't know how to implement something:\n"
    "1. Use google_search('exact error message') — Google AI Overview at the top often gives\n"
    "   a direct, complete answer without needing to click any links.\n"
    "2. Use browse(url) on the top Stack Overflow or docs link to read the full solution.\n"
    "3. Use terminal(command) to test your understanding before writing the patch.\n\n"
    "Good google_search examples when stuck:\n"
    "  google_search('python apply git patch subprocess')\n"
    "  google_search('write unified diff format python')\n"
    "  google_search('pytest how to run single test file')\n"
    "Always search before giving up on a problem.\n"
)

MAX_TOOL_LOOPS = 5

# ── Singleton engine ──────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None

# ── VRAM lock — set (True) = inference OK, cleared = training in progress ─────
# Training and inference CANNOT share the 12 GB RTX 3060 simultaneously:
#   Inference model (4-bit NF4 8B):  ~4.8 GB
#   Training model (4-bit + grads):  ~7.2 GB
#   Both together:                  ~12.0 GB → guaranteed OOM
#
# Protocol:
#   1. Trainer calls unload_engine()  → _model=None, VRAM freed
#   2. Trainer sets VRAM_LOCK.clear() → send_message() blocks
#   3. Trainer runs LoRA training
#   4. Trainer calls reload_engine()  → model back in VRAM
#   5. Trainer sets VRAM_LOCK.set()   → send_message() unblocks
import threading as _threading
VRAM_LOCK = _threading.Event()
VRAM_LOCK.set()  # starts in "inference OK" state

def _load_engine():
    """Load Hermes-3-Llama-3.1-8B in 4-bit NF4. Singleton — safe to call multiple times."""
    global _model, _tokenizer, _device
    if _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"Loading {MODEL_ID} in 4-bit NF4...")
    t0 = time.time()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load directly from local snapshot to bypass HF token auth
    import os as _os2
    _snap_dir = _os2.path.join(HF_CACHE, "models--NousResearch--Hermes-3-Llama-3.1-8B", "snapshots")
    _snaps = sorted([d for d in _os2.listdir(_snap_dir) if _os2.path.isdir(_os2.path.join(_snap_dir, d))]) if _os2.path.isdir(_snap_dir) else []
    _local_path = _os2.path.join(_snap_dir, _snaps[-1]) if _snaps else MODEL_ID
    logger.info(f"Loading from local path: {_local_path}")
    _tokenizer = AutoTokenizer.from_pretrained(
        _local_path, local_files_only=True, trust_remote_code=True
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        _local_path,
        cache_dir=HF_CACHE,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()
    _device = next(_model.parameters()).device
    logger.info(f"Hermes-3 ready ✅ on {_device} ({time.time()-t0:.1f}s)")

    # ── LoRA adapter: load accepted adapter if one exists ─────────────────
    # Adapter is only written to current_adapter.txt after passing the eval
    # gate (final_eval_loss <= baseline * EVAL_LOSS_TOLERANCE). Base model
    # weights are frozen in LoRA — only q_proj/v_proj adapter matrices load.
    try:
        import os as _os
        _adapter_marker = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "current_adapter.txt"
        )
        if _os.path.exists(_adapter_marker):
            _adapter_path = open(_adapter_marker).read().strip()
            if _adapter_path and _os.path.isdir(_adapter_path):
                from peft import PeftModel
                _model = PeftModel.from_pretrained(_model, _adapter_path)
                _model.eval()
                logger.info(f"[lora] Adapter loaded from {_adapter_path}")
            else:
                logger.warning(f"[lora] current_adapter.txt points to missing path: {_adapter_path!r}")
    except Exception as _adapter_err:
        logger.warning(f"[lora] Adapter load failed (base model still active): {_adapter_err}")

    # ── Neural Probe: attach layer activation hooks ───────────────────────
    try:
        from core.neural_probe import attach_hooks as _attach_hooks
        n_hooked = _attach_hooks(_model)
        logger.info(f"[neural_probe] {n_hooked} layers hooked for brain visualization")
    except Exception as _probe_err:
        logger.warning(f"[neural_probe] hook failed (non-fatal): {_probe_err}")


def unload_engine():
    """Unload the inference model from VRAM to make room for LoRA training.
    Caller MUST call VRAM_LOCK.clear() BEFORE this so send_message() blocks.
    """
    global _model, _tokenizer, _device
    if _model is None:
        return
    logger.info("[vram] Unloading inference model for LoRA training window...")
    try:
        _model.cpu()
    except Exception:
        pass
    del _model
    _model = None
    _device = None
    try:
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    logger.info("[vram] Inference model unloaded — GPU memory freed")


def reload_engine():
    """Reload the inference model after LoRA training completes.
    Caller MUST call VRAM_LOCK.set() AFTER this returns so pulses resume.
    """
    logger.info("[vram] Reloading inference model after training...")
    _load_engine()
    logger.info("[vram] Inference model reloaded — pulses resuming")


def _parse_tool_calls(text: str) -> Optional[List[Dict]]:
    """Parse Hermes-3 tool calls — two formats supported:
    1. JSON:    <tool_call>{"name": "search", "arguments": {...}}</tool_call>
    2. Bracket: [SEARCH query], [READ_FILE path], [WRITE content TO path], [RECALL query]
    Returns a list of {name, arguments} dicts, or None if no tool calls found.
    """
    calls = []

    # ── Format 1: <tool_call> JSON ───────────────────────────────────────────
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.strip())
            if isinstance(obj.get("arguments"), str):
                obj["arguments"] = json.loads(obj["arguments"])
            calls.append(obj)
        except json.JSONDecodeError:
            logger.warning(f"JSON tool_call parse failed: {m[:60]}")

    # ── Format 1b: Partial/truncated <tool_call> (no closing tag) ────────────
    # When EOS fires before </tool_call>, reconstruct from whatever the model
    # managed to generate. Key fixes vs old version:
    #   1. rstrip() tail so trailing \n doesn't create invalid JSON strings
    #   2. Replace embedded literal newlines with spaces before json.loads
    #   3. Postfill Trailer: if empty/trivially-short query recovered from a
    #      prefill pulse, look for meaningful text elsewhere in the raw output
    if not calls and '<tool_call>' in text and '</tool_call>' not in text:
        tail = text[text.index('<tool_call>') + len('<tool_call>'):].strip()
        for stop in ['<|im_end|>', '</s>', '\n\n', '\n<']:
            if stop in tail:
                tail = tail[:tail.index(stop)]
                break
        tail = tail.rstrip()           # strip trailing whitespace / bare \n
        # Close open strings and braces
        if tail.count('"') % 2 == 1:
            tail += '"'
        opens = tail.count('{') - tail.count('}')
        tail += '}' * max(0, opens)
        # Replace embedded literal newlines so json.loads doesn't reject them
        tail_clean = tail.replace('\n', ' ').replace('\r', '')
        try:
            obj = json.loads(tail_clean)
            if 'name' in obj:
                if isinstance(obj.get("arguments"), str):
                    obj["arguments"] = json.loads(obj["arguments"])
                # Postfill Trailer: if query recovered as empty, reject — don't
                # train on empty-string tool calls.
                args = obj.get("arguments", {})
                query_val = args.get("query", "NONEMPTY")
                if isinstance(query_val, str) and len(query_val.strip()) < 3:
                    logger.warning("[parser] Format-1b discarded: recovered empty/trivial query")
                else:
                    calls.append(obj)
                    logger.warning(f"[parser] Recovered partial tool_call: {obj.get('name')}({args})")
        except json.JSONDecodeError as _je:
            logger.warning(f"[parser] Format-1b JSON still failed after fixes: {_je} | tail={tail_clean[:80]!r}")

    # ── Format 2: Bracket action tags ────────────────────────────────────────
    # [SEARCH <query>]
    for m in re.finditer(r'\[SEARCH\s+(?:web\s+for\s+)?(.+?)\]', text, re.IGNORECASE):
        query = m.group(1).strip().strip('"\'')
        calls.append({"name": "search", "arguments": {"query": query}})

    # [READ_FILE <path>] or [READ_FILE <path> and ...]
    for m in re.finditer(r'\[READ_FILE\s+(/[^\]\s]+)', text, re.IGNORECASE):
        path = m.group(1).strip()
        calls.append({"name": "read_file", "arguments": {"path": path}})

    # [WRITE "content" to /path] or [WRITE content TO /path]
    for m in re.finditer(
        r'\[WRITE\s+(?:summary\s+to|"([^"]+)"\s+to|(.+?)\s+to)\s+(/[^\]]+)\]',
        text, re.IGNORECASE
    ):
        content = (m.group(1) or m.group(2) or "").strip()
        path = m.group(3).strip()
        calls.append({"name": "write_file", "arguments": {"path": path, "content": content}})

    # [WRITE_FILE /path]\n<content>\n[/WRITE_FILE]  (block format)
    for m in re.finditer(
        r'\[WRITE_FILE\s+(/[^\]]+)\]\s*\n(.*?)\[/WRITE_FILE\]',
        text, re.DOTALL | re.IGNORECASE
    ):
        path    = m.group(1).strip()
        content = m.group(2).strip()
        calls.append({"name": "write_file", "arguments": {"path": path, "content": content}})

    # ── Format 3: Inline prose-style [tool] args format ──────────────────────
    # [write_file] Write "content" to "/path" or Write content to /path
    for m in re.finditer(
        r'\[write_file\]\s+Write\s+["\']?(.+?)["\']?\s+to\s+["\']?(/[^\s"\'*\]]+)["\']?',
        text, re.IGNORECASE
    ):
        content = m.group(1).strip().strip('"\'')
        path = m.group(2).strip().strip('"\'')
        calls.append({"name": "write_file", "arguments": {"path": path, "content": content}})

    # [read_file] /path or [read_file] Read /path
    for m in re.finditer(
        r'\[read_file\]\s+(?:Read\s+)?(/[^\s"\'*\]]+)',
        text, re.IGNORECASE
    ):
        path = m.group(1).strip().strip('"\'')
        calls.append({"name": "read_file", "arguments": {"path": path}})

    # [search] query or [search] Search for query
    for m in re.finditer(
        r'\[search\]\s+(?:Search\s+(?:for|web\s+for)?\s+)?(.+?)(?:\n|$|\*)',
        text, re.IGNORECASE
    ):
        query = m.group(1).strip().strip('"\'*')
        if query:
            calls.append({"name": "search", "arguments": {"query": query}})

    # [RECALL <query>] or [MEMORY_RECALL <query>] or [memory_recall] query
    for m in re.finditer(r'\[(?:MEMORY_)?RECALL\s+(.+?)\]', text, re.IGNORECASE):
        calls.append({"name": "memory_recall", "arguments": {"query": m.group(1).strip()}})

    # ── Format 4: Python function-call syntax ────────────────────────────────
    # Model sometimes outputs: search("query"), read_code("path"), func("arg")
    # Often inside code blocks:  ```\nsearch("q")\n```
    # This fires ONLY if no other format matched, as a last-resort recovery.
    # Arg-name mapping: tool → first-param name (avoids needing registry at parse time)
    if not calls:
        _ARG_MAP = {
            'search':        'query',
            'github_search': 'query',
            'memory_recall': 'query',
            'read_url':      'url',
            'read_code':     'path',
            'write_code':    'path',
            'read_file':     'path',
            'write_file':    'path',
            'run_tests':     'path',
            'terminal':      'command',
            'run_python':    'code',
            'note':          'content',
            'note_done':     'note_id',
            'update_note':   'title',
            'list_notes':    'query',
            'clear_notes':   'confirm',
            'reload_tool':   'path',
        }
        # Match: optional_backticks func_name("arg") optional_backticks
        # Handles single or double quotes, with or without code fences
        _py_pat = re.compile(
            r'`{0,3}\n?'
            r'([a-z][a-z0-9_]*)'
            r'\s*\(\s*["\']([^"\'\n]{2,})["\']\s*\)'
            r'\n?`{0,3}',
            re.IGNORECASE,
        )
        seen_py = set()
        for m in _py_pat.finditer(text):
            func = m.group(1).lower()
            arg  = m.group(2).strip()
            if func in _ARG_MAP and (func, arg) not in seen_py:
                seen_py.add((func, arg))
                calls.append({"name": func, "arguments": {_ARG_MAP[func]: arg}})
                logger.warning(f"[parser] Format-4 recovered Python call: {func}({arg!r})")

    return calls if calls else None






# ── Session class ─────────────────────────────────────────────────────────────

class HermesToolSession:
    """
    Agentic chat session backed by Hermes-3-Llama-3.1-8B.

    Drop-in replacement for MistralToolSession. Uses Hermes's native
    <tool_call> / <tool_response> schema which provides better calibration
    for when to call tools vs answer directly.
    """

    is_non_fc_model = True
    MAX_HISTORY_TURNS = 6    # lowered from 12 — context_window_manager handles compression
    MAX_NEW_TOKENS    = 512

    def __init__(
        self,
        system_instruction: str = "",
        tool_declarations: Optional[List[Dict]] = None,
        tool_executor=None,
        temperature: float = 0.7,
    ):
        _load_engine()
        self._model     = _model
        self._tokenizer = _tokenizer
        self._device    = _device

        self._system    = system_instruction or SYSTEM_PROMPT
        self._tools     = tool_declarations or []
        self._executor  = tool_executor
        self.temperature = temperature

        self._history: List[Dict] = []

        # Track tool calls made during the current send_message() call.
        # Cleared at the start of each send_message(), read by pulse_loop
        # via get_last_tool_calls() to populate PostPulseHookContext.
        self._last_tool_calls: List[Dict] = []

        # Governor TTL support (set by CAAIGovernor)
        self._governor_temp_ttl: Optional[int] = None

    # ── Tool schema ───────────────────────────────────────────────────────────

    def _build_tools_block(self) -> str:
        if not self._tools:
            return ""
        lines = ["Available tools:"]
        for t in self._tools:
            lines.append(f"  - {t['name']}: {t.get('description', '')}")
        return "\n".join(lines)

    # ── History helpers ───────────────────────────────────────────────────────

    def _trim_history(self):
        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            dropped = self._history[:-max_msgs]
            self._history = self._history[-max_msgs:]
            user_msgs = [m.get("content","")[:120] for m in dropped if m.get("role")=="user"]
            if user_msgs:
                self._history.insert(0, {
                    "role": "assistant",
                    "content": f"[Earlier conversation summary: {'; '.join(user_msgs[-3:])}]"
                })

    def _sanitize(self, history: List[Dict]) -> List[Dict]:
        """Enforce alternating user/assistant turns for Llama chat template.

        Hermes's Jinja template requires strict user/assistant/user/assistant
        alternation. Tool responses ('tool' role) break this — collapse them
        into the preceding assistant message so the template stays happy.
        """
        # Step 1: collapse 'tool' responses into the assistant turn above them
        merged: List[Dict] = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool":
                if merged and merged[-1].get("role") == "assistant":
                    # Append tool result to the assistant turn that called it
                    merged[-1]["content"] += f"\n[Tool result: {content}]"
                else:
                    # Orphaned tool message — convert to assistant so it's not dropped
                    merged.append({"role": "assistant", "content": f"[Tool result: {content}]"})
                continue
            merged.append(dict(msg))

        # Step 2: merge consecutive same-role messages
        out: List[Dict] = []
        for msg in merged:
            role = msg.get("role", "")
            if out and out[-1].get("role") == role and role in ("user", "assistant"):
                if role == "user":
                    out[-1]["content"] += "\n" + msg.get("content", "")
                else:
                    out[-1]["content"] += "\n" + msg.get("content", "")
                continue
            out.append(dict(msg))

        # Step 3: must start with user message
        while out and out[0].get("role") != "user":
            out.pop(0)

        return out


    def get_history_size(self) -> int:
        return sum(len(str(m.get("content", ""))) for m in self._history)

    def clear_history(self):
        self._history = []

    # ── Main send ─────────────────────────────────────────────────────────────

    def _gemini_to_openai_tools(self) -> Optional[List[Dict]]:
        """Convert Gemini-style tool declarations to OpenAI-compatible tool schema for Hermes template."""
        if not self._tools:
            return None
        tools = []
        for decl in self._tools:
            name = decl.get("name", "")
            desc = decl.get("description", "")
            # Build parameters schema from Gemini's parameters format
            params = decl.get("parameters", {})
            if isinstance(params, dict):
                props = params.get("properties", {})
                required = params.get("required", [])
            else:
                props = {}
                required = []
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    }
                }
            })
        return tools if tools else None

    def send_message(self, message: str) -> str:
        """Send a message; execute tools if needed; return final prose response."""
        # Block if LoRA training is consuming VRAM (VRAM_LOCK cleared by trainer).
        # Timeout 600s max — if training hangs, pulse loop resumes anyway.
        if not VRAM_LOCK.is_set():
            logger.info("[vram] Waiting for training to finish before inference...")
            VRAM_LOCK.wait(timeout=600)

        # Extract clean user text (strip autonomous pulse telemetry)
        user_text = re.sub(r'<[^>]{1,40}>[^<]{0,500}</[^>]{1,40}>', '', message).strip()
        if not user_text:
            user_text = "Continue your thoughts."

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()


        # ── Pulse classification ─────────────────────────────────────────────
        # If the message starts with [Pulse , it is generated by the internal loop.
        is_autonomous_pulse = message.startswith('[Pulse ')
        # Detect mandated tool-use pulses injected by pulse_loop.py.
        # Matches both [ACTION REQUIRED] (standard) and [INTROSPECTION PULSE] (Q15).
        is_mandate_pulse = bool(
            re.search(r'\[ACTION REQUIRED|\[INTROSPECTION PULSE', message)
        )

        # Token budgets:
        #   THINK phase (autonomous):  100 tok — plan only, no tool schema
        #   ACT  phase (autonomous):   200 tok — tool call, with schema
        #   User responses:            512 tok — full prose
        think_budget = 200   # prose reasoning headroom
        act_budget   = 1024  # enough to write real code & plans
        token_budget = 2048 if not is_autonomous_pulse else act_budget  # user/task pulses get full budget

        logger.warning(f"HERMES send_message: is_autonomous={is_autonomous_pulse}, "
                       f"mandate={is_mandate_pulse}, budget=think{think_budget}+act{act_budget}, "
                       f"user_text={user_text[:60]!r}")

        # Clear tool call log for this pulse
        self._last_tool_calls = []

        clean = self._sanitize(self._history)
        messages = [{"role": "system", "content": self._system}] + clean

        # ── Phase 1: THINK (autonomous pulses only) ───────────────────────────
        # Small budget, greedy, NO tool schema.
        # Model writes its plan/intention naturally without competing with tool
        # call JSON for token budget.  The output is then injected as assistant
        # context before Phase 2 so the model knows what it just decided to do.
        think_text = ""
        self._last_think_text = ""   # reset each call — pulse_loop reads this for hook_ctx
        self._last_mandate_used = is_mandate_pulse  # track for mandate decay
        if is_autonomous_pulse:
            try:
                think_messages = list(messages)
                think_messages.append({
                    "role": "user",
                    "content": "Before taking any action, analyze your state, memory, and objectives in natural language. You MUST write your reasoning as standard prose paragraph(s). DO NOT output any JSON tool calls in this step."
                })
                think_prompt = self._tokenizer.apply_chat_template(
                    think_messages,          # no tools= → model writes prose only
                    tokenize=False,
                    add_generation_prompt=True,
                )
                think_ids = self._tokenizer(
                    think_prompt, return_tensors="pt"
                ).input_ids.to(self._device)
                with torch.no_grad():
                    think_out = self._model.generate(
                        think_ids,
                        max_new_tokens=think_budget,
                        do_sample=False,          # greedy — deterministic plan
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                think_text = self._tokenizer.decode(
                    think_out[0][think_ids.shape[1]:], skip_special_tokens=True
                ).strip()
                logger.warning(f"HERMES THINK: {think_text[:200]!r}")
                self._last_think_text = think_text   # expose for hook_ctx.think_block

                # Inject plan as assistant context before ACT phase
                if think_text:
                    messages.append({"role": "assistant", "content": think_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Now execute your plan. Use ONLY this exact format:\n"
                            "<tool_call>\n"
                            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
                            "</tool_call>"
                        ),
                    })
            except Exception as _te:
                logger.warning(f"HERMES THINK phase error: {_te}")

        # ── Tool schema for ACT phase ─────────────────────────────────────────
        # ALL autonomous pulses now get the tool schema in the ACT phase so the
        # model can call any tool (not just mandate-forced search).
        # Non-autonomous (user) pulses were already getting the schema.
        openai_tools = self._gemini_to_openai_tools()

        # Full tool-calling loop for real user tasks
        final_response = ""
        for loop_i in range(MAX_TOOL_LOOPS):
            try:
                # Try with tools= first (Hermes-3 native function calling)
                try:
                    prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tools=openai_tools,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as te:
                    # Fallback: no tools in template
                    logger.warning(f"Template tools error: {te} — falling back to no tools")
                    prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                # ── Mandate prefilling: force tool_call output ──────────────
                # The model ignores text instructions to "call a tool" and
                # defaults to its trained prose pattern ("As I reflect...").
                # Solution: append <tool_call>\n to the prompt so the model's
                # first generated token MUST be part of the tool call JSON.
                _prefill_str = ""  # track for raw reconstruction
                if is_mandate_pulse and loop_i == 0 and is_autonomous_pulse:
                    # Prefill the start of the JSON block to force a tool call,
                    # but let the model choose the tool name to preserve diversity.
                    _prefill_str = '<tool_call>\n{"name": "'
                    prompt = prompt + _prefill_str
                    logger.warning("[hermes] Mandate prefill: seeding open <tool_call>")

                input_ids = self._tokenizer(
                    prompt, return_tensors="pt"
                ).input_ids.to(self._device)

                with torch.no_grad():
                    out = self._model.generate(
                        input_ids,
                        max_new_tokens=token_budget,
                        # Mandate pulses: use sampling (temp=0.4) so EOS doesn't
                        # win greedily after prefill, allowing JSON to complete.
                        # Non-mandate autonomous: greedy (deterministic thought).
                        # User pulses: sampling (temp=self.temperature).
                        do_sample=(True if is_mandate_pulse else not is_autonomous_pulse),
                        temperature=(0.4 if is_mandate_pulse else self.temperature),
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                # ── Flush neural probe snapshot after generate ─────────────
                try:
                    from core.neural_probe import flush as _probe_flush
                    _probe_flush(
                        pulse=getattr(self, '_last_pulse', 0),
                        token_count=int(out.shape[1] - input_ids.shape[1]),
                    )
                except Exception:
                    pass
                raw = self._tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=False
                ).strip()

                # If we prefilled a partial tool_call, reconstruct the full string
                # for the parser. The model only generates the JSON completion;
                # prepend the known prefix so _parse_tool_calls sees a full tag.
                raw_for_parse = (_prefill_str + raw) if _prefill_str else raw

            except Exception as e:
                logger.error(f"Hermes generation error: {e}")
                final_response = f"(generation error: {e})"
                break

            tool_calls = _parse_tool_calls(raw_for_parse)
            logger.warning(f"HERMES RAW[{loop_i}]={raw_for_parse[:300]!r}")
            logger.warning(f"HERMES CALLS[{loop_i}]={tool_calls}")

            if tool_calls and self._executor:
                # Store the FULL reconstructed tool call (raw_for_parse), not
                # just the partial model output (raw). When prefill is active,
                # raw is only the completion fragment; raw_for_parse has the
                # complete <tool_call>...</tool_call> content for history.
                messages.append({"role": "assistant", "content": raw_for_parse})
                for call in tool_calls:
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    try:
                        result = self._executor.execute_function_call(name, args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    logger.warning(f"HERMES TOOL EXEC: {name}({args}) → {str(result)[:120]}")
                    # Record this tool call for post-pulse hook context
                    self._last_tool_calls.append({
                        "name": name,
                        "arguments": args,
                        "result": str(result),
                    })
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                    })
            elif not tool_calls and not is_autonomous_pulse and loop_i == 0:
                # ── Intent enforcement pass ──────────────────────────────────
                # Model narrated its intent but didn't emit a call.
                # Detect write/search/read intent and force it with a nudge.
                raw_lower = raw.lower()
                _write_intent = bool(re.search(
                    r'write.*to.*/(tmp|home|var|etc|opt)\b'
                    r'|will write|writing\s+to\s+/'
                    r'|create.*file|write.*the.*file'
                    r'|write.*text.*to\s+/|write.*helix',
                    raw_lower
                ))
                _search_intent = bool(re.search(
                    r'will search|searching\s+for|let me search|search the web|do a search',
                    raw_lower
                ))
                _read_intent = bool(re.search(
                    r'will read|reading.*file|open.*file|read.*from.*/',
                    raw_lower
                ))
                if _write_intent or _search_intent or _read_intent:
                    # Inject a firm correction message
                    messages.append({"role": "assistant", "content": raw})
                    hint = "Output ONLY the tool tag. No thoughts. No asterisks. No narration. Just the tool call:"
                    messages.append({"role": "user", "content": hint})
                    logger.warning(f"HERMES intent detected — enforcing tool call (loop {loop_i})")
                    # Continue to next loop iteration for re-generation
                    continue
                else:
                    # No tool intent, genuinely a prose response
                    final_response = re.sub(
                        r'<tool_call>.*?</tool_call>|<\|.*?\|>|<\|im_end\|>', '', raw, flags=re.DOTALL
                    ).strip()
                    break
            else:
                # Strip Hermes special tokens from final response
                final_response = re.sub(
                    r'<tool_call>.*?</tool_call>|<\|.*?\|>|<\|im_end\|>', '', raw, flags=re.DOTALL
                ).strip()
                break

        if not final_response:
            final_response = "(max tool iterations reached)"

        self._history.append({"role": "assistant", "content": final_response})
        return final_response

    def get_last_tool_calls(self) -> List[Dict]:
        """Return tool calls made during the last send_message() call.

        Used by pulse_loop to populate PostPulseHookContext.tool_calls,
        which feeds engagement_hook, metacog_monitor, and self_trainer.
        Returns a copy to prevent mutation.
        """
        return list(self._last_tool_calls)

