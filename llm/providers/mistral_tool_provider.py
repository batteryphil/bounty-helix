"""
Helix — Mistral-7B-Instruct-v0.3 Tool-Calling Provider

Uses Mistral's native function-calling format to execute Helix tools:
  [TOOL_CALLS] [{"name": "search", "arguments": {"query": "..."}}]

This is a full agentic provider: it can loop through multiple tool calls
per turn until the model produces a final prose response.

Architecture:
  - Model: mistralai/Mistral-7B-Instruct-v0.3 (4-bit NF4, ~4.1GB VRAM)
  - KV cache grows linearly with context but resets every ~4K tokens
  - At 4K tokens: +512MB KV = ~4.6GB total — safe on RTX 3060

Tool call format (Mistral v0.3):
  [TOOL_CALLS] [{"name": "fn_name", "arguments": {...}}]

Tool result format:
  {"role": "tool", "tool_call_id": "...", "content": "..."}
"""

import json
import logging
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from llm.providers.base import ChatSession

logger = logging.getLogger("helix.llm.providers.mistral")

# ── Singleton engine ──────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None

MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
HF_CACHE  = str(Path(__file__).resolve().parents[3] / "hf_cache")

SYSTEM_PROMPT = (
    "You are Helix, an autonomous AI agent with access to tools.\n\n"
    "## Tool Use Rules\n"
    "Use tools when the question requires current, specific, or external information. "
    "Answer directly when you already know the answer from training.\n\n"
    "## Tool Call Formats (use either)\n\n"
    "JSON format:\n"
    "  [TOOL_CALLS] [{\"name\": \"search\", \"arguments\": {\"query\": \"...\"}}]\n\n"
    "Bracket format:\n"
    "  [SEARCH your query here]\n"
    "  [READ_FILE /absolute/path/to/file]\n"
    "  [WRITE_FILE /absolute/path/to/output.md]\n"
    "  Content to write goes here\n"
    "  [/WRITE_FILE]\n"
    "  [RECALL topic or keywords]\n\n"
    "## Examples\n\n"
    "EXAMPLE 1 — Search (current events):\n"
    "User: What happened at Google I/O this week?\n"
    "→ [TOOL_CALLS] [{\"name\": \"search\", \"arguments\": {\"query\": \"Google I/O 2025 announcements\"}}]\n\n"
    "EXAMPLE 2 — Read file:\n"
    "User: Summarize /home/user/notes.txt\n"
    "→ [READ_FILE /home/user/notes.txt]\n\n"
    "EXAMPLE 3 — Write file:\n"
    "User: Write the summary to /home/user/summary.md\n"
    "→ [WRITE_FILE /home/user/summary.md]\n"
    "  # Summary\n"
    "  The main points are...\n"
    "  [/WRITE_FILE]\n\n"
    "EXAMPLE 4 — Direct answer (well-known fact):\n"
    "User: What is the capital of France?\n"
    "→ Paris.\n\n"
    "Never narrate what you are about to do. Just emit the action.\n"
    "Keep responses concise."
)


MAX_TOOL_LOOPS = 3   # max tool call → result → re-generate cycles per turn


def _rand_id(n: int = 9) -> str:
    """Generate a Mistral-compatible 9-char alphanumeric tool call ID."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _load_engine():
    """Load Mistral-7B in 4-bit NF4. Idempotent singleton."""
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"Loading {MODEL_ID} in 4-bit NF4 (first load, ~10s)...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        cache_dir=HF_CACHE,
    )
    _model.eval()

    _device = next(_model.parameters()).device
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Mistral-7B loaded ✅  VRAM: {vram:.2f} GB")

    return _model, _tokenizer, _device


def _helix_to_mistral_tools(tool_declarations: list) -> list:
    """
    Convert Helix tool declarations (dict-based) to Mistral's OpenAI-style
    function schema format.

    Helix tools look like:
        {"name": "search", "description": "...", "parameters": {...}}

    Mistral expects:
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    """
    if not tool_declarations:
        return []

    # Whitelist: tools that actually work without Gemini's native FC backend
    # These are the tools the ToolExecutor can run directly
    USABLE_TOOLS = {
        "search", "read_url", "browse", "read_file", "write_file",
        "append_file", "note", "note_done", "list_notes", "clear_notes",
        "update_note", "memory_recall", "reply", "verbalize", "terminal",
        "journal", "reset_context",
    }

    result = []
    for decl in tool_declarations:
        name = decl.get("name", "")
        if name not in USABLE_TOOLS:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": decl.get("description", ""),
                "parameters": decl.get("parameters", {
                    "type": "object", "properties": {}, "required": []
                }),
            }
        })
    return result


def _parse_tool_calls(raw_text: str) -> Optional[List[Dict]]:
    """
    Extract tool calls from Mistral's output.

    Handles three formats:
    1. [TOOL_CALLS] [{"name": "...", "arguments": {...}}]  (native Mistral JSON)
    2. Bracket action tags: [SEARCH query], [READ_FILE /path], [WRITE_FILE /path]...[/WRITE_FILE]
    3. [RECALL query]
    """
    calls = []

    # ── Format 1: Mistral native [TOOL_CALLS] JSON ───────────────────────────
    m = re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])', raw_text, re.DOTALL)
    if m:
        try:
            tc = json.loads(m.group(1))
            for call in tc:
                if isinstance(call.get("arguments"), str):
                    try:
                        call["arguments"] = json.loads(call["arguments"])
                    except Exception:
                        call["arguments"] = {"input": call["arguments"]}
            calls.extend(tc)
        except Exception:
            pass

    # ── Format 2: Bracket action tags ────────────────────────────────────────
    # [SEARCH <query>]
    for m in re.finditer(r'\[SEARCH\s+(?:web\s+for\s+)?(.+?)\]', raw_text, re.IGNORECASE):
        calls.append({"name": "search", "arguments": {"query": m.group(1).strip().strip('"\''"'")}})

    # [READ_FILE /path]
    for m in re.finditer(r'\[READ_FILE\s+(/[^\]\s]+)', raw_text, re.IGNORECASE):
        calls.append({"name": "read_file", "arguments": {"path": m.group(1).strip()}})

    # [WRITE_FILE /path]\n<content>\n[/WRITE_FILE]
    for m in re.finditer(
        r'\[WRITE_FILE\s+(/[^\]]+)\]\s*\n(.*?)\[/WRITE_FILE\]',
        raw_text, re.DOTALL | re.IGNORECASE
    ):
        calls.append({"name": "write_file", "arguments": {
            "path": m.group(1).strip(), "content": m.group(2).strip()
        }})

    # [WRITE "content" to /path]
    for m in re.finditer(
        r'\[WRITE\s+(?:summary\s+to|"([^"]+)"\s+to|(.+?)\s+to)\s+(/[^\]]+)\]',
        raw_text, re.IGNORECASE
    ):
        content = (m.group(1) or m.group(2) or "").strip()
        calls.append({"name": "write_file", "arguments": {
            "path": m.group(3).strip(), "content": content
        }})

    # [RECALL query]
    for m in re.finditer(r'\[(?:MEMORY_)?RECALL\s+(.+?)\]', raw_text, re.IGNORECASE):
        calls.append({"name": "memory_recall", "arguments": {"query": m.group(1).strip()}})

    return calls if calls else None




class MistralToolSession(ChatSession):
    """
    Full agentic session using Mistral-7B-Instruct-v0.3 with native tool calling.

    Flow per user message:
      1. Build prompt with system + history + tools
      2. Generate → check for [TOOL_CALLS]
      3. If tool call: execute via ToolExecutor, append result, go to 2
      4. If prose: return final response to user

    Tool execution is real: search fires DuckDuckGo, read_url fetches pages,
    write_file writes to disk, memory_recall queries the vector store, etc.
    """

    # Signals pulse_loop to use simple text delivery path
    is_non_fc_model = True

    MAX_HISTORY_TURNS = 10
    MAX_NEW_TOKENS    = 512
    TEMPERATURE       = 0.7
    TOP_P             = 0.9

    def __init__(
        self,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        tool_declarations: list = None,
        tool_executor=None,
        **kwargs,
    ):
        self.temperature        = temperature
        self.max_output_tokens  = max_output_tokens
        self._tool_executor     = tool_executor
        self._raw_tool_decls    = tool_declarations or []
        self._mistral_tools     = _helix_to_mistral_tools(self._raw_tool_decls)

        # Keep system prompt short for local model
        if system_instruction and len(system_instruction) < 800:
            self._system = system_instruction.strip()
        else:
            self._system = SYSTEM_PROMPT

        self._history: List[Dict] = []
        self._model     = None
        self._tokenizer = None
        self._device    = None

        n_tools = len(self._mistral_tools)
        logger.info(f"MistralToolSession created — {n_tools} tools active")

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tokenizer, self._device = _load_engine()

    # ── History helpers ───────────────────────────────────────────────

    def _trim_history(self):
        """Keep only the last MAX_HISTORY_TURNS pairs; summarize dropped turns."""
        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            dropped = self._history[:-max_msgs]
            self._history = self._history[-max_msgs:]
            # Summarize dropped turns and prepend as a system note
            summary = self._summarize_old_turns(dropped)
            if summary:
                # Insert summary as the oldest assistant turn so context is preserved
                self._history.insert(0, {
                    "role": "assistant",
                    "content": f"[Earlier conversation summary: {summary}]"
                })

    def _summarize_old_turns(self, turns: List[Dict]) -> str:
        """Produce a 1-2 sentence summary of dropped history turns without
        calling the model — just extract the key user messages."""
        user_msgs = [
            m.get("content", "")[:120]
            for m in turns
            if m.get("role") == "user" and m.get("content")
        ]
        if not user_msgs:
            return ""
        joined = "; ".join(user_msgs[-3:])  # last 3 dropped user messages
        return f"Previously discussed: {joined}"

    def _sanitize_for_mistral(self, history: List[Dict]) -> List[Dict]:
        """
        Mistral's chat template requires strictly alternating user/assistant turns.
        Consecutive same-role messages (e.g. from autonomous pulses adding extra
        assistant turns) cause a TemplateError.

        Strategy:
        - Consecutive user messages: collapse into one (join with newline)
        - Consecutive assistant messages: keep only the LAST one (most recent)
        - tool / tool_calls messages: kept as-is (part of the FC sequence)
        - Ensure history starts with a user message
        """
        if not history:
            return history

        sanitized = []
        for msg in history:
            role = msg.get("role", "")
            # Tool messages always pass through
            if role == "tool" or "tool_calls" in msg:
                sanitized.append(msg)
                continue

            if sanitized:
                last = sanitized[-1]
                last_role = last.get("role", "")
                if last_role == role and role in ("user", "assistant"):
                    if role == "user":
                        # Merge consecutive user messages
                        last["content"] = last["content"] + "\n" + msg.get("content", "")
                    else:
                        # Replace old assistant turn with newer one
                        sanitized[-1] = msg
                    continue
            sanitized.append(dict(msg))

        # Must start with a user message
        while sanitized and sanitized[0].get("role") != "user":
            sanitized.pop(0)

        return sanitized

    def get_history_size(self) -> int:
        return sum(
            len(str(m.get("content", "") or m.get("tool_calls", "")))
            for m in self._history
        )

    # ── Core generation ───────────────────────────────────────────────

    def _generate(self, messages: List[Dict], greedy: bool = False, max_new_tokens: int = None) -> str:
        """Run one forward pass and return the decoded raw output text."""
        tools = self._mistral_tools if self._mistral_tools else None
        max_tokens = max_new_tokens or self.max_output_tokens
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            # Fallback: no tools if template chokes
            logger.warning(f"Template error with tools: {e} — retrying without tools")
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        input_ids = self._tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=not greedy,
                temperature=self.temperature if not greedy else 1.0,
                top_p=self.TOP_P,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        raw = self._tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=False,
        )
        return raw

    def _execute_tool(self, name: str, args: Dict) -> str:
        """Execute a tool via Helix's ToolExecutor or fallback."""
        if self._tool_executor is not None:
            try:
                result = self._tool_executor.execute_function_call(name, args)
                if result is None:
                    return "(tool returned no output)"
                if isinstance(result, dict):
                    return json.dumps(result)
                return str(result)
            except Exception as e:
                logger.error(f"Tool '{name}' failed: {e}")
                return f"Tool error: {e}"

        # No executor — try basic web search inline
        if name == "search" or name == "web_search":
            try:
                from tools.web_search import WebSearch
                ws = WebSearch()
                results = ws.search_web(args.get("query", ""), max_results=3)
                if results:
                    return "\n".join(
                        f"[{i+1}] {r.get('title','')}: {r.get('body','')[:200]}"
                        for i, r in enumerate(results)
                    )
                return "No results found."
            except Exception as e:
                return f"Search failed: {e}"

        return f"(tool '{name}' not available without executor)"

    # ── Main entry point ──────────────────────────────────────────────

    def _extract_user_text(self, message: str) -> str:
        """Strip Helix telemetry wrappers and return clean user text."""
        import re

        # Pattern: They said: "<msg>"
        m = re.search(r'They said:\s*["\u201c](.+?)["\u201d]', message, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Pattern: User message: / User: "<msg>"
        m = re.search(r'(?:User message|User):\s*["\u201c](.+?)["\u201d]',
                      message, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()[:600]

        # Autonomous pulse — strip XML/telemetry
        clean = re.sub(r'<[^>]+>.*?</[^>]+>', '', message, flags=re.DOTALL)
        clean = re.sub(r'\[.*?\]', '', clean)
        clean = re.sub(r'\d+\.\d+\s*\|[^\n]+', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        if len(clean) < 20:
            return "Reflect briefly on what you are currently processing."

        return clean[:400].strip()

    def send_message(self, message: str) -> str:
        """Send a message, handling tool call loops, and return final response."""
        self._ensure_loaded()

        user_text = self._extract_user_text(message)
        if not user_text:
            return "(no message)"

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        # Detect autonomous pulse vs real user task
        is_user_task = bool(
            re.search(r'They said:|User message:|User:', message, re.IGNORECASE)
            or re.search(r'["\u201c].{10,}["\u201d]', message)
        )

        # Autonomous pulses get a short reflection (no tools, max 80 tokens)
        # to avoid polluting history with 500-word essays
        if not is_user_task:
            clean_history = self._sanitize_for_mistral(self._history)
            messages = [{"role": "system", "content": self._system}] + clean_history
            try:
                input_ids = self._tokenizer(
                    self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ),
                    return_tensors="pt"
                ).input_ids.to(self._device)
                with torch.no_grad():
                    out = self._model.generate(
                        input_ids, max_new_tokens=80, do_sample=True,
                        temperature=0.8, pad_token_id=self._tokenizer.eos_token_id
                    )
                pulse_resp = re.sub(
                    r'</s>|<s>|\[INST\]|\[/INST\]', '',
                    self._tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=False)
                ).strip()
                self._history.append({"role": "assistant", "content": pulse_resp})
                logger.info(f"Mistral pulse: {pulse_resp[:80]}")
                return pulse_resp
            except Exception as e:
                logger.warning(f"Pulse generation failed: {e}")
                self._history.pop()  # remove failed user turn
                return "(pulse skipped)"

        # Build message list: system + sanitized history (Mistral requires strict alternation)
        clean_history = self._sanitize_for_mistral(self._history)
        messages = [{"role": "system", "content": self._system}] + clean_history

        logger.warning(f"ENTERING TOOL LOOP for: {user_text[:100]!r}")

        t0 = time.time()

        final_response = ""

        for loop_i in range(MAX_TOOL_LOOPS):
            try:
                raw = self._generate(messages, greedy=(loop_i >= 1))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.error("OOM during generation")
                return "(out of memory — try a shorter message)"

            calls = _parse_tool_calls(raw)
            logger.warning(f"RAW[{loop_i}]={raw[:300]!r}")
            logger.warning(f"CALLS[{loop_i}]={calls}")

            if calls:
                # ── Tool call loop ────────────────────────────────
                call_ids = [_rand_id() for _ in calls]

                # Append model's tool-call turn
                messages.append({
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c.get("arguments", {})),
                            },
                        }
                        for c, cid in zip(calls, call_ids)
                    ],
                })

                # Execute each tool
                for call, cid in zip(calls, call_ids):
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    logger.warning(f"TOOL EXEC: {name}({json.dumps(args)[:80]})")
                    result = self._execute_tool(name, args)
                    logger.warning(f"TOOL RESULT ({name}): {result[:200]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": result,
                    })
                # After tool execution: do ONE greedy summary pass (256 tokens) and return
                summary_raw = self._generate(messages, greedy=True, max_new_tokens=256)
                # If the summary itself is a prose answer (no more tool calls) → done
                if not _parse_tool_calls(summary_raw):
                    final_response = re.sub(r'</s>|<s>|\[INST\]|\[/INST\]', '', summary_raw).strip()
                    break
                # Otherwise continue the loop with the new raw
                raw = summary_raw
                calls = _parse_tool_calls(raw)
                continue
            else:
                # ── Final prose response ──────────────────────────
                # Strip Mistral special tokens from raw output
                final_response = re.sub(r'</s>|<s>|\[INST\]|\[/INST\]', '', raw).strip()
                break
        else:
            # Exhausted loops — use last raw as response
            final_response = re.sub(r'</s>|<s>|\[INST\]|\[/INST\]', '', raw).strip() if 'raw' in dir() else "(max tool iterations reached)"

        elapsed = time.time() - t0
        tokens_est = len(self._tokenizer.encode(final_response))
        logger.info(
            f"Mistral: {tokens_est} tokens in {elapsed:.1f}s "
            f"({tokens_est/max(elapsed,0.001):.0f} tok/s) | "
            f"loops={loop_i+1}"
        )

        # Store final response in history
        self._history.append({"role": "assistant", "content": final_response})
        return final_response

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def reset(self):
        self._history = []
