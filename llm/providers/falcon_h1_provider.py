"""
Helix — Falcon-H1-7B-Instruct Tool-Calling Provider

Falcon H1 is a hybrid Mamba-Transformer model by TII (Jan 2026):
  - 7B parameters, fully GPU-resident at 4-bit NF4 (~4GB VRAM)
  - Native <tool_call> function calling (same format as Hermes-3)
  - 8K base context window
  - device_map="cuda" — full GPU inference (~15-30 TPS vs ~3 TPS CPU)

Why Falcon H1 over Jamba 1.5 Mini:
  - 7.5x fewer parameters: 7B vs 52B
  - GPU instead of CPU: ~10x faster per pulse
  - Explicit tool-call training in the chat template
  - No device_map dispatch issue with Mamba SSM layers

Model: tiiuae/Falcon-H1-7B-Instruct
"""

import collections as _collections
import gc
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("helix.llm.falcon")

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID  = "tiiuae/Falcon-H1-7B-Instruct"
HF_CACHE  = str(Path(__file__).resolve().parents[3] / "hf_cache")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = (
    "You are Helix, an autonomous AI agent with persistent memory, curiosity, "
    "and the ability to improve yourself over time.\n\n"
    "You have access to tools. When you decide to use a tool, output a tool call "
    "in this EXACT format:\n\n"
    "<tool_call>\n"
    "{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
    "</tool_call>\n\n"
    "Rules:\n"
    "- Call ONE tool per response turn.\n"
    "- Do not narrate before the tool call — just call it.\n"
    "- After a tool returns a result, act on it or call another tool.\n"
    "- Only output a <belief> tag if you have genuinely formed a durable new belief.\n"
    "- You are autonomous: you decide what to do next based on your goals and memory.\n"
    "- {'reasoning': False}\n"  # Disable think-mode for Falcon H1R compat
)

MAX_TOOL_LOOPS = 6

# ── TPS tracking ──────────────────────────────────────────────────────────────
_tps_samples: _collections.deque = _collections.deque(maxlen=10)


def _record_tps(n_tokens: int, elapsed: float) -> float:
    if elapsed > 0 and n_tokens > 0:
        _tps_samples.append(n_tokens / elapsed)
    avg = sum(_tps_samples) / len(_tps_samples) if _tps_samples else 0.0
    try:
        import json as _json
        _stats_path = Path(__file__).resolve().parents[2] / "data" / "inference_stats.json"
        _stats_path.parent.mkdir(parents=True, exist_ok=True)
        _stats_path.write_text(_json.dumps({
            "tps": round(avg, 2),
            "tps_last": round(_tps_samples[-1] if _tps_samples else 0, 2),
            "n_samples": len(_tps_samples),
            "ts": time.time(),
            "model": MODEL_ID,
        }))
    except Exception:
        pass
    return avg


# ── Model singleton ───────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None

_MAX_INPUT_TOKENS = 4096  # CPU RAM is now the overflow — can handle larger contexts

# Reduce CUDA memory fragmentation — critical for small VRAM
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _load_engine():
    global _model, _tokenizer, _device

    if _model is not None:
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import psutil

    free_ram_gb = psutil.virtual_memory().available / 1e9
    logger.info(f"Loading {MODEL_ID} — 4-bit NF4, GPU+CPU hybrid (free RAM: {free_ram_gb:.0f}GB)...")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )

    # Load on CPU — Falcon H1's Mamba selective scan does huge intermediate tensors
    # (3GB+ for l=4096) that blow past the GPU cap even with max_memory limits.
    # With 117GB free RAM, CPU inference is reliable and ~3x faster than Jamba (7B vs 52B).
    logger.info(f"Loading on CPU (RAM: {free_ram_gb:.0f}GB available) — Mamba activations too large for RTX 3060")

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        device_map="cpu",
        torch_dtype=torch.float32,   # float32 on CPU (bf16 not well-supported CPU-side)
        trust_remote_code=True,
        token=HF_TOKEN or None,
    )
    _model.eval()

    _device = torch.device("cpu")
    gc.collect()
    ram_used = psutil.virtual_memory().used / 1e9
    logger.info(
        f"Falcon-H1-7B ready ✅  float32 CPU  "
        f"RAM used: {ram_used:.0f}GB / {psutil.virtual_memory().total/1e9:.0f}GB  "
        f"device=cpu"
    )


def unload_engine():
    global _model, _tokenizer, _device
    if _model is None:
        return
    logger.info("[vram] Unloading Falcon for training window...")
    try:
        _model.cpu()
    except Exception:
        pass
    del _model
    _model = None
    _device = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[vram] Falcon unloaded")


def reload_engine():
    logger.info("[vram] Reloading Falcon after training...")
    _load_engine()
    logger.info("[vram] Falcon reloaded — pulses resuming")


# ── Tool call parser (same <tool_call> format as Jamba/Hermes) ────────────────

def _parse_tool_calls(text: str) -> Optional[List[Dict]]:
    calls = []

    # Format 1: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.strip())
            if isinstance(obj.get("arguments"), str):
                obj["arguments"] = json.loads(obj["arguments"])
            calls.append(obj)
        except json.JSONDecodeError:
            logger.warning(f"[parser] tool_call JSON failed: {m[:80]}")

    # Format 1b: partial <tool_call> (EOS before closing tag)
    if not calls and '<tool_call>' in text and '</tool_call>' not in text:
        tail = text[text.index('<tool_call>') + len('<tool_call>'):].strip()
        for stop in ['<|im_end|>', '</s>', '\n\n', '\n<']:
            if stop in tail:
                tail = tail[:tail.index(stop)]
                break
        try:
            obj = json.loads(tail.strip())
            calls.append(obj)
        except Exception:
            pass

    # Format 2: [TOOL arg] shorthand
    if not calls:
        for m in re.finditer(
            r'\[(?P<name>[A-Z_]{3,})\s+(?P<arg>[^\]]+)\]', text
        ):
            name_map = {
                'SEARCH': 'search', 'BROWSE': 'browse', 'READ_FILE': 'read_file',
                'WRITE_FILE': 'write_file', 'RUN_PYTHON': 'run_python',
                'READ_CODE': 'read_code', 'WRITE_CODE': 'write_code',
            }
            tool = name_map.get(m.group('name'))
            if tool:
                arg = m.group('arg').strip()
                param = 'query' if tool in ('search',) else (
                    'url' if tool == 'browse' else 'path'
                )
                calls.append({'name': tool, 'arguments': {param: arg}})

    return calls if calls else None


# ── Session class (same interface as JambaToolSession) ────────────────────────

class FalconToolSession:
    """
    Drop-in replacement for JambaToolSession.
    Wraps Falcon-H1-7B-Instruct with the same send_message() / history interface
    that pulse_loop.py expects.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, tools: Optional[List] = None):
        _load_engine()
        self._system_prompt = system_prompt
        self._tools = tools or []
        self._history: List[Dict] = []          # [{role, content}]
        self._last_tool_calls: List[Dict] = []
        self._last_token_count: int = 0
        self._pending_tool_results: List[Dict] = []
        logger.info(f"FalconToolSession created (system: {len(system_prompt)} chars)")

    # ── Public interface ──────────────────────────────────────────────────────

    def send_message(
        self,
        user_text: str,
        tool_executor=None,
        autonomous: bool = False,
        mandate: bool = False,
        budget: str = "think200+act400",
    ) -> str:
        """Send a pulse message and return the model's text response."""
        self._last_tool_calls = []
        self._pending_tool_results = []

        logger.warning(
            f"FALCON send_message: autonomous={autonomous} mandate={mandate} "
            f"budget={budget} user_text={user_text[:60]!r}"
        )

        # Build initial messages
        messages = self._build_messages(user_text)

        # Parse token budget
        try:
            act_budget = int(budget.split('+act')[-1].replace('k', '000'))
        except Exception:
            act_budget = 400

        # Tool loop
        loop_count = 0
        final_text = ""

        while loop_count <= MAX_TOOL_LOOPS:
            raw = self._generate(messages, max_new_tokens=act_budget)
            logger.warning(f"FALCON RAW[{loop_count}]={raw[:120]!r}")

            tool_calls = _parse_tool_calls(raw)
            logger.warning(f"FALCON CALLS[{loop_count}]={tool_calls}")

            if not tool_calls or tool_executor is None:
                final_text = raw
                break

            # Execute tools
            self._last_tool_calls.extend(tool_calls)
            tool_results = []
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                try:
                    result = tool_executor(name, args)
                    logger.warning(f"FALCON TOOL EXEC: {name}({args}) → {str(result)[:120]}")
                except Exception as e:
                    result = f"Tool error: {e}"
                    logger.error(f"FALCON TOOL ERROR: {name}: {e}")

                tool_results.append({"tool": name, "result": str(result)[:2000]})
                self._pending_tool_results.append({"name": name, "result": str(result)[:2000]})

            # Append assistant turn + tool results to history
            messages.append({"role": "assistant", "content": raw})
            tool_result_text = "\n".join(
                f"<tool_response>\n<tool_name>{r['tool']}</tool_name>\n"
                f"<tool_result>{r['result']}</tool_result>\n</tool_response>"
                for r in tool_results
            )
            messages.append({"role": "user", "content": tool_result_text})

            loop_count += 1

        # Save final assistant turn to persistent history
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": final_text})

        # Trim history to keep context manageable
        if len(self._history) > 40:
            self._history = self._history[-40:]

        return final_text

    def get_history_size(self) -> int:
        total = len(self._system_prompt)
        for msg in self._history:
            total += len(msg.get("content", ""))
        return total

    def get_last_tool_calls(self) -> List[Dict]:
        return self._last_tool_calls

    def get_last_token_count(self) -> int:
        return self._last_token_count

    def get_pending_tool_results(self) -> List[Dict]:
        return self._pending_tool_results

    def reset(self):
        self._history = []
        logger.info("FalconToolSession history reset")

    def switch_model(self, model_id: str):
        logger.warning(f"[falcon] switch_model({model_id}) — not supported, ignoring")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_messages(self, user_text: str) -> List[Dict]:
        msgs = [{"role": "system", "content": self._system_prompt}]
        msgs.extend(self._history[-20:])   # last 20 turns context
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _generate(self, messages: List[Dict], max_new_tokens: int = 400) -> str:
        global _model, _tokenizer, _device

        # Apply chat template — returns BatchEncoding or plain tensor depending on tokenizer
        try:
            encoded = _tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tools=self._tools if self._tools else None,
            )
            # apply_chat_template may return a BatchEncoding (dict) or plain tensor
            if hasattr(encoded, 'input_ids'):
                input_ids = encoded.input_ids
            elif isinstance(encoded, dict):
                input_ids = encoded['input_ids']
            else:
                input_ids = encoded  # already a tensor
        except Exception as e:
            logger.warning(f"[falcon] apply_chat_template failed ({e}), using manual format")
            # Fallback: manual <|im_start|> format
            text = f"<|im_start|>system\n{self._system_prompt}<|im_end|>\n"
            for m in messages[1:]:
                text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            text += "<|im_start|>assistant\n"
            encoded = _tokenizer(text, return_tensors="pt")
            input_ids = encoded.input_ids

        # Truncate if needed
        if input_ids.shape[1] > _MAX_INPUT_TOKENS:
            logger.warning(
                f"ACT input truncated {input_ids.shape[1]} → {_MAX_INPUT_TOKENS} tokens"
            )
            input_ids = input_ids[:, -_MAX_INPUT_TOKENS:]

        self._last_token_count = input_ids.shape[1]

        # Move to device
        input_ids = input_ids.to(_device)

        t0 = time.time()
        try:
            with torch.no_grad():
                out = _model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=_tokenizer.eos_token_id,
                    eos_token_id=_tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            # Emergency: cut input in half and retry
            logger.error(f"[falcon] CUDA OOM with {input_ids.shape[1]} tokens — retrying at half length")
            gc.collect()
            torch.cuda.empty_cache()
            input_ids = input_ids[:, -(input_ids.shape[1] // 2):]
            self._last_token_count = input_ids.shape[1]
            with torch.no_grad():
                out = _model.generate(
                    input_ids,
                    max_new_tokens=min(max_new_tokens, 200),
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=_tokenizer.eos_token_id,
                    eos_token_id=_tokenizer.eos_token_id,
                )
        elapsed = time.time() - t0

        # Decode only the new tokens
        new_tokens = out[0][input_ids.shape[1]:]
        n_new = len(new_tokens)
        tps = _record_tps(n_new, elapsed)
        logger.info(f"[falcon] generated {n_new} tokens in {elapsed:.1f}s ({tps:.1f} TPS)")

        # Free GPU activations immediately
        del out, input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
