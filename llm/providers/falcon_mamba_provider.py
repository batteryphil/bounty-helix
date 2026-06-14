"""
Helix — Falcon-Mamba-7B-Instruct Provider

Pure Mamba (zero KV cache) — attention-free, constant memory regardless of
sequence length. Loaded in 4-bit NF4 via bitsandbytes (~4.5GB VRAM).

Model: tiiuae/falcon-mamba-7b-instruct
Architecture: Pure Mamba SSM (no attention layers)
VRAM:  ~4.6GB @ 4-bit NF4 (leaves plenty of headroom on RTX 3060)
"""

import logging
import time
import torch
from pathlib import Path
from typing import Optional, List, Dict

from llm.providers.base import ChatSession

logger = logging.getLogger("helix.llm.providers.falcon_mamba")

# ── Singleton engine ──────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_device = None

MODEL_ID = "tiiuae/falcon-mamba-7b-instruct"
HF_CACHE  = str(Path(__file__).resolve().parents[3] / "hf_cache")

SYSTEM_PROMPT = (
    "You are Helix, a thoughtful and curious AI assistant running locally. "
    "You are honest, helpful, and concise."
)


def _load_engine():
    """Load Falcon-Mamba-7B in 4-bit NF4. Idempotent singleton."""
    global _model, _tokenizer, _model_device

    if _model is not None:
        return _model, _tokenizer, _model_device

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"Loading {MODEL_ID} in 4-bit NF4 (first load, ~30s)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=HF_CACHE,
    )
    _model.eval()

    _model_device = next(_model.parameters()).device
    vram = torch.cuda.memory_allocated() / 1e9
    logger.info(f"Falcon-Mamba loaded ✅  VRAM: {vram:.2f} GB")

    return _model, _tokenizer, _model_device


class FalconMambaSession(ChatSession):
    """
    Chat session backed by Falcon-Mamba-7B-Instruct (pure Mamba, 4-bit).

    Uses the model's built-in chat template. History is maintained per-session
    and compressed to avoid exceeding the context window.
    """

    # Tell pulse_loop this is not a Gemini FC model — route raw output to callback
    is_non_fc_model = True

    MAX_HISTORY_TURNS = 8    # keep last 8 turns max
    MAX_NEW_TOKENS    = 512
    TEMPERATURE       = 0.7
    TOP_P             = 0.9

    def __init__(
        self,
        system_instruction: str = "",
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        **kwargs,
    ):
        self.temperature       = temperature
        self.max_output_tokens = max_output_tokens

        # Build a clean, short system prompt (don't pass the giant Helix manifold)
        # The 7B model can handle a real system prompt — keep it focused.
        if system_instruction and len(system_instruction) < 600:
            self._system = system_instruction.strip()
        else:
            self._system = SYSTEM_PROMPT

        self._history: List[Dict[str, str]] = []  # [{role, content}, ...]
        self._model    = None
        self._tokenizer = None
        self._device   = None

    def _ensure_loaded(self):
        if self._model is None:
            self._model, self._tokenizer, self._device = _load_engine()

    def _extract_user_text(self, message: str) -> str:
        """
        Extract just the actual user chat message from Helix's pulse message.

        User-triggered: '[HH:MM:SS] ... They said: "<msg>"'
        Autonomous: preconscious XML/telemetry blocks — no user quote.
        For autonomous pulses return a quiet self-reflection prompt.
        """
        import re

        # Pattern 1: They said: "<msg>"  (user-triggered pulse)
        m = re.search(r'They said:\s*["\u201c](.+?)["\u201d]', message, re.DOTALL)
        if m:
            return m.group(1).strip()

        # Pattern 2: User message: / User: "<msg>"
        m = re.search(r'(?:User message|User):\s*["\u201c](.+?)["\u201d]',
                      message, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()[:500]

        # Autonomous pulse — strip all XML/tag blocks and telemetry noise
        clean = re.sub(r'<[^>]+>.*?</[^>]+>', '', message, flags=re.DOTALL)
        clean = re.sub(r'\[.*?\]', '', clean)
        clean = re.sub(r'\d+\.\d+\s*\|[^\n]+', '', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        if len(clean) < 20:
            return "Take a moment to reflect. What are you thinking about right now?"

        return clean[:300].strip()

    def send_message(self, message: str) -> str:
        """Send a message and return the model's reply."""
        self._ensure_loaded()

        user_text = self._extract_user_text(message)
        if not user_text:
            return "(no message)"

        # Append user turn
        self._history.append({"role": "user", "content": user_text})

        # Trim history to MAX_HISTORY_TURNS
        if len(self._history) > self.MAX_HISTORY_TURNS * 2:
            self._history = self._history[-(self.MAX_HISTORY_TURNS * 2):]

        # Build messages list with system prompt
        messages = [{"role": "system", "content": self._system}] + self._history

        # Apply the model's chat template
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_ids = self._tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(self._device)

        # Generate
        t0 = time.time()
        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=self.max_output_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=TOP_P if hasattr(self, '_top_p') else 0.9,
                    use_cache=True,   # Mamba recurrent state, NOT KV cache
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            logger.error("OOM during generation — clearing CUDA cache")
            torch.cuda.empty_cache()
            return "(out of memory — try a shorter message)"

        elapsed = time.time() - t0
        response = self._tokenizer.decode(
            output_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Append assistant turn to history
        self._history.append({"role": "assistant", "content": response})

        tokens_gen = output_ids.shape[1] - input_ids.shape[1]
        logger.info(f"Falcon-Mamba: {tokens_gen} tokens in {elapsed:.1f}s "
                    f"({tokens_gen/elapsed:.0f} tok/s)")

        return response

    def get_history(self) -> List[Dict]:
        return list(self._history)

    def get_history_size(self) -> int:
        """Return total character count of history (for pulse_loop status)."""
        return sum(len(m.get("content", "")) for m in self._history)

    def reset(self):
        self._history = []
