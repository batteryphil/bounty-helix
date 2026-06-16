import logging
import re
import threading
from typing import List, Optional
from pathlib import Path

from llm.providers.llama_cpp_provider import LlamaCppSession

logger = logging.getLogger("helix.core.cpu_coprocessor")

class CPUCoprocessor:
    """1.58-bit BitNet CPU Coprocessor for asynchronous cognitive offloading.
    
    Runs entirely on the CPU to avoid contending with the primary Hermes GPU model
    for VRAM. Specialized in context compression and semantic extraction.
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(CPUCoprocessor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_path: str = ""):
        if self._initialized:
            return

        # Prefer newly downloaded compatible model; fall back to legacy name
        _base = Path(__file__).parent.parent / "models"
        if model_path:
            self.model_path = Path(__file__).parent.parent / model_path
        elif (_base / "coprocessor-new.gguf").exists():
            self.model_path = _base / "coprocessor-new.gguf"
        else:
            self.model_path = _base / "coprocessor-0.5b.gguf"
        self._llm: Optional[LlamaCppSession] = None
        self._model_lock = threading.Lock()
        
        # Load lazily to not block import
        self._initialized = True

    def _get_llm(self, system_instruction: str) -> LlamaCppSession:
        """Initialize or reconfigure the LLM session with new system instructions."""
        with self._model_lock:
            if self._llm is None:
                logger.info(f"[CPUCoprocessor] Loading BitNet model on CPU: {self.model_path}")
                self._llm = LlamaCppSession(
                    model_path=str(self.model_path),
                    system_instruction=system_instruction,
                    n_ctx=8192,   # 2B BitNet can handle 8K context on CPU
                    n_gpu_layers=0,  # strictly CPU — leaves full VRAM for Hermes
                    temperature=0.3,
                    max_output_tokens=800
                )
            else:
                # Update system prompt and clear history
                self._llm.system_instruction = system_instruction
                self._llm.history = []
            return self._llm

    def compress_context(self, raw_text: str, max_words: int = 200) -> str:
        """Compress a large block of text into dense bullet points.
        
        Used to summarize large read_url or search results before they hit 
        the primary GPU model's context window.
        """
        if not self.model_path.exists():
            logger.warning("[CPUCoprocessor] Model not found, skipping compression.")
            # Fallback to simple truncation
            words = raw_text.split()
            return " ".join(words[:max_words]) + "... (truncated)"
            
        prompt = (
            "You are an expert synthesizer. Distill the following text into "
            "a dense, concise bulleted list of the most critical facts and insights. "
            "Ignore all fluff, navigation links, and irrelevant noise.\n\n"
            f"TEXT TO SUMMARIZE:\n{raw_text[:8000]}"
        )
        
        sys_prompt = "You output only the requested summary as a bulleted list. No intro, no outro."
        try:
            llm = self._get_llm(system_instruction=sys_prompt)
            summary = llm.send_message(prompt)
            logger.info(f"[CPUCoprocessor] Compressed {len(raw_text)} chars to {len(summary)} chars.")
            return summary.strip()
        except Exception as e:
            logger.error(f"[CPUCoprocessor] Compression failed: {e}")
            words = raw_text.split()
            return " ".join(words[:max_words]) + "... (truncated on error)"

    def extract_beliefs(self, text: str) -> List[str]:
        """Extract explicit beliefs about capabilities, identity, or the world.
        
        Replaces the TF-IDF regex fallback.
        """
        if not self.model_path.exists():
            return []
            
        prompt = (
            "Extract any explicit beliefs or realizations from the following text. "
            "Output each belief wrapped in a <belief> tag. "
            "If there are no beliefs, output 'NONE'.\n\n"
            f"TEXT:\n{text[:4000]}"
        )
        
        sys_prompt = "You are a cognitive extractor. Output ONLY valid <belief> XML tags."
        try:
            llm = self._get_llm(system_instruction=sys_prompt)
            response = llm.send_message(prompt)
            
            beliefs = []
            for match in re.finditer(r"<belief>(.*?)</belief>", response, re.IGNORECASE | re.DOTALL):
                b = match.group(1).strip()
                if b and len(b) > 10:
                    beliefs.append(b)
            return beliefs
        except Exception as e:
            logger.error(f"[CPUCoprocessor] Belief extraction failed: {e}")
            return []

    def inference_fallback(self, prompt: str, system: str = "", max_tokens: int = 512) -> str:
        """Run a full inference pass on CPU when the GPU model OOMs.

        This is the OOM escape hatch: when Hermes CUDA generate() throws
        RuntimeError (out of memory), the pulse loop calls this method to
        get a response via the 1.58-bit BitNet on CPU instead.
        No VRAM used — pure CPU inference.

        Args:
            prompt:     The user/pulse message to respond to.
            system:     System instruction (uses a compressed bounty prompt if empty).
            max_tokens: Max tokens to generate (kept low to stay fast on CPU).
        Returns:
            The generated text response string.
        """
        if not self.model_path.exists():
            logger.warning("[CPUCoprocessor] Model not found — cannot run inference fallback.")
            return '{"name": "bounty_status", "arguments": {}}'  # safe no-op tool call

        _sys = system or (
            "You are Helix, an autonomous bounty agent. "
            "Output a single tool call as JSON: {\"name\": \"tool_name\", \"arguments\": {}}. "
            "Pick the most useful next action for completing the Rustchain bounty."
        )
        try:
            llm = self._get_llm(system_instruction=_sys)
            llm.max_output_tokens = max_tokens
            response = llm.send_message(prompt)
            logger.info(f"[CPUCoprocessor] OOM fallback inference OK ({len(response)} chars).")
            return response.strip()
        except Exception as e:
            logger.error(f"[CPUCoprocessor] OOM fallback inference failed: {e}")
            return '{"name": "bounty_status", "arguments": {}}'

# Global singleton
coprocessor = CPUCoprocessor()
