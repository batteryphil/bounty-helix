"""
Helix — Titan MIMO Local Provider

Implements the ChatSession interface backed by the Titan 2.7B MIMO inference engine.
Runs 100% locally on the RTX 3060 — no API calls, no cloud dependency.

Architecture:
  - Parses Helix pulse meta-tags ([JOURNAL:], [REMEMBER:], [BELIEF_FORM], etc.)
  - Maps tags to MIMO arm bias vectors so the right arm activates per cognitive task
  - Handles context compression automatically when history exceeds Titan's window
  - Logs (prompt, response) pairs to a replay buffer for overnight fine-tuning

Arm → Helix subsystem mapping:
  Arm 0  General Language      → pulse_loop internal monologue
  Arm 1  Symbolic Math         → physics_engine Lagrangian calculations
  Arm 2  Logical Reasoning     → belief_detector / belief_consolidator
  Arm 3  Code Syntax           → tool call generation
  Arm 4  Factual Recall        → memory_manager semantic search
  Arm 5  Summarization         → context_compressor
  Arm 6  Creative Writing      → cognitive_journal journaling
  Arm 7  Instruction Following → orchestrator tool dispatch
"""

import logging
import os
import sys
import json
import time
import re
import socket
import torch
from pathlib import Path
from typing import Optional, List, Dict

from llm.providers.base import ChatSession

logger = logging.getLogger("helix.llm.providers.titan")

# ── Path bootstrap: allow running from any working directory ──────────────────
_HERE = Path(__file__).resolve().parent          # Helix-AGI/llm/providers/
_PROJECT = _HERE.parents[2]                      # analysis_project/
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ── Arm bias profiles (8 arms) ───────────────────────────────────────────────
# Additive boosts applied to gate logits before softmax.
# Values: 0.0 = neutral, 2.0 = strong preference.
# Order: [General, Math, Logical, Code, Factual, Summary, Creative, Instruction]
_ARM_PROFILES: Dict[str, List[float]] = {
    "journal":      [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 0.3],
    "remember":     [0.2, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.3],
    "belief":       [0.2, 0.5, 2.0, 0.0, 0.5, 0.3, 0.0, 0.5],
    "tool_call":    [0.2, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 2.0],
    "math_physics": [0.2, 2.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.3],
    "compress":     [0.2, 0.0, 0.3, 0.0, 0.5, 2.0, 0.3, 0.5],
    "default":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


# ── Tag → profile mapping ────────────────────────────────────────────────────
_TAG_TO_PROFILE = {
    r"\[JOURNAL[:\]]":          "journal",
    r"\[NOTE[:\]]":             "journal",
    r"\[REMEMBER[:\]]":         "remember",
    r"\[BELIEF_FORM\]":         "belief",
    r"\[BELIEF_CONSOLIDAT":     "belief",
    r"tool_call|function_call": "tool_call",
    r"Ω|lagrangian|manifold|curvature|stability": "math_physics",
    r"\[COMPRESS\]|context_limit": "compress",
}


def _detect_arm_profile(text: str) -> str:
    """Scan message text for Helix meta-tags and return best arm profile name."""
    text_lower = text.lower()
    for pattern, profile in _TAG_TO_PROFILE.items():
        if re.search(pattern, text, re.IGNORECASE):
            return profile
    return "default"


def _compress_history(history: List[Dict], max_chars: int = 3000) -> List[Dict]:
    """Trim oldest history turns to stay within Titan's context window.
    Keeps the system turn (index 0) and all recent turns.
    """
    if not history:
        return history
    total = sum(len(m.get("content", "")) for m in history)
    while total > max_chars and len(history) > 2:
        removed = history.pop(1)  # remove oldest non-system turn
        total -= len(removed.get("content", ""))
    return history


class TitanSession(ChatSession):
    """
    Chat session backed by Titan 2.7B MIMO local inference.

    Drop-in replacement for GeminiSession / OllamaSession.
    The pulse loop sees only the ChatSession interface.
    """

    # Max chars of history to keep — Titan context ~512 tokens ≈ 2048 chars.
    # Helix's context_compressor handles deeper compression upstream.
    MAX_HISTORY_CHARS = 2048

    # Replay buffer: log every (prompt, response) pair for overnight fine-tuning
    REPLAY_BUFFER_PATH = _PROJECT / "helix_replay_buffer.jsonl"

    def __init__(
        self,
        system_instruction: str,
        temperature: float = 0.85,
        max_output_tokens: int = 512,
        enable_deep_think: bool = False,
        checkpoint: str = "auto",
    ):
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_deep_think = enable_deep_think
        self.checkpoint = checkpoint
        self.is_non_fc_model = True
        self.history: List[Dict[str, str]] = []
        self._engine = None  # lazy-loaded on first send_message()
        self._checkpoint = checkpoint

        logger.info(
            f"TitanSession created — temp={temperature}, "
            f"max_tokens={max_output_tokens}, deep_think={enable_deep_think}"
        )

    # ── Lazy engine load ──────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._engine is not None:
            return
        try:
            from titan_inference import get_engine
            logger.info("Reusing pre-loaded Titan singleton from VRAM...")
            self._engine = get_engine(self._checkpoint)
            logger.info("Titan engine ready ✓")
            
            # Setup UDP telemetry for direct neural feed
            self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_addr = ("127.0.0.1", 5051) # Dashboard
            self._udp_addr_sentinel = ("127.0.0.1", 5052) # StabilitySentinel
            
            # Setup Manifold Projection Matrix
            device = self._engine.model.lm_head.weight.device
            d_model = self._engine.model.d_model
            self._W_proj = torch.nn.Linear(d_model, 8, bias=False).to(device)
            torch.nn.init.orthogonal_(self._W_proj.weight)
            self._W_proj.requires_grad_(False)
            
            # Register forward hook on norm_f (last layer before LM head)
            def telemetry_hook(module, inputs, output):
                with torch.no_grad():
                    # output shape: [B, L, d_model] -> get last token
                    hidden_state = output[:, -1, :].float()
                    
                    # 1. Entropy Extraction (\Omega)
                    p = torch.nn.functional.softmax(hidden_state, dim=-1)
                    entropy = -(p * torch.log(p + 1e-9)).sum(dim=-1).mean().item()
                    max_entropy = torch.log(torch.tensor(d_model, dtype=torch.float32)).item()
                    omega = entropy / max_entropy
                    
                    # 2. Manifold Projection
                    # Ensure W_proj is on the exact same device and dtype as hidden_state
                    if self._W_proj.weight.device != hidden_state.device or self._W_proj.weight.dtype != hidden_state.dtype:
                        self._W_proj = self._W_proj.to(device=hidden_state.device, dtype=hidden_state.dtype)
                    coords = self._W_proj(hidden_state).squeeze(0).tolist()
                    
                    # 3. Fire-and-forget UDP telemetry
                    payload = json.dumps({
                        "type": "telemetry",
                        "omega": round(omega, 4),
                        "manifold_8d": [round(c, 4) for c in coords]
                    }).encode('utf-8')
                    try:
                        self._udp_sock.sendto(payload, self._udp_addr)
                        self._udp_sock.sendto(payload, self._udp_addr_sentinel)
                    except Exception:
                        pass
                        
            self._engine.model.norm_f.register_forward_hook(telemetry_hook)
            logger.info("Attached Lagrangian Sentinel telemetry hook.")
            
        except Exception as e:
            logger.error(f"Failed to load Titan: {e}")
            raise RuntimeError(
                f"Titan inference engine failed to initialize: {e}\n"
                "Ensure Phase 1 training has produced a checkpoint at "
                "checkpoints_2.7b/phase_1.pt"
            ) from e

    # ── Core interface ────────────────────────────────────────────────────────
    def send_message(self, message: str) -> str:
        """Send a pulse message to Titan and return generated text."""
        self._ensure_loaded()

        # Detect which cognitive arm profile this pulse needs
        profile_name = _detect_arm_profile(message)
        arm_bias = _ARM_PROFILES[profile_name]
        if profile_name != "default":
            logger.debug(f"Arm profile: {profile_name} | bias applied to gate logits")

        # Build prompt: system + compressed history + new message
        self.history.append({"role": "user", "content": message})
        self.history = _compress_history(self.history, self.MAX_HISTORY_CHARS)

        prompt = self._build_prompt()

        # Generate
        t0 = time.time()
        try:
            response_tokens = []
            for token, arm_info in self._engine.stream(
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_output_tokens,
                arm_bias=arm_bias,
            ):
                response_tokens.append(token)

            response = "".join(response_tokens).strip()
            elapsed = time.time() - t0
            tps = len(response_tokens) / max(elapsed, 0.001)
            logger.debug(
                f"Titan generated {len(response_tokens)} tokens "
                f"in {elapsed:.1f}s ({tps:.1f} tok/s) | profile={profile_name}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Titan generation error: {e}")
            response = f"[Titan internal error: {str(e)[:120]}]"

        # Store in history
        self.history.append({"role": "assistant", "content": response})

        # Log to replay buffer for overnight fine-tuning
        self._log_replay(prompt, response, profile_name)

        return response

    def get_history_size(self) -> int:
        total = len(self.system_instruction)
        for msg in self.history:
            total += len(msg.get("content", ""))
        return total

    # ── Prompt assembly ───────────────────────────────────────────────────────
    def _build_prompt(self) -> str:
        """Assemble a flat prompt using the exact training format.
        
        The model was trained on allenai/big-reasoning-traces with format:
            User: <question>
            Assistant: <think>
        
        The [SYSTEM]...[/SYSTEM] wrapper was never seen during training.
        Passing it causes the model to treat it as continuation text and
        output garbage. Only the last user turn is used as the prompt.
        The small 1.4B model cannot follow a complex system prompt anyway.
        """
        # Extract just the last user message from history
        last_user_msg = ""
        for msg in reversed(self.history):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        # Strip the giant helix system injection — use only the raw user message.
        # The system prompt was causing the model to output medical/scientific gibberish
        # because it was treating the Lagrangian telemetry as text to continue.
        # Extract just the user's actual chat message if present.
        # The pulse message format is: "[HH:MM:SS] User is talking... They said: \"<msg>\""
        import re
        chat_match = re.search(r'They said: "(.+?)"', last_user_msg, re.DOTALL)
        if chat_match:
            user_text = chat_match.group(1).strip()
        else:
            # Fallback: use the raw message, truncated to 200 chars
            user_text = last_user_msg.strip()[-200:]
        
        return f"User: {user_text}\nAssistant: <think>\n"

    # ── Replay buffer ─────────────────────────────────────────────────────────
    def _log_replay(self, prompt: str, response: str, profile: str):
        """Append (prompt, response) to JSONL replay buffer for nightly fine-tuning."""
        try:
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "profile": profile,
                "prompt": prompt[-1000:],  # trim to avoid huge files
                "response": response,
            }
            with open(self.REPLAY_BUFFER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Replay buffer write failed: {e}")
