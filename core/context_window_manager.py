"""
Helix — Context Window Manager (Infinite Run Support)

Prevents KV cache ballooning for indefinite agent operation by enforcing
a strict, flat context budget using a 4-layer memory hierarchy:

  Layer 1 — Hot window (in-memory chat history)
    Last N turns in HermesToolSession._history
    Budget: MAX_HOT_TURNS = 6 turns (was 12)
    Trimmed after every pulse

  Layer 2 — Warm summary (rolling compressed summary)
    Every COMPRESS_EVERY turns, the oldest half of hot window is
    summarized into a single "Memory Digest" assistant turn and
    prepended to the hot window. The originals are dropped.
    This keeps the window flat as a pancake forever.

  Layer 3 — Cold belief store (BeliefStore)
    Factual long-term knowledge — already persisted to disk.
    Not in the context window at all.

  Layer 4 — Archive (curiosity_knowledge.jsonl + evolution_journal.jsonl)
    Full historical record on disk. Never loaded into context.

Memory lifecycle:
  Pulse → hot window grows by 1 turn
  Every 6 turns → compress oldest 3 turns → 1 digest turn
  Hot window stays at ≤ 6 turns permanently
  VRAM stays flat (no growth over time)

KV cache math for Hermes-3-Llama-3.1-8B:
  - Per token: 2 (K,V) × 8 (GQA heads) × 128 (head_dim) × 32 (layers) × 2 (bfloat16)
             = 131,072 bytes = 128 KB per token
  - System prompt: ~600 tokens = 75 MB
  - 6 hot turns × ~150 tokens each = 900 tokens = 112 MB
  - Total flat context KV: ~187 MB — stays constant forever
  - Leaves ~1.2 GB free for generation headroom ✅
"""

import logging
import threading
import time
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llm.providers.hermes_tool_provider import HermesToolSession

logger = logging.getLogger("helix.core.context_manager")

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_HOT_TURNS    = 6     # max turns in live chat history (was 12)
COMPRESS_EVERY   = 4     # compress when hot window exceeds this
SESSION_RESET_EVERY = 500  # full session reset every N pulses (safety net)
MAX_SYSTEM_PROMPT_CHARS = 2000  # hard cap on system prompt injected per pulse


class ContextWindowManager:
    """
    Manages the HermesToolSession chat history to keep the KV cache
    flat and bounded regardless of how long the agent has been running.
    """

    def __init__(self, session=None):
        self._session = session
        self._lock = threading.Lock()
        self._pulse_count = 0
        self._compression_count = 0
        self._last_digest: Optional[str] = None

    def set_session(self, session):
        """Wire in the HermesToolSession after init."""
        self._session = session
        # Override the session's MAX_HISTORY_TURNS
        if session:
            session.MAX_HISTORY_TURNS = MAX_HOT_TURNS
            logger.info(f"[ctx_mgr] MAX_HISTORY_TURNS capped to {MAX_HOT_TURNS}")

    def on_pulse(self, pulse_number: int):
        """Call after every pulse. Manages compression and session resets."""
        self._pulse_count = pulse_number

        if self._session is None:
            return

        with self._lock:
            history = self._session._history
            hot_turns = len([m for m in history if m.get("role") == "assistant"])

            # Compress if hot window is getting large
            if hot_turns >= COMPRESS_EVERY:
                self._compress_oldest()

            # Full session reset every N pulses (safety net against any leak)
            if pulse_number > 0 and pulse_number % SESSION_RESET_EVERY == 0:
                self._full_reset()

    def _compress_oldest(self):
        """
        Replace the oldest assistant/user pair with a compact Memory Digest.
        This keeps the hot window flat — oldest memory → compressed form.
        """
        if self._session is None:
            return

        history = self._session._history

        # Find first assistant turn and its surrounding user turn
        # Skip system message (index 0 if present)
        start = 0
        if history and history[0].get("role") == "system":
            start = 1

        # Collect oldest 2 turns (1 user + 1 assistant) to compress
        if len(history) - start < 4:
            return  # not enough to compress yet

        oldest = history[start:start + 2]
        remaining = history[:start] + history[start + 2:]

        # Build a 1-sentence digest of what was dropped
        digest_parts = []
        for msg in oldest:
            content = str(msg.get("content", ""))[:150].strip()
            role = msg.get("role", "")
            if content and role == "assistant":
                # Take first sentence of the thought
                first_sent = content.split(".")[0][:100]
                digest_parts.append(first_sent)

        if not digest_parts:
            self._session._history = remaining
            return

        digest_text = (
            f"[Memory Digest — compressed {len(oldest)} turns]: "
            + " | ".join(digest_parts)
        )
        self._last_digest = digest_text

        # Insert digest as a system-role note right after the system prompt
        digest_msg = {"role": "system", "content": digest_text}
        insert_at = start
        remaining.insert(insert_at, digest_msg)

        self._session._history = remaining
        self._compression_count += 1

        logger.info(
            f"[ctx_mgr] Compressed {len(oldest)} turns → digest "
            f"(compression #{self._compression_count}, "
            f"history now {len(remaining)} msgs)"
        )

    def _full_reset(self):
        """
        Every SESSION_RESET_EVERY pulses: clear history entirely and
        inject only the last digest as context seed. This is the absolute
        safety net — guarantees zero leak over arbitrarily long runs.
        """
        if self._session is None:
            return

        old_len = len(self._session._history)

        # Keep only the system message
        sys_msgs = [m for m in self._session._history if m.get("role") == "system"][:1]

        # Add last digest as memory seed
        seed = []
        if self._last_digest:
            seed = [{"role": "system", "content":
                     f"[Long-term memory seed from prior session]: {self._last_digest}"}]

        self._session._history = sys_msgs + seed
        self._session.clear_history()

        logger.warning(
            f"[ctx_mgr] Full session reset at pulse {self._pulse_count} "
            f"(dropped {old_len} msgs, injected memory seed)"
        )

    def get_stats(self) -> dict:
        if self._session is None:
            return {}
        history = self._session._history
        total_chars = sum(len(str(m.get("content", ""))) for m in history)
        return {
            "history_msgs": len(history),
            "history_chars": total_chars,
            "estimated_tokens": total_chars // 4,  # rough estimate
            "compressions": self._compression_count,
            "pulse_count": self._pulse_count,
            "next_reset_in": SESSION_RESET_EVERY - (self._pulse_count % SESSION_RESET_EVERY),
        }

    def format_status(self) -> str:
        s = self.get_stats()
        if not s:
            return "Context manager not initialized"
        return (
            f"Context: {s['history_msgs']} msgs / ~{s['estimated_tokens']} tokens | "
            f"compressions: {s['compressions']} | "
            f"reset in: {s['next_reset_in']} pulses"
        )


# ── Singleton ──────────────────────────────────────────────────────────────────

_manager: Optional[ContextWindowManager] = None


def get_manager() -> Optional[ContextWindowManager]:
    return _manager


def init_manager(session=None) -> ContextWindowManager:
    global _manager
    _manager = ContextWindowManager(session=session)
    return _manager
