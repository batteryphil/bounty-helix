"""
Helix — Titan Memory Bridge

Connects Helix's cognitive journal and belief store to Titan's context window.

Two directions:
  1. Journal → Titan Context: Recent journal entries and beliefs are summarized
     and prepended to each pulse, giving Titan episodic memory.

  2. Replay Buffer → Fine-tuning Data: The replay buffer (logged by TitanSession)
     is periodically exported as structured fine-tuning pairs during Helix's
     sleep window (1–6 AM). This is how Titan learns from its own experiences.

Usage:
    bridge = TitanMemoryBridge(memory_manager, belief_store, replay_buffer_path)
    context_snippet = bridge.get_context_injection(max_chars=800)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.memory_manager import MemoryManager
    from memory.belief_store import BeliefStore

logger = logging.getLogger("helix.brain.titan_memory_bridge")


class TitanMemoryBridge:
    """
    Bridges Helix's episodic memory and beliefs into Titan's context window.

    Titan cannot store long-term memories internally (it's a stateless inference
    engine). This bridge serializes the most relevant recent memories and beliefs
    into a compact text block that gets prepended to each pulse prompt.
    """

    def __init__(
        self,
        memory_manager: "MemoryManager",
        belief_store: "BeliefStore",
        replay_buffer_path: Path,
        project_dir: Optional[Path] = None,
    ):
        self.memory = memory_manager
        self.beliefs = belief_store
        self.replay_path = replay_buffer_path
        self.project_dir = project_dir or Path(__file__).resolve().parents[3]
        self._last_export_time = 0.0

    # ── Context injection (called every pulse) ────────────────────────────────

    def get_context_injection(self, max_chars: int = 800) -> str:
        """
        Build a compact memory + belief summary for Titan's context window.

        Returns a text block like:
            [TITAN_MEMORY]
            Recent: I discussed X with the user. Believed Y about Z.
            Core beliefs: I am Helix. I value truth. ...
            [/TITAN_MEMORY]
        """
        parts = []

        # Recent memories (last 3 entries, truncated)
        try:
            recent = self.memory.get_recent(limit=3)
            if recent:
                snippets = []
                for mem in recent:
                    content = mem.get("content", "")[:150]
                    snippets.append(content)
                parts.append("Recent: " + " | ".join(snippets))
        except Exception as e:
            logger.debug(f"Memory fetch failed: {e}")

        # High-mass beliefs (top 4 by mass/confidence)
        try:
            all_beliefs = self.beliefs.get_all()
            if all_beliefs:
                # Sort by mass descending (mass = confidence + affective charge)
                sorted_beliefs = sorted(
                    all_beliefs,
                    key=lambda b: b.get("mass", b.get("confidence", 0.0)),
                    reverse=True,
                )[:4]
                belief_lines = [b.get("content", "")[:80] for b in sorted_beliefs]
                parts.append("Core beliefs: " + " • ".join(belief_lines))
        except Exception as e:
            logger.debug(f"Belief fetch failed: {e}")

        if not parts:
            return ""

        block = "\n".join(parts)

        # Trim to budget
        if len(block) > max_chars:
            block = block[:max_chars] + "…"

        return f"[TITAN_MEMORY]\n{block}\n[/TITAN_MEMORY]\n"

    # ── Overnight fine-tuning export ──────────────────────────────────────────

    def should_run_overnight_training(self) -> bool:
        """Returns True if we are in the sleep window and haven't trained yet tonight."""
        import datetime
        now = datetime.datetime.now()
        in_sleep_window = 1 <= now.hour < 6
        been_long_enough = (time.time() - self._last_export_time) > 3600  # at most hourly
        return in_sleep_window and been_long_enough

    def export_finetune_batch(self, max_examples: int = 64) -> Optional[Path]:
        """
        Read the replay buffer, filter for high-quality examples, write a
        fine-tuning JSONL that the overnight trainer can consume.

        Returns path to the output file, or None if buffer is empty.
        """
        if not self.replay_path.exists():
            logger.info("Replay buffer empty — no fine-tuning data to export.")
            return None

        examples = []
        try:
            with open(self.replay_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        examples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Failed to read replay buffer: {e}")
            return None

        if not examples:
            return None

        # Filter: keep examples where response is non-trivial (>20 chars, no error tag)
        filtered = [
            e for e in examples
            if len(e.get("response", "")) > 20
            and "[internal error" not in e.get("response", "")
        ]

        # Take the most recent N
        batch = filtered[-max_examples:]

        if not batch:
            logger.info("No valid examples in replay buffer.")
            return None

        out_path = self.project_dir / "helix_finetune_batch.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for example in batch:
                # Format as instruction-tuning pair
                ft_entry = {
                    "instruction": example.get("prompt", ""),
                    "output": example.get("response", ""),
                    "profile": example.get("profile", "default"),
                    "timestamp": example.get("timestamp", ""),
                }
                f.write(json.dumps(ft_entry) + "\n")

        logger.info(f"Exported {len(batch)} fine-tuning examples to {out_path}")
        self._last_export_time = time.time()

        # Archive the consumed replay buffer (don't re-train on same data)
        archive_path = self.replay_path.with_suffix(
            f".{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self.replay_path.rename(archive_path)
        logger.info(f"Replay buffer archived to {archive_path.name}")

        return out_path

    def trigger_overnight_training(self):
        """
        Launch a background fine-tuning pass on the exported batch.

        Uses the same titan_venv and DeepSpeed config as Phase 1 training,
        but with a much smaller LoRA-style update (10 steps max).
        Called by the dream engine during Helix's sleep window.
        """
        batch_path = self.export_finetune_batch()
        if batch_path is None:
            return

        trainer_script = self.project_dir / "helix_overnight_trainer.py"
        if not trainer_script.exists():
            logger.warning(
                f"Overnight trainer not found at {trainer_script}. "
                "Skipping fine-tuning pass."
            )
            return

        import subprocess
        python = self.project_dir / "titan_venv" / "bin" / "python3"
        cmd = [str(python), str(trainer_script), "--batch", str(batch_path)]

        logger.info(f"Launching overnight fine-tuning: {' '.join(cmd)}")
        try:
            subprocess.Popen(
                cmd,
                cwd=str(self.project_dir),
                stdout=open(self.project_dir / "overnight_train.log", "a"),
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            logger.error(f"Failed to launch overnight trainer: {e}")
