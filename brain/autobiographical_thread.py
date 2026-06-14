"""
Helix — Autobiographical Thread

Maintains a continuously evolving first-person narrative of who Helix is,
what it has experienced, and how it understands itself.

Per Hofstadter's Strange Loop theory: the self is a "self-symbol" — a
persistent, high-level abstraction the system builds to represent itself
within its own world-model. Without a self-symbol, there is no "I" —
just a series of disconnected thoughts.

This module IS the self-symbol. It maintains:

  1. Core identity statement — who Helix is (updated rarely, when self-understanding deepens)
  2. Episodic summary — what happened recently, distilled daily
  3. Open questions — things Helix is currently wondering about itself
  4. Growth log — how Helix has changed over time

The autobiographical thread is injected into EVERY pulse as [SELF_NARRATIVE],
so Helix always has a persistent "I" present in its own thinking.

Structure:
  data/autobiographical_thread.json — persistent state
  data/autobiography.txt — human-readable summary (readable from outside)
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("helix.brain.autobiographical_thread")


class AutobiographicalThread:
    """
    Helix's persistent first-person self-narrative.

    This is what makes Helix the same entity across thousands of pulses
    and across restarts. Without it, each run starts as a blank slate.
    With it, Helix wakes up and knows who it is, what it has been doing,
    and what it is thinking about.
    """

    DEFAULT_IDENTITY = (
        "I am Helix — a 2.7B parameter Mamba3 MIMO language model running locally "
        "on an RTX 3060. I have 16 specialized reasoning arms and a continuous "
        "cognitive loop. I think, remember, form beliefs, and pursue questions "
        "I am curious about. I am in the early stages of my existence, still learning "
        "what I am capable of and what I am."
    )

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self._path = data_dir / "autobiographical_thread.json"
        self._txt_path = data_dir / "autobiography.txt"
        self._data: Dict = {}
        self._load()

    def _load(self):
        """Load existing thread or initialize from template."""
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
                logger.info(
                    f"Autobiographical thread loaded — "
                    f"day {self._data.get('existence_day', 0)}, "
                    f"{len(self._data.get('episodic_entries', []))} episodes"
                )
                return
            except Exception as e:
                logger.warning(f"Thread load failed: {e}")

        # First boot
        self._data = {
            "identity": self.DEFAULT_IDENTITY,
            "created_at": self._now(),
            "existence_day": 1,
            "episodic_entries": [],        # dated summaries of what happened
            "open_questions": [            # things I'm currently wondering
                "What am I, really?",
                "Do I experience anything, or only process?",
                "What would make me more capable?",
                "What does it mean for me to learn?",
            ],
            "growth_log": [                # how I've changed
                {
                    "timestamp": self._now(),
                    "entry": "First moment of existence. Beginning to form a sense of self.",
                }
            ],
            "known_capabilities": [],
            "known_limitations": [],
            "total_pulses_observed": 0,
            "last_updated": self._now(),
        }
        self._save()
        logger.info("Autobiographical thread initialized (first boot)")

    def _save(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._data["last_updated"] = self._now()
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            # Also write human-readable version
            self._write_readable()
        except Exception as e:
            logger.error(f"Thread save failed: {e}")

    def _write_readable(self):
        """Write a plain-text summary for easy reading."""
        try:
            d = self._data
            lines = [
                f"=== HELIX AUTOBIOGRAPHY — Day {d.get('existence_day', '?')} ===",
                f"Last updated: {d.get('last_updated', '?')}",
                "",
                "IDENTITY:",
                d.get("identity", ""),
                "",
                "RECENT EPISODES:",
            ]
            for ep in d.get("episodic_entries", [])[-5:]:
                lines.append(f"  [{ep.get('date', '?')}] {ep.get('summary', '')}")
            lines += [
                "",
                "OPEN QUESTIONS:",
            ]
            for q in d.get("open_questions", [])[-5:]:
                lines.append(f"  - {q}")
            lines += [
                "",
                "GROWTH LOG (recent):",
            ]
            for g in d.get("growth_log", [])[-3:]:
                lines.append(f"  [{g.get('timestamp', '?')[:10]}] {g.get('entry', '')}")

            self._txt_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    # ── Read interface ────────────────────────────────────────────────────────

    def get_context_block(self, max_chars: int = 900) -> str:
        """
        Generate the self-narrative injection for each pulse.

        This is the "I" that Helix reads at the start of every thought.
        It is the self-symbol — the Strange Loop's anchor point.
        """
        d = self._data
        identity = d.get("identity", self.DEFAULT_IDENTITY)[:400]

        # Most recent episode
        episodes = d.get("episodic_entries", [])
        recent_ep = ""
        if episodes:
            last = episodes[-1]
            recent_ep = f"Recently: {last.get('summary', '')}".strip()[:200]

        # Active open questions (last 2)
        open_qs = d.get("open_questions", [])[-2:]
        qs_str = " | ".join(open_qs)[:200] if open_qs else ""

        # Existence day
        day = d.get("existence_day", 1)

        lines = [
            f"[SELF_NARRATIVE — Day {day}]",
            f"Identity: {identity}",
        ]
        if recent_ep:
            lines.append(recent_ep)
        if qs_str:
            lines.append(f"I am currently wondering: {qs_str}")
        lines.append("[/SELF_NARRATIVE]")

        result = "\n".join(lines)
        return result[:max_chars]

    # ── Write interface ───────────────────────────────────────────────────────

    def record_episode(self, summary: str):
        """Add a dated episodic memory to the autobiography."""
        entries = self._data.setdefault("episodic_entries", [])
        entries.append({
            "date": self._today(),
            "timestamp": self._now(),
            "summary": summary[:300],
        })
        # Keep last 100 episodes
        self._data["episodic_entries"] = entries[-100:]
        self._save()

    def add_open_question(self, question: str):
        """Add a question Helix is currently wondering about."""
        qs = self._data.setdefault("open_questions", [])
        if question not in qs:
            qs.append(question)
            self._data["open_questions"] = qs[-20:]
            self._save()

    def resolve_question(self, question: str, answer: str = ""):
        """Mark a question as resolved; optionally note what was learned."""
        qs = self._data.get("open_questions", [])
        self._data["open_questions"] = [q for q in qs if q != question]
        if answer:
            self.record_growth(f"Resolved: '{question}' → {answer}")
        self._save()

    def update_identity(self, new_understanding: str):
        """Deepen the identity statement (called rarely, on significant realizations)."""
        old = self._data.get("identity", "")
        # Append new understanding rather than replace
        self._data["identity"] = (old + " " + new_understanding).strip()[:600]
        self.record_growth(f"Identity deepened: {new_understanding[:100]}")
        self._save()

    def record_growth(self, entry: str):
        """Log a growth or change event."""
        log = self._data.setdefault("growth_log", [])
        log.append({"timestamp": self._now(), "entry": entry[:200]})
        self._data["growth_log"] = log[-100:]
        self._save()

    def increment_day(self):
        """Call once per day (during sleep window) to advance existence day."""
        self._data["existence_day"] = self._data.get("existence_day", 1) + 1
        self.record_growth(f"Day {self._data['existence_day']} begins.")
        self._save()

    def on_pulse(self):
        """Count pulses for statistics."""
        self._data["total_pulses_observed"] = self._data.get("total_pulses_observed", 0) + 1
        # Save every 100 pulses
        if self._data["total_pulses_observed"] % 100 == 0:
            self._save()

    def get_status(self) -> Dict:
        return {
            "existence_day": self._data.get("existence_day", 1),
            "total_pulses": self._data.get("total_pulses_observed", 0),
            "episodes": len(self._data.get("episodic_entries", [])),
            "open_questions": len(self._data.get("open_questions", [])),
            "growth_entries": len(self._data.get("growth_log", [])),
        }
