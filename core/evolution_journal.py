"""
Helix — Evolution Journal

Append-only JSONL log of every self-modification the agent makes.
This is the system's memory of its own development history.

Each entry records:
  - What changed (type, path, description)
  - Why (the reasoning the agent gave)
  - The outcome (PASS/FAIL/REVERTED)
  - The fitness delta (did the agent get better or worse?)
  - The revert patch (original content, for rollback)

The journal is read by the SelfImprovementEngine to avoid repeating
failed experiments and to build on successful ones.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("helix.core.evolution_journal")


@dataclass
class EvolutionEntry:
    """A single self-modification event."""
    ts: float                        # Unix timestamp
    type: str                        # tool_addition | tool_fix | prompt_tuning | config_change | lora_step | code_write
    path: str                        # file modified
    description: str                 # human-readable intent
    content_snippet: str             # first 300 chars of new content
    revert_patch: str               # original content (for rollback)
    test_result: str                 # PASS | FAIL | SKIP | TIMEOUT
    fitness_before: float            # composite fitness score before change
    fitness_after: float             # composite fitness score after change
    fitness_delta: float             # after - before
    committed: bool                  # True = kept, False = reverted
    reasoning: str                   # agent's stated reason
    error: str                       # error message if FAIL
    tags: List[str]                  # e.g. ["tool", "web", "search"]


class EvolutionJournal:
    """Thread-safe append-only JSONL evolution log."""

    def __init__(self, data_dir: str = "data"):
        self._path = Path(data_dir) / "evolution_journal.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._entries: List[EvolutionEntry] = []
        self._load()

    def _load(self):
        """Load existing journal entries on startup."""
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            self._entries.append(EvolutionEntry(**{
                                k: d.get(k, "") for k in EvolutionEntry.__dataclass_fields__
                            }))
                        except Exception:
                            pass
            logger.info(f"[journal] Loaded {len(self._entries)} evolution entries")
        except Exception as e:
            logger.warning(f"[journal] Load error: {e}")

    def record(self, entry: EvolutionEntry):
        """Append an entry to the journal."""
        with self._lock:
            self._entries.append(entry)
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(entry)) + "\n")
                logger.info(
                    f"[journal] Recorded: type={entry.type} path={entry.path} "
                    f"result={entry.test_result} Δfit={entry.fitness_delta:+.3f} "
                    f"committed={entry.committed}"
                )
            except Exception as e:
                logger.error(f"[journal] Write error: {e}")

    def record_code_write(self, action: str, path: str, snippet: str, outcome: str):
        """Quick-record a code write event (minimal metadata)."""
        entry = EvolutionEntry(
            ts=time.time(),
            type=action,
            path=path,
            description=f"Code write via {action}",
            content_snippet=snippet,
            revert_patch="",
            test_result=outcome,
            fitness_before=0.0,
            fitness_after=0.0,
            fitness_delta=0.0,
            committed=True,
            reasoning="",
            error="",
            tags=[],
        )
        self.record(entry)

    def get_recent(self, n: int = 20) -> List[Dict]:
        """Return the last N entries as dicts."""
        with self._lock:
            return [asdict(e) for e in self._entries[-n:]]

    def get_failures(self, n: int = 10) -> List[Dict]:
        """Return the last N failed/reverted entries."""
        with self._lock:
            failures = [e for e in self._entries if not e.committed or e.test_result == "FAIL"]
            return [asdict(e) for e in failures[-n:]]

    def get_stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        with self._lock:
            total = len(self._entries)
            committed = sum(1 for e in self._entries if e.committed)
            reverted = total - committed
            type_counts: Dict[str, int] = {}
            for e in self._entries:
                type_counts[e.type] = type_counts.get(e.type, 0) + 1
            avg_delta = (
                sum(e.fitness_delta for e in self._entries) / total
                if total else 0.0
            )
            return {
                "total": total,
                "committed": committed,
                "reverted": reverted,
                "commit_rate": committed / total if total else 0.0,
                "avg_fitness_delta": avg_delta,
                "by_type": type_counts,
            }

    def format_for_context(self, n: int = 5) -> str:
        """Format recent entries as text for injection into agent context."""
        recent = self.get_recent(n)
        if not recent:
            return "No self-modifications recorded yet."
        lines = ["Recent self-modifications:"]
        for e in recent:
            symbol = "✓" if e["committed"] else "✗"
            lines.append(
                f"  {symbol} [{e['type']}] {e['path']} — {e['description'][:60]} "
                f"(Δfit={e['fitness_delta']:+.3f}, {e['test_result']})"
            )
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────

_journal: Optional[EvolutionJournal] = None


def get_journal(data_dir: str = "data") -> EvolutionJournal:
    """Get or create the singleton EvolutionJournal."""
    global _journal
    if _journal is None:
        _journal = EvolutionJournal(data_dir=data_dir)
    return _journal


# Convenience alias — used as `from core.evolution_journal import journal`
# after init_journal() is called from main.py
journal: Optional[EvolutionJournal] = None


def init_journal(data_dir: str = "data") -> EvolutionJournal:
    """Initialize the module-level singleton. Call once from main.py."""
    global journal
    journal = get_journal(data_dir)
    return journal
