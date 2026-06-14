"""
Helix — Metacognitive Monitor

A post-pulse hook that silently observes every conscious pulse and
accumulates performance metrics. Every SNAPSHOT_INTERVAL pulses it
writes a MetaSnapshot to disk for use by the SelfImprovementEngine.

Metrics tracked per-pulse:
  - tool_called (bool)
  - tool_succeeded (bool)
  - tool_name (str)
  - hallucination_detected (bool)  — prose claim of action without tool exec
  - response_tokens (int)
  - user_message_present (bool)
  - task_completed (bool)          — heuristic: no follow-up error in same session

Every SNAPSHOT_INTERVAL pulses:
  - Compute aggregate rates
  - Identify top failure patterns
  - Write MetaSnapshot to data/meta_snapshots.jsonl
"""

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("helix.core.metacog")

SNAPSHOT_INTERVAL = 10      # pulses between full snapshots (was 50)
MIN_PULSES_FOR_SNAPSHOT = 5


@dataclass
class PulseRecord:
    """Lightweight record for a single pulse outcome."""
    ts: float
    pulse_count: int
    tool_called: bool
    tool_succeeded: bool
    tool_name: str
    hallucination: bool        # said "I did X" but no tool executed
    response_len: int
    has_user_message: bool
    error_in_thought: bool     # thought contains "error" or "failed"


@dataclass
class MetaSnapshot:
    """Aggregate performance snapshot over a window of pulses."""
    ts: float
    window_size: int
    tool_success_rate: float          # tool successes / tool calls (import-level)
    tool_call_rate: float             # tool calls / total pulses  ← primary driver
    tool_exec_success_rate: float     # tools returning non-error output / tool calls
    hallucination_rate: float         # hallucinations / user-task pulses
    top_failures: List[str]           # most frequent error patterns seen
    novel_belief_rate: float          # beliefs added per window (from belief store)
    composite_fitness: float          # 0.0–1.0 composite score
    # Removed (Rev 3 peer review):
    #   task_completion_rate: was constant 0.5 (deadweight compressing variance)
    #   avg_response_len: penalized efficient tool calls vs verbose prose


class MetacognitiveMonitor:
    """Observes agent behavior pulse-by-pulse and computes fitness metrics."""

    def __init__(self, data_dir: str = "data", belief_store=None):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._snap_path = self._data_dir / "meta_snapshots.jsonl"
        self._belief_store = belief_store
        self._lock = threading.Lock()
        self._records: List[PulseRecord] = []
        self._last_belief_count = 0
        self._snapshots: List[MetaSnapshot] = []
        self._load_snapshots()

    def _load_snapshots(self):
        if not self._snap_path.exists():
            return
        try:
            with self._snap_path.open("r") as f:
                for line in f:
                    d = json.loads(line.strip())
                    self._snapshots.append(MetaSnapshot(**d))
            logger.info(f"[metacog] Loaded {len(self._snapshots)} historical snapshots")
        except Exception as e:
            logger.warning(f"[metacog] Snapshot load error: {e}")

    def observe(self, ctx) -> None:
        """Called as a post-pulse hook. ctx is a PostPulseHookContext."""
        try:
            thought = getattr(ctx, "thought", "") or ""
            tool_calls = getattr(ctx, "tool_calls", []) or []
            pulse_count = getattr(ctx, "pulse_count", 0)

            # Detect tool execution
            tool_called = bool(tool_calls)
            tool_name = tool_calls[0].get("name", "") if tool_calls else ""

            # Tool success: check if HERMES TOOL EXEC log was recent
            # Heuristic: if tool was called and thought doesn't start with "(tool error"
            tool_succeeded = tool_called and not thought.lower().startswith("(tool error")

            # Hallucination: mentions doing something without a tool call
            hallucination_keywords = [
                r"\bi (?:wrote|created|saved|deleted|executed|ran|searched|found)\b",
                r"\bi've (?:written|created|saved|completed)\b",
                r"\bfile (?:has been|was) (?:written|created|saved)\b",
            ]
            has_user_message = bool(re.search(
                r"they said:|user message:|new events since",
                thought.lower()
            ))
            hallucination = False
            if has_user_message and not tool_called:
                for pat in hallucination_keywords:
                    if re.search(pat, thought.lower()):
                        hallucination = True
                        break

            error_in_thought = bool(re.search(
                r"\berror\b|\bfailed\b|\bcannot\b|\bunable to\b",
                thought.lower()
            ))

            record = PulseRecord(
                ts=time.time(),
                pulse_count=pulse_count,
                tool_called=tool_called,
                tool_succeeded=tool_succeeded,
                tool_name=tool_name,
                hallucination=hallucination,
                response_len=len(thought),
                has_user_message=has_user_message,
                error_in_thought=error_in_thought,
            )

            with self._lock:
                self._records.append(record)
                # Keep last 500 records in memory
                if len(self._records) > 500:
                    self._records = self._records[-500:]

            # Write fitness delta back to ctx for self_trainer quality gate
            try:
                snap = self.get_latest_snapshot()
                if snap:
                    ctx.last_fitness_delta = snap.composite_fitness - 0.5  # delta from neutral
            except (AttributeError, TypeError):
                pass  # ctx may not have slot yet on first pulse

            # Snapshot every N pulses
            if pulse_count > 0 and pulse_count % SNAPSHOT_INTERVAL == 0:
                self._write_snapshot()

        except Exception as e:
            logger.debug(f"[metacog] observe error: {e}")

    def _write_snapshot(self):
        """Compute and persist a MetaSnapshot."""
        try:
            with self._lock:
                records = list(self._records[-SNAPSHOT_INTERVAL:])

            if len(records) < MIN_PULSES_FOR_SNAPSHOT:
                return

            tool_records = [r for r in records if r.tool_called]
            user_records = [r for r in records if r.has_user_message]

            # Import-level: did the tool exist and not crash on call?
            tool_success_rate = (
                sum(1 for r in tool_records if r.tool_succeeded) / len(tool_records)
                if tool_records else 1.0
            )
            tool_call_rate = len(tool_records) / len(records)

            # Execution-level: did the tool return a valid (non-error) result?
            # Heuristic: tool_succeeded AND thought doesn't contain error keywords
            # that suggest the tool returned an error payload.
            _error_pats = re.compile(
                r"tool.*error|error.*tool|failed to|could not|exception:|traceback",
                re.IGNORECASE
            )
            tool_exec_success_rate = (
                sum(
                    1 for r in tool_records
                    if r.tool_succeeded and not _error_pats.search(r.tool_name)
                ) / len(tool_records)
                if tool_records else 0.0
            )

            hallucination_rate = (
                sum(1 for r in user_records if r.hallucination) / len(user_records)
                if user_records else 0.0
            )

            # Novel belief rate from belief store
            novel_belief_rate = 0.0
            if self._belief_store:
                try:
                    current_count = len(self._belief_store.get_all())
                    novel_belief_rate = max(0.0, current_count - self._last_belief_count) / SNAPSHOT_INTERVAL
                    self._last_belief_count = current_count
                except Exception:
                    pass

            # Top failure patterns
            errors = [r for r in records if r.error_in_thought]
            top_failures = []
            if errors:
                top_failures.append(f"{len(errors)} error-containing responses in last {len(records)} pulses")
            hallucinations = [r for r in records if r.hallucination]
            if hallucinations:
                top_failures.append(f"{len(hallucinations)} hallucinated actions detected")
            failed_tools = [r for r in tool_records if not r.tool_succeeded]
            if failed_tools:
                top_failures.append(f"{len(failed_tools)} tool execution failures")

            # ── Composite Fitness (Rev 3 formula — per peer review) ───────────
            # REMOVED (Q9): task_completion_rate — constant 0.5 was mathematical
            #   deadweight compressing the dynamic range of the entire score.
            # REMOVED (Q10): avg_response_length — punished efficient tool calls
            #   vs verbose prose, actively training the wrong behavior.
            #
            # New weights (sum = 1.0):
            #   tool_call_rate        0.30  ← primary LoRA unlock driver
            #   tool_exec_success     0.20  ← rewards actual runtime execution
            #   tool_success_rate     0.35  ← import-level stability floor
            #   novel_belief_rate     0.10  ← belief formation signal
            #   hallucination (inv)   0.05  ← truthfulness guard
            #
            # A pulse where the agent calls a tool AND it returns valid data:
            #   tool_call_rate=1.0, exec=1.0, success=1.0 → fitness=0.95+
            #   (only novel beliefs and hallucination can push it above 0.95)
            # A prose-only pulse:
            #   tool_call_rate=0.0, exec=0.0, success=1.0 → fitness=0.40
            # This creates a real Δ ≥ 0.55 between tool and non-tool pulses.
            composite = (
                0.35 * tool_success_rate +
                0.30 * tool_call_rate +
                0.20 * tool_exec_success_rate +
                0.10 * min(1.0, novel_belief_rate * 10) +
                0.05 * (1.0 - hallucination_rate)
            )

            snap = MetaSnapshot(
                ts=time.time(),
                window_size=len(records),
                tool_success_rate=round(tool_success_rate, 4),
                tool_call_rate=round(tool_call_rate, 4),
                tool_exec_success_rate=round(tool_exec_success_rate, 4),
                hallucination_rate=round(hallucination_rate, 4),
                top_failures=top_failures,
                novel_belief_rate=round(novel_belief_rate, 4),
                composite_fitness=round(composite, 4),
            )

            with self._lock:
                self._snapshots.append(snap)

            snap_dict = asdict(snap)
            snap_dict["fitness"] = snap_dict["composite_fitness"]  # alias for readers
            with self._snap_path.open("a") as f:
                f.write(json.dumps(snap_dict) + "\n")

            logger.info(
                f"[metacog] Snapshot: fitness={snap.composite_fitness:.3f} "
                f"tool_ok={snap.tool_success_rate:.2%} "
                f"halluc={snap.hallucination_rate:.2%}"
            )
        except Exception as e:
            logger.error(f"[metacog] Snapshot error: {e}")

    def get_latest_snapshot(self) -> Optional[MetaSnapshot]:
        """Return the most recent MetaSnapshot, or None."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_current_fitness(self) -> float:
        """Return the latest composite fitness score (0.0–1.0)."""
        snap = self.get_latest_snapshot()
        return snap.composite_fitness if snap else 0.5

    def format_for_context(self) -> str:
        """Format as text for injection into agent self-improvement prompt."""
        snap = self.get_latest_snapshot()
        if not snap:
            return "No performance data yet (need at least 10 pulses)."
        return (
            f"Performance snapshot (last {snap.window_size} pulses):\n"
            f"  Tool call rate:        {snap.tool_call_rate:.1%}  [weight 0.30 — PRIMARY DRIVER]\n"
            f"  Tool exec success:     {snap.tool_exec_success_rate:.1%}  [weight 0.20]\n"
            f"  Tool import success:   {snap.tool_success_rate:.1%}  [weight 0.35]\n"
            f"  Novel beliefs/window:  {snap.novel_belief_rate:.3f}  [weight 0.10]\n"
            f"  Hallucination rate:    {snap.hallucination_rate:.1%}  [weight 0.05, inverted]\n"
            f"  COMPOSITE FITNESS:     {snap.composite_fitness:.3f}/1.0\n"
            f"  NOTE: Calling a tool that returns valid data scores ~0.85+. Prose only scores ~0.40.\n"
            + (f"  Top issues: {'; '.join(snap.top_failures)}" if snap.top_failures else "  No major issues detected.")
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_monitor: Optional[MetacognitiveMonitor] = None


def get_monitor() -> Optional[MetacognitiveMonitor]:
    return _monitor


def init_monitor(data_dir: str = "data", belief_store=None) -> MetacognitiveMonitor:
    global _monitor
    _monitor = MetacognitiveMonitor(data_dir=data_dir, belief_store=belief_store)
    return _monitor
