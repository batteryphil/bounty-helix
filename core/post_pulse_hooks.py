"""
Helix — Post-Pulse Hook System

Lightweight hooks that run after each pulse completes, without blocking
the next pulse. These are the "subconscious" background processes that
observe patterns, update state, and maintain cognitive hygiene.

Inspired by Claude Code's post-sampling hooks (backgroundHousekeeping.ts,
skillImprovement.ts) — but simpler, because Helix is single-threaded
with background tasks, not event-driven.

Each hook receives a PostPulseHookContext containing:
  - thought: the model's output from this pulse
  - events: incoming events that triggered the pulse
  - pulse_count: monotonic pulse counter
  - tool_calls: list of tool call dicts from this pulse
  - spatial_state: current 8D spatial state snapshot
  - active_toolsets: set of currently enabled toolset names
  - memory_id: short-term memory ID of the stored thought (provenance)
  - lagrangian_before: sentinel snapshot BEFORE the pulse
  - lagrangian_after: sentinel snapshot AFTER the pulse

The lagrangian_before/after pair enables hooks to compute stability
deltas — measuring the perturbation a pulse caused, not just the
absolute atmospheric state.

Hooks MUST be non-blocking. If a hook needs to do LLM work, it should
queue it for the next idle period (similar to the dream engine).

Usage:
    from core.post_pulse_hooks import register_hook, run_hooks

    # Registration (at startup in main.py):
    def my_hook(ctx: PostPulseHookContext):
        if ctx.tool_calls:
            logger.info("Tools used: %s", [tc['name'] for tc in ctx.tool_calls])

    register_hook(my_hook, name="tool_logger")

    # Execution (at end of _pulse() in pulse_loop.py):
    run_hooks(hook_context)
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("helix.core.post_pulse_hooks")


class PostPulseHookContext:
    """Read-only context snapshot passed to each hook.

    Contains everything a background hook needs to observe the
    current pulse's results without modifying the main loop state.

    The lagrangian_before/after pair captures the sentinel state
    delta across the pulse, enabling hooks to measure the stability
    perturbation a pulse caused (not the noisy absolute state).
    """

    __slots__ = (
        "thought", "events", "pulse_count", "tool_calls",
        "spatial_state", "active_toolsets",
        "memory_id", "lagrangian_before", "lagrangian_after",
        "injected_belief_ids",
        # Q4 (Gemini Pass 11): THINK phase output — must be in training tuples
        # so LoRA learns reasoning is a mandatory precursor to tool execution.
        # If this is empty, the tuple is EXCLUDED from training data.
        "think_block",          # str: Phase 1 THINK output for this pulse
        # Bonus Q (Gemini Pass 11): mandate tracking for decay mechanism
        "mandate_used",         # bool: True if pulse_loop injected a mandate this pulse
        # Writable by hooks — read by self_trainer quality gate
        "novel_belief_added",   # True if belief_detector stored a new belief this pulse
        "last_fitness_delta",   # float set by fitness_evaluator hook
    )

    def __init__(
        self,
        thought: str = "",
        events: Optional[List[str]] = None,
        pulse_count: int = 0,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        spatial_state: Optional[Dict[str, Any]] = None,
        active_toolsets: Optional[Set[str]] = None,
        memory_id: int = -1,
        lagrangian_before: Optional[Dict[str, Any]] = None,
        lagrangian_after: Optional[Dict[str, Any]] = None,
        injected_belief_ids: Optional[List[str]] = None,
        think_block: str = "",
        mandate_used: bool = False,
    ):
        self.thought = thought
        self.events = events or []
        self.pulse_count = pulse_count
        self.tool_calls = tool_calls or []
        self.spatial_state = spatial_state or {}
        self.active_toolsets = active_toolsets or {"core"}
        self.memory_id = memory_id
        self.lagrangian_before = lagrangian_before or {}
        self.lagrangian_after = lagrangian_after or {}
        self.injected_belief_ids = injected_belief_ids or []
        self.think_block = think_block       # Phase 1 THINK output
        self.mandate_used = mandate_used     # True if mandate was injected
        self.novel_belief_added = False      # set True by belief_detector_hook
        self.last_fitness_delta = 0.0        # set by fitness_evaluator hook


# Type alias for hook functions
PostPulseHook = Callable[[PostPulseHookContext], None]

# Registry — hooks run in registration order
_hooks: List[PostPulseHook] = []
_hook_names: List[str] = []
_lock = threading.Lock()


def register_hook(hook: PostPulseHook, name: str = ""):
    """Register a post-pulse hook. Hooks run in registration order.

    Args:
        hook: Callable that receives a PostPulseHookContext.
              Must be non-blocking.
        name: Human-readable name for logging. Defaults to the
              function's __name__.
    """
    display_name = name or getattr(hook, "__name__", "anonymous")
    with _lock:
        _hooks.append(hook)
        _hook_names.append(display_name)
    logger.info("Post-pulse hook registered: %s", display_name)



# Persistent hook worker — one long-lived daemon thread processes all pulses
# sequentially from a queue. Avoids spawning 90+ dead threads over a long run.
import queue as _queue
_hook_queue: _queue.Queue = _queue.Queue(maxsize=4)  # max 4 queued pulses, then drop oldest
_hook_worker_started = False


def _ensure_hook_worker():
    """Start the persistent hook worker thread once."""
    global _hook_worker_started
    if _hook_worker_started:
        return

    def _worker():
        while True:
            try:
                ctx, hooks = _hook_queue.get(timeout=60)
                for hook, name in hooks:
                    try:
                        hook(ctx)
                    except Exception as e:
                        logger.warning("Post-pulse hook '%s' failed: %s", name, e, exc_info=True)
                _hook_queue.task_done()
            except _queue.Empty:
                continue  # stay alive
            except Exception as e:
                logger.warning("Hook worker error: %s", e)

    t = threading.Thread(target=_worker, daemon=True, name="hook-worker")
    t.start()
    _hook_worker_started = True


def run_hooks(context: PostPulseHookContext):
    """Queue this pulse's hooks to run in the persistent hook worker thread.

    Non-blocking: returns immediately. Hooks run sequentially in a single
    long-lived daemon thread — no per-pulse thread spawning.
    If the queue is full (4 pending), the oldest is dropped and this one queued.
    """
    with _lock:
        hooks = list(zip(_hooks, _hook_names))

    _ensure_hook_worker()
    try:
        _hook_queue.put_nowait((context, hooks))
    except _queue.Full:
        # Queue full — drop oldest, add newest
        try:
            _hook_queue.get_nowait()
            _hook_queue.task_done()
        except _queue.Empty:
            pass
        try:
            _hook_queue.put_nowait((context, hooks))
        except _queue.Full:
            pass


def run_hooks_sync(context: PostPulseHookContext):
    """Synchronous fallback — blocks until all hooks complete.
    Use only when the next action genuinely depends on hook output.
    """
    with _lock:
        hooks = list(zip(_hooks, _hook_names))
    for hook, name in hooks:
        try:
            hook(context)
        except Exception as e:
            logger.warning(
                "Post-pulse hook '%s' failed: %s", name, e,
                exc_info=True,
            )


def get_registered_hooks() -> List[str]:
    """Return names of all registered hooks (for diagnostics)."""
    with _lock:
        return list(_hook_names)
