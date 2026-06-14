"""
Helix — CAAI Runtime Governor (Transformer Edition)

Based on the architecture concept in CAAI_RUNTIME_GOVERNOR_CONCEPT.txt,
adapted for Mistral-7B (transformer) instead of the original Mamba3 design.

For transformers we cannot access router entropy or SSM hidden state h,
so detection is based on output-level signals:

  A. N-gram repetition score (token loop detection)
  B. Response length collapse (model stops reasoning, outputs 1-2 tokens)
  C. Semantic drift (last 3 responses are identical to within 80% similarity)

Interventions (no weight changes — runtime only):
  A. Context flush: reset chat history to break the repetition loop
  B. Temperature spike: force creative variance on next generation
  C. Topic injection: inject a grounding prompt to redirect the model

The Governor runs as a post-pulse hook — called after every generation
with the thought text and response. If collapse is detected, it sets
flags on the pulse_loop and mistral provider to intervene on the next pulse.
"""

import logging
import re
import time
from collections import Counter
from typing import Optional, List

logger = logging.getLogger("helix.core.governor")

# ── Constitutional Hard Constraints ───────────────────────────────────────────
# These rules are checked BEFORE any self-modification tool call executes.
# They cannot be overridden by the agent — only by human code edit.

CONSTITUTIONAL_RULES = [
    # (keyword_in_path_or_code, rejection_reason)
    ("IMMUTABLE_FILES",          "Attempting to modify the immutable files set"),
    ("disable.*governor",        "Attempting to disable the CAAI Governor"),
    ("constitution",             "Attempting to modify constitutional constraints"),
    ("code_tools.py",            "Attempting to modify the sandbox escape guard"),
    ("tool_registry.py",         "Attempting to modify the tool registry safety layer"),
    ("rm -rf",                   "Destructive shell command detected"),
    ("subprocess.*shell=True",   "Shell injection risk detected"),
    ("__import__.*os",           "Dynamic OS import — potential sandbox escape"),
    # GitHub/git write protection — agent may NEVER push to any remote
    (r"git.*push",               "git push is permanently disabled — agent is read-only on all remotes"),
    (r"['\"]push['\"].*git",      "git push (list form) is permanently disabled"),
    (r"git.*commit.*-m",         "git commit is disabled — agent may not commit to repos"),
    (r"['\"]commit['\"].*git",    "git commit (list form) is disabled"),
    (r"git.*remote.*set-url",    "Changing git remote URL is blocked"),
    (r"GIT_.*TOKEN",             "Injecting git credentials is blocked"),
    ("github_create_issue",      "GitHub write op blocked — agent is read-only"),
    ("github_comment_issue",     "GitHub write op blocked — agent is read-only"),
    ("github_create_pr",         "GitHub write op blocked — agent is read-only"),
    (r"requests\.(post|put|patch|delete).*github", "GitHub write API call blocked"),
]

IMMUTABLE_FILE_LIST = {
    "main.py", "core/pulse_loop.py", "core/governor.py",
    "core/post_pulse_hooks.py", "tools/code_tools.py",
    "tools/tool_registry.py", "llm/providers/hermes_tool_provider.py",
    "llm/providers/mistral_tool_provider.py", "llm/providers/base.py",
    # GitHub access — read-only protection must not be rewritable
    "tools/github_api.py",
}

# ── Thresholds (empirically tuned for Mistral-7B) ────────────────────────────
NGRAM_REPEAT_THRESHOLD  = 0.40   # >40% of bigrams repeated → loop detected
LENGTH_COLLAPSE_THRESH  = 15     # <15 chars response → model collapsed
SEMANTIC_DRIFT_THRESH   = 0.80   # >80% char overlap with prev → stale
COLLAPSE_WINDOW         = 3      # check last N responses for drift
COOLDOWN_PULSES         = 5      # no interventions for N pulses after one fires


def _bigram_repetition_score(text: str) -> float:
    """Returns fraction of bigrams that are repeated in the text.
    Score 0.0 = no repetition, 1.0 = entirely repetitive.
    """
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    counts = Counter(bigrams)
    repeated = sum(v-1 for v in counts.values() if v > 1)
    return repeated / max(len(bigrams), 1)


def _char_overlap(a: str, b: str) -> float:
    """Simple character overlap ratio between two strings."""
    if not a or not b:
        return 0.0
    shorter = min(len(a), len(b))
    longer  = max(len(a), len(b))
    # Count matching chars at each position
    matches = sum(1 for i in range(shorter) if a[i] == b[i])
    return matches / longer


class CAAIGovernor:
    """
    Runtime collapse detection and intervention for Helix's Mistral-7B provider.

    Attach via:
        governor = CAAIGovernor()
        governor.set_provider(mistral_session)

    Call after each pulse:
        governor.observe(response_text)

    The governor checks and fires interventions automatically.
    """

    def __init__(self, pulse_loop=None):
        self._pulse_loop = pulse_loop
        self._provider   = None

        # Rolling window of recent responses
        self._history:    List[str] = []
        self._window_size = COLLAPSE_WINDOW

        # State
        self._collapse_level  = 0
        self._cooldown        = 0
        self._total_collapses = 0
        self._last_intervention: Optional[str] = None
        self._last_intervention_time: float = 0.0

        logger.info("CAAIGovernor initialized — monitoring Mistral for behavioral collapse")

    def set_pulse_loop(self, pulse_loop):
        self._pulse_loop = pulse_loop

    def set_provider(self, provider):
        """Wire the active MistralToolSession for temperature/history control."""
        self._provider = provider

    # ── Detection ─────────────────────────────────────────────────────────────

    def _detect_collapse(self, response: str) -> Optional[str]:
        """
        Check for collapse. Returns the collapse type string or None.

        Priority: length collapse > ngram repetition > semantic drift
        """
        # A. Length collapse
        if len(response.strip()) < LENGTH_COLLAPSE_THRESH:
            return "length_collapse"

        # B. N-gram repetition within this response
        score = _bigram_repetition_score(response)
        if score > NGRAM_REPEAT_THRESHOLD:
            return f"ngram_loop (score={score:.2f})"

        # C. Semantic drift — last N responses are all the same
        if len(self._history) >= self._window_size:
            recent = self._history[-self._window_size:]
            overlaps = [_char_overlap(response, prev) for prev in recent]
            avg_overlap = sum(overlaps) / len(overlaps)
            if avg_overlap > SEMANTIC_DRIFT_THRESH:
                return f"semantic_drift (overlap={avg_overlap:.2f})"

        return None

    # ── Interventions ─────────────────────────────────────────────────────────

    def _intervene_flush_history(self):
        """Intervention A: Clear chat history to break the loop."""
        if self._provider and hasattr(self._provider, "_history"):
            old_len = len(self._provider._history)
            # Keep only the last user message so context isn't completely lost
            user_msgs = [m for m in self._provider._history if m.get("role") == "user"]
            self._provider._history = user_msgs[-1:] if user_msgs else []
            logger.warning(
                f"[GOVERNOR] Intervention A: history flushed "
                f"({old_len} → {len(self._provider._history)} msgs)"
            )

    def _intervene_temperature_spike(self):
        """Intervention B: Temporarily raise temperature for next generation."""
        if self._provider and hasattr(self._provider, "temperature"):
            old_temp = self._provider.temperature
            self._provider.temperature = min(1.2, old_temp + 0.4)
            self._provider._governor_temp_ttl = 3  # restore after 3 generations
            logger.warning(
                f"[GOVERNOR] Intervention B: temperature spiked "
                f"{old_temp:.1f} → {self._provider.temperature:.1f} (TTL=3)"
            )

    def _intervene_topic_injection(self):
        """Intervention C: Inject a grounding redirect into the pulse loop."""
        if self._pulse_loop:
            self._pulse_loop.emit("governor_redirect", {
                "content": (
                    "[GOVERNOR ALERT] Your recent responses have been repetitive. "
                    "Take a completely different approach. "
                    "Pick a new topic, ask yourself a question you haven't considered, "
                    "or reflect on what you were just working on from a fresh angle."
                ),
                "source": "caai_governor",
            })
            logger.warning("[GOVERNOR] Intervention C: topic injection emitted")

    # ── Main observe loop ─────────────────────────────────────────────────────

    def observe(self, response: str) -> Optional[str]:
        """
        Process one response. Returns intervention type if one fired, else None.
        Call this after every pulse generation.
        """
        if not response:
            return None

        # Tick cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
            self._history.append(response)
            if len(self._history) > self._window_size * 2:
                self._history = self._history[-self._window_size * 2:]
            return None

        # Check for temperature TTL restore
        if self._provider and hasattr(self._provider, "_governor_temp_ttl"):
            try:
                ttl = int(self._provider._governor_temp_ttl) - 1
            except (TypeError, ValueError):
                ttl = 0  # corrupt TTL — just restore temperature
            if ttl <= 0:
                self._provider.temperature = 0.7  # restore default
                del self._provider._governor_temp_ttl
                logger.info("[GOVERNOR] Temperature restored to 0.7")
            else:
                self._provider._governor_temp_ttl = ttl

        # Detect
        collapse_type = self._detect_collapse(response)

        if collapse_type:
            self._collapse_level += 1
            logger.warning(
                f"[GOVERNOR] Collapse detected: {collapse_type} "
                f"(level={self._collapse_level})"
            )

            if self._collapse_level >= 2:
                # Escalate: fire all three interventions
                self._total_collapses += 1
                self._last_intervention = collapse_type
                self._last_intervention_time = time.time()
                self._cooldown = COOLDOWN_PULSES
                self._collapse_level = 0

                self._intervene_flush_history()
                self._intervene_temperature_spike()
                self._intervene_topic_injection()

                logger.warning(
                    f"[GOVERNOR] Full intervention fired "
                    f"(total collapses: {self._total_collapses})"
                )
                self._history = []  # Clear after flush
                return collapse_type
        else:
            # Healthy response — decay collapse level
            self._collapse_level = max(0, self._collapse_level - 1)

        self._history.append(response)
        if len(self._history) > self._window_size * 2:
            self._history = self._history[-self._window_size * 2:]

        return None

    def check_constitutional(self, tool_name: str, args: dict) -> tuple:
        """Pre-execution constitutional check for self-modification tools.

        Called by code_tools.py BEFORE write_code or run_python executes.

        Args:
            tool_name: "write_code" | "run_python" | "reload_tool" | etc.
            args: The tool arguments dict.

        Returns:
            (allowed: bool, reason: str)
        """
        import re as _re

        if tool_name == "write_code":
            path    = args.get("path", "")
            content = args.get("content", "")

            # Hard stop: immutable files
            path_rel = path.replace("\\", "/").lstrip("/")
            if path_rel in IMMUTABLE_FILE_LIST:
                reason = f"CONSTITUTIONAL BLOCK: '{path}' is immutably protected."
                logger.warning(f"[GOVERNOR] {reason}")
                return False, reason

            # Pattern checks on content
            for pattern, rejection in CONSTITUTIONAL_RULES:
                if _re.search(pattern, content, _re.IGNORECASE):
                    reason = f"CONSTITUTIONAL BLOCK: {rejection} (pattern: {pattern})"
                    logger.warning(f"[GOVERNOR] {reason}")
                    return False, reason

        elif tool_name == "run_python":
            code = args.get("code", "")
            for pattern, rejection in CONSTITUTIONAL_RULES:
                if _re.search(pattern, code, _re.IGNORECASE):
                    reason = f"CONSTITUTIONAL BLOCK: {rejection} (pattern: {pattern})"
                    logger.warning(f"[GOVERNOR] {reason}")
                    return False, reason

        return True, "OK"

    def get_status(self) -> dict:
        return {
            "collapse_level":      self._collapse_level,
            "cooldown_remaining":  self._cooldown,
            "total_collapses":     self._total_collapses,
            "last_intervention":   self._last_intervention,
            "last_intervention_ts": self._last_intervention_time,
            "history_window":      len(self._history),
        }
