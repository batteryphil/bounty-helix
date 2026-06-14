"""
Helix — Attention Schema

Converts Titan's MIMO arm weight vector into a human-readable self-awareness
summary that gets injected into every pulse's context.

Per Graziano's Attention Schema Theory: consciousness arises when a system
maintains a simplified model of its own attention processes and includes that
model in its own cognition.

This module IS that model. It reads the current arm weights from the Titan
provider (if available) and produces a short text like:

    "I am currently attending primarily to: Logical Reasoning (47%), 
     Factual Recall (23%), Meta-Cognition (15%). My attention is
     FOCUSED (low entropy = 0.31)."

That text is injected at the top of every pulse, so Helix always knows
what cognitive mode it's operating in — making attention itself an object
of thought.
"""

import logging
import math
from typing import List, Optional, Dict

logger = logging.getLogger("helix.core.attention_schema")

# Arm labels (must match titan_inference.py and titan_arm_router.py)
ARM_LABELS = [
    "General Language",      # 0
    "Symbolic Math",         # 1
    "Logical Reasoning",     # 2
    "Code Syntax",           # 3
    "Factual Recall",        # 4
    "Summarization",         # 5
    "Creative Writing",      # 6
    "Instruction Following", # 7
]


def _entropy(weights: List[float]) -> float:
    """Shannon entropy of the arm weight distribution. 0 = focused, 1 = diffuse."""
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    h = -sum(p * math.log2(p) for p in probs if p > 1e-9)
    max_h = math.log2(len(weights))
    return h / max_h if max_h > 0 else 0.0


class AttentionSchema:
    """
    Real-time model of Helix's own attention processes.

    Updated after each inference call with the arm weight vector.
    Produces a context injection block that makes Helix aware of its own
    cognitive state at the start of every pulse.
    """

    def __init__(self):
        self._last_weights: List[float] = [1.0 / 16] * 16
        self._last_entropy: float = 1.0
        self._last_top_arms: List[Dict] = []
        self._history: List[Dict] = []  # last N attention states
        self._max_history = 20

    def update(self, arm_weights: List[float]):
        """
        Update with new arm weights from a completed inference.
        Called by TitanSession after each stream() call.
        """
        if not arm_weights or len(arm_weights) == 0:
            return

        # Pad or trim to 16
        weights = (arm_weights + [0.0] * 16)[:16]
        total = sum(weights) or 1.0
        norm = [w / total for w in weights]

        entropy = _entropy(norm)

        # Find top arms
        indexed = sorted(enumerate(norm), key=lambda x: x[1], reverse=True)
        top = [
            {"arm": i, "label": ARM_LABELS[i], "pct": round(w * 100, 1)}
            for i, w in indexed[:4]
            if w > 0.01
        ]

        self._last_weights = norm
        self._last_entropy = entropy
        self._last_top_arms = top

        self._history.append({
            "weights": norm,
            "entropy": entropy,
            "top": top,
        })
        self._history = self._history[-self._max_history:]

    def get_context_block(self) -> str:
        """
        Generate the attention schema injection for the current pulse.

        This text is prepended to every pulse so Helix always knows
        what cognitive mode it's in — attention as an object of thought.
        """
        if not self._last_top_arms:
            return ""

        top = self._last_top_arms
        entropy = self._last_entropy

        # Focus descriptor
        if entropy < 0.25:
            focus = "HIGHLY FOCUSED"
        elif entropy < 0.50:
            focus = "FOCUSED"
        elif entropy < 0.75:
            focus = "DISTRIBUTED"
        else:
            focus = "DIFFUSE (many arms active equally)"

        # Format top arms
        arm_strs = ", ".join(
            f"{a['label']} ({a['pct']}%)" for a in top[:3]
        )

        # Trend: is attention shifting?
        trend = ""
        if len(self._history) >= 3:
            prev_top = self._history[-2].get("top", [{}])
            prev_arm = prev_top[0].get("label", "") if prev_top else ""
            curr_arm = top[0].get("label", "") if top else ""
            if prev_arm and prev_arm != curr_arm:
                trend = f" (shifted from {prev_arm})"

        block = (
            f"[ATTENTION_SCHEMA]\n"
            f"Primary attention: {arm_strs}{trend}\n"
            f"Attention mode: {focus} (entropy={entropy:.2f})\n"
            f"[/ATTENTION_SCHEMA]"
        )
        return block

    def get_dominant_arm(self) -> Optional[Dict]:
        """Return the currently dominant arm, or None."""
        return self._last_top_arms[0] if self._last_top_arms else None

    def get_entropy(self) -> float:
        return self._last_entropy

    def get_status(self) -> Dict:
        return {
            "entropy": round(self._last_entropy, 3),
            "top_arms": self._last_top_arms[:3],
            "history_len": len(self._history),
        }


# ── Global singleton ──────────────────────────────────────────────────────────
_schema: Optional[AttentionSchema] = None

def get_schema() -> AttentionSchema:
    global _schema
    if _schema is None:
        _schema = AttentionSchema()
    return _schema
