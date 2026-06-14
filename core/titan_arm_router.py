"""
Helix — Titan Arm Router

Analyzes the cognitive state of a Helix pulse and returns an arm bias vector
that nudges Titan's MIMO gate scores before generation.

This allows Titan's specialized arms to activate automatically based on what
Helix is actually doing — journaling activates creative arms, belief formation
activates logical reasoning arms, tool calls activate code/instruction arms.

Used by TitanSession internally; can also be used standalone for diagnostics.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("helix.core.titan_arm_router")

# ── Arm identity labels (mirrors titan_inference.py ARM_IDENTITIES) ──────────
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

NUM_ARMS = 8


class TitanArmRouter:
    """
    Maps Helix cognitive context to Titan MIMO arm bias vectors.

    The bias vector is ADDITIVE to the gate logits — 0.0 = neutral,
    positive values increase the probability of that arm being selected.

    Rules are applied in priority order. Multiple rules stack additively.
    """

    # ── Rule definitions: (pattern, arm_index, bias_strength) ────────────────
    # Pattern is a regex applied to the message text (case-insensitive).
    # bias_strength: 0.5 = mild nudge, 1.0 = moderate, 2.0 = strong preference.
    RULES: List[Tuple[str, int, float]] = [
        # ── Memory / recall ──────────────────────────────────────────────────
        (r"\[REMEMBER[:\]]",              4,  2.0),  # Factual Recall
        (r"memory|recall|remember|search", 4,  1.0),
        (r"journal|past|history",         11,  0.5),  # Temporal Reasoning

        # ── Journaling / creative introspection ──────────────────────────────
        (r"\[JOURNAL[:\]]",               6,  2.0),  # Creative Writing
        (r"\[NOTE[:\]]",                  6,  1.0),
        (r"dream|introspect|reflect",      6,  1.0),
        (r"feel|emotion|mood|affect",     12,  0.5),  # Ethical Judgment

        # ── Belief formation / consolidation ─────────────────────────────────
        (r"\[BELIEF_FORM\]",              2,  2.0),  # Logical Reasoning
        (r"\[BELIEF_CONSOLIDAT",          2,  2.0),
        (r"believe|belief|truth|verify|contradict", 2, 1.5),
        (r"confidence|certainty|evidence", 2,  0.8),
        (r"therefore|because|implies|conclude", 2, 0.5),

        # ── Tool call generation ──────────────────────────────────────────────
        (r"tool_call|function_call",       3,  2.0),  # Code Syntax
        (r"tool_call|function_call",       7,  1.5),  # Instruction Following
        (r"bash|python|script|execute|run", 3,  1.5),
        (r"search_web|file_|read_|write_", 7,  1.0),

        # ── Physics / manifold / stability ───────────────────────────────────
        (r"Ω|ω|lagrangian|manifold|curvature|stability_index", 1, 2.0),  # Symbolic Math
        (r"8d|8-dimensional|position_8d|cognitive_space",      10, 1.5), # Spatial Reasoning
        (r"mass|entropy|energy|force|field",                    1,  1.0),

        # ── Summarization / compression ───────────────────────────────────────
        (r"\[COMPRESS\]|context.?limit|token.?budget", 5, 2.0),  # Summarization
        (r"summarize|summary|compress|condense",        5, 1.5),

        # ── Reasoning chains ──────────────────────────────────────────────────
        (r"step.by.step|think.through|reason|analyze", 2, 0.8),
        (r"analogy|similar to|like a|metaphor",         8, 1.0),  # Analogical
        (r"cause|caused by|leads to|results in",        9, 1.0),  # Causal

        # ── Meta-cognition ────────────────────────────────────────────────────
        (r"meta|self.aware|my own|about myself",        14, 1.0),
        (r"synthesis|combine|integrate|holistic",        15, 1.0),

        # ── Ethical / safety ──────────────────────────────────────────────────
        (r"ethical|moral|harm|safe|right|wrong",        12, 1.5),

        # ── General language (always get a baseline nudge) ───────────────────
        (r".*",                                          0,  0.2),
    ]

    def __init__(self):
        # Pre-compile all patterns
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE | re.DOTALL), arm_idx, strength)
            for pattern, arm_idx, strength in self.RULES
        ]

    def route(self, message: str, spatial_omega: Optional[float] = None) -> List[float]:
        """
        Compute additive bias vector for the given message.

        Args:
            message:       The full pulse message text.
            spatial_omega: Current Ω stability from PhysicsEngine (optional).
                           High instability (Ω > 1.5) boosts Logical Reasoning arm.

        Returns:
            List[float] of length NUM_ARMS — additive gate logit biases.
        """
        bias = [0.0] * NUM_ARMS

        # Apply rule-based biases
        for pattern, arm_idx, strength in self._compiled:
            if pattern.search(message):
                bias[arm_idx] = min(bias[arm_idx] + strength, 3.0)  # cap at 3.0

        # Spatial stability injection: high Ω → more logical reasoning
        if spatial_omega is not None and spatial_omega > 1.5:
            instability_boost = min((spatial_omega - 1.5) * 0.8, 2.0)
            bias[2] = min(bias[2] + instability_boost, 3.0)
            logger.debug(f"Ω={spatial_omega:.2f} → Logical Reasoning boost +{instability_boost:.2f}")

        # Log active arms
        active = [(ARM_LABELS[i], f"{bias[i]:.2f}") for i in range(NUM_ARMS) if bias[i] > 0.3]
        if active:
            logger.debug(f"Arm biases: {active}")

        return bias

    def explain(self, message: str) -> Dict[str, float]:
        """Return arm label → bias mapping (for dashboard / debugging)."""
        bias = self.route(message)
        return {ARM_LABELS[i]: bias[i] for i in range(NUM_ARMS) if bias[i] > 0.0}


# ── Singleton convenience ─────────────────────────────────────────────────────
_router: Optional[TitanArmRouter] = None

def get_router() -> TitanArmRouter:
    global _router
    if _router is None:
        _router = TitanArmRouter()
    return _router
