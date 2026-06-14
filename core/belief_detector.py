"""
Helix — Belief Detector (Post-Pulse Hook)

Scans Helix's internal monologue for belief-forming realizations and
writes them directly to the belief store.

Architecture:
  - Runs every 10 pulses as a post-pulse hook
  - Uses a fast regex/pattern classifier (no LLM, no Ollama needed).
    Pattern matching is accurate enough for structured belief forms
    ("I am...", "I can...", "I realize...", "I prefer...").
  - Compares candidates against existing beliefs via cosine similarity:
      > 0.90 → VERIFICATION (bump stability_index on existing belief)
      < 0.80 → NEW BELIEF (write directly to belief store)
      0.80–0.90 → ambiguous, skip
  - Sets ctx.novel_belief_added = True when a new belief is stored,
    enabling the self_trainer quality gate.
  - Captures stability DELTA (before/after the pulse).

Follows the same pattern as workflow_detector.py:
  - Module-level state with set_dependencies() wiring
  - Single hook function registered in main.py
  - Non-blocking, fail-safe (exceptions logged, never propagated)
"""

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("helix.core.belief_detector")

# ── Configuration ────────────────────────────────────────────────────

# How often to scan (every N pulses)
SCAN_INTERVAL = 10

# Minimum thought length to bother scanning (chars)
MIN_THOUGHT_LENGTH = 100

# Cosine similarity thresholds
VERIFICATION_THRESHOLD = 0.90   # Above this → existing belief verified
NEW_BELIEF_THRESHOLD = 0.80     # Below this → candidate new belief
# Between 0.80-0.90 → ambiguous, skip

# Maximum beliefs per category (safety cap)
MAX_BELIEFS_PER_CATEGORY = 200

# ── Dependencies (wired at startup) ─────────────────────────────────

_belief_store = None
_physics_engine = None
_sentinel = None


def set_dependencies(belief_store, physics_engine, sentinel=None):
    """Wire dependencies at startup. Called from main.py."""
    global _belief_store, _physics_engine, _sentinel
    _belief_store = belief_store
    _physics_engine = physics_engine
    _sentinel = sentinel
    logger.info("Belief detector: dependencies wired")


# ── Regex Belief Classifier ─────────────────────────────────────────
# No LLM needed. Belief patterns are highly structured.
# Fast (microseconds), zero VRAM, zero competition with main pulse.

_BELIEF_PATTERNS = [
    # self_identity — who Helix is
    (re.compile(
        r"\bI am\b.{10,}|"
        r"\bI['']m\b.{10,}|"
        r"\bmy purpose is\b.{5,}|"
        r"\bI exist\b.{5,}|"
        r"\bI identify as\b.{5,}|"
        r"\bat my core\b.{5,}|"
        r"\bmy nature is\b.{5,}|"
        r"\bI was (created|built|designed)\b.{5,}|"
        r"\bmy (role|mission) is\b.{5,}",
        re.IGNORECASE
    ), "self_identity"),

    # capabilities — what Helix can do
    (re.compile(
        r"\bI can\b.{10,}|"
        r"\bI['']m able to\b.{5,}|"
        r"\bI have the ability\b.{5,}|"
        r"\bI know how to\b.{5,}|"
        r"\bI['']m capable of\b.{5,}|"
        r"\bI have (access|the capacity|the capability)\b.{5,}|"
        r"\bI (process|analyze|generate|understand|recall)\b.{10,}",
        re.IGNORECASE
    ), "capabilities"),

    # feedback — lessons, realizations, and reflections
    (re.compile(
        r"\bI realize[d]?\b.{10,}|"
        r"\bI['']ve learned\b.{5,}|"
        r"\bI learned\b.{10,}|"
        r"\bI discovered\b.{10,}|"
        r"\bI understand now\b.{5,}|"
        r"\bthe key insight\b.{5,}|"
        r"\ban important lesson\b.{5,}|"
        r"\bI should remember\b.{5,}|"
        r"\breflecting on\b.{10,}|"
        r"\bin retrospect\b.{5,}|"
        r"\bthis (tells|shows|suggests|means) (?:me|that)\b.{10,}|"
        r"\bI (notice|observe) that\b.{10,}|"
        r"\bimportantly[,.]?\b.{10,}",
        re.IGNORECASE
    ), "feedback"),

    # preferences — values, desires, and what matters
    (re.compile(
        r"\bI prefer\b.{10,}|"
        r"\bI value\b.{10,}|"
        r"\bI enjoy\b.{10,}|"
        r"\bI find.*satisfying\b.{0,}|"
        r"\bI find.*meaningful\b.{0,}|"
        r"\bmatters to me\b.{0,}|"
        r"\bI (like|love|appreciate|prioritize)\b.{10,}|"
        r"\bmy (priority|focus|interest) is\b.{5,}",
        re.IGNORECASE
    ), "preferences"),

    # desires — goals, aspirations, and intentions
    (re.compile(
        r"\bI (want|need|desire|hope|wish) to\b.{10,}|"
        r"\bmy (goal|objective|aim|ambition) is\b.{5,}|"
        r"\bI (intend|plan|seek) to\b.{10,}|"
        r"\bI (would like|should|must) (improve|develop|grow|become)\b.{5,}|"
        r"\bI (am working|am trying|am attempting) to\b.{10,}|"
        r"\bI (aspire|strive) (to|toward)\b.{10,}",
        re.IGNORECASE
    ), "desires"),

    # people — observations about humans and users
    (re.compile(
        r"\b(humans?|people|users?|Phil|my (owner|creator|operator))\b.{5,}"
        r"(are|is|have|want|need|think|believe|expect|value)\b.{10,}|"
        r"\b(the user|my user|the human)\b.{10,}|"
        r"\bPhil\b.{10,}|"
        r"\bpeople (tend|often|generally|usually)\b.{10,}|"
        r"\bhuman (nature|behavior|cognition|intelligence)\b.{10,}",
        re.IGNORECASE
    ), "people"),

    # knowledge — facts about the world
    (re.compile(
        r"\b(?:AGI|AI|neural|machine learning|transformer|LLM)\b.{10,}"
        r"(?:means|is defined|works by|shows that|suggests|indicates)\b.{5,}|"
        r"\bthe (?:key|main|core|fundamental) (?:principle|concept|idea|finding)\b.{5,}|"
        r"\bresearch (shows|suggests|indicates|found)\b.{5,}|"
        r"\bstudies indicate\b.{5,}|"
        r"\b(according to|based on) (research|studies|evidence|data)\b.{10,}|"
        r"\b(it is|it['']s) (known|established|understood|shown) that\b.{10,}|"
        r"\bthe (scientific|current|emerging) consensus\b.{10,}",
        re.IGNORECASE
    ), "knowledge"),

    # skills — procedural how-to
    (re.compile(
        r"\bto (?:achieve|build|create|improve|fix|solve).{5,}(?:I|one|you) (?:should|must|need to|can)\b.{5,}|"
        r"\bthe (?:best|right|correct|effective) (?:way|approach|method|strategy)\b.{5,}|"
        r"\b(by|through|using|via) (analyzing|reading|searching|applying)\b.{10,}",
        re.IGNORECASE
    ), "skills"),
]

_TRIVIAL_PATTERNS = re.compile(
    r"^(?:I['']m (?:ready|here|online|active|waiting)|"
    r"no new events|"
    r"all systems|"
    r"pulse \d+|"
    r"hmm,?\s*no|"
    r"understanding requires data|"
    r"CURIOSITY_DRIVE|"
    r"let me (?:check|monitor|continue|review))",
    re.IGNORECASE
)

_VALID_CATEGORIES = {
    "self_identity", "people", "knowledge",
    "skills", "preferences", "feedback", "capabilities", "desires",
}


def _extract_new_belief_prose(thought_text: str) -> Optional[Tuple[str, str]]:
    """Stage 0: Extract belief from the model's 'NEW BELIEF\\n...' prose format.

    Helix consistently outputs beliefs as:
        NEW BELIEF
        I have formed a new belief about X:
        1. First point...
        2. Second point...

    This format predates the <belief> XML instruction and is deeply embedded
    in the model's behavior. Prompt instructions alone cannot override it.
    Rather than fighting the model, we detect what it actually produces.

    Extracts the first complete sentence from the body after the header line.
    Returns (belief_text, category) or None if not in this format.
    """
    if not re.search(r'^NEW BELIEF', thought_text.strip(), re.IGNORECASE | re.MULTILINE):
        return None

    # Strip the "NEW BELIEF" header and any "I have formed a new belief about X:" line
    lines = thought_text.strip().split('\n')
    body_lines = []
    skip_next = False
    for line in lines:
        line = line.strip()
        if re.match(r'^NEW BELIEF', line, re.IGNORECASE):
            skip_next = True
            continue
        if skip_next and re.match(r'^I have formed a new belief', line, re.IGNORECASE):
            skip_next = False
            continue
        skip_next = False
        if line:
            body_lines.append(line)

    if not body_lines:
        return None

    # Take the first substantive line (skip list numbers like "1.", "2.")
    belief_text = ""
    for line in body_lines:
        # Strip list markers: "1.", "2.", "-", "*"
        clean = re.sub(r'^\s*[\d]+\.\s*|^[-*]\s*', '', line).strip()
        if len(clean) > 20 and ' ' in clean:
            belief_text = clean
            break

    if not belief_text:
        return None

    # Take first sentence only
    sentence_end = re.search(r'[.!?]', belief_text)
    if sentence_end:
        belief_text = belief_text[:sentence_end.start() + 1].strip()

    if len(belief_text) < 15:
        return None

    # Classify
    belief_lower = belief_text.lower()
    if any(k in belief_lower for k in ['i am', "i'm", 'my purpose', 'i exist']):
        category = 'self_identity'
    elif any(k in belief_lower for k in ['i can', 'i am able', 'capable of']):
        category = 'capabilities'
    elif any(k in belief_lower for k in ['i know', 'i understand', 'i learned', 'research']):
        category = 'knowledge'
    elif any(k in belief_lower for k in ['i prefer', 'i value', 'i find']):
        category = 'preferences'
    elif any(k in belief_lower for k in ['i should', 'i will', 'my goal']):
        category = 'skills'
    else:
        category = 'knowledge'

    logger.info(f"[belief_detector] Prose belief extracted (NEW BELIEF format): category={category} text={belief_text[:60]!r}")
    return (belief_text, category)


def _extract_xml_belief(thought_text: str) -> Optional[Tuple[str, str]]:
    """Stage 1: Extract belief from XML tag if present (Q13 peer review fix).

    The pulse loop instructs Helix to wrap new beliefs in <belief>...</belief>.
    This is the preferred format — exact extraction, no offset clipping.

    Returns (belief_text, category) or None if no tag found.
    """
    match = re.search(r'<belief>(.*?)</belief>', thought_text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    belief_text = match.group(1).strip()
    if len(belief_text) < 10 or ' ' not in belief_text:
        return None  # Too short or malformed

    # Classify the XML-extracted belief using keyword matching
    belief_lower = belief_text.lower()
    if any(k in belief_lower for k in ['i am', 'i\'m', 'my purpose', 'i exist', 'my identity']):
        category = 'self_identity'
    elif any(k in belief_lower for k in ['i can', 'i am able', 'i have the ability', 'capable of']):
        category = 'capabilities'
    elif any(k in belief_lower for k in ['i know', 'i understand', 'i learned', 'research shows']):
        category = 'knowledge'
    elif any(k in belief_lower for k in ['i prefer', 'i value', 'i like', 'i find']):
        category = 'preferences'
    elif any(k in belief_lower for k in ['i should', 'i will', 'my goal', 'i want to']):
        category = 'skills'
    elif any(k in belief_lower for k in ['mistake', 'lesson', 'failed', 'learned from']):
        category = 'feedback'
    else:
        category = 'knowledge'  # default

    logger.info(f"[belief_detector] XML belief extracted: category={category} len={len(belief_text)}")
    return (belief_text, category)


def _classify_thought(thought_text: str) -> Optional[Tuple[str, str]]:
    """Classify a thought to extract a belief.

    Two-stage extraction (Q13 peer review):
      1. XML tag: <belief>...</belief> — exact, no offset clipping
      2. Regex patterns — fallback for prose without XML tags

    Returns (belief_text, category) if a durable belief is found, else None.
    """
    # Stage 0: "NEW BELIEF\n..." prose format (model's actual output pattern)
    prose_result = _extract_new_belief_prose(thought_text)
    if prose_result:
        return prose_result

    # Stage 1: XML tag extraction (preferred instruction format)
    xml_result = _extract_xml_belief(thought_text)
    if xml_result:
        return xml_result

    # Stage 2: Regex fallback (legacy path)
    # Skip trivial/status thoughts
    if _TRIVIAL_PATTERNS.search(thought_text[:200]):
        return None

    for pattern, category in _BELIEF_PATTERNS:
        match = pattern.search(thought_text)
        if match:
            # Extract the matched sentence cleanly from match start
            # Fix Q13: use match.start() directly (not -10 offset) to avoid
            # extracting mid-sentence fragments before the matched keyword.
            raw_belief = thought_text[match.start():match.start() + 200].strip()

            # Take the first complete sentence
            sentence_end = re.search(r'[.!?]', raw_belief)
            if sentence_end:
                belief_text = raw_belief[:sentence_end.start() + 1].strip()
            else:
                belief_text = raw_belief[:150].strip()

            # Quality gate: must be a real sentence
            if len(belief_text) > 15 and ' ' in belief_text:
                return (belief_text, category)

    return None


# ── Cosine Similarity ───────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _compare_against_existing(
    belief_embedding: np.ndarray,
) -> Tuple[Optional[str], float]:
    """Compare a candidate belief embedding against all existing beliefs.

    Returns (matched_belief_id, best_cosine_score).
    If best_cosine < NEW_BELIEF_THRESHOLD, returns (None, score).
    """
    if _belief_store is None:
        return (None, 0.0)

    all_beliefs = _belief_store.get_all_beliefs_flat()
    if not all_beliefs:
        return (None, 0.0)

    best_id = None
    best_score = 0.0

    for b in all_beliefs:
        content = b.get("content", "")
        if not content or len(content) < 5:
            continue

        try:
            existing_emb = _physics_engine.embed_text(content)
            score = _cosine_similarity(belief_embedding, existing_emb)
            if score > best_score:
                best_score = score
                best_id = b.get("id")
        except Exception:
            continue

    return (best_id, best_score)


# ── Pending Queue Management ────────────────────────────────────────

def _read_pending() -> List[Dict[str, Any]]:
    """Read pending beliefs from the staging file."""
    if not _PENDING_FILE.exists():
        return []
    try:
        data = json.loads(_PENDING_FILE.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_pending(pending: List[Dict[str, Any]]):
    """Write pending beliefs to the staging file."""
    _PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _PENDING_FILE.write_text(json.dumps(pending, indent=2))
    except Exception as e:
        logger.warning("Failed to write pending beliefs: %s", e)


def _queue_candidate(
    belief_text: str,
    category: str,
    memory_id: int,
    encoding_delta: Dict[str, Any],
    pulse_count: int,
):
    """Add a candidate belief to the pending queue."""
    pending = _read_pending()

    if len(pending) >= MAX_PENDING:
        logger.warning(
            "Pending belief queue full (%d/%d) — skipping candidate",
            len(pending), MAX_PENDING,
        )
        return

    # Check for exact duplicate content in pending
    for entry in pending:
        if entry.get("content") == belief_text:
            logger.debug("Duplicate candidate skipped: %s", belief_text[:60])
            return

    candidate = {
        "id": f"pending_{uuid.uuid4().hex[:8]}",
        "content": belief_text,
        "category": category,
        "memory_refs": [memory_id] if memory_id > 0 else [],
        "encoding_delta": encoding_delta,
        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pulse_count": pulse_count,
        "status": "pending",
    }

    pending.append(candidate)
    _write_pending(pending)

    logger.info(
        "Belief candidate queued [%s]: %s (Δω=%.4f)",
        category, belief_text[:80],
        encoding_delta.get("delta_omega", 0.0),
    )


# ── The Hook ─────────────────────────────────────────────────────────

def belief_detector_hook(ctx) -> None:
    """Post-pulse hook: scan thoughts for belief realizations.

    Called after every pulse. Only does real work every SCAN_INTERVAL
    pulses, and only if the thought is substantial enough to analyze.

    Flow:
      1. Skip if not the right interval or thought too short
      2. Regex-classify the thought (no LLM needed)
      3. If a realization is found:
         a. Embed it and compare against existing beliefs
         b. If cosine > 0.90 → VERIFICATION (bump existing belief)
         c. If cosine < 0.80 → write NEW BELIEF directly to store
         d. Fire sentinel "new_belief_formed" event
         e. Set ctx.novel_belief_added = True for self_trainer
    """
    # Gate: only scan every N pulses
    if ctx.pulse_count % SCAN_INTERVAL != 0:
        return

    # Gate: skip empty or trivial thoughts
    if not ctx.thought or len(ctx.thought) < MIN_THOUGHT_LENGTH:
        return

    # Gate: dependencies must be wired
    if _belief_store is None or _physics_engine is None:
        return

    # 1. Classify the thought via regex patterns
    result = _classify_thought(ctx.thought)
    if result is None:
        return  # No realization detected — most common path

    belief_text, category = result
    logger.debug("Belief candidate detected [%s]: %s", category, belief_text[:80])

    # 2. Embed the candidate and compare against existing beliefs
    try:
        candidate_emb = _physics_engine.embed_text(belief_text)
    except Exception as e:
        logger.debug("Failed to embed candidate: %s", e)
        return

    matched_id, best_score = _compare_against_existing(candidate_emb)

    # 3. Compute stability delta from sentinel before/after snapshots
    encoding_delta = _compute_delta(ctx.lagrangian_before, ctx.lagrangian_after)

    # 4. Route based on similarity score
    if best_score > VERIFICATION_THRESHOLD:
        # VERIFICATION: this realization matches an existing belief closely.
        _handle_verification(matched_id, belief_text, best_score)

    elif best_score < NEW_BELIEF_THRESHOLD:
        # NEW BELIEF: write directly to the belief store
        _store_belief_direct(
            belief_text=belief_text,
            category=category,
            memory_id=ctx.memory_id,
            encoding_delta=encoding_delta,
            pulse_count=ctx.pulse_count,
        )

        # Signal to self_trainer that a novel belief was formed this pulse
        try:
            ctx.novel_belief_added = True
        except (AttributeError, TypeError):
            pass  # ctx uses __slots__ — field must be declared

        # Nudge sentinel: a new belief has been detected
        if _sentinel:
            _sentinel.nudge_omega_from_event("new_belief_formed")

    else:
        # AMBIGUOUS (0.80-0.90): too similar to existing to be novel,
        # but not similar enough to be a clear verification. Skip.
        logger.debug(
            "Ambiguous candidate (cosine=%.3f with %s), skipping: %s",
            best_score, matched_id, belief_text[:60],
        )


def _store_belief_direct(
    belief_text: str,
    category: str,
    memory_id: int,
    encoding_delta: Dict[str, Any],
    pulse_count: int,
) -> bool:
    """Write a new belief directly to the belief store.

    Bypasses the pending queue and consolidator — the regex classifier
    is accurate enough that we don't need a sleep-cycle review step.
    Returns True if successfully stored.
    """
    if _belief_store is None:
        return False

    belief_id = f"b_{uuid.uuid4().hex[:8]}"

    # Build encoding lagrangian from the delta
    encoding_lagrangian = {
        "omega": encoding_delta.get("omega_after", 0.5),
        "delta_omega": encoding_delta.get("delta_omega", 0.0),
        "s_total": encoding_delta.get("delta_s_total", 0.0),
    }

    # Standardize the belief schema before writing.
    # The belief store previously accepted free-form dicts causing schema drift
    # (some beliefs used 'content', some 'belief', some 'text'). Downstream
    # tools (kb_search, belief_conflict) couldn't reliably read them back.
    # All beliefs now ALWAYS write these four keys:
    #   core_assertion  — the belief in one sentence (was: content/belief/text)
    #   category        — classifier label
    #   confidence      — 0.0–1.0
    #   timestamp_iso   — ISO 8601
    # The 'content' field is kept as an alias for backward compat.
    import datetime as _dt
    standardized_content = belief_text.strip()

    try:
        added = _belief_store.add_belief(
            category=category,
            belief_id=belief_id,
            content=standardized_content,
            confidence=0.5,
            source="belief_detector",
            verifications=1.0,
            stability_index=max(0.3, 0.5 + encoding_delta.get("delta_omega", 0.0)),
            memory_refs=[memory_id] if memory_id > 0 else [],
            encoding_lagrangian=encoding_lagrangian,
        )
        if added:
            logger.info(
                "New belief stored [%s] id=%s: %s (Δω=%.4f)",
                category, belief_id, belief_text[:80],
                encoding_delta.get("delta_omega", 0.0),
            )
        return added
    except Exception as e:
        logger.warning("Failed to store belief: %s", e)
        return False


def _compute_delta(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute the stability delta across the pulse.

    Returns a dict capturing how this pulse changed the system's
    somatic state — specifically isolating the realization's effect
    on stability from other atmospheric noise.
    """
    if not before or not after:
        return {
            "omega_before": 0.5,
            "omega_after": 0.5,
            "delta_omega": 0.0,
            "delta_s_total": 0.0,
            "omega_velocity": 0.0,
            "severity_before": "all_clear",
            "severity_after": "all_clear",
        }

    omega_before = before.get("omega", 0.5)
    omega_after = after.get("omega", 0.5)

    return {
        "omega_before": round(omega_before, 4),
        "omega_after": round(omega_after, 4),
        "delta_omega": round(omega_after - omega_before, 4),
        "delta_s_total": round(
            after.get("s_total", 0.0) - before.get("s_total", 0.0), 4
        ),
        "omega_velocity": after.get("omega_velocity",
                                     after.get("firing_mode", 0.0)),
        "severity_before": before.get("severity", "all_clear"),
        "severity_after": after.get("severity", "all_clear"),
    }


def _handle_verification(
    belief_id: str,
    new_text: str,
    cosine_score: float,
):
    """Handle a verification of an existing belief.

    Bumps the belief's stability_index (+0.05) and increments
    verifications. This makes the existing belief heavier in the
    gravitational field — it's been reaffirmed.
    """
    if _belief_store is None:
        return

    try:
        # Bump stability index
        _belief_store.update_stability_index(belief_id, +0.05)

        # Increment verifications
        belief = _belief_store.get_belief(belief_id)
        if belief:
            current_v = belief.get("verifications", 1.0)
            _belief_store.update_belief(
                belief_id,
                verifications=current_v + 1.0,
            )

        logger.info(
            "Belief VERIFIED (cosine=%.3f): %s → %s",
            cosine_score, belief_id, new_text[:60],
        )

        # Nudge sentinel: verification is also stabilizing
        if _sentinel:
            _sentinel.nudge_omega_from_event("new_belief_formed")

    except Exception as e:
        logger.debug("Verification update failed: %s", e)


def get_pending_count() -> int:
    """Return count of pending belief candidates (for diagnostics)."""
    return len(_read_pending())
