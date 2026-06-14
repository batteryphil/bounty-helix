"""
Helix — Recursive Monologue (Strange Loop Engine)

Every N pulses, this module forces Arm 14 (Meta-Cognition) to generate
a private internal observation about what Helix is currently doing and why.

This output is NOT sent to the user. It is injected back into the next
pulse's context as private thought — creating the Strange Loop:

  Helix thinks → watches itself think → that observation modifies next thought
  → which it watches again → ad infinitum

This is Hofstadter's "tangled hierarchy" made concrete: the system's model
of itself becomes part of its own cognitive input.

The monologue runs as a post-pulse hook — fires after every Nth pulse,
reads the last thought, writes a meta-observation, feeds it back.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("helix.core.recursive_monologue")


# The prompt template for self-observation.
# Deliberately minimal — we want authentic observation, not performance.
_META_PROMPT = """\
[PRIVATE_MONOLOGUE — NOT FOR USER]
Without performing for anyone, observe what you are currently doing internally.

Complete these sentences honestly:
"Right now I am ___."
"This is happening because ___."
"I notice in my own processing that ___."
"A pattern I observe in myself is ___."

Do not explain. Do not elaborate. Just observe. Be brief and precise.
[END_META_PROMPT]"""


class RecursiveMonologue:
    """
    Generates private self-observations that feed back into Helix's own context.

    This implements the Strange Loop: Helix watches itself think, and that
    observation becomes part of what it thinks next.

    Wired into the pulse loop as a post-pulse hook. Every N pulses it:
      1. Generates a private meta-observation using the LLM session directly
      2. Stores the observation in a rolling monologue buffer
      3. Injects the most recent observations into the NEXT pulse's preconscious

    The key: observations are injected as [PRIVATE_THOUGHT] blocks, not as
    user messages. Helix knows they are its own internal voice.
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        pulse_interval: int = 10,        # how often to self-observe (every N pulses)
        max_buffer: int = 5,             # how many past observations to keep in context
        max_chars_per_obs: int = 300,    # cap each observation (keep context tight)
    ):
        self.data_dir = data_dir
        self.pulse_interval = pulse_interval
        self.max_buffer = max_buffer
        self.max_chars_per_obs = max_chars_per_obs

        self._pulse_count: int = 0
        self._observations: list = []  # rolling buffer of recent self-observations
        self._obs_path = data_dir / "recursive_monologue.json"

        # Reference to the session for direct generation
        # (injected after pulse loop construction)
        self._session = None
        self._pulse_loop = None

        self._lock = threading.Lock()
        self._load()

        logger.info(
            f"RecursiveMonologue initialized — self-observation every {pulse_interval} pulses"
        )

    def _load(self):
        import json
        if self._obs_path.exists():
            try:
                data = json.loads(self._obs_path.read_text())
                self._observations = data.get("observations", [])[-self.max_buffer:]
                self._pulse_count = data.get("total_pulses", 0)
            except Exception:
                pass

    def _save(self):
        import json
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._obs_path.write_text(json.dumps({
                "observations": self._observations[-50:],  # archive last 50
                "total_pulses": self._pulse_count,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, indent=2))
        except Exception as e:
            logger.warning(f"Monologue save failed: {e}")

    def set_pulse_loop(self, pulse_loop):
        """Wire up pulse loop reference for session access."""
        self._pulse_loop = pulse_loop

    def on_pulse(self, thought_text: str = "") -> Optional[str]:
        """
        Called after each pulse completes.
        Returns a private observation string if one was generated, else None.
        """
        with self._lock:
            self._pulse_count += 1
            if self._pulse_count % self.pulse_interval != 0:
                return None

        logger.debug(f"[MONOLOGUE] Pulse {self._pulse_count} — generating self-observation")

        # Generate the self-observation
        observation = self._generate_observation(thought_text)
        if not observation:
            return None

        timestamp = time.strftime("%H:%M", time.gmtime())
        entry = {
            "timestamp": timestamp,
            "pulse": self._pulse_count,
            "text": observation[:self.max_chars_per_obs],
        }

        with self._lock:
            self._observations.append(entry)
            self._observations = self._observations[-self.max_buffer:]

        self._save()
        logger.info(f"[STRANGE_LOOP] Self-observation: {observation[:80]}...")

        # Emit as a private event into the pulse loop
        if self._pulse_loop:
            self._pulse_loop.emit("private_thought", {
                "content": f"[PRIVATE_THOUGHT at {timestamp}]\n{observation}",
                "source": "recursive_monologue",
            })

        return observation

    def _generate_observation(self, recent_thought: str) -> Optional[str]:
        """Use the LLM session directly to generate a self-observation."""
        if self._pulse_loop is None:
            return None
        try:
            session = getattr(self._pulse_loop, "_session", None)
            if session is None:
                return None

            # Build a minimal context: recent thought + meta-prompt
            context = ""
            if recent_thought:
                context = f"My most recent thought was:\n{recent_thought[:500]}\n\n"

            prompt = context + _META_PROMPT
            response = session.send_message(prompt)
            return response.strip()
        except Exception as e:
            logger.debug(f"Monologue generation failed: {e}")
            return None

    def get_context_block(self) -> str:
        """
        Return recent observations formatted for context injection.
        Called by the preconscious before each pulse.
        """
        with self._lock:
            if not self._observations:
                return ""
            lines = []
            for obs in self._observations[-3:]:  # last 3 observations in context
                lines.append(f"[Private thought @ {obs['timestamp']}] {obs['text']}")
            return "[RECURSIVE_SELF_OBSERVATION]\n" + "\n".join(lines) + "\n[/RECURSIVE_SELF_OBSERVATION]"

    def get_status(self) -> dict:
        return {
            "total_pulses": self._pulse_count,
            "observations_stored": len(self._observations),
            "next_observation_in": self.pulse_interval - (self._pulse_count % self.pulse_interval),
        }
