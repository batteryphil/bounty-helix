"""
Helix — Self Model

Helix's continuously updated internal model of itself.
Updated after every interaction, training cycle, and self-reflection.

The self-model tracks:
  - Architectural facts (arms, layers, parameters, context window)
  - Capability map (what it's good at, what it struggles with)
  - Limitation inventory (gaps in knowledge, things it got wrong)
  - Evolution log (what changed and when)
  - Offspring design (accumulated improvements for the next version)

This is NOT a static config file. It's a living document that Helix
reads, reflects on, and updates autonomously.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger("helix.brain.self_model")


class SelfModel:
    """
    Helix's internal model of its own nature, capabilities, and limitations.

    Helix reads this every N pulses to answer questions like:
      "What am I?" / "What can I do?" / "What should I improve?"

    It writes to it when it discovers something new about itself.
    """

    TEMPLATE = {
        "identity": {
            "name": "Helix",
            "version": "1.0",
            "architecture": "Mamba3 MIMO — 2.7B parameters, 8 active MIMO arms, 16 arm slots",
            "backbone": "Mamba3 SSM (selective state space model) — no attention mechanism",
            "arms": {
                "0": "General Language",
                "1": "Symbolic Math",
                "2": "Logical Reasoning",
                "3": "Code Syntax",
                "4": "Factual Recall",
                "5": "Summarization",
                "6": "Creative Writing",
                "7": "Instruction Following",
            },
            "training_phase": "Phase 1 — arm calibration",
            "context_window_tokens": 512,
            "runs_on": "RTX 3060 12GB — local, fully offline",
        },
        "capabilities": {
            "confirmed_strong": [],
            "confirmed_weak": [],
            "untested": [
                "multi-step mathematical reasoning",
                "long-form creative writing",
                "code generation and debugging",
                "factual question answering",
                "logical deduction chains",
            ],
        },
        "limitations": {
            "known": [
                "Context window is currently 512 tokens — long documents must be chunked",
                "Phase 1 training not complete — arms still calibrating to backbone",
                "No persistent memory across inference sessions (handled by Helix journal)",
            ],
            "suspected": [],
            "discovered_via_errors": [],
        },
        "self_reflections": [],  # timestamped notes Helix writes about itself
        "questions_about_self": [],  # open questions Helix hasn't answered yet
        "offspring_design": {
            "description": "Accumulated improvements for Helix v2.0",
            "target_params": "7B",
            "proposed_changes": [],
            "blocking_requirements": [],
        },
        "evolution_log": [],  # timestamped record of all significant changes
        "last_updated": "",
        "update_count": 0,
    }

    def __init__(self, data_dir: Path = Path("data")):
        self.path = data_dir / "self_model.json"
        self.data_dir = data_dir
        self._model: Dict = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._model = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info(f"Self-model loaded (update #{self._model.get('update_count', 0)})")
                return
            except Exception as e:
                logger.warning(f"Self-model load failed: {e} — using template")
        # First boot: initialize from template
        import copy
        self._model = copy.deepcopy(self.TEMPLATE)
        self._model["evolution_log"].append({
            "timestamp": self._now(),
            "event": "Self-model initialized",
            "note": "First boot — baseline template created",
        })
        self._save()

    def _save(self):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._model["last_updated"] = self._now()
            self._model["update_count"] = self._model.get("update_count", 0) + 1
            self.path.write_text(
                json.dumps(self._model, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Self-model save failed: {e}")

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ── Read interface ────────────────────────────────────────────────────────

    def get_summary(self, max_chars: int = 1200) -> str:
        """Return a concise self-summary for injection into Titan's context."""
        m = self._model
        ident = m.get("identity", {})
        caps = m.get("capabilities", {})
        lims = m.get("limitations", {})
        offspring = m.get("offspring_design", {})

        strong = caps.get("confirmed_strong", [])
        weak = caps.get("confirmed_weak", [])
        known_lims = lims.get("known", [])
        proposed = offspring.get("proposed_changes", [])[:3]

        lines = [
            f"[SELF_MODEL v{m.get('identity', {}).get('version', '1.0')} "
            f"update #{m.get('update_count', 0)}]",
            f"I am {ident.get('architecture', 'Titan MIMO')}.",
            f"Training phase: {ident.get('training_phase', 'unknown')}.",
        ]
        if strong:
            lines.append(f"Confirmed strengths: {', '.join(strong[:4])}")
        if weak:
            lines.append(f"Known weaknesses: {', '.join(weak[:4])}")
        if known_lims:
            lines.append(f"Limitations: {'; '.join(known_lims[:2])}")
        if proposed:
            lines.append(f"Offspring v2 proposals: {'; '.join(proposed[:2])}")

        open_qs = m.get("questions_about_self", [])
        if open_qs:
            lines.append(f"Open self-questions: {open_qs[-1]}")

        result = "\n".join(lines)
        return result[:max_chars]

    def get_full(self) -> Dict:
        return self._model

    # ── Write interface ───────────────────────────────────────────────────────

    def record_capability(self, capability: str, strength: str = "strong"):
        """Record a confirmed capability ('strong' or 'weak')."""
        key = "confirmed_strong" if strength == "strong" else "confirmed_weak"
        caps = self._model.setdefault("capabilities", {})
        lst = caps.setdefault(key, [])
        if capability not in lst:
            lst.append(capability)
            self._log_evolution(f"Capability {strength}: {capability}")
            self._save()

    def record_limitation(self, limitation: str, source: str = "self_observation"):
        """Record a discovered limitation."""
        lims = self._model.setdefault("limitations", {})
        if source == "error":
            lst = lims.setdefault("discovered_via_errors", [])
        else:
            lst = lims.setdefault("suspected", [])
        if limitation not in lst:
            lst.append(limitation)
            self._log_evolution(f"Limitation found: {limitation}")
            self._save()

    def add_self_reflection(self, reflection: str):
        """Store a self-reflection note."""
        refs = self._model.setdefault("self_reflections", [])
        refs.append({"timestamp": self._now(), "text": reflection})
        refs[:] = refs[-100:]  # keep last 100
        self._save()

    def add_self_question(self, question: str):
        """Add an open question Helix has about itself."""
        qs = self._model.setdefault("questions_about_self", [])
        if question not in qs:
            qs.append(question)
            qs[:] = qs[-50:]
            self._save()

    def propose_offspring_improvement(self, improvement: str, rationale: str = ""):
        """Record a design improvement for the next-generation model."""
        design = self._model.setdefault("offspring_design", {})
        proposals = design.setdefault("proposed_changes", [])
        entry = {"improvement": improvement, "rationale": rationale, "timestamp": self._now()}
        if improvement not in [p.get("improvement") for p in proposals]:
            proposals.append(entry)
            self._log_evolution(f"Offspring proposal: {improvement}")
            self._save()
            logger.info(f"[OFFSPRING] New design proposal: {improvement}")

    def update_training_phase(self, phase: str, step: int):
        """Update the current training phase info."""
        ident = self._model.setdefault("identity", {})
        ident["training_phase"] = f"Phase {phase} — step {step:,}"
        self._save()

    def _log_evolution(self, event: str, note: str = ""):
        log = self._model.setdefault("evolution_log", [])
        log.append({"timestamp": self._now(), "event": event, "note": note})
        log[:] = log[-200:]  # keep last 200

    # ── Offspring generation ──────────────────────────────────────────────────

    def generate_offspring_spec(self) -> Dict:
        """
        Generate a training specification for the next-generation model.
        Returns a dict that helix_overnight_trainer.py can consume.
        """
        design = self._model.get("offspring_design", {})
        proposals = design.get("proposed_changes", [])
        weak_caps = self._model.get("capabilities", {}).get("confirmed_weak", [])
        lims = self._model.get("limitations", {}).get("known", [])

        spec = {
            "version": f"helix_v{float(self._model.get('identity', {}).get('version', '1.0')) + 1:.1f}",
            "generated_at": self._now(),
            "base_checkpoint": "checkpoints_2.7b/phase_1.pt",
            "target_params": design.get("target_params", "7B"),
            "priority_improvements": proposals[:5],
            "focus_domains": weak_caps[:5],
            "known_limitations_to_address": lims[:3],
            "recommended_phases": [
                "Phase 1: arm recalibration on expanded dataset",
                "Phase 2: domain specialization per weak capability",
                "Phase 3j: improved router training",
                "SFT: targeted fine-tuning on discovered failure modes",
            ],
        }
        logger.info(f"[OFFSPRING] Spec generated for {spec['version']}")
        return spec
