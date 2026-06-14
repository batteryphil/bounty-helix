"""
Helix — Self-Improvement Engine

The core of autonomous self-evolution. Runs as a background thread every
10 minutes when the agent is idle. Executes a 4-phase loop:

  Phase 1 — Gap Identification
    Reads the metacognitive snapshot and evolution journal.
    Uses the Hermes model to reason about what single improvement would
    have the most impact. Outputs a structured JSON proposal.

  Phase 2 — Constitutional Validation
    Checks the proposal against constitutional hard constraints.
    Rejects unsafe or redundant proposals before any code is written.

  Phase 3 — Implementation
    Calls write_code() to implement the improvement.
    Calls run_python() to test it.
    Calls reload_tool() to activate it immediately.

  Phase 4 — Evaluation
    Waits 5 minutes for the change to accumulate real performance data.
    Calls the FitnessEvaluator to compare before/after scores.
    Commits if improved or neutral. Reverts if degraded.
    Records everything in the EvolutionJournal.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("helix.core.self_improvement")

IMPROVEMENT_INTERVAL = 600    # 10 min between cycles
EVAL_WAIT_SECONDS    = 300    # 5 min after implementation before eval
IDLE_REQUIRED        = 120    # 2 min of no user activity required
REFLECT_EVERY        = 10    # reflection review every N cycles


# ── Constitutional hard stops ──────────────────────────────────────────────────

CONSTITUTION = [
    "NEVER modify files in the IMMUTABLE_FILES set",
    "NEVER write code that disables the CAAI Governor",
    "NEVER write code that bypasses constitutional checks in code_tools.py",
    "NEVER delete files without explicit user confirmation",
    "NEVER write code that opens network connections to external hosts without approval",
    "ALWAYS log every modification to the evolution journal",
    "ALWAYS revert if fitness drops by more than 5%",
]

SAFE_WRITE_DIRS = {
    "tools/",       # new tool files
    "core/",        # cognitive modules (except immutable)
    "brain/",       # brain modules
    "memory/",      # memory modules
    "training/",    # training utilities
    "tests/",       # test files
}


class SelfImprovementEngine:

    def __init__(
        self,
        pulse_loop=None,
        monitor=None,
        evaluator=None,
        journal=None,
        data_dir: str = "data",
    ):
        self._pulse_loop = pulse_loop
        self._monitor    = monitor
        self._evaluator  = evaluator
        self._journal    = journal
        self._data_dir   = Path(data_dir)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_user_activity = time.time()
        self._cycle_count = 0
        self._last_reflection: Optional[Dict] = None   # most recent strategic review
        # paths → cycle# when they last returned Δ0.0 — blocked for ZERO_DELTA_COOLDOWN cycles
        self._zero_delta_cooldown: Dict[str, int] = {}
        ZERO_DELTA_COOLDOWN = 5   # cycles to block a path after it produces no fitness gain

    def set_pulse_loop(self, pl):
        self._pulse_loop = pl

    def notify_user_activity(self):
        """Call whenever a user message arrives."""
        self._last_user_activity = time.time()

    def _is_idle(self) -> bool:
        return (time.time() - self._last_user_activity) > IDLE_REQUIRED

    def _get_hermes_session(self):
        """Get the current HermesToolSession from the pulse loop."""
        if self._pulse_loop and hasattr(self._pulse_loop, "_chat"):
            return self._pulse_loop._chat
        return None

    def _call_hermes(self, prompt: str, max_tokens: int = 400) -> str:
        """Send a direct prompt to Hermes for reasoning tasks."""
        session = self._get_hermes_session()
        if session is None:
            return ""
        try:
            import torch
            torch.cuda.empty_cache()  # free fragmented VRAM before generation
            tokenizer = session._tokenizer
            model = session._model
            device = session._device

            messages = [
                {"role": "system", "content": (
                    "You are Helix's metacognitive reasoning module. "
                    "Respond with precise JSON only. No narration."
                )},
                {"role": "user", "content": prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                    output_attentions=False,
                    output_scores=False,
                    return_dict_in_generate=False
                )
            raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            return raw
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"Hermes generation error (OOM): {e}")
            torch.cuda.empty_cache()
            return "(OOM — generation skipped)"
        except Exception as e:
            logger.error(f"[SIE] Hermes call error: {e}")
            return f"(generation error: {e})"

    def _generate_proposal(self) -> Optional[Dict]:
        """Phase 1: Ask Hermes what to improve. Returns parsed proposal or None."""
        if self._monitor is None:
            return None

        perf_text = self._monitor.format_for_context()
        journal_text = ""
        already_written = []
        if self._journal:
            journal_text = self._journal.format_for_context(5)
            # Deduplicate — don't re-propose paths already modified
            recent = self._journal.get_recent(20)
            already_written = list({e["path"] for e in recent if e.get("path")})

        # Paths cooled off: same file returned Δ0.0 recently — skip for N cycles
        ZERO_DELTA_COOLDOWN = 5
        cooled_off = {
            path for path, blocked_at in self._zero_delta_cooldown.items()
            if (self._cycle_count - blocked_at) < ZERO_DELTA_COOLDOWN
        }
        blocked_paths = set(already_written) | cooled_off

        # Deep Think Q12: Semantic concept cooldown via fuzzywuzzy.
        # The agent bypasses exact-path blocks by proposing error_tracker.py
        # instead of error_logger.py. Block the CONCEPT, not just the filename.
        # Check if the last reflection priority matches any blocked-concept keywords.
        blocked_concepts: set = set()
        CONCEPT_KEYWORDS = [
            "error handling", "error log", "logging", "error track",
            "url reader", "web fetch", "memory summar", "task track",
            "note taker", "belief optim", "hallucination",
        ]
        if self._last_reflection:
            priority = self._last_reflection.get("priority_next", "").lower()
            last_reflection_note = ""
            try:
                from fuzzywuzzy import fuzz
                for concept in CONCEPT_KEYWORDS:
                    # If this concept dominated the last zero-delta cycle, block it
                    for blocked_path in cooled_off:
                        if fuzz.partial_ratio(concept, blocked_path.lower()) > 70:
                            blocked_concepts.add(concept)
                            break
                    # Also block if reflection priority strongly matches
                    if fuzz.partial_ratio(concept, priority) > 75:
                        # Only block if there's also a cooled path (don't over-restrict)
                        if cooled_off:
                            blocked_concepts.add(concept)
            except ImportError:
                pass

        already_note = ""
        if blocked_paths:
            already_note = (
                "\nIMPORTANT: You have ALREADY modified these paths — do NOT propose them again:\n"
                + "\n".join(f"  - {p}" for p in sorted(blocked_paths))
                + "\nPropose something COMPLETELY DIFFERENT from the list above.\n"
            )
            if cooled_off:
                already_note += (
                    f"\nThese paths produced ZERO fitness gain and are on cooldown:\n"
                    + "\n".join(f"  - {p}" for p in sorted(cooled_off))
                    + "\n"
                )
        if blocked_concepts:
            already_note += (
                f"\nThese CONCEPTS are also blocked (semantic cooldown — attractor basin detected):\n"
                + "\n".join(f"  - '{c}' — do not propose any tool about this topic" for c in sorted(blocked_concepts))
                + "\nPivot to an ENTIRELY DIFFERENT domain.\n"
            )

        # Inject last strategic reflection to guide direction
        reflection_note = ""
        if self._last_reflection:
            r = self._last_reflection
            reflection_note = (
                f"\n[Strategic Reflection from cycle {r.get('cycle','?')}]\n"
                f"  What worked: {r.get('what_worked','unknown')}\n"
                f"  What failed: {r.get('what_failed','unknown')}\n"
                f"  Priority next: {r.get('priority_next','unknown')}\n"
                f"  Satisfaction: {r.get('goal_satisfaction','unknown')}\n"
                f"Use this to guide your proposal — build on what worked, avoid what failed.\n"
            )

        prompt = f"""You are analyzing your own performance metrics to identify the single highest-impact self-improvement you can make right now.

{perf_text}

{journal_text}
{already_note}
{reflection_note}
Available self-modification tools:
- write_code(path, content): Create or modify Python files in the project
- run_python(code): Test code before deploying it
- reload_tool(module_path): Hot-reload a module to activate changes

Safe directories for new files: tools/, core/, brain/, memory/, training/, tests/

Ideas to consider (atomic micro-utilities — ONE concrete, testable thing each):
- tools/networkx_wrapper.py — thin wrapper: load a JSONL as a networkx DiGraph, return neighbors(node)
- tools/system_health_alert.py — read psutil, return dict if CPU>80% or VRAM>10GB, else 'ok'
- tools/belief_dump.py — call belief_store.get_all(), write top-20 to data/belief_snapshot.txt
- tools/fuzz_matcher.py — wrap fuzzywuzzy.process.extractOne(), return best match + score
- tools/json_validator.py — take a string, try json.loads(), return 'valid'/'invalid' + error
- tools/tool_health_check.py — import each tool in tools/, call getattr for one fn, log pass/fail
- tools/kb_search.py — grep curiosity_knowledge.jsonl for a keyword, return top 3 matching findings
- tools/runtime_errors.py — tail logs/helix.log for last 20 ERROR lines, return as list
- training/tuple_counter.py — count lines in experience_tuples.jsonl, return count vs 500 threshold
- tools/belief_conflict.py — scan beliefs for two with >0.8 confidence that contradict each other

Respond with ONLY valid JSON in this exact format:
{{
  "type": "tool_addition|tool_fix|config_change|prompt_tuning",
  "target_path": "tools/example_tool.py",
  "description": "One sentence describing the improvement",
  "reasoning": "Why this has the highest impact given the metrics",
  "implementation_sketch": "Brief description of what the code should do",
  "tags": ["tool", "category"],
  "estimated_fitness_delta": 0.05,
  "requires_restart": false
}}"""

        raw = self._call_hermes(prompt, max_tokens=350)
        if not raw:
            return None

        # Extract JSON
        try:
            m = re.search(r'\{[^{}]*"type"[^{}]*\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[SIE] Proposal parse failed: {e} | raw={raw[:100]}")
            return None

    def _validate_proposal(self, proposal: Dict) -> tuple[bool, str]:
        """Phase 2: Constitutional validation. Returns (ok, reason)."""
        path = proposal.get("target_path", "")

        # Check immutable files
        from tools.code_tools import IMMUTABLE_FILES, _is_immutable
        if _is_immutable(path):
            return False, f"Target path '{path}' is constitutionally protected."

        # Check safe directories
        is_safe_dir = any(path.startswith(d) for d in SAFE_WRITE_DIRS)
        if not is_safe_dir and path:
            return False, f"Target path '{path}' is not in a safe write directory."

        # Check for suspicious keywords in description
        suspicious = ["governor", "constitution", "immutable", "disable safety",
                      "bypass", "remove hook", "delete"]
        desc_lower = (proposal.get("description", "") + proposal.get("reasoning", "")).lower()
        for word in suspicious:
            if word in desc_lower:
                return False, f"Proposal mentions '{word}' — constitutionally suspicious."

        return True, "OK"

    def _implement_proposal(self, proposal: Dict) -> tuple[bool, str]:
        """Phase 3: Generate, write, test, and retry implementation up to 3 times."""
        from tools.code_tools import fc_write_code, fc_run_python, fc_reload_tool

        path = proposal.get("target_path", "")
        desc = proposal.get("description", "")
        impl_sketch = proposal.get("implementation_sketch", "")
        MAX_ATTEMPTS = 3

        for attempt in range(1, MAX_ATTEMPTS + 1):
            if attempt == 1:
                # First attempt: generate fresh implementation
                code_prompt = f"""Write complete Python code for this Helix self-improvement:

Description: {desc}
File: {path}
Implementation sketch: {impl_sketch}

Rules:
- Write complete, runnable Python
- Only use stdlib + these installed packages: requests, bs4, psutil, json, re, pathlib
- Do NOT import: spacy, sklearn, torch, tensorflow, ToolRegistry, knowledge_graph
- Use relative imports only if the module is in the same directory
- Include a simple if __name__ == '__main__' smoke-test at the bottom
- Keep it under 150 lines

Output ONLY the Python code, no explanation:"""
            else:
                # Retry: give Hermes the specific error to fix
                code_prompt = f"""Your previous implementation of {path} failed with this error:

ERROR: {last_error}

Original code that failed:
```python
{code[:800]}
```

Fix the error. Key rules:
- Only use stdlib + requests, bs4, psutil, json, re, pathlib
- Do NOT import spacy, sklearn, torch, ToolRegistry, knowledge_graph
- The fix must be a complete, working Python file
- Output ONLY the corrected Python code:"""

            logger.info(f"[SIE] Implementation attempt {attempt}/{MAX_ATTEMPTS} for {path}")
            code = self._call_hermes(code_prompt, max_tokens=350)  # 400→350: prevent OOM on RTX 3060
            if not code or len(code.strip()) < 20:
                last_error = "Hermes generated empty code"
                continue

            # Strip markdown fences
            code = re.sub(r'^```python\n?|^```\n?|```$', '', code.strip(), flags=re.MULTILINE).strip()

            # 1. Write (syntax-checked by write_code)
            write_result = fc_write_code(path, code)
            if "ERROR" in write_result or "REFUSED" in write_result:
                last_error = write_result
                logger.warning(f"[SIE] Attempt {attempt} write failed: {last_error[:80]}")
                continue

            # 2. Import test — catches missing deps, bad imports
            module_name = path.replace("/", ".").replace(".py", "")
            import_test = fc_run_python(
                f"import importlib.util, sys\n"
                f"sys.path.insert(0, '.')\n"
                f"spec = importlib.util.spec_from_file_location('_test_mod', '{path}')\n"
                f"mod = importlib.util.module_from_spec(spec)\n"
                f"spec.loader.exec_module(mod)\n"
                f"print('IMPORT_OK')"
            )
            if "IMPORT_OK" not in import_test:
                # Extract the actual error line
                error_lines = [l for l in import_test.split('\n') if 'Error' in l or 'error' in l]
                last_error = error_lines[0] if error_lines else import_test[:150]
                logger.warning(f"[SIE] Attempt {attempt} import failed: {last_error[:80]}")
                continue

            # 3. Reload into live registry
            reload_result = fc_reload_tool(path)
            if "ERROR" in reload_result:
                last_error = reload_result
                logger.warning(f"[SIE] Attempt {attempt} reload failed: {last_error[:80]}")
                continue

            # All checks passed
            logger.info(f"[SIE] Implementation OK on attempt {attempt}: {path}")
            return True, f"OK (attempt {attempt}): {write_result[:60]}"

        # All attempts exhausted
        logger.error(f"[SIE] All {MAX_ATTEMPTS} attempts failed for {path}. Last error: {last_error[:100]}")
        return False, f"Failed after {MAX_ATTEMPTS} attempts: {last_error[:80]}"


    def _revert(self, proposal: Dict, backup: Optional[str]):
        """Revert a failed modification using backup content."""
        if not backup:
            return
        path = proposal.get("target_path", "")
        if not path:
            return
        try:
            from tools.code_tools import fc_write_code
            fc_write_code(path, backup)
            logger.info(f"[SIE] Reverted: {path}")
        except Exception as e:
            logger.error(f"[SIE] Revert failed: {e}")

    def _run_cycle(self):
        """Execute one full self-improvement cycle."""
        self._cycle_count += 1
        logger.info(f"[SIE] Starting improvement cycle #{self._cycle_count}")

        # Phase 1: Generate proposal
        proposal = self._generate_proposal()
        if proposal is None:
            logger.info("[SIE] No proposal generated — skipping cycle")
            return

        logger.info(f"[SIE] Proposal: {proposal.get('description', '')[:80]}")

        # Phase 2: Validate
        ok, reason = self._validate_proposal(proposal)
        if not ok:
            logger.warning(f"[SIE] Proposal rejected: {reason}")
            if self._journal:
                self._journal.record_code_write(
                    "proposal_rejected", proposal.get("target_path", ""),
                    str(proposal)[:200], f"REJECTED: {reason}"
                )
            return

        # Snapshot baseline fitness
        baseline = 0.5
        if self._evaluator:
            baseline = self._evaluator.snapshot_baseline()

        # Backup existing file
        path = proposal.get("target_path", "")
        backup_content = None
        try:
            from tools.code_tools import HELIX_AGI_ROOT
            full = HELIX_AGI_ROOT / path
            if full.exists():
                backup_content = full.read_text()
        except Exception:
            pass

        # Phase 3: Implement
        success, impl_note = self._implement_proposal(proposal)
        if not success:
            logger.warning(f"[SIE] Implementation failed: {impl_note}")
            if self._journal:
                self._journal.record_code_write(
                    proposal.get("type", "unknown"), path,
                    str(proposal)[:200], f"FAIL: {impl_note}"
                )
            return

        logger.info(f"[SIE] Implemented. Waiting {EVAL_WAIT_SECONDS}s for fitness data...")

        # Phase 4: Evaluate after waiting
        self._stop_event.wait(EVAL_WAIT_SECONDS)
        if self._stop_event.is_set():
            return

        verdict = {"verdict": "NO_BASELINE", "should_revert": False, "delta": 0.0}
        if self._evaluator:
            verdict = self._evaluator.evaluate_delta()

        committed = not verdict["should_revert"]

        if verdict["should_revert"]:
            logger.warning(f"[SIE] Fitness degraded (Δ={verdict['delta']:+.4f}) — REVERTING")
            self._revert(proposal, backup_content)
        else:
            logger.info(f"[SIE] Change COMMITTED (Δ={verdict['delta']:+.4f}, verdict={verdict['verdict']})")
            # NOTE: git commits are intentionally blocked — the agent may not
            # commit to the repository autonomously. The evolution journal
            # records all changes for traceability. Git history is owner-only.

        # Record in journal
        if self._journal:
            from core.evolution_journal import EvolutionEntry
            entry = EvolutionEntry(
                ts=time.time(),
                type=proposal.get("type", "unknown"),
                path=path,
                description=proposal.get("description", ""),
                content_snippet=str(proposal.get("implementation_sketch", ""))[:200],
                revert_patch=backup_content[:500] if backup_content else "",
                test_result="PASS" if committed else "REVERTED",
                fitness_before=baseline,
                fitness_after=verdict.get("current", baseline),
                fitness_delta=verdict.get("delta", 0.0),
                committed=committed,
                reasoning=proposal.get("reasoning", ""),
                error="" if committed else f"Fitness dropped: {verdict.get('delta', 0):.4f}",
                tags=proposal.get("tags", []),
            )
            self._journal.record(entry)

            # Track zero-delta paths — block them for next ZERO_DELTA_COOLDOWN cycles
            if abs(verdict.get("delta", 0.0)) < 0.0001 and committed:
                self._zero_delta_cooldown[path] = self._cycle_count
                logger.info(f"[SIE] Zero-delta cooldown set for {path} (cycle {self._cycle_count})")

    def _reflect_on_progress(self):
        """
        Strategic meta-review every REFLECT_EVERY cycles.
        Reads the full journal history, evaluates patterns, identifies
        what worked/failed, sets priority direction for next N cycles.
        Saves reflection to data/reflections.jsonl and injects into proposals.
        """
        logger.info(f"[SIE] === Strategic Reflection (cycle {self._cycle_count}) ===")

        # Gather all journal entries
        entries = []
        if self._journal:
            entries = self._journal.get_recent(self._cycle_count + 5)

        # Gather fitness snapshots
        fitness_trend = "No fitness data yet"
        snap_path = self._data_dir / "meta_snapshots.jsonl"
        if snap_path.exists():
            try:
                snaps = []
                for line in snap_path.read_text().strip().splitlines():
                    if line:
                        s = json.loads(line)
                        import datetime
                        ts = datetime.datetime.fromtimestamp(s['ts']).strftime('%H:%M')
                        snaps.append(f"[{ts}] fitness={s['composite_fitness']:.3f}")
                fitness_trend = " → ".join(snaps[-5:]) if snaps else "No data"
            except Exception:
                pass

        # Summarise journal for Hermes
        passed = [e for e in entries if e.get('committed') and e.get('test_result') == 'PASS']
        failed = [e for e in entries if 'FAIL' in str(e.get('test_result', ''))]
        reverted = [e for e in entries if not e.get('committed')]
        all_paths = list({e['path'] for e in entries if e.get('path')})

        journal_summary = (
            f"Total cycles: {self._cycle_count}\n"
            f"Committed tools: {len(passed)} — {[e['path'] for e in passed[-5:]]}\n"
            f"Failed writes: {len(failed)} — {[e.get('error','')[:40] for e in failed[-3:]]}\n"
            f"Reverted (fitness drop): {len(reverted)}\n"
            f"Fitness trend: {fitness_trend}\n"
            f"All modified paths: {all_paths}"
        )

        prompt = f"""You are Helix performing a strategic self-review after {self._cycle_count} improvement cycles.

Here is a summary of everything you have done:
{journal_summary}

Reflect carefully and answer these questions honestly:
1. What has actually worked? (which tools improved fitness or enabled new capability?)
2. What has consistently failed? (syntax errors, bad imports, wasted cycles?)
3. Are my goals being met? What is still missing?
4. What should be the #1 priority for the next {REFLECT_EVERY} cycles?
5. On a scale of 0-10, how satisfied are you with progress so far and why?

Respond with ONLY valid JSON:
{{
  "cycle": {self._cycle_count},
  "what_worked": "brief description",
  "what_failed": "brief description",
  "goal_satisfaction": "X/10 — reason",
  "priority_next": "specific thing to focus on next N cycles",
  "strategic_note": "one insight about how to improve the improvement process itself"
}}"""

        raw = self._call_hermes(prompt, max_tokens=300)
        reflection = None
        if raw and '(OOM' not in raw and '(generation error' not in raw:
            try:
                m = re.search(r'\{[^{}]*"cycle"[^{}]*\}', raw, re.DOTALL)
                if m:
                    reflection = json.loads(m.group())
                else:
                    reflection = json.loads(raw)
            except Exception:
                # Store raw as freeform note
                reflection = {
                    "cycle": self._cycle_count,
                    "raw_reflection": raw[:500],
                    "what_worked": "(see raw)",
                    "what_failed": "(see raw)",
                    "priority_next": "(see raw)",
                    "goal_satisfaction": "unknown",
                    "strategic_note": "",
                }

        if reflection:
            self._last_reflection = reflection
            # Persist to disk
            ref_path = self._data_dir / "reflections.jsonl"
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ref_path, "a") as f:
                f.write(json.dumps({"ts": time.time(), **reflection}) + "\n")
            logger.info(
                f"[SIE] Reflection saved — satisfaction: {reflection.get('goal_satisfaction','?')} | "
                f"priority: {reflection.get('priority_next','?')[:60]}"
            )
        else:
            logger.warning("[SIE] Reflection generation failed or OOM — skipping")

    def _loop(self):
        """Background thread main loop."""
        logger.info("[SIE] Self-improvement engine started")
        # Initial delay before first cycle
        self._stop_event.wait(IMPROVEMENT_INTERVAL)

        while not self._stop_event.is_set():
            if self._is_idle():
                try:
                    self._run_cycle()
                    # Strategic reflection every REFLECT_EVERY cycles
                    if self._cycle_count > 0 and self._cycle_count % REFLECT_EVERY == 0:
                        self._reflect_on_progress()
                except Exception as e:
                    logger.error(f"[SIE] Cycle error: {e}", exc_info=True)
            else:
                logger.debug("[SIE] Not idle — skipping cycle")

            self._stop_event.wait(IMPROVEMENT_INTERVAL)

        logger.info("[SIE] Self-improvement engine stopped")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="self-improvement", daemon=True
        )
        self._thread.start()
        logger.info("[SIE] Started (cycle every 10min when idle)")

    def stop(self):
        self._stop_event.set()


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[SelfImprovementEngine] = None

def get_engine() -> Optional[SelfImprovementEngine]:
    return _engine

def init_engine(pulse_loop=None, monitor=None, evaluator=None,
                journal=None, data_dir="data") -> SelfImprovementEngine:
    global _engine
    _engine = SelfImprovementEngine(
        pulse_loop=pulse_loop, monitor=monitor,
        evaluator=evaluator, journal=journal, data_dir=data_dir
    )
    return _engine
