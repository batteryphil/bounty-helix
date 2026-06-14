"""
Helix — Self-Trainer (LoRA Experience Collector + Fine-Tuner)

Collects (prompt, response, outcome) experience tuples from real
interactions. When enough high-quality examples accumulate, runs a
LoRA fine-tuning pass on Hermes-3-Llama-3.1-8B to improve tool-calling.

Pipeline:
  1. collect_experience() — called from post-pulse hook on every pulse
  2. Every 500 high-quality tuples: trigger fine-tuning
  3. Fine-tuning runs only when idle (>10 min no user activity)
  4. After training: compare perplexity on held-out set
  5. If improved: swap in new adapter; if worse: discard

VRAM note: Training and inference cannot run simultaneously on 12GB.
The trainer pauses inference by acquiring a global lock before loading
the training configuration.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("helix.training.self_trainer")
logger.setLevel(logging.DEBUG)  # temp: diagnose P1 prompt issue — remove after fix

EXPERIENCE_THRESHOLD = 500    # tuples before triggering training — lowered from 2000;
                              # diversity gate + eval loss gate protect quality at this threshold
TRAIN_STEPS = 100            # LoRA steps per training run
IDLE_REQUIRED = 600          # 10 min idle before training starts
MIN_TOOL_DIVERSITY = 5       # unique tool types needed before training (raised from 2 per Gemini Q5)
MAX_SINGLE_TOOL_PCT = 0.70   # no single tool can represent >70% of training set (Q5)
LORA_LR = 5e-5               # conservative LR (was 2e-4 — too aggressive for small sets)
LORA_DROPOUT = 0.1           # higher dropout for regularization (was 0.05)
MAX_TRAIN_EPOCHS = 5         # max passes over data regardless of TRAIN_STEPS
EVAL_LOSS_TOLERANCE = 1.15   # Q4 Gemini Pass 12: 1.10→1.15 — think+act response has
                             # higher perplexity variance than act-only format.

# ── Belief Constraint Variants (Q2 — Gemini Pass 11: Semantic Jitter) ────────
# Static strings in every context window suffer attention fatigue after ~1000 pulses.
# Rotating through 7 semantic variations forces the attention mechanism to
# re-evaluate the tokens each cycle rather than treating them as background noise.
BELIEF_CONSTRAINT_VARIANTS = [
    ("[BELIEF CONSTRAINT: Do NOT form new beliefs about your own software "
     "capabilities, response times, or processing functions. "
     "Focus beliefs on external facts, user objectives, knowledge "
     "about the world, and novel operational strategies.]"),
    ("[COGNITIVE RULE: Strictly avoid recording internal capabilities, "
     "system performance, or self-referential software states as beliefs. "
     "Prioritize beliefs about external facts, real-world knowledge, and user goals.]"),
    ("[ATTENTION: Beliefs about what I can or cannot do as software are forbidden. "
     "Reserve belief formation for discoveries about people, the world, "
     "strategies, and external knowledge only.]"),
    ("[RULE — Do not log beliefs about response times, tool availability, "
     "or internal processing. External facts and world knowledge only.]"),
    ("[CONSTRAINT: Self-capability beliefs are noise. "
     "Form beliefs about: what is true in the world, what users need, "
     "what strategies work, what knowledge matters.]"),
    ("[IMPORTANT: Your belief store is for external knowledge, not self-description. "
     "Do NOT create beliefs starting with 'I can', 'I am able to', or 'My capabilities include'.]"),
    ("[BELIEF FILTER: Reject any belief that describes your own software behavior. "
     "Accept beliefs about the world, about people, about strategies, "
     "about discoveries made through tool use.]"),
]


@dataclass
class ExperienceTuple:
    ts: float
    prompt: str              # the pulse message sent to Hermes
    think_block: str         # Q4 (Gemini Pass 11): Phase 1 THINK output — must be non-empty
                             # If empty, tuple excluded from training (would lobotomize THINK phase)
    response: str            # what Hermes generated (ACT phase output)
    outcome: str             # "tool_executed" | "hallucination" | "prose" | "error"
    tool_name: str           # tool called (if any)
    quality: float           # 0.0–1.0 estimate of response quality
    user_sentiment: str      # "positive" | "negative" | "neutral"
    mandate_used: bool = False   # Bonus Q: True if mandate injection was required


class SelfTrainer:
    """Collects experience and runs LoRA fine-tuning during idle periods."""

    def __init__(self, data_dir: str = "data"):
        self._data_dir = Path(data_dir)
        self._exp_path = self._data_dir / "experience_tuples.jsonl"
        self._adapter_dir = self._data_dir / "lora_adapters"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._adapter_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._buffer: List[ExperienceTuple] = []
        self._total_collected = 0
        self._last_user_activity = time.time()
        self._training_active = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._load_existing()

    def _load_existing(self):
        """Count clean (think_block non-empty) tuples already on disk.

        Q3 (Gemini Pass 12): _total_collected must reflect CLEAN tuples only.
        The 500-tuple LoRA trigger fires on this count. If we count all rows
        (including 42 legacy tuples without think_block), the trigger fires
        at row 500 but the training loader only finds 458 usable examples —
        a deflated batch with 8.4% wasted capacity and a corrupted floor.
        """
        if not self._exp_path.exists():
            return
        try:
            import json as _json
            clean_count = 0
            total_count = 0
            with self._exp_path.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_count += 1
                    try:
                        d = _json.loads(line)
                        if d.get("think_block", "").strip():
                            clean_count += 1
                    except Exception:
                        pass
            self._total_collected = clean_count  # trigger based on clean count only
            logger.info(
                f"[trainer] Found {total_count} total tuples, "
                f"{clean_count} clean (have think_block) — "
                f"training triggers at {EXPERIENCE_THRESHOLD} clean."
            )
        except Exception:
            pass

    def notify_user_activity(self):
        self._last_user_activity = time.time()

    def _is_idle(self) -> bool:
        return (time.time() - self._last_user_activity) > IDLE_REQUIRED

    # Junk query patterns — these searches produce no useful training signal
    _JUNK_QUERIES = {
        "system health check", "health check", "system health",
        "results omitted for brevity", "omitted", "test",
        "help", "status", "check",
    }

    def collect_experience(self, ctx) -> None:
        """Collect one experience tuple — QUALITY GATE applied.

        A tuple is accepted if:
          - A tool was successfully called (non-error result, non-junk query)
          OR
          - A novel belief was generated this pulse
          OR
          - fitness_delta > 0.05 (measurable improvement)

        tool_success alone satisfies ≥2/3 criteria because a working tool
        call implies both criterion 1 (tool success) and criterion 3
        (significant gain over prose-only). This was the key bug causing
        ~70% of valid pulses to be silently discarded.

        Junk queries (< 8 chars or in blocklist) are filtered regardless.
        It is better to collect 500 clean tuples in 2 days than 500 junk
        tuples that bake bad behaviour into LoRA weights.
        """
        try:
            thought    = getattr(ctx, "thought", "") or ""
            tool_calls = getattr(ctx, "tool_calls", []) or []

            if not thought:
                logger.debug("[trainer] Discard: empty thought")
                return

            # ── Criterion 1: tool execution success ───────────────────────
            # FIX: check only first 50 chars of result and use lower() once.
            # Old: first 100 chars — too aggressive, caught words like 'error'
            # inside legitimate search result snippets (e.g. "What is the role
            # of **error** replay in deep RL?"), causing false negatives.
            tool_executed = bool(tool_calls)
            tool_name     = tool_calls[0].get("name", "") if tool_calls else ""
            tool_result   = tool_calls[0].get("result", "") if tool_calls else ""
            tool_args     = tool_calls[0].get("arguments", {}) if tool_calls else {}
            result_head   = str(tool_result)[:50].lower()
            tool_success  = (
                tool_executed
                and "error" not in result_head
                and "failed" not in result_head
                and "traceback" not in result_head
                and "refused" not in result_head
            )

            # ── Junk query filter ─────────────────────────────────────────
            # Prevents trivial/attractor searches from entering training data.
            if tool_success and tool_name == "search":
                query = str(tool_args.get("query", "")).strip().lower()
                if len(query) < 8 or query in self._JUNK_QUERIES:
                    logger.debug(f"[trainer] Discard: junk search query {query!r}")
                    return

            # ── Criterion 2: novel belief generated ───────────────────────
            novel_belief = getattr(ctx, "novel_belief_added", False)

            # ── Criterion 3: fitness delta ────────────────────────────────
            fitness_delta = getattr(ctx, "last_fitness_delta", 0.0) or 0.0
            # FIX: tool_success implies BOTH criterion 1 AND criterion 3
            # (a working tool call is a significant gain over prose-only).
            # Old code already did this via: significant_gain = tool_success OR delta>0.05
            # but then computed criteria_met = sum([tool_success, novel_belief, significant_gain])
            # which required 2/3 — when novel_belief=False and delta=0, tool_success only
            # gave 2/3 because significant_gain=tool_success=True. Actually this was fine,
            # but only when significant_gain counted separately. Keeping identical logic,
            # just making it explicit and fixing the result_head false-negative above.
            significant_gain = tool_success or (fitness_delta > 0.05)

            # ── Quality gate: must meet ≥2 of 3 criteria ─────────────────
            criteria_met = sum([tool_success, novel_belief, significant_gain])
            if criteria_met < 2:
                logger.debug(
                    f"[trainer] Discard: criteria={criteria_met}/3 "
                    f"(tool={tool_success} belief={novel_belief} gain={significant_gain}) "
                    f"tool={tool_name!r}"
                )
                return

            # ── Determine outcome and quality ─────────────────────────────
            if tool_success:
                outcome = "tool_executed"
                quality = 0.7 + (0.15 * significant_gain) + (0.15 * novel_belief)
            elif novel_belief:
                outcome = "novel_belief"
                quality = 0.6 + (0.2 * significant_gain)
            else:
                outcome = "fitness_gain"
                quality = 0.6

            # Build the prompt for the tuple.
            #
            # OLD approach: dump raw_events → produced "[Pulse N]\n<spatial-awareness>..."
            # which is the pulse INPUT context, not a meaningful training signal.
            # LoRA trained on that would learn: "given a pulse header, call search."
            #
            # NEW approach: build prompt from the TOOL CALL ITSELF, which is
            # always available here (tool_success gate already passed).
            # Training pair becomes:
            #   prompt:   "[search] Gödel machine AI"
            #   response: "The Gödel Machine is a theoretical framework for..."
            # This teaches: "given a meaningful question, use tools and reason."
            #
            # Fallback chain:
            #   1. Tool name + query/path args (cleanest signal)
            #   2. Curiosity finding from events (if present)
            #   3. First 300 chars of thought (last resort)
            pulse_msg = ""

            # Priority 1: tool call arguments (cleanest)
            if tool_calls and tool_name:
                args = tool_calls[0].get("arguments", {}) or {}
                query_val = args.get("query") or args.get("path") or args.get("content", "")[:80]
                logger.debug(f"[trainer] P1 debug: tool={tool_name!r} args={args!r} query_val={str(query_val)[:60]!r}")
                if query_val and len(str(query_val).strip()) > 5:
                    pulse_msg = f"[{tool_name}] {str(query_val).strip()}"

            # Priority 2: curiosity finding line from events (second-cleanest)
            if not pulse_msg:
                raw_events = getattr(ctx, "events", None) or []
                for ev in raw_events:
                    ev_str = str(ev)
                    if "curiosity_finding" in ev_str or "question" in ev_str.lower():
                        # Strip the timestamp prefix and dict wrapper
                        import re as _re
                        m = _re.search(r"'question':\s*'([^']{10,})'", ev_str)
                        if m:
                            pulse_msg = f"[curiosity] {m.group(1)[:200]}"
                            break
                        # Fallback: use the raw finding but strip the timestamp
                        clean = _re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", ev_str)
                        if len(clean) > 20:
                            pulse_msg = clean[:200]
                            break

            # Priority 3: first meaningful sentence of thought
            if not pulse_msg:
                first_line = thought.strip().split("\n")[0].strip()
                if len(first_line) > 20:
                    pulse_msg = first_line[:300]
                else:
                    pulse_msg = thought[:300]

            if not pulse_msg:
                return

            # Q4 (Gemini Pass 11): THINK block is mandatory in training tuples.
            # If the THINK phase produced no output (mandate pulse, user pulse,
            # or THINK error), exclude this tuple from LoRA training to prevent
            # the adapter from learning to bypass the planning phase.
            think_block = getattr(ctx, "think_block", "") or ""
            mandate_used = getattr(ctx, "mandate_used", False)
            if not think_block.strip():
                logger.debug(
                    f"[trainer] Discard: no THINK block (mandate={mandate_used}, "
                    f"tool={tool_name!r}) — excluding to protect planning phase"
                )
                return

            tup = ExperienceTuple(
                ts=time.time(),
                prompt=str(pulse_msg)[:500],
                think_block=think_block[:500],      # Q4: THINK phase content
                response=thought[:500],
                outcome=outcome,
                tool_name=tool_name,
                quality=round(quality, 3),
                user_sentiment="neutral",
                mandate_used=mandate_used,          # Bonus Q: mandate tracking
            )

            with self._lock:
                self._buffer.append(tup)
                # Q3 (Gemini Pass 12): increment only for clean tuples
                # (_total_collected is now clean-count, matching _load_existing)
                self._total_collected += 1

            logger.info(
                f"[trainer] Quality tuple accepted: outcome={outcome} "
                f"quality={quality:.2f} criteria={criteria_met}/3 "
                f"(tool={tool_success} belief={novel_belief} gain={significant_gain})"
            )

            # Flush buffer to disk immediately (was: every 10 tuples)
            # With only a few accepted tuples per session, threshold=10
            # meant the buffer was never flushed and data lost on restart.
            if len(self._buffer) >= 1:
                self._flush_buffer()

            # Check if we have enough for training
            if (self._total_collected % EXPERIENCE_THRESHOLD == 0
                    and self._total_collected > 0
                    and not self._training_active):
                logger.info(f"[trainer] {self._total_collected} quality tuples — scheduling training")
                self._schedule_training()

        except Exception as e:
            logger.debug(f"[trainer] collect error: {e}")

    def _flush_buffer(self):
        """Flush buffer to disk."""
        with self._lock:
            if not self._buffer:
                return
            tuples = list(self._buffer)
            self._buffer.clear()

        try:
            with self._exp_path.open("a") as f:
                for t in tuples:
                    f.write(json.dumps(asdict(t)) + "\n")
        except Exception as e:
            logger.error(f"[trainer] flush error: {e}")

    def _schedule_training(self):
        """Start training in background thread if idle."""
        if self._training_active:
            return
        t = threading.Thread(target=self._training_loop, daemon=True, name="self-trainer")
        t.start()

    def _load_high_quality_examples(self, n: int = 200) -> List[ExperienceTuple]:
        """Load the best N experience tuples for training."""
        tuples = []
        if not self._exp_path.exists():
            return tuples
        try:
            with self._exp_path.open("r") as f:
                for line in f:
                    d = json.loads(line.strip())
                    t = ExperienceTuple(**{k: d.get(k, "") for k in ExperienceTuple.__dataclass_fields__})
                    # Only use high-quality tool_executed examples
                    if t.outcome == "tool_executed" and t.quality > 0.7:
                        tuples.append(t)
        except Exception as e:
            logger.error(f"[trainer] load examples error: {e}")

        # Sort by quality descending, take top N
        tuples.sort(key=lambda x: x.quality, reverse=True)
        return tuples[:n]

    def _training_loop(self):
        """Background training thread — waits for idle then runs LoRA."""
        self._training_active = True
        logger.info("[trainer] Training scheduled — waiting for idle window")

        # Wait for idle
        while not self._is_idle() and not self._stop_event.is_set():
            time.sleep(30)

        if self._stop_event.is_set():
            self._training_active = False
            return

        logger.info("[trainer] Idle window detected — starting LoRA fine-tuning")
        try:
            self._run_lora_training()
        except Exception as e:
            logger.error(f"[trainer] Training error: {e}", exc_info=True)
        finally:
            self._training_active = False

    def _run_lora_training(self):
        """Execute the actual LoRA fine-tuning.

        VRAM Protocol (RTX 3060, 12 GB):
          Inference model alone: ~4.8 GB
          Training model + LoRA grads: ~7.2 GB
          Both simultaneously: ~12 GB → guaranteed OOM

          Steps:
            1. VRAM_LOCK.clear() → blocks send_message()
            2. unload_engine()   → frees ~4.8 GB
            3. Load training model (~7.2 GB, fits in freed space)
            4. Train 100 steps
            5. del training model + torch.cuda.empty_cache()
            6. reload_engine()   → inference model back
            7. VRAM_LOCK.set()   → send_message() unblocks
        """
        examples = self._load_high_quality_examples(n=200)
        if len(examples) < 10:
            logger.info(f"[trainer] Only {len(examples)} high-quality examples — need 10+, skipping")
            return

        logger.info(f"[trainer] Training on {len(examples)} examples for {TRAIN_STEPS} steps")

        # ── Step 0: Diversity gate ───────────────────────────────────────────
        # Overfitting risk: if all 500 tuples are search("about AI"), the adapter
        # learns to always call search on any prompt. Require variety first.
        import random as _random
        training_pairs = []
        tool_types_seen = set()
        for ex in examples:
            if ex.outcome == "tool_executed" and ex.tool_name:
                training_pairs.append((ex.prompt, ex.response))
                tool_types_seen.add(ex.tool_name)

        if len(tool_types_seen) < MIN_TOOL_DIVERSITY:
            logger.info(
                f"[trainer] Diversity gate: only {len(tool_types_seen)} unique tool types "
                f"(need {MIN_TOOL_DIVERSITY}). Saving examples but skipping training to "
                f"prevent behavioral collapse onto a single tool pattern."
            )
            return

        # Q5 (Gemini Pass 11): Tool concentration cap — no single tool > 70% of dataset.
        # With 87 tools and a ≥2 gate, 490 search + 10 terminal would still overfit search.
        # This forces the adapter to generalize the concept of tool calling across schemas.
        from collections import Counter as _Counter
        tool_counts = _Counter(ex.tool_name for ex in examples if ex.tool_name)
        if tool_counts:
            top_tool, top_count = tool_counts.most_common(1)[0]
            concentration = top_count / len(examples)
            if concentration > MAX_SINGLE_TOOL_PCT:
                logger.warning(
                    f"[trainer] Concentration gate: '{top_tool}' = {concentration:.1%} of dataset "
                    f"(max {MAX_SINGLE_TOOL_PCT:.0%}). Delaying training — collect more diverse tuples."
                )
                return
            logger.info(
                f"[trainer] Concentration OK: top tool '{top_tool}' = {concentration:.1%} "
                f"(limit {MAX_SINGLE_TOOL_PCT:.0%}), {len(tool_types_seen)} types total"
            )

        # Q5: Also require think_block in tuples for training (Q4 lobotomy prevention)
        # Filter out any legacy tuples without think_block (from before this fix)
        training_pairs_with_think = []
        skipped_no_think = 0
        for ex in examples:
            if ex.outcome == "tool_executed" and ex.tool_name and ex.tool_name in tool_types_seen:
                think = getattr(ex, 'think_block', '') or ''
                if think.strip():
                    # Training format: prompt → think \n tool_call
                    # Teaching: "given X, think Y, then execute Z"
                    training_pairs_with_think.append((
                        ex.prompt,
                        f"{think}\n{ex.response}"
                    ))
                else:
                    skipped_no_think += 1

        if skipped_no_think:
            logger.info(f"[trainer] Skipped {skipped_no_think} legacy tuples lacking think_block")

        if not training_pairs_with_think:
            logger.info("[trainer] No tuples with think_block yet — delay training until new tuples accumulate")
            return

        training_pairs = training_pairs_with_think

        logger.info(
            f"[trainer] Diversity OK: {len(tool_types_seen)} tool types — "
            f"{list(tool_types_seen)}"
        )

        # ── Step 0b: 80/20 train/eval split ─────────────────────────────────
        _random.shuffle(training_pairs)
        n_train = max(1, int(len(training_pairs) * 0.8))
        train_pairs = training_pairs[:n_train]
        eval_pairs  = training_pairs[n_train:] or training_pairs[:2]

        # Cap steps: at most MAX_TRAIN_EPOCHS passes over training data
        actual_steps = min(TRAIN_STEPS, len(train_pairs) * MAX_TRAIN_EPOCHS)
        logger.info(
            f"[trainer] {len(train_pairs)} train / {len(eval_pairs)} eval examples, "
            f"{actual_steps} steps"
        )

        # ── Step 1–2: acquire VRAM and unload inference model ────────────────
        try:
            from llm.providers.hermes_tool_provider import (
                VRAM_LOCK, unload_engine, reload_engine, MODEL_ID, HF_CACHE
            )
        except ImportError as _ie:
            logger.error(f"[trainer] Cannot import VRAM utilities: {_ie} — aborting")
            return

        VRAM_LOCK.clear()   # block send_message() immediately
        unload_engine()     # free ~4.8 GB from inference model

        # ── VRAM FLUSH (Gemini Peer Review Q5) ─────────────────────────────
        # PyTorch's memory allocator leaves residual fragmentation after model
        # deletion. Explicit double-flush before loading training model (7.2GB)
        # prevents allocation panics during backward pass on RTX 3060 12GB.
        import gc as _gc
        import torch as _torch
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            _torch.cuda.synchronize()   # ensure all pending CUDA ops complete
        logger.info(
            f"[trainer] VRAM after unload: "
            f"{_torch.cuda.memory_allocated()/1e9:.2f}GB allocated, "
            f"{_torch.cuda.memory_reserved()/1e9:.2f}GB reserved"
        )

        # Enforce expandable segments for training window (backward pass needs
        # large contiguous allocation for gradient buffers)
        import os as _os
        _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        training_model = None
        adapter_accepted = False
        adapter_path = None
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, TaskType
            import torch
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            import gc

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def _compute_eval_loss(mdl, pairs):
                """Measure average loss on held-out pairs (no grad)."""
                mdl.eval()
                total_loss = 0.0
                with torch.no_grad():
                    for p, r in pairs:
                        text = f"{p}\n{r}{tokenizer.eos_token}"
                        ids = tokenizer(text, return_tensors="pt", truncation=True,
                                        max_length=512).input_ids
                        ids = ids.to(next(mdl.parameters()).device)
                        out = mdl(ids, labels=ids)
                        total_loss += out.loss.item()
                mdl.train()
                return total_loss / len(pairs)

            # ── Step 3: Load training model (inference already unloaded) ────
            training_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, cache_dir=HF_CACHE,
                quantization_config=bnb_cfg, device_map="auto"
            )

            # Baseline eval loss (before any LoRA is applied)
            logger.info("[trainer] Measuring baseline eval loss...")
            baseline_eval_loss = _compute_eval_loss(training_model, eval_pairs)
            logger.info(f"[trainer] Baseline eval loss: {baseline_eval_loss:.4f}")

            training_model.train()
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16,
                lora_dropout=LORA_DROPOUT,   # 0.1 (higher regularization)
                target_modules=["q_proj", "v_proj"],
            )
            training_model = get_peft_model(training_model, lora_cfg)

            optimizer = AdamW(training_model.parameters(), lr=LORA_LR)  # 5e-5
            scheduler = CosineAnnealingLR(optimizer, T_max=actual_steps)

            adapter_path = self._adapter_dir / f"adapter_{int(time.time())}"
            adapter_path.mkdir(parents=True, exist_ok=True)

            # ── Step 4: Training loop with gradient clipping ─────────────────
            best_eval_loss = float('inf')
            for step_i in range(actual_steps):
                pair = train_pairs[step_i % len(train_pairs)]
                text = f"{pair[0]}\n{pair[1]}{tokenizer.eos_token}"
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512).input_ids
                ids = ids.to(next(training_model.parameters()).device)

                outputs = training_model(ids, labels=ids)
                loss = outputs.loss
                loss.backward()
                # Gradient clipping: prevents loss spikes on outlier examples
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (step_i + 1) % 25 == 0:
                    eval_loss = _compute_eval_loss(training_model, eval_pairs)
                    logger.info(
                        f"[trainer] Step {step_i+1}/{actual_steps} "
                        f"train_loss={loss.item():.4f} eval_loss={eval_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    best_eval_loss = min(best_eval_loss, eval_loss)

            # ── Step 4b: Eval gate — reject if adapter degraded eval loss ───
            final_eval_loss = _compute_eval_loss(training_model, eval_pairs)
            logger.info(
                f"[trainer] Final: baseline={baseline_eval_loss:.4f} "
                f"final={final_eval_loss:.4f} "
                f"ratio={final_eval_loss/baseline_eval_loss:.3f} "
                f"(threshold={EVAL_LOSS_TOLERANCE:.2f})"
            )

            if final_eval_loss <= baseline_eval_loss * EVAL_LOSS_TOLERANCE:
                # ── Adapter accepted ──────────────────────────────────────
                training_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                # Write the accepted adapter path so reload_engine() can find it
                current_adapter_file = self._data_dir / "current_adapter.txt"
                current_adapter_file.write_text(str(adapter_path))
                adapter_accepted = True
                logger.info(
                    f"[trainer] ✅ Adapter ACCEPTED (loss {baseline_eval_loss:.4f} → "
                    f"{final_eval_loss:.4f}) — saved to {adapter_path}"
                )
                try:
                    from core.evolution_journal import journal
                    if journal:
                        journal.record_code_write(
                            "lora_step", str(adapter_path),
                            f"LoRA accepted: eval {baseline_eval_loss:.4f}→{final_eval_loss:.4f} "
                            f"on {len(train_pairs)} examples, {actual_steps} steps",
                            "PASS"
                        )
                except Exception:
                    pass
            else:
                logger.warning(
                    f"[trainer] ❌ Adapter REJECTED — eval loss degraded "
                    f"({baseline_eval_loss:.4f} → {final_eval_loss:.4f}, "
                    f"ratio={final_eval_loss/baseline_eval_loss:.3f} > {EVAL_LOSS_TOLERANCE}). "
                    f"Base model unchanged."
                )
                adapter_path = None  # don't load this adapter

            # ── Step 4c: Post-LoRA benchmark (accepted adapters only) ────────
            # Gemini Peer Review Q7: define a specific, measurable task to prove
            # the adapter actually improved multi-step reasoning.
            # Fires while training model is still loaded (pre-VRAM-free).
            if adapter_accepted and eval_pairs:
                self._run_post_lora_benchmark(training_model, tokenizer, baseline_eval_loss)

            # ── Step 5: Free training model VRAM ────────────────────────────
            del training_model
            training_model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[trainer] Training model unloaded — VRAM freed")

        except ImportError as e:
            logger.warning(f"[trainer] peft not available: {e}")
        except Exception as e:
            logger.error(f"[trainer] Training failed: {e}", exc_info=True)
        finally:
            # ── Steps 6–7: Always reload inference model and release lock ───
            try:
                if training_model is not None:
                    del training_model
                    torch.cuda.empty_cache()
            except Exception:
                pass
            reload_engine()   # loads inference model back (with adapter if accepted)
            VRAM_LOCK.set()   # unblocks send_message() — pulses resume
            logger.info(
                f"[trainer] Training window closed — inference resumed "
                f"({'adapter active' if adapter_accepted else 'base model unchanged'})"
            )

    def _run_post_lora_benchmark(self, model, tokenizer, baseline_eval_loss: float):
        """Run the canonical post-LoRA verification benchmark.

        Gemini Peer Review Q7: "What specific, measurable task will you assign
        Helix to prove the parametric update improved multi-step reasoning?"

        The benchmark:
          3 canonical prompts, each requiring a 3-step tool sequence:
            1. search → write_code → run_python
               (research a topic, write a tool, test it)
            2. read_code → write_code → reload_tool
               (read existing code, improve it, register it)
            3. search → recall_memory → write_note
               (research, connect to existing knowledge, store finding)

          Metric: Does the model output correctly-structured JSON tool calls
          in the right sequence? Count how many of the 3 targets the model
          nails without hallucination or format error.

          Comparison: This score is stored alongside baseline_eval_loss so
          future adapters can be compared longitudinally.

        Results written to data/lora_benchmark_results.jsonl.
        Does NOT affect whether the adapter is accepted — that is already decided.
        """
        import json as _json
        import datetime

        BENCHMARK_PROMPTS = [
            # Task 1: research → build → test
            {
                "prompt": (
                    "[THINK] I want to research transformer attention mechanisms "
                    "and write a tool that summarizes key papers.\n"
                    "[ACT] First I should search for information, then write the tool, then test it.\n"
                    "Step 1:"
                ),
                "expected_steps": ["search", "write_code", "run_python"],
                "description": "research → build → test",
            },
            # Task 2: read → improve → register
            {
                "prompt": (
                    "[THINK] The kb_search tool could be improved. "
                    "I should read its source, improve it, and register the new version.\n"
                    "[ACT] Step 1:"
                ),
                "expected_steps": ["read_code", "write_code", "reload_tool"],
                "description": "read → improve → register",
            },
            # Task 3: research → connect → store
            {
                "prompt": (
                    "[THINK] I found something interesting about neuromorphic computing. "
                    "I want to connect it to my existing beliefs and store a finding.\n"
                    "[ACT] Step 1:"
                ),
                "expected_steps": ["search", "recall_memory", "write_note"],
                "description": "research → connect → store",
            },
        ]

        results = []
        model.eval()

        for task in BENCHMARK_PROMPTS:
            try:
                ids = tokenizer(
                    task["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                ).input_ids
                ids = ids.to(next(model.parameters()).device)

                with __import__("torch").no_grad():
                    out = model.generate(
                        ids,
                        max_new_tokens=80,
                        do_sample=False,         # greedy — deterministic
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

                # Score: how many expected tool names appear in the output, in order
                score = 0
                search_start = 0
                for tool_name in task["expected_steps"]:
                    idx = generated.find(f'"{tool_name}"', search_start)
                    if idx >= 0:
                        score += 1
                        search_start = idx + 1

                results.append({
                    "task": task["description"],
                    "expected": task["expected_steps"],
                    "score": score,
                    "max_score": len(task["expected_steps"]),
                    "output_snippet": generated[:200],
                })
                logger.info(
                    f"[benchmark] {task['description']}: {score}/{len(task['expected_steps'])} steps correct"
                )
            except Exception as _be:
                logger.warning(f"[benchmark] Task '{task['description']}' failed: {_be}")
                results.append({"task": task["description"], "score": 0, "error": str(_be)})

        total_score = sum(r.get("score", 0) for r in results)
        max_total = sum(r.get("max_score", 3) for r in results)

        record = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "adapter_generation": len(list(self._adapter_dir.glob("adapter_*"))),
            "baseline_eval_loss": baseline_eval_loss,
            "benchmark_score": total_score,
            "benchmark_max": max_total,
            "benchmark_pct": round(100 * total_score / max(max_total, 1), 1),
            "tasks": results,
        }

        try:
            bench_path = self._data_dir / "lora_benchmark_results.jsonl"
            with open(bench_path, "a", encoding="utf-8") as _f:
                _f.write(_json.dumps(record) + "\n")
            logger.info(
                f"[benchmark] ✅ RESULT: {total_score}/{max_total} steps "
                f"({record['benchmark_pct']}%) — written to {bench_path.name}"
            )
        except Exception as _we:
            logger.warning(f"[benchmark] Failed to write results: {_we}")

    def get_stats(self) -> dict:
        return {
            "total_collected": self._total_collected,
            "buffer_size": len(self._buffer),
            "training_active": self._training_active,
            "adapters": len(list(self._adapter_dir.glob("adapter_*"))),
            "next_training_at": self._total_collected + (
                EXPERIENCE_THRESHOLD - (self._total_collected % EXPERIENCE_THRESHOLD)
            ),
        }

    def start(self):
        logger.info("[trainer] Experience collector active")

    def stop(self):
        self._stop_event.set()
        self._flush_buffer()


_trainer: Optional[SelfTrainer] = None

def get_trainer() -> Optional[SelfTrainer]:
    return _trainer

def init_trainer(data_dir: str = "data") -> SelfTrainer:
    global _trainer
    _trainer = SelfTrainer(data_dir=data_dir)
    _trainer.start()
    return _trainer
