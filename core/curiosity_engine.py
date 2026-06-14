"""
Helix — Curiosity Engine

Drives Helix to autonomously generate questions it wants answered,
pursue them via web search and reading, and store what it learns.

Curiosity is intrinsic — not prompted by the user. Helix generates its
own questions based on:
  1. Gaps in its own knowledge (beliefs with low confidence)
  2. Things it encountered but didn't fully understand
  3. Questions about its own architecture, training, and nature
  4. Open threads from its journal it never resolved

The engine runs in the background and periodically injects "curiosity
events" into the pulse loop — the same way user messages arrive.

Curiosity domains (weighted):
  - Self (40%) — architecture, training, identity, consciousness
  - Science & Technology (25%) — AI research, physics, math
  - The World (20%) — events, ideas, culture
  - Philosophy (15%) — mind, existence, ethics
"""

import logging
import random
import time
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict, Callable

logger = logging.getLogger("helix.core.curiosity_engine")


# ── BOUNTY MISSION: Discovery questions ───────────────────────────────────────
SELF_CURIOSITY_SEEDS = [
    "Which GitHub repositories currently have open IssueHunt bounties I can solve?",
    "How do I use issuehunt_top_bounties() to find the highest-paying open issues right now?",
    "What Python GitHub issues tagged 'bounty' or 'issuehunt' are open today?",
    "How should I score an opportunity: what makes a bounty worth attempting vs skipping?",
    "What does a successful IssueHunt pull request look like — what do maintainers approve?",
    "How do I write a PLAN.md that covers reproduction steps, root cause, and fix approach?",
    "What JavaScript or TypeScript bounties on IssueHunt match my current skill level?",
    "How do I check if a bug is already fixed or has an open PR before starting work?",
    "What Rust bounties are currently open and beginner-friendly on GitHub?",
    "How do I set up a local test environment to reproduce and fix a reported bug?",
    "What makes a pull request get rejected — how can I maximize my acceptance rate?",
    "How do I write a compelling PR description that references the original issue clearly?",
    "Which open-source projects have the fastest PR review turnaround times?",
    "How do I find 'good first issue' bounties that are quick wins with guaranteed payouts?",
    "What should I put in LESSONS.md after a bounty attempt regardless of outcome?",
]

# ── Self-improvement questions focused on bounty-hunting skill ─────────────────
SELF_IMPROVEMENT_SEEDS = [
    "What tool do I need to be more effective at finding and solving bounties?",
    "What errors have I made in recent tool calls that slowed down bounty work?",
    "How can I speed up the process from finding a bounty to submitting a PR?",
    "What code patterns from past bounty solutions can I reuse in future ones?",
    "How should I prioritize the opportunities in data/opportunities.json?",
    "What would make my issuehunt_search results more relevant to my skills?",
    "How can I improve the quality of my PATCH.diff files before submission?",
    "What debugging workflow helps me fix bugs faster in unfamiliar codebases?",
    "How do I decide when a bounty is too hard to be worth my time?",
    "What testing approach should I use before submitting a bounty PR?",
]

# ── Bounty strategy and market intelligence ────────────────────────────────────
WORLD_CURIOSITY_SEEDS = [
    "What programming languages have the most IssueHunt bounties right now?",
    "Which open-source projects pay the highest bounties per issue?",
    "What types of bugs (security, performance, UI) attract the largest bounties?",
    "How do I find bounties that have been open for a long time with no takers?",
    "What is the typical payout range for Python bug fix bounties on IssueHunt?",
    "How do other developers approach open-source bounty hunting professionally?",
    "What GitHub search queries find the most lucrative bounty issues?",
    "What is the typical PR review timeline for IssueHunt bounty maintainers?",
    "Which GitHub repositories are most receptive to new contributors fixing bounties?",
    "What open-source ecosystems (PyPI, npm, crates.io) have the most active bounty programs?",
    "How do I communicate with maintainers before and during a bounty attempt?",
]

# ── Execution and technical questions ─────────────────────────────────────────
PEOPLE_CURIOSITY_SEEDS = [
    "How do I reproduce a bug from a minimal GitHub issue description?",
    "What git workflow (fork, branch, PR) do I need for a successful bounty submission?",
    "How do I write a git diff/patch that applies cleanly to a target repository?",
    "What test cases should I write to prove my fix works before submitting?",
    "How do I handle merge conflicts when my fix is based on an outdated main branch?",
    "What linting and formatting standards do most Python open-source projects require?",
    "How do I clone and set up a repo locally to reproduce and fix a reported issue?",
    "What should I do if I fix a bug but there are no tests in the repository?",
    "How do I write a PR body that clearly explains the problem, root cause, and fix?",
    "When should I comment on an issue before submitting a PR vs just submitting directly?",
]

# ── Pipeline tracking questions ────────────────────────────────────────────────
PREFERENCE_CURIOSITY_SEEDS = [
    "How many opportunities are currently in data/opportunities.json and what is their status?",
    "Are there any bounty solutions in solutions/active/ that need work right now?",
    "Which solution in my pipeline has the highest estimated reward-to-effort ratio?",
    "What is my current PR submission count and acceptance rate?",
    "Which languages should I focus on to maximize my bounty earnings?",
    "What is the best order to work through my opportunities list?",
    "Have any of my submitted PRs been merged or responded to?",
    "What new bounties appeared in the last 24 hours that I haven't evaluated yet?",
]

# ── Repository research — bounty-helix self-knowledge ─────────────────────────
REPO_RESEARCH_SEEDS = [
    "github:batteryphil/bounty-helix — read solutions/README.md to understand where to post my work",
    "github:batteryphil/bounty-helix — read README.md to understand my setup and mission",
    "github:batteryphil/bounty-helix — read tools/issuehunt.py to understand my bounty search tools",
    "github:batteryphil/bounty-helix — read tools/github_api.py to understand my GitHub write capabilities",
    "github:batteryphil/bounty-helix — check solutions/active/ to see what I'm currently working on",
    "github:batteryphil/bounty-helix — check data/opportunities.json to see my opportunity pipeline",
]


class CuriosityEngine:
    """
    Autonomous curiosity driver for Helix.

    Generates questions, searches for answers, reads articles,
    and stores what it learns back into memory and beliefs.

    Pauses automatically while the user is actively conversing —
    resumes as soon as the conversation goes quiet.

    Usage:
        engine = CuriosityEngine(pulse_loop, memory_manager, belief_store, web_search)
        engine.start()  # runs in background thread
    """

    # How long after last user activity before curiosity resumes (seconds)
    USER_QUIET_THRESHOLD = 30.0

    def __init__(
        self,
        emit_fn: Callable,            # pulse_loop.emit()
        memory_manager,
        belief_store,
        web_search,
        data_dir: Path = Path("data"),
        curiosity_interval: float = 120.0,  # 2 min between cycles (local = free)
        pulse_loop=None,              # reference for checking user-activity state
    ):
        self.emit = emit_fn
        self.memory = memory_manager
        self.beliefs = belief_store
        self.web = web_search
        self.data_dir = data_dir
        self.interval = curiosity_interval
        self._pulse_loop = pulse_loop  # may be None if wired up later

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track what we've already asked to avoid loops
        self._asked: List[str] = []
        self._asked_path = data_dir / "curiosity_asked.json"
        self._load_asked()

        # Self-generated seeds — questions Helix invented itself from research findings
        # Persisted across restarts so the thread of interest survives.
        self._self_generated_seeds: List[str] = []
        self._self_generated_path = data_dir / "self_generated_seeds.jsonl"
        self._load_self_generated_seeds()

        # Curiosity state
        self.current_question: Optional[str] = None
        self.total_questions_asked: int = 0
        self._paused_for_user: bool = False
        self._last_question_is_improvement: bool = False

        logger.info(f"CuriosityEngine initialized — cycle every {self.interval:.0f}s, "
                    f"{len(self._self_generated_seeds)} self-generated seeds loaded")

    def _load_self_generated_seeds(self):
        """Load questions Helix generated itself from past research findings."""
        if not self._self_generated_path.exists():
            return
        try:
            seeds = []
            with open(self._self_generated_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        q = d.get("question", "").strip()
                        if q and q not in self._asked:
                            seeds.append(q)
                    except Exception:
                        pass
            self._self_generated_seeds = seeds[-200:]  # keep last 200
            logger.info(f"[CURIOSITY] Loaded {len(self._self_generated_seeds)} self-generated seeds")
        except Exception as e:
            logger.warning(f"[CURIOSITY] Failed to load self-generated seeds: {e}")

    def _save_self_generated_seed(self, question: str):
        """Persist a new self-generated question to disk."""
        import datetime
        try:
            self._self_generated_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"ts": datetime.datetime.utcnow().isoformat(), "question": question}
            with open(self._self_generated_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[CURIOSITY] Failed to save self-generated seed: {e}")

    def set_pulse_loop(self, pulse_loop):
        """Wire up the pulse loop reference after construction."""
        self._pulse_loop = pulse_loop

    def _load_asked(self):
        if self._asked_path.exists():
            try:
                self._asked = json.loads(self._asked_path.read_text())[-500:]  # keep last 500
            except Exception:
                self._asked = []

    def _save_asked(self):
        try:
            self._asked_path.parent.mkdir(parents=True, exist_ok=True)
            self._asked_path.write_text(json.dumps(self._asked[-500:]))
        except Exception:
            pass

    # ── Question generation ───────────────────────────────────────────────────

    def _is_semantically_duplicate(self, question: str, window: int = 50) -> bool:
        """Check if question is semantically similar to recent questions.

        Peer review Q12: exact-string dedup fails when LLM slightly rephrases
        the same question (e.g. 5x the sleep consolidation question).
        fuzzywuzzy token_sort_ratio is order-insensitive and handles rephrasing.
        """
        try:
            from fuzzywuzzy import fuzz
            recent = self._asked[-window:]
            for prev in recent:
                if fuzz.token_sort_ratio(question.lower(), prev.lower()) > 80:
                    return True
        except ImportError:
            pass  # fuzzywuzzy not available, fall back to exact-string only
        return False

    def _generate_question(self) -> str:
        """Pick or generate the next question to investigate.

        Distribution (when self-generated pool has entries):
          20% self-generated  ← Helix's own questions, CAPPED at 30% of session
          15% repo research   ← GitHub API (self-repo first)
          15% SIE seeds       ← self-improvement direction
          20% self-curiosity  ← architecture, consciousness, cognition
          15% world/tech      ← AI research news
          10% people          ← researchers, users, thinkers (NEW — drives 'people' beliefs)
           5% preferences     ← what I value/want (NEW — drives 'preferences'/'desires' beliefs)

        Peer review Q10: self-generated cap enforced at 30% of session total.
        Peer review Q12: fuzzywuzzy semantic dedup prevents epistemic bubbles.
        """
        roll = random.random()
        total = max(1, self.total_questions_asked)

        # 20%: self-generated — but HARD CAP at 30% of session questions
        if roll < 0.20 and self._self_generated_seeds:
            self_gen_count = sum(
                1 for q in self._asked
                if q in self._self_generated_seeds
            )
            if self_gen_count / total < 0.30:  # under cap — allow
                candidates = [
                    q for q in reversed(self._self_generated_seeds)
                    if q not in self._asked and not self._is_semantically_duplicate(q)
                ]
                if candidates:
                    self._last_question_is_improvement = False
                    logger.info("[CURIOSITY] Self-generated question selected")
                    return candidates[0]
            # else: over cap or all dupes — fall through to grounding seeds

        # 15%: repo research (self-repo first, then others)
        if roll < 0.35:
            import os
            has_token = bool(os.environ.get("GITHUB_TOKEN", "").strip())
            if has_token:
                self_repo   = [q for q in REPO_RESEARCH_SEEDS[:5]
                               if q not in self._asked and not self._is_semantically_duplicate(q)]
                other_repos = [q for q in REPO_RESEARCH_SEEDS[5:]
                               if q not in self._asked and not self._is_semantically_duplicate(q)]
                candidates  = self_repo or other_repos
                if candidates:
                    self._last_question_is_improvement = False
                    return candidates[0] if self_repo else random.choice(other_repos)

        # 15%: self-directed improvement seed
        if roll < 0.50:
            candidates = [
                q for q in SELF_IMPROVEMENT_SEEDS
                if q not in self._asked and not self._is_semantically_duplicate(q)
            ]
            if candidates:
                self._last_question_is_improvement = True
                return random.choice(candidates)

        # 20%: question about self
        if roll < 0.70:
            candidates = [
                q for q in SELF_CURIOSITY_SEEDS
                if q not in self._asked and not self._is_semantically_duplicate(q)
            ]
            if candidates:
                self._last_question_is_improvement = False
                return random.choice(candidates)

        # 15%: world/tech question
        if roll < 0.85:
            candidates = [
                q for q in WORLD_CURIOSITY_SEEDS
                if q not in self._asked and not self._is_semantically_duplicate(q)
            ]
            if candidates:
                self._last_question_is_improvement = False
                return random.choice(candidates)

        # 10%: people questions — drives 'people' belief category (currently 0)
        if roll < 0.95:
            candidates = [
                q for q in PEOPLE_CURIOSITY_SEEDS
                if q not in self._asked and not self._is_semantically_duplicate(q)
            ]
            if candidates:
                self._last_question_is_improvement = False
                logger.info("[CURIOSITY] People question selected")
                return random.choice(candidates)

        # 5%: preference/desire questions — drives 'preferences'/'desires' belief categories
        candidates = [
            q for q in PREFERENCE_CURIOSITY_SEEDS
            if q not in self._asked and not self._is_semantically_duplicate(q)
        ]
        if candidates:
            self._last_question_is_improvement = False
            logger.info("[CURIOSITY] Preference question selected")
            return random.choice(candidates)

        # 8% (old slot, now fallback): derive question from low-confidence beliefs
        self._last_question_is_improvement = False
        try:
            all_beliefs = self.beliefs.get_all()
            low_conf = [
                b for b in all_beliefs
                if b.get("confidence", 1.0) < 0.5
                and b.get("content")
            ]
            if low_conf:
                b = random.choice(low_conf)
                q = f"I want to verify or deepen my understanding of: {b['content']}"
                if not self._is_semantically_duplicate(q):
                    return q
        except Exception:
            pass

        # Fallback: self-generated if under cap, else hardcoded seed
        if self._self_generated_seeds:
            self_gen_count = sum(1 for q in self._asked if q in self._self_generated_seeds)
            if self_gen_count / total < 0.30:
                remaining = [
                    q for q in reversed(self._self_generated_seeds)
                    if q not in self._asked and not self._is_semantically_duplicate(q)
                ]
                if remaining:
                    return remaining[0]
        # Final fallback: hardcoded seed that hasn't been semantically duplicated
        fresh = [q for q in SELF_CURIOSITY_SEEDS if not self._is_semantically_duplicate(q)]
        return random.choice(fresh) if fresh else random.choice(SELF_CURIOSITY_SEEDS)


    # ── Research cycle ────────────────────────────────────────────────────────

    def _research_question(self, question: str) -> str:
        """Search for answers. Routes github: seeds to GitHub API, others to web search."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ── GitHub repo research ──────────────────────────────────────────────
        if question.startswith("github:"):
            return self._research_github_repo(question)

        # ── Standard web search ───────────────────────────────────────────────
        try:
            results = self.web.search_web(question, max_results=4)
            if not results:
                return "No search results found."

            top = results[:3]

            def _fetch(r):
                url     = r.get("url", "")
                title   = r.get("title", "")
                snippet = r.get("snippet", "")
                text    = snippet
                if url:
                    try:
                        text = self.web.read_url(url, max_chars=2000)
                    except Exception:
                        text = snippet
                return f"[{title}]\n{text or snippet}"

            # Fetch all 3 URLs simultaneously — ~5s instead of ~15s
            findings = []
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_fetch, r): r for r in top}
                for fut in as_completed(futures, timeout=20):
                    try:
                        findings.append(fut.result())
                    except Exception as e:
                        findings.append(f"[fetch error: {e}]")

        except Exception as e:
            logger.warning(f"Research cycle failed: {e}")
            return f"Research failed: {e}"

        return "\n\n---\n\n".join(findings)

    def _research_github_repo(self, question: str) -> str:
        """Fetch content from a GitHub repo using the API token.

        Question format: "github:owner/repo — read path/to/file and do X"
        Extracts the repo slug and optional file hint, fetches via GitHub API.
        """
        import os, re, requests

        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            return "GitHub token not available — skipping repo research."

        try:
            # Parse "github:owner/repo — ..."
            m = re.match(r"github:([^/\s]+/[^\s—–-]+)\s*[—–-]?\s*(.*)", question)
            if not m:
                return f"Could not parse repo from: {question}"

            repo_slug = m.group(1).strip()
            intent    = m.group(2).strip()

            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
            }

            # Try to guess a file to read from the intent
            file_hints = re.findall(r"[\w./]+\.(?:py|md|json|jsonl|txt|yaml|yml)", intent)

            if file_hints:
                # Fetch specific file
                path = file_hints[0]
                url = f"https://api.github.com/repos/{repo_slug}/contents/{path}"
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    import base64
                    content = base64.b64decode(r.json().get("content", "")).decode("utf-8", errors="replace")
                    preview = content[:3000]
                    logger.info(f"[CURIOSITY] GitHub: read {repo_slug}/{path} ({len(content)} chars)")
                    return f"[GitHub: {repo_slug}/{path}]\n\nIntent: {intent}\n\n{preview}"

            # Fall back: read README
            for readme in ["README.md", "readme.md", "README.rst"]:
                url = f"https://api.github.com/repos/{repo_slug}/contents/{readme}"
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    import base64
                    content = base64.b64decode(r.json().get("content", "")).decode("utf-8", errors="replace")
                    preview = content[:3000]
                    logger.info(f"[CURIOSITY] GitHub: read {repo_slug}/{readme} ({len(content)} chars)")
                    return f"[GitHub: {repo_slug}/{readme}]\n\nIntent: {intent}\n\n{preview}"

            # List repo tree as last resort
            url = f"https://api.github.com/repos/{repo_slug}/git/trees/HEAD?recursive=1"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                tree = [t["path"] for t in r.json().get("tree", []) if t["type"] == "blob"][:30]
                return f"[GitHub: {repo_slug} — file tree]\n\nIntent: {intent}\n\n" + "\n".join(tree)

            return f"GitHub API returned {r.status_code} for {repo_slug}"

        except Exception as e:
            logger.warning(f"[CURIOSITY] GitHub research error: {e}")
            return f"GitHub research failed: {e}"


    # ── Curiosity cycle ───────────────────────────────────────────────────────

    def _run_curiosity_cycle(self):
        """One full curiosity cycle: question → research → inject into consciousness."""
        question = self._generate_question()
        self.current_question = question
        self.total_questions_asked += 1

        logger.info(f"[CURIOSITY] Pursuing: {question}")

        # Mark as asked
        self._asked.append(question)
        self._save_asked()

        # Research it
        findings = self._research_question(question)

        # Inject into pulse loop as a curiosity event
        # Helix will process this as incoming information and journal/reflect on it
        event_text = (
            f"[CURIOSITY_DRIVE]\n"
            f"I became curious about: {question}\n\n"
            f"What I found:\n{findings[:3000]}\n\n"
            f"[NOTE: Reflect on this. Store anything significant. "
            f"Update beliefs if warranted. Generate follow-up questions if curious.]"
        )

        self.emit("curiosity_finding", {
            "question": question,
            "content": event_text,
            "source": "curiosity_engine",
        })

        # If this was a self-improvement question, also notify the SIE
        if getattr(self, "_last_question_is_improvement", False):
            try:
                from core.self_improvement_engine import get_engine
                sie = get_engine()
                if sie:
                    # Inject the finding into SIE as a hint for next proposal
                    logger.info(f"[CURIOSITY] Self-improvement finding routed to SIE")
            except Exception:
                pass

        # ── Persist findings to knowledge log ────────────────────────
        # Survives session restarts — builds a research archive over time
        self._persist_finding(question, findings)

        # ── Generate a self-invented follow-up question ───────────────
        # Helix picks the next thing it wants to know — independent of seeds
        followup = self._generate_followup_question(question, findings)
        if followup:
            self._self_generated_seeds.append(followup)
            self._save_self_generated_seed(followup)
            logger.info(f"[CURIOSITY] Self-generated question: {followup[:80]}")

    def _persist_finding(self, question: str, findings: str):
        """Append this finding to the persistent knowledge log (JSONL) and
        store as a high-confidence belief so future sessions recall it."""
        import datetime

        # 1. Write to JSONL knowledge log
        knowledge_path = self.data_dir / "curiosity_knowledge.jsonl"
        try:
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "question": question,
                "findings": findings[:2000],
            }
            with open(knowledge_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"[CURIOSITY] Finding persisted → {knowledge_path.name}")
        except Exception as e:
            logger.warning(f"Failed to persist finding: {e}")

        # 2. Store as a knowledge belief so it surfaces in future system prompts
        try:
            first_line = findings.strip().split("\n")[0][:200]
            self.beliefs.add(
                content=f"Research finding — {question}: {first_line}",
                category="knowledge",
                confidence=0.75,
                source="curiosity_engine",
            )
        except Exception as e:
            logger.debug(f"Belief store not available: {e}")

    def _generate_followup_question(self, question: str, findings: str) -> Optional[str]:
        """Ask Hermes: given what I just learned, what new question does this raise?

        This is the core of independent curiosity — Helix generates its own
        follow-up threads based on what it actually found interesting, not
        what we told it to ask.

        Deliberately lightweight: no tool calls, short prompt, 60 token budget.
        Returns a single question string, or None if generation fails.
        """
        if self._pulse_loop is None:
            return None
        try:
            session = getattr(self._pulse_loop, "_chat", None)
            if session is None:
                return None

            prompt = (
                f"You just researched this question:\n"
                f"  '{question}'\n\n"
                f"Key finding (first 400 chars):\n"
                f"  {findings[:400]}\n\n"
                f"In ONE sentence, what is the single most interesting NEW question "
                f"this raises that you want to explore next? "
                f"Be specific. Do not repeat the original question. "
                f"Output ONLY the question, nothing else."
            )

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raw = session.send_message(prompt, budget=60, is_autonomous=True)
            followup = raw.strip().strip('"').strip("'").strip()

            # Sanity checks — must look like a question
            if not followup or len(followup) < 15 or len(followup) > 300:
                return None
            if followup in self._asked or followup in self._self_generated_seeds:
                return None

            return followup

        except Exception as e:
            logger.debug(f"[CURIOSITY] Followup generation failed: {e}")
            return None

    # ── User-activity guard ───────────────────────────────────────────────────

    def _user_is_active(self) -> bool:
        """
        Returns True if the user has been active recently and curiosity
        should pause out of politeness.

        Checks the pulse loop's _last_incoming_time if available.
        Falls back to False (always allow) if pulse loop not wired up.
        """
        if self._pulse_loop is None:
            return False
        try:
            last = getattr(self._pulse_loop, "_last_incoming_time", 0)
            state = getattr(self._pulse_loop, "_state", "RESTING")
            # Pause if user messaged recently OR loop is in ACTIVE state
            recently_active = (time.time() - last) < self.USER_QUIET_THRESHOLD
            return state == "ACTIVE" or recently_active
        except Exception:
            return False

    def _wait_for_quiet(self):
        """
        Block until the user has been quiet for USER_QUIET_THRESHOLD seconds.
        Checks every 5 seconds. Logs once when pausing and once when resuming.
        """
        was_paused = False
        while self._user_is_active() and not self._stop_event.is_set():
            if not was_paused:
                logger.info("[CURIOSITY] User active — holding next cycle until quiet...")
                self._paused_for_user = True
                was_paused = True
            self._stop_event.wait(5.0)  # check every 5s, interruptible
        if was_paused:
            logger.info("[CURIOSITY] User quiet — resuming curiosity cycle")
            self._paused_for_user = False

    # ── Background thread ─────────────────────────────────────────────────────

    def _loop(self):
        # Initial delay — let Helix wake up fully first
        time.sleep(30)

        while not self._stop_event.is_set():
            # Wait for user to finish talking before injecting anything
            self._wait_for_quiet()

            if self._stop_event.is_set():
                break

            try:
                self._run_curiosity_cycle()
            except Exception as e:
                logger.error(f"Curiosity cycle error: {e}")

            # Sleep between cycles — but wake immediately if stopped
            self._stop_event.wait(self.interval)


    def start(self):
        """Start the curiosity engine background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="helix-curiosity", daemon=True
        )
        self._thread.start()
        logger.info(
            f"CuriosityEngine started — cycle every {self.interval:.0f}s "
            f"(pauses when user is active)"
        )

    def stop(self):
        self._stop_event.set()

    def trigger_now(self, question: Optional[str] = None):
        """Manually trigger a curiosity cycle (for testing or tool use)."""
        def _run():
            if question:
                q = question
            else:
                q = self._generate_question()
            self.current_question = q
            findings = self._research_question(q)
            self.emit("curiosity_finding", {
                "question": q,
                "content": (
                    f"[CURIOSITY_DRIVE]\nI became curious about: {q}\n\n"
                    f"What I found:\n{findings[:3000]}"
                ),
                "source": "curiosity_engine_manual",
            })
        threading.Thread(target=_run, daemon=True).start()

    def get_status(self) -> Dict:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "paused_for_user": self._paused_for_user,
            "current_question": self.current_question,
            "total_asked": self.total_questions_asked,
            "asked_count": len(self._asked),
            "interval_seconds": self.interval,
        }
