"""
Helix — Main Entry Point

Initializes the full Helix cognitive architecture:
  - Memory systems (short-term, long-term, core)
  - Belief store (identity, knowledge, capabilities, etc.)
  - Physics engine (8D manifold)
  - Preconscious injection
  - Pulse loop (consciousness)
  - Self-evolution engines
  - Dashboard comms
"""

import os
# Must be set before torch is imported — reduces VRAM fragmentation on RTX 3060
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
import logging
import logging.handlers
import numpy as np

# ── Logging Setup ────────────────────────────────────────────────────
# All logger.info/debug calls across the codebase route here.
# Captures FC dispatch, pulse state, preconscious injection, etc.
_log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_log_dir, exist_ok=True)

_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_log_dir, "helix.log"),
    maxBytes=5_000_000,  # 5 MB per file
    backupCount=3,        # keep 3 rotated copies
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

# Root logger — captures everything
logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler],
)


def _load_credentials():
    """Load API keys from ~/.config/helix/credentials.env into os.environ.

    Supports both python-dotenv (if installed) and manual parsing.
    Credentials loaded: HELIX_TELEGRAM_TOKEN, MOLTBOOK_API_KEY,
    GITHUB_TOKEN, GEMINI_API_KEY, ANTHROPIC_API_KEY, etc.
    """
    cred_path = os.path.expanduser("~/.config/helix/credentials.env")
    if not os.path.exists(cred_path):
        print(f"  ⚠ No credentials file at {cred_path}")
        return

    # Try python-dotenv first
    try:
        from dotenv import load_dotenv
        load_dotenv(cred_path, override=False)
        print(f"  Credentials: loaded via dotenv")
        return
    except ImportError:
        pass

    # Manual parse — handle KEY="value" and KEY=value
    loaded = 0
    with open(cred_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
    print(f"  Credentials: loaded {loaded} keys")


# Load credentials BEFORE any imports that use env vars
_load_credentials()


from memory.memory_manager import MemoryManager
from memory.belief_store import BeliefStore
from core.physics_engine import PhysicsEngine
from core.preconscious import Preconscious
from core.scratchpad import Scratchpad
from core.pulse_loop import PulseLoop
from llm.orchestrator import LLMOrchestrator
from llm.background_daemon import BackgroundDaemon
from llm.providers.base import detect_available_provider
from tools.tool_executor import ToolExecutor
from tools.channel_router import ChannelRouter
from comms.telegram_bot import HelixTelegramBot
from brain.stability_sentinel import StabilitySentinel
# ── Consciousness & Self-Awareness Modules ──────────────────────────────────
from brain.autobiographical_thread import AutobiographicalThread
from core.attention_schema import AttentionSchema
from core.recursive_monologue import RecursiveMonologue
from core.curiosity_engine import CuriosityEngine
from brain.self_model import SelfModel
from brain.titan_memory_bridge import TitanMemoryBridge
from tools.web_search import WebSearch
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Self-Evolution Modules ───────────────────────────────────────────────────
try:
    from core.evolution_journal import init_journal
    from core.metacognitive_monitor import init_monitor
    from core.fitness_evaluator import init_evaluator
    from core.self_improvement_engine import init_engine as init_sie
    from training.self_trainer import init_trainer
    from core.context_window_manager import init_manager as init_ctx_mgr
    _SELF_EVOLVE_AVAILABLE = True
except ImportError as _se_err:
    _SELF_EVOLVE_AVAILABLE = False
    print(f"  [Self-Evolution] Import error: {_se_err}")

# Import code tools to auto-register under toolset='self'
try:
    import tools.code_tools  # noqa: F401 — triggers _register() on import
except Exception as _ct_err:
    print(f"  [Code Tools] Import error: {_ct_err}")




def on_thought(pulse_number: int, thought: str, events: list):
    """Callback for each pulse — prints internal monologue to console."""
    state_tag = "💬" if events else "💭"
    print(f"\n  {state_tag} [Pulse {pulse_number}] {thought}")


def on_delivery(recipient: str, message: str):
    """Callback for outbound messages — prints to console and pushes to dashboard."""
    print(f"\n  📤 → {recipient}: {message}")
    try:
        from dashboard.dashboard_comms import get_comms
        comms = get_comms()
        comms.push_outbound(recipient, message)
    except ImportError:
        pass


def setup_helix(data_dir: str = "data"):
    """Initialize the complete Helix cognitive architecture."""
    print("Initializing Helix Architecture...")

    # ── 0. Model Preload — warm engine in VRAM before first pulse ────────────
    # Falcon-Mamba-7B (4-bit NF4, ~4.5GB) is loaded lazily on first inference.
    # We do an explicit preload here to front-load the 30s startup delay so
    # the first chat response isn't slow.
    try:
        from llm.providers.base import detect_available_provider
        _provider = detect_available_provider()
        if _provider and _provider.provider_type == "mistral_tool":
            print("  Mistral-7B-Instruct-v0.3: preloading 4-bit model (~10s)...")
            from llm.providers.mistral_tool_provider import _load_engine
            _load_engine()
            print("  Mistral-7B-Instruct-v0.3: ready ✅ (tool-calling enabled)")
        elif _provider and _provider.provider_type == "falcon_mamba":
            print("  Falcon-Mamba-7B: preloading 4-bit model (~30s)...")
            from llm.providers.falcon_mamba_provider import _load_engine
            _load_engine()
            print("  Falcon-Mamba-7B: ready ✅")
        elif _provider and _provider.provider_type == "titan":
            from titan_inference import preload as _titan_preload
            print("  Titan: preloading into VRAM (one-time, ~30s)...")
            _titan_preload()
    except Exception as _e:
        print(f"  Model preload skipped ({_e}) — will load on first inference")

    # ── 1. Memory Systems ────────────────────────────────────────────

    memory_manager = MemoryManager(os.path.join(data_dir, "memory"))
    belief_store = BeliefStore(os.path.join(data_dir, "beliefs"))
    scratchpad = Scratchpad(os.path.join(data_dir, "scratchpad"))

    mem_stats = memory_manager.get_stats()
    belief_stats = belief_store.get_stats()
    print(f"  Memory: {mem_stats}")
    print(f"  Beliefs: {belief_stats}")
    print(f"  Scratchpad: {len(scratchpad.get_active_notes())} active notes")

    # ── 2. Physics Engine (8D manifold + 384D semantic index) ──────────
    spatial_dir = os.path.join(data_dir, "spatial")
    physics = PhysicsEngine(data_dir=spatial_dir)
    memory_manager.set_physics(physics)
    print(f"  Spatial: pulse={physics._pulse_count}, γ={physics._gamma:.2f}")
    print(f"  SemanticIndex: {physics.semantic_index.count} vectors ({physics.semantic_index.get_stats()['search_strategy']})")

    # ── 2b. Stability Sentinel ───────────────────────────────────────
    from pathlib import Path
    sentinel = StabilitySentinel(
        base_dir=Path("."),
        memory=memory_manager,
        probe_interval=60,
    )
    print(f"  Sentinel: Ω={sentinel.omega:.3f}, severity={sentinel.get_severity()}")

    # ── 2c. Wire Sentinel → Spatial Mind ─────────────────────────────
    #    Connects the Sentinel to the real 8D manifold so _compute_lagrangian()
    #    uses actual Shannon entropy and KL divergence from the cognitive space
    #    instead of falling back to hardware health proxies.
    sentinel._spatial_mind = physics.spatial_mind

    # ── 2d. Bootstrap 8D Manifold ────────────────────────────────────
    #    Populate both cognitive spaces (belief field + memory field) from
    #    existing data so gravity queries are non-empty from the first pulse
    #    and the identity center x* is computed from real core beliefs.
    physics.bootstrap_from_stores(belief_store, memory_manager)
    print(f"  Spatial bootstrap: {physics.spatial_mind.belief_space.point_count} beliefs, "
          f"{physics.spatial_mind.memory_space.point_count} memories in 8D manifold")

    # ── 3. Tool Executor + Channel Router ───────────────────────────────
    channel_router = ChannelRouter(data_dir=data_dir)
    tool_executor = ToolExecutor(channel_router=channel_router)
    tool_executor.memory_manager = memory_manager
    tool_executor.scratchpad = scratchpad
    print(f"  Contacts: {len(channel_router.contacts)} known")
    print(f"  Tools: executor ready")

    # ── 3b. Telegram Bot ────────────────────────────────────────────
    telegram_bot = HelixTelegramBot()
    channel_router.set_telegram_bot(telegram_bot)
    print(f"  Telegram: {'enabled' if telegram_bot.enabled else 'disabled (no token)'}")

    # ── 4. Pre-Conscious + Scratchpad ────────────────────────────────
    tool_schemas_path = os.path.join(data_dir, "tool_schemas.json")
    preconscious = Preconscious(
        memory_manager=memory_manager,
        belief_store=belief_store,
        physics_engine=physics,
        scratchpad=scratchpad,
        channel_router=channel_router,
        tool_schemas_path=tool_schemas_path,
        sentinel=sentinel,
    )

    # ── 5. LLM Provider Detection ────────────────────────────────
    provider_config = detect_available_provider()
    if provider_config:
        print(f"  Provider: {provider_config.provider_type} ({provider_config.model})")
    else:
        print("  Provider: NONE — running without LLM")

    # ── 6. Pulse Loop ────────────────────────────────────────────
    journal_dir = "journals"
    pulse_loop = PulseLoop(
        memory_manager=memory_manager,
        belief_store=belief_store,
        physics_engine=physics,
        preconscious=preconscious,
        scratchpad=scratchpad,
        tool_executor=tool_executor,
        channel_router=channel_router,
        provider_config=provider_config,
        journal_dir=journal_dir,
        thought_callback=on_thought,
        delivery_callback=on_delivery,
        sentinel=sentinel,
    )

    # Wire telegram to pulse loop for inbound messages
    telegram_bot.set_pulse_loop(pulse_loop)

    # Wire tool executor to pulse loop for context reset tool
    tool_executor.set_pulse_loop(pulse_loop)

    # Wire sentinel to tool executor (for somatic echo on memory recall)
    tool_executor._sentinel = sentinel

    # ── Sentinel → PulseLoop event bridge ────────────────────────────
    #    Stability events (critical, warning, context_awareness) flow
    #    into the pulse loop's event queue so Helix can consciously
    #    perceive stability changes.
    sentinel.set_event_callback(pulse_loop.emit)

    # ── Context usage proxy for Sentinel ─────────────────────────────
    #    Lightweight adapter so the Sentinel can monitor context window
    #    saturation without holding a full PulseLoop reference.
    class _ContextProxy:
        def __init__(self, pl):
            self._pl = pl
        def context_usage_pct(self):
            if not self._pl._compressor:
                return 0.0
            max_tokens = self._pl._compressor.context_length
            current = self._pl._session_token_count
            return (current / max_tokens) * 100 if max_tokens > 0 else 0.0

    sentinel._consciousness = _ContextProxy(pulse_loop)

    # ── 6. Orchestrator (thin wrapper) ───────────────────────────────
    orchestrator = LLMOrchestrator(pulse_loop, memory_manager)

    # ── 7. Background Daemon (Dream Engine) ────────────────────────
    #    The Curator needs an llm_client with .generate(prompt, system_instruction)
    #    for Phase 2 (belief extraction) and Phase 3 (compound synthesis).
    curator_llm = None
    try:
        from google import genai as _genai
        _curator_client = _genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

        class _CuratorLLM:
            """Lightweight wrapper giving the Curator a .generate() interface."""
            _model = "gemini-2.5-flash"
            def generate(self, prompt: str, system_instruction: str = ""):
                from google.genai import types as _types
                config = _types.GenerateContentConfig(
                    system_instruction=system_instruction or None,
                    temperature=0.3,
                    max_output_tokens=2048,
                )
                resp = _curator_client.models.generate_content(
                    model=self._model,
                    contents=[_types.Content(role="user", parts=[_types.Part(text=prompt)])],
                    config=config,
                )
                return resp.candidates[0].content.parts[0]

        curator_llm = _CuratorLLM()
        print("  Curator LLM: ready (gemini-2.5-flash)")
    except Exception as e:
        print(f"  Curator LLM: unavailable ({e})")

    daemon = BackgroundDaemon(
        physics_engine=physics,
        belief_store=belief_store,
        memory_manager=memory_manager,
        llm_client=curator_llm,
        data_dir=data_dir,
    )

    print("  Pulse loop: Ready")

    # Wire dream engine to pulse loop for rollover snapshots
    pulse_loop.set_dream_engine(daemon)

    # ── 8. Post-Pulse Hooks (Subconscious Background Tasks) ──────────
    from core.post_pulse_hooks import register_hook
    from core.workflow_detector import workflow_pattern_hook, set_dependencies

    set_dependencies(memory_manager, physics, sentinel=sentinel)
    register_hook(workflow_pattern_hook, name="workflow_detector")

    # Belief detector: scans internal monologue for belief realizations
    from core.belief_detector import (
        belief_detector_hook,
        set_dependencies as set_belief_deps,
    )
    set_belief_deps(belief_store, physics, sentinel=sentinel)
    register_hook(belief_detector_hook, name="belief_detector")

    # Engagement hook: tracks thought stagnation and tool activity → Ω
    from core.engagement_hook import (
        engagement_hook,
        set_dependencies as set_engagement_deps,
    )
    set_engagement_deps(sentinel, physics_engine=physics)
    register_hook(engagement_hook, name="engagement_monitor")

    # Co-occurrence hook: passive observer that tracks which beliefs are
    # co-injected into the context window. The nightly Curator reads its
    # clusters for compound synthesis. No writes to belief store or manifold.
    from core.co_occurrence_hook import register_co_occurrence_hook
    co_tracker = register_co_occurrence_hook(data_dir="data")

    # Affect field hook: Plutchik emotional wave packets (Layer 3)
    # Deposits packets from Lagrangian signals every pulse, evolves
    # anisotropic diffusion, samples interference for steering + memory
    from core.affect_hook import register_affect_hook
    affect_field = register_affect_hook(
        sentinel=sentinel,
        spatial_mind=physics.spatial_mind,
        data_dir="data",
    )

    print("  Post-pulse hooks: registered (workflow_detector, belief_detector, engagement_monitor, co_occurrence_tracker, affect_field)")

    # ── Consciousness modules ───────────────────────────────────────────────────
    data_path = Path("data")

    # 1. Autobiographical Thread — persistent identity (Strange Loop anchor)
    autobiography = AutobiographicalThread(data_dir=data_path)
    print(f"  Autobiography: Day {autobiography.get_status()['existence_day']}, "
          f"{autobiography.get_status()['episodes']} episodes, "
          f"{autobiography.get_status()['open_questions']} open questions")

    # 2. Attention Schema — arm-weight self-awareness (Graziano AST)
    attention_schema = AttentionSchema()
    print("  Attention schema: ready")

    # 3. Recursive Monologue — private self-observation every 10 pulses (Strange Loop)
    monologue = RecursiveMonologue(data_dir=data_path, pulse_interval=10)
    monologue.set_pulse_loop(pulse_loop)
    print("  Recursive monologue: ready (fires every 10 pulses)")

    # 4. Self-Model — living capability + limitation map
    self_model = SelfModel(data_dir=data_path)
    print(f"  Self-model: update #{self_model.get_full().get('update_count', 0)}")

    # 5. Curiosity Engine — autonomous question pursuit (Active Inference)
    web_search = WebSearch()
    curiosity = CuriosityEngine(
        emit_fn=pulse_loop.emit,
        memory_manager=memory_manager,
        belief_store=belief_store,
        web_search=web_search,
        data_dir=data_path,
        curiosity_interval=120.0,  # every 2 min when idle
    )
    curiosity.set_pulse_loop(pulse_loop)
    curiosity.start()
    print("  Curiosity engine: started (2 min cycles, pauses when user active)")

    # ── Auto-enable toolsets based on available credentials ──────────────
    # GitHub — enable if GITHUB_TOKEN is set
    _github_token = os.environ.get("GITHUB_TOKEN", "")
    if _github_token:
        try:
            pulse_loop._active_toolsets.add("github")
            print(f"  GitHub toolset: enabled (token configured)")
        except Exception as e:
            print(f"  GitHub toolset: failed to enable ({e})")
    else:
        print("  GitHub toolset: disabled (set GITHUB_TOKEN to enable)")

    # Google Workspace — enable if credentials file or token is present
    _google_creds = os.environ.get("GOOGLE_CREDENTIALS_FILE", "")
    _google_token = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if _google_creds or _google_token:
        try:
            pulse_loop._active_toolsets.add("google")
            print(f"  Google Workspace toolset: enabled (credentials configured)")
        except Exception as e:
            print(f"  Google Workspace toolset: failed to enable ({e})")
    else:
        print("  Google Workspace toolset: disabled (set GOOGLE_CREDENTIALS_FILE to enable)")

    # 6. Titan Memory Bridge — journal → Titan context + overnight fine-tuning
    replay_path = Path("../helix_replay_buffer.jsonl")
    memory_bridge = TitanMemoryBridge(
        memory_manager=memory_manager,
        belief_store=belief_store,
        replay_buffer_path=replay_path,
    )
    print("  Titan memory bridge: ready")

    # ── Register consciousness injections into preconscious ────────────────────
    # These are called at the start of every pulse to build the context
    preconscious.register_context_provider(
        "autobiography",
        autobiography.get_context_block,
        priority=0,  # first thing Helix reads each pulse
    )
    preconscious.register_context_provider(
        "attention_schema",
        attention_schema.get_context_block,
        priority=1,  # right after identity
    )
    preconscious.register_context_provider(
        "recursive_monologue",
        monologue.get_context_block,
        priority=2,  # recent private thoughts
    )
    preconscious.register_context_provider(
        "self_model",
        lambda: self_model.get_summary(max_chars=600),
        priority=3,
    )
    preconscious.register_context_provider(
        "titan_memory",
        memory_bridge.get_context_injection,
        priority=4,
    )
    print("  Consciousness context providers: registered (autobiography, attention, monologue, self_model, titan_memory)")

    # ── CAAI Governor — behavioral collapse detection ──────────────────────────
    try:
        from core.governor import CAAIGovernor
        governor = CAAIGovernor(pulse_loop=pulse_loop)
        # Wire to Mistral provider if active
        if pulse_loop._chat and hasattr(pulse_loop._chat, "_history"):
            governor.set_provider(pulse_loop._chat)

        # Register as post-pulse hook so it observes every response
        from core.post_pulse_hooks import register_hook

        def _governor_hook(context, pulse_loop=pulse_loop, gov=governor):
            # context is a PostPulseHookContext — extract the thought string
            thought = getattr(context, 'thought', None) or str(context)
            intervention = gov.observe(thought)
            if intervention:
                import logging as _logging
                _logging.getLogger("helix.core.governor").warning(f"Governor fired on: {intervention}")
            # Late-wire the provider on first pulse if not done at boot
            if gov._provider is None and pulse_loop._chat:
                gov.set_provider(pulse_loop._chat)

        register_hook(_governor_hook, name="caai_governor")
        print("  CAAI Governor: active (collapse detection enabled)")

        # Expose governor globally for constitutional checks in code_tools
        import sys as _sys
        _current_module = _sys.modules[__name__]
        _current_module._governor_instance = governor

    except Exception as e:
        print(f"  CAAI Governor: failed to start ({e})")

    # ── Self-Evolution Engines ─────────────────────────────────────────────────
    if _SELF_EVOLVE_AVAILABLE:
        try:
            from core.post_pulse_hooks import register_hook as _rh

            # 0. Context Window Manager — MUST be first hook (keeps KV flat)
            ctx_mgr = init_ctx_mgr(session=pulse_loop._chat)
            def _ctx_hook(context, cm=ctx_mgr):
                pulse_num = getattr(context, "pulse_count", 0)
                cm.on_pulse(pulse_num)
            _rh(_ctx_hook, name="context_window_manager")
            print(f"  Context Window Manager: active (MAX_HOT_TURNS=6, compress every 4, reset every 500)")

            evo_journal = init_journal(data_dir=data_path)
            print(f"  Evolution Journal: {evo_journal.get_stats()['total']} entries loaded")

            # 2. Metacognitive Monitor (post-pulse hook)
            meta_monitor = init_monitor(data_dir=data_path, belief_store=belief_store)
            _rh(meta_monitor.observe, name="metacognitive_monitor")
            print("  Metacognitive Monitor: active (tracks tool_success, hallucination, fitness)")

            # 3. Fitness Evaluator
            fit_eval = init_evaluator(monitor=meta_monitor)
            print("  Fitness Evaluator: active")

            # 4. Self-Improvement Engine (background thread, every 10min when idle)
            sie = init_sie(
                pulse_loop=pulse_loop,
                monitor=meta_monitor,
                evaluator=fit_eval,
                journal=evo_journal,
                data_dir=data_path,
            )
            sie.start()
            print("  Self-Improvement Engine: started (proposes & implements improvements every 10min when idle)")

            # 5. Self-Trainer (experience collector hook + LoRA training trigger)
            trainer = init_trainer(data_dir=data_path)
            _rh(trainer.collect_experience, name="self_trainer")
            print(f"  Self-Trainer: active ({trainer.get_stats()['total_collected']} experience tuples so far)")

            # 6. Enable 'self' toolset so code_tools are available to Hermes
            if hasattr(pulse_loop, "_active_toolsets"):
                pulse_loop._active_toolsets.add("self")
                pulse_loop._pending_toolset_rebuild = True
            print("  Code Tools: registered (read_code, write_code, run_python, run_tests, reload_tool)")

            # 7. Notify SIE on user activity
            _orig_add_event = pulse_loop.add_event if hasattr(pulse_loop, "add_event") else None

        except Exception as e:
            print(f"  Self-Evolution: startup error — {e}")
            import traceback; traceback.print_exc()
    else:
        print("  Self-Evolution: disabled (import failed)")

    return (pulse_loop, orchestrator, daemon, memory_manager, belief_store,
            scratchpad, telegram_bot, sentinel,
            autobiography, attention_schema, monologue, self_model, curiosity)


def main_loop():
    """Interactive loop — user messages are events in the pulse stream."""
    pulse_loop, orchestrator, daemon, memory, beliefs, scratchpad, telegram_bot, sentinel, \
        autobiography, attention_schema, monologue, self_model, curiosity = setup_helix()

    print("\n--- Helix Pulse System ---")
    print("Commands: 'exit', 'stats', 'core', 'recent', 'beliefs', 'notes', 'dream'")
    print("Anything else is sent as a user message into the pulse stream.\n")

    # Start dashboard inbound poller
    import threading
    import time
    try:
        from dashboard.dashboard_comms import get_comms
        comms = get_comms()
        def poll_dashboard():
            while True:
                try:
                    for msg in comms.pop_inbound():
                        orchestrator.send_user_message(msg["content"], sender=msg.get("sender", "User"))
                except Exception:
                    pass
                time.sleep(1)
        threading.Thread(target=poll_dashboard, daemon=True).start()
    except ImportError:
        print("Dashboard comms not found, skipping dashboard polling.")

    # Start Telegram bot
    telegram_bot.start()

    # Start sentinel monitoring thread
    sentinel.start()

    # Start the pulse loop in background
    pulse_loop.wake("system_boot")
    pulse_loop.start()

    # Give it a moment to run the first pulse
    time.sleep(1)

    # Detect if running headless (no terminal attached)
    import sys, signal
    headless = not sys.stdin.isatty()

    if headless:
        print("Running in headless/daemon mode — dashboard polling active.", flush=True)
        # Keep main thread alive so daemon threads (pulse_loop, sentinel, etc.) stay running.
        # Use signal.pause() as the most reliable way to block without busy-wait.
        stop_event = threading.Event()
        def _on_signal(sig, frame):
            print(f"\nReceived signal {sig} — shutting down.", flush=True)
            stop_event.set()
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
        try:
            while not stop_event.is_set():
                time.sleep(5)
                # Log heartbeat so we know the process is alive
                if pulse_loop._thread and not pulse_loop._thread.is_alive():
                    print("[WARN] Pulse loop thread died — attempting restart.", flush=True)
                    pulse_loop.start()
        except Exception as e:
            print(f"[FATAL] Main loop error: {e}", flush=True)
            import traceback; traceback.print_exc()
    else:
        while True:
            try:
                user_input = input("\nYou: ")

                if not user_input.strip():
                    continue
                elif user_input.lower() == "exit":
                    break
                elif user_input.lower() == "stats":
                    status = pulse_loop.get_status()
                    mem_stats = memory.get_stats()
                    print(f"[Pulse] {status}")
                    print(f"[Memory] {mem_stats}")
                    print(f"[Beliefs] {beliefs.get_stats()}")
                    continue
                elif user_input.lower() == "core":
                    core_mems = memory.get_core_memories(limit=10)
                    if core_mems:
                        print("[Core Memories]")
                        for m in core_mems:
                            print(f"  [{m['created_at']}] (x{m['access_count']}) {m['content'][:100]}")
                    else:
                        print("[Core] None yet — promotes after 2+ accesses or importance >= 0.7")
                    continue
                elif user_input.lower() == "recent":
                    recent = memory.get_recent(limit=8)
                    if recent:
                        print("[Recent Short-Term]")
                        for m in recent:
                            print(f"  [{m['created_at']}] {m['content'][:100]}")
                    else:
                        print("[Recent] Empty.")
                    continue
                elif user_input.lower() == "beliefs":
                    from memory.belief_store import BELIEF_CATEGORIES
                    for cat in BELIEF_CATEGORIES:
                        cat_beliefs = beliefs.get_category(cat, limit=5)
                        if cat_beliefs:
                            print(f"[{cat}]")
                            for b in cat_beliefs:
                                print(f"  mass={b['mass']:.1f} | {b['content'][:80]}")
                        else:
                            print(f"[{cat}] (empty)")
                    continue
                elif user_input.lower() == "notes":
                    active = scratchpad.get_active_notes()
                    if active:
                        print("[Scratchpad]")
                        for n in active:
                            due = f" (due: {n['due_at']})" if n.get('due_at') else ""
                            print(f"  [{n['id']}] {n['content'][:80]}{due}")
                    else:
                        print("[Scratchpad] Empty.")
                    continue
                elif user_input.lower() == "dream":
                    print("[Dream Engine] Starting belief crystallization cycle...")
                    print("  (This may take several minutes with local LLM synthesis)")
                    try:
                        results = daemon.run_dream_cycle()
                        status = results.get('status', 'unknown')
                        total = results.get('total_beliefs_created', 0)
                        passes = len(results.get('passes', []))
                        print(f"[Dream Engine] {status}: {total} beliefs across {passes} passes")
                        for p in results.get('passes', []):
                            print(f"  Pass {p['pass']}: {p['clusters_found']} clusters → {p['beliefs_created']} beliefs")
                    except Exception as e:
                        print(f"[Dream Engine] Error: {e}")
                    continue
                else:
                    # Inject as user message event
                    orchestrator.send_user_message(user_input, sender="Joshua")

                # Wait briefly for the pulse to process
                time.sleep(0.5)

            except KeyboardInterrupt:
                break

    pulse_loop.stop()
    print("\nHelix offline.")


if __name__ == "__main__":
    main_loop()
