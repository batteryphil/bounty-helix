"""
Helix — Base Chat Session Interface

All LLM providers implement this interface. The pulse loop only knows
about ChatSession — it never imports provider-specific code directly.

To add a new provider:
    1. Create a new file in llm/providers/
    2. Implement a class that extends ChatSession
    3. Register it in get_provider() below
"""

import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger("helix.llm.providers.base")


class ChatSession(ABC):
    """Abstract chat session — the only interface the pulse loop sees."""

    @abstractmethod
    def send_message(self, message: str) -> str:
        """Send a user-turn message and return the assistant response."""
        ...

    @abstractmethod
    def get_history_size(self) -> int:
        """Return approximate character count of all messages in the session."""
        ...


class ProviderConfig:
    """Configuration for a specific LLM provider.

    Each provider's config is a simple dataclass. New providers just
    add their own fields. The pulse loop reads provider-agnostic
    fields (model, context_window) and passes the rest through.
    """

    def __init__(
        self,
        provider_type: str,          # "ollama", "llama_cpp", "gemini", "anthropic"
        model: str,                  # Model name or path
        context_window: int = 128_000,
        temperature: float = 0.8,
        max_output_tokens: int = 2048,
        options: Optional[Dict[str, Any]] = None,
    ):
        self.provider_type = provider_type
        self.model = model
        self.context_window = context_window
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.options = options or {}


def create_session(
    config: ProviderConfig,
    system_instruction: str,
    tool_declarations: list = None,
    tool_executor=None,
    preconscious=None,
) -> ChatSession:
    """Factory: create a ChatSession from a ProviderConfig.

    This is the ONLY place provider-specific imports happen.
    Adding a new provider = adding an elif branch here.

    Args:
        config: Provider configuration.
        system_instruction: System prompt text.
        tool_declarations: Optional Gemini FunctionDeclaration dicts (Gemini only).
        tool_executor: Optional ToolExecutor for function call handling (Gemini only).
        preconscious: Optional Preconscious for belief enrichment on tool returns.
    """
    if config.provider_type == "falcon_h1":
        from llm.providers.falcon_h1_provider import FalconToolSession
        return FalconToolSession(
            system_prompt=system_instruction,
        )

    if config.provider_type == "jamba":
        from llm.providers.jamba_tool_provider import JambaToolSession
        return JambaToolSession(
            system_instruction=system_instruction,
            tool_declarations=tool_declarations,
            tool_executor=tool_executor,
        )

    if config.provider_type == "hermes_tool":
        from llm.providers.hermes_tool_provider import HermesToolSession
        return HermesToolSession(
            system_instruction=system_instruction,
            tool_declarations=tool_declarations,
            tool_executor=tool_executor,
        )

    if config.provider_type == "qwen_gguf":
        from llm.providers.qwen_gguf_provider import QwenGGUFSession
        return QwenGGUFSession(
            system_instruction=system_instruction,
            tool_declarations=tool_declarations,
            tool_executor=tool_executor,
            temperature=config.temperature,
            model_path=config.model,
        )

    if config.provider_type == "mistral_tool":
        from llm.providers.mistral_tool_provider import MistralToolSession
        return MistralToolSession(
            system_instruction=system_instruction,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            tool_declarations=tool_declarations,
            tool_executor=tool_executor,
        )

    elif config.provider_type == "falcon_mamba":
        from llm.providers.falcon_mamba_provider import FalconMambaSession
        return FalconMambaSession(
            system_instruction=system_instruction,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )

    elif config.provider_type == "titan":
        from llm.providers.titan_provider import TitanSession
        return TitanSession(
            system_instruction=system_instruction,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            enable_deep_think=config.options.get("deep_think", False),
            checkpoint=config.model,
        )

    elif config.provider_type == "gemini":
        from llm.providers.gemini_provider import GeminiSession
        return GeminiSession(
            model=config.model,
            system_instruction=system_instruction,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            tool_declarations=tool_declarations,
            tool_executor=tool_executor,
            preconscious=preconscious,
        )

    elif config.provider_type == "ollama":
        from llm.providers.ollama_provider import OllamaSession
        return OllamaSession(
            model=config.model,
            system_instruction=system_instruction,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            options=config.options,
        )

    elif config.provider_type == "llama_cpp":
        from llm.providers.llama_cpp_provider import LlamaCppSession
        return LlamaCppSession(
            model_path=config.model,
            system_instruction=system_instruction,
            n_ctx=config.context_window,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            n_gpu_layers=config.options.get("n_gpu_layers", -1),
        )

    else:
        raise ValueError(
            f"Unknown provider type: {config.provider_type}. "
            f"Supported: falcon_h1, jamba, hermes_tool, falcon_mamba, titan, gemini, ollama, llama_cpp"
        )


def detect_available_provider() -> Optional[ProviderConfig]:
    """Auto-detect the best available LLM backend.

    Priority: Falcon-Mamba-7B (highest, pure Mamba 4-bit) > Titan > Gemini > Ollama > llama.cpp

    Falcon-Mamba-7B is the primary conscious mind — pure SSM, no KV cache,
    4.5GB VRAM, instruction-tuned, coherent output.
    """
    import os
    import glob

    _here    = os.path.dirname(os.path.abspath(__file__))
    _project = os.path.normpath(os.path.join(_here, "..", "..", ".."))

    # 0. Qwen2.5-7B-Instruct GGUF — #1 priority.
    #    Native tool calling, runs completely on CPU via llama.cpp
    qwen_gguf_path = os.path.join(_project, "hf_cache", "qwen2.5-7b", "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf")
    if os.path.exists(qwen_gguf_path):
        logger.info(f"Auto-detected Qwen2.5-7B-Instruct GGUF at {qwen_gguf_path} — CPU inference active")
        return ProviderConfig(
            provider_type="qwen_gguf",
            model=qwen_gguf_path,
            context_window=8192,
            temperature=0.7,
            max_output_tokens=512,
        )

    # 1. Hermes-3-Llama-3.1-8B — #2 priority.
    #    Best agentic tool calling, 4-bit NF4, GPU-resident ~5GB VRAM.
    #    NOTE: Falcon-H1-7B is downloaded but incompatible with RTX 3060 —
    #    its Mamba selective scan creates 3GB+ intermediate tensors that OOM.
    for hermes_base in [
        "/data/hf_cache/hf_cache",                          # old 3TB volume path
        os.path.join(_project, "hf_cache"),                  # project-local cache
        os.path.expanduser("~/.cache/huggingface/hub"),      # HF default cache
    ]:
        hermes_cache = os.path.join(hermes_base, "models--NousResearch--Hermes-3-Llama-3.1-8B")
        if os.path.isdir(hermes_cache):
            hermes_weights = glob.glob(
                os.path.join(hermes_cache, "**", "*.safetensors"), recursive=True
            )
            if hermes_weights:
                logger.info(
                    f"Auto-detected Hermes-3-Llama-3.1-8B at {hermes_base} — "
                    "4-bit NF4, GPU-resident, native <tool_call> calling"
                )
                return ProviderConfig(
                    provider_type="hermes_tool",
                    model="NousResearch/Hermes-3-Llama-3.1-8B",
                    context_window=8192,
                    temperature=0.7,
                    max_output_tokens=512,
                    options={"cache_dir": hermes_base},
                )

    # 1. AI21-Jamba-1.5-Mini — second priority (256K context, CPU inference)
    jamba_cache = os.path.join(_project, "hf_cache",
                               "models--ai21labs--AI21-Jamba-1.5-Mini")
    if os.path.isdir(jamba_cache):
        jamba_weights = glob.glob(
            os.path.join(jamba_cache, "**", "*.safetensors"), recursive=True
        )
        if jamba_weights:
            logger.info(
                "Auto-detected AI21-Jamba-1.5-Mini — hybrid Mamba/Transformer MoE, "
                "native tool calling, 256K context, CPU inference"
            )
            return ProviderConfig(
                provider_type="jamba",
                model="ai21labs/AI21-Jamba-1.5-Mini",
                context_window=262144,
                temperature=0.7,
                max_output_tokens=512,
            )
        else:
            logger.info("Jamba cache dir exists but weights not yet downloaded — skipping")
    if os.path.isdir(hermes_cache):
        hermes_weights = glob.glob(os.path.join(hermes_cache, "**", "*.safetensors"), recursive=True)
        if hermes_weights:
            logger.info(
                "Auto-detected Hermes-3-Llama-3.1-8B (4-bit NF4) — "
                "optimized for agentic function calling, ~5.0GB VRAM"
            )
            return ProviderConfig(
                provider_type="hermes_tool",
                model="NousResearch/Hermes-3-Llama-3.1-8B",
                context_window=8192,
                temperature=0.7,
                max_output_tokens=512,
            )
        else:
            logger.info("Hermes-3 cache dir exists but weights not yet downloaded — skipping")

    # 1. Mistral-7B-Instruct-v0.3 — tool-calling capable
    mistral_cache = os.path.join(_project, "hf_cache",
                                 "models--mistralai--Mistral-7B-Instruct-v0.3")
    if os.path.isdir(mistral_cache):
        logger.info(
            "Auto-detected Mistral-7B-Instruct-v0.3 (4-bit NF4) — "
            "tool-calling capable, ~4.1GB VRAM"
        )
        return ProviderConfig(
            provider_type="mistral_tool",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            context_window=32768,
            temperature=0.7,
            max_output_tokens=512,
        )

    # 1. Falcon-Mamba-7B — conversation-only fallback (no tool calls)
    falcon_cache = os.path.join(_project, "hf_cache",
                                "models--tiiuae--falcon-mamba-7b-instruct")
    if os.path.isdir(falcon_cache):
        logger.info(
            "Auto-detected Falcon-Mamba-7B-Instruct (4-bit NF4) — "
            "pure Mamba, zero KV cache, ~4.5GB VRAM (no tool calling)"
        )
        return ProviderConfig(
            provider_type="falcon_mamba",
            model="tiiuae/falcon-mamba-7b-instruct",
            context_window=8192,
            temperature=0.7,
            max_output_tokens=512,
        )

    # 1. Titan MIMO — fallback local model
    titan_ckpt = os.path.join(_project, "legacy_1.4b_project", "titan_checkpoints", "phase_sft20_best.pt")
    if os.path.exists(titan_ckpt):
        logger.info(
            f"Auto-detected Titan 1.4B MIMO checkpoint at {titan_ckpt} — "
            "using Titan as primary conscious mind (100% local)"
        )
        return ProviderConfig(
            provider_type="titan",
            model=titan_ckpt,
            context_window=4096,
            temperature=0.85,
            max_output_tokens=512,
        )

    # 1. Gemini API — primary conscious mind
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        logger.info("Auto-detected Gemini API key — using gemini-3-flash-preview")
        return ProviderConfig(
            provider_type="gemini",
            model="gemini-3-flash-preview",
            context_window=1_000_000,
            temperature=0.8,
            max_output_tokens=8192,
        )

    # 2. Ollama fallback (local models)
    try:
        import ollama
        models = ollama.list()
        model_names = [m.model for m in models.models]

        preferred = [
            "granite4.1:8b",
            "granite4.1:3b",
        ]
        for pref in preferred:
            if pref in model_names:
                logger.info(f"Auto-detected Ollama with {pref} (Gemini key not found)")
                return ProviderConfig(
                    provider_type="ollama",
                    model=pref,
                    context_window=64_000,
                    options={"num_ctx": 64_000},
                )

        if model_names:
            first = model_names[0]
            logger.info(f"Auto-detected Ollama with {first} (fallback)")
            return ProviderConfig(
                provider_type="ollama",
                model=first,
                context_window=64_000,
                options={"num_ctx": 64_000},
            )
    except Exception:
        pass

    # 3. llama-cpp-python fallback
    try:
        import llama_cpp
        model_path = (
            "/home/nemo/.ollama/models/blobs/"
            "sha256-afb54ad43a39f947407f5cabc59856348d70e072baa5c62d436332157c151bcd"
        )
        if os.path.exists(model_path):
            gpu = "Vulkan" if llama_cpp.llama_supports_gpu_offload() else "CPU"
            logger.info(f"Auto-detected llama.cpp ({gpu})")
            return ProviderConfig(
                provider_type="llama_cpp",
                model=model_path,
                context_window=64_000,
                options={"n_gpu_layers": -1},
            )
    except ImportError:
        pass

    logger.warning("No LLM backend detected")
    return None
