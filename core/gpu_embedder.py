"""
Helix — GPU Embedding Utility

Thin wrapper around HuggingFace transformers to embed text using
all-MiniLM-L6-v2 on CUDA, with CPU fallback. Replaces the chromadb
DefaultEmbeddingFunction (CPU-only) for a ~3-4x speedup per pulse.

This is a module-level singleton so the model loads once and is reused
across all callers (spatial_mind, physics_engine, preconscious).

Usage:
    from core.gpu_embedder import embed, embed_batch

    vec = embed("some text")          # → np.ndarray shape (384,)
    vecs = embed_batch(["a", "b"])    # → np.ndarray shape (N, 384)
"""

import logging
import threading
import numpy as np

logger = logging.getLogger("helix.core.gpu_embedder")

_lock = threading.Lock()
_tokenizer = None
_model = None
_device = None


def _load():
    """Load the embedding model once. Thread-safe."""
    global _tokenizer, _model, _device
    if _model is not None:
        return True
    try:
        from transformers import AutoTokenizer, AutoModel

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # Force CPU — Hermes-3 (8B 4-bit) needs the full GPU during inference.
        # The mini embedder on CPU is ~300-500ms, which is fine since embedding
        # runs in the async hook thread and doesn't block the pulse loop.
        _device = "cpu"
        logger.info(f"[gpu_embedder] Loading {model_name} on CPU (GPU reserved for Hermes-3)...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name).to(_device)
        _model.eval()
        logger.info(f"[gpu_embedder] ✅ Embedder ready on CPU")
        return True
    except Exception as e:
        logger.warning(f"[gpu_embedder] Load failed: {e}")
        return False


def embed_batch(texts: list) -> np.ndarray:
    """Embed a list of texts. Returns (N, 384) float32 array."""
    with _lock:
        if not _load():
            return np.zeros((len(texts), 384), dtype=np.float32)

    try:
        import torch
        encoded = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(_device) for k, v in encoded.items()}
        with torch.no_grad():
            out = _model(**encoded)
        # Mean pooling over token embeddings
        attn = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.debug(f"[gpu_embedder] embed_batch failed: {e}")
        return np.zeros((len(texts), 384), dtype=np.float32)


def embed(text: str) -> np.ndarray:
    """Embed a single text string. Returns (384,) float32 array."""
    result = embed_batch([text])
    return result[0]


def get_device() -> str:
    """Return the device the embedder is running on."""
    with _lock:
        _load()
    return _device or "cpu"
