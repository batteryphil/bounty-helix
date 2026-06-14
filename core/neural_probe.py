"""
Helix — Neural Probe

Attaches PyTorch forward hooks to each transformer layer in the Hermes-3
(LLaMA-3.1 8B) model and captures per-layer activation statistics after
every inference pass.

Output: data/neural_activations.json — updated after each generate() call.
{
  "layers": [0.43, 0.71, ...],   # 32 floats, mean-abs activation per layer
  "attn_heads": [0.5, ...],      # 32 floats, attention head entropy proxy
  "ts": 1718000000.0,            # unix timestamp of last update
  "pulse": 42,                   # pulse number that triggered this
  "token_count": 124             # tokens generated
}

The values are normalized to [0, 1] over the session's running max so the
visualization is relative (not absolute), which is visually stable.
"""

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("helix.core.neural_probe")

_OUTPUT_PATH = Path("data/neural_activations.json")
_lock = threading.Lock()

# Running max per layer for normalization (keeps viz stable across sessions)
_layer_max: List[float] = []
_hooks = []
_registered = False
_current_activations: List[float] = []
_current_attn: List[float] = []


def _safe_norm(val: float, layer_idx: int) -> float:
    """Normalize val against running max for this layer."""
    global _layer_max
    while len(_layer_max) <= layer_idx:
        _layer_max.append(1e-6)
    if val > _layer_max[layer_idx]:
        _layer_max[layer_idx] = val
    m = _layer_max[layer_idx]
    return min(1.0, val / m) if m > 0 else 0.0


def attach_hooks(model) -> int:
    """
    Attach forward hooks to every transformer layer in the model.
    Returns the number of layers hooked.

    Works with LLaMA-based models (Hermes-3, Mistral, etc.) where
    layers are at model.model.layers.
    """
    global _registered, _hooks, _current_activations, _current_attn

    if _registered:
        return len(_hooks)

    layers = None
    # LLaMA / Mistral / Hermes architecture
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    # Falcon-Mamba or other flat architectures
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    elif hasattr(model, "layers"):
        layers = model.layers

    if layers is None:
        logger.warning("[neural_probe] Could not find transformer layers — probe disabled")
        return 0

    n = len(layers)
    _current_activations = [0.0] * n
    _current_attn = [0.0] * n
    logger.info(f"[neural_probe] Attaching hooks to {n} transformer layers")

    def _make_layer_hook(idx):
        def hook(module, input, output):
            try:
                import torch
                # output is typically (hidden_states, ...) or just hidden_states
                hs = output[0] if isinstance(output, tuple) else output
                if hs is not None and hasattr(hs, "float"):
                    # Mean absolute activation across all tokens and hidden dims
                    act = hs.detach().float().abs().mean().item()
                    if not math.isnan(act) and not math.isinf(act):
                        _current_activations[idx] = _safe_norm(act, idx)

                # Attention entropy proxy — if attention weights are available
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1]
                    if attn is not None and hasattr(attn, "float"):
                        # Entropy of attention distribution = activity measure
                        a = attn.detach().float()
                        a = a + 1e-9
                        entropy = -(a * a.log()).sum(dim=-1).mean().item()
                        norm_e = min(1.0, entropy / 4.0)  # ~4 nats max
                        if not math.isnan(norm_e):
                            _current_attn[idx] = norm_e
            except Exception:
                pass
        return hook

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(_make_layer_hook(i))
        _hooks.append(h)

    _registered = True
    logger.info(f"[neural_probe] ✅ {n} hooks registered — brain visualization active")
    return n


def flush(pulse: int = 0, token_count: int = 0):
    """
    Write current activation snapshot to disk.
    Called after each generate() call.
    """
    if not _registered or not _current_activations:
        return

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "layers": [round(v, 4) for v in _current_activations],
        "attn_heads": [round(v, 4) for v in _current_attn],
        "ts": time.time(),
        "pulse": pulse,
        "token_count": token_count,
        "n_layers": len(_current_activations),
    }

    with _lock:
        try:
            _OUTPUT_PATH.write_text(json.dumps(snapshot))
        except Exception as e:
            logger.debug(f"[neural_probe] flush error: {e}")


def get_latest() -> Optional[dict]:
    """Read and return the latest activation snapshot from disk."""
    try:
        if _OUTPUT_PATH.exists():
            return json.loads(_OUTPUT_PATH.read_text())
    except Exception:
        pass
    return None


def remove_hooks():
    """Detach all hooks (call on shutdown)."""
    global _registered, _hooks
    for h in _hooks:
        try: h.remove()
        except: pass
    _hooks.clear()
    _registered = False
