#!/usr/bin/env python3
"""
Helix — Cognitive Dashboard

Real-time monitoring dashboard for the Helix cognitive architecture.
Runs alongside main.py, reads files only — never modifies Helix state.

Usage:
    python dashboard/dashboard.py              # default: localhost:5050
    python dashboard/dashboard.py --port 8080  # custom port
"""

import argparse
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Suppress Flask's default request logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

# ── Path Configuration ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.resolve()
LOG_PATH = BASE_DIR / "logs" / "helix.log"
BELIEFS_DIR = BASE_DIR / "data" / "beliefs"
SPATIAL_DIR = BASE_DIR / "data" / "spatial"

# Spatial state files written by SpatialMind.save_state()
BELIEF_STATE = SPATIAL_DIR / "belief_space_state.json"
MEMORY_STATE = SPATIAL_DIR / "memory_space_state.json"
ATTENTION_NPY = SPATIAL_DIR / "attention_center.npy"
ATTENTION_PREV = SPATIAL_DIR / "attention_center_prev.npy"
ATTENTION_GAMMA = SPATIAL_DIR / "attention_center_gamma.npy"

# Log lines matching these patterns are excluded from the thought stream
NOISE_PATTERNS = re.compile(
    r"telegram_bot|comms\.|chromadb|chroma_db|rate_limit|"
    r"429|socket|httpcore|httpx|urllib3|google\.auth|"
    r"werkzeug|PIL\.|fontTools|"
    r"dashboard_comms|/api/messages|dashboard.*poll",
    re.IGNORECASE,
)

# Tab filters for the thought stream
TAB_FILTERS = {
    "thoughts": re.compile(
        r"\[thought\]|Pulse \d+|internal monologue|💭|💬|_pulse\b",
        re.IGNORECASE,
    ),
    "tools": re.compile(
        r"FC tools used|tool_result|tool_call|send_message|function.call|🔧",
        re.IGNORECASE,
    ),
    "beliefs": re.compile(
        r"belief_detector|[Cc]o.occurrence|[Hh]ebbian|[Bb]elief.*added|"
        r"[Bb]elief.*merged|[Ww]ired.*relation|confidence|attrition",
        re.IGNORECASE,
    ),
    "spatial": re.compile(
        r"[Ss]patial|attention|gravity|bootstrap|manifold|entropy|"
        r"identity.center|[Kk][Dd][Tt]ree|drift",
        re.IGNORECASE,
    ),
}

CATEGORY_COLORS = {
    "self_identity": "#FFD700",
    "knowledge": "#4A9EFF",
    "skills": "#4ADE80",
    "capabilities": "#A78BFA",
    "people": "#F472B6",
    "preferences": "#FB923C",
    "feedback": "#94A3B8",
}


# ── Log Tailer ────────────────────────────────────────────────────────

class LogTailer:
    """Tails helix.log and caches recent lines per tab."""

    def __init__(self, path: Path, max_lines: int = 200):
        self._path = path
        self._max = max_lines
        self._offset = 0
        self._lines: Dict[str, deque] = {
            tab: deque(maxlen=max_lines) for tab in TAB_FILTERS
        }
        self._all = deque(maxlen=max_lines)

    def poll(self) -> None:
        if not self._path.exists():
            return
        try:
            size = self._path.stat().st_size
            if size < self._offset:
                self._offset = 0
            with open(self._path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                new_lines = f.readlines()
                self._offset = f.tell()
            for raw in new_lines:
                line = raw.rstrip()
                if not line or NOISE_PATTERNS.search(line):
                    continue
                self._all.append(line)
                for tab, pattern in TAB_FILTERS.items():
                    if pattern.search(line):
                        self._lines[tab].append(line)
        except Exception:
            pass

    def get(self, tab: str = "thoughts", since: int = 0) -> List[str]:
        self.poll()
        buf = self._lines.get(tab, self._all)
        return list(buf)[since:]


# ── Spatial Data Reader ───────────────────────────────────────────────

def _load_npy(path: Path) -> Optional[np.ndarray]:
    try:
        if path.exists():
            return np.load(str(path))
    except Exception:
        pass
    return None


def _pca_3d(positions: np.ndarray) -> np.ndarray:
    """Project Nx8 positions to Nx3 via PCA (numpy only)."""
    if len(positions) < 2:
        return positions[:, :3] if positions.shape[1] >= 3 else positions
    centered = positions - positions.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Top 3 components (eigh returns ascending order)
    top3 = eigenvectors[:, -3:][:, ::-1]
    return centered @ top3


def read_spatial() -> Dict[str, Any]:
    """Read all spatial state files and return JSON-ready dict."""
    points = []
    all_positions = []
    point_meta = []

    # Read belief space
    if BELIEF_STATE.exists():
        try:
            with open(BELIEF_STATE) as f:
                state = json.load(f)
            for pid, data in state.items():
                pos = data.get("position", [])
                if len(pos) >= 3:
                    all_positions.append(pos)
                    point_meta.append({
                        "id": pid,
                        "type": data.get("type", "belief"),
                        "content": (data.get("content", "") or "")[:120],
                        "mass": data.get("confidence", 0.5),
                        "category": "belief",
                    })
        except Exception:
            pass

    # Read memory space
    if MEMORY_STATE.exists():
        try:
            with open(MEMORY_STATE) as f:
                state = json.load(f)
            for pid, data in state.items():
                pos = data.get("position", [])
                if len(pos) >= 3:
                    all_positions.append(pos)
                    point_meta.append({
                        "id": pid,
                        "type": "memory",
                        "content": (data.get("content", "") or "")[:80],
                        "mass": data.get("importance", 0.3),
                        "category": "memory",
                    })
        except Exception:
            pass

    # Also read belief category for coloring
    belief_categories = {}
    if BELIEFS_DIR.exists():
        for cat_file in BELIEFS_DIR.glob("*.json"):
            cat_name = cat_file.stem
            try:
                with open(cat_file) as f:
                    beliefs = json.load(f)
                for b in beliefs:
                    bid = b.get("id", "")
                    if bid:
                        belief_categories[bid] = cat_name
            except Exception:
                pass

    # Project to 3D
    projected = []
    if all_positions:
        pos_array = np.array(all_positions, dtype=np.float32)
        proj_3d = _pca_3d(pos_array)
        for i, meta in enumerate(point_meta):
            cat = belief_categories.get(meta["id"], meta["category"])
            projected.append({
                "x": round(float(proj_3d[i, 0]), 4),
                "y": round(float(proj_3d[i, 1]), 4),
                "z": round(float(proj_3d[i, 2]), 4),
                "id": meta["id"],
                "type": meta["type"],
                "content": meta["content"],
                "mass": round(meta["mass"], 3),
                "category": cat,
                "color": CATEGORY_COLORS.get(cat, "#666666"),
            })

    # Attention center
    attn = {"x": 0, "y": 0, "z": 0}
    attn_prev = {"x": 0, "y": 0, "z": 0}
    identity = {"x": 0, "y": 0, "z": 0}

    if all_positions:
        pos_array = np.array(all_positions, dtype=np.float32)
        mean = pos_array.mean(axis=0)
        centered = pos_array - mean
        cov = np.cov(centered, rowvar=False)
        _, eigvec = np.linalg.eigh(cov)
        top3 = eigvec[:, -3:][:, ::-1]

        ac = _load_npy(ATTENTION_NPY)
        if ac is not None and len(ac) == len(mean):
            proj = (ac - mean) @ top3
            attn = {"x": round(float(proj[0]), 4), "y": round(float(proj[1]), 4), "z": round(float(proj[2]), 4)}

        ap = _load_npy(ATTENTION_PREV)
        if ap is not None and len(ap) == len(mean):
            proj = (ap - mean) @ top3
            attn_prev = {"x": round(float(proj[0]), 4), "y": round(float(proj[1]), 4), "z": round(float(proj[2]), 4)}

    gamma = 0.5
    g = _load_npy(ATTENTION_GAMMA)
    if g is not None:
        gamma = round(float(g[0]), 3)

    return {
        "points": projected,
        "attention": attn,
        "attention_prev": attn_prev,
        "identity": identity,
        "gamma": gamma,
        "point_count": len(projected),
    }


def read_status() -> Dict[str, Any]:
    """Read belief stats and last known gauges from log."""
    stats = {"total": 0, "categories": {}}
    if BELIEFS_DIR.exists():
        for cat_file in BELIEFS_DIR.glob("*.json"):
            try:
                with open(cat_file) as f:
                    beliefs = json.load(f)
                cat = cat_file.stem
                stats["categories"][cat] = len(beliefs)
                stats["total"] += len(beliefs)
            except Exception:
                pass

    gamma = 0.5
    g = _load_npy(ATTENTION_GAMMA)
    if g is not None:
        gamma = round(float(g[0]), 3)

    # Priority 1: pulse_count.txt — written on EVERY pulse, always accurate.
    # status.json is only written every 5-10 pulses so it always lags.
    pulse = 0
    PULSE_COUNT_FILE = BASE_DIR / "data" / "pulse_count.txt"
    if PULSE_COUNT_FILE.exists():
        try:
            pulse = int(PULSE_COUNT_FILE.read_text().strip())
        except Exception:
            pass

    # Priority 2: status.json for state, pid, and ts (not pulse).
    omega = 0.5
    state = "UNKNOWN"
    ts = 0
    pid = 0
    STATUS_FILE = BASE_DIR / "data" / "status.json"
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                heartbeat = json.load(f)
            state = heartbeat.get("state", "UNKNOWN")
            if pulse == 0:
                pulse = heartbeat.get("pulse", 0)  # fallback only
            ts    = heartbeat.get("ts", 0)
            pid   = heartbeat.get("pid", 0)
        except Exception:
            pass

    # Priority 2: log tail for omega (and pulse/state fallback if file missing).
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 200_000))  # last ~200KB
                tail = f.read().decode("utf-8", errors="replace")
            for line in tail.split("\n"):
                m = re.search(r"Ω=([0-9.]+)", line)
                if m:
                    omega = float(m.group(1))
                if pulse == 0:
                    m = re.search(r"Pulse (\d+)", line)
                    if m:
                        pulse = int(m.group(1))
                if state == "UNKNOWN":
                    m = re.search(r"→ (DORMANT|RESTING|REGULAR|ACTIVE)", line)
                    if m:
                        state = m.group(1)
                    if "state=" in line.lower():
                        m2 = re.search(
                            r"state['\"]?\s*[:=]\s*['\"]?(DORMANT|RESTING|REGULAR|ACTIVE)",
                            line, re.I,
                        )
                        if m2:
                            state = m2.group(1).upper()
        except Exception:
            pass

    # Read TPS from inference_stats.json written by jamba_tool_provider
    tps = 0.0
    tps_last = 0.0
    STATS_FILE = BASE_DIR / "data" / "inference_stats.json"
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE) as f:
                stats = json.load(f)
            tps      = stats.get("tps", 0.0)
            tps_last = stats.get("tps_last", 0.0)
        except Exception:
            pass

    submissions_count = 0
    try:
        sol_dir = BASE_DIR / "solutions"
        for sub in ["submitted", "accepted", "rejected"]:
            d = sol_dir / sub
            if d.exists():
                submissions_count += len([x for x in d.iterdir() if x.is_dir()])
    except Exception:
        pass

    active_bounty = None
    try:
        active_dir = BASE_DIR / "solutions" / "active"
        if active_dir.exists():
            newest_json = None
            newest_time = 0
            for d in active_dir.iterdir():
                if d.is_dir():
                    idx = d / "index.json"
                    if idx.exists():
                        mtime = idx.stat().st_mtime
                        if mtime > newest_time:
                            newest_time = mtime
                            newest_json = idx
            if newest_json:
                with open(newest_json) as f:
                    active_bounty = json.load(f)
    except Exception:
        pass

    return {
        "beliefs": stats,
        "omega": omega,
        "gamma": gamma,
        "pulse": pulse,
        "state": state,
        "ts": ts,
        "pid": pid,
        "tps": tps,
        "tps_last": tps_last,
        "active_bounty": active_bounty,
        "submissions": submissions_count,
    }


# ── Flask App ─────────────────────────────────────────────────────────

def create_app():
    import sys
    from flask import Flask, jsonify, request, send_from_directory
    # Support both: python dashboard/dashboard.py (standalone) and import as package
    try:
        from dashboard.dashboard_comms import get_comms
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).parent))
        from dashboard_comms import get_comms

    app = Flask(__name__, static_folder=None)
    tailer = LogTailer(LOG_PATH)
    comms = get_comms()

    @app.route("/")
    def index():
        html_path = Path(__file__).parent / "dashboard_ui.html"
        return html_path.read_text(encoding="utf-8")

    @app.route("/api/logs")
    def api_logs():
        tab = request.args.get("tab", "thoughts")
        since = int(request.args.get("since", 0))
        lines = tailer.get(tab, since)
        return jsonify({"lines": lines, "total": since + len(lines)})

    @app.route("/api/spatial")
    def api_spatial():
        return jsonify(read_spatial())

    @app.route("/api/status")
    def api_status():
        return jsonify(read_status())

    # ── Chat Endpoints ────────────────────────────────────────────

    @app.route("/api/messages", methods=["POST"])
    def api_send_message():
        """Browser sends a message to Helix."""
        data = request.get_json(force=True, silent=True) or {}
        sender = (data.get("sender") or "User").strip()
        # Accept either 'content' (legacy) or 'message' (dashboard chat panel)
        content = (data.get("content") or data.get("message") or "").strip()
        if not content:
            return jsonify({"error": "No content"}), 400
        comms.push_inbound(sender, content)
        return jsonify({"ok": True})

    @app.route("/api/messages/pending")
    def api_pending_messages():
        """Pulse loop poller consumes inbound messages."""
        pending = comms.pop_inbound()
        return jsonify({"messages": pending})

    @app.route("/api/messages/outbound")
    def api_outbound_messages():
        """Browser polls for Helix's replies."""
        since = int(request.args.get("since", 0))
        messages = comms.get_outbound(since)
        total = comms.get_outbound_count()
        return jsonify({"messages": messages, "total": total})

    @app.route("/api/activations")
    def api_activations():
        """Real-time cognitive activation data for the live visualization panel."""
        data = {
            "fitness_history": [],   # last N composite_fitness values
            "tool_call_rate": 0.0,
            "tool_success_rate": 1.0,
            "novel_belief_rate": 0.0,
            "hallucination_rate": 0.0,
            "last_tool": "",
            "last_belief": "",
            "pulse_count": 0,
            "active_components": {},  # component → activity level 0-1
        }
        # Read meta_snapshots for fitness history & rates
        snap_path = BASE_DIR / "data" / "meta_snapshots.jsonl"
        if snap_path.exists():
            try:
                snaps = []
                with snap_path.open() as f:
                    for line in f:
                        try: snaps.append(json.loads(line.strip()))
                        except: pass
                # last 30 snapshots for the waveform
                recent = snaps[-30:]
                data["fitness_history"] = [round(s.get("composite_fitness", 0), 4) for s in recent]
                if snaps:
                    last = snaps[-1]
                    data["tool_call_rate"]     = last.get("tool_call_rate", 0)
                    data["tool_success_rate"]  = last.get("tool_success_rate", 1)
                    data["novel_belief_rate"]  = min(1.0, last.get("novel_belief_rate", 0) * 10)
                    data["hallucination_rate"] = last.get("hallucination_rate", 0)
            except Exception:
                pass

        # Parse recent log lines for last tool + belief + pulse count
        if LOG_PATH.exists():
            try:
                with open(LOG_PATH, "rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 30000))
                    tail = f.read().decode("utf-8", errors="replace")
                for line in reversed(tail.split("\n")):
                    if not data["last_tool"] and "FC tools used:" in line:
                        m = re.search(r"FC tools used: \[(.+?)\]", line)
                        if m: data["last_tool"] = m.group(1).replace("'", "").split(",")[0].strip()
                    if not data["last_belief"] and "Belief added" in line:
                        m = re.search(r"Belief added[^:]*: (.{0,60})", line)
                        if m: data["last_belief"] = m.group(1).strip()
                    if not data["pulse_count"]:
                        m = re.search(r"Pulse (\d+)", line)
                        if m: data["pulse_count"] = int(m.group(1))
                    if data["last_tool"] and data["pulse_count"]:
                        break
            except Exception:
                pass

        # Synthetic component activity from rates
        data["active_components"] = {
            "Pulse Loop":    min(1.0, 0.9 if data["pulse_count"] > 0 else 0),
            "Tool Executor": data["tool_call_rate"],
            "Belief Store":  data["novel_belief_rate"],
            "Curiosity":     0.7,  # always running
            "Memory":        0.6,
            "SIE":           0.4,
        }
        return jsonify(data)

    @app.route("/api/neural")
    def api_neural():
        """Per-layer activation snapshot for the 3D brain visualization."""
        try:
            snap_path = BASE_DIR / "data" / "neural_activations.json"
            if snap_path.exists():
                data = json.loads(snap_path.read_text())
                return jsonify(data)
        except Exception:
            pass
        # Return flat defaults (32 layers, all quiet) if not yet available
        return jsonify({
            "layers": [0.0] * 32,
            "attn_heads": [0.0] * 32,
            "ts": 0,
            "pulse": 0,
            "token_count": 0,
            "n_layers": 32,
        })

    @app.route("/api/tuples")
    def api_tuples():
        """LoRA training dataset progress — clean tuple count, tool distribution, THINK samples."""
        TUPLES_PATH = BASE_DIR / "data" / "experience_tuples.jsonl"
        THRESHOLD = 500
        data = {
            "clean": 0, "total": 0, "legacy": 0, "threshold": THRESHOLD,
            "pct": 0.0, "eta_hours": None,
            "tool_dist": {}, "unique_tools": 0,
            "top_tool_pct": 0.0,
            "diversity_ok": False, "concentration_ok": False,
            "lora_gen": 0,
        }
        if TUPLES_PATH.exists():
            try:
                from collections import Counter as _C
                tool_counts = _C()
                clean, total = 0, 0
                for line in TUPLES_PATH.open():
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        d = json.loads(line)
                        if d.get("think_block", "").strip():
                            clean += 1
                            tn = d.get("tool_name", "")
                            if tn:
                                tool_counts[tn] += 1
                    except Exception:
                        pass
                data["clean"]   = clean
                data["total"]   = total
                data["legacy"]  = total - clean
                data["pct"]     = round(clean / THRESHOLD * 100, 1)
                data["unique_tools"] = len(tool_counts)
                data["diversity_ok"] = len(tool_counts) >= 5
                if tool_counts and clean > 0:
                    top_n = tool_counts.most_common(1)[0][1]
                    top_pct = top_n / clean
                    data["top_tool_pct"] = round(top_pct * 100, 1)
                    data["concentration_ok"] = top_pct <= 0.70
                data["tool_dist"] = dict(tool_counts.most_common(8))
                if clean > 5:
                    data["eta_hours"] = round((THRESHOLD - clean) / 30, 1)
            except Exception:
                pass
        adapter_txt = BASE_DIR / "data" / "current_adapter.txt"
        adapters_dir = BASE_DIR / "data" / "lora_adapters"
        if adapter_txt.exists():
            data["lora_gen"] = 1
        elif adapters_dir.exists():
            data["lora_gen"] = len([x for x in adapters_dir.iterdir() if x.is_dir()])
        return jsonify(data)

    @app.route("/api/think")
    def api_think():
        """Last 5 HERMES THINK blocks from the log."""
        result = []
        if LOG_PATH.exists():
            try:
                with open(LOG_PATH, "rb") as f:
                    f.seek(max(0, f.seek(0, 2) - 80_000))
                    tail = f.read().decode("utf-8", errors="replace")
                for line in tail.split("\n"):
                    m = re.search(r"HERMES THINK: '(.+)'$", line)
                    if m:
                        think = m.group(1).replace("\\n", " ").strip()
                        ts_m = re.search(r"(\d{2}:\d{2}:\d{2})", line)
                        ts = ts_m.group(1) if ts_m else ""
                        result.append({
                            "ts": ts,
                            "text": think[:200],
                            "is_json": think.strip().startswith("{"),
                        })
                result = result[-5:]
            except Exception:
                pass
        return jsonify({"thinks": result})

    return app



def main():
    parser = argparse.ArgumentParser(description="Helix Cognitive Dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Port (default: 5050)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"╔══════════════════════════════════════════╗")
    print(f"║     HELIX COGNITIVE DASHBOARD            ║")
    print(f"║     http://{args.host}:{args.port}             ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"  Log: {LOG_PATH}")
    print(f"  Beliefs: {BELIEFS_DIR}")
    print(f"  Spatial: {SPATIAL_DIR}")
    print()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
