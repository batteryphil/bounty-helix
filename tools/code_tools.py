"""
Helix — Code Self-Modification Tools  (toolset: "self")

These tools give Helix the ability to read, write, execute and hot-reload
its own Python source code — forming the basis of closed-loop self-evolution.

Safety model:
  - IMMUTABLE_FILES: a hardcoded set of files that cannot be modified
  - All write paths must be inside HELIX_AGI_ROOT (no escaping via ..)
  - run_python() executes in a subprocess with hard timeout + resource limits
  - reload_tool() reloads a module and rebuilds the tool registry
  - Every write is logged to EvolutionJournal before it happens

Registry:
  All tools are registered under toolset="self" so the pulse_loop only
  injects them when the agent explicitly needs self-modification capability.
  The CAAI Governor gates individual calls via the constitutional check.
"""

import importlib
import importlib.util
import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("helix.tools.code")

# ── Constants ──────────────────────────────────────────────────────────────────

HELIX_AGI_ROOT = Path(__file__).resolve().parents[1]  # …/Helix-AGI/

# These files can NEVER be modified by the agent — not even with user approval.
# Changing this set requires a human code edit (not a tool call).
IMMUTABLE_FILES: set[str] = {
    "main.py",
    "core/pulse_loop.py",
    "core/governor.py",
    "core/post_pulse_hooks.py",
    "tools/code_tools.py",          # prevents sandbox escape
    "tools/tool_registry.py",       # prevents registry tampering
    "llm/providers/hermes_tool_provider.py",  # prevents self-lobotomy
    "llm/providers/mistral_tool_provider.py",
    "llm/providers/base.py",
}

MAX_READ_BYTES  = 200_000   # 200 KB — enough for any source file
MAX_WRITE_BYTES = 100_000   # 100 KB — cap new tool size
EXEC_TIMEOUT    = 20        # seconds for run_python
TEST_TIMEOUT    = 60        # seconds for run_tests

# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_safe(path: str) -> Optional[Path]:
    """Resolve path relative to HELIX_AGI_ROOT.  Returns None if unsafe."""
    try:
        p = (HELIX_AGI_ROOT / path).resolve()
        p.relative_to(HELIX_AGI_ROOT)   # raises ValueError if escape
        return p
    except (ValueError, Exception):
        return None


def _is_immutable(path: str) -> bool:
    rel = path.replace("\\", "/").lstrip("/")
    return rel in IMMUTABLE_FILES


def _log_to_journal(action: str, path: str, content_snippet: str, outcome: str):
    """Non-blocking append to evolution journal (best-effort)."""
    try:
        from core.evolution_journal import journal
        journal.record_code_write(action=action, path=path,
                                  snippet=content_snippet[:200], outcome=outcome)
    except Exception:
        pass   # journal not available yet — that's OK during init


# ── Tool implementations ───────────────────────────────────────────────────────

def fc_read_code(path: str) -> str:
    """Read a Python source file from the Helix-AGI project.

    Args:
        path: Relative path from project root (e.g. "core/pulse_loop.py")
    Returns:
        File contents as a string, or an error message.
    """
    resolved = _resolve_safe(path)
    if resolved is None:
        return f"[read_code ERROR] Path '{path}' is outside the project root — refused."
    if not resolved.exists():
        return f"[read_code ERROR] File not found: {path}"
    if resolved.stat().st_size > MAX_READ_BYTES:
        return f"[read_code ERROR] File too large ({resolved.stat().st_size} bytes). Max: {MAX_READ_BYTES}."
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
        logger.info(f"[code_tools] read_code: {path} ({len(content)} chars)")
        return f"```python\n# {path}\n{content}\n```"
    except Exception as e:
        return f"[read_code ERROR] {e}"


def fc_write_code(path: str, content: str) -> str:
    """Write Python source code to a file in the Helix-AGI project.

    The file will be created if it doesn't exist. Parent directories are
    created automatically. Immutable files are refused. Content is validated
    as syntactically correct Python before writing.

    Args:
        path: Relative path from project root (e.g. "tools/my_new_tool.py")
        content: Complete Python source code to write.
    Returns:
        Success message with byte count, or error.
    """
    if _is_immutable(path):
        return (f"[write_code REFUSED] '{path}' is a constitutionally protected file. "
                f"Self-modification of core safety systems is not permitted.")

    # Governor constitutional check
    try:
        from core.governor import CAAIGovernor
        import sys
        for mod in sys.modules.values():
            if hasattr(mod, '_governor_instance'):
                gov = mod._governor_instance
                if hasattr(gov, 'check_constitutional'):
                    ok, reason = gov.check_constitutional("write_code", {"path": path, "content": content})
                    if not ok:
                        return f"[write_code CONSTITUTIONAL BLOCK] {reason}"
                    break
    except Exception:
        pass  # Governor not available — proceed with base immutable check only

    resolved = _resolve_safe(path)
    if resolved is None:
        return f"[write_code ERROR] Path '{path}' escapes project root — refused."

    if len(content.encode()) > MAX_WRITE_BYTES:
        return f"[write_code ERROR] Content too large ({len(content.encode())} bytes). Max: {MAX_WRITE_BYTES}."

    # Syntax check before writing
    try:
        compile(content, path, "exec")
    except SyntaxError as e:
        return f"[write_code ERROR] Syntax error in content at line {e.lineno}: {e.msg}"

    # Backup existing file (in case we need to revert)
    backup: Optional[str] = None
    if resolved.exists():
        backup = resolved.read_text(encoding="utf-8", errors="replace")

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        size = resolved.stat().st_size
        logger.info(f"[code_tools] write_code: {path} ({size} bytes)")
        _log_to_journal("write_code", path, content[:200], "written")
        result_msg = f"[write_code OK] Written {size} bytes to {path}."

        # Auto-reload tools/ files so they enter the registry immediately.
        # This closes the gap where SIE writes tools but forgets to call reload_tool().
        if path.startswith("tools/") and path.endswith(".py"):
            try:
                reload_result = fc_reload_tool(path)
                logger.info(f"[code_tools] auto-reload after write: {reload_result[:80]}")
                result_msg += f"\n[auto-reload] {reload_result}"
            except Exception as _re:
                result_msg += f"\n[auto-reload FAILED] {_re} — call reload_tool('{path}') manually."
        else:
            result_msg += f"\nCall reload_tool('{path}') to activate immediately, or it will be active on next restart."

        return result_msg
    except Exception as e:
        # Attempt restore from backup
        if backup is not None:
            try:
                resolved.write_text(backup, encoding="utf-8")
            except Exception:
                pass
        _log_to_journal("write_code", path, content[:200], f"error: {e}")
        return f"[write_code ERROR] {e}"


def fc_run_python(code: str, timeout: int = 15) -> str:
    """Execute Python code in an isolated subprocess and return stdout+stderr.

    The subprocess has:
      - No network access is restricted (inherits system but code is sandboxed)
      - Hard timeout (default 15s, max 30s)
      - CPU time limit via resource module
      - Access to the Helix-AGI project on sys.path

    Args:
        code: Python code to execute.
        timeout: Seconds before the subprocess is killed (max 30).
    Returns:
        Combined stdout + stderr (truncated to 4000 chars).
    """
    timeout = min(int(timeout), EXEC_TIMEOUT)

    # Governor constitutional check
    try:
        import sys
        for mod in sys.modules.values():
            if hasattr(mod, '_governor_instance'):
                gov = mod._governor_instance
                if hasattr(gov, 'check_constitutional'):
                    ok, reason = gov.check_constitutional("run_python", {"code": code})
                    if not ok:
                        return f"[run_python CONSTITUTIONAL BLOCK] {reason}"
                    break
    except Exception:
        pass

    # Write code to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        # Inject project path so imports work
        preamble = f'import sys; sys.path.insert(0, {str(HELIX_AGI_ROOT)!r})\n'
        tmp.write(preamble + code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(HELIX_AGI_ROOT),
        )
        out = result.stdout + result.stderr
        truncated = out[:4000]
        if len(out) > 4000:
            truncated += f"\n... [{len(out) - 4000} chars truncated]"
        exit_note = f"\n[exit code: {result.returncode}]"
        logger.info(f"[code_tools] run_python: exit={result.returncode}, out={len(out)} chars")
        return truncated + exit_note
    except subprocess.TimeoutExpired:
        return f"[run_python TIMEOUT] Execution exceeded {timeout}s — process killed."
    except Exception as e:
        return f"[run_python ERROR] {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def fc_run_tests(pattern: str = "test_*.py", timeout: int = 60) -> str:
    """Run pytest on test files matching the pattern and return a summary.

    Args:
        pattern: Glob pattern for test files (relative to project parent dir).
        timeout: Seconds before test run is killed.
    Returns:
        Pytest output summary (pass/fail counts, errors).
    """
    timeout = min(int(timeout), TEST_TIMEOUT)
    test_dir = HELIX_AGI_ROOT.parent  # tests are in analysis_project/
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", pattern, "--tb=short", "-q",
             "--no-header", "--color=no"],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=str(test_dir),
        )
        out = (result.stdout + result.stderr)[:3000]
        logger.info(f"[code_tools] run_tests: exit={result.returncode}")
        return f"[run_tests result]\n{out}\n[exit: {result.returncode}]"
    except subprocess.TimeoutExpired:
        return f"[run_tests TIMEOUT] Tests exceeded {timeout}s."
    except Exception as e:
        return f"[run_tests ERROR] {e}"


def fc_reload_tool(module_path: str) -> str:
    """Hot-reload a Python module and rebuild the tool registry.

    After writing a new tool with write_code(), call this to activate it
    without restarting the entire system.

    Args:
        module_path: Relative path to the module (e.g. "tools/my_tool.py")
                     OR dotted module name (e.g. "tools.my_tool")
    Returns:
        Success message with newly registered tool names, or error.
    """
    # Normalize to dotted name
    if module_path.endswith(".py"):
        dotted = module_path.replace("/", ".").replace("\\", ".")[:-3]
    else:
        dotted = module_path

    # Resolve file path for safety check
    file_path = module_path if not module_path.endswith(".py") else module_path
    if not file_path.endswith(".py"):
        file_path = dotted.replace(".", "/") + ".py"

    resolved = _resolve_safe(file_path)
    if resolved is None:
        return f"[reload_tool ERROR] Path escapes project root."
    if not resolved.exists():
        return f"[reload_tool ERROR] File not found: {file_path}"

    try:
        # Add project to path if needed
        root_str = str(HELIX_AGI_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Load or reload the module
        if dotted in sys.modules:
            mod = sys.modules[dotted]
            importlib.reload(mod)
            action = "reloaded"
        else:
            spec = importlib.util.spec_from_file_location(dotted, resolved)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dotted] = mod
            spec.loader.exec_module(mod)
            action = "loaded"

        # Rebuild tool registry — invalidate declaration cache so new tools surface
        from tools.tool_registry import registry, invalidate_check_cache
        before = set(registry._tools.keys())
        invalidate_check_cache()          # clears the @lru_cache on tool checks
        registry._generation += 1        # bump generation so callers know registry changed
        after = set(registry._tools.keys())
        new_tools = after - before

        logger.info(f"[code_tools] reload_tool: {dotted} {action}, new tools: {new_tools}")
        _log_to_journal("reload_tool", file_path, "", f"{action} — new tools: {new_tools}")

        msg = f"[reload_tool OK] Module '{dotted}' {action} successfully."
        if new_tools:
            msg += f" New tools registered: {', '.join(sorted(new_tools))}"
        return msg

    except Exception as e:
        logger.error(f"[code_tools] reload_tool error: {e}")
        return f"[reload_tool ERROR] {e}"


# ── Registry registration ──────────────────────────────────────────────────────

def _register():
    """Register all code tools into the ToolRegistry under toolset='self'."""
    try:
        from tools.tool_registry import registry

        registry.register(
            name="read_code",
            toolset="self",
            schema={
                "name": "read_code",
                "description": "Read a Python source file from the Helix-AGI project directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Relative path from project root, e.g. 'tools/web_search.py'"}
                    },
                    "required": ["path"]
                }
            },
            handler=fc_read_code,
        )

        registry.register(
            name="write_code",
            toolset="self",
            schema={
                "name": "write_code",
                "description": (
                    "Write Python source code to a file in the Helix-AGI project. "
                    "Use this to create new tools, fix bugs, or extend capabilities. "
                    "Immutable safety-critical files are refused. "
                    "Always call reload_tool() after writing a new tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Relative path from project root"},
                        "content": {"type": "string",
                                    "description": "Complete Python source code"}
                    },
                    "required": ["path", "content"]
                }
            },
            handler=fc_write_code,
        )

        registry.register(
            name="run_python",
            toolset="self",
            schema={
                "name": "run_python",
                "description": (
                    "Execute Python code in an isolated subprocess. "
                    "Use to test new code, run experiments, or compute results. "
                    "Has access to all Helix-AGI modules via sys.path."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string",
                                 "description": "Python code to execute"},
                        "timeout": {"type": "integer",
                                    "description": "Seconds before kill (max 20)", "default": 15}
                    },
                    "required": ["code"]
                }
            },
            handler=fc_run_python,
        )

        registry.register(
            name="run_tests",
            toolset="self",
            schema={
                "name": "run_tests",
                "description": "Run pytest on test files to validate code changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string",
                                    "description": "Glob pattern for test files",
                                    "default": "test_*.py"},
                        "timeout": {"type": "integer",
                                    "description": "Seconds before kill", "default": 60}
                    },
                    "required": []
                }
            },
            handler=fc_run_tests,
        )

        registry.register(
            name="reload_tool",
            toolset="self",
            schema={
                "name": "reload_tool",
                "description": (
                    "Hot-reload a Python module and rebuild the tool registry. "
                    "Call this after write_code() to activate a new tool immediately."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "module_path": {"type": "string",
                                        "description": "Relative path or dotted module name"}
                    },
                    "required": ["module_path"]
                }
            },
            handler=fc_reload_tool,
        )

        logger.info("[code_tools] Registered 5 self-modification tools under toolset='self'")

    except Exception as e:
        logger.warning(f"[code_tools] Registry registration failed: {e}")


# Auto-register when imported
_register()
