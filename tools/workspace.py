"""
Helix — Workspace Tools

Simple, direct tools for working inside a cloned repository.
These wrap the raw terminal/file tools with automatic path resolution
so the agent doesn't have to construct absolute paths manually.

Usage pattern:
  ws_ls(slug)                    → list files in repo root
  ws_read(slug, "src/foo.py")    → read a source file
  ws_run(slug, "pytest -x -q")   → run a command in the repo
  ws_tree(slug)                  → show repo file structure
  ws_diff(slug)                  → show current git diff
"""

import os
import subprocess
from pathlib import Path

PROJ = Path(__file__).parent.parent.resolve()
WORKSPACE = PROJ / "workspace"
SOLUTIONS = PROJ / "solutions"


def _repo_dir(slug: str) -> Path | None:
    """Find the cloned repo directory for a bounty slug or repo name."""
    if not WORKSPACE.exists():
        # Last resort: pick the workspace dir with the most slug chars in common
    best, best_score = None, 0
    slug_lower = slug.lower().replace("-","").replace("_","")
    for d in candidates:
        if not d.is_dir():
            continue
        dname = d.name.lower().replace("-","").replace("_","")
        score = sum(1 for c in dname if c in slug_lower)
        if score > best_score:
            best, best_score = d, score
    if best and best_score > 5:
        return best
    return None
    if not slug:
        return None
    # Accept "owner/repo" format directly
    normalized = slug.replace("/", "_").replace("-", "_")
    slug_base = slug.rsplit("-issue-", 1)[0].replace("-", "_", 1) if "-issue-" in slug else normalized
    candidates = list(WORKSPACE.iterdir())
    # Exact name match
    for d in candidates:
        if d.is_dir() and d.name == slug_base:
            return d
    # Partial match — any workspace dir whose name contains key parts
    for d in candidates:
        if not d.is_dir():
            continue
        dname = d.name.replace("-", "_").lower()
        sbase = slug_base.lower()
        if sbase in dname or dname in sbase:
            return d
        # Try matching the repo owner+name ignoring case/separators
        parts = [p for p in sbase.split("_") if len(p) > 2]
        if all(p in dname for p in parts):
            return d
    return None


def _run(cmd: str, cwd: Path, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=str(cwd), timeout=timeout,
                           env={**os.environ})
        out = (r.stdout or "") + (f"\nSTDERR: {r.stderr}" if r.stderr else "")
        if r.returncode != 0:
            out += f"\n(exit {r.returncode})"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def ws_ls(slug: str, subdir: str = "") -> str:
    """
    List files in the cloned repo for this bounty.
    Call with just the slug to see the root, or pass a subdir.

    Args:
        slug: Bounty slug e.g. 'SecureBananaLabs-bug-bounty-issue-2885'
        subdir: Optional subdirectory to list

    Returns:
        File listing with sizes
    """
    repo = _repo_dir(slug)
    if not repo:
        dirs = [d.name for d in WORKSPACE.iterdir()] if WORKSPACE.exists() else []
        return f"No repo found for '{slug}'.\nWorkspace contains: {dirs}\nRun bounty_clone_repo(repo) first."
    target = repo / subdir if subdir else repo
    return _run(f"ls -la {target}", repo)


def ws_tree(slug: str, depth: int = 2) -> str:
    """
    Show the file tree of the cloned repo (like 'find' with depth limit).
    Use this first to understand the repo structure before reading files.

    Args:
        slug: Bounty slug
        depth: How deep to show (default 2)
    """
    repo = _repo_dir(slug)
    if not repo:
        return f"No repo found for '{slug}'. Run bounty_clone_repo(repo) first."
    result = _run(f"find . -not -path '*/.git/*' -maxdepth {depth} | sort", repo)
    return f"Repo structure ({repo.name}):\n{result}"


def ws_read(slug: str, filepath: str) -> str:
    """
    Read a source file from the cloned repo.
    Use ws_tree() first to find the right path.

    Args:
        slug: Bounty slug
        filepath: Relative path inside the repo e.g. 'src/main.py'

    Returns:
        File contents (truncated at 6000 chars)
    """
    repo = _repo_dir(slug)
    if not repo:
        return f"No repo found for '{slug}'. Run bounty_clone_repo(repo) first."
    target = repo / filepath
    if not target.exists():
        # Try searching for the file
        result = _run(f"find . -name '{Path(filepath).name}' -not -path '*/.git/*'", repo)
        return f"File not found: {filepath}\nFiles with that name:\n{result}"
    try:
        content = target.read_text(errors="replace")
        if len(content) > 6000:
            return content[:6000] + f"\n... (truncated, {len(content)} total chars)"
        return content
    except Exception as e:
        return f"Read failed: {e}"


def ws_run(slug: str, command: str, timeout: int = 60) -> str:
    """
    Run a shell command inside the cloned repo.
    Use this to: install deps, run tests, check Python version, reproduce bugs.

    Args:
        slug: Bounty slug
        command: Shell command e.g. 'python3 -m pytest tests/ -x -q'
                 Or 'pip install -e .' or 'python3 --version'
        timeout: Max seconds (default 60)

    Returns:
        Command output
    """
    repo = _repo_dir(slug)
    if not repo:
        return f"No repo found for '{slug}'. Run bounty_clone_repo(repo) first."
    return _run(command, repo, timeout=timeout)


def ws_write(slug: str, filepath: str, content: str) -> str:
    """
    Write a file directly into the cloned repo.
    Use this to implement your fix — write the corrected source file.

    Args:
        slug: Bounty slug
        filepath: Relative path inside the repo e.g. 'src/utils.py'
        content: Full file content to write

    Returns:
        Confirmation with path
    """
    repo = _repo_dir(slug)
    if not repo:
        return f"No repo found for '{slug}'. Run bounty_clone_repo(repo) first."
    target = repo / filepath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"✅ Written {len(content)} chars to {repo.name}/{filepath}"


def ws_diff(slug: str) -> str:
    """
    Show the git diff of all changes made in the cloned repo.
    Use this to generate the patch after implementing your fix.

    Args:
        slug: Bounty slug

    Returns:
        Unified diff of all changes — this is your PATCH.diff content
    """
    repo = _repo_dir(slug)
    if not repo:
        return f"No repo found for '{slug}'."
    diff = _run("git diff", repo)
    if not diff or diff == "(no output)":
        diff = _run("git diff HEAD", repo)
    return diff or "No changes detected. Have you written your fix yet?"


def ws_install_deps(slug: str) -> str:
    """
    Install the repo's dependencies so tests can run.
    Tries: pip install -e . / pip install -r requirements.txt / npm install

    Args:
        slug: Bounty slug
    """
    repo = _repo_dir(slug)
    if not repo:
        return f"No repo found for '{slug}'."

    results = []
    if (repo / "requirements.txt").exists():
        results.append(_run("pip install -r requirements.txt -q", repo, timeout=120))
    if (repo / "setup.py").exists() or (repo / "pyproject.toml").exists():
        results.append(_run("pip install -e . -q", repo, timeout=120))
    if (repo / "package.json").exists():
        results.append(_run("npm install --silent", repo, timeout=120))

    if not results:
        return "No recognized dependency file found (requirements.txt, setup.py, pyproject.toml, package.json)"
    return "\n".join(results)
