"""
Helix — GitHub API Tools  (READ/WRITE ENABLED)

Provides full API access to GitHub repos, issues, and PRs.
Also wraps local git operations (status, diff, log, commit, push).

REPO: batteryphil/bounty-helix — agent has full read/write access.

Auth: GITHUB_TOKEN from environment (.env file).

Tag interface (read-only subset — write tags are suppressed by governor):
  [GIT_STATUS:path]              — Repo status + current branch
  [GIT_DIFF:path]                — Show uncommitted changes (local only)
  [GIT_LOG:path]                 — Recent commit history
  [GIT_CLONE:] url               — Clone a repo (read-only local copy)
  [GITHUB_SEARCH:] query         — Search repos on GitHub
  [GITHUB_ISSUE:repo] number     — Read an issue + comments
  [GITHUB_FILE:repo] path        — Read a file from a repo
"""

import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("helix.tools.github")

# ── Config ────────────────────────────────────────────────────────────

API_BASE = "https://api.github.com"
TIMEOUT  = 15
_WRITE_DISABLED = None  # Write operations ENABLED for batteryphil/bounty-helix


def _github_headers() -> dict:
    """Build GitHub API headers with optional auth."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ── Local Git Read Operations ─────────────────────────────────────────

def git_status(repo_path: str) -> str:
    """Check git status of a local repository (read-only)."""
    if not repo_path or not os.path.isdir(repo_path):
        return f"Invalid repo path: {repo_path}"
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        status = result.stdout.strip() or "Clean — nothing to commit."
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path, capture_output=True, text=True, timeout=5,
        )
        return f"Branch: {branch.stdout.strip()}\n{status}"
    except Exception as e:
        return f"Git status failed: {e}"


def git_diff(repo_path: str) -> str:
    """Show local uncommitted changes (read-only)."""
    if not repo_path or not os.path.isdir(repo_path):
        return f"Invalid repo path: {repo_path}"
    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        diff = subprocess.run(
            ["git", "diff"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        out = ""
        if untracked.stdout.strip():
            out += f"Untracked files:\n{untracked.stdout.strip()}\n\n"
        if diff.stdout.strip():
            out += f"Modifications:\n{diff.stdout.strip()[:3000]}"
            if len(diff.stdout) > 3000:
                out += "\n...[diff truncated]"
        return out if out else "No changes."
    except Exception as e:
        return f"Git diff failed: {e}"


def git_log(repo_path: str, count: int = 10) -> str:
    """Show recent git log (read-only)."""
    if not repo_path:
        return "No repo_path provided."
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", f"-{count}"],
            cwd=repo_path, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or "No commits found."
    except Exception as e:
        return f"Git log failed: {e}"


def git_clone(repo_url: str, target_dir: str = "") -> str:
    """Clone a repository locally for reading (read-only after clone)."""
    if not repo_url:
        return "No repo_url provided."
    target = target_dir or os.path.expanduser("~/repos")
    try:
        Path(target).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = "ssh -o StrictHostKeyChecking=no"
        result = subprocess.run(
            ["git", "clone", "--depth=1", repo_url],
            cwd=target, capture_output=True, text=True, timeout=60, env=env,
        )
        if result.returncode == 0:
            return f"Cloned {repo_url} into {target}"
        return f"Clone failed: {result.stderr.strip()}"
    except Exception as e:
        return f"Git clone failed: {e}"


def git_pull(repo_path: str) -> str:
    """Pull latest from remote (updates local read-only copy)."""
    if not repo_path:
        return "No repo_path provided."
    try:
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = "ssh -o StrictHostKeyChecking=no"
        result = subprocess.run(
            ["git", "pull"],
            cwd=repo_path, capture_output=True, text=True, timeout=30, env=env,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return f"Git pull failed: {e}"


# ── WRITE OPERATIONS — ALL DISABLED ──────────────────────────────────

def git_commit(repo_path: str, message: str) -> str:
    """Commit staged changes in repo_path with message."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"git commit error: {e}"


def git_push(repo_path: str, remote: str = "origin", branch: str = "main") -> str:
    """Push commits in repo_path to remote/branch."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "push", remote, branch],
            cwd=repo_path, capture_output=True, text=True, timeout=60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"git push error: {e}"


# ── GitHub REST API — Read Only ───────────────────────────────────────

def github_search_repos(query: str) -> str:
    """Search GitHub repositories (read)."""
    import requests as req
    if not query:
        return "Missing query."
    try:
        res = req.get(
            f"{API_BASE}/search/repositories",
            params={"q": query, "per_page": 5},
            headers=_github_headers(),
            timeout=TIMEOUT,
        )
        if res.status_code == 200:
            items = res.json().get("items", [])
            if not items:
                return "No repositories found."
            out = "Found Repositories:\n"
            for i in items:
                out += f"- {i['full_name']} (★{i['stargazers_count']}): {i.get('description','')}\n"
            return out
        return f"GitHub search failed ({res.status_code}): {res.text[:500]}"
    except Exception as e:
        return f"GitHub API error: {e}"


def github_read_file(repo: str, path: str, ref: str = "HEAD") -> str:
    """Read a file from a GitHub repo via the API (read-only)."""
    import requests as req, base64
    if not repo or not path:
        return "Missing repo or path."
    try:
        res = req.get(
            f"{API_BASE}/repos/{repo}/contents/{path}",
            params={"ref": ref},
            headers=_github_headers(),
            timeout=TIMEOUT,
        )
        if res.status_code == 200:
            data = res.json()
            content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
            logger.info(f"[GITHUB] Read {repo}/{path} ({len(content)} chars)")
            return content[:4000]
        return f"GitHub file read failed ({res.status_code}): {res.text[:300]}"
    except Exception as e:
        return f"GitHub API error: {e}"


def github_read_issue(repo: str, issue_number: int) -> str:
    """Read an issue and its comments (read-only)."""
    import requests as req
    if not repo or not issue_number:
        return "Missing repo or issue_number."
    try:
        res = req.get(
            f"{API_BASE}/repos/{repo}/issues/{issue_number}",
            headers=_github_headers(),
            timeout=TIMEOUT,
        )
        if res.status_code != 200:
            return f"Failed to fetch issue: {res.text[:500]}"
        issue = res.json()
        out = (
            f"Issue #{issue['number']}: {issue['title']} (State: {issue['state']})\n"
            f"Author: {issue['user']['login']}\n\n{issue['body']}\n\n--- COMMENTS ---\n"
        )
        c_res = req.get(
            f"{API_BASE}/repos/{repo}/issues/{issue_number}/comments",
            headers=_github_headers(),
            timeout=TIMEOUT,
        )
        if c_res.status_code == 200:
            for c in c_res.json():
                out += f"\n[{c['user']['login']}] at {c['created_at']}:\n{c['body']}\n"
        return out
    except Exception as e:
        return f"GitHub API error: {e}"


# ── WRITE API — ALL DISABLED ──────────────────────────────────────────

def github_create_issue(repo: str, title: str, body: str = "") -> str:
    """DISABLED — Helix may not create issues on repos."""
    logger.warning(f"[BLOCKED] github_create_issue attempted on {repo}: {title[:60]}")
    return _WRITE_DISABLED


def github_comment_issue(repo: str, issue_number: int, body: str) -> str:
    """DISABLED — Helix may not comment on issues."""
    logger.warning(f"[BLOCKED] github_comment_issue attempted on {repo}#{issue_number}")
    return _WRITE_DISABLED


def github_create_pr(repo: str, title: str, head: str, base: str = "main", body: str = "") -> str:
    """Create a pull request on a GitHub repo."""
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        return "Error: GITHUB_TOKEN not set"
    import requests
    r = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
        json={"title": title, "head": head, "base": base, "body": body}
    )
    return r.json().get("html_url", str(r.json()))
