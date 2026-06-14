"""
Helix — Bounty Workflow Tool

Provides a structured, file-based workflow for solving GitHub bounties.
Each issue gets its own directory under solutions/active/ so the agent
can work without loading the full context each time.

Workflow:
  1. bounty_claim(repo, issue_num)     → creates solutions/active/<slug>/
  2. bounty_clone_repo(repo)           → clones repo to workspace/<slug>/
  3. bounty_run(slug, command)         → runs command inside the cloned repo
  4. bounty_write_plan(slug, content)  → writes PLAN.md
  5. bounty_write_patch(slug, content) → writes PATCH.diff
  6. bounty_write_pr(slug, content)    → writes PR_DESCRIPTION.md
  7. bounty_apply_patch(slug)          → applies the patch to the cloned repo
  8. bounty_submit(slug)               → forks, branches, pushes, opens PR
  9. bounty_move(slug, status)         → moves to submitted/accepted/rejected
"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("helix.tools.bounty")

PROJ = Path(__file__).parent.parent.resolve()
SOLUTIONS_DIR = PROJ / "solutions"
WORKSPACE_DIR = PROJ / "workspace"
GITHUB_TOKEN = lambda: os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_USER = lambda: os.environ.get("GITHUB_USER", "batteryphil").strip()
ISSUEHUNT_USER = lambda: os.environ.get("ISSUEHUNT_USERNAME", "batteryphil").strip()


def _slug(repo: str, issue_num: int) -> str:
    return f"{repo.replace('/', '-')}-issue-{issue_num}"


def _solution_dir(slug: str, status: str = "active") -> Path:
    return SOLUTIONS_DIR / status / slug


def _run(cmd: str, cwd: Path, timeout: int = 120) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           cwd=str(cwd), timeout=timeout,
                           env={**os.environ, "GITHUB_TOKEN": GITHUB_TOKEN()})
        out = r.stdout[:4000] + (r.stderr[:1000] if r.stderr else "")
        if r.returncode != 0:
            out += f"\n(exit {r.returncode})"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def bounty_claim(repo: str, issue_num: int, title: str = "", labels: str = "",
                 reward_estimate: str = "$0", difficulty: str = "unknown") -> str:
    """
    Claim a bounty issue — creates the solution directory and index file.
    Call this first when you decide to attempt a bounty.

    Args:
        repo: GitHub repo (e.g. 'psf/requests')
        issue_num: Issue number
        title: Issue title
        labels: Comma-separated labels
        reward_estimate: Estimated payout (e.g. '$50')
        difficulty: easy/medium/hard

    Returns:
        Path to the solution directory.
    """
    slug = _slug(repo, issue_num)
    sol_dir = _solution_dir(slug, "active")
    sol_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "repo": repo,
        "issue": issue_num,
        "url": f"https://github.com/{repo}/issues/{issue_num}",
        "title": title,
        "labels": labels,
        "reward_estimate": reward_estimate,
        "difficulty": difficulty,
        "status": "active",
        "claimed_at": __import__("datetime").datetime.utcnow().isoformat(),
    }
    (sol_dir / "index.json").write_text(json.dumps(index, indent=2))

    # Stub files
    for fname in ["PLAN.md", "PATCH.diff", "PR_DESCRIPTION.md", "LESSONS.md"]:
        p = sol_dir / fname
        if not p.exists():
            p.write_text(f"# {fname.replace('.md','').replace('.diff','')}\n\n_To be filled in._\n")

    logger.info(f"[BOUNTY] Claimed {repo}#{issue_num} → {sol_dir}")
    return (
        f"✅ Bounty claimed: {repo}#{issue_num}\n"
        f"   Directory: {sol_dir}\n"
        f"   Next: bounty_clone_repo('{repo}') then bounty_write_plan('{slug}', ...)"
    )


def bounty_clone_repo(repo: str) -> str:
    """
    Clone a GitHub repository into workspace/ for local editing and testing.

    Args:
        repo: GitHub repo slug (e.g. 'psf/requests')

    Returns:
        Path to cloned repo, or error message.
    """
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    repo_dir = WORKSPACE_DIR / repo.replace("/", "_")

    if repo_dir.exists():
        # Pull latest instead of re-cloning
        result = _run("git pull", repo_dir)
        return f"Repo already cloned at {repo_dir}\n{result}"

    token = GITHUB_TOKEN()
    if token:
        url = f"https://{GITHUB_USER()}:{token}@github.com/{repo}.git"
    else:
        url = f"https://github.com/{repo}.git"

    result = _run(f"git clone --depth=50 {url} {repo_dir}", WORKSPACE_DIR, timeout=120)
    if repo_dir.exists():
        return f"✅ Cloned {repo} → {repo_dir}\n{result}"
    return f"Clone failed:\n{result}"


def bounty_run(slug: str = "", command: str = "", repo: str = "", issue_num: int = 0) -> str:
    if not slug and repo and issue_num:
        slug = _slug(repo, issue_num)
    elif not slug and repo:
        # Try to find by repo name in workspace
        workspace_dirs = list(WORKSPACE_DIR.iterdir()) if WORKSPACE_DIR.exists() else []
        for d in workspace_dirs:
            if repo.replace("/","_") in d.name:
                slug = d.name
                break
    """
    Run a shell command inside the cloned repo for this bounty.
    Use this to: reproduce the bug, run tests, verify your fix.

    Args:
        slug: Bounty slug (e.g. 'psf-requests-issue-1234')
        command: Shell command to run (e.g. 'python -m pytest tests/ -x -q')

    Returns:
        Command output (stdout + stderr, truncated to 4000 chars).
    """
    # Find the cloned repo — parse repo from slug
    # slug format: owner-repo-issue-N  →  find workspace dir
    workspace_dirs = list(WORKSPACE_DIR.iterdir()) if WORKSPACE_DIR.exists() else []
    repo_dir = None
    for d in workspace_dirs:
        if d.is_dir() and slug.replace("-issue-" + slug.split("-issue-")[-1], "").replace("-", "_") in d.name:
            repo_dir = d
            break

    if not repo_dir:
        # Fallback: use the slug prefix directly
        candidate = WORKSPACE_DIR / slug.rsplit("-issue-", 1)[0].replace("-", "_", 1)
        if candidate.exists():
            repo_dir = candidate

    if not repo_dir or not repo_dir.exists():
        return (
            f"No cloned repo found for '{slug}'.\n"
            f"Run bounty_clone_repo(repo) first.\n"
            f"Workspace contents: {[d.name for d in workspace_dirs]}"
        )

    logger.info(f"[BOUNTY] Run in {repo_dir}: {command}")
    return _run(command, repo_dir, timeout=120)


def bounty_write_plan(slug: str = "", content: str = "", repo: str = "", issue_num: int = 0, plan: str = "") -> str:
    if not slug and repo and issue_num:
        slug = _slug(repo, issue_num)
    if not content and plan:
        content = plan
    """
    Write the PLAN.md for this bounty — your analysis and fix approach.
    Include: problem description, root cause, fix approach, test plan.

    Args:
        slug: Bounty slug
        content: Full markdown content for PLAN.md
    """
    sol_dir = _solution_dir(slug, "active")
    if not sol_dir.exists():
        return f"No active solution for '{slug}'. Run bounty_claim() first."
    (sol_dir / "PLAN.md").write_text(content)
    logger.info(f"[BOUNTY] PLAN.md written for {slug}")
    return f"✅ PLAN.md written ({len(content)} chars) → {sol_dir}/PLAN.md"


def bounty_write_patch(slug: str = "", content: str = "", repo: str = "", issue_num: int = 0, patch: str = "") -> str:
    if not slug and repo and issue_num:
        slug = _slug(repo, issue_num)
    if not content and patch:
        content = patch
    """
    Write the PATCH.diff — the actual code fix as a unified diff.

    Args:
        slug: Bounty slug
        content: Unified diff content
    """
    sol_dir = _solution_dir(slug, "active")
    if not sol_dir.exists():
        return f"No active solution for '{slug}'. Run bounty_claim() first."
    (sol_dir / "PATCH.diff").write_text(content)
    logger.info(f"[BOUNTY] PATCH.diff written for {slug}")
    return f"✅ PATCH.diff written ({len(content)} chars) → {sol_dir}/PATCH.diff"


def bounty_write_pr(slug: str = "", content: str = "", repo: str = "", issue_num: int = 0, title: str = "", body: str = "", head: str = "", base: str = "") -> str:
    if not slug and repo and issue_num:
        slug = _slug(repo, issue_num)
    if not content and body:
        content = body
    elif not content and title:
        content = f"## {title}\n\n{body}"
    """
    Write the PR_DESCRIPTION.md — the pull request body.
    ALWAYS include the IssueHunt attribution line at the bottom.

    Args:
        slug: Bounty slug
        content: PR description markdown
    """
    sol_dir = _solution_dir(slug, "active")
    if not sol_dir.exists():
        return f"No active solution for '{slug}'. Run bounty_claim() first."

    # Auto-append IssueHunt attribution if missing
    ih_user = ISSUEHUNT_USER()
    attribution = f"\n\n---\n> IssueHunt contributor: @{ih_user} (https://issuehunt.io/u/{ih_user})"
    if "issuehunt.io" not in content.lower():
        content += attribution

    (sol_dir / "PR_DESCRIPTION.md").write_text(content)
    logger.info(f"[BOUNTY] PR_DESCRIPTION.md written for {slug}")
    return f"✅ PR_DESCRIPTION.md written → {sol_dir}/PR_DESCRIPTION.md"


def bounty_apply_patch(slug: str = "", repo: str = "", issue_num: int = 0) -> str:
    if not slug and repo and issue_num:
        slug = _slug(repo, issue_num)
    """
    Apply the PATCH.diff to the cloned repo for this bounty.
    Run bounty_run(slug, 'python -m pytest ...') after this to verify.

    Args:
        slug: Bounty slug
    """
    sol_dir = _solution_dir(slug, "active")
    patch_file = sol_dir / "PATCH.diff"
    if not patch_file.exists():
        return f"No PATCH.diff found for '{slug}'. Write it first with bounty_write_patch()."

    # Find cloned repo
    workspace_dirs = list(WORKSPACE_DIR.iterdir()) if WORKSPACE_DIR.exists() else []
    repo_dir = None
    slug_base = slug.rsplit("-issue-", 1)[0].replace("-", "_", 1)
    for d in workspace_dirs:
        if slug_base in d.name and d.is_dir():
            repo_dir = d
            break

    if not repo_dir:
        return f"Cloned repo not found. Run bounty_clone_repo() first."

    result = _run(f"git apply --check {patch_file}", repo_dir)
    if "error" in result.lower():
        return f"Patch check failed (may not apply cleanly):\n{result}"

    result = _run(f"git apply {patch_file}", repo_dir)
    return f"Patch applied:\n{result}"


def bounty_submit(slug: str = "", repo: str = "", issue_num: int = 0) -> str:
    if not slug and repo and issue_num:
        slug = _slug(repo, issue_num)
    """
    Submit a completed bounty: fork the target repo, push fix as a branch,
    and create a pull request via GitHub API.
    IssueHunt attribution is auto-included in the PR body.

    Args:
        slug: Bounty slug (e.g. 'psf-requests-issue-1234')

    Returns:
        PR URL on success, error message on failure.
    """
    import requests

    sol_dir = _solution_dir(slug, "active")
    if not sol_dir.exists():
        return f"No active solution for '{slug}'."

    index = json.loads((sol_dir / "index.json").read_text())
    repo = index["repo"]
    issue_num = index["issue"]
    pr_body = (sol_dir / "PR_DESCRIPTION.md").read_text()
    patch = (sol_dir / "PATCH.diff").read_text()
    plan = (sol_dir / "PLAN.md").read_text()

    token = GITHUB_TOKEN()
    user = GITHUB_USER()
    if not token:
        return "Error: GITHUB_TOKEN not set."

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    # 1. Fork the repo
    fork_r = requests.post(
        f"https://api.github.com/repos/{repo}/forks",
        headers=headers, timeout=30
    )
    if fork_r.status_code not in (200, 202):
        return f"Fork failed ({fork_r.status_code}): {fork_r.text[:300]}"

    fork_full_name = fork_r.json().get("full_name", f"{user}/{repo.split('/')[-1]}")
    logger.info(f"[BOUNTY] Forked {repo} → {fork_full_name}")

    import time; time.sleep(5)  # GitHub needs a moment to set up the fork

    # 2. Clone fork and apply patch
    repo_name = repo.split("/")[-1]
    fork_dir = WORKSPACE_DIR / f"fork_{repo_name}"

    if not fork_dir.exists():
        _run(f"git clone --depth=50 https://{user}:{token}@github.com/{fork_full_name}.git {fork_dir}",
             WORKSPACE_DIR, timeout=120)

    branch = f"helix-fix-issue-{issue_num}"
    _run(f"git checkout -b {branch}", fork_dir)
    _run(f"git config user.email '{user}@users.noreply.github.com'", fork_dir)
    _run(f"git config user.name '{user}'", fork_dir)

    # Apply the patch
    patch_path = sol_dir / "PATCH.diff"
    apply_result = _run(f"git apply {patch_path}", fork_dir)
    if "error" in apply_result.lower():
        return f"Patch did not apply cleanly:\n{apply_result}"

    _run("git add -A", fork_dir)
    commit_title = index.get("title", "Bug fix")[:60]
    _run(f'git commit -m "Fix #{issue_num}: {commit_title}"', fork_dir)
    push_result = _run(f"git push origin {branch}", fork_dir)

    if "error" in push_result.lower() and "already exists" not in push_result.lower():
        return f"Push failed:\n{push_result}"

    # 3. Open PR
    pr_r = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        headers=headers,
        json={
            "title": f"Fix #{issue_num}: {index.get('title', 'Bug fix')}",
            "head": f"{user}:{branch}",
            "base": "main",
            "body": pr_body,
        },
        timeout=30,
    )

    if pr_r.status_code in (200, 201):
        pr_url = pr_r.json().get("html_url", "")
        bounty_move(slug, "submitted")
        logger.info(f"[BOUNTY] PR submitted: {pr_url}")
        return f"🎉 PR submitted!\n   {pr_url}\n   Moved to solutions/submitted/"
    else:
        return f"PR creation failed ({pr_r.status_code}): {pr_r.text[:400]}"


def bounty_move(slug: str, status: str) -> str:
    """
    Move a solution to a different status folder.

    Args:
        slug: Bounty slug
        status: 'submitted', 'accepted', or 'rejected'
    """
    import shutil
    valid = {"submitted", "accepted", "rejected", "active"}
    if status not in valid:
        return f"Invalid status '{status}'. Use: {', '.join(valid)}"

    for src_status in ["active", "submitted", "accepted", "rejected"]:
        src = _solution_dir(slug, src_status)
        if src.exists():
            dst = _solution_dir(slug, status)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            logger.info(f"[BOUNTY] {slug} moved {src_status} → {status}")
            return f"✅ {slug} moved to solutions/{status}/"

    return f"No solution found for '{slug}' in any status folder."


def bounty_status() -> str:
    """
    Show a summary of all current bounty solutions across all statuses.
    Use this to get a quick overview of your pipeline without reading every file.
    """
    output = "📊 Bounty Pipeline Summary\n" + "=" * 50 + "\n"
    total = 0
    for status in ["active", "submitted", "accepted", "rejected"]:
        d = SOLUTIONS_DIR / status
        if not d.exists():
            continue
        items = [x for x in d.iterdir() if x.is_dir() and x.name != ".gitkeep"]
        if items:
            output += f"\n{status.upper()} ({len(items)}):\n"
            for item in items:
                idx_path = item / "index.json"
                if idx_path.exists():
                    try:
                        idx = json.loads(idx_path.read_text())
                        output += f"  • {idx['repo']}#{idx['issue']}: {idx.get('title','')[:60]}\n"
                        output += f"    Reward: {idx.get('reward_estimate','?')} | Difficulty: {idx.get('difficulty','?')}\n"
                    except Exception:
                        output += f"  • {item.name}\n"
                total += 1

    if total == 0:
        output += "\nNo bounties claimed yet. Use issuehunt_top_bounties() to find opportunities.\n"

    output += f"\nTotal: {total} bounties in pipeline\n"
    return output


def bounty_read_plan(slug: str) -> str:
    """Read the PLAN.md for a bounty without loading the full solution context."""
    for status in ["active", "submitted", "accepted", "rejected"]:
        p = _solution_dir(slug, status) / "PLAN.md"
        if p.exists():
            return p.read_text()
    return f"No PLAN.md found for '{slug}'."


def bounty_easy_search(max_results: int = 10) -> str:
    """
    Search specifically for easy, quick-win bounties:
    - Documentation improvements
    - Test additions
    - Small 1-10 line bug fixes
    - Issues tagged 'good first issue' + bounty label

    These have the highest success probability for automated solving.
    """
    import requests

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = GITHUB_TOKEN()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    results = []
    queries = [
        'label:"good first issue" label:"bounty" state:open',
        'label:"good first issue" label:"issuehunt" state:open',
        'label:"documentation" label:"bounty" state:open',
        'label:"tests" label:"bounty" state:open',
        'label:"easy" label:"bounty" state:open',
    ]

    for q in queries:
        try:
            r = requests.get(
                "https://api.github.com/search/issues",
                params={"q": q, "sort": "updated", "order": "desc", "per_page": 5},
                headers=headers, timeout=15
            )
            if r.status_code == 200:
                for item in r.json().get("items", []):
                    labels = [l["name"] for l in item.get("labels", [])]
                    results.append({
                        "title": item["title"],
                        "url": item["html_url"],
                        "repo": item["repository_url"].replace("https://api.github.com/repos/", ""),
                        "issue_num": item["number"],
                        "labels": labels,
                        "comments": item["comments"],
                        "updated": item["updated_at"][:10],
                        "body_len": len(item.get("body") or ""),
                    })
        except Exception as e:
            logger.warning(f"Easy search failed for '{q}': {e}")

    # Deduplicate
    seen, unique = set(), []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    # Sort: prefer issues with short bodies (simpler problems) and few comments
    unique.sort(key=lambda x: (x["body_len"], x["comments"]))
    unique = unique[:max_results]

    if not unique:
        return "No easy bounties found. Try issuehunt_top_bounties() for broader search."

    out = f"🎯 Easy Bounties — Best Probability of Success ({len(unique)} found)\n"
    out += "=" * 60 + "\n"
    for i, issue in enumerate(unique, 1):
        out += (
            f"\n#{i} {issue['title']}\n"
            f"   Repo:    {issue['repo']} (#{issue['issue_num']})\n"
            f"   URL:     {issue['url']}\n"
            f"   Labels:  {', '.join(issue['labels'])}\n"
            f"   Updated: {issue['updated']} | Comments: {issue['comments']}\n"
        )

    out += (
        "\n" + "=" * 60 + "\n"
        "Recommended workflow:\n"
        "1. bounty_claim(repo, issue_num, title, labels, reward, difficulty)\n"
        "2. bounty_clone_repo(repo)\n"
        "3. bounty_run(slug, 'python -m pytest -x -q')  ← reproduce failure\n"
        "4. bounty_write_plan(slug, '# Plan\\n...')\n"
        "5. bounty_write_patch(slug, unified_diff)\n"
        "6. bounty_apply_patch(slug)\n"
        "7. bounty_run(slug, 'python -m pytest -x -q')  ← verify fix\n"
        "8. bounty_write_pr(slug, pr_body)\n"
        "9. bounty_submit(slug)  ← forks, pushes, opens PR automatically\n"
    )
    return out
