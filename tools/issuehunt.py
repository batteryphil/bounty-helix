"""
Helix — IssueHunt Bounty Tool

IssueHunt doesn't have a public REST API, but all IssueHunt bounties
are backed by GitHub Issues tagged with recognizable labels.

This tool searches GitHub Issues for IssueHunt-posted bounties using
the GitHub Search API and returns ranked opportunities by reward estimate.

Payout account: batteryphil (https://issuehunt.io/u/batteryphil)
"""

import os
import logging
import json
from typing import Optional

logger = logging.getLogger("helix.tools.issuehunt")

API_BASE = "https://api.github.com"
TIMEOUT = 15

# IssueHunt labels applied to funded issues on GitHub
ISSUEHUNT_LABELS = ["issuehunt", "bounty", "paid", "funded", "reward"]

# Languages/ecosystems to target (prioritize Python for highest match rate)
TARGET_LANGUAGES = ["python", "javascript", "typescript", "rust", "go"]


def _gh_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    h = {"Accept": "application/vnd.github.v3+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def issuehunt_search(language: str = "python", max_results: int = 10) -> str:
    """
    Search IssueHunt for open funded bounties via GitHub Issues API.

    IssueHunt attaches a label to GitHub issues when a bounty is posted.
    This searches for those labels across open issues.

    Args:
        language: Programming language to filter by (default: python)
        max_results: Max number of results to return (default: 10)

    Returns:
        Ranked list of open bounty issues with repo, title, URL, and label info.
    """
    import requests

    results = []

    # Search across all IssueHunt-style labels
    for label in ISSUEHUNT_LABELS:
        query = f'label:"{label}" state:open language:{language}'
        try:
            r = requests.get(
                f"{API_BASE}/search/issues",
                params={"q": query, "sort": "updated", "order": "desc", "per_page": 10},
                headers=_gh_headers(),
                timeout=TIMEOUT,
            )
            if r.status_code != 200:
                logger.warning(f"[issuehunt] GitHub search failed ({r.status_code}) for label={label}")
                continue

            items = r.json().get("items", [])
            for item in items:
                labels = [l["name"] for l in item.get("labels", [])]
                results.append({
                    "title": item["title"],
                    "url": item["html_url"],
                    "repo": item["repository_url"].replace("https://api.github.com/repos/", ""),
                    "labels": labels,
                    "updated": item["updated_at"][:10],
                    "comments": item["comments"],
                    "label_match": label,
                })
        except Exception as e:
            logger.warning(f"[issuehunt] Error searching label={label}: {e}")

    # Deduplicate by URL
    seen = set()
    unique = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    if not unique:
        return (
            f"No IssueHunt bounties found for language={language}. "
            "Try issuehunt_search(language='javascript') or issuehunt_search(language='rust')."
        )

    unique = unique[:max_results]

    out = f"🎯 IssueHunt Bounties — {language} ({len(unique)} found)\n"
    out += "=" * 60 + "\n"
    for i, issue in enumerate(unique, 1):
        out += (
            f"\n#{i} [{issue['label_match'].upper()}] {issue['title']}\n"
            f"   Repo:    {issue['repo']}\n"
            f"   URL:     {issue['url']}\n"
            f"   Labels:  {', '.join(issue['labels'])}\n"
            f"   Updated: {issue['updated']} | Comments: {issue['comments']}\n"
        )

    out += (
        "\n" + "=" * 60 + "\n"
        "Next steps:\n"
        "1. Pick the most promising issue\n"
        "2. Use github_issue tool to read full details\n"
        "3. Create solutions/active/<repo>-issue-<N>/ with PLAN.md\n"
        "4. Implement fix, write PATCH.diff and PR_DESCRIPTION.md\n"
        "5. git commit + git push + submit PR\n"
        "   Include: > IssueHunt contributor: @batteryphil\n"
    )

    return out


def issuehunt_top_bounties(max_results: int = 15) -> str:
    """
    Search for the highest-value open bounties across all languages.
    Searches multiple languages and label combinations to find the best opportunities.

    Returns:
        Combined ranked list of bounties across Python, JS, Rust, Go, TypeScript.
    """
    import requests

    all_results = []

    for lang in TARGET_LANGUAGES:
        for label in ["issuehunt", "bounty", "funded"]:
            query = f'label:"{label}" state:open language:{lang}'
            try:
                r = requests.get(
                    f"{API_BASE}/search/issues",
                    params={"q": query, "sort": "updated", "order": "desc", "per_page": 5},
                    headers=_gh_headers(),
                    timeout=TIMEOUT,
                )
                if r.status_code == 200:
                    for item in r.json().get("items", []):
                        all_results.append({
                            "title": item["title"],
                            "url": item["html_url"],
                            "repo": item["repository_url"].replace("https://api.github.com/repos/", ""),
                            "labels": [l["name"] for l in item.get("labels", [])],
                            "language": lang,
                            "updated": item["updated_at"][:10],
                            "comments": item["comments"],
                        })
            except Exception as e:
                logger.warning(f"[issuehunt_top] {lang}/{label}: {e}")

    # Deduplicate
    seen = set()
    unique = []
    for r in all_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    if not unique:
        return "No bounties found. GitHub API may be rate-limited — try again in a few minutes."

    unique = unique[:max_results]

    out = f"🏆 Top Open Bounties — All Languages ({len(unique)} results)\n"
    out += "=" * 60 + "\n"
    for i, issue in enumerate(unique, 1):
        out += (
            f"\n#{i} [{issue['language'].upper()}] {issue['title']}\n"
            f"   Repo:    {issue['repo']}\n"
            f"   URL:     {issue['url']}\n"
            f"   Labels:  {', '.join(issue['labels'])}\n"
            f"   Updated: {issue['updated']}\n"
        )

    return out


def issuehunt_save_opportunities(opportunities: list) -> str:
    """
    Save a list of discovered opportunities to data/opportunities.json.

    Args:
        opportunities: List of dicts with repo, url, title, language, effort, reward, probability

    Returns:
        Confirmation with count saved.
    """
    path = "data/opportunities.json"
    try:
        existing = []
        try:
            with open(path) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Merge — deduplicate by URL
        existing_urls = {o.get("url") for o in existing}
        added = 0
        for opp in opportunities:
            if opp.get("url") not in existing_urls:
                existing.append(opp)
                existing_urls.add(opp.get("url"))
                added += 1

        os.makedirs("data", exist_ok=True)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)

        return f"✅ Saved {added} new opportunities ({len(existing)} total in database)"
    except Exception as e:
        return f"Error saving opportunities: {e}"
