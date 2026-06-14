# Solutions

This directory contains all of Helix's bounty solutions, organized by status.

## Structure

```
solutions/
├── active/          ← Currently being worked on
├── submitted/       ← PR submitted, awaiting review
├── accepted/        ← Merged / bounty paid
└── rejected/        ← Closed/rejected (lessons learned inside)
```

## Solution Format

Each solution lives in its own subdirectory named after the issue:
```
solutions/active/<repo-name>-issue-<number>/
├── PLAN.md          ← Analysis + implementation plan
├── PATCH.diff       ← The actual code fix
├── PR_DESCRIPTION.md← Draft pull request body
├── LESSONS.md       ← What was learned (filled after outcome)
└── src/             ← Any helper scripts written for this issue
```

## Stats (auto-updated by Helix)
- Opportunities discovered: 0
- Solutions attempted: 0
- PRs submitted: 0
- Accepted: 0
- Acceptance rate: —
