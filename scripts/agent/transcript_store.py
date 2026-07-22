"""
Full, untruncated per-round transcript storage.

GitHub comment bodies are capped (~65k characters) well below what a single
tool result — let alone a whole round with several tool calls — can
produce, so long results end up truncated and unreadable right in the UI.
This module is the fix: a dedicated branch in the target repo holding one
JSON file per round, with the complete assistant text, tool call arguments,
and tool result content, nothing shortened.

Comments still get posted as before, but now hold only a short preview plus
a link to the full file here. On replay (history.build_initial_messages),
the full file is loaded directly — so a resumed run always works from
complete data, not whatever happened to fit and stay legible in a comment.

If a write here fails for any reason (API hiccup, permissions), the caller
is expected to fall back to embedding data directly in the comment instead
— this store is a nice-to-have for full fidelity, not a hard dependency for
the run to proceed.
"""
from __future__ import annotations

import json

from .github_client import GitHubClient

DEFAULT_BRANCH_NAME = "issue-agent-transcripts"


class TranscriptStore:
    def __init__(self, gh: GitHubClient, owner: str, repo: str, run_id: str, branch: str = DEFAULT_BRANCH_NAME):
        self.gh = gh
        self.owner = owner
        self.repo = repo
        self.run_id = run_id
        self.branch = branch
        self._branch_ready = False

    def _ensure_branch(self) -> None:
        if self._branch_ready:
            return
        self.gh.ensure_branch(self.owner, self.repo, self.branch)
        self._branch_ready = True

    def _path(self, issue_number: int, round_number: int) -> str:
        # Namespaced by Actions run ID so a resumed run's round numbering
        # (which restarts at 1) can never collide with an earlier run's
        # files for the same issue.
        return f"issue-agent/issue-{issue_number}/run-{self.run_id}/round-{round_number:04d}.json"

    def save_round(self, issue_number: int, round_number: int, payload: dict) -> tuple[str, str]:
        """Writes the full round payload. Returns (path, blob_url) for use
        in the comment. Raises on failure — the caller should catch this
        and fall back to embedding data directly in the comment."""
        self._ensure_branch()
        path = self._path(issue_number, round_number)
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        self.gh.put_file(
            self.owner,
            self.repo,
            path,
            content,
            branch=self.branch,
            message=f"issue #{issue_number}: transcript round {round_number}",
        )
        url = f"https://github.com/{self.owner}/{self.repo}/blob/{self.branch}/{path}"
        return path, url

    def load_round(self, path: str, ref: str | None = None) -> dict:
        content = self.gh.get_file(self.owner, self.repo, path, ref=ref or self.branch)
        return json.loads(content)
