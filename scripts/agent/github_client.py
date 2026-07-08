"""
Thin wrapper around the GitHub REST API. Uses `requests` directly rather than
a full SDK, since we only need a handful of endpoints.

The token used here must be a PAT (classic with `repo` scope, or fine-grained
with Contents/Pull requests/Issues write access) granted on the TARGET repo —
the workflow's automatic GITHUB_TOKEN only has access to the repo the
workflow itself lives in, not arbitrary other repos you point this tool at.
"""
from __future__ import annotations

import requests

API_ROOT = "https://api.github.com"


class GitHubClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    def _check(self, resp: requests.Response, context: str):
        if not resp.ok:
            raise RuntimeError(
                f"GitHub API error during {context}: {resp.status_code} {resp.text[:500]}"
            )
        return resp

    def get_default_branch(self, owner: str, repo: str) -> str:
        resp = self.session.get(f"{API_ROOT}/repos/{owner}/{repo}")
        self._check(resp, "fetching repo metadata")
        return resp.json()["default_branch"]

    def get_issue(self, owner: str, repo: str, number: int) -> dict:
        resp = self.session.get(f"{API_ROOT}/repos/{owner}/{repo}/issues/{number}")
        self._check(resp, f"fetching issue #{number}")
        issue = resp.json()

        comments = []
        comments_resp = self.session.get(
            f"{API_ROOT}/repos/{owner}/{repo}/issues/{number}/comments"
        )
        if comments_resp.ok:
            comments = comments_resp.json()
        issue["comments_data"] = comments
        return issue

    def add_issue_comment(self, owner: str, repo: str, number: int, body: str):
        resp = self.session.post(
            f"{API_ROOT}/repos/{owner}/{repo}/issues/{number}/comments",
            json={"body": body},
        )
        self._check(resp, f"commenting on issue #{number}")
        return resp.json()

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        head: str,
        base: str,
        title: str,
        body: str,
        draft: bool = True,
    ) -> dict:
        resp = self.session.post(
            f"{API_ROOT}/repos/{owner}/{repo}/pulls",
            json={
                "title": title,
                "head": head,
                "base": base,
                "body": body,
                "draft": draft,
            },
        )
        self._check(resp, "creating pull request")
        return resp.json()
