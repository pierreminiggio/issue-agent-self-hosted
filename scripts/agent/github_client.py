"""
Thin wrapper around the GitHub REST API. Uses `requests` directly rather than
a full SDK, since we only need a handful of endpoints.

The token used here must be a PAT (classic with `repo` scope, or fine-grained
with Contents/Pull requests/Issues write access) granted on the TARGET repo —
the workflow's automatic GITHUB_TOKEN only has access to the repo the
workflow itself lives in, not arbitrary other repos you point this tool at.
"""
from __future__ import annotations

import base64

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

    def get_authenticated_login(self) -> str:
        """Returns the login of the account that owns `token`. This is the
        single trust anchor the whole "trust the messenger" model rests on:
        only issue/comment content authored by this exact login is ever
        treated as trusted input — see agent/history.py.
        """
        resp = self.session.get(f"{API_ROOT}/user")
        self._check(resp, "fetching authenticated user")
        return resp.json()["login"]

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

    def branch_exists(self, owner: str, repo: str, branch: str) -> bool:
        resp = self.session.get(f"{API_ROOT}/repos/{owner}/{repo}/git/ref/heads/{branch}")
        if resp.status_code == 404:
            return False
        self._check(resp, f"checking for branch '{branch}'")
        return True

    def find_open_pr_for_branch(self, owner: str, repo: str, branch: str) -> dict | None:
        """Returns the open PR whose head is `branch`, if one exists. Used to
        avoid opening a duplicate PR when resuming work on an issue that
        already has one — pushing new commits to the same branch updates
        that existing PR automatically, no extra API call needed for that
        part."""
        resp = self.session.get(
            f"{API_ROOT}/repos/{owner}/{repo}/pulls",
            params={"head": f"{owner}:{branch}", "state": "open"},
        )
        self._check(resp, f"listing pull requests for branch '{branch}'")
        prs = resp.json()
        return prs[0] if prs else None

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

    # --- Used by transcript_store.py to persist full, untruncated round
    # data outside of comment bodies (comments are capped well below what a
    # single tool result can produce) ---

    def ensure_branch(self, owner: str, repo: str, branch: str) -> None:
        """Creates `branch` pointing at the default branch's current tip if
        it doesn't already exist. No-op if it does."""
        ref_resp = self.session.get(f"{API_ROOT}/repos/{owner}/{repo}/git/ref/heads/{branch}")
        if ref_resp.status_code == 200:
            return
        if ref_resp.status_code != 404:
            self._check(ref_resp, f"checking for branch '{branch}'")

        default_branch = self.get_default_branch(owner, repo)
        base_ref = self.session.get(f"{API_ROOT}/repos/{owner}/{repo}/git/ref/heads/{default_branch}")
        self._check(base_ref, f"reading default branch ref '{default_branch}'")
        sha = base_ref.json()["object"]["sha"]

        create_resp = self.session.post(
            f"{API_ROOT}/repos/{owner}/{repo}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": sha},
        )
        self._check(create_resp, f"creating branch '{branch}'")

    def get_file(self, owner: str, repo: str, path: str, ref: str) -> str:
        resp = self.session.get(
            f"{API_ROOT}/repos/{owner}/{repo}/contents/{path}", params={"ref": ref}
        )
        if resp.status_code == 404:
            raise FileNotFoundError(f"{path}@{ref} not found in {owner}/{repo}")
        self._check(resp, f"reading {path}@{ref}")
        return base64.b64decode(resp.json()["content"]).decode("utf-8")

    def put_file(self, owner: str, repo: str, path: str, content: str, branch: str, message: str) -> dict:
        """Creates or updates a file at `path` on `branch`. Each round's
        transcript path is unique (includes the Actions run ID and round
        number) so this is a create in the overwhelming common case; the
        update path only matters if the exact same run+round is written
        twice, e.g. a retried step."""
        body = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        resp = self.session.put(f"{API_ROOT}/repos/{owner}/{repo}/contents/{path}", json=body)
        if resp.status_code == 422:
            existing = self.session.get(
                f"{API_ROOT}/repos/{owner}/{repo}/contents/{path}", params={"ref": branch}
            )
            if existing.ok:
                body["sha"] = existing.json()["sha"]
                resp = self.session.put(f"{API_ROOT}/repos/{owner}/{repo}/contents/{path}", json=body)
        self._check(resp, f"writing {path}@{branch}")
        return resp.json()
