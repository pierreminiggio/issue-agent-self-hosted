"""
Entry point: given a target repo + issue number, verify the issue is
trusted, clone the repo, run the orchestrator (cloud-model-driven, tools
executed locally) against it, post the conversation to the issue as it
happens, and — if the model finished with changes — push a branch and open
a draft pull request.

Required environment variables:
  TARGET_REPO       "owner/name" of the repo to work on
  ISSUE_NUMBER      issue number to resolve
  GH_TOKEN          PAT with write access to TARGET_REPO (contents, pull
                     requests, issues). This PAT's own account is also the
                     sole trusted author for issue/comment content — see
                     agent/history.py.

At least one of:
  GROQ_API_KEY
  GEMINI_API_KEY

Optional:
  DRAFT_PR                  "true"/"false", default "true"
  MAX_ITERATIONS             default 40
  AGENT_TIME_BUDGET_SECONDS  default 14400 (4h) — leaves headroom under the
                              GitHub Actions 6h job limit for setup/git
                              operations either side of the loop
  GROQ_MODEL / GEMINI_MODEL  override default model per provider
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agent import history
from agent.github_client import GitHubClient
from agent.orchestrator import Orchestrator
from agent.providers.base import Provider
from agent.providers.gemini_provider import GeminiProvider
from agent.providers.groq_provider import GroqProvider
from agent.tools import ToolExecutor

WORKDIR = Path("target-repo")


def env(name: str, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        print(f"ERROR: required environment variable {name} is not set.")
        raise SystemExit(1)
    return val


def run_git(args: list, cwd=None, mask: str | None = None):
    """Runs a git command, raising with output on failure. `mask` (if given)
    is a secret substring redacted from any printed error output — used so a
    failed clone/push never leaks the PAT into the Actions log."""
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        out = proc.stdout + proc.stderr
        if mask:
            out = out.replace(mask, "***")
        print(f"ERROR: git {' '.join(args if not mask else ['<redacted>'])} failed:\n{out}")
        raise SystemExit(1)
    return proc.stdout


def build_providers() -> list[Provider]:
    providers: list[Provider] = []
    groq_key = env("GROQ_API_KEY")
    gemini_key = env("GEMINI_API_KEY")
    if groq_key:
        providers.append(GroqProvider(groq_key, model=env("GROQ_MODEL") or "llama-3.3-70b-versatile"))
    if gemini_key:
        providers.append(GeminiProvider(gemini_key, model=env("GEMINI_MODEL") or "gemini-2.0-flash"))
    if not providers:
        print("ERROR: at least one of GROQ_API_KEY or GEMINI_API_KEY must be set.")
        raise SystemExit(1)
    return providers


def main():
    target_repo = env("TARGET_REPO", required=True)
    issue_number = int(env("ISSUE_NUMBER", required=True))
    token = env("GH_TOKEN", required=True)
    draft_pr = env("DRAFT_PR", "true").lower() != "false"
    max_iterations = int(env("MAX_ITERATIONS", "40"))
    time_budget = int(env("AGENT_TIME_BUDGET_SECONDS", str(4 * 3600)))

    owner, repo = target_repo.split("/", 1)
    gh = GitHubClient(token)

    print("Identifying trusted account ...")
    trusted_login = gh.get_authenticated_login()

    print(f"Fetching issue #{issue_number} from {target_repo} ...")
    issue = gh.get_issue(owner, repo, issue_number)

    try:
        trusted_comments = history.filter_trusted(issue, trusted_login)
    except history.UntrustedIssueError as e:
        print(f"REFUSING TO RUN: {e}")
        gh.add_issue_comment(owner, repo, issue_number, history.render_refusal_comment(trusted_login))
        raise SystemExit(1)

    initial_messages = history.build_initial_messages(issue, trusted_comments)

    default_branch = gh.get_default_branch(owner, repo)
    branch_name = f"agent/issue-{issue_number}"
    clone_url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"

    print(f"Cloning {target_repo} (branch {default_branch}) ...")
    run_git(["clone", "--depth", "50", "--branch", default_branch, clone_url, str(WORKDIR)], mask=token)
    run_git(["checkout", "-b", branch_name], cwd=WORKDIR)
    run_git(["config", "user.name", "issue-agent[bot]"], cwd=WORKDIR)
    run_git(["config", "user.email", "issue-agent-bot@users.noreply.github.com"], cwd=WORKDIR)

    providers = build_providers()
    print(f"Configured providers (in failover order): {[p.name for p in providers]}")

    executor = ToolExecutor(str(WORKDIR))

    def post_comment(body: str) -> None:
        gh.add_issue_comment(owner, repo, issue_number, body)

    orchestrator = Orchestrator(
        providers=providers,
        executor=executor,
        post_comment=post_comment,
        max_rounds=max_iterations,
        max_seconds=time_budget,
    )

    print("Starting orchestrator ...")
    result = orchestrator.run(target_repo, issue_number, initial_messages)
    print(f"Orchestrator finished: status={result.status} rounds_used={result.rounds_used}")

    if result.status == "error":
        print(f"ERROR: {result.error}")
        gh.add_issue_comment(
            owner, repo, issue_number, f"⚠️ The coding agent stopped due to an error: {result.error}"
        )
        raise SystemExit(1)

    if result.status == "waiting_for_human":
        print(f"Waiting for human input: {result.question}")
        return  # question comment already posted by the orchestrator

    # status == "finished"
    diff = run_git(["status", "--porcelain"], cwd=WORKDIR)
    if not diff.strip():
        print("Agent finished but no files were changed; nothing to commit.")
        gh.add_issue_comment(
            owner, repo, issue_number,
            f"The coding agent finished without making any file changes.\n\nSummary: {result.summary}",
        )
        return

    run_git(["add", "-A"], cwd=WORKDIR)
    run_git(["commit", "-m", f"Automated fix for #{issue_number}\n\n{result.summary}"], cwd=WORKDIR)
    run_git(["push", "-u", clone_url, branch_name], cwd=WORKDIR, mask=token)

    body = (
        f"{result.summary}\n\n"
        f"---\n"
        f"⚠️ This pull request was generated automatically by a cloud-LLM-driven coding agent "
        f"in response to #{issue_number}. It has not been reviewed by a human. Please review "
        f"the diff carefully before merging.\n\n"
        f"The full step-by-step conversation between the local agent and the cloud model is "
        f"posted as comments on #{issue_number}."
    )
    pr = gh.create_pull_request(
        owner, repo,
        head=branch_name,
        base=default_branch,
        title=f"Fix #{issue_number}: {issue['title']}",
        body=body,
        draft=draft_pr,
    )
    print(f"Opened pull request: {pr['html_url']}")


if __name__ == "__main__":
    main()
