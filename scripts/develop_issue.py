"""
Entry point: given a target repo + issue number, clone the repo, run the
local-LLM coding agent against it, and (if it produced changes) push a branch
and open a draft pull request back on that repo.

Required environment variables:
  TARGET_REPO       "owner/name" of the repo to work on
  ISSUE_NUMBER      issue number to resolve
  GH_TOKEN          PAT with write access to TARGET_REPO (contents, pull
                     requests, issues)
  MODEL_PATH        path to the local GGUF model file

Optional:
  DRAFT_PR                  "true"/"false", default "true"
  MAX_ITERATIONS             default 40
  AGENT_TIME_BUDGET_SECONDS  default 14400 (4h) — leaves headroom under the
                              GitHub Actions 6h job limit for setup/model
                              download/git operations either side of the loop
  LLM_N_CTX                  default 16384
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from llama_cpp import Llama

from agent.github_client import GitHubClient
from agent.llm_agent import AgentLoop
from agent.repo_tools import RepoTools
from agent.test_runner import detect_test_command, run_tests

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
        out = (proc.stdout + proc.stderr)
        if mask:
            out = out.replace(mask, "***")
        print(f"ERROR: git {' '.join(args if not mask else ['<redacted>'])} failed:\n{out}")
        raise SystemExit(1)
    return proc.stdout


def main():
    target_repo = env("TARGET_REPO", required=True)
    issue_number = int(env("ISSUE_NUMBER", required=True))
    token = env("GH_TOKEN", required=True)
    model_path = env("MODEL_PATH", required=True)
    draft_pr = env("DRAFT_PR", "true").lower() != "false"
    max_iterations = int(env("MAX_ITERATIONS", "40"))
    time_budget = int(env("AGENT_TIME_BUDGET_SECONDS", str(4 * 3600)))
    n_ctx = int(env("LLM_N_CTX", "16384"))

    owner, repo = target_repo.split("/", 1)

    if not os.path.exists(model_path):
        print(f"ERROR: model file not found at {model_path}")
        raise SystemExit(1)

    gh = GitHubClient(token)

    print(f"Fetching issue #{issue_number} from {target_repo} ...")
    issue = gh.get_issue(owner, repo, issue_number)
    default_branch = gh.get_default_branch(owner, repo)

    branch_name = f"agent/issue-{issue_number}"
    clone_url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"

    print(f"Cloning {target_repo} (branch {default_branch}) ...")
    run_git(["clone", "--depth", "50", "--branch", default_branch, clone_url, str(WORKDIR)], mask=token)
    run_git(["checkout", "-b", branch_name], cwd=WORKDIR)
    run_git(["config", "user.name", "issue-agent[bot]"], cwd=WORKDIR)
    run_git(["config", "user.email", "issue-agent-bot@users.noreply.github.com"], cwd=WORKDIR)

    detected = detect_test_command(str(WORKDIR))
    if detected:
        install_cmds, test_cmd, description = detected
        print(f"Detected test runner: {description}")

        def run_tests_fn():
            return run_tests(str(WORKDIR), install_cmds, test_cmd)
    else:
        print("No test runner detected; the agent will proceed without automated verification.")

        def run_tests_fn():
            return (
                "No test runner could be detected for this project (checked for pytest/"
                "pyproject, package.json, go.mod, Cargo.toml, Makefile). Tests were not run."
            )

    print(f"Loading model from {model_path} (n_ctx={n_ctx}) ...")
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=os.cpu_count(), verbose=False)

    tools = RepoTools(str(WORKDIR))
    loop = AgentLoop(
        llm=llm,
        repo_tools=tools,
        run_tests_fn=run_tests_fn,
        max_iterations=max_iterations,
        max_seconds=time_budget,
    )

    print("Starting agent loop ...")
    result = loop.run(target_repo, issue)
    print(f"Agent loop finished. finished={result['finished']} modified_files={result['modified_files']}")

    if not result["modified_files"]:
        message = (
            "The self-hosted coding agent looked at this issue but did not make any code "
            f"changes.\n\nReason given: {result['summary']}"
        )
        print("No files modified; commenting on the issue instead of opening a PR.")
        gh.add_issue_comment(owner, repo, issue_number, message)
        return

    diff = run_git(["status", "--porcelain"], cwd=WORKDIR)
    if not diff.strip():
        print("Agent reported modified files but working tree is clean; nothing to commit.")
        gh.add_issue_comment(
            owner, repo, issue_number,
            "The self-hosted coding agent attempted changes for this issue but no net "
            "difference remained afterward.",
        )
        return

    run_git(["add", "-A"], cwd=WORKDIR)
    run_git(["commit", "-m", f"Automated fix for #{issue_number}\n\n{result['summary']}"], cwd=WORKDIR)
    run_git(["push", "-u", clone_url, branch_name], cwd=WORKDIR, mask=token)

    test_note = ""
    if result["tests_ever_run"] and result["last_test_result"]:
        test_note = f"\n\n---\n**Last test run:**\n```\n{result['last_test_result'][-2000:]}\n```"
    elif not result["tests_ever_run"]:
        test_note = "\n\n---\n**Note:** the agent never ran the test suite before finishing."

    body = (
        f"{result['summary']}\n\n"
        f"---\n"
        f"⚠️ This pull request was generated automatically by a self-hosted, local LLM coding "
        f"agent in response to #{issue_number}. It has not been reviewed by a human. Please "
        f"review the diff carefully before merging.\n\n"
        f"Modified files: {', '.join(result['modified_files'])}"
        f"{test_note}"
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
