"""
Tool definitions and executor for the coding agent.

Architecture note: this module is the entire "local agent" now. It does not
reason about anything — it has no model in it. It just:
  1. describes, in a provider-agnostic JSON-schema shape, which tools exist
  2. executes a tool call a cloud model asked for, against the sandboxed
     checkout via RepoTools/test_runner
  3. returns the result as plain text for the cloud model to read

All actual intelligence (deciding what to read, what to write, when it's
done) lives in the cloud provider. This module never calls read_file(),
write_file() etc. on its own initiative.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Callable

from . import repo_tools, test_runner

# Directories/files ignored when building the project tree/manifest sent to
# the cloud model, same spirit as repo_tools.SKIP_DIRS but also covering
# obvious non-source noise so the tree stays small enough to be useful.
IGNORE_TREE_DIRS = repo_tools.SKIP_DIRS | {"uploads", "logs", "tmp", ".idea", ".vscode"}
MAX_TREE_ENTRIES = 2000

# Every tool the cloud model may call, described once in OpenAI-style JSON
# schema. Both providers translate this shared list into their own
# tool-calling wire format, so adding a tool means editing exactly one place.
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_project_tree",
        "description": (
            "Return the repository's directory/file tree (paths only, no content). "
            "Always call this first, before reading any files, to orient yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern (e.g. 'src/**/*.php'). Defaults to everything.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. 'src/**/*.php'"},
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read the full text content of one file at the given repo-relative path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search all files for a regex or literal substring, returning matching lines with file:line.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Regex or literal text to search for"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a brand-new file with the given content. Fails if the file already exists — "
            "use edit_file to modify an existing file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace one exact, unique occurrence of old_str with new_str in an existing file. "
            "old_str must match the file's current content exactly (whitespace included) and "
            "must appear exactly once — include enough surrounding context to make it unique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_str": {"type": "string"},
                "new_str": {"type": "string"},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "run_tests",
        "description": (
            "Run this repo's test suite(s) deterministically — either generic detection based on "
            "project files present, or a hardcoded setup/command for repos that need it (you "
            "don't choose the command). May run more than one suite (e.g. a backend and a "
            "frontend test runner); all of them must pass for this to report success. Returns "
            "pass/fail per suite and an output tail."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "ask_user",
        "description": (
            "Ask the human a clarifying question about the feature/issue instead of guessing. "
            "This ends the current run: the question is posted on the issue and the agent stops "
            "until a human replies. Use sparingly — prefer inspecting the code yourself first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Signal that the requested feature/fix is fully implemented. Ends the run and opens "
            "the pull request. If this repository has a detected test suite, this call is "
            "refused (with an explanation, so you can act on it) unless run_tests has been "
            "called since your last file edit and reported ALL TEST SUITES PASSED — you'll need "
            "to call run_tests and fix any failures first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Human-readable summary of what was changed, for the PR description.",
                },
            },
            "required": ["summary"],
        },
    },
]


def build_project_tree(root: str) -> str:
    root_path = Path(root).resolve()
    lines: list[str] = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_TREE_DIRS and not d.startswith(".git"))
        rel_dir = Path(dirpath).relative_to(root_path)
        for name in sorted(filenames):
            rel = (rel_dir / name).as_posix() if str(rel_dir) != "." else name
            lines.append(rel)
            count += 1
            if count >= MAX_TREE_ENTRIES:
                lines.append(f"... truncated at {MAX_TREE_ENTRIES} entries")
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(empty repository)"


class ToolExecutionError(Exception):
    """Raised for tool-call shapes we refuse to execute (unknown tool, bad args)."""


class ToolExecutor:
    """Executes tool calls against a single sandboxed repo checkout.

    `on_terminal` is invoked when the model calls `ask_user` or `finish`,
    letting the orchestrator end the loop and post the right kind of comment
    without this class needing to know about GitHub at all.

    `target_repo` ("owner/name") is used only to look up a hardcoded
    per-repo test override (test_runner.REPO_OVERRIDES) — it's never used to
    let the model influence what command runs.
    """

    def __init__(
        self,
        repo_root: str,
        on_terminal: Callable[[str, dict], None] | None = None,
        target_repo: str | None = None,
    ):
        self.tools = repo_tools.RepoTools(repo_root)
        self.repo_root = repo_root
        self.on_terminal = on_terminal
        self.target_repo = target_repo
        self.finished = False
        self.finish_summary: str | None = None
        self.pending_question: str | None = None
        # None = not yet run this session; True/False = outcome of the most
        # recent run_tests call. finish is refused while this isn't True, if
        # a test plan is detected at all — see _test_plan().
        self.last_tests_passed: bool | None = None
        self._test_plan_cache = "unset"  # sentinel distinct from a real None (no plan found)

    def _test_plan(self):
        if self._test_plan_cache == "unset":
            self._test_plan_cache = test_runner.detect_test_plan(self.repo_root, self.target_repo)
        return self._test_plan_cache

    def execute(self, name: str, args: dict[str, Any]) -> str:
        args = args or {}
        try:
            if name == "get_project_tree":
                return build_project_tree(self.repo_root)
            if name == "list_files":
                return self.tools.list_files(args.get("pattern", "**/*"))
            if name == "read_file":
                return self.tools.read_file(args["path"])
            if name == "search_code":
                return self.tools.search_code(args["query"])
            if name == "write_file":
                self.last_tests_passed = None  # file changed since the last test run
                return self.tools.write_file(args["path"], args.get("content", ""))
            if name == "edit_file":
                self.last_tests_passed = None  # file changed since the last test run
                return self.tools.edit_file(args["path"], args["old_str"], args["new_str"])
            if name == "run_tests":
                plan = self._test_plan()
                if plan is None:
                    return "No recognized test suite/config found in this repository; nothing was run."
                setup_cmds, test_cmds, description = plan
                output = test_runner.run_tests(self.repo_root, setup_cmds, test_cmds)
                self.last_tests_passed = output.startswith("ALL TEST SUITES PASSED")
                return output
            if name == "ask_user":
                self.pending_question = args.get("question", "").strip() or "(no question text provided)"
                if self.on_terminal:
                    self.on_terminal("ask_user", args)
                return "Question recorded. Ending this run; a human will need to reply before the next run."
            if name == "finish":
                plan = self._test_plan()
                if plan is not None and not self.last_tests_passed:
                    return (
                        "ERROR: cannot finish yet — this repository has a detected test suite, but "
                        "it either hasn't been run since your most recent file change, or the last "
                        "run reported failures. Call run_tests, fix whatever it reports, and only "
                        "call finish once it reports ALL TEST SUITES PASSED. Do not open the pull "
                        "request with failing tests."
                    )
                self.finished = True
                self.finish_summary = args.get("summary", "").strip() or "(no summary provided)"
                if self.on_terminal:
                    self.on_terminal("finish", args)
                return "Marked as finished. Opening the pull request now."
        except KeyError as e:
            raise ToolExecutionError(f"Missing required argument {e} for tool '{name}'")
        except repo_tools.PathEscapeError as e:
            return f"ERROR: {e}"
        raise ToolExecutionError(f"Unknown tool: {name!r}")
