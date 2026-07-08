"""
Sandboxed file operations the agent is allowed to perform against the cloned
target repo. Every entry point resolves its path against `root` and refuses
to operate outside it, so a model that's been prompt-injected (e.g. via the
issue body) into writing e.g. "../../../etc/something" or an absolute path
cannot escape the checkout, and cannot see anything outside it either.

There is deliberately no generic shell/exec tool anywhere in this module —
see README.md for why.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

# Directories we never walk into: VCS internals, dependency/build output, and
# caches. These are large, rarely what an issue is actually about, and would
# otherwise blow past list_files/search_code result limits with noise.
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "venv", ".venv", "__pycache__",
    "dist", "build", "target", ".next", ".cache", ".pytest_cache",
    "coverage", ".mypy_cache", ".tox", "egg-info",
}

MAX_READ_CHARS = 12_000
MAX_LIST_RESULTS = 300
MAX_SEARCH_RESULTS = 60
# Skip anything obviously binary/huge rather than trying to read it as text.
MAX_SEARCHABLE_FILE_BYTES = 1_000_000


class PathEscapeError(Exception):
    pass


class RepoTools:
    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def _resolve(self, rel_path: str) -> Path:
        # Reject absolute paths outright rather than letting Path() silently
        # treat them as absolute and escape root on join.
        if not rel_path or os.path.isabs(rel_path):
            raise PathEscapeError(f"Path must be relative to the repo root: {rel_path!r}")
        candidate = (self.root / rel_path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise PathEscapeError(f"Path escapes the repository: {rel_path!r}")
        return candidate

    def _iter_files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".git")]
            for name in filenames:
                yield Path(dirpath) / name

    def list_files(self, pattern: str = "**/*") -> str:
        pattern = pattern or "**/*"
        matches = []
        for path in self._iter_files():
            rel = path.relative_to(self.root).as_posix()
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern.lstrip("*/")):
                matches.append(rel)
            if len(matches) >= MAX_LIST_RESULTS:
                break
        matches.sort()
        if not matches:
            return f"No files matched pattern '{pattern}'."
        suffix = "" if len(matches) < MAX_LIST_RESULTS else f"\n... truncated at {MAX_LIST_RESULTS} results"
        return "\n".join(matches) + suffix

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.exists():
            return f"ERROR: file not found: {path}"
        if target.is_dir():
            return f"ERROR: {path} is a directory, not a file. Use list_files on it instead."
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"ERROR: could not read {path}: {e}"
        if len(text) > MAX_READ_CHARS:
            return (
                text[:MAX_READ_CHARS]
                + f"\n... [truncated, file is {len(text)} chars total; "
                "consider search_code to find the relevant section instead]"
            )
        return text

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content if content is not None else "", encoding="utf-8")
        return f"Wrote {len(content or '')} chars to {path}"

    def search_code(self, query: str) -> str:
        if not query:
            return "ERROR: search_code requires a non-empty query."
        try:
            regex = re.compile(query)
        except re.error:
            # Fall back to a literal substring search if the model's query
            # isn't valid regex (e.g. contains unescaped special chars).
            regex = re.compile(re.escape(query))

        results = []
        for path in self._iter_files():
            try:
                if path.stat().st_size > MAX_SEARCHABLE_FILE_BYTES:
                    continue
            except OSError:
                continue
            rel = path.relative_to(self.root).as_posix()
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            results.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                            if len(results) >= MAX_SEARCH_RESULTS:
                                break
            except (UnicodeDecodeError, OSError):
                continue
            if len(results) >= MAX_SEARCH_RESULTS:
                break

        if not results:
            return f"No matches for '{query}'."
        suffix = "" if len(results) < MAX_SEARCH_RESULTS else f"\n... truncated at {MAX_SEARCH_RESULTS} matches"
        return "\n".join(results) + suffix
