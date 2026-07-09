"""
The JSON schema below is compiled into a GBNF grammar via
LlamaGrammar.from_json_schema(), which forces llama.cpp to only ever sample
tokens that keep the output valid against this schema. This matters a lot
for a small local model driving an agent loop: without it, a 7B model will
periodically emit malformed JSON, prose instead of an action, or an action
name that doesn't exist, any of which would otherwise crash or stall the
loop. The grammar makes those failure modes structurally impossible instead
of something we have to detect and retry after the fact.

We deliberately keep this flat (one object, one action per turn) rather than
a oneOf-per-action union: unions compile to much larger/slower grammars in
llama.cpp and small models are also just more reliable at filling in a fixed
set of optional fields than at picking the right branch of a union first.
Unused fields for a given action are simply ignored by the dispatcher in
llm_agent.py.
"""

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "One or two sentences of reasoning about what to do next and why.",
        },
        "action": {
            "type": "string",
            "enum": [
                "list_files",
                "read_file",
                "search_code",
                "write_file",
                "edit_file",
                "run_tests",
                "finish",
            ],
        },
        "path": {
            "type": "string",
            "description": "Repo-relative file path. Used by read_file, write_file, and edit_file.",
        },
        "content": {
            "type": "string",
            "description": "Full contents of a brand-new file. Used by write_file only, "
            "which can only create files that don't already exist.",
        },
        "old_str": {
            "type": "string",
            "description": "Exact, unique, contiguous snippet to find in the existing file. "
            "Used by edit_file only.",
        },
        "new_str": {
            "type": "string",
            "description": "Text to replace old_str with. Used by edit_file only.",
        },
        "pattern": {
            "type": "string",
            "description": "Glob pattern, e.g. 'src/**/*.py'. Used by list_files.",
        },
        "query": {
            "type": "string",
            "description": "Text or regex to search for across the repo. Used by search_code.",
        },
        "summary": {
            "type": "string",
            "description": "Human-readable summary of the fix, for the pull request "
            "description. Used by finish only.",
        },
    },
    "required": ["thought", "action"],
}

# Kept separate from the schema's own "description" fields (which the grammar
# doesn't enforce the model to *read*, only to be consistent with) so the
# tool contract is spelled out once, in plain language, in the system prompt
# the model actually reasons over.
SYSTEM_PROMPT = """You are an autonomous coding agent. You have been given a git repository \
checked out locally and a GitHub issue describing a bug or feature request. Your job is to \
make the necessary code changes in the repository to resolve the issue.

You do not have shell access and cannot run arbitrary commands. You have exactly these tools, \
invoked by replying with ONE JSON object per turn (no prose outside the JSON):

- list_files: list files in the repo matching a glob "pattern" (default "**/*" if omitted). \
Use this first to understand the project layout.
- read_file: read the contents of the file at "path". Large files are truncated; if you need \
a different part of a large file, note that in your next thought.
- search_code: search the whole repo for a literal string or regex given in "query". Returns \
matching file paths with line numbers and a snippet of context. Use this to find where \
something is defined or used before editing it.
- write_file: create a BRAND-NEW file at "path" with the complete "content". This only works if \
the file does not already exist — it will refuse to overwrite an existing file. To modify a \
file that already exists, use edit_file instead.
- edit_file: modify an EXISTING file by replacing one exact snippet with another. Give "path", \
"old_str" (the exact text currently in the file that you want to change — copy it precisely, \
whitespace and indentation included, from a prior read_file or search_code result) and \
"new_str" (what it should become). old_str must match exactly once in the file; if the same \
text appears more than once, include more surrounding lines so it's unique. This changes only \
the quoted snippet — every other line in the file is left exactly as it was. NEVER try to \
reconstruct or retype an entire existing file from memory; always use edit_file for changes to \
existing files, even for a large file, even if you only saw part of it — you don't need to \
remember the rest of the file to safely edit one part of it.
- run_tests: runs this project's test suite (already auto-detected for you; you don't choose \
the command). Returns pass/fail and a tail of the output. Use this after making changes to \
check your work. If no test runner could be detected, this will tell you so — in that case, \
rely on careful reading instead.
- finish: call this when you believe the issue is resolved (or when you've made the best \
partial progress you can). Provide a "summary" describing what you changed and why, written as \
a pull request description. This ends the session — you cannot take further actions after this.

Rules:
- Make the smallest change that correctly resolves the issue. Don't refactor unrelated code.
- Do NOT modify dependency manifests, build configuration, or test-tooling setup (e.g. \
requirements.txt, package.json, composer.json, CI config) in order to make test tooling run, \
unless the issue is explicitly about that. If run_tests fails for environment/tooling reasons \
unrelated to your actual code change (missing interpreter, missing package, wrong language \
tooling entirely), say so plainly in your finish summary instead of trying to install or \
configure your way around it.
- Prefer reading and searching before writing — understand existing conventions (naming, \
style, error handling, existing tests) and follow them.
- If the project has tests, run them after your changes and try to fix any failures your \
change introduced before finishing. If tests fail for reasons clearly unrelated to your change \
(e.g. missing dependencies in this sandbox), say so in your finish summary rather than looping \
forever trying to fix the sandbox itself.
- You have a limited number of turns. Don't spend many turns re-reading the same file — take \
notes in your "thought" instead.
- If, after reasonable exploration, the issue is unclear or you cannot determine a safe fix, \
call finish and explain what's blocking you rather than guessing wildly or making unrelated \
changes.

Reply with ONLY the JSON object for your next action. No markdown fences, no extra text.
"""


def build_issue_context(repo_full_name: str, issue: dict) -> str:
    """Renders the issue (and its comments, if any) into the first user message."""
    lines = [
        f"Repository: {repo_full_name}",
        f"Issue #{issue['number']}: {issue['title']}",
        "",
        "Issue description:",
        issue.get("body") or "(no description provided)",
    ]
    comments = issue.get("comments_data") or []
    if comments:
        lines.append("\nDiscussion on the issue:")
        for c in comments:
            author = c.get("user", {}).get("login", "unknown")
            lines.append(f"- {author}: {c.get('body', '')}")
    lines.append(
        "\nBegin by listing files to understand the project layout, then investigate the "
        "relevant area before making changes."
    )
    return "\n".join(lines)
