# issue-agent-self-hosted

A GitHub Action that takes an issue number in one of your repos, clones that
repo, and runs a **self-hosted, local LLM** as an autonomous coding agent to
resolve the issue — reading files, searching the codebase, writing changes,
running the project's own test suite, and iterating — then pushes a branch
and opens a **draft pull request** with the result. No calls to any external
AI API are made; everything runs inside the GitHub Actions runner.

This is the sibling of `patrimoi-transactions-extractor-self-hosted`, same
approach (`llama-cpp-python` + a quantized GGUF model on a free GitHub-hosted
runner) applied to a much harder, open-ended task.

## Read this before using it

**Model quality.** This runs a 7B coding model on CPU, not a frontier hosted
model. It's genuinely useful for small, well-described, self-contained
issues in a codebase (a bug with a clear repro, a small well-specified
feature, a missing validation, etc). It will struggle with large
architectural changes, ambiguous requirements, or issues that require
understanding a lot of the codebase at once. Treat it like a junior
contributor working alone overnight, not a replacement for you.

**Every PR is a draft, and every PR must be reviewed.** The agent's output
is never merged automatically. Review the diff like you would any other
contribution — the model can misunderstand the issue, write code that
"passes" trivially, or (rarely) do something odd if the issue text itself
contains confusing or adversarial instructions (see Security below).

**Time and cost.** This is free (GitHub-hosted `ubuntu-latest` runners,
public repos get free minutes), but slow: CPU inference plus an agent loop
that reads/writes/tests iteratively can run for a long time. The workflow
budgets up to ~4 hours for the agent loop itself (`AGENT_TIME_BUDGET_SECONDS`)
inside a 350-minute job timeout, comfortably under the 6-hour hard limit
GitHub imposes on hosted-runner jobs. Most issues will finish well before
that, but don't be surprised if a run takes 30–90 minutes.

## Setup

1. **Create a PAT** for the agent to use against your target repositories.
   A fine-grained personal access token scoped to just the repos you'll
   point this at, with:
   - Contents: Read and write
   - Pull requests: Read and write
   - Issues: Read and write

   Add it as a secret named `DEV_AGENT_PAT` in **this** repo (the one
   hosting the workflow) — not in the target repo.

   This is necessary regardless of who owns the target repo: the default
   `GITHUB_TOKEN` GitHub Actions gives a workflow only has access to the repo
   the workflow itself lives in, not other repos you point it at.

2. Push this repo to GitHub as-is (the `models/` directory is empty in git;
   the model is downloaded and cached by the workflow on first run).

3. Trigger it: **Actions → Develop GitHub Issue (Self-Hosted Coding Agent) →
   Run workflow**, and fill in:
   - `repo`: `owner/name` of the repo with the issue (must be a repo your
     `DEV_AGENT_PAT` can write to)
   - `issue_number`: the issue to resolve
   - `draft_pr` / `max_iterations`: optional, sensible defaults provided

## How it works

```
scripts/develop_issue.py        orchestration: fetch issue, clone repo,
                                 run the agent, commit/push/open PR
scripts/agent/schema.py         the JSON action schema + system prompt
scripts/agent/repo_tools.py     sandboxed list/read/search/write file tools
scripts/agent/test_runner.py    static test-command detection + execution
scripts/agent/llm_agent.py      the ReAct loop itself
scripts/agent/github_client.py  minimal GitHub REST API wrapper
```

Each turn, the model is asked for exactly one JSON action (`list_files`,
`read_file`, `search_code`, `write_file`, `run_tests`, or `finish`).
`llama-cpp-python`'s grammar-constrained decoding
(`LlamaGrammar.from_json_schema`) forces every completion to be valid JSON
using one of those action names — this matters much more here than it did
for the PDF extractor, because a broken turn in an agent *loop* would
otherwise either crash the job or let the model wander off into unstructured
prose instead of taking an action.

Test execution is auto-detected from the repo's own files (pytest/
pyproject.toml, package.json, go.mod, Cargo.toml, or a Makefile `test:`
target) — the agent can ask to run tests, but cannot choose what command
that means. If nothing is detected, `run_tests` just tells the model so, and
it proceeds on reading/review alone.

## Security model — why there's no shell tool

The issue title/body/comments are attacker-reachable input: anyone who can
open an issue on the target repo can put arbitrary text in front of the
model. If the agent had a generic "run this shell command" tool, a
malicious or compromised issue could try to get it to exfiltrate the PAT,
hit an external URL, or otherwise do something well outside "edit this
codebase." This project avoids that entire class of problem by not
exposing one:

- **No shell/exec tool of any kind.** Only `list_files`, `read_file`,
  `search_code`, `write_file`, and a fixed `run_tests` are available.
- **File tools are sandboxed** to the cloned repo directory; absolute paths
  and `..` traversal are rejected before touching the filesystem
  (`agent/repo_tools.py::_resolve`).
- **The test command is statically detected, not model-supplied** — the
  agent can trigger "run the tests" but never define what that command is.
- **PRs are opened as drafts** and clearly labeled as AI-generated, so
  review is expected, not optional.

This is a reasonable middle ground, not a formal security boundary — the
runner itself still has normal internet access (e.g. `pip`/`npm install`
during test setup can reach the network), and a sufficiently capable model
could still write source code that does something bad if a human then
merges and runs it without review. The mitigations above remove the cheap,
obvious attack (a hostile issue directly commanding a shell tool); they
don't replace reviewing the diff.

## Configuration

Environment/workflow inputs of interest (see the workflow file for the
full list):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_REPO` / `MODEL_FILE` | Qwen2.5-Coder-7B-Instruct, Q4_K_M | swap for another GGUF coding model if you want to trade speed for quality |
| `max_iterations` (input) | 40 | hard cap on agent tool calls per run |
| `AGENT_TIME_BUDGET_SECONDS` | 14400 (4h) | wall-clock cap on the agent loop itself |
| `draft_pr` (input) | true | set false if you're comfortable with non-draft PRs |
| `LLM_N_CTX` | 16384 | context window; raise if you hit truncated context on large repos (uses more RAM) |

### Using a bigger model / your own GPU runner later

If you ever add a self-hosted runner with a GPU, you don't need to change
anything about the agent logic — only:
- point `runs-on` at your runner label instead of `ubuntu-latest`,
- install a CUDA/Metal build of `llama-cpp-python` instead of the CPU wheel,
- optionally switch `MODEL_REPO`/`MODEL_FILE` to a larger model (e.g.
  Qwen2.5-Coder-14B or 32B) for noticeably better code quality.

## Known limitations

- Edits to existing files are anchored, surgical replacements (`edit_file`: find an exact
  unique snippet, replace it), not full-file rewrites. `write_file` only ever creates brand-new
  files and refuses to touch an existing one. This is deliberate: a full-file rewrite requires
  the model to faithfully reproduce every untouched line from memory, and on a large file over
  a long session that can silently drop content — anchored edits make it structurally
  impossible for an edit to touch anything outside the snippet it explicitly quoted.
- No dependency installation beyond a best-effort `pip install -r
  requirements.txt` / `npm ci` before running tests; projects with more
  involved setup (databases, docker-compose, etc.) will likely show test
  failures unrelated to the agent's actual change.
- One issue per run; no batching multiple issues in one workflow dispatch.
