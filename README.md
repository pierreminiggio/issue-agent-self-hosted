# issue-agent-self-hosted

A GitHub Action that takes an issue number in one of your repos, clones that
repo, and runs an autonomous coding agent to resolve the issue — reading
files, searching the codebase, writing changes, running the project's own
test suite, and iterating — then pushes a branch and opens a **draft pull
request** with the result.

## Architecture

The "local agent" (this repo's Python code, running inside the GitHub
Actions runner) does **no reasoning of its own**. It has no model in it.
Its whole job is:

- fetch the issue and its comments, and refuse to run at all unless the
  issue was opened by a trusted account (see Security below)
- hand the conversation to a cloud model (Groq or Gemini) that can call
  tools (`get_project_tree`, `read_file`, `search_code`, `write_file`,
  `edit_file`, `run_tests`, `ask_user`, `finish`)
- execute whatever tool call the cloud model asks for, sandboxed to the
  repo checkout, and hand the result straight back
- post every round of that exchange to the issue as a comment, so you can
  watch the conversation happen in real time
- once the model calls `finish`, commit whatever changed and open a draft PR

```
scripts/develop_issue.py            entry point: trust check, clone, run, PR
scripts/agent/history.py            trust filtering + transcript comment format
scripts/agent/orchestrator.py       the round loop + provider failover
scripts/agent/tools.py              tool schema + the executor (repo_tools/test_runner wrapper)
scripts/agent/repo_tools.py         sandboxed list/read/search/write/edit file ops
scripts/agent/test_runner.py        static test-command detection + execution
scripts/agent/providers/base.py     provider-agnostic interface both LLMs implement
scripts/agent/providers/groq_provider.py
scripts/agent/providers/gemini_provider.py
scripts/agent/github_client.py      minimal GitHub REST API wrapper
```

Earlier versions of this project ran a local 7B GGUF model
(`llama-cpp-python`) as the reasoning engine, with a grammar-constrained
single-JSON-action loop to keep it from producing garbage. That's gone: a
cloud model with real tool-calling does the reasoning now, and the local
side is a thin, deterministic tool executor plus GitHub glue.

## Read this before using it

**Every PR is a draft, and every PR must be reviewed.** The agent's output
is never merged automatically. Review the diff like you would any other
contribution.

**Watch the conversation, not just the diff.** Every round — the model's
reasoning, which tool it called, and what came back — is posted as a
comment on the issue as it happens. If a run goes somewhere you don't
expect, you'll see it there before the PR even opens.

## Setup

1. **Create a PAT** for the agent to use. A fine-grained personal access
   token scoped to the repos you'll point this at, with:
   - Contents: Read and write
   - Pull requests: Read and write
   - Issues: Read and write

   Add it as a secret named `DEV_AGENT_PAT` in **this** repo. This is
   necessary regardless of who owns the target repo: the default
   `GITHUB_TOKEN` a workflow gets only has access to the repo the workflow
   itself lives in.

   **This PAT's own account is also the one trusted account this whole
   project runs on — see Security below.** Only issues (and comments) it
   itself authored are ever acted on.

2. **Add cloud provider keys** as secrets: `GROQ_API_KEY`, `GEMINI_API_KEY`,
   or both. At least one is required. If both are set, Groq is tried first
   each round and Gemini is used as a live failover if Groq is rate-limited
   or erroring — mid-conversation, with the same message history.

3. Push this repo to GitHub as-is.

4. **Open (or already have) the issue you want resolved — from the
   `DEV_AGENT_PAT` account itself.** See Security below for why.

5. Trigger it: **Actions → Develop GitHub Issue (Cloud-LLM Coding Agent) →
   Run workflow**, and fill in:
   - `repo`: `owner/name` of the repo with the issue
   - `issue_number`: the issue to resolve
   - `draft_pr` / `max_iterations`: optional, sensible defaults provided

## Talking to the agent, and resuming a run

Because the local side keeps no memory of its own between runs, all
conversation state lives on the issue itself, as comments:

- Each round the cloud model takes is posted as one comment: a hidden JSON
  block (so a later run can reconstruct the exact conversation) plus a
  human-readable rendition underneath it.
- If the model calls `ask_user`, the run stops and posts the question as a
  comment. **Reply on the issue from the `DEV_AGENT_PAT` account** with your
  answer, then re-run the workflow on the same issue — it will replay the
  full prior conversation (including your reply) and continue from there.
- Want to redirect a run that's already finished or is heading the wrong
  way? Post a plain comment (no special formatting needed) from the
  `DEV_AGENT_PAT` account and re-run the workflow.

## Security model — "trust the messenger"

Issue titles/bodies/comments are attacker-reachable input: on a public repo,
anyone can open an issue or comment on one. Rather than trying to sanitize
or detect adversarial text, this project sidesteps the problem by never
reading content whose author it doesn't already trust:

- **Exactly one GitHub account is trusted**: whoever owns `DEV_AGENT_PAT`.
- Before anything else, the agent fetches the issue and checks
  `issue.user.login`. **If the issue itself wasn't opened by that account,
  the run refuses entirely** — no comments are read, nothing is sent to a
  model, nothing is cloned. A short comment explaining the refusal is
  posted (safe, since it's the bot's own text, not a reflection of
  anything untrusted).
- Every comment on the issue is checked the same way, one by one, by author
  login *before* its body is read at all. Anything from a different account
  is discarded at that point and never reaches the model, a log line, or a
  file.
- Comments that do pass the check are split into two kinds: ones the agent
  itself posted (identified by an invisible marker, replayed as structured
  history) and everything else from that same trusted account (treated as
  plain instructions/feedback).

On top of that, the usual tool-level sandboxing still applies:

- **No shell/exec tool of any kind.** Only `get_project_tree`, `list_files`,
  `read_file`, `search_code`, `write_file`, `edit_file`, and a fixed
  `run_tests` are available.
- **File tools are sandboxed** to the cloned repo directory; absolute paths
  and `..` traversal are rejected before touching the filesystem.
- **The test command is statically detected, not model-supplied.**
- **PRs are opened as drafts** and clearly labeled as AI-generated.

This is a practical mitigation, not a formal security boundary — the runner
still has normal internet access, and a capable model could still write
code that does something bad if merged without review. What it removes is
the cheap, obvious attack: a hostile public issue or comment trying to
smuggle instructions to the model. It doesn't replace reviewing the diff.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` / `GEMINI_API_KEY` | — | at least one required; both enables failover |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | override Groq model |
| `GEMINI_MODEL` | `gemini-2.0-flash` | override Gemini model |
| `max_iterations` (input) | 40 | hard cap on tool-call rounds per run |
| `AGENT_TIME_BUDGET_SECONDS` | 6000 | wall-clock cap on the orchestrator loop |
| `draft_pr` (input) | true | set false if you're comfortable with non-draft PRs |

## Known limitations

- Edits to existing files are anchored, surgical replacements (`edit_file`:
  find an exact unique snippet, replace it), not full-file rewrites.
  `write_file` only ever creates brand-new files. This is deliberate: a
  full-file rewrite requires the model to faithfully reproduce every
  untouched line from memory, and on a large file that can silently drop
  content — anchored edits make that structurally impossible.
- No dependency installation beyond a best-effort `pip install -r
  requirements.txt` / `npm ci` before running tests; projects with more
  involved setup (databases, docker-compose, etc.) will likely show test
  failures unrelated to the agent's actual change.
- One issue per run; no batching multiple issues in one workflow dispatch.
- The trust model means only the `DEV_AGENT_PAT` account's own issues can be
  worked on — a contributor filing an issue on your behalf doesn't work
  unless you re-post it (or its content) yourself from that account.
