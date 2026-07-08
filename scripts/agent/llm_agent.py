"""
The agentic loop: repeatedly asks the local LLM for one JSON action, executes
it against the sandboxed RepoTools, feeds the result back as an observation,
and repeats until the model calls finish() or we run out of iterations/time.

Grammar-constrained decoding (via ACTION_SCHEMA) guarantees the JSON is
well-formed and uses a valid action name, but it can't guarantee the *values*
make sense (e.g. a path that doesn't exist) — that's handled by the tool
implementations returning a clear "ERROR: ..." string, which becomes the
next observation, so the model can course-correct on its own next turn
exactly the way it would from a real tool failure.
"""
from __future__ import annotations

import json
import time

from llama_cpp import Llama, LlamaGrammar

from .repo_tools import RepoTools, PathEscapeError
from .schema import ACTION_SCHEMA, SYSTEM_PROMPT, build_issue_context

MAX_OBSERVATION_CHARS = 6000


class AgentLoop:
    def __init__(
        self,
        llm: Llama,
        repo_tools: RepoTools,
        run_tests_fn,
        max_iterations: int = 40,
        max_seconds: int = 4 * 3600,
        max_tokens_per_turn: int = 1200,
    ):
        self.llm = llm
        self.tools = repo_tools
        self.run_tests_fn = run_tests_fn
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.max_tokens_per_turn = max_tokens_per_turn
        self.grammar = LlamaGrammar.from_json_schema(json.dumps(ACTION_SCHEMA))
        self.modified_files = set()
        self.tests_ever_run = False
        self.last_test_result = None

    def run(self, repo_full_name: str, issue: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_issue_context(repo_full_name, issue)},
        ]

        start = time.time()
        transcript = []

        for iteration in range(1, self.max_iterations + 1):
            if time.time() - start > self.max_seconds:
                transcript.append({"event": "timeout", "iteration": iteration})
                return self._result(
                    finished=False,
                    summary="Stopped: time budget exhausted before the agent called finish().",
                    transcript=transcript,
                )

            action = self._get_next_action(messages)
            if action is None:
                # Model produced something the grammar still couldn't rescue
                # (e.g. empty completion). Nudge once instead of crashing the job.
                messages.append(
                    {
                        "role": "user",
                        "content": "OBSERVATION: your last reply could not be parsed as the "
                        "required JSON action. Reply with exactly one valid JSON action.",
                    }
                )
                continue

            messages.append({"role": "assistant", "content": json.dumps(action)})
            transcript.append({"iteration": iteration, "action": action})

            act = action.get("action")
            if act == "finish":
                summary = action.get("summary") or "(agent did not provide a summary)"
                return self._result(finished=True, summary=summary, transcript=transcript)

            observation = self._dispatch(act, action)
            transcript.append({"iteration": iteration, "observation": observation[:500]})
            messages.append({"role": "user", "content": f"OBSERVATION:\n{observation}"})

        return self._result(
            finished=False,
            summary=f"Stopped: reached the {self.max_iterations}-iteration limit before "
            "the agent called finish().",
            transcript=transcript,
        )

    def _get_next_action(self, messages):
        try:
            resp = self.llm.create_chat_completion(
                messages=messages,
                grammar=self.grammar,
                temperature=0.2,
                max_tokens=self.max_tokens_per_turn,
            )
        except Exception as e:
            print(f"WARNING: local inference call failed: {e}")
            return None

        raw = resp["choices"][0]["message"]["content"].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"WARNING: could not parse model output as JSON: {raw[:300]!r}")
            return None

    def _dispatch(self, act: str, action: dict) -> str:
        try:
            if act == "list_files":
                return self.tools.list_files(action.get("pattern", "**/*"))
            if act == "read_file":
                return self.tools.read_file(action.get("path", ""))
            if act == "search_code":
                return self.tools.search_code(action.get("query", ""))
            if act == "write_file":
                path = action.get("path", "")
                result = self.tools.write_file(path, action.get("content", ""))
                self.modified_files.add(path)
                return result
            if act == "run_tests":
                self.tests_ever_run = True
                self.last_test_result = self.run_tests_fn()
                return self._truncate(self.last_test_result)
            return f"ERROR: unknown action '{act}'"
        except PathEscapeError as e:
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: tool '{act}' raised an unexpected exception: {e}"

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OBSERVATION_CHARS:
            return text
        return text[:MAX_OBSERVATION_CHARS] + "\n... [observation truncated]"

    def _result(self, finished: bool, summary: str, transcript: list) -> dict:
        return {
            "finished": finished,
            "summary": summary,
            "modified_files": sorted(self.modified_files),
            "tests_ever_run": self.tests_ever_run,
            "last_test_result": self.last_test_result,
            "transcript": transcript,
        }
