"""
Test command detection and execution.

Deliberately NOT model-chosen: the command run here is picked by static,
deterministic rules based on which project files are present, never by
letting the agent supply its own shell string. That's the difference between
"the repo owner's own test suite runs" and "an attacker-controlled issue body
gets to run arbitrary commands in CI" — the agent can ask us to run tests,
but not choose what running tests means.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 600
TAIL_CHARS = 4000


def detect_test_command(root: str):
    """Returns (install_cmds: list[list[str]], test_cmd: list[str], description: str) or None."""
    root = Path(root)

    if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists() or (root / "tests").is_dir():
        install = []
        if (root / "requirements.txt").exists():
            install.append(["pip", "install", "-q", "-r", "requirements.txt"])
        if (root / "requirements-dev.txt").exists():
            install.append(["pip", "install", "-q", "-r", "requirements-dev.txt"])
        return install, ["python", "-m", "pytest", "-q"], "pytest"

    if (root / "package.json").exists():
        install = [["npm", "ci", "--silent"]] if (root / "package-lock.json").exists() else [["npm", "install", "--silent"]]
        return install, ["npm", "test", "--silent"], "npm test"

    if (root / "go.mod").exists() and shutil.which("go"):
        return [], ["go", "test", "./..."], "go test"

    if (root / "Cargo.toml").exists() and shutil.which("cargo"):
        return [], ["cargo", "test"], "cargo test"

    if (root / "Makefile").exists():
        try:
            makefile_text = (root / "Makefile").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            makefile_text = ""
        if "\ntest:" in ("\n" + makefile_text):
            return [], ["make", "test"], "make test"

    return None


def run_tests(root: str, install_cmds, test_cmd, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    root = str(root)
    log_parts = []

    for cmd in install_cmds:
        if not shutil.which(cmd[0]):
            log_parts.append(f"(skipping install step, '{cmd[0]}' not available in this runner)")
            continue
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                log_parts.append(
                    f"Install step {' '.join(cmd)} exited {proc.returncode} (continuing anyway):\n"
                    + (proc.stdout[-1000:] + proc.stderr[-1000:])
                )
        except subprocess.TimeoutExpired:
            log_parts.append(f"Install step {' '.join(cmd)} timed out; continuing anyway.")
        except OSError as e:
            log_parts.append(f"Install step {' '.join(cmd)} failed to start: {e}")

    if not shutil.which(test_cmd[0]):
        return (
            "\n".join(log_parts)
            + f"\nERROR: test command '{test_cmd[0]}' is not available in this runner, "
            "so tests could not actually be executed."
        )

    try:
        proc = subprocess.run(test_cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
        status = "PASSED" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
        output = (proc.stdout + "\n" + proc.stderr).strip()
        tail = output[-TAIL_CHARS:]
        log_parts.append(f"Test run ({' '.join(test_cmd)}): {status}\n--- output tail ---\n{tail}")
    except subprocess.TimeoutExpired:
        log_parts.append(f"Test run ({' '.join(test_cmd)}) timed out after {timeout}s.")
    except OSError as e:
        log_parts.append(f"Test run ({' '.join(test_cmd)}) failed to start: {e}")

    return "\n".join(log_parts)
