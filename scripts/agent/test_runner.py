"""
Test command detection and execution.

Deliberately NOT model-chosen: the commands run here are picked by static,
deterministic rules — either generic detection based on which project files
are present, or a hardcoded per-repo override — never by letting the agent
supply its own shell string. That's the difference between "the repo
owner's own test suite runs" and "an attacker-controlled issue body gets to
run arbitrary commands in CI": the agent can ask us to run tests, but never
choose what running tests means.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 600
TAIL_CHARS = 4000

# Repos with test setup too particular for generic detection to get right —
# multi-step environment setup, more than one test runner that both need to
# pass, etc. Add an entry here rather than trying to make detect_test_command
# smart enough to infer it; the whole point of this file is that the agent
# never gets to choose these commands itself.
#
# pierreminiggio/cms: PHPUnit (tests/) + Jest (__tests__/), mirroring
# bin/test. index.php reads ./.htaccess directly at runtime (not committed —
# see AGENTS.md), so any functional test needs .htaccess-dev copied into
# place first; env.php isn't strictly required (H\AppEnv::load no-ops if
# missing) but is copied for parity with the documented local dev setup.
REPO_OVERRIDES: dict[str, dict] = {
    "pierreminiggio/cms": {
        "setup_cmds": [
            ["cp", ".htaccess-dev", ".htaccess"],
            ["cp", "env-dev-base.php", "env.php"],
            ["composer", "install", "--no-interaction", "--prefer-dist"],
            ["npm", "ci"],
        ],
        "test_cmds": [
            ["php", "vendor/bin/phpunit", "--color", "--bootstrap", "vendor/autoload.php", "tests/"],
            ["npm", "run", "test", "--silent"],
        ],
        "description": "pierreminiggio/cms: phpunit (tests/) + jest (__tests__/)",
    },
}


def detect_test_plan(root: str, target_repo: str | None = None):
    """Returns (setup_cmds: list[list[str]], test_cmds: list[list[str]], description: str) or None.

    Checks REPO_OVERRIDES first (exact "owner/name" match); falls back to
    generic file-based detection otherwise.
    """
    override = REPO_OVERRIDES.get(target_repo or "")
    if override:
        return override["setup_cmds"], override["test_cmds"], override["description"]

    detected = _detect_generic_test_command(root)
    if detected is None:
        return None
    setup_cmds, test_cmd, description = detected
    return setup_cmds, [test_cmd], description


def _detect_generic_test_command(root: str):
    """Returns (setup_cmds, test_cmd, description) or None.

    Checked in an order that puts unambiguous, language-specific config files
    first (composer.json, package.json, go.mod, Cargo.toml) and only falls
    back to Python/pytest if there's actual Python evidence — not just a
    folder happening to be named "tests", which plenty of non-Python
    projects (e.g. PHPUnit, Jest) also use. Matching on that folder name
    alone previously caused false positives on PHP/JS repos.
    """
    root = Path(root)

    # PHP (composer-based projects)
    if (root / "composer.json").exists():
        install = [["composer", "install", "--no-interaction", "--prefer-dist"]] if shutil.which("composer") else []
        has_test_script = False
        try:
            import json as _json
            data = _json.loads((root / "composer.json").read_text(encoding="utf-8"))
            has_test_script = "test" in (data.get("scripts") or {})
        except (OSError, ValueError):
            pass
        if has_test_script:
            return install, ["composer", "test"], "composer test"
        if (root / "phpunit.xml").exists() or (root / "phpunit.xml.dist").exists():
            # vendor/bin/phpunit won't exist until `composer install` runs (it's
            # normally gitignored), so we rely on the checked-in phpunit config
            # as the signal, not the binary's presence at detection time.
            phpunit_bin = root / "vendor" / "bin" / "phpunit"
            return install, ["php", str(phpunit_bin)], "phpunit"

    # Node / JS
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

    # Python: require actual Python evidence, not just a "tests" folder name,
    # since that name is also common in PHP/JS/Go projects.
    has_python_evidence = (
        (root / "pyproject.toml").exists()
        or (root / "setup.py").exists()
        or (root / "setup.cfg").exists()
        or (root / "requirements.txt").exists()
        or (root / "pytest.ini").exists()
        or any(root.glob("*.py"))
    )
    if has_python_evidence:
        install = [["pip", "install", "-q", "pytest"]]
        if (root / "requirements.txt").exists():
            install.append(["pip", "install", "-q", "-r", "requirements.txt"])
        if (root / "requirements-dev.txt").exists():
            install.append(["pip", "install", "-q", "-r", "requirements-dev.txt"])
        return install, ["python", "-m", "pytest", "-q"], "pytest"

    return None


def run_tests(root: str, setup_cmds, test_cmds, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Runs every setup command in order (best-effort — a failed setup step
    is logged but doesn't abort the run, in case it's non-essential), then
    every test command in order. ALL test commands must pass for the overall
    result to be considered passing — the summary line callers check for
    (`ALL TEST SUITES PASSED`) only appears when every one of them exited 0.
    """
    root = str(root)
    log_parts = []

    for cmd in setup_cmds:
        if not shutil.which(cmd[0]):
            log_parts.append(f"(skipping setup step, '{cmd[0]}' not available in this runner)")
            continue
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                log_parts.append(
                    f"Setup step {' '.join(cmd)} exited {proc.returncode} (continuing anyway):\n"
                    + (proc.stdout[-1000:] + proc.stderr[-1000:])
                )
        except subprocess.TimeoutExpired:
            log_parts.append(f"Setup step {' '.join(cmd)} timed out; continuing anyway.")
        except OSError as e:
            log_parts.append(f"Setup step {' '.join(cmd)} failed to start: {e}")

    all_passed = True
    for test_cmd in test_cmds:
        if not shutil.which(test_cmd[0]):
            log_parts.append(
                f"ERROR: test command '{test_cmd[0]}' is not available in this runner; "
                "this suite could not be executed."
            )
            all_passed = False
            continue
        try:
            proc = subprocess.run(test_cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
            status = "PASSED" if proc.returncode == 0 else f"FAILED (exit code {proc.returncode})"
            if proc.returncode != 0:
                all_passed = False
            output = (proc.stdout + "\n" + proc.stderr).strip()
            tail = output[-TAIL_CHARS:]
            log_parts.append(f"Test run ({' '.join(test_cmd)}): {status}\n--- output tail ---\n{tail}")
        except subprocess.TimeoutExpired:
            log_parts.append(f"Test run ({' '.join(test_cmd)}) timed out after {timeout}s.")
            all_passed = False
        except OSError as e:
            log_parts.append(f"Test run ({' '.join(test_cmd)}) failed to start: {e}")
            all_passed = False

    summary = "ALL TEST SUITES PASSED" if all_passed else "AT LEAST ONE TEST SUITE FAILED"
    return summary + "\n\n" + "\n\n".join(log_parts)
