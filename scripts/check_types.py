#!/usr/bin/env python3
"""New-code-only type gate: run mypy on iopaint/**/*.py changed vs BASE (default origin/main).

Old code is never back-ported: a mypy error only fails the gate when it is anchored on a line
ADDED in the diff (tracked files via `git diff -U0`, untracked new files count as all-added).
Errors on pre-existing lines are printed as info and never fail.

Run it with an interpreter that has the runtime deps (pydantic, loguru, ...), e.g.:
    python scripts/check_types.py [BASE_REF]
mypy is taken from that same interpreter (`sys.executable -m mypy`); override it
with MYPY=/path/to/mypy.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys

MYPY_FLAGS = [
    "--disallow-untyped-defs",
    "--check-untyped-defs",
    "--no-implicit-optional",
    # third-party plugin deps (onnxruntime, rembg, ...) ship no stubs; requiring stubs would
    # make the gate unsatisfiable without adding dependencies.
    "--ignore-missing-imports",
]

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
ERR_RE = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error:")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True)


def changed_py_files(base: str) -> list[str]:
    tracked = git("diff", "--name-only", base, "--", "*.py").splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", "*.py").splitlines()
    return sorted({f for f in tracked + untracked if f.startswith("iopaint/")})


def is_tracked(path: str) -> bool:
    """Index/HEAD knows the file? Untracked files never show up in `git diff` output."""
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            capture_output=True,
        ).returncode
        == 0
    )


def added_lines(base: str, path: str) -> set[int]:
    """Line numbers (new-file coordinates) added vs base; empty set = unchanged.

    An untracked file is not in any diff, so every line of it counts as added —
    without this branch its mypy errors would be filed under "pre-existing".
    """
    if not is_tracked(path):
        with open(path, encoding="utf-8") as f:
            return set(range(1, sum(1 for _ in f) + 1))
    out: set[int] = set()
    for line in git("diff", "-U0", base, "--", path).splitlines():
        m = HUNK_RE.match(line)
        if m:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            out.update(range(start, start + count))
    return out


def mypy_command() -> list[str]:
    """Pick the mypy that belongs to the running interpreter.

    `shutil.which("mypy")` may find an unrelated distro binary built for
    another Python (/usr/bin/mypy on 3.14 here) whose parser chokes on
    site-packages code from this env. Prefer `sys.executable -m mypy` so the
    checker and `--python-executable` always come from the same env; keep
    MYPY=... as an escape hatch.
    """
    override = os.environ.get("MYPY")
    if override:
        return [override]
    if importlib.util.find_spec("mypy") is not None:
        return [sys.executable, "-m", "mypy"]
    exe = shutil.which("mypy")
    return [exe] if exe else []


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    files = changed_py_files(base)
    if not files:
        print(f"check_types: no changed iopaint/**/*.py vs {base} — nothing to check.")
        return 0

    mypy = mypy_command()
    if not mypy:
        print("check_types: mypy not found (set MYPY=/path/to/mypy).", file=sys.stderr)
        return 2

    added = {f: added_lines(base, f) for f in files}
    proc = subprocess.run(
        [*mypy, *MYPY_FLAGS, "--python-executable", sys.executable, *files],
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # 2 = usage/runtime error, not "type errors found"
        sys.stderr.write(proc.stdout + proc.stderr)
        return proc.returncode

    new_errors: list[str] = []
    old_errors = 0
    for line in proc.stdout.splitlines():
        m = ERR_RE.match(line)
        if not m:
            continue
        path, lineno = m.group("file"), int(m.group("line"))
        if path in added and lineno in added[path]:
            new_errors.append(line)
        else:
            old_errors += 1

    for line in new_errors:
        print(line)
    if old_errors:
        print(f"check_types: {old_errors} pre-existing error(s) on unchanged lines (ignored).")
    if new_errors:
        print(f"check_types: {len(new_errors)} error(s) on newly added lines — fix before committing.")
        return 1
    print(f"check_types: OK ({len(files)} changed file(s), no errors on added lines).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
