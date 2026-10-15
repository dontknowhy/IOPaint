# Type checking (new code only — local only, never in CI)

Status: **local opt-in script + this doc**; a human decision fixed it as **local-only — mypy must never be wired
into the CI environment**. No mypy/pyright *config* exists in the repo (`pyproject.toml` only declares the build
backend), CI is unchanged, and nothing here runs automatically. `mypy==2.4.0` is pinned as a dev dependency in
`requirements-dev.txt` / `environment.yml` (dev tools); CI's `backend` job installs only `requirements.txt`, so
mypy stays out of CI by both decision and dependency.

## Why mypy never runs in CI

- Decided by the human: type checking is a local pre-commit habit, not a CI gate — do not add a mypy install or a
  `check_types.py` step to any workflow.
- Even setting the decision aside, the codebase carries substantial pre-existing typing debt (e.g. `iopaint/schema.py`
  alone reports ~10 errors from `@property` decorator stacking), so a whole-tree or whole-file gate would be red on
  day one.

## What exists now

`scripts/check_types.py` — a new-code-only checker:

- Computes changed files: `git diff --name-only <BASE> -- '*.py'` plus untracked files, filtered to `iopaint/**`
  (`<BASE>` defaults to `origin/main`, pass another ref as argv[1]).
- Runs mypy with a strict-ish subset on just those files:
  `--disallow-untyped-defs --check-untyped-defs --no-implicit-optional --ignore-missing-imports`
  (`--ignore-missing-imports` because plugin deps like onnxruntime/rembg ship no stubs; requiring stubs would mean
  adding dependencies).
- Parses `git diff -U0` hunks to know which line numbers are *newly added* in each changed file, and keeps only mypy
  errors anchored on those lines. Errors on pre-existing lines are counted and printed as info, never fatal — that is
  how "old files are not back-ported" is enforced mechanically.
- Exit 1 only when at least one error lands on an added line; exit 0 otherwise. `MYPY=/path/to/mypy` overrides the
  binary; run it with an interpreter that has the runtime deps so imports resolve:
  `--python-executable` is pinned to the running interpreter.

Usage (with the same interpreter you use to run IOPaint, so imports resolve):

```bash
python scripts/check_types.py              # vs origin/main
python scripts/check_types.py HEAD         # vs last commit
```

Caveat: an added line is attributed by file+line number only; moving an old function without editing it keeps its
errors attributed to its (now shifted) old lines — which is fine — but rewriting an old function *in place* counts
all its lines as added, so its pre-existing errors inside the rewritten region become fatal. That is deliberate:
rewriting a region is the moment to fix its types.

## Interpreter requirement

Running the type check must use an interpreter that has the required dependencies installed (this doc deliberately
does not name a specific environment — the point is that mypy and the runtime deps are importable there).
