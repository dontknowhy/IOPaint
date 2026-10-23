# Type checking (new code only — wired into the CI `backend` job)

Status: **CI gate + local script + this doc**. The Phase 1 human decision ("local-only, mypy must never enter CI")
has been overturned by the human for Phase 2: `.github/workflows/ci.yml` installs `requirements-dev.txt` (pin
`mypy==2.4.0`) in the `backend` job and runs `scripts/check_types.py` as its own step, and `scripts/check.sh` runs
the same command locally. The repo's only mypy *config* is the `[tool.mypy]` stanza in `pyproject.toml`, which loads
the pydantic plugin; every strict flag still comes from the script. `mypy==2.4.0` stays pinned only in
`requirements-dev.txt` / `environment.yml` — never in `requirements.txt`, which remains runtime-only.

## What CI runs

- Job `backend`, step **`Type check (new code only)`**, after the fast suite:
  `python scripts/check_types.py ${{ github.event.pull_request.base.sha || 'origin/main' }}`
  (`check_types.py` defaults to `origin/main` when no ref is passed).
- **Base ref** — PR events use `pull_request.base.sha`, because a shallow PR clone fetches the merge ref and has no
  usable `origin/main`; push / `workflow_dispatch` events fall back to `origin/main`, which equals `HEAD` there, so
  those runs print "no changed … — nothing to check" and pass. The gate therefore bites on pull requests, which is
  where new code enters the tree.
- **Dependencies** — `pip install -r requirements-dev.txt` in that job; the pip cache key lists both
  `requirements.txt` and `requirements-dev.txt`, so a pin bump invalidates it. Installing dev tools does not change
  what the test suite itself sees.
- **Local mirror** — `scripts/check.sh` runs the same `scripts/check_types.py` invocation after the fast suite, but
  skips with a printed notice when mypy is not importable (same philosophy as `test_webui.py` self-skipping without
  playwright), so a runtime-only install still gets a green `check.sh`.
- Shape of a run: exit 0 when no `iopaint/**/*.py` changed vs the base, or when no mypy error lands on an added
  line; exit 1 when at least one does; exit 2 when mypy is missing (CI never hits 2).

## Why the whole tree is not checked

The codebase carries substantial pre-existing typing debt: with the gate's own flag set and the pydantic plugin, a
whole-tree run reports **3777 errors in 182 of 233 files** (`mypy iopaint --disallow-untyped-defs
--check-untyped-defs --no-implicit-optional --ignore-missing-imports`), e.g. `iopaint/schema.py` alone ~10 from
`@property` decorator stacking. Without the plugin the same run reports 6888, because it misreads pydantic
constructors such as `InpaintRequest()` as missing arguments. A whole-tree or whole-file gate would still be red on
day one. The gate's diff anchoring is what keeps it honest instead of vacuous: it only forgives errors on lines the
diff did not touch (a single-file run prints the rest as "pre-existing errors on unchanged lines (ignored)" — 2222
for `test_runtime_gaps.py`).

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
- Exit 1 only when at least one error lands on an added line; exit 0 otherwise. mypy is resolved from the running
  interpreter (`sys.executable -m mypy`) when importable, so it always comes from the same env as
  `--python-executable`; `MYPY=/path/to/mypy` overrides that choice. Run it with an interpreter that has the runtime
  deps so imports resolve.

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
