# Coding principles

How every change in this repo is written: the change should look as if the project's original author wrote it, not like
an attempt to show off your ability. When unsure, choose "less, dumber, closer to the status quo". AGENTS.md carries the
one-paragraph summary; this file is the long form.

## 1. Read before writing

- Before touching anything, read at least: the target file, 1–2 files of the same kind, and the related tests. Stop once
  that is enough; do not scan the whole repo.
- Reuse existing helpers / constants / types; re-inventing a wheel is forbidden.
- When no existing pattern covers the task, state your approach in the reply first, then start.

## 2. Minimal change

- Change only the code directly related to the task. Do not "optimize" unrelated code along the way (unless the human
  **specifically** asked), do not re-sort imports, do not reformat.
- One task, one thing; do not expand the scope along the way. A bug spotted nearby gets recorded, not fixed in passing
  (same exception as above).
- Keep the code as simple as possible: clear structure, no redundant abstraction — fewer characters or a clever trick is
  not the point; readability beats line count.
- For a large change, present the approach and the impact surface first; start once confirmed.
- If 3 edited lines do, do not rewrite the function; but **never twist the code to keep the diff small** — rewrite when a
  rewrite is right, and the price is explaining clearly why.

## 3. Follow existing conventions

- Naming, directory structure, error handling, logging and test style always follow the existing code.
- Do not introduce a new abstraction layer, new dependency, or new design pattern unless the task explicitly requires
  it.
- A new dependency needs a stated reason and consent first.

## 4. Against over-engineering

- No interfaces, parameters or config options reserved for "possible future needs".
- No helper with a single call site; no factory / strategy / dependency injection for simple logic.
- No docstrings, type annotations, logging or try-catch on functions that were not asked for.
- No defensive code to handle "impossible" inputs.
- Comments are written only when the *why* is not obvious; never restate what the code does.

Repo adaptation: the type-annotation bullet means "no annotations the neighbouring code does not already carry". This
repo has a new-code-only mypy gate (`scripts/check_types.py`, see docs/agents/type-check.md), so match the file's existing
style rather than typing an untyped module, and keep new or changed lines type-clean where the file is already typed.

## 5. Honesty and boundaries

- For an API, library version or config option you are unsure about, verify it first or say explicitly "I'm not sure";
  fabricating is forbidden.
- Do not modify code you have not understood. Do not delete code that "looks useless" (it may be referenced
  dynamically).
- State every implicit assumption (input range, concurrency, null, ordering) explicitly in the reply.
- Do not change tests on your own to make them pass; when a test fails, suspect the implementation first.
- Do not run destructive commands (`rm -rf`, force push, database changes) unless explicitly authorized.

## 6. Tests

- Bug fix: write the test that reproduces it first, then fix. When done, a test that can fail must be green.
- New feature: cover the normal path + 1–2 edges; 100% coverage is not a goal.
- A test must be able to fail. Do not write assertions that are always true.

Repo adaptation: the fast offline suite is `pytest -m "not slow"` from the repo root; weight-heavy files declare a
module-level `pytestmark = pytest.mark.slow` (see AGENTS.md → Tests, and docs/agents/slow-tests.md before running them).
Frontend tests are Vitest, under `cd web_app`.

## 7. Delivery

- Self-check after finishing: does lint / type check / test pass (`bash scripts/check.sh`: fast pytest + npm lint + npm
  test + build, then the new-code type check — see AGENTS.md → Health check)? Are there unrelated edits in the diff?
- Say what you changed, why, and what you did not do or are unsure about.
- If the task itself is ambiguous, ask first; do not guess your way through a big pile.
