# Review prompt templates

Documentation only — these are templates for a human to paste into a *separate* review session. They do not change
any workflow in this repo. Language of the templates: match the review session's own convention (CN/EN both fine, per
the repo's mix rule).

## 1. Adversarial review

```
You are reviewing the diff <range or PR>. Assume the author is competent but the diff hides exactly three defects:
one P0 (correctness — wrong result, crash, data loss), one P1 performance regression, one P1 concurrency / edge-case
bug. Hunt for them. Rules:
- Cite every finding as file:line and explain the concrete failing scenario (inputs/state that trigger it).
- If after genuine effort you cannot find a defect in a category, say so explicitly: "P0: none found — searched
  <what you checked>". Never invent a defect to satisfy the quota, and never soften a real finding because the quota
  was already filled.
- Rank by blast radius, not by how easy the fix is.
Do not comment on style; the fast suite and lint already cover that.
```

## 2. Test quality review

```
You are reviewing only the tests in the diff. Flag exactly these five smells, each with file:line and a one-line
reason:
1. No-assertion tests (setup + call, nothing checked — including "asserts" that only check `is not None`).
2. Over-mocking (mocking the unit under test's collaborators so thoroughly the test passes against a broken
   implementation — name what SHOULD have been real).
3. Test that copies the implementation (same expression on both sides of the assert; a rewrite of the algorithm
   inside the test).
4. Happy-path-only coverage (no error branch, no boundary value, no empty/degenerate input).
5. `raises(...)` without matching the exception message/type detail (a `raises(ValueError)` that would also pass on
   an unrelated ValueError raised deeper).
For each smell found: the minimal change that fixes it. If a category has no instances, say "none found".
```

## 3. Mutation testing (how to run it, and the "surviving mutant = false coverage" rule)

Mutation testing is a *local investigation tool*, not a gate here (pinned in `requirements-dev.txt` /
`environment.yml` as a dev dependency, but **not** wired into CI or `scripts/check.sh`). `mutmut` 3.x reads its
config from `[tool.mutmut]` in `pyproject.toml` at import time — there is no `--paths-to-mutate` CLI flag any
more, so switch the target by editing `only_mutate` there:

```bash
mutmut run                    # baseline + every mutant (writes to gitignored mutants/)
mutmut results                # list surviving mutants
mutmut show <id>              # inspect one survivor
rm -rf mutants                # scratch dir; safe to delete at any time
```

Measured on `only_mutate = ["iopaint/helper.py"]` (2026-10, one 8-core machine): 662 mutants in **70 s**
(19.8 mutations/s), 517 killed / 143 survived / 2 timeout → **78.4 % kill rate**. Survivors cluster in
`adjust_mask` (29), `only_keep_largest_contour` (20), `download_model` (14), `gen_frontend_mask` (12) — image-path
and network-path code where tests assert the happy path only.

Two hard-won constraints, both baked into the `[tool.mutmut]` comments — do not "simplify" them away:

- **`only_mutate` must point at a file whose functions are undecorated.** mutmut 3.8 skips every function carrying
  a decorator (except a bare `@staticmethod`/`@classmethod`), see the `FunctionDef` branch in
  `mutmut/mutation/file_mutation.py`. `iopaint/schema.py` is nearly all `@property`/`@computed_field`/
  `@model_validator`, so it yields **0 usable mutants** and the run aborts with "could not find any test case for
  any mutant". `iopaint/helper.py` (22 top-level plain functions) works.
- **`test_missing_package_version_recorded_as_na` is deselected in the config.** It does `importlib.reload(runtime)`
  and thereby rebuilds the `GpuInfo` dataclass; every later `GpuInfo(...)` comparison then fails on class identity
  (`GpuInfo(0, "FakeGPU", 0) != GpuInfo(0, "FakeGPU", 0)`). Normal `pytest` never trips this because the test sits
  at the end of `test_runtime_gaps.py`; mutmut runs its per-mutant subset in set order and can put it first. Fix
  properly (restore the module once in a fixture) only if the test itself is being edited — that is business-test
  code, outside this doc's scope.

Judgment rule: **a surviving mutant is a test that claims coverage it does not deliver.** For each survivor, decide:

- The mutant is *equivalent* (behavior genuinely unchanged, e.g. `x + 0`)? → note it and move on.
- The mutant changes behavior no test notices? → that is a real gap; write the missing assertion, then re-run to see
  the mutant die.

Do not "fix" a survivor by deleting the mutant or skipping the test; fix it by making an assertion that distinguishes
the mutant from the original. Report: total mutants, killed, survived, equivalent, and the 3 highest-value new
assertions the survivors suggested.
