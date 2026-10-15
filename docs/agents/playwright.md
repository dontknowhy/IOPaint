# Playwright webui test details

Playwright webui tests must not draw on the canvas right after closing a Radix Select popup or a modal Dialog: both
leave `pointer-events: none` on `<body>` for the duration of their close animation (measured ~300ms), so every mouse
event falls through to `<html>` (`elementFromPoint` at the canvas centre returns `HTML`), the stroke is silently lost
and no `/inpaint` request ever fires — the symptom looks like the app is broken but a plain page draws fine.
`iopaint/tests/test_webui.py:_wait_interactive()` waits for `document.body.style.pointerEvents !== 'none'` before
interacting; call it after any Select/Dialog interaction that precedes drawing. Related: the Editor's bottom brush
slider only mounts once an image is loaded, so a test asserting on it must upload a picture first, and the brush
slider is told apart from the cv2 radius slider by `aria-valuemax` (200 vs 100).

General context: `test_webui.py` drives the real built frontend, so `iopaint/web_app` must exist (it `skip`s itself
otherwise) — see "The backend serves the built frontend" in AGENTS.md for how the dist gets copied there.
