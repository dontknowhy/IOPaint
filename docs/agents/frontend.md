# Frontend notes

- One zustand store in `web_app/src/lib/states.ts` (`persist` + `immer` + `createWithEqualityFn`); the `Settings` type
  lives there too, magic numbers in `web_app/src/lib/const.ts`. `model` is deliberately not persisted.
- `web_app/src/lib/api.ts` is a single axios instance whose response interceptor converts errors into the user-facing
  message (`getErrorMessage`). `API_ENDPOINT` is `VITE_BACKEND + "/api/v1"` in dev and `/api/v1` in production builds.
- socket.io: the server mounts `/ws` **before** the `/` static mount and requires `socketio_path="/ws/socket.io"`
  (`api.py:Api.__init__`); the client hardcodes the same path in `components/DiffusionProgress.tsx`. The `/ws` mount must stay
  ahead of `/`, or the static mount shadows it and progress events 404.
- `components/Editor.tsx` (~1900 lines) paints the mask into a separate offscreen canvas: committed strokes live in
  `lineGroups` while the in-progress stroke is in `strokeRef`, and the canvas is only redrawn as "committed + current
  stroke". The mask is scaled by `maskScale`, re-expanded to full image size in `generateMask` (`lib/utils.ts`) and
  encoded with `canvasToBlob` (not `toDataURL`). Backend `gen_frontend_mask` mirrors the `ffcc00bb` brush color.
- Shared hooks: `useDragResize`, `useImage`, `useInputImage`, `useHotkey`, `useResolution` (viewport breakpoints, not
  model resolution).
