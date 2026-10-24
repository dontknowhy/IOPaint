# Frontend notes

- One zustand store in `web_app/src/lib/states.ts` (`persist` + `immer` + `createWithEqualityFn`); the `Settings` type
  lives there too, magic numbers in `web_app/src/lib/const.ts`. `model` is deliberately not persisted.
- `web_app/src/lib/api.ts` is a single axios instance whose response interceptor converts errors into the user-facing
  message (`getErrorMessage`). `API_ENDPOINT` is `VITE_BACKEND + "/api/v1"` in dev and `/api/v1` in production builds.
- socket.io: the server mounts `/ws` **before** the `/` static mount and requires `socketio_path="/ws/socket.io"`
  (`api.py:Api.__init__`); the client hardcodes the same path in `components/DiffusionProgress.tsx`. The `/ws` mount must stay
  ahead of `/`, or the static mount shadows it and progress events 404.
- `components/Editor.tsx` (~2000 lines) paints the mask into a separate offscreen canvas: committed strokes live in
  `lineGroups` while the in-progress stroke is in `strokeRef`, and the canvas is only redrawn as "committed + current
  stroke". The mask is scaled by `maskScale`, re-expanded to full image size in `generateMask` (`lib/utils.ts`) and
  encoded with `canvasToBlob` (not `toDataURL`). `generateMask` asks for a `willReadFrequently` context on purpose: a
  GPU-backed canvas loses its bitmap when Chromium's GPU process crashes, which reaches the backend as an empty mask.
  Backend `gen_frontend_mask` mirrors the `ffcc00bb` brush color.
- Shared hooks: `useDragResize`, `useImage`, `useInputImage`, `useHotkey`, `useResolution` (viewport breakpoints, not
  model resolution).
- The display canvas is capped by `IMAGE_CANVAS_MAX_PIXELS` (16Mi), so `ctx.setTransform` maps image coordinates onto a
  smaller bitmap that CSS stretches back to the original size. Take the transform from `fitCanvasBitmap` (`lib/utils.ts`),
  never from a single uniform scale: when the bitmap size rounds up but the drawn rect stays `width*scale`, the last row
  ends up ~55% covered and Firefox/WebKit paint it semi-transparent (alpha 142/128), which reads as a 1–2px transparent
  gradient strip along the image edge once CSS scales it back up.
