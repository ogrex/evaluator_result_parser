# Viewer smoke tests

Headless-Chrome regression gate for the Three.js viewer. Run it before and
after any change to `templates/viewer_three.html`, `static/viewer_three/app/*`,
or the `/viewer/three/*` endpoints.

```bash
# one-time fixture
.venv/bin/python tools/viewer_smoke/gen_frame_bin.py

# local server (dataset dir may be empty; endpoints are mocked in-browser)
mkdir -p t4datasets_smoke
.venv/bin/t4-server --host 127.0.0.1 --port 8765 --data-dir ./t4datasets_smoke &

python3 tools/viewer_smoke/smoke_full.py
```

Covers: scene boot, frame scrubbing (LRU cache + reused point buffers),
postMessage trust guard, debug gating, metrics charts via the shared WebGL
renderer, and the point color-only update path.

## Traps

- **Templates are `lru_cache`d per server process.** Restart the local server
  after editing `templates/*.html`. Static JS is read fresh from disk.
- **A killed server may survive** and a replacement then dies with
  `Address already in use` while `curl /health` still answers from the old
  process. Always verify the serving PID's start time after a restart.
- Requires `playwright` (pip) and system Google Chrome (`channel="chrome"`);
  `--enable-unsafe-swiftshader` keeps WebGL alive in headless mode.
