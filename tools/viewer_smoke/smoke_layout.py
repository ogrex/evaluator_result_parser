"""Layout stability gate for the Three.js viewer's scrubber row.

The scrub track is `flex:1`, so any sibling whose text changes length steals
width from it and the slider visibly resizes while frames load. This asserts the
track's geometry is a function of the window only — never of the status text,
the frame counter, or the session state.

Prerequisites are the same as smoke_full.py:
  - playwright + system Chrome (channel="chrome")
  - a local server:  .venv/bin/t4-server --host 127.0.0.1 --port 8765 --data-dir ./t4datasets_smoke
  - the fixture:     .venv/bin/python tools/viewer_smoke/gen_frame_bin.py

RESTART the local server after editing templates/*.html; they are lru_cached.

Usage: python3 tools/viewer_smoke/smoke_layout.py
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"
HERE = Path(__file__).parent
frame_bin = (HERE / "frame.bin").read_bytes()
failures = []

# Real strings the status line cycles through, shortest to longest.
STATUS = [
    "loading...",
    "ready · cached=1/400 · 0/768 MB",
    "loading frame 12 · 3.4/8.1 MB (37%)",
    "loading frame 287 · 12.7/12.7 MB (100%)",
    "ready · cached=213/400 · 512/768 MB",
    "error: frame 15: HTTP 500 Internal Server Error on a very long url path",
    "share link copied · a1b2c3d4",
    "x",
]
# The other readouts in the row that change length.
READOUTS = [
    (0, 39, "session local-only"),
    (287, 399, "imported very_long_name.json"),
    (9, 9, "session a1b2c3d4"),
]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True,
                                args=["--enable-unsafe-swiftshader"])
    for width in (1600, 1280, 1100):
        page = browser.new_page(viewport={"width": width, "height": 880})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.route("**/viewer/three/meta*", lambda r: r.fulfill(
            content_type="application/json",
            body=json.dumps({"t4dataset_id": "smoke_ds", "scenario_name": "smoke_scene",
                             "version": None, "total_frames": 400, "format_version": "T4V3D002",
                             "binary_endpoint_template": "/viewer/three/frame.bin?frame_index={frame_index}"})))
        page.route("**/viewer/three/frame.bin*", lambda r: r.fulfill(
            content_type="application/octet-stream", body=frame_bin))
        page.goto(f"{BASE}/viewer/three?t4dataset_id=smoke_ds&scenario_name=smoke_scene",
                  wait_until="networkidle")
        page.wait_for_timeout(1500)

        def geom():
            return tuple(page.evaluate(
                "(()=>{const r=document.getElementById('slider').getBoundingClientRect();"
                "return [Math.round(r.width*100)/100, Math.round(r.left*100)/100]})()"))

        seen = set()
        for text in STATUS:
            page.evaluate("t=>{const e=document.getElementById('status');e.textContent=t;e.title=t;}", text)
            page.wait_for_timeout(50)
            seen.add(geom())
        for frame, total, session in READOUTS:
            page.evaluate(
                "a=>{document.getElementById('frameTxt').value=String(a[0]);"
                "document.getElementById('frameTotal').textContent='/ '+a[1];"
                "document.getElementById('shareSessionState').textContent=a[2];}",
                [frame, total, session])
            page.wait_for_timeout(50)
            seen.add(geom())
        # An error also reveals the dismiss button, adding a sibling to the row.
        page.evaluate("document.getElementById('statusDismiss').hidden=false")
        page.wait_for_timeout(50)
        seen.add(geom())

        if len(seen) != 1:
            failures.append(f"width={width}: scrub track moved across readout changes: {sorted(seen)}")
        else:
            print(f"width={width}: track stable at {seen.pop()}")
        if errors:
            failures.append(f"width={width}: page errors {errors[:3]}")
        page.close()
    browser.close()

if failures:
    print("SMOKE LAYOUT FAIL")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("SMOKE LAYOUT PASS")
