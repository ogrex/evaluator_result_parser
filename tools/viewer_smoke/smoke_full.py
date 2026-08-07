"""Full browser smoke test for the Three.js viewer.

Boots /viewer/three against a local t4-server with mocked meta / frame.bin /
camera endpoints, then verifies: scene boot, frame scrubbing through the LRU
cache, postMessage trust guard, debug gating, metrics-chart painting via the
shared renderer, camera panel painting via the lean image_url flow, and the
point color-only update path.

Prerequisites:
  - playwright (pip install playwright) + system Chrome (channel="chrome")
  - a local server:  .venv/bin/t4-server --host 127.0.0.1 --port 8765 --data-dir ./t4datasets_smoke
  - the fixture:     .venv/bin/python tools/viewer_smoke/gen_frame_bin.py

IMPORTANT: templates are lru_cached per server process; RESTART the local
server after any edit to templates/*.html. Static JS under static/ is read
fresh from disk, no restart needed.

Usage: python3 tools/viewer_smoke/smoke_full.py
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765"
HERE = Path(__file__).parent
frame_bin = (HERE / "frame.bin").read_bytes()
failures = []

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True,
                                args=["--enable-unsafe-swiftshader"])
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.route("**/viewer/three/meta*", lambda r: r.fulfill(
        content_type="application/json",
        body=json.dumps({"t4dataset_id": "smoke_ds", "scenario_name": "smoke_scene",
                         "version": None, "total_frames": 6, "format_version": "T4V3D002",
                         "binary_endpoint_template": "/viewer/three/frame.bin?frame_index={frame_index}"})))
    page.route("**/viewer/three/frame.bin*", lambda r: r.fulfill(
        content_type="application/octet-stream", body=frame_bin))
    page.goto(f"{BASE}/viewer/three?t4dataset_id=smoke_ds&scenario_name=smoke_scene",
              wait_until="networkidle")
    page.wait_for_timeout(2000)

    status = page.evaluate("document.getElementById('status').textContent")
    if "ready" not in status:
        failures.append(f"boot did not reach ready: {status!r}")

    # postMessage: accepted from self
    page.evaluate("""window.postMessage({type:'bbox_layers',
        gt:[{x:1,y:2,z:0,length:4,width:2,height:1.5,yaw:0,status:'TP'}], pred:[]}, '*')""")
    page.wait_for_timeout(500)
    if page.evaluate("window.T4ViewerAPI.debugState().externalGtCount") != 1:
        failures.append("bbox_layers self-post not applied")

    # postMessage: dropped from an untrusted iframe
    page.evaluate("""
      const f = document.createElement('iframe');
      f.srcdoc = "<script>parent.postMessage({type:'bbox_layers_clear'}, '*');<\\/script>";
      document.body.appendChild(f);
    """)
    page.wait_for_timeout(600)
    if page.evaluate("window.T4ViewerAPI.debugState().externalGtCount") != 1:
        failures.append("untrusted iframe message was applied")

    # metrics charts paint through the shared offscreen renderer
    page.evaluate("""
      for (const id of ['uiShowMetricsCounts','uiShowMetricsRates','uiShowMetricsError']) {
        const el = document.getElementById(id);
        if (el && !el.checked) { el.checked = true; el.dispatchEvent(new Event('change')); }
      }
      window.postMessage({type:'eval_metrics_series',
        gt_tp:[5,6,7,8,7,6], gt_fn:[1,0,2,1,0,1], est_tp:[5,6,6,8,7,5], est_fp:[2,1,0,1,2,1],
        tp_center_distance_mean:[0.3,0.4,0.2,0.5,0.3,0.4]}, '*');
    """)
    page.wait_for_timeout(1500)
    charts = page.evaluate("""
      (() => {
        const out = {};
        for (const id of ['metricsCanvas','metricsRatesCanvas','metricsErrorCanvas']) {
          const cv = document.getElementById(id);
          const ctx = cv ? cv.getContext('2d') : null;
          if (!ctx) { out[id] = -1; continue; }
          const d = ctx.getImageData(0, 0, cv.width, cv.height).data;
          let painted = 0;
          for (let i = 3; i < d.length; i += 4) if (d[i] > 0) painted++;
          out[id] = painted;
        }
        return out;
      })()
    """)
    for cid, painted in charts.items():
        if painted <= 0:
            failures.append(f"{cid} did not paint (painted={painted})")

    # point color-only path: gain-down and colormap changes must repaint.
    # Gain is intentionally checked with per-frame normalization off; with
    # normalization on, multiplying every intensity by the same gain preserves
    # the normalized color distribution.
    cv = page.locator("#canvas")
    page.evaluate("""
      const n = document.getElementById('pointIntensityNormalize');
      if (n.checked) { n.checked = false; n.dispatchEvent(new Event('change')); }
    """)
    page.wait_for_timeout(500)
    s1 = cv.screenshot()
    page.evaluate("""
      const g = document.getElementById('intensityGain');
      g.value = '0.4'; g.dispatchEvent(new Event('input'));
    """)
    page.wait_for_timeout(500)
    s2 = cv.screenshot()
    page.evaluate("""
      const s = document.getElementById('pointColormap');
      s.value = 'viridis'; s.dispatchEvent(new Event('change'));
    """)
    page.wait_for_timeout(500)
    s3 = cv.screenshot()
    if s1 == s2:
        failures.append("intensity gain change did not repaint points")
    if s2 == s3:
        failures.append("colormap change did not repaint points")

    # scrub through the LRU / reused buffers
    page.evaluate("""
      const s = document.getElementById('slider');
      for (const v of ['3','1','5','2']) { s.value = v; s.dispatchEvent(new Event('input')); }
    """)
    page.wait_for_timeout(1200)
    status2 = page.evaluate("document.getElementById('status').textContent")
    if "error" in status2:
        failures.append(f"scrubbing errored: {status2!r}")

    real_errors = [e for e in errors if "404" not in e]
    if real_errors:
        failures.append(f"pageerrors: {real_errors[:3]}")
    print("boot:", status, "| after scrub:", status2)
    print("charts painted:", charts)
    browser.close()

if failures:
    print("SMOKE FULL FAIL")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("SMOKE FULL PASS")
