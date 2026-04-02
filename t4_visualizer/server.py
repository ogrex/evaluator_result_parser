"""FastAPI server that exposes the render_frame API over HTTP.

Usage::

    # Default: scan ./t4datasets with search_depth=1
    t4-server

    # Custom data directory and port
    t4-server --data-dir /mnt/t4data --port 8080

    # Limit in-memory Tier4 cache to 4 datasets
    t4-server --data-dir /mnt/t4data --tier4-cache 4

Endpoints::

    POST /render
        Accepts a JSON body matching RenderRequest (includes ``target_objects``).
        Returns JSON matching RenderResponse.

    GET  /render
        Same render as POST using query parameters (no ``target_objects``).
        If ``format`` is omitted, **browsers** (``Accept: text/html``) receive an
        HTML page with embedded PNGs; typical API clients get JSON. You can force
        ``format=json`` or ``format=html``.

    GET  /render/view
    GET  /render/html
        Same query parameters as ``GET /render`` but always return ``text/html``
        with embedded PNGs (iframes, bookmarks). Prefer these URLs over
        ``GET /render`` when embedding in ``<iframe src="...">``.

    GET  /
        Landing page with server status, quick links, and usage examples.

    GET  /health
        Returns {"status": "ok"}.

    GET  /datasets
        Lists dataset IDs found under the configured data_dir.

    GET  /datasets/browser
        Interactive, read-only browser for datasets and scenarios.

    GET  /datasets/{t4dataset_id}/scenarios
        Lists scenes in that dataset (name, token, description, nbr_samples).
        Optional query parameter: ``version`` (same as ``POST /render``).

    GET  /datasets/{t4dataset_id}/availability
        Returns whether the dataset id exists under ``data_dir`` (same lookup as
        render). JSON: ``available``, optional ``dataset_path`` when present.

Example request body::

    {
        "t4dataset_id": "abc123",
        "scenario_name": "scene-0001",
        "frame_index": 5,
        "target_objects": [
            {"uuid": "", "x": 10.5, "y": 2.3, "z": 0.5, "label": "car"}
        ],
        "crop_cameras": true
    }

Example response body::

    {
        "sample_token": "deadbeef...",
        "timestamp_us": 1609459200000000,
        "images": [
            {"label": "CAM_FRONT", "png_base64": "iVBORw0KGgo..."}
        ],
        "elapsed_ms": 1234.5,
        "tier4_load_ms": 344.3,
        "render_ms": 890.2
    }

    The same timings are also sent as response headers (``X-Server-Elapsed-Ms``,
    ``X-Server-Tier4-Load-Ms``, ``X-Server-Render-Ms``) and ``Server-Timing`` for
    clients that prefer headers over the JSON body.

Example GET (JSON)::

    GET /render?t4dataset_id=...&scenario_name=...&frame_index=4

Example GET (HTML viewer)::

    GET /render?t4dataset_id=...&scenario_name=...&frame_index=4&format=html
    GET /render/view?t4dataset_id=...&scenario_name=...&frame_index=4
    GET /render/html?t4dataset_id=...&scenario_name=...&frame_index=4
"""

from __future__ import annotations

import argparse
import base64
import datetime
import html
import json
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tier4 in-memory cache
# ---------------------------------------------------------------------------

class _Tier4Cache:
    """Thread-safe LRU cache for Tier4 instances keyed by dataset path."""

    def __init__(self, max_size: int = 8):
        self._cache: Dict[Path, object] = {}   # Path → Tier4
        self._order: List[Path] = []           # LRU order (most-recent last)
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._loads = 0
        self._evictions = 0

    def get(self, path: Path):
        """Return cached Tier4 for *path*, or None if not present."""
        with self._lock:
            if path in self._cache:
                self._order.remove(path)
                self._order.append(path)
                self._hits += 1
                return self._cache[path]
            self._misses += 1
        return None

    def put(self, path: Path, t4) -> None:
        """Store *t4* for *path*, evicting the LRU entry if over capacity."""
        with self._lock:
            if path in self._cache:
                self._order.remove(path)
            elif len(self._cache) >= self._max_size:
                evict = self._order.pop(0)
                del self._cache[evict]
                self._evictions += 1
            self._cache[path] = t4
            self._order.append(path)

    def load(self, path: Path, version: Optional[str] = None):
        """Return a Tier4 instance for *path*, loading it if not cached."""
        t4 = self.get(path)
        if t4 is not None:
            return t4

        try:
            from t4_devkit import Tier4
        except ImportError as exc:
            raise ImportError(
                "t4_devkit is not installed. "
                "Install with: pip install git+https://github.com/tier4/t4-devkit.git"
            ) from exc

        from t4_visualizer.downloader import find_t4_root, prepare_dataset_root, patch_missing_t4_tables
        t4_root = find_t4_root(path)
        t4_root = prepare_dataset_root(t4_root)
        patch_missing_t4_tables(t4_root)
        kwargs = {"version": version} if version else {}
        t4 = Tier4(str(t4_root), **kwargs)
        with self._lock:
            self._loads += 1
        self.put(path, t4)
        return t4

    def stats(self) -> Dict[str, object]:
        """Return in-memory cache counters and keys for diagnostics pages."""
        with self._lock:
            return {
                "max_size": self._max_size,
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "loads": self._loads,
                "evictions": self._evictions,
                "keys": [str(p) for p in self._order],
            }


class _DatasetPathCache:
    """Thread-safe TTL cache for :func:`find_dataset_in_dir` results.

    Caches both hits (resolved ``Path``) and misses (``None``) so repeated
    availability checks and renders avoid rescanning large directory trees.
    Positive entries are dropped if the path disappears before TTL expires.
    Set *ttl_s* to ``0`` to disable caching.
    """

    def __init__(self, ttl_s: float = 30.0, max_entries: int = 4096):
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._lock = threading.Lock()
        # key -> (Optional[Path], deadline monotonic); None path = negative cache
        self._entries: Dict[str, Tuple[Optional[Path], float]] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

    @staticmethod
    def _key(data_dir: Path, search_depth: int, t4dataset_id: str) -> str:
        return f"{data_dir.resolve()}|{search_depth}|{t4dataset_id}"

    def resolve(
        self,
        data_dir: Path,
        search_depth: int,
        t4dataset_id: str,
        find_fn,
    ) -> Optional[Path]:
        """Return cached path, call *find_fn* () -> Optional[Path] on miss."""
        if self._ttl_s <= 0:
            return find_fn()

        key = self._key(data_dir, search_depth, t4dataset_id)
        now = time.monotonic()

        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                path, deadline = hit
                if now < deadline:
                    if path is None:
                        self._hits += 1
                        return None
                    try:
                        if path.exists():
                            self._hits += 1
                            return path
                    except OSError:
                        pass
                    del self._entries[key]
            self._misses += 1

        found = find_fn()
        deadline = now + self._ttl_s
        with self._lock:
            while len(self._entries) >= self._max_entries:
                try:
                    self._entries.pop(next(iter(self._entries)))
                    self._evictions += 1
                except StopIteration:
                    break
            self._entries[key] = (found, deadline)
            self._stores += 1
        return found

    def stats(self) -> Dict[str, object]:
        """Return TTL cache counters and occupancy for diagnostics endpoints."""
        now = time.monotonic()
        with self._lock:
            active = 0
            for _, deadline in self._entries.values():
                if now < deadline:
                    active += 1
            return {
                "ttl_s": self._ttl_s,
                "max_entries": self._max_entries,
                "entries": len(self._entries),
                "active_entries": active,
                "hits": self._hits,
                "misses": self._misses,
                "stores": self._stores,
                "evictions": self._evictions,
            }


# ---------------------------------------------------------------------------
# Pydantic models (request / response)
# Must be defined at module level — Pydantic v2 cannot resolve forward
# references for classes defined inside a function scope.
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field as _Field

    class TargetObjectIn(BaseModel):
        uuid: str = ""
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
        label: str = ""
        width: float = 0.0
        length: float = 0.0
        height: float = 0.0
        yaw: float = 0.0

    class RenderRequest(BaseModel):
        t4dataset_id: str
        scenario_name: str
        frame_index: int
        target_objects: List[TargetObjectIn] = _Field(default_factory=list)
        cameras: Optional[List[str]] = None
        show_annotations: bool = True
        version: Optional[str] = None
        crop_cameras: bool = False
        crop_padding: int = 40
        crop_min_size: int = 300

    class ImageOut(BaseModel):
        label: str
        png_base64: str

    class RenderResponse(BaseModel):
        sample_token: str
        timestamp_us: int
        images: List[ImageOut]
        elapsed_ms: float
        tier4_load_ms: float
        render_ms: float

    class ScenarioOut(BaseModel):
        name: str
        token: str
        description: str = ""
        nbr_samples: int = 0

    class ScenariosListResponse(BaseModel):
        t4dataset_id: str
        scenarios: List[ScenarioOut]
        version: Optional[str] = None
        total_scenarios: int = 0
        total_frames: int = 0
        min_frames: int = 0
        max_frames: int = 0

    class DatasetAvailabilityResponse(BaseModel):
        """Result of ``GET /datasets/{id}/availability``."""

        t4dataset_id: str
        available: bool
        dataset_path: Optional[str] = None

except ImportError:
    pass  # Proper error is raised inside _build_app when fastapi is missing


# ---------------------------------------------------------------------------

def _build_app(
    data_dir: Path,
    search_depth: int,
    tier4_cache_size: int,
    dataset_path_cache_ttl_s: float = 30.0,
    visibility_mode: str = "public",
):
    """Construct and return the FastAPI application."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query
        from fastapi.encoders import jsonable_encoder
        from fastapi.responses import HTMLResponse, JSONResponse, Response
    except ImportError as exc:
        raise ImportError(
            "fastapi and pydantic are required for the server. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    from t4_visualizer.batch import find_dataset_in_dir
    from t4_visualizer.downloader import list_webauto_annotation_dataset_ids
    from t4_visualizer.visualize import (
        TargetObject,
        VisualizationRequest,
        render_frame,
    )

    app = FastAPI(title="T4 Visualizer", version="0.1.0")
    _cache = _Tier4Cache(max_size=tier4_cache_size)
    _path_cache = _DatasetPathCache(ttl_s=dataset_path_cache_ttl_s)
    _debug_visibility = str(visibility_mode).strip().lower() == "debug"

    def _public_error(status_code: int, code: str, message: str, hint: Optional[str] = None):
        detail: Dict[str, object] = {"code": code, "message": message}
        if hint:
            detail["hint"] = hint
        raise HTTPException(status_code=status_code, detail=detail)

    def _safe_error(status_code: int, code: str, message: str, exc: Optional[Exception] = None):
        detail: Dict[str, object] = {"code": code, "message": message}
        if _debug_visibility and exc is not None:
            detail["debug"] = str(exc)
        raise HTTPException(status_code=status_code, detail=detail)

    @app.middleware("http")
    async def _log_http_requests(request, call_next):
        method = request.method.upper()
        if method not in ("GET", "POST"):
            return await call_next(request)
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        query = request.url.query
        query_txt = f"?{query}" if query else ""
        print(
            f"[http] {method} {request.url.path}{query_txt} "
            f"-> {response.status_code} ({elapsed_ms:.1f} ms)"
        )
        return response

    # Syncs with system / browser theme via prefers-color-scheme (no JS).
    _RENDER_VIEW_CSS = """
    :root {
      color-scheme: light dark;
      --page-bg: #f4f4f6;
      --text: #18181c;
      --meta-bg: #ffffff;
      --meta-border: #e4e4e8;
      --dt: #52525c;
      --figcaption: #63636b;
      --timings: #71717b;
      --img-border: #d4d4d8;
      --img-shadow: rgba(15, 23, 42, 0.12);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --page-bg: #141418;
        --text: #e8e8ed;
        --meta-bg: #1e1e24;
        --meta-border: #2c2c34;
        --dt: #8e8e9a;
        --figcaption: #a0a0ac;
        --timings: #6e6e78;
        --img-border: #2c2c34;
        --img-shadow: rgba(0, 0, 0, 0.35);
      }
    }
    body {
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      margin: 0;
      background: var(--page-bg);
      color: var(--text);
      line-height: 1.45;
    }
    h1 { font-size: 1.1rem; font-weight: 600; margin: 0 0 0.75rem 0; }
    .meta {
      padding: 0.75rem 1.25rem;
      background: var(--meta-bg);
      border-bottom: 1px solid var(--meta-border);
    }
    .meta summary {
      cursor: pointer;
      font-size: 1.1rem;
      font-weight: 600;
      list-style: none;
      user-select: none;
    }
    .meta summary::-webkit-details-marker { display: none; }
    .meta summary::before {
      content: "▸";
      display: inline-block;
      margin-right: 0.45rem;
      transition: transform 120ms ease;
    }
    .meta[open] summary::before { transform: rotate(90deg); }
    .meta .meta-content { margin-top: 0.75rem; }
    .meta dl { display: grid; grid-template-columns: 9rem 1fr; gap: 0.35rem 1rem;
               margin: 0; font-size: 0.8125rem; }
    .meta dt { color: var(--dt); margin: 0; }
    .meta dd { margin: 0; word-break: break-all; }
    main { padding: 1rem 1.25rem 2rem; max-width: min(100%, 140rem); margin: 0 auto; }
    figure { margin: 1.25rem 0; }
    figure img {
      display: block; max-width: 100%; height: auto;
      border: 1px solid var(--img-border); border-radius: 6px;
      box-shadow: 0 4px 24px var(--img-shadow);
    }
    figcaption { margin-top: 0.5rem; font-size: 0.8rem; color: var(--figcaption); }
    .timings { margin-top: 0.75rem; font-size: 0.75rem; color: var(--timings); }
    """

    def _render_get_query(
        t4dataset_id: str = Query(..., description="Dataset id (see GET /datasets)"),
        scenario_name: str = Query(
            ...,
            description="Scene name from GET /datasets/{id}/scenarios",
        ),
        frame_index: int = Query(..., ge=0, description="Frame index within the scene"),
        cameras: Optional[str] = Query(
            None,
            description="Comma-separated camera channels (omit for all)",
        ),
        show_annotations: bool = Query(True),
        version: Optional[str] = Query(None),
        crop_cameras: bool = Query(False),
        crop_padding: int = Query(40, ge=0),
        crop_min_size: int = Query(300, ge=1),
    ):
        """Query parameters shared by ``GET /render`` and ``GET /render/view``."""
        from types import SimpleNamespace

        return SimpleNamespace(
            t4dataset_id=t4dataset_id,
            scenario_name=scenario_name,
            frame_index=frame_index,
            cameras=cameras,
            show_annotations=show_annotations,
            version=version,
            crop_cameras=crop_cameras,
            crop_padding=crop_padding,
            crop_min_size=crop_min_size,
        )

    def _parse_cameras_csv(cameras: Optional[str]) -> Optional[List[str]]:
        if not cameras or not str(cameras).strip():
            return None
        parts = [c.strip() for c in str(cameras).split(",") if c.strip()]
        return parts or None

    def _effective_render_format(
        accept_header: Optional[str],
        explicit: Optional[str],
        sec_fetch_dest: Optional[str],
    ) -> str:
        """Return ``json`` or ``html``. When *explicit* is omitted, use Sec-Fetch-Dest and Accept.

        Browsers loading a document or iframe often send ``Accept: */*`` without
        ``text/html``, so we treat ``Sec-Fetch-Dest: document`` / ``iframe`` as HTML.
        """
        if explicit is not None and str(explicit).strip() != "":
            fmt = str(explicit).strip().lower()
            if fmt not in ("json", "html"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid format: use 'json', 'html', or omit for auto (browser→html).",
                )
            return fmt
        dest = (sec_fetch_dest or "").strip().lower()
        if dest in ("document", "iframe"):
            return "html"
        accept = (accept_header or "").lower()
        if "text/html" in accept:
            return "html"
        return "json"

    def _html_response(body: str, hdrs: Dict[str, str]) -> HTMLResponse:
        """Return HTML with an explicit charset so browsers render the page, not raw text."""
        return HTMLResponse(
            content=body,
            media_type="text/html; charset=utf-8",
            headers=hdrs,
        )

    def _resolve_dataset(t4dataset_id: str) -> Path:
        path = _path_cache.resolve(
            data_dir,
            search_depth,
            t4dataset_id,
            lambda: find_dataset_in_dir(data_dir, t4dataset_id, search_depth),
        )
        if path is None:
            _public_error(
                status_code=404,
                code="dataset_not_found",
                message=f"Dataset '{t4dataset_id}' was not found.",
                hint="Call GET /datasets to inspect visible dataset IDs.",
            )
        return path

    def _run_render(
        *,
        t4dataset_id: str,
        dataset_path: Path,
        scenario_name: str,
        frame_index: int,
        target_objects: List[TargetObject],
        cameras: Optional[List[str]],
        show_annotations: bool,
        version: Optional[str],
        crop_cameras: bool,
        crop_padding: int,
        crop_min_size: int,
    ):
        """Execute render_frame and build :class:`RenderResponse` plus timing headers."""
        request = VisualizationRequest(
            dataset_path=dataset_path,
            scenario_name=scenario_name,
            frame_index=frame_index,
            target_objects=target_objects,
            cameras=cameras,
            show_annotations=show_annotations,
            version=version,
            crop_cameras=crop_cameras,
            crop_padding=crop_padding,
            crop_min_size=crop_min_size,
        )

        t0 = time.perf_counter()
        try:
            t4 = _cache.load(dataset_path, version=version)
            t1 = time.perf_counter()
            result = render_frame(request, t4=t4)
            t2 = time.perf_counter()
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="render_failed",
                message="Failed to render the requested frame.",
                exc=exc,
            )

        elapsed_ms = round((t2 - t0) * 1000.0, 3)
        tier4_load_ms = round((t1 - t0) * 1000.0, 3)
        render_ms = round((t2 - t1) * 1000.0, 3)

        payload = RenderResponse(
            sample_token=result.sample_token,
            timestamp_us=result.timestamp_us,
            images=[
                ImageOut(label=img.label, png_base64=base64.b64encode(img.data).decode())
                for img in result.images
            ],
            elapsed_ms=elapsed_ms,
            tier4_load_ms=tier4_load_ms,
            render_ms=render_ms,
        )
        hdrs = {
            "X-Server-Elapsed-Ms": str(elapsed_ms),
            "X-Server-Tier4-Load-Ms": str(tier4_load_ms),
            "X-Server-Render-Ms": str(render_ms),
            "Server-Timing": (
                f"tier4-load;dur={tier4_load_ms}, render;dur={render_ms}, total;dur={elapsed_ms}"
            ),
        }
        return payload, hdrs

    def _get_scenario_sample(t4, scenario_name: str, frame_index: int):
        from t4_visualizer.visualize import find_sample_by_scene_and_index

        try:
            return find_sample_by_scene_and_index(t4, scenario_name, frame_index)
        except ValueError as exc:
            _public_error(
                status_code=404,
                code="scenario_not_found",
                message=f"Scenario '{scenario_name}' was not found.",
                hint="Call GET /datasets/{id}/scenarios to inspect valid scenario names.",
            )
        except IndexError as exc:
            _public_error(
                status_code=400,
                code="frame_index_out_of_range",
                message=str(exc),
            )
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="scenario_sample_lookup_failed",
                message="Failed to resolve sample for scenario/frame.",
                exc=exc,
            )

    def _pointcloud_and_boxes_for_sample(t4, sample):
        import numpy as np
        from t4_visualizer.visualize import _load_pointcloud, list_lidar_channels

        lidar_channels = list_lidar_channels(t4, sample)
        if not lidar_channels:
            return np.zeros((0, 4), dtype=np.float32), []
        lidar_channel = lidar_channels[0]
        token = sample.data.get(lidar_channel)
        if token is None:
            return np.zeros((0, 4), dtype=np.float32), []
        data_path, boxes_3d, _ = t4.get_sample_data(
            token, as_3d=True, as_sensor_coord=True
        )
        # Schema-first decode:
        # - .pcd.bin is defined by t4-devkit as (x,y,z,intensity,ring_idx) float32[5]
        # - .bin in some legacy datasets can be float32[4] (x,y,z,intensity)
        # We preserve a cautious fallback chain for compatibility.
        points = None
        path_txt = str(data_path).lower()
        if path_txt.endswith(".pcd.bin") or path_txt.endswith(".bin"):
            try:
                raw = np.fromfile(data_path, dtype=np.float32)
                stride_order = (5, 4, 6, 3) if path_txt.endswith(".pcd.bin") else (4, 5, 6, 3)
                for ncols in stride_order:
                    if raw.size % ncols != 0:
                        continue
                    pts = raw.reshape(-1, ncols)
                    xyz = pts[:, :3]
                    # Basic sanity: avoid wildly implausible decode.
                    p99 = np.percentile(np.abs(xyz), 99, axis=0)
                    if np.any(p99 > 2000):
                        continue
                    if ncols >= 4:
                        points = pts[:, :4]
                    else:
                        points = np.column_stack(
                            [pts[:, :3], np.ones(len(pts), dtype=np.float32)]
                        )
                    break
            except Exception:
                points = None
        if points is None:
            points = _load_pointcloud(data_path)
        if points is None or points.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32), boxes_3d or []
        points = np.asarray(points)
        if points.shape[1] >= 4:
            out = points[:, :4].astype(np.float32, copy=False)
        else:
            pad = np.ones((points.shape[0], 1), dtype=np.float32)
            out = np.concatenate([points[:, :3].astype(np.float32, copy=False), pad], axis=1)
        # Drop non-finite points early to prevent viewport artifacts.
        finite_mask = np.isfinite(out).all(axis=1)
        if finite_mask is not None and finite_mask.size == out.shape[0]:
            out = out[finite_mask]
        return out, boxes_3d or []

    def _camera_overlay_payload_for_sample(
        t4,
        sample,
        camera: Optional[str] = None,
        show_annotations: bool = True,
    ) -> Dict[str, object]:
        from t4_visualizer.visualize import list_camera_channels

        channels = list_camera_channels(t4, sample)
        if not channels:
            return {
                "camera": None,
                "available_cameras": [],
                "image_base64": "",
                "image_format": "jpeg",
                "boxes_2d": [],
                "width": 0,
                "height": 0,
            }
        channel = camera if camera in channels else channels[0]
        token = sample.data.get(channel)
        if token is None:
            _public_error(404, "camera_token_not_found", f"Camera token not found for channel '{channel}'.")
        if show_annotations:
            data_path, boxes_2d, _ = t4.get_sample_data(token, as_3d=False, as_sensor_coord=True)
        else:
            data_path, _, _ = t4.get_sample_data(token, as_3d=False)
            boxes_2d = []
        sample_data = t4.get("sample_data", token)
        img_path = Path(str(data_path))
        try:
            img_bytes = img_path.read_bytes()
        except Exception as exc:
            _safe_error(500, "camera_image_read_failed", "Failed to read camera image.", exc=exc)
        fmt = img_path.suffix.lower().lstrip(".") or "jpeg"
        width = int(getattr(sample_data, "width", 0) or 0)
        height = int(getattr(sample_data, "height", 0) or 0)
        try:
            # Fallback to image probing only if table metadata is unavailable.
            if width <= 0 or height <= 0:
                from PIL import Image

                with Image.open(img_path) as im:
                    width, height = int(im.width), int(im.height)
        except Exception:
            pass
        box_rows = []
        for b in boxes_2d or []:
            try:
                roi = getattr(b, "roi", None)
                if roi is None or len(roi) < 4:
                    continue
                x0, y0, x1, y1 = float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])
                box_rows.append(
                    {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "label": str(getattr(b, "label", "") or ""),
                    }
                )
            except Exception:
                continue
        return {
            "camera": channel,
            "available_cameras": channels,
            "image_base64": base64.b64encode(img_bytes).decode("ascii"),
            "image_format": fmt,
            "boxes_2d": box_rows,
            "width": width,
            "height": height,
        }

    def _pack_viewer_frame_binary(
        *,
        frame_index: int,
        sample_token: str,
        timestamp_us: int,
        points_xyz_i,
        boxes_3d,
    ):
        """
        Binary wire format (little-endian):
          magic[8]        : b'T4V3D002'
          header_len      : uint32
          frame_index     : uint32
          timestamp_us    : uint64
          point_count     : uint32
          box_count       : uint32
          sample_token_len: uint16
          sample_token    : UTF-8 bytes
          points          : point_count * float32[4]  # x,y,z,intensity
          box_corners_f32 : box_count * float32[24]   # 8 corners * xyz
          box_labels_json : UTF-8 JSON list[str], length-prefixed uint32
        """
        import numpy as np

        token_bytes = str(sample_token).encode("utf-8")
        pts = np.asarray(points_xyz_i, dtype=np.float32)
        if pts.size == 0:
            pts = np.zeros((0, 4), dtype=np.float32)
        point_count = int(pts.shape[0])

        box_rows = []
        labels = []
        for box in boxes_3d:
            try:
                corners = box.corners()
                flat = [float(v) for v in corners.reshape(-1).tolist()]
            except Exception:
                # Fallback: axis-aligned cuboid from center/size.
                center = getattr(box, "center", [0.0, 0.0, 0.0])
                size = getattr(box, "size", [0.0, 0.0, 0.0])
                cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
                sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
                hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
                pts = [
                    [cx + hx, cy + hy, cz + hz], [cx + hx, cy - hy, cz + hz],
                    [cx + hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz - hz],
                    [cx - hx, cy + hy, cz + hz], [cx - hx, cy - hy, cz + hz],
                    [cx - hx, cy - hy, cz - hz], [cx - hx, cy + hy, cz - hz],
                ]
                flat = [float(v) for p in pts for v in p]
            box_rows.append(flat)
            labels.append(str(getattr(box, "label", "") or ""))
        box_arr = np.asarray(box_rows, dtype=np.float32) if box_rows else np.zeros((0, 24), dtype=np.float32)
        label_blob = json.dumps(labels, ensure_ascii=True).encode("utf-8")
        box_count = int(box_arr.shape[0])

        header = struct.pack(
            "<8sIIQIIH",
            b"T4V3D002",
            34,  # fixed header bytes
            int(frame_index),
            int(timestamp_us),
            point_count,
            box_count,
            len(token_bytes),
        ) + token_bytes
        return b"".join(
            [
                header,
                pts.tobytes(order="C"),
                box_arr.tobytes(order="C"),
                struct.pack("<I", len(label_blob)),
                label_blob,
            ]
        )

    def _render_html_page(payload: RenderResponse, q) -> str:
        """Build a self-contained HTML document with embedded PNG data URLs."""
        esc = html.escape
        rows = [
            ("Dataset", esc(q.t4dataset_id)),
            ("Scenario", esc(q.scenario_name)),
            ("Frame index", str(q.frame_index)),
            ("Sample token", esc(payload.sample_token)),
            ("Timestamp (µs)", str(payload.timestamp_us)),
        ]
        if q.cameras:
            rows.append(("Cameras filter", esc(q.cameras)))
        rows.append(("Images", str(len(payload.images))))

        dl_parts = []
        for dt, dd in rows:
            dl_parts.append(f"<dt>{esc(dt)}</dt><dd>{dd}</dd>")

        fig_parts = []
        for im in payload.images:
            fig_parts.append(
                "<figure>"
                f'<img src="data:image/png;base64,{im.png_base64}" '
                f'alt="{esc(im.label)}">'
                f"<figcaption>{esc(im.label)}</figcaption>"
                "</figure>"
            )

        title = f"{q.scenario_name[:48]}…" if len(q.scenario_name) > 48 else q.scenario_name
        return (
            "<!DOCTYPE html>"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)} — frame {q.frame_index}</title>"
            f"<style>{_RENDER_VIEW_CSS}</style>"
            "</head><body>"
            '<details class="meta"><summary>T4 frame render</summary><div class="meta-content"><dl>'
            + "".join(dl_parts)
            + "</dl>"
            '<p class="timings">'
            f"elapsed_ms={payload.elapsed_ms} · tier4_load_ms={payload.tier4_load_ms} · "
            f"render_ms={payload.render_ms}"
            "</p></div></details>"
            "<main>"
            + "".join(fig_parts)
            + "</main></body></html>"
        )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/datasets/browser")
    def datasets_browser_page():
        """Interactive read-only browser for datasets and scenarios."""
        page = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>T4 Dataset Browser</title>
  <style>
    :root{
      color-scheme: light dark;
      --bg:#f5f7fb; --card:#ffffff; --muted:#576075; --text:#101322;
      --border:#d8e0f0; --accent:#245bda; --ok:#0e7a42; --warn:#9f3f11;
      --shadow:rgba(8,18,50,0.12);
    }
    @media (prefers-color-scheme: dark){
      :root{
        --bg:#0f1420; --card:#171e2d; --muted:#9aa8c2; --text:#e6ebf8;
        --border:#28334a; --accent:#79a6ff; --ok:#4bd08d; --warn:#ff9f73;
        --shadow:rgba(0,0,0,0.35);
      }
    }
    *{box-sizing:border-box}
    body{margin:0;background:
      radial-gradient(1200px 520px at 20% -10%, color-mix(in srgb, var(--accent) 15%, transparent), transparent 70%),
      radial-gradient(900px 500px at 90% 0%, color-mix(in srgb, var(--ok) 10%, transparent), transparent 68%),
      var(--bg);
      color:var(--text);font-family:system-ui,-apple-system,Segoe UI,sans-serif}
    .wrap{max-width:1400px;margin:0 auto;padding:1.1rem}
    .hero{display:flex;align-items:end;justify-content:space-between;gap:1rem;margin-bottom:1rem}
    h1{margin:0;font-size:1.45rem}
    .muted{color:var(--muted)}
    .actions a{color:var(--accent);text-decoration:none;margin-left:.9rem}
    .grid{display:grid;grid-template-columns:minmax(260px,320px) minmax(340px,1fr) minmax(340px,1fr);gap:.9rem}
    .grid-diag{display:grid;grid-template-columns:1fr 1fr;gap:.9rem;margin-top:.9rem}
    .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.85rem;box-shadow:0 8px 26px var(--shadow)}
    .title{font-weight:700;font-size:1rem;margin:0 0 .7rem}
    input,select{width:100%;padding:.58rem .65rem;border:1px solid var(--border);border-radius:9px;background:transparent;color:var(--text)}
    .pill{display:inline-block;padding:.15rem .55rem;border:1px solid var(--border);border-radius:999px;font-size:.78rem;color:var(--muted)}
    .list{margin-top:.65rem;max-height:65vh;overflow:auto;border:1px solid var(--border);border-radius:9px}
    .dataset-item{padding:.55rem .65rem;border-bottom:1px solid var(--border);cursor:pointer}
    .dataset-item:last-child{border-bottom:none}
    .dataset-item:hover,.dataset-item.active{background:color-mix(in srgb, var(--accent) 12%, transparent)}
    .k{color:var(--muted);font-size:.84rem}
    .v{word-break:normal;overflow-wrap:anywhere;line-break:auto}
    .row{display:grid;grid-template-columns:minmax(120px,160px) minmax(0,1fr);gap:.5rem;margin:.42rem 0;align-items:start}
    .status.ok{color:var(--ok)} .status.warn{color:var(--warn)}
    table{width:100%;border-collapse:collapse;font-size:.9rem}
    th,td{padding:.5rem;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}
    th{font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
    .toolbar{display:flex;gap:.5rem;margin:.55rem 0}
    .toolbar input{max-width:340px}
    .empty,.error,.loading{padding:.7rem;border:1px dashed var(--border);border-radius:8px;color:var(--muted)}
    .error{color:var(--warn)}
    .retry{margin-top:.6rem;padding:.45rem .65rem;border:1px solid var(--border);border-radius:8px;background:transparent;color:var(--text);cursor:pointer}
    .diag pre{max-height:28vh;overflow:auto;border:1px solid var(--border);padding:.5rem;border-radius:8px;background:color-mix(in srgb, var(--card) 75%, transparent)}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    .tiny{font-size:.8rem;color:var(--muted)}
    @media (max-width:1200px){.grid{grid-template-columns:1fr}}
    @media (max-width:1200px){.grid-diag{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div>
        <h1>T4 Dataset Browser</h1>
        <div class="muted">Read-only explorer for datasets and scenarios on this server.</div>
      </div>
      <div class="actions">
        <a href="/">Home</a>
        <a href="/datasets">Raw /datasets</a>
        <a href="/docs">API docs</a>
      </div>
    </div>
    <div class="grid">
      <section class="card">
        <div class="title">Datasets <span id="datasetCount" class="pill">0</span></div>
        <input id="datasetSearch" type="search" placeholder="Filter by dataset id">
        <div id="datasetList" class="list" aria-label="Dataset list"></div>
      </section>
      <section class="card">
        <div class="title">Dataset Details</div>
        <div id="detailsState" class="empty">Select a dataset from the left.</div>
        <div id="detailsPanel" style="display:none">
          <div class="row"><div class="k">dataset_id</div><div id="dId" class="v"></div></div>
          <div class="row"><div class="k">availability</div><div id="dAvail" class="v"></div></div>
          <div class="row"><div class="k">dataset_path</div><div id="dPath" class="v"></div></div>
          <div class="row"><div class="k">quick links</div><div id="dLinks" class="v"></div></div>
        </div>
      </section>
      <section class="card">
        <div class="title">Scenarios</div>
        <div class="toolbar">
          <input id="scenarioSearch" type="search" placeholder="Filter by name/description/token" disabled>
          <select id="scenarioSort" disabled>
            <option value="name">Sort: name</option>
            <option value="nbr_samples">Sort: frame count</option>
          </select>
        </div>
        <div id="scenarioState" class="empty">Select a dataset to load scenarios.</div>
        <div id="scenarioPanel" style="display:none;max-height:65vh;overflow:auto">
          <div id="scenarioMeta" class="tiny" style="margin-bottom:.5rem"></div>
          <table>
            <thead><tr><th>name</th><th>token</th><th>description</th><th>frames</th><th>actions</th></tr></thead>
            <tbody id="scenarioRows"></tbody>
          </table>
        </div>
      </section>
    </div>
    <div class="grid-diag">
      <section class="card diag">
        <div class="title">Server Diagnostics</div>
        <div id="diagState" class="loading">Loading diagnostics...</div>
        <pre id="diagJson" style="display:none"></pre>
      </section>
      <section class="card">
        <div class="title">Frame Summary</div>
        <div class="toolbar">
          <input id="frameOffset" type="number" min="0" value="0" placeholder="offset" disabled>
          <input id="frameLimit" type="number" min="1" max="500" value="50" placeholder="limit" disabled>
          <button class="retry" id="loadFramesBtn" disabled>Load</button>
        </div>
        <div id="frameState" class="empty">Select a scenario action to load frame summaries.</div>
        <div id="framePanel" style="display:none;max-height:42vh;overflow:auto">
          <table>
            <thead><tr><th>idx</th><th>timestamp_us</th><th>sample_token</th><th>render</th></tr></thead>
            <tbody id="frameRows"></tbody>
          </table>
        </div>
      </section>
    </div>
  </div>
  <script>
    const el = (id) => document.getElementById(id);
    const state = {
      dataDir: "",
      datasets: [],
      selectedId: null,
      scenarios: [],
      selectedScenario: null,
      scenarioMeta: { total_scenarios: 0, total_frames: 0, min_frames: 0, max_frames: 0 }
    };

    function setDatasetCount(n){ el("datasetCount").textContent = String(n); }
    function showDatasetList(items){
      const root = el("datasetList");
      if (!items.length){
        root.innerHTML = '<div class="empty">No datasets found.</div>';
        return;
      }
      root.innerHTML = items.map((id) => {
        const active = id === state.selectedId ? " active" : "";
        return `<div class="dataset-item${active}" data-id="${id}"><code>${id}</code></div>`;
      }).join("");
      root.querySelectorAll(".dataset-item").forEach((node) => {
        node.addEventListener("click", () => selectDataset(node.dataset.id));
      });
    }

    function filterDatasets(){
      const q = el("datasetSearch").value.trim().toLowerCase();
      const filtered = !q ? state.datasets : state.datasets.filter((d) => d.toLowerCase().includes(q));
      showDatasetList(filtered);
    }

    async function getJson(url){
      const res = await fetch(url, { headers: { "Accept": "application/json" } });
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch (_) {}
      if (!res.ok) {
        const detail = data && (data.detail || data.hint) ? (data.detail || data.hint) : `HTTP ${res.status}`;
        throw new Error(detail);
      }
      return data;
    }

    function renderDetails({ id, avail, path }){
      el("detailsState").style.display = "none";
      el("detailsPanel").style.display = "block";
      el("dId").innerHTML = `<code>${id}</code>`;
      el("dAvail").innerHTML = avail ? '<span class="status ok">available</span>' : '<span class="status warn">not found</span>';
      el("dPath").innerHTML = path
        ? `<code style="display:block;max-width:100%;overflow:auto;white-space:nowrap">${path}</code>`
        : '<span class="muted">(none)</span>';
      const qs = `?t4dataset_id=${encodeURIComponent(id)}`;
      el("dLinks").innerHTML =
        `<a href="/datasets/${encodeURIComponent(id)}/availability" target="_blank" rel="noopener">availability</a> · ` +
        `<a href="/datasets/${encodeURIComponent(id)}/scenarios" target="_blank" rel="noopener">scenarios</a> · ` +
        `<a href="/render/html${qs}&scenario_name=...&frame_index=0" target="_blank" rel="noopener">render/html template</a>`;
    }

    function showDetailsError(err, id){
      el("detailsPanel").style.display = "none";
      el("detailsState").style.display = "block";
      el("detailsState").className = "error";
      el("detailsState").innerHTML =
        `Failed to load details for <code>${id}</code>: ${err.message}` +
        `<div><button class="retry" id="retryDetails">Retry</button></div>`;
      const btn = el("retryDetails");
      if (btn) btn.onclick = () => selectDataset(id);
    }

    function renderScenarios(list, meta){
      const q = el("scenarioSearch").value.trim().toLowerCase();
      const sortBy = el("scenarioSort").value;
      let rows = list.slice();
      if (q){
        rows = rows.filter((s) =>
          (s.name || "").toLowerCase().includes(q) ||
          (s.description || "").toLowerCase().includes(q) ||
          (s.token || "").toLowerCase().includes(q)
        );
      }
      rows.sort((a,b) => sortBy === "nbr_samples"
        ? (Number(b.nbr_samples || 0) - Number(a.nbr_samples || 0))
        : String(a.name || "").localeCompare(String(b.name || ""))
      );
      const tbody = el("scenarioRows");
      if (!rows.length){
        tbody.innerHTML = `<tr><td colspan="5" class="muted">No scenarios match current filter.</td></tr>`;
      } else {
        tbody.innerHTML = rows.map((s) =>
          `<tr>
            <td><code>${s.name || ""}</code></td>
            <td class="mono">${s.token || ""}</td>
            <td>${s.description || ""}</td>
            <td>${s.nbr_samples ?? 0}</td>
            <td>
              <a href="#" data-scenario="${encodeURIComponent(s.name || "")}" class="load-frames">frames</a> ·
              <a href="/render/html?t4dataset_id=${encodeURIComponent(state.selectedId)}&scenario_name=${encodeURIComponent(s.name || "")}&frame_index=0" target="_blank" rel="noopener">render</a> ·
              <a href="/viewer/three?t4dataset_id=${encodeURIComponent(state.selectedId)}&scenario_name=${encodeURIComponent(s.name || "")}&frame_index=0" target="_blank" rel="noopener">3d</a>
            </td>
          </tr>`
        ).join("");
      }
      const totalScenarios = meta && Number(meta.total_scenarios || 0);
      const totalFrames = meta && Number(meta.total_frames || 0);
      const minFrames = meta && Number(meta.min_frames || 0);
      const maxFrames = meta && Number(meta.max_frames || 0);
      el("scenarioMeta").textContent =
        `total_scenarios=${totalScenarios} total_frames=${totalFrames} min_frames=${minFrames} max_frames=${maxFrames}`;
      el("scenarioState").style.display = "none";
      el("scenarioPanel").style.display = "block";
      tbody.querySelectorAll(".load-frames").forEach((n) => {
        n.addEventListener("click", async (ev) => {
          ev.preventDefault();
          const scenario = decodeURIComponent(n.dataset.scenario || "");
          state.selectedScenario = scenario;
          await loadFrameSummary();
        });
      });
    }

    async function loadFrameSummary(){
      if (!state.selectedId || !state.selectedScenario){
        return;
      }
      const offset = Math.max(0, Number(el("frameOffset").value || 0));
      const limit = Math.max(1, Math.min(500, Number(el("frameLimit").value || 50)));
      el("frameState").className = "loading";
      el("frameState").style.display = "block";
      el("framePanel").style.display = "none";
      el("frameState").textContent =
        `Loading frames for ${state.selectedScenario} (offset=${offset}, limit=${limit})...`;
      try {
        const data = await getJson(
          `/datasets/${encodeURIComponent(state.selectedId)}/scenarios/${encodeURIComponent(state.selectedScenario)}/frames/summary?offset=${offset}&limit=${limit}`
        );
        const rows = Array.isArray(data.frames) ? data.frames : [];
        const tbody = el("frameRows");
        if (!rows.length){
          tbody.innerHTML = `<tr><td colspan="4" class="muted">No frame rows in selected window.</td></tr>`;
        } else {
          tbody.innerHTML = rows.map((f) =>
            `<tr>
              <td>${f.frame_index ?? ""}</td>
              <td>${f.timestamp_us ?? ""}</td>
              <td class="mono">${f.sample_token ?? ""}</td>
              <td>
                <a href="/render/html?t4dataset_id=${encodeURIComponent(state.selectedId)}&scenario_name=${encodeURIComponent(state.selectedScenario)}&frame_index=${encodeURIComponent(String(f.frame_index ?? 0))}" target="_blank" rel="noopener">render</a> ·
                <a href="/viewer/three?t4dataset_id=${encodeURIComponent(state.selectedId)}&scenario_name=${encodeURIComponent(state.selectedScenario)}&frame_index=${encodeURIComponent(String(f.frame_index ?? 0))}" target="_blank" rel="noopener">3d</a>
              </td>
            </tr>`
          ).join("");
        }
        el("frameState").style.display = "none";
        el("framePanel").style.display = "block";
      } catch (err) {
        el("framePanel").style.display = "none";
        el("frameState").className = "error";
        el("frameState").style.display = "block";
        el("frameState").textContent = `Failed to load frame summary: ${err.message}`;
      }
    }

    async function loadDiagnostics(){
      try {
        const data = await getJson("/browser/diagnostics");
        el("diagState").style.display = "none";
        el("diagJson").style.display = "block";
        el("diagJson").textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        el("diagState").className = "error";
        el("diagState").textContent = `Failed to load diagnostics: ${err.message}`;
      }
    }

    async function selectDataset(id){
      state.selectedId = id;
      filterDatasets();

      el("detailsState").className = "loading";
      el("detailsState").style.display = "block";
      el("detailsPanel").style.display = "none";
      el("detailsState").textContent = "Loading dataset details...";

      el("scenarioState").className = "loading";
      el("scenarioState").style.display = "block";
      el("scenarioPanel").style.display = "none";
      el("scenarioState").textContent = "Loading scenarios...";
      el("scenarioSearch").disabled = true;
      el("scenarioSort").disabled = true;
      el("frameOffset").disabled = true;
      el("frameLimit").disabled = true;
      el("loadFramesBtn").disabled = true;
      state.selectedScenario = null;

      try {
        const avail = await getJson(`/datasets/${encodeURIComponent(id)}/availability`);
        renderDetails({ id, avail: !!avail.available, path: avail.dataset_path || "" });
      } catch (err) {
        showDetailsError(err, id);
      }

      try {
        const data = await getJson(`/datasets/${encodeURIComponent(id)}/scenarios`);
        state.scenarios = Array.isArray(data.scenarios) ? data.scenarios : [];
        state.scenarioMeta = data || { total_scenarios: 0, total_frames: 0, min_frames: 0, max_frames: 0 };
        el("scenarioSearch").disabled = false;
        el("scenarioSort").disabled = false;
        el("frameOffset").disabled = false;
        el("frameLimit").disabled = false;
        el("loadFramesBtn").disabled = false;
        renderScenarios(state.scenarios, state.scenarioMeta);
      } catch (err) {
        el("scenarioPanel").style.display = "none";
        el("scenarioState").className = "error";
        el("scenarioState").style.display = "block";
        el("scenarioState").innerHTML =
          `Failed to load scenarios for <code>${id}</code>: ${err.message}` +
          `<div><button class="retry" id="retryScenarios">Retry</button></div>`;
        const btn = el("retryScenarios");
        if (btn) btn.onclick = () => selectDataset(id);
      }
    }

    async function init(){
      const listRoot = el("datasetList");
      listRoot.innerHTML = '<div class="loading">Loading datasets...</div>';
      try {
        const data = await getJson("/datasets");
        state.dataDir = data.data_dir || "";
        state.datasets = Array.isArray(data.datasets) ? data.datasets : [];
        setDatasetCount(state.datasets.length);
        filterDatasets();
        if (state.datasets.length) {
          await selectDataset(state.datasets[0]);
        } else {
          el("detailsState").className = "empty";
          el("detailsState").textContent = "No datasets available on this server.";
          el("scenarioState").className = "empty";
          el("scenarioState").textContent = "No scenarios to show.";
        }
      } catch (err) {
        listRoot.innerHTML = `<div class="error">Failed to load /datasets: ${err.message}<div><button class="retry" id="retryDatasets">Retry</button></div></div>`;
        const btn = el("retryDatasets");
        if (btn) btn.onclick = () => init();
      }
      await loadDiagnostics();
    }

    el("datasetSearch").addEventListener("input", filterDatasets);
    el("scenarioSearch").addEventListener("input", () => renderScenarios(state.scenarios, state.scenarioMeta));
    el("scenarioSort").addEventListener("change", () => renderScenarios(state.scenarios, state.scenarioMeta));
    el("loadFramesBtn").addEventListener("click", () => loadFrameSummary());
    init();
  </script>
</body>
</html>"""
        return HTMLResponse(content=page, media_type="text/html; charset=utf-8")

    @app.get("/")
    def index_page():
        """Human-friendly landing page for quick server introspection."""
        esc = html.escape
        try:
            top_level_dirs = sorted(
                p.name for p in data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )[:12]
        except OSError:
            top_level_dirs = []
        try:
            dataset_ids = list_webauto_annotation_dataset_ids(data_dir)
        except OSError:
            dataset_ids = []
        dataset_count = len(dataset_ids)
        sample_ids = dataset_ids[:8]
        datasets_block = (
            "<br>".join(f"<code>{esc(did)}</code>" for did in sample_ids)
            if sample_ids
            else "(none)"
        )

        links = [
            ("Health", "/health"),
            ("Dataset Browser", "/datasets/browser"),
            ("Datasets", "/datasets"),
            ("Browser Diagnostics", "/browser/diagnostics"),
            ("OpenAPI JSON", "/openapi.json"),
            ("Swagger UI", "/docs"),
            ("ReDoc", "/redoc"),
        ]
        link_html = "".join(
            f'<li><a href="{href}" target="_blank" rel="noopener">{esc(label)}</a></li>'
            for label, href in links
        )
        top_dirs = "<br>".join(esc(name) for name in top_level_dirs) if top_level_dirs else "(none)"
        page = (
            "<!DOCTYPE html>"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>T4 Visualizer Server</title>"
            "<style>"
            ":root{color-scheme:light dark;--bg:#f6f7fb;--fg:#111318;--card:#fff;--border:#d9deea;--muted:#566072;--link:#1f5fe0;}"
            "@media (prefers-color-scheme: dark){:root{--bg:#111318;--fg:#e7ebf5;--card:#1a1f29;--border:#2a3342;--muted:#9ba7bd;--link:#79a6ff;}}"
            "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--fg);}"
            "main{max-width:980px;margin:0 auto;padding:1.25rem;}"
            "h1{margin:0 0 .5rem;font-size:1.35rem;}"
            "p{margin:.25rem 0 .75rem;color:var(--muted);}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:.9rem;margin-top:1rem;}"
            ".card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:.9rem 1rem;}"
            "h2{font-size:1rem;margin:0 0 .6rem;} ul{margin:.2rem 0 0 1.1rem;padding:0;} li{margin:.35rem 0;}"
            "a{color:var(--link);text-decoration:none;} a:hover{text-decoration:underline;}"
            "code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.9em;}"
            "pre{overflow:auto;white-space:pre-wrap;background:var(--bg);border:1px dashed var(--border);padding:.6rem;border-radius:8px;}"
            "</style></head><body><main>"
            "<h1>T4 Visualizer Server</h1>"
            "<p>Quick status and links for operators and API users.</p>"
            '<div class="grid">'
            '<section class="card"><h2>Runtime</h2>'
            f"<div><strong>data_dir</strong><br><code>{esc(str(data_dir))}</code></div>"
            f'<div style="margin-top:.55rem;"><strong>search_depth</strong> <code>{search_depth}</code></div>'
            f'<div style="margin-top:.55rem;"><strong>dataset_path_cache_ttl_s</strong> <code>{dataset_path_cache_ttl_s:g}</code></div>'
            "</section>"
            '<section class="card"><h2>Quick Links</h2><ul>' + link_html + "</ul></section>"
            '<section class="card"><h2>Top-level directories</h2>'
            f"<code>{top_dirs}</code></section>"
            '<section class="card"><h2>T4 datasets on this server</h2>'
            f"<div><strong>count</strong> <code>{dataset_count}</code></div>"
            '<div style="margin-top:.55rem;"><strong>sample IDs</strong><br>'
            f"{datasets_block}"
            "</div>"
            '<div style="margin-top:.55rem;"><a href="/datasets" target="_blank" rel="noopener">View full dataset list</a></div>'
            "</section>"
            '<section class="card"><h2>Example calls</h2>'
            "<pre>/datasets\n/datasets/{t4dataset_id}/availability\n/render/html?t4dataset_id=...&scenario_name=...&frame_index=0</pre>"
            "</section>"
            "</div></main></body></html>"
        )
        return HTMLResponse(content=page, media_type="text/html; charset=utf-8")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "visibility_mode": "debug" if _debug_visibility else "public",
        }

    @app.get("/datasets")
    def list_datasets():
        """Return dataset IDs plus discovery diagnostics under the configured data_dir."""
        resolved = str(data_dir.resolve())
        try:
            all_top_level_dirs = sorted(
                p.name for p in data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        except OSError:
            all_top_level_dirs = []
        top_level_dirs = all_top_level_dirs[:32]
        top_level_dir_count = len(all_top_level_dirs)
        ann_present = (data_dir / "annotation_dataset").is_dir()
        if not data_dir.exists():
            out: Dict[str, object] = {
                "data_dir": str(data_dir),
                "data_dir_resolved": resolved if _debug_visibility else "(redacted)",
                "datasets": [],
                "top_level_dirs": [],
                "top_level_dir_count": 0,
                "annotation_dataset_present": False,
                "hint": "data_dir does not exist on this host — check --data-dir.",
            }
            return out

        try:
            uniq = list_webauto_annotation_dataset_ids(data_dir)
        except OSError:
            uniq = []
        payload: Dict[str, object] = {
            "data_dir": str(data_dir),
            "data_dir_resolved": resolved if _debug_visibility else "(redacted)",
            "datasets": uniq,
            "top_level_dirs": top_level_dirs,
            "top_level_dir_count": top_level_dir_count,
            "annotation_dataset_present": ann_present,
            "runtime": {
                "search_depth": search_depth,
                "dataset_path_cache_ttl_s": dataset_path_cache_ttl_s,
                "visibility_mode": "debug" if _debug_visibility else "public",
            },
        }
        if not uniq:
            if not top_level_dirs:
                payload["hint"] = (
                    "No subdirectories under data_dir — confirm --data-dir on this host."
                )
            elif not ann_present and len(top_level_dirs) <= 8:
                payload["hint"] = (
                    "No T4 datasets listed. Data is often under a grouped folder "
                    "(e.g. annotation_dataset or a project folder like j6gen6_3) "
                    "as <group>/<uuid>/<version>/."
                )
            else:
                payload["hint"] = (
                    "No UUID-shaped folder names found under data_dir or "
                    "one level inside grouped folders (see top_level_dirs)."
                )
        return payload

    @app.get(
        "/datasets/{t4dataset_id}/availability",
        response_model=DatasetAvailabilityResponse,
        tags=["datasets"],
        summary="Check whether a dataset id is available under data_dir",
    )
    def dataset_availability(t4dataset_id: str):
        """Return whether *t4dataset_id* resolves under the configured ``data_dir``.

        Uses the same lookup as ``POST /render`` and ``GET /datasets/.../scenarios``
        (:func:`t4_visualizer.batch.find_dataset_in_dir`). Does not load Tier4.
        Results are cached briefly (see ``--dataset-path-cache-ttl``).
        """
        found = _path_cache.resolve(
            data_dir,
            search_depth,
            t4dataset_id,
            lambda: find_dataset_in_dir(data_dir, t4dataset_id, search_depth),
        )
        if found is not None:
            return DatasetAvailabilityResponse(
                t4dataset_id=t4dataset_id,
                available=True,
                dataset_path=str(found.resolve()) if _debug_visibility else None,
            )
        return DatasetAvailabilityResponse(
            t4dataset_id=t4dataset_id,
            available=False,
            dataset_path=None,
        )

    @app.get(
        "/datasets/{t4dataset_id}/scenarios",
        response_model=ScenariosListResponse,
        tags=["datasets"],
        summary="List scenarios in a dataset",
    )
    def list_dataset_scenarios(
        t4dataset_id: str,
        version: Optional[str] = None,
    ):
        """Return each scene's name (for ``scenario_name`` in ``POST /render``) and frame count."""
        dataset_path = _resolve_dataset(t4dataset_id)
        from t4_visualizer.visualize import list_scene_summaries

        try:
            t4 = _cache.load(dataset_path, version=version)
            raw = list_scene_summaries(t4)
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="scenarios_list_failed",
                message="Failed to list scenarios for dataset.",
                exc=exc,
            )

        frames = [int(row.get("nbr_samples") or 0) for row in raw]
        total_frames = sum(frames)

        return ScenariosListResponse(
            t4dataset_id=t4dataset_id,
            scenarios=[ScenarioOut(**row) for row in raw],
            version=version,
            total_scenarios=len(raw),
            total_frames=total_frames,
            min_frames=min(frames) if frames else 0,
            max_frames=max(frames) if frames else 0,
        )

    @app.get(
        "/browser/diagnostics",
        tags=["server"],
        summary="Diagnostics for dataset browser and server runtime",
    )
    def browser_diagnostics():
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tier4_stats = _cache.stats()
        if not _debug_visibility:
            tier4_stats["keys"] = [f"{len(tier4_stats.get('keys', []))} dataset(s) cached"]
        out: Dict[str, object] = {
            "status": "ok",
            "timestamp_utc": now_iso,
            "visibility_mode": "debug" if _debug_visibility else "public",
            "runtime": {
                "search_depth": search_depth,
                "dataset_path_cache_ttl_s": dataset_path_cache_ttl_s,
                "tier4_cache_size_limit": tier4_cache_size,
            },
            "caches": {
                "dataset_path_cache": _path_cache.stats(),
                "tier4_cache": tier4_stats,
            },
        }
        if _debug_visibility:
            out["runtime"]["data_dir"] = str(data_dir)
            out["runtime"]["data_dir_resolved"] = str(data_dir.resolve())
        return out

    @app.get(
        "/datasets/{t4dataset_id}/scenarios/{scenario_name}/frames/summary",
        tags=["datasets"],
        summary="List lightweight frame metadata for one scenario",
    )
    def scenario_frames_summary(
        t4dataset_id: str,
        scenario_name: str,
        version: Optional[str] = None,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        from t4_visualizer.visualize import list_scene_summaries

        try:
            t4 = _cache.load(dataset_path, version=version)
            scenes = list_scene_summaries(t4)
            scene_meta = next((s for s in scenes if s.get("name") == scenario_name), None)
            if scene_meta is None:
                _public_error(
                    status_code=404,
                    code="scenario_not_found",
                    message=f"Scenario '{scenario_name}' was not found.",
                    hint="Call GET /datasets/{id}/scenarios to inspect valid scenario names.",
                )
            total = int(scene_meta.get("nbr_samples") or 0)
            rows = []
            if offset < total:
                token = next((s.first_sample_token for s in t4.scene if s.name == scenario_name), None)
                if token is None:
                    _public_error(
                        status_code=404,
                        code="scenario_first_sample_missing",
                        message=f"Scenario '{scenario_name}' has no first sample token.",
                    )
                idx = 0
                while idx < offset and token:
                    sample = t4.get("sample", token)
                    token = sample.next or None
                    idx += 1
                remaining = min(limit, max(0, total - offset))
                for i in range(remaining):
                    if not token:
                        break
                    sample = t4.get("sample", token)
                    rows.append(
                        {
                            "frame_index": offset + i,
                            "sample_token": str(sample.token),
                            "timestamp_us": int(sample.timestamp),
                            "has_next": bool(getattr(sample, "next", None)),
                            "has_prev": bool(getattr(sample, "prev", None)),
                        }
                    )
                    token = sample.next or None
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="frame_summary_failed",
                message="Failed to list frame summary for scenario.",
                exc=exc,
            )

        return {
            "t4dataset_id": t4dataset_id,
            "scenario_name": scenario_name,
            "version": version,
            "offset": offset,
            "limit": limit,
            "total_frames": total,
            "frames": rows,
        }

    @app.get(
        "/viewer/three/schema",
        tags=["viewer"],
        summary="Binary schema for Three.js frame payload",
    )
    def viewer_three_schema():
        return {
            "format_version": "T4V3D002",
            "endianness": "little",
            "header_layout": [
                "magic:8",
                "header_len:uint32",
                "frame_index:uint32",
                "timestamp_us:uint64",
                "point_count:uint32",
                "box_count:uint32",
                "sample_token_len:uint16",
                "sample_token:utf8 bytes",
            ],
            "body_layout": {
                "points_f32": "[point_count][4] -> x,y,z,intensity",
                "box_corners_f32": "[box_count][24] -> 8 corners * xyz (x forward, y left, z up)",
                "box_labels_json": "uint32 length + utf8 json list[str]",
            },
        }

    @app.get(
        "/viewer/three/meta",
        tags=["viewer"],
        summary="Metadata bootstrap for Three.js viewer",
    )
    def viewer_three_meta(
        t4dataset_id: str,
        scenario_name: str,
        version: Optional[str] = None,
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        from t4_visualizer.visualize import list_scene_summaries

        try:
            t4 = _cache.load(dataset_path, version=version)
            scenes = list_scene_summaries(t4)
            scene_meta = next((s for s in scenes if s.get("name") == scenario_name), None)
            if scene_meta is None:
                _public_error(
                    status_code=404,
                    code="scenario_not_found",
                    message=f"Scenario '{scenario_name}' was not found.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="viewer_meta_failed",
                message="Failed to prepare viewer metadata.",
                exc=exc,
            )

        return {
            "t4dataset_id": t4dataset_id,
            "scenario_name": scenario_name,
            "version": version,
            "total_frames": int(scene_meta.get("nbr_samples") or 0),
            "format_version": "T4V3D002",
            "binary_endpoint_template": (
                f"/viewer/three/frame.bin?t4dataset_id={t4dataset_id}"
                f"&scenario_name={scenario_name}&frame_index={{frame_index}}"
                f"{f'&version={version}' if version else ''}"
            ),
        }

    @app.get(
        "/viewer/three/frame.bin",
        tags=["viewer"],
        summary="Binary 3D frame payload for Three.js viewer",
    )
    def viewer_three_frame_binary(
        t4dataset_id: str,
        scenario_name: str,
        frame_index: int = Query(..., ge=0),
        version: Optional[str] = None,
        response_format: str = Query("binary", alias="format"),
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        try:
            t4 = _cache.load(dataset_path, version=version)
            sample = _get_scenario_sample(t4, scenario_name, frame_index)
            points, boxes_3d = _pointcloud_and_boxes_for_sample(t4, sample)
            if response_format == "json":
                if not _debug_visibility:
                    _public_error(403, "json_debug_only", "JSON frame format is allowed in debug mode only.")
                return {
                    "frame_index": frame_index,
                    "sample_token": str(sample.token),
                    "timestamp_us": int(sample.timestamp),
                    "points": points[:, :4].tolist(),
                    "boxes": [
                        {
                            "center": [float(v) for v in getattr(b, "center", [0.0, 0.0, 0.0])[:3]],
                            "size": [float(v) for v in getattr(b, "size", [0.0, 0.0, 0.0])[:3]],
                            "label": str(getattr(b, "label", "") or ""),
                        }
                        for b in boxes_3d
                    ],
                }
            if response_format != "binary":
                _public_error(400, "invalid_format", "Use format=binary or format=json.")
            blob = _pack_viewer_frame_binary(
                frame_index=frame_index,
                sample_token=str(sample.token),
                timestamp_us=int(sample.timestamp),
                points_xyz_i=points,
                boxes_3d=boxes_3d,
            )
            return Response(
                content=blob,
                media_type="application/octet-stream",
                headers={
                    "X-T4V-Format": "T4V3D002",
                    "X-T4V-Frame-Index": str(frame_index),
                    "X-T4V-Sample-Token": str(sample.token),
                    "X-T4V-Timestamp-Us": str(sample.timestamp),
                    "X-T4V-Point-Fields": "x,y,z,intensity",
                    "X-T4V-Box-Fields": "8corners_xyz",
                },
            )
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="viewer_frame_binary_failed",
                message="Failed to build binary frame payload.",
                exc=exc,
            )

    @app.get(
        "/viewer/three/frames/window",
        tags=["viewer"],
        summary="Frame window metadata for prefetch planning",
    )
    def viewer_three_frames_window(
        t4dataset_id: str,
        scenario_name: str,
        center: int = Query(..., ge=0),
        radius: int = Query(2, ge=0, le=20),
        version: Optional[str] = None,
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        from t4_visualizer.visualize import list_scene_summaries

        try:
            t4 = _cache.load(dataset_path, version=version)
            scenes = list_scene_summaries(t4)
            scene_meta = next((s for s in scenes if s.get("name") == scenario_name), None)
            if scene_meta is None:
                _public_error(404, "scenario_not_found", f"Scenario '{scenario_name}' was not found.")
            total = int(scene_meta.get("nbr_samples") or 0)
            lo = max(0, center - radius)
            hi = min(total - 1, center + radius) if total > 0 else -1
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(500, "viewer_window_failed", "Failed to compute prefetch window.", exc=exc)

        frames = []
        for i in range(lo, hi + 1):
            frames.append(
                {
                    "frame_index": i,
                    "binary_url": (
                        f"/viewer/three/frame.bin?t4dataset_id={t4dataset_id}"
                        f"&scenario_name={scenario_name}&frame_index={i}"
                        f"{f'&version={version}' if version else ''}"
                    ),
                }
            )
        return {
            "t4dataset_id": t4dataset_id,
            "scenario_name": scenario_name,
            "version": version,
            "total_frames": total,
            "center": center,
            "radius": radius,
            "frames": frames,
        }

    @app.get(
        "/viewer/three/camera-overlay",
        tags=["viewer"],
        summary="Camera image + 2D annotation payload for Three.js overlay viewport",
    )
    def viewer_three_camera_overlay(
        t4dataset_id: str,
        scenario_name: str,
        frame_index: int = Query(..., ge=0),
        camera: Optional[str] = None,
        version: Optional[str] = None,
        show_annotations: bool = Query(True),
        all_cameras: bool = Query(False),
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        try:
            t4 = _cache.load(dataset_path, version=version)
            sample = _get_scenario_sample(t4, scenario_name, frame_index)
            payload = _camera_overlay_payload_for_sample(
                t4, sample, camera=camera, show_annotations=show_annotations
            )
            if all_cameras:
                cams = payload.get("available_cameras", []) or []
                all_rows = []
                for ch in cams:
                    row = _camera_overlay_payload_for_sample(
                        t4, sample, camera=str(ch), show_annotations=show_annotations
                    )
                    all_rows.append(row)
                payload["cameras_payload"] = all_rows
            payload.update(
                {
                    "t4dataset_id": t4dataset_id,
                    "scenario_name": scenario_name,
                    "frame_index": frame_index,
                    "sample_token": str(sample.token),
                    "timestamp_us": int(sample.timestamp),
                }
            )
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="viewer_camera_overlay_failed",
                message="Failed to generate camera overlay payload.",
                exc=exc,
            )

    @app.get("/viewer/three")
    def viewer_three_page(
        t4dataset_id: str = Query(...),
        scenario_name: str = Query(...),
        frame_index: int = Query(0, ge=0),
        version: Optional[str] = Query(None),
    ):
        esc = html.escape
        qs = (
            f"t4dataset_id={esc(t4dataset_id)}&scenario_name={esc(scenario_name)}"
            f"&frame_index={frame_index}"
            f"{f'&version={esc(version)}' if version else ''}"
        )
        page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>T4 Three.js Viewer</title>
<style>
  :root{{--bg:#060812;--panel:#0f1324cc;--text:#e8edff;--muted:#9aa6d4;--accent:#6ea8ff;--ok:#4bd08d;}}
  html,body{{margin:0;height:100%;background:radial-gradient(1200px 600px at 20% 0%,#1a2552 0%,#060812 65%);color:var(--text);font-family:ui-sans-serif,system-ui,sans-serif;}}
  #app{{display:grid;grid-template-rows:auto 1fr;height:100%;}}
  .top{{display:flex;gap:.7rem;align-items:center;padding:.7rem .9rem;background:var(--panel);backdrop-filter:blur(8px);border-bottom:1px solid #24315f;}}
  .pill{{font-size:.82rem;color:var(--muted)}}
  .controls{{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap}}
  button,input[type=range],select{{background:#111a35;color:var(--text);border:1px solid #2a3d78;border-radius:8px;padding:.3rem .5rem}}
  #canvasWrap{{position:relative;min-height:0}}
  #canvas{{width:100%;height:100%;display:block}}
  #hud{{position:absolute;right:12px;top:12px;background:var(--panel);border:1px solid #2b3f7d;border-radius:10px;padding:.6rem;min-width:260px}}
  #status{{font-size:.82rem;color:var(--muted)}}
  a{{color:var(--accent)}}
</style>
</head><body><div id="app">
  <div class="top">
    <strong>Three.js Viewer</strong>
    <span class="pill">dataset=<code>{esc(t4dataset_id)}</code></span>
    <span class="pill">scenario=<code>{esc(scenario_name)}</code></span>
    <div class="controls">
      <button id="playBtn">Play</button>
      <label>speed <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select>x</label>
      <button id="camReset">Reset Cam</button>
      <button id="camTop">Top</button>
      <button id="camFollow">Follow</button>
      <button id="toggleOverlay2d">Cam Viewport</button>
      <select id="overlayCamera" style="min-width:150px"></select>
      <input id="slider" type="range" min="0" max="0" value="0" />
      <span id="frameTxt">frame 0/0</span>
    </div>
    <a href="/datasets/browser">Back</a>
  </div>
  <div id="canvasWrap"><canvas id="canvas"></canvas>
    <div id="overlayWrap" style="position:absolute;left:12px;bottom:12px;width:min(46vw,820px);height:min(40vh,360px);display:none;border:1px solid #2b3f7d;border-radius:10px;overflow:hidden;background:#090d1a">
      <canvas id="overlayCanvas" style="width:100%;height:100%;display:block"></canvas>
    </div>
    <div id="hud">
      <div id="status">loading...</div>
      <div id="meta"></div>
      <hr style="border-color:#24315f;opacity:.6">
      <div style="display:grid;gap:.4rem;font-size:.82rem">
        <div>projection: <strong id="projMode">perspective</strong></div>
        <label>point size <input id="pointSize" type="range" min="0.02" max="0.25" step="0.01" value="0.08"></label>
        <label>point opacity <input id="pointOpacity" type="range" min="0.2" max="1.0" step="0.05" value="0.95"></label>
        <label>intensity gain <input id="intensityGain" type="range" min="0.4" max="2.5" step="0.1" value="1.0"></label>
        <label>box opacity <input id="boxOpacity" type="range" min="0.2" max="1.0" step="0.05" value="0.9"></label>
      </div>
    </div>
  </div>
</div>
<script type="importmap">{{"imports":{{"three":"https://esm.sh/three@0.161.0","three/addons/":"https://esm.sh/three@0.161.0/examples/jsm/"}}}}</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

const params = new URLSearchParams("{qs}");
const dataset = params.get("t4dataset_id");
const scenario = params.get("scenario_name");
const startFrame = Number(params.get("frame_index") || "0");
const version = params.get("version");
const qv = version ? `&version=${{encodeURIComponent(version)}}` : "";
const metaUrl = `/viewer/three/meta?t4dataset_id=${{encodeURIComponent(dataset)}}&scenario_name=${{encodeURIComponent(scenario)}}${{qv}}`;

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(canvas.clientWidth || window.innerWidth, canvas.clientHeight || (window.innerHeight-64), false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x070b16, 90, 230);
const perspCamera = new THREE.PerspectiveCamera(74, 1, 0.1, 1000);
const orthoCamera = new THREE.OrthographicCamera(-20, 20, 20, -20, 0.1, 1000);
let camera = perspCamera;
camera.position.set(-12, -8, 4.5);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0,0,0); controls.update();
scene.add(new THREE.AmbientLight(0x8aa6ff, 0.7));
const dl = new THREE.DirectionalLight(0xffffff, 0.9); dl.position.set(12, -10, 30); scene.add(dl);
const grid = new THREE.GridHelper(180, 90, 0x2b3a6f, 0x1a2342);
grid.rotation.x = Math.PI / 2; // XY ground plane (z-up)
scene.add(grid);

const pointsGeom = new THREE.BufferGeometry();
const pointsMat = new THREE.PointsMaterial({{size:0.08, vertexColors:true, transparent:true, opacity:0.95}});
const pointsObj = new THREE.Points(pointsGeom, pointsMat);
scene.add(pointsObj);
const boxesGroup = new THREE.Group(); scene.add(boxesGroup);
const egoAxes = new THREE.AxesHelper(2.8); scene.add(egoAxes);
const egoBody = new THREE.Mesh(
  new THREE.BoxGeometry(4.6, 1.9, 1.6),
  new THREE.MeshBasicMaterial({{color:0x6ea8ff, wireframe:true, transparent:true, opacity:0.45}})
);
egoBody.position.set(0,0,0.8); scene.add(egoBody);
const cache = new Map(); const MAX_CACHE = 28;
let totalFrames = 0; let playing = false; let frame = startFrame; let fps = 6;
let lastFrameTs = 0;
let followEgo = true;
let intensityGain = 1.0;
let overlayEnabled = false;
let overlayData = null;
let overlayTexture = null;
let overlayImageKey = "";
const overlayTextureCache = new Map();

const overlayCanvas = document.getElementById("overlayCanvas");
const overlayRenderer = new THREE.WebGLRenderer({{canvas: overlayCanvas, antialias:true, alpha:true}});
overlayRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
overlayRenderer.outputColorSpace = THREE.SRGBColorSpace;
overlayRenderer.toneMapping = THREE.NoToneMapping;
const overlayScene = new THREE.Scene();
const overlayCamera3d = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
overlayCamera3d.position.set(0,0,2);
const overlayImgMesh = new THREE.Mesh(
  new THREE.PlaneGeometry(2,2),
  new THREE.MeshBasicMaterial({{color:0xffffff, toneMapped:false}})
);
overlayScene.add(overlayImgMesh);
const overlayBoxes = new THREE.Group(); overlayScene.add(overlayBoxes);

const slider = document.getElementById("slider");
const frameTxt = document.getElementById("frameTxt");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
function setStatus(t){{ statusEl.textContent = t; }}
function setFrameText(){{ frameTxt.textContent = `frame ${{frame}}/${{Math.max(0,totalFrames-1)}}`; slider.value = String(frame); }}

function readFloat32Copied(buf, off, count){{
  const bytes = new Uint8Array(buf, off, count * 4);
  const copied = new Uint8Array(bytes.length);
  copied.set(bytes);
  return new Float32Array(copied.buffer);
}}

function parseFrameBuffer(buf){{
  const dv = new DataView(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (magic !== "T4V3D001" && magic !== "T4V3D002") throw new Error(`unexpected format ${{magic}}`);
  const frameIndex = dv.getUint32(12, true);
  const timestampUs = Number(dv.getBigUint64(16, true));
  const pointCount = dv.getUint32(24, true);
  const boxCount = dv.getUint32(28, true);
  const tokenLen = dv.getUint16(32, true);
  let off = 34;
  const sampleToken = new TextDecoder().decode(new Uint8Array(buf, off, tokenLen)); off += tokenLen;
  const pts = readFloat32Copied(buf, off, pointCount * 4); off += pointCount * 16;
  const boxStride = magic === "T4V3D002" ? 24 : 7;
  const boxes = readFloat32Copied(buf, off, boxCount * boxStride); off += boxCount * boxStride * 4;
  const labelLen = dv.getUint32(off, true); off += 4;
  const labels = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, off, labelLen)));
  return {{ frameIndex, timestampUs, sampleToken, pointCount, boxCount, pts, boxes, labels, format: magic }};
}}

async function fetchFrame(i){{
  if (cache.has(i)) return cache.get(i);
  const url = `/viewer/three/frame.bin?t4dataset_id=${{encodeURIComponent(dataset)}}&scenario_name=${{encodeURIComponent(scenario)}}&frame_index=${{i}}${{qv}}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`frame ${{i}}: HTTP ${{res.status}}`);
  const parsed = parseFrameBuffer(await res.arrayBuffer());
  cache.set(i, parsed);
  if (cache.size > MAX_CACHE) {{
    const k = cache.keys().next().value;
    cache.delete(k);
  }}
  return parsed;
}}

async function fetchCameraOverlay(i){{
  if (!overlayEnabled) return null;
  const camSel = document.getElementById("overlayCamera");
  const selected = camSel && camSel.value ? camSel.value : "";
  const allMode = selected === "__ALL__";
  const cam = selected && !allMode ? `&camera=${{encodeURIComponent(selected)}}` : "";
  const allFlag = allMode ? "&all_cameras=true" : "";
  const url = `/viewer/three/camera-overlay?t4dataset_id=${{encodeURIComponent(dataset)}}&scenario_name=${{encodeURIComponent(scenario)}}&frame_index=${{i}}${{qv}}${{cam}}${{allFlag}}&show_annotations=true`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`camera overlay HTTP ${{res.status}}`);
  return await res.json();
}}

function ensureOverlayCameraOptions(payload){{
  const sel = document.getElementById("overlayCamera");
  const cams = Array.isArray(payload.available_cameras) ? payload.available_cameras : [];
  if (!sel || !cams.length) return;
  const prev = sel.value;
  sel.innerHTML = ['<option value="__ALL__">ALL CAMERAS</option>']
    .concat(cams.map((c) => `<option value="${{c}}">${{c}}</option>`))
    .join("");
  if (prev === "__ALL__") sel.value = "__ALL__";
  else sel.value = cams.includes(prev) ? prev : (payload.camera || cams[0]);
}}

function renderCameraOverlay(payload){{
  overlayData = payload;
  const wrap = document.getElementById("overlayWrap");
  if (!wrap || wrap.style.display === "none") return;
  const selected = (document.getElementById("overlayCamera") || {{value:""}}).value;
  const allMode = selected === "__ALL__";
  overlayCanvas.style.display = "block";
  const w = wrap.clientWidth || 640;
  const h = wrap.clientHeight || 360;
  overlayRenderer.setSize(w, h, false);
  while (overlayScene.children.length) overlayScene.remove(overlayScene.children[0]);
  const rows = allMode
    ? (Array.isArray(payload.cameras_payload) ? payload.cameras_payload : [payload])
    : [payload];
  const n = Math.max(1, rows.length);
  const cols = Math.ceil(Math.sqrt(n));
  const tileW = 2 / cols;
  const tileH = 2 / cols;
  for (let i = 0; i < rows.length; i++) {{
    const p = rows[i];
    const r = Math.floor(i / cols);
    const c = i % cols;
    const cx = -1 + tileW * c + tileW / 2;
    const cy = 1 - tileH * r - tileH / 2;
    const iw = Math.max(1, Number(p.width || 1));
    const ih = Math.max(1, Number(p.height || 1));
    const imgAspect = iw / ih;
    const tileAspect = (w * tileW / 2) / (h * tileH / 2);
    let sx = tileW, sy = tileH;
    if (imgAspect > tileAspect) sy = tileW / imgAspect;
    else sx = tileH * imgAspect;

    const key = `${{p.sample_token || payload.sample_token || ""}}:${{p.camera || ""}}:${{p.image_format || ""}}:${{(p.image_base64 || "").length}}`;
    let tex = overlayTextureCache.get(key);
    if (!tex) {{
      const img = new Image();
      img.src = `data:${{p.image_format === "png" ? "image/png" : "image/jpeg"}};base64,${{p.image_base64 || ""}}`;
      tex = new THREE.Texture(img);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      overlayTextureCache.set(key, tex);
      if (overlayTextureCache.size > 24) {{
        const first = overlayTextureCache.keys().next().value;
        const old = overlayTextureCache.get(first);
        if (old) old.dispose();
        overlayTextureCache.delete(first);
      }}
    }}
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(sx, sy),
      new THREE.MeshBasicMaterial({{map: tex, color:0xffffff, toneMapped:false}})
    );
    plane.position.set(cx, cy, 0);
    overlayScene.add(plane);

    const bxs = Array.isArray(p.boxes_2d) ? p.boxes_2d : [];
    for (const b of bxs) {{
      const nx0 = ((b.x0 / iw) - 0.5) * sx + cx;
      const nx1 = ((b.x1 / iw) - 0.5) * sx + cx;
      const ny0 = (0.5 - (b.y0 / ih)) * sy + cy;
      const ny1 = (0.5 - (b.y1 / ih)) * sy + cy;
      const pts = [nx0,ny0,0, nx1,ny0,0, nx1,ny1,0, nx0,ny1,0, nx0,ny0,0];
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
      const m = new THREE.LineBasicMaterial({{color:0xff5ec4, transparent:true, opacity:0.95}});
      overlayScene.add(new THREE.Line(g, m));
    }}
  }}
  overlayRenderer.render(overlayScene, overlayCamera3d);
}}

async function prefetchAround(i){{
  const tasks = [];
  for (let d = 1; d <= 3; d++) {{
    for (const j of [i + d, i - d]) {{
      if (j >= 0 && j < totalFrames && !cache.has(j)) {{
        tasks.push(fetchFrame(j).catch(() => null));
      }}
    }}
  }}
  await Promise.all(tasks);
}}

function setFrameData(data){{
  const pos = new Float32Array(data.pointCount * 3);
  const col = new Float32Array(data.pointCount * 3);
  for (let i=0;i<data.pointCount;i++) {{
    const x = data.pts[i*4], y=data.pts[i*4+1], z=data.pts[i*4+2], inten=data.pts[i*4+3];
    pos[i*3]=x; pos[i*3+1]=y; pos[i*3+2]=z;
    const t = Math.max(0, Math.min(1, inten * intensityGain));
    col[i*3]=0.15+0.85*t; col[i*3+1]=0.35+0.5*(1-t); col[i*3+2]=1.0-0.6*t;
  }}
  pointsGeom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  pointsGeom.setAttribute("color", new THREE.BufferAttribute(col, 3));
  pointsGeom.computeBoundingSphere();
  while (boxesGroup.children.length) boxesGroup.remove(boxesGroup.children[0]);
  const edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]
  ];
  for (let i=0;i<data.boxCount;i++) {{
    if (data.format === "T4V3D002") {{
      const base = i * 24;
      const linePts = [];
      for (const [a,b] of edges) {{
        const a0 = base + a * 3;
        const b0 = base + b * 3;
        linePts.push(
          data.boxes[a0], data.boxes[a0 + 1], data.boxes[a0 + 2],
          data.boxes[b0], data.boxes[b0 + 1], data.boxes[b0 + 2]
        );
      }}
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(linePts, 3));
      const m = new THREE.LineBasicMaterial({{
        color:0xffd166, transparent:true,
        opacity:Number(document.getElementById("boxOpacity").value || "0.9")
      }});
      const lines = new THREE.LineSegments(g, m);
      boxesGroup.add(lines);
    }} else {{
      const cx=data.boxes[i*7], cy=data.boxes[i*7+1], cz=data.boxes[i*7+2];
      const sx=data.boxes[i*7+3], sy=data.boxes[i*7+4], sz=data.boxes[i*7+5], yaw=data.boxes[i*7+6];
      const g = new THREE.BoxGeometry(sx, sy, sz);
      const m = new THREE.MeshBasicMaterial({{color:0xffd166, wireframe:true, transparent:true, opacity:Number(document.getElementById("boxOpacity").value || "0.9")}});
      const box = new THREE.Mesh(g, m);
      box.position.set(cx, cy, cz);
      box.rotation.set(0, 0, yaw);
      boxesGroup.add(box);
    }}
  }}
  metaEl.innerHTML = `<div>sample=<code>${{data.sampleToken}}</code></div><div>timestamp_us=${{data.timestampUs}}</div><div>points=${{data.pointCount}} boxes=${{data.boxCount}}</div>`;
}}

async function showFrame(i){{
  frame = Math.max(0, Math.min(totalFrames-1, i));
  setFrameText();
  setStatus(`loading frame ${{frame}} ...`);
  const data = await fetchFrame(frame);
  setFrameData(data);
  if (followEgo) {{
    controls.target.set(0, 0, 0);
    controls.update();
  }}
  if (overlayEnabled) {{
    const payload = await fetchCameraOverlay(frame);
    if (payload) {{
      ensureOverlayCameraOptions(payload);
      renderCameraOverlay(payload);
    }}
  }}
  setStatus(`ready · cached=${{cache.size}}`);
  prefetchAround(frame);
}}

document.getElementById("playBtn").addEventListener("click", () => {{
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "Pause" : "Play";
}});
document.getElementById("speed").addEventListener("change", (e) => {{
  fps = 6 * Number(e.target.value || "1");
}});
slider.addEventListener("input", () => showFrame(Number(slider.value)));
document.getElementById("camReset").addEventListener("click", () => {{
  followEgo = false;
  camera.position.set(-12,-8,4.5); controls.target.set(0,0,0); controls.update();
}});
document.getElementById("camTop").addEventListener("click", () => {{
  followEgo = false;
  camera.position.set(0,0,60); controls.target.set(0,0,0); controls.update();
}});
document.getElementById("camFollow").addEventListener("click", () => {{
  followEgo = true;
  camera.position.set(-10,-2,3.8); controls.target.set(0,0,0); controls.update();
  setStatus("follow ego: on");
}});
document.getElementById("toggleOverlay2d").addEventListener("click", () => {{
  const wrap = document.getElementById("overlayWrap");
  const btn = document.getElementById("toggleOverlay2d");
  const show = wrap.style.display === "none";
  wrap.style.display = show ? "block" : "none";
  overlayEnabled = show;
  btn.textContent = show ? "Hide Cam" : "Cam Viewport";
  if (show) showFrame(frame).catch((err) => setStatus(`overlay error: ${{err.message}}`));
}});
document.getElementById("overlayCamera").addEventListener("change", () => {{
  if (overlayEnabled) {{
    showFrame(frame).catch((err) => setStatus(`overlay error: ${{err.message}}`));
  }}
}});
document.getElementById("pointSize").addEventListener("input", (e) => {{
  pointsMat.size = Number(e.target.value || "0.08");
}});
document.getElementById("pointOpacity").addEventListener("input", (e) => {{
  pointsMat.opacity = Number(e.target.value || "0.95");
}});
document.getElementById("intensityGain").addEventListener("input", async (e) => {{
  intensityGain = Number(e.target.value || "1.0");
  const data = cache.get(frame);
  if (data) setFrameData(data);
}});
document.getElementById("boxOpacity").addEventListener("input", () => {{
  for (const child of boxesGroup.children) {{
    if (child.material) child.material.opacity = Number(document.getElementById("boxOpacity").value || "0.9");
  }}
}});
document.getElementById("projMode").addEventListener("click", () => {{
  const wasPersp = camera === perspCamera;
  const next = wasPersp ? orthoCamera : perspCamera;
  next.position.copy(camera.position);
  next.up.set(0, 0, 1);
  if (wasPersp) {{
    const fr = 22;
    orthoCamera.left = -fr; orthoCamera.right = fr; orthoCamera.top = fr; orthoCamera.bottom = -fr;
    orthoCamera.updateProjectionMatrix();
    document.getElementById("projMode").textContent = "orthographic";
  }} else {{
    perspCamera.fov = 74;
    perspCamera.updateProjectionMatrix();
    document.getElementById("projMode").textContent = "perspective";
  }}
  controls.object = next;
  camera = next;
  controls.update();
}});
window.addEventListener("resize", () => {{
  const w = canvas.clientWidth || window.innerWidth;
  const h = canvas.clientHeight || (window.innerHeight-64);
  renderer.setSize(w, h, false);
  if (camera.isPerspectiveCamera) {{
    camera.aspect = w / h;
  }} else {{
    const fr = 22;
    camera.left = -fr * (w / h);
    camera.right = fr * (w / h);
    camera.top = fr;
    camera.bottom = -fr;
  }}
  camera.updateProjectionMatrix();
}});

function animate(ts){{
  requestAnimationFrame(animate);
  if (playing && totalFrames > 0 && ts - lastFrameTs > (1000 / Math.max(1, fps))) {{
    lastFrameTs = ts;
    const nxt = frame + 1 >= totalFrames ? 0 : frame + 1;
    showFrame(nxt).catch((err) => setStatus(`error: ${{err.message}}`));
  }}
  controls.update();
  renderer.render(scene, camera);
  if (overlayEnabled && overlayData) overlayRenderer.render(overlayScene, overlayCamera3d);
}}

(async function boot(){{
  setStatus("bootstrapping...");
  const metaRes = await fetch(metaUrl);
  if (!metaRes.ok) throw new Error(`meta HTTP ${{metaRes.status}}`);
  const meta = await metaRes.json();
  totalFrames = Number(meta.total_frames || 0);
  slider.max = String(Math.max(0, totalFrames - 1));
  camera = perspCamera;
  document.getElementById("projMode").textContent = "perspective";
  camera.position.set(-10, -2, 3.8);
  controls.target.set(0,0,0);
  controls.update();
  await showFrame(Math.min(startFrame, Math.max(0, totalFrames - 1)));
  animate(0);
}})().catch((err) => {{
  setStatus(`boot error: ${{err.message}}`);
}});
</script></body></html>"""
        return HTMLResponse(content=page, media_type="text/html; charset=utf-8")

    @app.post("/render", response_model=RenderResponse)
    def render_post(body: RenderRequest):
        """Render a single frame and return base64-encoded PNG images.

        Server-side timings are in the JSON body and duplicated on response headers
        so any HTTP client can read them without parsing JSON.
        """
        print(
            "[render:POST] "
            f"dataset={body.t4dataset_id} "
            f"scenario={body.scenario_name} "
            f"frame={body.frame_index} "
            f"targets={len(body.target_objects)} "
            f"cameras={body.cameras} "
            f"show_annotations={body.show_annotations} "
            f"crop_cameras={body.crop_cameras} "
            f"crop_padding={body.crop_padding} "
            f"crop_min_size={body.crop_min_size} "
            f"version={body.version}"
        )
        dataset_path = _resolve_dataset(body.t4dataset_id)

        target_objects = [
            TargetObject(
                uuid=o.uuid,
                x=o.x, y=o.y, z=o.z,
                label=o.label,
                width=o.width, length=o.length, height=o.height,
                yaw=o.yaw,
            )
            for o in body.target_objects
        ]

        payload, hdrs = _run_render(
            t4dataset_id=body.t4dataset_id,
            dataset_path=dataset_path,
            scenario_name=body.scenario_name,
            frame_index=body.frame_index,
            target_objects=target_objects,
            cameras=body.cameras,
            show_annotations=body.show_annotations,
            version=body.version,
            crop_cameras=body.crop_cameras,
            crop_padding=body.crop_padding,
            crop_min_size=body.crop_min_size,
        )
        return JSONResponse(content=jsonable_encoder(payload), headers=hdrs)

    @app.get("/render")
    def render_get(
        q=Depends(_render_get_query),
        response_format: Optional[str] = Query(
            None,
            alias="format",
            description="json or html; omit to auto (Sec-Fetch-Dest / Accept; curl→json).",
        ),
        accept: Optional[str] = Header(None, include_in_schema=False),
        sec_fetch_dest: Optional[str] = Header(
            None,
            alias="Sec-Fetch-Dest",
            include_in_schema=False,
        ),
    ):
        """Render one frame via query string (no ``target_objects``; use POST for those)."""
        print(
            "[render:GET] "
            f"dataset={q.t4dataset_id} "
            f"scenario={q.scenario_name} "
            f"frame={q.frame_index} "
            f"targets=0 "
            f"cameras={q.cameras} "
            f"show_annotations={q.show_annotations} "
            f"crop_cameras={q.crop_cameras} "
            f"crop_padding={q.crop_padding} "
            f"crop_min_size={q.crop_min_size} "
            f"version={q.version}"
        )
        fmt = _effective_render_format(accept, response_format, sec_fetch_dest)
        dataset_path = _resolve_dataset(q.t4dataset_id)
        cam_list = _parse_cameras_csv(q.cameras)
        payload, hdrs = _run_render(
            t4dataset_id=q.t4dataset_id,
            dataset_path=dataset_path,
            scenario_name=q.scenario_name,
            frame_index=q.frame_index,
            target_objects=[],
            cameras=cam_list,
            show_annotations=q.show_annotations,
            version=q.version,
            crop_cameras=q.crop_cameras,
            crop_padding=q.crop_padding,
            crop_min_size=q.crop_min_size,
        )
        if fmt == "html":
            return _html_response(_render_html_page(payload, q), hdrs)
        return JSONResponse(content=jsonable_encoder(payload), headers=hdrs)

    def _render_get_html_always(q):
        """Shared handler: HTML page with embedded PNGs (same query params as GET /render)."""
        print(
            "[render:GET:HTML] "
            f"dataset={q.t4dataset_id} "
            f"scenario={q.scenario_name} "
            f"frame={q.frame_index} "
            f"targets=0 "
            f"cameras={q.cameras} "
            f"show_annotations={q.show_annotations} "
            f"crop_cameras={q.crop_cameras} "
            f"crop_padding={q.crop_padding} "
            f"crop_min_size={q.crop_min_size} "
            f"version={q.version}"
        )
        dataset_path = _resolve_dataset(q.t4dataset_id)
        cam_list = _parse_cameras_csv(q.cameras)
        payload, hdrs = _run_render(
            t4dataset_id=q.t4dataset_id,
            dataset_path=dataset_path,
            scenario_name=q.scenario_name,
            frame_index=q.frame_index,
            target_objects=[],
            cameras=cam_list,
            show_annotations=q.show_annotations,
            version=q.version,
            crop_cameras=q.crop_cameras,
            crop_padding=q.crop_padding,
            crop_min_size=q.crop_min_size,
        )
        return _html_response(_render_html_page(payload, q), hdrs)

    @app.get("/render/view")
    def render_get_view(q=Depends(_render_get_query)):
        """Same parameters as ``GET /render`` but always returns an HTML page with PNGs."""
        return _render_get_html_always(q)

    @app.get("/render/html")
    def render_get_html(q=Depends(_render_get_query)):
        """Same as ``GET /render/view`` — explicit path for iframe ``src`` and bookmarks."""
        return _render_get_html_always(q)

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Serve the T4 Visualizer render API over HTTP."
    )
    parser.add_argument(
        "--data-dir", default="./t4datasets", metavar="PATH",
        help="Directory that contains T4 datasets (default: ./t4datasets).",
    )
    parser.add_argument(
        "--search-depth", type=int, default=1, metavar="N",
        help="Sub-levels to search for datasets under --data-dir (default: 1).",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", metavar="HOST",
        help="Bind host (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port", type=int, default=8000, metavar="PORT",
        help="Bind port (default: 8000).",
    )
    parser.add_argument(
        "--tier4-cache", type=int, default=8, metavar="N",
        help="Max number of Tier4 instances to keep in memory (default: 8).",
    )
    parser.add_argument(
        "--dataset-path-cache-ttl", type=float, default=30.0, metavar="SEC",
        help=(
            "Seconds to cache dataset id → path lookups (0 = disable). "
            "Default: 30."
        ),
    )
    parser.add_argument(
        "--reload", action="store_true", default=False,
        help="Enable uvicorn auto-reload (development only).",
    )
    parser.add_argument(
        "--visibility-mode",
        choices=("public", "debug"),
        default="public",
        help=(
            "Field visibility in JSON responses: "
            "'public' redacts sensitive host paths; 'debug' exposes internals."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point for the ``t4-server`` command."""
    args = _parse_args(argv)

    try:
        import uvicorn
    except ImportError as exc:
        print(
            "uvicorn is required to run the server. "
            "Install with: pip install fastapi uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Data directory : {data_dir}")
    print(f"  Search depth   : {args.search_depth}")
    print(f"  Tier4 cache    : {args.tier4_cache} datasets")
    print(f"  Visibility mode: {args.visibility_mode}")
    ttl = args.dataset_path_cache_ttl
    print(
        f"  Path lookup cache: "
        f"{'off' if ttl <= 0 else f'{ttl:g}s TTL'}"
    )
    print(f"  Listening on   : http://{args.host}:{args.port}")

    app = _build_app(
        data_dir=data_dir,
        search_depth=args.search_depth,
        tier4_cache_size=args.tier4_cache,
        dataset_path_cache_ttl_s=ttl,
        visibility_mode=args.visibility_mode,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
