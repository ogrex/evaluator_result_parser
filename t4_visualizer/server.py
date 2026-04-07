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
import math
import struct
import sys
import threading
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

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

    class CameraOverlayExtrasBody(BaseModel):
        """Optional GT/EST boxes (same schema as viewer ``bbox_layers`` postMessage) to project onto the image."""

        pred: List[Dict[str, Any]] = _Field(default_factory=list)
        gt: List[Dict[str, Any]] = _Field(default_factory=list)

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
        from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
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
    _vehicle_mesh_dir = Path(__file__).resolve().parent.parent / "assets" / "sample_vehicle_description" / "mesh"
    if _vehicle_mesh_dir.is_dir():
        from starlette.staticfiles import StaticFiles

        app.mount(
            "/viewer/assets/vehicle-mesh",
            StaticFiles(directory=str(_vehicle_mesh_dir)),
            name="vehicle_mesh",
        )
    _cache = _Tier4Cache(max_size=tier4_cache_size)
    _path_cache = _DatasetPathCache(ttl_s=dataset_path_cache_ttl_s)
    _debug_visibility = str(visibility_mode).strip().lower() == "debug"
    _templates_dir = Path(__file__).resolve().parent / "templates"

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

    @lru_cache(maxsize=16)
    def _load_template(filename: str) -> str:
        path = _templates_dir / filename
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            _safe_error(
                status_code=500,
                code="template_load_failed",
                message=f"Failed to load template '{filename}'.",
                exc=exc,
            )

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

    def _resolve_viewer_scenario_name(t4, scenario_name: Optional[str]) -> str:
        """Resolve scenario for viewer endpoints.

        Rules:
        - exact match when provided and found
        - if dataset has exactly one scenario, use it even when missing/wrong
        - otherwise require explicit valid scenario name
        """
        from t4_visualizer.visualize import list_scene_summaries

        raw = list_scene_summaries(t4)
        names = [str(r.get("name") or "") for r in raw if str(r.get("name") or "")]
        if not names:
            _public_error(404, "scenario_not_found", "No scenarios found in dataset.")
        wanted = (scenario_name or "").strip()
        if wanted and wanted in names:
            return wanted
        if len(names) == 1:
            return names[0]
        if not wanted:
            _public_error(
                400,
                "scenario_required",
                "scenario_name is required when dataset contains multiple scenarios.",
            )
        _public_error(
            404,
            "scenario_not_found",
            f"Scenario '{wanted}' was not found.",
            hint="Call GET /datasets/{id}/scenarios to inspect valid scenario names.",
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

    def _boxes_3d_ego_for_camera_projection(t4, sample) -> List:
        """3D boxes in ``base_link`` (ego), for projecting onto calibrated cameras.

        ``get_sample_data(..., as_sensor_coord=True)`` returns boxes in the *sensor* frame
        (e.g. LiDAR), but :func:`t4_visualizer.visualize._project_ego_to_cam` expects
        ego-frame points — use ``as_sensor_coord=False`` here only for overlay math.
        """
        from t4_visualizer.visualize import list_lidar_channels

        lidar_channels = list_lidar_channels(t4, sample)
        if not lidar_channels:
            return []
        token = sample.data.get(lidar_channels[0])
        if token is None:
            return []
        _path, boxes_3d, _ = t4.get_sample_data(
            token, as_3d=True, as_sensor_coord=False
        )
        return list(boxes_3d or [])

    def _box_xy_bev_m(box) -> Optional[Tuple[float, float]]:
        """Horizontal center for range filtering (t4 Box3D uses ``position``)."""
        pos = getattr(box, "position", None)
        if pos is not None:
            try:
                return float(pos[0]), float(pos[1])
            except (TypeError, ValueError, IndexError):
                pass
        center = getattr(box, "center", None)
        if center is not None:
            try:
                if hasattr(center, "__len__") and len(center) >= 2:
                    return float(center[0]), float(center[1])
            except (TypeError, ValueError, IndexError):
                pass
        return None

    def _box_label_str(box) -> str:
        sl = getattr(box, "semantic_label", None)
        if sl is not None:
            name = getattr(sl, "name", None)
            if name:
                return str(name)
        return str(getattr(box, "label", "") or "")

    def _box_xy_from_eval_dict(box: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        try:
            x = float(box.get("x", box.get("cx", 0.0)))
            y = float(box.get("y", box.get("cy", 0.0)))
            return x, y
        except (TypeError, ValueError):
            return None

    def _external_eval_boxes_to_2d_rows(
        t4,
        camera_token: str,
        boxes: Optional[List[Dict[str, Any]]],
        width: int,
        height: int,
        yaw_off: float,
        swap_lw: bool,
        max_range_m: float,
        max_boxes: int,
    ) -> List[Dict[str, object]]:
        from t4_visualizer.visualize import project_external_eval_box_to_image_roi

        if not boxes or width <= 0 or height <= 0:
            return []
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for b in boxes:
            if not isinstance(b, dict):
                continue
            xy = _box_xy_from_eval_dict(b)
            if xy is None:
                continue
            d = math.hypot(xy[0], xy[1])
            if d > float(max_range_m):
                continue
            candidates.append((d, b))
        candidates.sort(key=lambda t: t[0])
        candidates = candidates[: int(max_boxes)]
        rows: List[Dict[str, object]] = []
        for _d, b in candidates:
            try:
                roi = project_external_eval_box_to_image_roi(
                    t4,
                    str(camera_token),
                    b,
                    width,
                    height,
                    yaw_offset=yaw_off,
                    swap_lw=swap_lw,
                )
                if roi is None:
                    continue
                u_min, v_min, u_max, v_max, _vis = roi
                rows.append(
                    {
                        "x0": u_min,
                        "y0": v_min,
                        "x1": u_max,
                        "y1": v_max,
                        "label": str(b.get("label", b.get("class", "")) or ""),
                        "status": str(b.get("status", "TP") or "TP").upper(),
                    }
                )
            except Exception:
                continue
        return rows

    def _camera_overlay_payload_for_sample(
        t4,
        sample,
        camera: Optional[str] = None,
        show_annotations: bool = True,
        *,
        boxes_3d_scene: Optional[List] = None,
        max_range_m: float = 120.0,
        max_scene_boxes: int = 128,
        extra_pred_boxes: Optional[List[Dict[str, Any]]] = None,
        extra_gt_boxes: Optional[List[Dict[str, Any]]] = None,
        external_yaw_offset: float = math.pi / 2,
        external_swap_lw: bool = False,
    ) -> Dict[str, object]:
        import math

        from t4_visualizer.visualize import list_camera_channels, project_box3d_to_image_roi

        channels = list_camera_channels(t4, sample)
        if not channels:
            return {
                "camera": None,
                "available_cameras": [],
                "image_base64": "",
                "image_format": "jpeg",
                "boxes_2d": [],
                "boxes_2d_pred": [],
                "boxes_2d_eval_gt": [],
                "width": 0,
                "height": 0,
                "boxes_2d_source": None,
                "projection": None,
            }
        channel = camera if camera in channels else channels[0]
        token = sample.data.get(channel)
        if token is None:
            _public_error(404, "camera_token_not_found", f"Camera token not found for channel '{channel}'.")
        data_path, _, _ = t4.get_sample_data(token, as_3d=False)
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
        box_rows: List[Dict[str, object]] = []
        boxes_2d_pred: List[Dict[str, object]] = []
        boxes_2d_eval_gt: List[Dict[str, object]] = []
        proj_meta: Optional[Dict[str, object]] = None
        if show_annotations and width > 0 and height > 0:
            scene_boxes = boxes_3d_scene
            if scene_boxes is None:
                scene_boxes = _boxes_3d_ego_for_camera_projection(t4, sample)
            scene_boxes = scene_boxes or []
            # Range filter in BEV (ego xy), then nearest-first so dense scenes stay readable.
            candidates = []
            for b in scene_boxes:
                try:
                    xy = _box_xy_bev_m(b)
                    if xy is None:
                        continue
                    d = math.hypot(xy[0], xy[1])
                    if d > float(max_range_m):
                        continue
                    candidates.append((d, b))
                except Exception:
                    continue
            candidates.sort(key=lambda t: t[0])
            in_range = len(candidates)
            candidates = candidates[: int(max_scene_boxes)]
            for _d, b in candidates:
                try:
                    roi = project_box3d_to_image_roi(t4, str(token), b, width, height)
                    if roi is None:
                        continue
                    u_min, v_min, u_max, v_max, _vis = roi
                    box_rows.append(
                        {
                            "x0": u_min,
                            "y0": v_min,
                            "x1": u_max,
                            "y1": v_max,
                            "label": _box_label_str(b),
                        }
                    )
                except Exception:
                    continue
            proj_meta = {
                "source": "scene_3d",
                "max_range_m": float(max_range_m),
                "max_scene_boxes": int(max_scene_boxes),
                "in_range": in_range,
                "drawn": len(box_rows),
            }
        if width > 0 and height > 0:
            boxes_2d_pred = _external_eval_boxes_to_2d_rows(
                t4,
                str(token),
                extra_pred_boxes,
                width,
                height,
                external_yaw_offset,
                external_swap_lw,
                max_range_m,
                max_scene_boxes,
            )
            boxes_2d_eval_gt = _external_eval_boxes_to_2d_rows(
                t4,
                str(token),
                extra_gt_boxes,
                width,
                height,
                external_yaw_offset,
                external_swap_lw,
                max_range_m,
                max_scene_boxes,
            )
            if proj_meta is not None:
                proj_meta = {
                    **proj_meta,
                    "external_pred_drawn": len(boxes_2d_pred),
                    "external_gt_drawn": len(boxes_2d_eval_gt),
                }
            elif boxes_2d_pred or boxes_2d_eval_gt:
                proj_meta = {
                    "source": "external_eval",
                    "external_pred_drawn": len(boxes_2d_pred),
                    "external_gt_drawn": len(boxes_2d_eval_gt),
                }
        return {
            "camera": channel,
            "available_cameras": channels,
            "image_base64": base64.b64encode(img_bytes).decode("ascii"),
            "image_format": fmt,
            "boxes_2d": box_rows,
            "boxes_2d_pred": boxes_2d_pred,
            "boxes_2d_eval_gt": boxes_2d_eval_gt,
            "width": width,
            "height": height,
            "boxes_2d_source": ("scene_3d" if show_annotations else None),
            "projection": proj_meta,
        }

    def _camera_calibration_payload_for_sample(
        t4,
        sample,
        camera: Optional[str] = None,
        *,
        all_cameras: bool = False,
    ) -> Dict[str, object]:
        """Return camera calibration metadata for one sample.

        The payload is intentionally JSON-friendly so browser code and external
        tools can use it directly without depending on t4-devkit model classes.
        Extrinsics are reported as the calibrated sensor pose in ego/base_link:
        rotation is ``sensor -> ego`` quaternion ``[w, x, y, z]`` and
        translation is the sensor origin in ego meters.
        """
        from t4_visualizer.visualize import list_camera_channels

        channels = list_camera_channels(t4, sample)
        if not channels:
            return {
                "camera": None,
                "available_cameras": [],
                "sample_token": str(sample.token),
                "timestamp_us": int(sample.timestamp),
                "calibration": None,
                "calibrations": [],
            }

        requested_channel = camera if camera in channels else channels[0]
        selected_channels = channels if all_cameras else [requested_channel]
        rows: List[Dict[str, object]] = []

        for channel_name in selected_channels:
            token = sample.data.get(channel_name)
            if token is None:
                continue
            sample_data = t4.get("sample_data", token)
            calibrated_sensor = t4.get("calibrated_sensor", sample_data.calibrated_sensor_token)
            sensor = t4.get("sensor", calibrated_sensor.sensor_token)

            intrinsics_raw = getattr(calibrated_sensor, "camera_intrinsic", None)
            intrinsics: List[List[float]] = []
            if intrinsics_raw is not None:
                try:
                    intrinsics = [
                        [float(v) for v in row]
                        for row in intrinsics_raw
                    ]
                except TypeError:
                    intrinsics = []
            distortion_raw = getattr(calibrated_sensor, "camera_distortion", None)
            if distortion_raw is None:
                distortion_raw = getattr(calibrated_sensor, "distortion_coefficients", None)
            distortion = None
            if distortion_raw is not None:
                try:
                    distortion = [float(v) for v in distortion_raw]
                except TypeError:
                    distortion = None

            translation_raw = getattr(calibrated_sensor, "translation", None)
            if translation_raw is None:
                translation_raw = [0.0, 0.0, 0.0]
            rotation_raw = getattr(calibrated_sensor, "rotation", None)
            if rotation_raw is None:
                rotation_raw = [1.0, 0.0, 0.0, 0.0]
            translation = [float(v) for v in translation_raw]
            if hasattr(rotation_raw, "elements"):
                rotation = [float(v) for v in rotation_raw.elements]
            else:
                rotation = [float(v) for v in rotation_raw]

            rows.append(
                {
                    "camera": channel_name,
                    "sample_data_token": str(token),
                    "sample_token": str(sample.token),
                    "timestamp_us": int(sample.timestamp),
                    "sensor_token": str(getattr(sensor, "token", "") or ""),
                    "sensor_modality": str(getattr(sensor, "modality", "") or ""),
                    "sensor_channel": str(getattr(sensor, "channel", channel_name) or channel_name),
                    "calibrated_sensor_token": str(getattr(calibrated_sensor, "token", "") or ""),
                    "image_width": int(getattr(sample_data, "width", 0) or 0),
                    "image_height": int(getattr(sample_data, "height", 0) or 0),
                    "image_format": str(getattr(getattr(sample_data, "fileformat", None), "value", "") or ""),
                    "camera_intrinsic": intrinsics,
                    "camera_distortion": distortion,
                    "translation_ego_m": translation,
                    "rotation_sensor_to_ego_wxyz": rotation,
                }
            )

        single = next((row for row in rows if row.get("camera") == requested_channel), rows[0] if rows else None)
        return {
            "camera": requested_channel if rows else None,
            "available_cameras": channels,
            "sample_token": str(sample.token),
            "timestamp_us": int(sample.timestamp),
            "calibration": single,
            "calibrations": rows,
        }

    @lru_cache(maxsize=8)
    def _load_lanelet_graph(map_path_txt: str):
        """Load lanelet2 OSM once and return nodes/ways/relation way roles."""
        map_path = Path(map_path_txt)
        root = ET.parse(str(map_path)).getroot()
        nodes_latlon_ele: Dict[str, Tuple[float, float, float]] = {}
        nodes_localxy_ele: Dict[str, Tuple[float, float, float]] = {}
        for n in root.findall("node"):
            nid = n.attrib.get("id")
            if not nid:
                continue
            lat_txt = n.attrib.get("lat")
            lon_txt = n.attrib.get("lon")
            if lat_txt is None or lon_txt is None:
                continue
            lat = float(lat_txt)
            lon = float(lon_txt)
            ele = 0.0
            lx = None
            ly = None
            for t in n.findall("tag"):
                k = t.attrib.get("k")
                if k in ("ele", "height", "z"):
                    try:
                        ele = float(t.attrib.get("v", "0"))
                    except ValueError:
                        ele = 0.0
                elif k in ("local_x", "x"):
                    try:
                        lx = float(t.attrib.get("v", "0"))
                    except ValueError:
                        lx = None
                elif k in ("local_y", "y"):
                    try:
                        ly = float(t.attrib.get("v", "0"))
                    except ValueError:
                        ly = None
            nodes_latlon_ele[nid] = (lat, lon, ele)
            if lx is not None and ly is not None:
                nodes_localxy_ele[nid] = (lx, ly, ele)

        ways: Dict[str, List[str]] = {}
        for w in root.findall("way"):
            wid = w.attrib.get("id")
            if not wid:
                continue
            refs = []
            for nd in w.findall("nd"):
                ref = nd.attrib.get("ref")
                if ref:
                    refs.append(ref)
            if refs:
                ways[wid] = refs

        lanelet_way_ids = set()
        role_of_way: Dict[str, str] = {}
        for rel in root.findall("relation"):
            is_lanelet = False
            for t in rel.findall("tag"):
                if t.attrib.get("k") == "type" and t.attrib.get("v") == "lanelet":
                    is_lanelet = True
                    break
            if not is_lanelet:
                continue
            for m in rel.findall("member"):
                if m.attrib.get("type") != "way":
                    continue
                ref = m.attrib.get("ref")
                if not ref:
                    continue
                lanelet_way_ids.add(ref)
                role_of_way[ref] = m.attrib.get("role", "unknown")
        return nodes_latlon_ele, nodes_localxy_ele, ways, lanelet_way_ids, role_of_way

    def _lanelet_lines_payload(
        dataset_path: Path,
        t4,
        scenario_name: str,
        frame_index: int,
        max_segments: int = 120000,
        clip_radius_m: float = 160.0,
    ) -> Dict[str, object]:
        """Return lanelet segments transformed into current ego frame."""
        map_path = dataset_path / "map" / "lanelet2_map.osm"
        if not map_path.exists():
            return {"available": False, "reason": f"map file not found: {map_path}", "segments": []}
        try:
            nodes_latlon_ele, nodes_localxy_ele, ways, lanelet_way_ids, role_of_way = _load_lanelet_graph(str(map_path))
        except Exception as exc:
            _safe_error(500, "lanelet_parse_failed", "Failed to parse lanelet2_map.osm.", exc=exc)

        sample = _get_scenario_sample(t4, scenario_name, frame_index)
        from t4_visualizer.visualize import list_lidar_channels

        lidar_channels = list_lidar_channels(t4, sample)
        if not lidar_channels:
            return {"available": False, "reason": "no lidar channel in sample", "segments": []}
        lidar_token = sample.data.get(lidar_channels[0])
        if lidar_token is None:
            return {"available": False, "reason": "no lidar sample_data token", "segments": []}
        sample_data = t4.get("sample_data", lidar_token)
        ego_pose = t4.get("ego_pose", sample_data.ego_pose_token)
        geocoord = getattr(ego_pose, "geocoordinate", None)
        ego_t = getattr(ego_pose, "translation", None) or [0.0, 0.0, 0.0]
        ego_tx = float(ego_t[0]) if len(ego_t) >= 1 else 0.0
        ego_ty = float(ego_t[1]) if len(ego_t) >= 2 else 0.0
        ego_tz = float(ego_t[2]) if len(ego_t) >= 3 else 0.0
        lat0 = None
        lon0 = None
        alt0 = ego_tz
        align_mode = "translation_xy"
        if geocoord and len(geocoord) >= 2:
            lat0 = float(geocoord[0])
            lon0 = float(geocoord[1])
            alt0 = float(geocoord[2]) if len(geocoord) >= 3 and geocoord[2] is not None else ego_tz
            align_mode = "geocoordinate"
        rot = getattr(ego_pose, "rotation", None) or [1.0, 0.0, 0.0, 0.0]
        try:
            w, x, y, z = [float(v) for v in rot]
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        except Exception:
            yaw = 0.0
        c = math.cos(-yaw)
        s = math.sin(-yaw)

        segments = []
        count = 0
        for wid in lanelet_way_ids:
            refs = ways.get(wid)
            if not refs or len(refs) < 2:
                continue
            role = role_of_way.get(wid, "unknown")
            for i in range(len(refs) - 1):
                p0 = nodes_latlon_ele.get(refs[i])
                p1 = nodes_latlon_ele.get(refs[i + 1])
                if p0 is None or p1 is None:
                    continue
                lat_a, lon_a, ele_a = p0
                lat_b, lon_b, ele_b = p1
                # Compute metric XY in global-ish frame.
                if align_mode == "geocoordinate":
                    gx0 = (lon_a - lon0) * 111320.0 * math.cos(math.radians(lat0))
                    gy0 = (lat_a - lat0) * 110540.0
                    gx1 = (lon_b - lon0) * 111320.0 * math.cos(math.radians(lat0))
                    gy1 = (lat_b - lat0) * 110540.0
                elif refs[i] in nodes_localxy_ele and refs[i + 1] in nodes_localxy_ele:
                    la = nodes_localxy_ele[refs[i]]
                    lb = nodes_localxy_ele[refs[i + 1]]
                    gx0, gy0 = float(la[0]) - ego_tx, float(la[1]) - ego_ty
                    gx1, gy1 = float(lb[0]) - ego_tx, float(lb[1]) - ego_ty
                else:
                    # Last-resort fallback: local map XY around first node.
                    # This keeps map visible (may be shifted when no geo info exists).
                    lat_ref, lon_ref, _ = next(iter(nodes_latlon_ele.values()))
                    gx0 = (lon_a - lon_ref) * 111320.0 * math.cos(math.radians(lat_ref))
                    gy0 = (lat_a - lat_ref) * 110540.0
                    gx1 = (lon_b - lon_ref) * 111320.0 * math.cos(math.radians(lat_ref))
                    gy1 = (lat_b - lat_ref) * 110540.0
                # map frame -> ego frame (x forward, y left)
                ex0 = c * gx0 - s * gy0
                ey0 = s * gx0 + c * gy0
                ex1 = c * gx1 - s * gy1
                ey1 = s * gx1 + c * gy1
                if clip_radius_m > 0:
                    if (ex0 * ex0 + ey0 * ey0 > clip_radius_m * clip_radius_m) and (
                        ex1 * ex1 + ey1 * ey1 > clip_radius_m * clip_radius_m
                    ):
                        continue
                segments.append(
                    {
                        "x0": ex0, "y0": ey0, "z0": float(ele_a - alt0),
                        "x1": ex1, "y1": ey1, "z1": float(ele_b - alt0),
                        "role": role,
                    }
                )
                count += 1
                if count >= max_segments:
                    break
            if count >= max_segments:
                break

        return {
            "available": True,
            "map_path": str(map_path) if _debug_visibility else "lanelet2_map.osm",
            "segment_count": len(segments),
            "segments": segments,
            "projection": "ego_local_xy_m",
            "origin_latlon": [lat0, lon0] if _debug_visibility else None,
            "ego_yaw_rad": yaw if _debug_visibility else None,
            "align_mode": align_mode,
            "warning": (
                "ego_pose.geocoordinate missing; using translation/local fallback alignment."
                if align_mode != "geocoordinate"
                else None
            ),
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
        page = _load_template("datasets_browser.html")
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
        scenario_name: Optional[str] = None,
        version: Optional[str] = None,
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        from t4_visualizer.visualize import list_scene_summaries

        try:
            t4 = _cache.load(dataset_path, version=version)
            resolved_scenario = _resolve_viewer_scenario_name(t4, scenario_name)
            scenes = list_scene_summaries(t4)
            scene_meta = next((s for s in scenes if s.get("name") == resolved_scenario), None)
            if scene_meta is None:
                _public_error(
                    status_code=404,
                    code="scenario_not_found",
                    message=f"Scenario '{resolved_scenario}' was not found.",
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
            "scenario_name": resolved_scenario,
            "version": version,
            "total_frames": int(scene_meta.get("nbr_samples") or 0),
            "format_version": "T4V3D002",
            "binary_endpoint_template": (
                f"/viewer/three/frame.bin?t4dataset_id={t4dataset_id}"
                f"&scenario_name={resolved_scenario}&frame_index={{frame_index}}"
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
        scenario_name: Optional[str] = None,
        frame_index: int = Query(..., ge=0),
        version: Optional[str] = None,
        response_format: str = Query("binary", alias="format"),
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        try:
            t4 = _cache.load(dataset_path, version=version)
            resolved_scenario = _resolve_viewer_scenario_name(t4, scenario_name)
            sample = _get_scenario_sample(t4, resolved_scenario, frame_index)
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
        scenario_name: Optional[str] = None,
        center: int = Query(..., ge=0),
        radius: int = Query(2, ge=0, le=20),
        version: Optional[str] = None,
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        from t4_visualizer.visualize import list_scene_summaries

        try:
            t4 = _cache.load(dataset_path, version=version)
            resolved_scenario = _resolve_viewer_scenario_name(t4, scenario_name)
            scenes = list_scene_summaries(t4)
            scene_meta = next((s for s in scenes if s.get("name") == resolved_scenario), None)
            if scene_meta is None:
                _public_error(404, "scenario_not_found", f"Scenario '{resolved_scenario}' was not found.")
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
                        f"&scenario_name={resolved_scenario}&frame_index={i}"
                        f"{f'&version={version}' if version else ''}"
                    ),
                }
            )
        return {
            "t4dataset_id": t4dataset_id,
            "scenario_name": resolved_scenario,
            "version": version,
            "total_frames": total,
            "center": center,
            "radius": radius,
            "frames": frames,
        }

    def _viewer_three_camera_overlay_core(
        t4dataset_id: str,
        scenario_name: Optional[str],
        frame_index: int,
        camera: Optional[str],
        version: Optional[str],
        show_annotations: bool,
        all_cameras: bool,
        max_range_m: float,
        max_scene_boxes: int,
        extra_pred_boxes: Optional[List[Dict[str, Any]]],
        extra_gt_boxes: Optional[List[Dict[str, Any]]],
        external_yaw_offset: float,
        external_swap_lw: bool,
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        t4 = _cache.load(dataset_path, version=version)
        resolved_scenario = _resolve_viewer_scenario_name(t4, scenario_name)
        sample = _get_scenario_sample(t4, resolved_scenario, frame_index)
        boxes_3d_scene = None
        if show_annotations:
            boxes_3d_scene = _boxes_3d_ego_for_camera_projection(t4, sample)
        payload = _camera_overlay_payload_for_sample(
            t4,
            sample,
            camera=camera,
            show_annotations=show_annotations,
            boxes_3d_scene=boxes_3d_scene,
            max_range_m=max_range_m,
            max_scene_boxes=max_scene_boxes,
            extra_pred_boxes=extra_pred_boxes,
            extra_gt_boxes=extra_gt_boxes,
            external_yaw_offset=external_yaw_offset,
            external_swap_lw=external_swap_lw,
        )
        if all_cameras:
            cams = payload.get("available_cameras", []) or []
            all_rows = []
            for ch in cams:
                row = _camera_overlay_payload_for_sample(
                    t4,
                    sample,
                    camera=str(ch),
                    show_annotations=show_annotations,
                    boxes_3d_scene=boxes_3d_scene,
                    max_range_m=max_range_m,
                    max_scene_boxes=max_scene_boxes,
                    extra_pred_boxes=extra_pred_boxes,
                    extra_gt_boxes=extra_gt_boxes,
                    external_yaw_offset=external_yaw_offset,
                    external_swap_lw=external_swap_lw,
                )
                all_rows.append(row)
            payload["cameras_payload"] = all_rows
        payload.update(
            {
                "t4dataset_id": t4dataset_id,
                "scenario_name": resolved_scenario,
                "frame_index": frame_index,
                "sample_token": str(sample.token),
                "timestamp_us": int(sample.timestamp),
            }
        )
        return payload

    @app.get(
        "/viewer/three/camera-overlay",
        tags=["viewer"],
        summary="Camera image + 2D annotation payload for Three.js overlay viewport",
    )
    def viewer_three_camera_overlay(
        t4dataset_id: str,
        scenario_name: Optional[str] = None,
        frame_index: int = Query(..., ge=0),
        camera: Optional[str] = None,
        version: Optional[str] = None,
        show_annotations: bool = Query(True),
        all_cameras: bool = Query(False),
        max_range_m: float = Query(
            120.0,
            ge=5.0,
            le=500.0,
            description="Only project 3D boxes whose ego-frame center (xy) is within this radius [m].",
        ),
        max_scene_boxes: int = Query(
            128,
            ge=1,
            le=500,
            description="Max number of nearest boxes (after range filter) to project per camera.",
        ),
        external_bbox_yaw_offset: float = Query(
            math.pi / 2,
            description="Yaw offset [rad] for external eval boxes (match viewer external_bbox_yaw_offset).",
        ),
        external_bbox_swap_lw: bool = Query(
            False,
            description="Swap length/width for external eval boxes (match viewer external_bbox_swap_lw).",
        ),
    ):
        try:
            return _viewer_three_camera_overlay_core(
                t4dataset_id,
                scenario_name,
                frame_index,
                camera,
                version,
                show_annotations,
                all_cameras,
                max_range_m,
                max_scene_boxes,
                None,
                None,
                external_bbox_yaw_offset,
                external_bbox_swap_lw,
            )
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="viewer_camera_overlay_failed",
                message="Failed to generate camera overlay payload.",
                exc=exc,
            )

    @app.get(
        "/viewer/three/camera-info",
        tags=["viewer"],
        summary="Camera calibration metadata for a viewer frame",
    )
    def viewer_three_camera_info(
        t4dataset_id: str,
        scenario_name: Optional[str] = None,
        frame_index: int = Query(..., ge=0),
        camera: Optional[str] = Query(
            None,
            description="Camera channel to return. Omit to use the first available camera.",
        ),
        version: Optional[str] = None,
        all_cameras: bool = Query(
            False,
            description="When true, include calibration rows for every camera in the sample.",
        ),
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        try:
            t4 = _cache.load(dataset_path, version=version)
            resolved_scenario = _resolve_viewer_scenario_name(t4, scenario_name)
            sample = _get_scenario_sample(t4, resolved_scenario, frame_index)
            payload = _camera_calibration_payload_for_sample(
                t4,
                sample,
                camera=camera,
                all_cameras=all_cameras,
            )
            payload.update(
                {
                    "t4dataset_id": t4dataset_id,
                    "scenario_name": resolved_scenario,
                    "frame_index": frame_index,
                    "version": version,
                }
            )
            return payload
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="viewer_camera_info_failed",
                message="Failed to build camera calibration payload.",
                exc=exc,
            )

    @app.post(
        "/viewer/three/camera-overlay",
        tags=["viewer"],
        summary="Camera overlay with GT/EST box arrays (same schema as bbox_layers postMessage)",
    )
    def viewer_three_camera_overlay_post(
        body: CameraOverlayExtrasBody = Body(default_factory=CameraOverlayExtrasBody),
        t4dataset_id: str = Query(..., description="Dataset id"),
        scenario_name: Optional[str] = None,
        frame_index: int = Query(..., ge=0),
        camera: Optional[str] = None,
        version: Optional[str] = None,
        show_annotations: bool = Query(True),
        all_cameras: bool = Query(False),
        max_range_m: float = Query(
            120.0,
            ge=5.0,
            le=500.0,
            description="Only project boxes whose ego-frame center (xy) is within this radius [m].",
        ),
        max_scene_boxes: int = Query(
            128,
            ge=1,
            le=500,
            description="Max number of nearest boxes (after range filter) per layer.",
        ),
        external_bbox_yaw_offset: float = Query(
            math.pi / 2,
            description="Yaw offset [rad] for external eval boxes.",
        ),
        external_bbox_swap_lw: bool = Query(False, description="Swap length/width for external eval boxes."),
    ):
        try:
            pred = list(body.pred) if body and body.pred else []
            gt = list(body.gt) if body and body.gt else []
            return _viewer_three_camera_overlay_core(
                t4dataset_id,
                scenario_name,
                frame_index,
                camera,
                version,
                show_annotations,
                all_cameras,
                max_range_m,
                max_scene_boxes,
                pred,
                gt,
                external_bbox_yaw_offset,
                external_bbox_swap_lw,
            )
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="viewer_camera_overlay_failed",
                message="Failed to generate camera overlay payload.",
                exc=exc,
            )

    @app.get(
        "/viewer/three/lanelet-lines",
        tags=["viewer"],
        summary="Lanelet line segments for Three.js map overlay",
    )
    def viewer_three_lanelet_lines(
        t4dataset_id: str,
        scenario_name: Optional[str] = None,
        frame_index: int = Query(0, ge=0),
        version: Optional[str] = None,
        max_segments: int = Query(120000, ge=1000, le=500000),
        clip_radius_m: float = Query(160.0, ge=20.0, le=500.0),
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        try:
            t4 = _cache.load(dataset_path, version=version)
            resolved_scenario = _resolve_viewer_scenario_name(t4, scenario_name)
            return _lanelet_lines_payload(
                dataset_path,
                t4=t4,
                scenario_name=resolved_scenario,
                frame_index=frame_index,
                max_segments=max_segments,
                clip_radius_m=clip_radius_m,
            )
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                status_code=500,
                code="viewer_lanelet_failed",
                message="Failed to build lanelet line payload.",
                exc=exc,
            )

    @app.get(
        "/viewer/three/debug/message-received",
        tags=["viewer"],
        summary="Debug ack: viewer received bbox_layers or bbox_layers_by_frame postMessage",
    )
    def viewer_three_debug_message_received(
        t4dataset_id: Optional[str] = None,
        scenario_name: Optional[str] = None,
        frame_index: Optional[int] = None,
        gt_count: int = Query(0, ge=0),
        pred_count: int = Query(0, ge=0),
        matched_count: int = Query(0, ge=0),
        message_type: Optional[str] = Query(
            None,
            description="bbox_layers | bbox_layers_by_frame (optional, for logging).",
        ),
        frames_count: Optional[int] = Query(
            None,
            ge=0,
            description="Number of frame keys when message_type=bbox_layers_by_frame.",
        ),
    ):
        print(
            "[viewer:message] "
            f"dataset={t4dataset_id} "
            f"scenario={scenario_name} "
            f"frame={frame_index} "
            f"gt={gt_count} pred={pred_count} matched={matched_count} "
            f"type={message_type or 'bbox_layers'} "
            f"frames_count={frames_count}"
        )
        return {
            "ok": True,
            "dataset": t4dataset_id,
            "scenario": scenario_name,
            "frame_index": frame_index,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "matched_count": matched_count,
            "message_type": message_type,
            "frames_count": frames_count,
        }

    @app.get("/viewer/three")
    def viewer_three_page(
        t4dataset_id: str = Query(...),
        scenario_name: Optional[str] = Query(None),
        frame_index: int = Query(0, ge=0),
        version: Optional[str] = Query(None),
    ):
        esc = html.escape
        qs_params = {
            "t4dataset_id": t4dataset_id,
            "frame_index": frame_index,
        }
        if scenario_name:
            qs_params["scenario_name"] = scenario_name
        if version:
            qs_params["version"] = version
        qs = urlencode(qs_params)
        scenario_label = scenario_name or "(auto)"
        tmpl = _load_template("viewer_three.html")
        page = (
            tmpl
            .replace("__DATASET_ID__", esc(t4dataset_id))
            .replace("__SCENARIO_NAME__", esc(scenario_label))
            .replace("__FRAME_INDEX__", str(frame_index))
            .replace("__QS__", qs)
        )
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
