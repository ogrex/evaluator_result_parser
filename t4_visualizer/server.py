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
import logging
import math
import os
import struct
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier4 in-memory cache
# ---------------------------------------------------------------------------

class _Tier4Cache:
    """Thread-safe LRU cache for Tier4 instances keyed by (dataset path, version)."""

    def __init__(self, max_size: int = 8):
        self._cache: Dict[Tuple[Path, Optional[str]], object] = {}   # (path, version) → Tier4
        self._order: List[Tuple[Path, Optional[str]]] = []           # LRU order (most-recent last)
        self._max_size = max_size
        self._lock = threading.Lock()
        self._load_locks: Dict[Tuple[Path, Optional[str]], threading.Lock] = {}
        self._hits = 0
        self._misses = 0
        self._loads = 0
        self._evictions = 0

    def get(self, path: Path, version: Optional[str] = None):
        """Return cached Tier4 for *(path, version)*, or None if not present."""
        key = (path, version)
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
        return None

    def put(self, path: Path, t4, version: Optional[str] = None) -> None:
        """Store *t4* for *(path, version)*, evicting the LRU entry if over capacity."""
        key = (path, version)
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._max_size:
                evict = self._order.pop(0)
                del self._cache[evict]
                self._evictions += 1
            self._cache[key] = t4
            self._order.append(key)

    def load(self, path: Path, version: Optional[str] = None):
        """Return a Tier4 instance for *(path, version)*, loading it if not cached.

        Concurrent requests for the same key share a single load (singleflight):
        one thread runs the expensive Tier4 construction while the others block
        on the per-key lock and then hit the cache.
        """
        key = (path, version)
        t4 = self.get(path, version)
        if t4 is not None:
            return t4

        with self._lock:
            load_lock = self._load_locks.setdefault(key, threading.Lock())
        with load_lock:
            t4 = self.get(path, version)
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
            self.put(path, t4, version)
        with self._lock:
            self._load_locks.pop(key, None)
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
                "keys": [
                    f"{p}@{v}" if v else str(p) for p, v in self._order
                ],
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
    def _key(data_dirs: List[Path], search_depth: int, t4dataset_id: str) -> str:
        # Create a unique key based on all data_dirs
        dirs_str = "|".join(sorted(str(d.resolve()) for d in data_dirs))
        return f"{dirs_str}|{search_depth}|{t4dataset_id}"

    def resolve(
        self,
        data_dirs: List[Path],
        search_depth: int,
        t4dataset_id: str,
        find_fn,
    ) -> Optional[Path]:
        """Return cached path, call *find_fn* () -> Optional[Path] on miss."""
        if self._ttl_s <= 0:
            return find_fn()

        key = self._key(data_dirs, search_depth, t4dataset_id)
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
# Three.js viewer binary frame protocol (T4V3D002)
# ---------------------------------------------------------------------------

VIEWER_FRAME_MAGIC = b"T4V3D002"
# Little-endian fixed header; the variable-length sample token follows it,
# so the body starts at VIEWER_FRAME_HEADER_LEN + sample_token_len.
# magic[8] + header_len:u32 + frame_index:u32 + timestamp_us:u64
# + point_count:u32 + box_count:u32 + sample_token_len:u16
VIEWER_FRAME_HEADER_FMT = "<8sIIQIIH"
VIEWER_FRAME_HEADER_LEN = struct.calcsize(VIEWER_FRAME_HEADER_FMT)


def viewer_frame_schema() -> Dict[str, object]:
    """Schema document for `/viewer/three/schema`, derived from the packer constants."""
    return {
        "format_version": VIEWER_FRAME_MAGIC.decode("ascii"),
        "endianness": "little",
        "header_struct_fmt": VIEWER_FRAME_HEADER_FMT,
        "header_len": VIEWER_FRAME_HEADER_LEN,
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
            "offset": "header_len + sample_token_len",
            "points_f32": "[point_count][4] -> x,y,z,intensity",
            "box_corners_f32": "[box_count][24] -> 8 corners * xyz (x forward, y left, z up)",
            "box_labels_json": "uint32 length + utf8 json list[str]",
        },
    }


def pack_viewer_frame_binary(
    *,
    frame_index: int,
    sample_token: str,
    timestamp_us: int,
    points_xyz_i,
    boxes_3d,
) -> bytes:
    """Pack one viewer frame into the T4V3D002 wire format (see viewer_frame_schema)."""
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
            corner_pts = [
                [cx + hx, cy + hy, cz + hz], [cx + hx, cy - hy, cz + hz],
                [cx + hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz - hz],
                [cx - hx, cy + hy, cz + hz], [cx - hx, cy - hy, cz + hz],
                [cx - hx, cy - hy, cz - hz], [cx - hx, cy + hy, cz - hz],
            ]
            flat = [float(v) for p in corner_pts for v in p]
        box_rows.append(flat)
        labels.append(str(getattr(box, "label", "") or ""))
    box_arr = np.asarray(box_rows, dtype=np.float32) if box_rows else np.zeros((0, 24), dtype=np.float32)
    label_blob = json.dumps(labels, ensure_ascii=True).encode("utf-8")
    box_count = int(box_arr.shape[0])

    header = struct.pack(
        VIEWER_FRAME_HEADER_FMT,
        VIEWER_FRAME_MAGIC,
        VIEWER_FRAME_HEADER_LEN,
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

    class TlrFrameAnnotationOut(BaseModel):
        x0: float
        y0: float
        x1: float
        y1: float
        label: str = ""
        instance_token: str = ""
        category_token: str = ""
        automatic_annotation: bool = False

    class TlrFrameViewOut(BaseModel):
        camera: str
        sample_token: str
        sample_data_token: str
        timestamp_us: int
        image_width: int
        image_height: int
        image_jpeg_base64: str
        annotation_count: int = 0
        annotations: List[TlrFrameAnnotationOut] = _Field(default_factory=list)

    class TlrFrameResponse(BaseModel):
        t4dataset_id: str
        scenario_name: str
        frame_index: int
        total_frames: int
        logical_frame_index: int = 0
        total_logical_frames: int = 0
        frame_mode: str = "sample"
        sample_token: str
        timestamp_us: int
        camera: str
        available_cameras: List[str] = _Field(default_factory=list)
        image_width: int
        image_height: int
        image_jpeg_base64: str
        annotation_count: int = 0
        annotations: List[TlrFrameAnnotationOut] = _Field(default_factory=list)
        views: List[TlrFrameViewOut] = _Field(default_factory=list)

    class CameraOverlayExtrasBody(BaseModel):
        """Optional GT/EST boxes (same schema as viewer ``bbox_layers`` postMessage) to project onto the image."""

        pred: List[Dict[str, Any]] = _Field(default_factory=list)
        gt: List[Dict[str, Any]] = _Field(default_factory=list)

    class ViewerSessionSaveBody(BaseModel):
        """Persisted viewer overlay payload used by upload/share links."""

        payload: Dict[str, Any] = _Field(default_factory=dict)
        source_name: Optional[str] = None
        t4dataset_id: Optional[str] = None
        scenario_name: Optional[str] = None
        frame_index: int = 0
        version: Optional[str] = None

except ImportError:
    pass  # Proper error is raised inside _build_app when fastapi is missing


# ---------------------------------------------------------------------------

def _build_app(
    data_dirs: List[Path],
    search_depth: int,
    tier4_cache_size: int,
    dataset_path_cache_ttl_s: float = 30.0,
    visibility_mode: str = "public",
):
    """Construct and return the FastAPI application."""
    # For backward compatibility, also expose the first data_dir as data_dir
    data_dir = data_dirs[0] if data_dirs else Path("./t4datasets")
    try:
        from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
        from fastapi.encoders import jsonable_encoder
        from fastapi.responses import HTMLResponse, JSONResponse, Response
    except ImportError as exc:
        raise ImportError(
            "fastapi and pydantic are required for the server. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    from t4_visualizer.batch import find_dataset_in_dirs
    from t4_visualizer.downloader import list_webauto_annotation_dataset_ids_multi
    from t4_visualizer.visualize import (
        TargetObject,
        VisualizationRequest,
        render_frame,
    )

    app = FastAPI(title="T4 Visualizer", version="0.1.0")
    _static_dir = Path(__file__).resolve().parent / "static"
    if _static_dir.is_dir():
        from starlette.staticfiles import StaticFiles

        app.mount(
            "/static",
            StaticFiles(directory=str(_static_dir)),
            name="static",
        )
    _repo_root = Path(__file__).resolve().parent.parent
    _vehicle_mesh_dir = _repo_root / "assets" / "sample_vehicle_description" / "mesh"
    if _vehicle_mesh_dir.is_dir():
        from starlette.staticfiles import StaticFiles

        app.mount(
            "/viewer/assets/vehicle-mesh",
            StaticFiles(directory=str(_vehicle_mesh_dir)),
            name="vehicle_mesh",
        )

    # Drop-in location for an optional (confidential, never-committed) ego vehicle
    # mesh. Install one by copying it here under exactly this name; nothing else
    # needs configuring. Absent, the viewer uses the bundled sample mesh.
    _CUSTOM_VEHICLE_MESH_PATH = (
        _repo_root / "assets" / "custom_vehicle_description" / "mesh" / "ego.dae"
    )
    # Served under a fixed URL so the deployment never reveals the real model name.
    _CUSTOM_VEHICLE_MESH_URL = "/viewer/assets/vehicle-mesh-custom/ego.dae"

    _custom_vehicle_mesh = (
        _CUSTOM_VEHICLE_MESH_PATH if _CUSTOM_VEHICLE_MESH_PATH.is_file() else None
    )
    if _custom_vehicle_mesh is not None:
        logger.info("Using custom ego vehicle mesh: %s", _custom_vehicle_mesh)

        @app.get(
            _CUSTOM_VEHICLE_MESH_URL,
            tags=["viewer"],
            summary="Custom ego vehicle mesh (Collada)",
            include_in_schema=False,
        )
        def viewer_custom_vehicle_mesh():
            """Serve the installed custom ego mesh under a name-neutral URL.

            Registered only when a custom mesh is present, so the route 404s
            otherwise and the viewer falls back to the bundled sample mesh.
            """
            from starlette.responses import FileResponse

            return FileResponse(
                str(_custom_vehicle_mesh),
                media_type="model/vnd.collada+xml",
            )

    @app.get(
        "/viewer/assets/vehicle-model.json",
        tags=["viewer"],
        summary="Ego vehicle mesh descriptor (custom mesh if installed, else bundled sample)",
    )
    def viewer_vehicle_model():
        """Describe the ego mesh the viewer should load, plus its base_link transform.

        ``rotation`` / ``offset`` place the mesh so that its origin coincides with
        ``base_link`` (rear axle centre, +X forward, Z up).
        """
        if _custom_vehicle_mesh is not None:
            return {
                "source": "custom",
                "url": _CUSTOM_VEHICLE_MESH_URL,
                # Authored Z-up with the origin already at base_link, facing +X.
                "rotation": [0.0, 0.0, 0.0],
                "offset": [0.0, 0.0, 0.0],
            }
        return {
            "source": "sample",
            "url": "/viewer/assets/vehicle-mesh/lexus.dae",
            # Y-up mesh authored facing -X, with its origin at the vehicle centre.
            "rotation": [-math.pi / 2, 0.0, math.pi],
            "offset": [2.79 * 0.5, 0.0, 0.0],
        }
    _cache = _Tier4Cache(max_size=tier4_cache_size)
    _path_cache = _DatasetPathCache(ttl_s=dataset_path_cache_ttl_s)
    _debug_visibility = str(visibility_mode).strip().lower() == "debug"
    _templates_dir = Path(__file__).resolve().parent / "templates"
    _viewer_session_dir: Optional[Path] = None
    _viewer_session_candidates = []
    env_session_dir = os.environ.get("T4_VIEWER_SESSION_DIR", "").strip()
    if env_session_dir:
        _viewer_session_candidates.append(Path(env_session_dir).expanduser())
    for dd in data_dirs:
        _viewer_session_candidates.append(dd / ".viewer_sessions")
    _viewer_session_candidates.append(Path("/tmp/t4_viewer_sessions"))
    for cand in _viewer_session_candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            _viewer_session_dir = cand
            break
        except OSError:
            continue

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

    # Session store hygiene: cap individual payloads and garbage-collect old
    # files so an unauthenticated client cannot fill the disk over time.
    _VIEWER_SESSION_MAX_BYTES = 32 * 1024 * 1024
    _VIEWER_SESSION_TTL_S = 30 * 24 * 3600
    _VIEWER_SESSION_MAX_FILES = 500

    def _prune_viewer_sessions() -> None:
        """Best-effort GC of the session dir: drop expired files, keep the newest N."""
        if _viewer_session_dir is None:
            return
        try:
            files = sorted(
                (p for p in _viewer_session_dir.glob("*.json")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return
        now = time.time()
        for idx, p in enumerate(files):
            try:
                if idx >= _VIEWER_SESSION_MAX_FILES or now - p.stat().st_mtime > _VIEWER_SESSION_TTL_S:
                    p.unlink()
            except OSError:
                continue

    def _normalize_viewer_session_id(raw: str) -> str:
        try:
            return str(uuid.UUID(str(raw)))
        except (ValueError, AttributeError, TypeError):
            _public_error(400, "invalid_session_id", "Viewer session id must be a valid UUID.")

    def _viewer_session_file(session_id: str) -> Path:
        if _viewer_session_dir is None:
            _safe_error(
                500,
                "viewer_session_store_unavailable",
                "Viewer session storage is unavailable on this server.",
            )
        return _viewer_session_dir / f"{_normalize_viewer_session_id(session_id)}.json"

    def _validate_viewer_session_payload(payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            _public_error(400, "invalid_viewer_payload", "Viewer session payload must be a JSON object.")
        has_any = False
        if "bbox_layers_by_frame" in payload:
            if payload["bbox_layers_by_frame"] is not None and not isinstance(payload["bbox_layers_by_frame"], dict):
                _public_error(
                    400,
                    "invalid_viewer_payload",
                    "'bbox_layers_by_frame' must be an object keyed by frame index.",
                )
            has_any = True
        if "bbox_layers" in payload:
            if payload["bbox_layers"] is not None and not isinstance(payload["bbox_layers"], dict):
                _public_error(400, "invalid_viewer_payload", "'bbox_layers' must be an object.")
            has_any = True
        if "eval_metrics_series" in payload:
            if payload["eval_metrics_series"] is not None and not isinstance(payload["eval_metrics_series"], dict):
                _public_error(400, "invalid_viewer_payload", "'eval_metrics_series' must be an object.")
            has_any = True
        if not has_any:
            _public_error(
                400,
                "invalid_viewer_payload",
                "Viewer session payload must include bbox_layers_by_frame, bbox_layers, or eval_metrics_series.",
            )
        return jsonable_encoder(payload)

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
        logger.info(
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
        # A dataset id is a single directory name; path separators or parent
        # references would let a client resolve paths outside --data-dir.
        if (
            not t4dataset_id
            or t4dataset_id in (".", "..")
            or "/" in t4dataset_id
            or "\\" in t4dataset_id
        ):
            _public_error(
                status_code=400,
                code="invalid_dataset_id",
                message=f"Dataset id '{t4dataset_id}' is not a valid dataset name.",
                hint="Dataset ids must be a bare directory name without path separators.",
            )
        path = _path_cache.resolve(
            data_dirs,
            search_depth,
            t4dataset_id,
            lambda: find_dataset_in_dirs(data_dirs, t4dataset_id, search_depth),
        )
        if path is not None:
            # Lexical containment check (no symlink resolution: data dirs may
            # legitimately symlink to a shared dataset mount).
            normalized = Path(os.path.normpath(path))
            if not any(
                normalized.is_relative_to(Path(os.path.normpath(d))) for d in data_dirs
            ):
                path = None
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
            exc_msg = str(exc)
            if "rosbag" in exc_msg.lower() or "invalid" in exc_msg.lower() or "not a t4" in exc_msg.lower():
                hint = f"The path '{dataset_path}' exists but may not be a valid T4 dataset."
            else:
                hint = f"Dataset path: {dataset_path}"
            _safe_error(
                status_code=500,
                code="render_failed",
                message=f"Failed to render frame: {exc_msg}",
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

    def _record_get(record: Any, key: str, default: Any = None) -> Any:
        if isinstance(record, dict):
            return record.get(key, default)
        return getattr(record, key, default)

    def _category_name_map(t4) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for row in list(getattr(t4, "category", []) or []):
            token = str(_record_get(row, "token", "") or "")
            if not token:
                continue
            name = str(_record_get(row, "name", "") or "")
            out[token] = name
        return out

    def _preferred_tlr_camera(available_cameras: List[str], requested: Optional[str]) -> str:
        if requested:
            wanted = str(requested).strip()
            if wanted in available_cameras:
                return wanted
            _public_error(
                404,
                "camera_not_found",
                f"Camera '{wanted}' was not found in this sample.",
                hint=f"Available cameras: {', '.join(available_cameras)}",
            )
        ranked = sorted(
            available_cameras,
            key=lambda name: (
                0 if "TRAFFIC_LIGHT" in str(name).upper() else 1,
                str(name),
            ),
        )
        return ranked[0]

    def _scenario_sample_rows(t4, scenario_name: str) -> List[Any]:
        scenes = list(getattr(t4, "scene", []) or [])
        scene_row = next(
            (row for row in scenes if str(_record_get(row, "name", "") or "") == scenario_name),
            None,
        )
        if scene_row is None:
            _public_error(
                404,
                "scenario_not_found",
                f"Scenario '{scenario_name}' was not found.",
                hint="Call GET /datasets/{id}/scenarios to inspect valid scenario names.",
            )

        ordered: List[Any] = []
        seen: set[str] = set()
        cur_token = str(_record_get(scene_row, "first_sample_token", "") or "")
        while cur_token:
            if cur_token in seen:
                break
            try:
                row = t4.get("sample", cur_token)
            except Exception:
                row = None
            if row is None:
                break
            ordered.append(row)
            seen.add(cur_token)
            cur_token = str(_record_get(row, "next", "") or "")
        if ordered:
            return ordered

        samples = list(getattr(t4, "sample", []) or [])
        scene_token = str(_record_get(scene_row, "token", "") or "")
        fallback = [
            row
            for row in samples
            if str(_record_get(row, "scene_token", "") or "") == scene_token
        ]
        fallback.sort(key=lambda row: int(_record_get(row, "timestamp", 0) or 0))
        out: List[Any] = []
        for row in fallback:
            token = str(_record_get(row, "token", "") or "")
            if not token:
                continue
            try:
                sample = t4.get("sample", token)
            except Exception:
                sample = row
            out.append(sample)
        return out

    def _sample_camera_names(t4, sample) -> List[str]:
        from t4_visualizer.visualize import list_camera_channels

        try:
            return list_camera_channels(t4, sample)
        except Exception:
            return []

    def _sample_primary_camera(t4, sample) -> str:
        channels = _sample_camera_names(t4, sample)
        if not channels:
            return ""
        return channels[0]

    def _tlr_logical_frames(t4, scenario_name: str) -> List[List[Any]]:
        ordered_samples = _scenario_sample_rows(t4, scenario_name)
        if not ordered_samples:
            return []

        primary_channels = [_sample_primary_camera(t4, sample) for sample in ordered_samples]
        valid_channels = [ch for ch in primary_channels if ch]
        unique_channels = sorted(set(valid_channels))
        all_single_camera = all(len(_sample_camera_names(t4, sample)) == 1 for sample in ordered_samples)
        all_traffic_light = bool(valid_channels) and all("TRAFFIC_LIGHT" in ch.upper() for ch in valid_channels)

        if not (all_single_camera and all_traffic_light and len(unique_channels) >= 2):
            return [[sample] for sample in ordered_samples]

        by_camera: Dict[str, List[Any]] = {cam: [] for cam in unique_channels}
        for sample, cam in zip(ordered_samples, primary_channels):
            if cam:
                by_camera.setdefault(cam, []).append(sample)

        lengths = [len(rows) for rows in by_camera.values() if rows]
        if not lengths:
            return [[sample] for sample in ordered_samples]

        logical_count = min(lengths)
        pair_gap_us = 50_000
        logical_frames: List[List[Any]] = []
        for idx in range(logical_count):
            frame_samples: List[Any] = []
            timestamps: List[int] = []
            for cam in unique_channels:
                rows = by_camera.get(cam) or []
                if idx >= len(rows):
                    continue
                sample = rows[idx]
                frame_samples.append(sample)
                timestamps.append(int(_record_get(sample, "timestamp", 0) or 0))
            if not frame_samples:
                continue
            if timestamps and (max(timestamps) - min(timestamps) > pair_gap_us):
                # If camera streams drift too much, fall back to raw per-sample frames.
                return [[sample] for sample in ordered_samples]
            frame_samples.sort(
                key=lambda sample: (
                    0 if "FAR" in _sample_primary_camera(t4, sample).upper() else 1,
                    _sample_primary_camera(t4, sample),
                    int(_record_get(sample, "timestamp", 0) or 0),
                )
            )
            logical_frames.append(frame_samples)

        if logical_frames and sum(len(frame) for frame in logical_frames) >= len(ordered_samples):
            return logical_frames

        return logical_frames or [[sample] for sample in ordered_samples]

    def _tlr_annotations_for_sample_data(t4, sample_data_token: str, boxes_2d) -> List[TlrFrameAnnotationOut]:
        category_names = _category_name_map(t4)
        annotations: List[TlrFrameAnnotationOut] = []
        object_ann_rows = list(getattr(t4, "object_ann", []) or [])
        for row in object_ann_rows:
            if str(_record_get(row, "sample_data_token", "") or "") != str(sample_data_token):
                continue
            bbox = _record_get(row, "bbox", None)
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in bbox]
            except (TypeError, ValueError):
                continue
            category_token = str(_record_get(row, "category_token", "") or "")
            label = category_names.get(category_token, category_token)
            annotations.append(
                TlrFrameAnnotationOut(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    label=label,
                    instance_token=str(_record_get(row, "instance_token", "") or ""),
                    category_token=category_token,
                    automatic_annotation=bool(_record_get(row, "automatic_annotation", False)),
                )
            )

        if annotations:
            return annotations

        for idx, box in enumerate(list(boxes_2d or [])):
            roi = getattr(box, "roi", None)
            if not isinstance(roi, (list, tuple)) or len(roi) != 4:
                continue
            try:
                x0, y0, x1, y1 = [float(v) for v in roi]
            except (TypeError, ValueError):
                continue
            annotations.append(
                TlrFrameAnnotationOut(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    label=str(getattr(box, "label", "") or ""),
                    instance_token=str(idx),
                    category_token="",
                    automatic_annotation=False,
                )
            )
        return annotations

    def _build_tlr_view(t4, sample, camera_name: str) -> TlrFrameViewOut:
        from PIL import Image

        sample_data = _record_get(sample, "data", {}) or {}
        sample_data_token = sample_data.get(camera_name)
        if sample_data_token is None:
            _public_error(
                404,
                "sample_data_not_found",
                f"Camera '{camera_name}' is not available in this frame.",
            )

        try:
            data_path, boxes_2d, _ = t4.get_sample_data(
                sample_data_token,
                as_3d=False,
                as_sensor_coord=True,
            )
        except Exception as exc:
            _safe_error(
                500,
                "tlr_frame_load_failed",
                "Failed to load TLR camera frame.",
                exc=exc,
            )

        annotations = _tlr_annotations_for_sample_data(t4, str(sample_data_token), boxes_2d)

        try:
            with Image.open(data_path) as img:
                width, height = img.size
            image_bytes = Path(data_path).read_bytes()
        except Exception as exc:
            _safe_error(
                500,
                "tlr_image_read_failed",
                "Failed to read TLR camera image.",
                exc=exc,
            )

        return TlrFrameViewOut(
            camera=camera_name,
            sample_token=str(_record_get(sample, "token", "") or ""),
            sample_data_token=str(sample_data_token),
            timestamp_us=int(_record_get(sample, "timestamp", 0) or 0),
            image_width=int(width),
            image_height=int(height),
            image_jpeg_base64=base64.b64encode(image_bytes).decode(),
            annotation_count=len(annotations),
            annotations=annotations,
        )

    def _dataset_profile_payload(
        t4dataset_id: str,
        dataset_path: Path,
        t4,
        version: Optional[str],
    ) -> Dict[str, object]:
        from t4_visualizer.visualize import list_camera_channels, list_lidar_channels

        scenes = list(getattr(t4, "scene", []) or [])
        samples = list(getattr(t4, "sample", []) or [])
        sample_annotation_count = len(list(getattr(t4, "sample_annotation", []) or []))
        object_ann_count = len(list(getattr(t4, "object_ann", []) or []))

        first_sample = samples[0] if samples else None
        camera_channels = list_camera_channels(t4, first_sample) if first_sample is not None else []
        lidar_channels = list_lidar_channels(t4, first_sample) if first_sample is not None else []
        scene_descriptions = [
            str(_record_get(scene, "description", "") or "")
            for scene in scenes
            if str(_record_get(scene, "description", "") or "")
        ]
        category_names = sorted(
            {
                str(_record_get(cat, "name", "") or "")
                for cat in list(getattr(t4, "category", []) or [])
                if str(_record_get(cat, "name", "") or "")
            }
        )
        scene_text = " ".join(scene_descriptions).upper()
        has_traffic_light_camera = any("TRAFFIC_LIGHT" in str(ch).upper() for ch in camera_channels)
        is_tlr = (
            object_ann_count > 0
            and (has_traffic_light_camera or "TLR" in scene_text or "TRAFFIC LIGHT" in scene_text)
            and sample_annotation_count == 0
        )
        kind = "tlr" if is_tlr else "standard"
        preferred_viewer = "tlr" if is_tlr else ("three" if lidar_channels else "render")

        return {
            "t4dataset_id": t4dataset_id,
            "available": True,
            "dataset_path": str(dataset_path.resolve()) if _debug_visibility else None,
            "version": version,
            "kind": kind,
            "preferred_viewer": preferred_viewer,
            "scene_count": len(scenes),
            "sample_count": len(samples),
            "camera_channels": camera_channels,
            "lidar_channels": lidar_channels,
            "has_lidar": bool(lidar_channels),
            "sample_annotation_count": sample_annotation_count,
            "object_ann_count": object_ann_count,
            "has_2d_annotations": object_ann_count > 0,
            "has_3d_annotations": sample_annotation_count > 0,
            "scene_descriptions": scene_descriptions,
            "categories": category_names,
            "supports": {
                "render": bool(camera_channels),
                "three": bool(lidar_channels),
                "tlr": is_tlr and bool(camera_channels),
            },
        }

    def _tlr_frame_payload(
        t4dataset_id: str,
        dataset_path: Path,
        t4,
        scenario_name: Optional[str],
        frame_index: int,
        camera: Optional[str],
        version: Optional[str],
    ) -> TlrFrameResponse:
        from t4_visualizer.visualize import list_scene_summaries

        resolved_scenario = _resolve_viewer_scenario_name(t4, scenario_name)
        logical_frames = _tlr_logical_frames(t4, resolved_scenario)
        if not logical_frames:
            _public_error(
                404,
                "frame_index_out_of_range",
                "No TLR frames were found for this scenario.",
            )
        if frame_index < 0 or frame_index >= len(logical_frames):
            _public_error(
                400,
                "frame_index_out_of_range",
                f"Frame index {frame_index} is out of range (0..{len(logical_frames) - 1}).",
            )
        samples_for_frame = logical_frames[frame_index]
        views: List[TlrFrameViewOut] = []
        available_cameras: List[str] = []
        for sample in samples_for_frame:
            for cam_name in _sample_camera_names(t4, sample):
                if cam_name not in available_cameras:
                    available_cameras.append(cam_name)
                views.append(_build_tlr_view(t4, sample, cam_name))
        if not available_cameras or not views:
            _public_error(
                404,
                "camera_not_found",
                "No camera channels are available for this frame.",
            )
        selected_camera = _preferred_tlr_camera(available_cameras, camera)
        primary_view = next((view for view in views if view.camera == selected_camera), views[0])

        scenes = list_scene_summaries(t4)
        scene_meta = next((s for s in scenes if s.get("name") == resolved_scenario), None)
        raw_total_frames = int(scene_meta.get("nbr_samples") or 0) if scene_meta else len(logical_frames)

        return TlrFrameResponse(
            t4dataset_id=t4dataset_id,
            scenario_name=resolved_scenario,
            frame_index=frame_index,
            total_frames=len(logical_frames),
            logical_frame_index=frame_index,
            total_logical_frames=len(logical_frames),
            frame_mode="paired" if len(samples_for_frame) > 1 else ("logical" if len(logical_frames) != raw_total_frames else "sample"),
            sample_token=primary_view.sample_token,
            timestamp_us=primary_view.timestamp_us,
            camera=primary_view.camera,
            available_cameras=available_cameras,
            image_width=primary_view.image_width,
            image_height=primary_view.image_height,
            image_jpeg_base64=primary_view.image_jpeg_base64,
            annotation_count=primary_view.annotation_count,
            annotations=primary_view.annotations,
            views=views,
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

    def _safe_float_or_none(v: Any) -> Optional[float]:
        try:
            out = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(out):
            return None
        return out

    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def _derive_eval_box_severity(box: Dict[str, Any], kind: str, status: str) -> Tuple[float, str]:
        preset = _safe_float_or_none(box.get("severity_score"))
        if preset is not None:
            return _clip01(preset), str(box.get("severity_reason", box.get("severity_label", "provided"))) or "provided"

        st = str(status or "TP").upper()
        if st == "FN":
            return 0.96, "missed ground truth"
        if st == "FP":
            conf = _safe_float_or_none(box.get("confidence"))
            base = 0.66 + 0.18 * (conf if conf is not None else 0.35)
            return _clip01(base), "false positive"

        components: List[Tuple[str, float]] = []
        x_err = _safe_float_or_none(box.get("x_error"))
        y_err = _safe_float_or_none(box.get("y_error"))
        z_err = _safe_float_or_none(box.get("z_error"))
        yaw_err = _safe_float_or_none(box.get("yaw_error"))
        center_d = _safe_float_or_none(box.get("center_distance"))
        plane_d = _safe_float_or_none(box.get("plane_distance"))
        dt_sec = _safe_float_or_none(box.get("pair_dt_sec"))
        if x_err is not None or y_err is not None:
            xy_mag = math.hypot(x_err or 0.0, y_err or 0.0)
            components.append(("xy error", min(1.0, xy_mag / 1.8) * 0.34))
        if z_err is not None:
            components.append(("z error", min(1.0, abs(z_err) / 1.0) * 0.16))
        if yaw_err is not None:
            components.append(("yaw error", min(1.0, abs(yaw_err) / 0.8) * 0.24))
        if center_d is not None:
            components.append(("center distance", min(1.0, center_d / 1.8) * 0.16))
        if plane_d is not None:
            components.append(("plane distance", min(1.0, plane_d / 1.8) * 0.12))
        if dt_sec is not None:
            components.append(("pair dt", min(1.0, abs(dt_sec) / 0.1) * 0.08))
        if not components:
            return 0.08 if kind == "GT" else 0.12, "low TP error"
        components.sort(key=lambda t: t[1], reverse=True)
        score = 0.1 + sum(v for _k, v in components)
        return _clip01(score), components[0][0]

    def _camera_overlay_eval_row(box: Dict[str, Any], *, kind: str, status: str, roi, layer_name: str) -> Dict[str, object]:
        u_min, v_min, u_max, v_max, _vis = roi
        sev, sev_reason = _derive_eval_box_severity(box, kind, status)
        row: Dict[str, object] = {
            "x0": u_min,
            "y0": v_min,
            "x1": u_max,
            "y1": v_max,
            "label": str(box.get("label", box.get("class", "")) or ""),
            "status": status,
            "kind": kind,
            "layer": layer_name,
            "uuid": str(box.get("uuid", box.get("id", box.get("track_id", box.get("object_id", "")))) or ""),
            "pair_uuid": str(box.get("pair_uuid", "") or ""),
            "severity_score": sev,
            "severity_reason": sev_reason,
        }
        passthrough_fields = [
            "confidence",
            "center_distance",
            "plane_distance",
            "pair_dt_sec",
            "x_error",
            "y_error",
            "z_error",
            "yaw_error",
            "vx",
            "vy",
            "frame_index",
            "unix_time",
            "frame_id",
            "topic_name",
            "run",
            "suite_name",
            "scenario_name",
            "t4dataset_id",
            "t4dataset_name",
            "source",
        ]
        for key in passthrough_fields:
            if key in box and box.get(key) is not None:
                row[key] = box.get(key)
        return row

    def _external_eval_boxes_to_2d_rows(
        t4,
        camera_token: str,
        boxes: Optional[List[Dict[str, Any]]],
        layer_name: str,
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
                status = str(b.get("status", "TP") or "TP").upper()
                kind = "EST" if layer_name == "pred" else "GT"
                rows.append(_camera_overlay_eval_row(b, kind=kind, status=status, roi=roi, layer_name=layer_name))
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
                "pred",
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
                "gt",
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
        def _as_float(value: object, default: float = 0.0) -> float:
            try:
                if value is None:
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        map_path = dataset_path / "map" / "lanelet2_map.osm"
        if not map_path.exists():
            return {"available": False, "reason": f"map file not found: {map_path}", "segments": []}
        try:
            nodes_latlon_ele, nodes_localxy_ele, ways, lanelet_way_ids, role_of_way = _load_lanelet_graph(str(map_path))
        except Exception as exc:
            _safe_error(500, "lanelet_parse_failed", "Failed to parse lanelet2_map.osm.", exc=exc)
        if not nodes_latlon_ele:
            return {
                "available": False,
                "reason": "lanelet map has no usable nodes with lat/lon coordinates",
                "segments": [],
            }

        sample = _get_scenario_sample(t4, scenario_name, frame_index)
        from t4_visualizer.visualize import list_lidar_channels

        lidar_channels = list_lidar_channels(t4, sample)
        if not lidar_channels:
            return {"available": False, "reason": "no lidar channel in sample", "segments": []}
        lidar_token = sample.data.get(lidar_channels[0])
        if lidar_token is None:
            return {"available": False, "reason": "no lidar sample_data token", "segments": []}
        sample_data = t4.get("sample_data", lidar_token)
        ego_pose_token = getattr(sample_data, "ego_pose_token", None)
        if not ego_pose_token:
            return {"available": False, "reason": "lidar sample_data missing ego_pose_token", "segments": []}
        ego_pose = t4.get("ego_pose", ego_pose_token)
        geocoord = getattr(ego_pose, "geocoordinate", None)
        ego_t = getattr(ego_pose, "translation", None)
        if ego_t is None:
            ego_t = [0.0, 0.0, 0.0]
        ego_tx = _as_float(ego_t[0]) if len(ego_t) >= 1 else 0.0
        ego_ty = _as_float(ego_t[1]) if len(ego_t) >= 2 else 0.0
        ego_tz = _as_float(ego_t[2]) if len(ego_t) >= 3 else 0.0
        lat0 = None
        lon0 = None
        alt0 = ego_tz
        align_mode = "translation_xy"
        if geocoord and len(geocoord) >= 2:
            lat_candidate = _as_float(geocoord[0], default=float("nan"))
            lon_candidate = _as_float(geocoord[1], default=float("nan"))
            if math.isfinite(lat_candidate) and math.isfinite(lon_candidate):
                lat0 = lat_candidate
                lon0 = lon_candidate
                alt0 = _as_float(geocoord[2], default=ego_tz) if len(geocoord) >= 3 else ego_tz
                align_mode = "geocoordinate"
        rot = getattr(ego_pose, "rotation", None)
        if rot is None:
            rot = [1.0, 0.0, 0.0, 0.0]
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
            '<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">'
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

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        favicon_path = _static_dir / "favicon.svg"
        if not favicon_path.is_file():
            _public_error(404, "favicon_not_found", "Favicon asset is not available.")
        return Response(
            content=favicon_path.read_text(encoding="utf-8"),
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/datasets/browser")
    def datasets_browser_page():
        """Interactive read-only browser for datasets and scenarios."""
        page = _load_template("datasets_browser.html")
        return HTMLResponse(content=page, media_type="text/html; charset=utf-8")

    @app.get("/")
    def index_page():
        """Human-friendly landing page for quick server introspection."""
        esc = html.escape
        # Collect top-level dirs from all data directories
        top_level_dirs_all = []
        for dd in data_dirs:
            try:
                top_level_dirs_all.extend(
                    p.name for p in dd.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                )
            except OSError:
                pass
        top_level_dirs = sorted(set(top_level_dirs_all))[:12]
        try:
            dataset_ids = list_webauto_annotation_dataset_ids_multi(data_dirs)
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
        data_dirs_html = "<br>".join(f"<code>{esc(str(dd))}</code>" for dd in data_dirs)
        page = (
            "<!DOCTYPE html>"
            '<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">'
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
            f"<div><strong>data_dirs</strong><br>{data_dirs_html}</div>"
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
        """Return dataset IDs plus discovery diagnostics under the configured data_dirs."""
        resolved = [str(dd.resolve()) for dd in data_dirs]
        # Collect top-level dirs from all data directories
        all_top_level_dirs = set()
        for dd in data_dirs:
            try:
                all_top_level_dirs.update(
                    p.name for p in dd.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                )
            except OSError:
                pass
        all_top_level_dirs_list = sorted(all_top_level_dirs)
        top_level_dirs = all_top_level_dirs_list[:32]
        top_level_dir_count = len(all_top_level_dirs_list)
        # Check if any data_dir has annotation_dataset
        ann_present = any((dd / "annotation_dataset").is_dir() for dd in data_dirs)
        # Check if any data_dir exists
        any_exists = any(dd.exists() for dd in data_dirs)
        if not any_exists:
            out: Dict[str, object] = {
                "data_dirs": [str(dd) for dd in data_dirs],
                "data_dirs_resolved": resolved if _debug_visibility else ["(redacted)"] * len(data_dirs),
                "datasets": [],
                "top_level_dirs": [],
                "top_level_dir_count": 0,
                "annotation_dataset_present": False,
                "hint": "None of the data_dirs exist on this host — check --data-dir.",
            }
            return out

        try:
            uniq = list_webauto_annotation_dataset_ids_multi(data_dirs)
        except OSError:
            uniq = []
        payload: Dict[str, object] = {
            "data_dirs": [str(dd) for dd in data_dirs],
            "data_dirs_resolved": resolved if _debug_visibility else ["(redacted)"] * len(data_dirs),
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
                    "No subdirectories under any data_dir — confirm --data-dir on this host."
                )
            elif not ann_present and len(top_level_dirs) <= 8:
                payload["hint"] = (
                    "No T4 datasets listed. Data is often under a grouped folder "
                    "(e.g. annotation_dataset or a project folder like j6gen6_3) "
                    "as <group>/<uuid>/<version>/."
                )
            else:
                payload["hint"] = (
                    "No UUID-shaped folder names found under data_dirs or "
                    "one level inside grouped folders (see top_level_dirs)."
                )
        return payload

    @app.get(
        "/datasets/{t4dataset_id}/availability",
        response_model=DatasetAvailabilityResponse,
        tags=["datasets"],
        summary="Check whether a dataset id is available under data_dirs",
    )
    def dataset_availability(t4dataset_id: str):
        """Return whether *t4dataset_id* resolves under any of the configured ``data_dirs``.

        Uses the same lookup as ``POST /render`` and ``GET /datasets/.../scenarios``
        (:func:`t4_visualizer.batch.find_dataset_in_dirs`). Does not load Tier4.
        Results are cached briefly (see ``--dataset-path-cache-ttl``).
        """
        found = _path_cache.resolve(
            data_dirs,
            search_depth,
            t4dataset_id,
            lambda: find_dataset_in_dirs(data_dirs, t4dataset_id, search_depth),
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
        "/datasets/{t4dataset_id}/profile",
        tags=["datasets"],
        summary="Inspect dataset modality and viewer capabilities",
    )
    def dataset_profile(
        t4dataset_id: str,
        version: Optional[str] = None,
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        try:
            t4 = _cache.load(dataset_path, version=version)
            return _dataset_profile_payload(t4dataset_id, dataset_path, t4, version)
        except HTTPException:
            raise
        except Exception as exc:
            exc_msg = str(exc)
            if "rosbag" in exc_msg.lower() or "invalid" in exc_msg.lower() or "not a t4" in exc_msg.lower():
                hint = f"The path '{dataset_path}' exists but may not be a valid T4 dataset."
            else:
                hint = f"Dataset path: {dataset_path}"
            _safe_error(
                500,
                "dataset_profile_failed",
                message=f"Failed to inspect dataset profile: {exc_msg}",
                exc=exc,
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
            # Check if this is not a valid T4 dataset (e.g., rosbag folder)
            exc_msg = str(exc)
            if "rosbag" in exc_msg.lower() or "invalid" in exc_msg.lower() or "not a t4" in exc_msg.lower():
                hint = f"The path '{dataset_path}' exists but may not be a valid T4 dataset."
            else:
                hint = f"Dataset path: {dataset_path}"
            _safe_error(
                status_code=500,
                code="scenarios_list_failed",
                message=f"Failed to load scenarios: {exc_msg}",
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
            "note": (
                "tier4_cache shows datasets that were successfully loaded. "
                "If a dataset fails to load, it will not appear here. "
                "Use GET /datasets/{id}/availability to check if a dataset path exists."
            ),
        }
        if _debug_visibility:
            out["runtime"]["data_dirs"] = [str(dd) for dd in data_dirs]
            out["runtime"]["data_dirs_resolved"] = [str(dd.resolve()) for dd in data_dirs]
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
            exc_msg = str(exc)
            if "rosbag" in exc_msg.lower() or "invalid" in exc_msg.lower() or "not a t4" in exc_msg.lower():
                hint = f"The path '{dataset_path}' exists but may not be a valid T4 dataset."
            else:
                hint = f"Dataset path: {dataset_path}"
            _safe_error(
                status_code=500,
                code="frame_summary_failed",
                message=f"Failed to load frame summary: {exc_msg}",
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
        "/viewer/three/session/{session_id}",
        tags=["viewer"],
        summary="Load a persisted viewer overlay session by UUID",
    )
    def viewer_three_session_get(session_id: str):
        path = _viewer_session_file(session_id)
        try:
            if not path.is_file():
                _public_error(404, "viewer_session_not_found", f"Viewer session '{session_id}' was not found.")
            return json.loads(path.read_text(encoding="utf-8"))
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                500,
                "viewer_session_load_failed",
                "Failed to load the requested viewer session.",
                exc=exc,
            )

    @app.post(
        "/viewer/three/session",
        tags=["viewer"],
        summary="Persist viewer overlay payload and return a shareable UUID link",
    )
    def viewer_three_session_create(body: ViewerSessionSaveBody):
        payload = _validate_viewer_session_payload(body.payload)
        session_id = str(uuid.uuid4())
        path = _viewer_session_file(session_id)
        record = {
            "session_id": session_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_name": str(body.source_name).strip() if body.source_name else None,
            "viewer_context": {
                "t4dataset_id": body.t4dataset_id,
                "scenario_name": body.scenario_name,
                "frame_index": int(body.frame_index or 0),
                "version": body.version,
            },
            "payload": payload,
        }
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > _VIEWER_SESSION_MAX_BYTES:
            _public_error(
                413,
                "viewer_session_too_large",
                f"Viewer session payload exceeds {_VIEWER_SESSION_MAX_BYTES // (1024 * 1024)} MB.",
                hint="Trim bbox_layers_by_frame or share fewer frames per session.",
            )
        _prune_viewer_sessions()
        try:
            path.write_text(encoded, encoding="utf-8")
        except Exception as exc:
            _safe_error(
                500,
                "viewer_session_save_failed",
                "Failed to persist the viewer session payload.",
                exc=exc,
            )

        share_url = None
        if body.t4dataset_id:
            share_params = {
                "t4dataset_id": body.t4dataset_id,
                "frame_index": int(body.frame_index or 0),
                "session_id": session_id,
            }
            if body.scenario_name:
                share_params["scenario_name"] = body.scenario_name
            if body.version:
                share_params["version"] = body.version
            share_url = f"/viewer/three?{urlencode(share_params)}"
        return {
            "ok": True,
            "session_id": session_id,
            "payload_url": f"/viewer/three/session/{session_id}",
            "share_url": share_url,
        }

    @app.get(
        "/viewer/three/schema",
        tags=["viewer"],
        summary="Binary schema for Three.js frame payload",
    )
    def viewer_three_schema():
        return viewer_frame_schema()

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
            "format_version": VIEWER_FRAME_MAGIC.decode("ascii"),
            "binary_endpoint_template": (
                "/viewer/three/frame.bin?"
                + urlencode(
                    {
                        "t4dataset_id": t4dataset_id,
                        "scenario_name": resolved_scenario,
                        **({"version": version} if version else {}),
                    }
                )
                + "&frame_index={frame_index}"
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
            blob = pack_viewer_frame_binary(
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
                    "X-T4V-Format": VIEWER_FRAME_MAGIC.decode("ascii"),
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
                        "/viewer/three/frame.bin?"
                        + urlencode(
                            {
                                "t4dataset_id": t4dataset_id,
                                "scenario_name": resolved_scenario,
                                "frame_index": i,
                                **({"version": version} if version else {}),
                            }
                        )
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
                if str(ch) == payload.get("camera"):
                    # The default camera is already computed above — reuse it
                    # (snapshot, so cameras_payload doesn't self-reference).
                    all_rows.append(dict(payload))
                    continue
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
        logger.info(
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
        session_id: Optional[str] = Query(None),
        compare_view: Optional[str] = Query(None),
        compare_mode: Optional[str] = Query(None),
        external_bbox_yaw_offset: Optional[str] = Query(None),
        external_bbox_swap_lw: Optional[str] = Query(None),
        external_bbox_alignment_version: Optional[str] = Query(None),
        hide_panels: Optional[str] = Query(None),
        embed: Optional[str] = Query(None),
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
        if session_id:
            qs_params["session_id"] = session_id
        if compare_view:
            qs_params["compare_view"] = compare_view
        if compare_mode:
            qs_params["compare_mode"] = compare_mode
        if external_bbox_yaw_offset is not None:
            qs_params["external_bbox_yaw_offset"] = external_bbox_yaw_offset
        if external_bbox_swap_lw is not None:
            qs_params["external_bbox_swap_lw"] = external_bbox_swap_lw
        if external_bbox_alignment_version is not None:
            qs_params["external_bbox_alignment_version"] = external_bbox_alignment_version
        if hide_panels is not None:
            qs_params["hide_panels"] = hide_panels
        if embed is not None:
            qs_params["embed"] = embed
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


    @app.get(
        "/viewer/tlr/debug/message-received",
        tags=["viewer"],
        summary="Debug ack: TLR viewer received tlr_eval_by_frame postMessage",
    )
    def viewer_tlr_debug_message_received(
        t4dataset_id: str,
        scenario_name: Optional[str] = Query(None),
        frame_index: int = Query(0, ge=0),
        message_type: Optional[str] = Query(
            None,
            description="Overlay message type received by the TLR viewer.",
        ),
        frames_count: Optional[int] = Query(
            None,
            ge=0,
            description="Number of frame keys when message_type=tlr_eval_by_frame.",
        ),
    ):
        logger.info(
            "[viewer:tlr-message] "
            f"dataset={t4dataset_id} "
            f"scenario={scenario_name or ''} "
            f"frame={frame_index} "
            f"type={message_type or 'tlr_eval_by_frame'} "
            f"frames={frames_count if frames_count is not None else ''}"
        )
        return {
            "ok": True,
            "t4dataset_id": t4dataset_id,
            "scenario_name": scenario_name,
            "frame_index": frame_index,
            "message_type": message_type,
            "frames_count": frames_count,
        }

    @app.get(
        "/viewer/tlr/frame",
        response_model=TlrFrameResponse,
        tags=["viewer"],
        summary="Traffic-light frame payload with image and 2D annotations",
    )
    def viewer_tlr_frame(
        t4dataset_id: str,
        scenario_name: Optional[str] = None,
        frame_index: int = Query(0, ge=0),
        camera: Optional[str] = None,
        version: Optional[str] = None,
    ):
        dataset_path = _resolve_dataset(t4dataset_id)
        try:
            t4 = _cache.load(dataset_path, version=version)
            return _tlr_frame_payload(
                t4dataset_id=t4dataset_id,
                dataset_path=dataset_path,
                t4=t4,
                scenario_name=scenario_name,
                frame_index=frame_index,
                camera=camera,
                version=version,
            )
        except HTTPException:
            raise
        except Exception as exc:
            _safe_error(
                500,
                "tlr_viewer_failed",
                "Failed to build the TLR frame payload.",
                exc=exc,
            )

    @app.get("/viewer/tlr")
    def viewer_tlr_page(
        t4dataset_id: str = Query(...),
        scenario_name: Optional[str] = Query(None),
        frame_index: int = Query(0, ge=0),
        camera: Optional[str] = Query(None),
        version: Optional[str] = Query(None),
    ):
        esc = html.escape
        qs_params = {
            "t4dataset_id": t4dataset_id,
            "frame_index": frame_index,
        }
        if scenario_name:
            qs_params["scenario_name"] = scenario_name
        if camera:
            qs_params["camera"] = camera
        if version:
            qs_params["version"] = version
        qs = urlencode(qs_params)
        scenario_label = scenario_name or "(auto)"
        tmpl = _load_template("viewer_tlr.html")
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
        logger.info(
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
        logger.info(
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
        logger.info(
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

_ENV_PREFIX = "T4_SERVER_"


def _env_key(name: str) -> str:
    return f"{_ENV_PREFIX}{name}"


def _set_server_runtime_env(args) -> None:
    """Expose app-construction settings for uvicorn worker subprocesses."""
    # Handle multiple data directories - serialize as JSON list
    data_dirs_input = args.data_dir if args.data_dir else ["./t4datasets"]
    data_dirs = [str(Path(d).expanduser().resolve()) for d in data_dirs_input]
    import json
    os.environ[_env_key("DATA_DIRS")] = json.dumps(data_dirs)
    os.environ[_env_key("SEARCH_DEPTH")] = str(args.search_depth)
    os.environ[_env_key("TIER4_CACHE")] = str(args.tier4_cache)
    os.environ[_env_key("DATASET_PATH_CACHE_TTL")] = str(args.dataset_path_cache_ttl)
    os.environ[_env_key("VISIBILITY_MODE")] = str(args.visibility_mode)


def create_app_from_env():
    """Uvicorn app factory used for multi-worker and reload mode."""
    import json
    data_dirs_raw = os.environ.get(_env_key("DATA_DIRS"), "[]")
    try:
        data_dirs_raw_list = json.loads(data_dirs_raw)
    except (json.JSONDecodeError, TypeError):
        data_dirs_raw_list = ["./t4datasets"]
    data_dirs = [Path(d).expanduser().resolve() for d in data_dirs_raw_list]
    search_depth = int(os.environ.get(_env_key("SEARCH_DEPTH"), "1"))
    tier4_cache_size = int(os.environ.get(_env_key("TIER4_CACHE"), "8"))
    dataset_path_cache_ttl_s = float(os.environ.get(_env_key("DATASET_PATH_CACHE_TTL"), "30.0"))
    visibility_mode = os.environ.get(_env_key("VISIBILITY_MODE"), "public")
    return _build_app(
        data_dirs=data_dirs,
        search_depth=search_depth,
        tier4_cache_size=tier4_cache_size,
        dataset_path_cache_ttl_s=dataset_path_cache_ttl_s,
        visibility_mode=visibility_mode,
    )

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Serve the T4 Visualizer render API over HTTP."
    )
    parser.add_argument(
        "--data-dir", action="append", default=[], metavar="PATH",
        help="Directory that contains T4 datasets (default: ./t4datasets). Can be specified multiple times.",
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
        "--workers", type=int, default=1, metavar="N",
        help="Number of uvicorn worker processes (default: 1).",
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
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.reload and args.workers != 1:
        parser.error("--reload cannot be used together with --workers > 1")
    return args


def main(argv=None):
    """Entry point for the ``t4-server`` command."""
    args = _parse_args(argv)
    # Request/telemetry lines go through the module logger; make them visible
    # by default without stealing uvicorn's own logging config.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        import uvicorn
    except ImportError as exc:
        print(
            "uvicorn is required to run the server. "
            "Install with: pip install fastapi uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    # Use provided data dirs or default
    data_dirs_input = args.data_dir if args.data_dir else ["./t4datasets"]
    data_dirs = [Path(d).expanduser().resolve() for d in data_dirs_input]
    for data_dir in data_dirs:
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Data directories : {data_dirs}")
    print(f"  Search depth      : {args.search_depth}")
    print(f"  Workers        : {args.workers}")
    print(f"  Tier4 cache    : {args.tier4_cache} datasets")
    print(f"  Visibility mode: {args.visibility_mode}")
    ttl = args.dataset_path_cache_ttl
    print(
        f"  Path lookup cache: "
        f"{'off' if ttl <= 0 else f'{ttl:g}s TTL'}"
    )
    print(f"  Listening on   : http://{args.host}:{args.port}")

    _set_server_runtime_env(args)

    uvicorn.run(
        "t4_visualizer.server:create_app_from_env",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
