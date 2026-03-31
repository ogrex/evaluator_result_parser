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
        Accepts a JSON body matching RenderRequest.
        Returns a JSON body matching RenderResponse.

    GET  /health
        Returns {"status": "ok"}.

    GET  /datasets
        Lists dataset IDs found under the configured data_dir.

    GET  /datasets/{t4dataset_id}/scenarios
        Lists scenes in that dataset (name, token, description, nbr_samples).
        Optional query parameter: ``version`` (same as ``POST /render``).

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
"""

from __future__ import annotations

import argparse
import base64
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

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

    def get(self, path: Path):
        """Return cached Tier4 for *path*, or None if not present."""
        with self._lock:
            if path in self._cache:
                self._order.remove(path)
                self._order.append(path)
                return self._cache[path]
        return None

    def put(self, path: Path, t4) -> None:
        """Store *t4* for *path*, evicting the LRU entry if over capacity."""
        with self._lock:
            if path in self._cache:
                self._order.remove(path)
            elif len(self._cache) >= self._max_size:
                evict = self._order.pop(0)
                del self._cache[evict]
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
        self.put(path, t4)
        return t4


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

except ImportError:
    pass  # Proper error is raised inside _build_app when fastapi is missing


# ---------------------------------------------------------------------------

def _build_app(data_dir: Path, search_depth: int, tier4_cache_size: int):
    """Construct and return the FastAPI application."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.encoders import jsonable_encoder
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError(
            "fastapi and pydantic are required for the server. "
            "Install with: pip install fastapi uvicorn"
        ) from exc

    from t4_visualizer.batch import find_dataset_in_dir
    from t4_visualizer.downloader import list_webauto_annotation_dataset_ids
    from t4_visualizer.visualize import (
        RenderImage,
        TargetObject,
        VisualizationRequest,
        render_frame,
    )

    app = FastAPI(title="T4 Visualizer", version="0.1.0")
    _cache = _Tier4Cache(max_size=tier4_cache_size)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dataset(t4dataset_id: str) -> Path:
        path = find_dataset_in_dir(data_dir, t4dataset_id, search_depth)
        if path is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Dataset '{t4dataset_id}' not found under {data_dir} "
                    f"(search_depth={search_depth})"
                ),
            )
        return path

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/datasets")
    def list_datasets():
        """Return dataset IDs visible under the configured data_dir."""
        resolved = str(data_dir.resolve())
        try:
            top_level_dirs = sorted(
                p.name for p in data_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )[:32]
        except OSError:
            top_level_dirs = []
        ann_present = (data_dir / "annotation_dataset").is_dir()
        if not data_dir.exists():
            out: Dict[str, object] = {
                "data_dir": str(data_dir),
                "data_dir_resolved": resolved,
                "datasets": [],
                "top_level_dirs": [],
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
            "data_dir_resolved": resolved,
            "datasets": uniq,
            "top_level_dirs": top_level_dirs,
            "annotation_dataset_present": ann_present,
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
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return ScenariosListResponse(
            t4dataset_id=t4dataset_id,
            scenarios=[ScenarioOut(**row) for row in raw],
            version=version,
        )

    @app.post("/render", response_model=RenderResponse)
    def render(body: RenderRequest):
        """Render a single frame and return base64-encoded PNG images.

        Server-side timings are in the JSON body and duplicated on response headers
        so any HTTP client can read them without parsing JSON.
        """
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

        request = VisualizationRequest(
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

        t0 = time.perf_counter()
        try:
            t4 = _cache.load(dataset_path, version=body.version)
            t1 = time.perf_counter()
            result = render_frame(request, t4=t4)
            t2 = time.perf_counter()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        elapsed_ms = round((t2 - t0) * 1000.0, 3)
        tier4_load_ms = round((t1 - t0) * 1000.0, 3)
        render_ms = round((t2 - t1) * 1000.0, 3)

        # Expose server-side timings to all clients (body + headers).
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
        return JSONResponse(content=jsonable_encoder(payload), headers=hdrs)

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
        "--reload", action="store_true", default=False,
        help="Enable uvicorn auto-reload (development only).",
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
    print(f"  Listening on   : http://{args.host}:{args.port}")

    app = _build_app(
        data_dir=data_dir,
        search_depth=args.search_depth,
        tier4_cache_size=args.tier4_cache,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
