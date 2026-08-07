"""Closure-free data helpers for the Three.js viewer endpoints.

Pointcloud decoding, dataset/eval box projection to camera images, and
lanelet OSM parsing. Everything here is pure with respect to the FastAPI
app: no routes, no caches, no error-shaping — those stay in server.py.
Extracted verbatim from the _build_app closure so they are importable and
unit-testable.
"""

import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

