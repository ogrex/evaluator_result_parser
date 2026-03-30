"""Visualize T4 dataset images and point clouds for a specific UUID and timestamp.

Usage:
    python visualize_by_uuid.py <dataset_path> <timestamp_us> [options]

Examples:
    # Visualize closest sample to a Unix timestamp (microseconds)
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000

    # Use Rerun for interactive 3D visualization
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000 --rerun

    # Save static plots to a directory
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000 --save-dir output/

    # Specify a dataset version (default: latest)
    python visualize_by_uuid.py /path/to/t4dataset 1609459200000000 --version annotation

Args:
    dataset_path: Path to the T4 dataset root directory (the UUID directory)
    timestamp_us: Target timestamp in microseconds (Unix time)
    --rerun: Use Rerun for interactive visualization instead of matplotlib
    --save-dir: Directory to save visualization outputs
    --version: Dataset version/annotation directory name (default: auto-detect)
    --cameras: Comma-separated list of camera channels to show (default: all)
    --no-annotations: Skip rendering 3D bounding boxes
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

_VISIBILITY_TIER_THRESHOLD = 0.5  # ROIs with visibility below this are deprioritised


@dataclass
class TargetObject:
    """One target object to highlight in the visualization (from CSV row)."""
    uuid: str           # instance_token in T4 dataset
    x: float = 0.0     # detection position in ego frame (from CSV)
    y: float = 0.0
    z: float = 0.0
    label: str = ""
    width: float = 0.0   # BBOX dimensions in ego frame [m]
    length: float = 0.0
    height: float = 0.0
    yaw: float = 0.0     # heading in ego frame [rad], rotation around z-axis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_token(box) -> str:
    """Extract the annotation token from a Box2D or Box3D object."""
    for attr in ("token", "annotation_token"):
        val = getattr(box, attr, None)
        if val:
            return str(val)
    return ""


def _project_ego_to_cam(t4, sample_data_token: str, point_ego) -> Optional[tuple]:
    """Project a 3D point in ego (base_link) frame to camera pixel (u, v).

    Uses calibrated_sensor from the T4 dataset (no rosbag / TF needed).

    Args:
        t4: Tier4 instance.
        sample_data_token: Token of the camera SampleData.
        point_ego: (x, y, z) in ego / base_link frame.

    Returns:
        (u, v) float pixel coordinates, or None if behind the camera or
        if the calibrated sensor has no intrinsics (i.e. LiDAR).
    """
    try:
        sd = t4.get("sample_data", sample_data_token)
        cs = t4.get("calibrated_sensor", sd.calibrated_sensor_token)

        # camera_intrinsic is empty for LiDAR
        K_raw = cs.camera_intrinsic
        if K_raw is None or len(K_raw) == 0:
            return None
        K = np.array(K_raw, dtype=float)
        if K.shape != (3, 3):
            return None

        t_ego = np.array(cs.translation, dtype=float)
        p = np.array(point_ego, dtype=float)

        # Rotation: sensor → ego (stored as [w, x, y, z] in T4)
        rot = cs.rotation
        if hasattr(rot, "inverse"):
            # Already a pyquaternion Quaternion
            q = rot
        else:
            # List/array [w, x, y, z]
            from pyquaternion import Quaternion
            q = Quaternion(np.array(rot, dtype=float))

        # ego → sensor:  p_cam = q^{-1} * (p_ego - t)
        p_cam = q.inverse.rotate(p - t_ego)

        if p_cam[2] <= 0.1:          # behind (or too close to) camera
            return None

        uvw = K @ p_cam
        return (float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]))

    except Exception:
        return None


def _project_bbox_to_roi(t4, sample_data_token: str, obj: "TargetObject",
                          img_w: int, img_h: int) -> Optional[tuple]:
    """Project the 8 corners of a detection BBOX onto a camera image.

    Returns (u_min, v_min, u_max, v_max, visibility) clipped to the image
    bounds, or None if no corner projects in front of the camera or the
    resulting ROI is empty / fully outside the image.

    ``visibility`` is the ratio of clipped area to raw projected area
    (0.0–1.0); 1.0 means the entire BBOX projection is within the image.

    Only corners with z > 0 in camera frame are used, so the result is a
    safe under-estimate when the box straddles the image plane.
    """
    if obj.width <= 0 or obj.length <= 0 or obj.height <= 0:
        return None

    hw, hl, hh = obj.width / 2.0, obj.length / 2.0, obj.height / 2.0
    cx, cy, cz = obj.x, obj.y, obj.z

    # 8 corners of the rotated bounding box in ego frame.
    # body-x (forward) = length direction, body-y (lateral) = width direction.
    # Rotation around ego z-axis by yaw:  ego = R(yaw) * body
    #   body-x unit in ego: ( cos(yaw),  sin(yaw), 0)
    #   body-y unit in ego: (-sin(yaw),  cos(yaw), 0)
    import math as _math
    c, s = _math.cos(obj.yaw), _math.sin(obj.yaw)
    corners = [
        (
            cx + lx * c - ly * s,
            cy + lx * s + ly * c,
            cz + dz * hh,
        )
        for lx in (-hl, hl)   # along length (body-x)
        for ly in (-hw, hw)   # along width  (body-y)
        for dz in (-1.0, 1.0)
    ]

    us, vs = [], []
    for corner in corners:
        uv = _project_ego_to_cam(t4, sample_data_token, corner)
        if uv is not None:        # None means behind the camera
            us.append(uv[0])
            vs.append(uv[1])

    if not us:
        return None

    # Raw bounding rect of projected corners
    u_min_raw, u_max_raw = min(us), max(us)
    v_min_raw, v_max_raw = min(vs), max(vs)

    # Clip to image bounds
    u_min = max(0.0, min(u_min_raw, float(img_w - 1)))
    u_max = max(0.0, min(u_max_raw, float(img_w - 1)))
    v_min = max(0.0, min(v_min_raw, float(img_h - 1)))
    v_max = max(0.0, min(v_max_raw, float(img_h - 1)))

    # Reject degenerate / fully-outside rectangles
    if u_max - u_min < 2 or v_max - v_min < 2:
        return None

    raw_area = (u_max_raw - u_min_raw) * (v_max_raw - v_min_raw)
    if raw_area > 0:
        clipped_area = (u_max - u_min) * (v_max - v_min)
        visibility = clipped_area / raw_area
    else:
        visibility = 1.0

    return (u_min, v_min, u_max, v_max, visibility)


def _get_target_ann_tokens(t4, sample, target_objects: List[TargetObject]) -> Set[str]:
    """Return sample_annotation tokens for target instance_tokens in this sample."""
    if not target_objects:
        return set()
    instance_tokens = {obj.uuid for obj in target_objects}
    result = set()
    try:
        for ann in t4.sample_annotation:
            if ann.sample_token == sample.token and ann.instance_token in instance_tokens:
                result.add(ann.token)
    except Exception as e:
        print(f"  WARNING: Could not resolve annotation tokens: {e}")
    if not result:
        print(
            f"  WARNING: No annotation found for instance_tokens {instance_tokens}. "
            "Detection position markers will be drawn in BEV as fallback."
        )
    return result


def find_closest_sample(t4, timestamp_us: int):
    """Return the Sample record whose timestamp is closest to *timestamp_us*.

    Args:
        t4: Tier4 instance already loaded with the dataset.
        timestamp_us: Target timestamp in **microseconds**.

    Returns:
        The closest Sample object, or None if no samples exist.
    """
    samples = t4.sample  # list[Sample]
    if not samples:
        return None
    best = min(samples, key=lambda s: abs(s.timestamp - timestamp_us))
    return best


def find_sample_by_scene_and_index(t4, scene_name: str, frame_index: int):
    """Return the frame_index-th sample in the named scene.

    Walks the linked list of samples starting from scene.first_sample_token.

    Args:
        t4: Tier4 instance already loaded with the dataset.
        scene_name: Value of the ``name`` field in the scene table.
        frame_index: 0-based index of the frame within the scene.

    Returns:
        The Sample object at position *frame_index*.

    Raises:
        ValueError: If no scene with *scene_name* is found.
        IndexError: If *frame_index* exceeds the number of samples in the scene.
    """
    scene = next((s for s in t4.scene if s.name == scene_name), None)
    if scene is None:
        available = t4.scene
        if len(available) == 1:
            print(
                f"  WARNING: Scene '{scene_name}' not found; "
                f"falling back to the only scene in dataset: '{available[0].name}'"
            )
            scene = available[0]
        else:
            raise ValueError(
                f"Scene '{scene_name}' not found in dataset.\n"
                f"Available scenes: {[s.name for s in available]}"
            )

    token = scene.first_sample_token
    for i in range(frame_index):
        sample = t4.get("sample", token)
        if not sample.next:
            raise IndexError(
                f"frame_index {frame_index} is out of range for scene '{scene_name}' "
                f"(scene contains only {i + 1} sample(s))."
            )
        token = sample.next
    return t4.get("sample", token)


def list_scene_summaries(t4) -> List[Dict[str, Any]]:
    """Return one entry per scene in the dataset (names usable as ``scenario_name`` in :func:`find_sample_by_scene_and_index`).

    Each dict has:

    - ``name``: Scene name (matches ``scene.name`` in the T4 tables).
    - ``token``: Scene record token.
    - ``description``: Human-readable description when present.
    - ``nbr_samples``: Number of frames (samples) in the scene; valid ``frame_index`` is ``0 .. nbr_samples - 1``.
    """
    rows: List[Dict[str, Any]] = []
    for scene in t4.scene:
        desc = getattr(scene, "description", None)
        nbr = getattr(scene, "nbr_samples", None)
        rows.append(
            {
                "name": scene.name,
                "token": scene.token,
                "description": (desc or "") if desc is not None else "",
                "nbr_samples": int(nbr) if nbr is not None else 0,
            }
        )
    rows.sort(key=lambda r: (r["name"], r["token"]))
    return rows


def list_camera_channels(t4, sample) -> List[str]:
    """Return camera channel names available for the given sample."""
    channels = []
    for channel, token in sample.data.items():
        sd = t4.get("sample_data", token)
        if sd.fileformat.value in ("jpg", "png"):
            channels.append(channel)
    return sorted(channels)


def list_lidar_channels(t4, sample) -> List[str]:
    """Return LiDAR channel names available for the given sample."""
    channels = []
    for channel, token in sample.data.items():
        sd = t4.get("sample_data", token)
        if sd.fileformat.value in ("pcd", "bin", "pcd.bin"):
            channels.append(channel)
    return sorted(channels)


# ---------------------------------------------------------------------------
# Static matplotlib visualization
# ---------------------------------------------------------------------------

def visualize_static(
    t4,
    sample,
    cameras: Optional[List[str]] = None,
    show_annotations: bool = True,
    save_dir: Optional[str] = None,
    filename_prefix: Optional[str] = None,
    target_objects: Optional[List[TargetObject]] = None,
    crop_cameras: bool = False,
    crop_padding: int = 40,
    crop_min_size: int = 300,
) -> None:
    """Render images and a bird's-eye-view point cloud with matplotlib.

    Args:
        t4: Tier4 instance.
        sample: Target Sample record.
        cameras: Camera channels to display (None = all cameras).
        show_annotations: Overlay 2D/3D bounding boxes on images.
        save_dir: If given, save figures here instead of showing interactively.
        filename_prefix: Prefix for saved filenames (default: timestamp).
        crop_cameras: If True, replace the full camera grid with a single
            cropped view of the camera showing the largest projected BBOX ROI.
            Output filename becomes ``{prefix}_visualization_crop.png``.
        crop_padding: Pixels of padding around the ROI (default 40).
        crop_min_size: Minimum crop dimension in pixels (default 300).
    """
    import matplotlib
    if save_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image  # Pillow

    ts_us = sample.timestamp
    prefix = filename_prefix if filename_prefix else str(ts_us)
    print(f"[visualize_static] Rendering sample token={sample.token}, timestamp={ts_us} us")

    available_cameras = list_camera_channels(t4, sample)
    selected_cameras = cameras if cameras else available_cameras
    selected_cameras = [c for c in selected_cameras if c in available_cameras]

    target_ann_tokens = _get_target_ann_tokens(t4, sample, target_objects or [])

    if not selected_cameras:
        print("  WARNING: No matching camera channels found in this sample.")

    lidar_channels = list_lidar_channels(t4, sample)
    if not lidar_channels:
        print("  WARNING: No LiDAR data found in this sample.")
    lidar_channel = lidar_channels[0] if lidar_channels else None

    _plot_combined(
        t4, sample, selected_cameras, lidar_channel, show_annotations,
        save_dir, prefix, target_ann_tokens, target_objects or [],
        crop_cameras=crop_cameras,
        crop_padding=crop_padding,
        crop_min_size=crop_min_size,
    )

    if not save_dir:
        plt.show()


# ---------------------------------------------------------------------------
# Programmatic render API  (returns bytes, no file I/O required by caller)
# ---------------------------------------------------------------------------

@dataclass
class VisualizationRequest:
    """Parameters for a single-frame render.  Passed to :func:`render_frame`."""

    dataset_path: Path
    """Local path to the T4 dataset root directory."""

    scenario_name: str
    """Scene name inside the dataset (e.g. ``"scene-0001"``)."""

    frame_index: int
    """0-based frame index within the scene."""

    target_objects: List[TargetObject] = field(default_factory=list)
    """Objects to highlight in the visualization."""

    cameras: Optional[List[str]] = None
    """Camera channels to render (``None`` = all available)."""

    show_annotations: bool = True
    """Overlay bounding boxes on camera images and BEV."""

    version: Optional[str] = None
    """T4 dataset version string (``None`` = auto-detect)."""

    crop_cameras: bool = False
    """Crop camera view to the region containing target objects."""

    crop_padding: int = 40
    crop_min_size: int = 300


@dataclass
class RenderImage:
    """One rendered figure as PNG bytes."""

    data: bytes
    """Raw PNG bytes."""

    label: str
    """``"combined"`` for standard layout; camera channel name for crop mode."""


@dataclass
class VisualizationResult:
    """Output of :func:`render_frame`."""

    images: List[RenderImage]
    """Rendered figures.  Standard mode: one item.  Crop mode: one per camera."""

    sample_token: str
    timestamp_us: int


def render_frame(
    request: VisualizationRequest,
    t4=None,
) -> VisualizationResult:
    """Render a single frame and return PNG bytes without writing to disk.

    Args:
        request: Rendering parameters.
        t4: Pre-loaded ``Tier4`` instance for *request.dataset_path*.
            When omitted the dataset is loaded inside this call.
            Pass a cached instance to avoid reloading the same dataset
            repeatedly (e.g. in a server or parallel worker).

    Returns:
        :class:`VisualizationResult` containing one or more PNG images.
    """
    import matplotlib
    matplotlib.use("Agg")

    import tempfile
    from pathlib import Path as _Path

    if t4 is None:
        try:
            from t4_devkit import Tier4
        except ImportError as exc:
            raise ImportError(
                "t4_devkit is not installed. "
                "Install with: pip install git+https://github.com/tier4/t4-devkit.git"
            ) from exc
        from t4_visualizer.downloader import find_t4_root, patch_missing_t4_tables
        t4_root = find_t4_root(_Path(request.dataset_path))
        patch_missing_t4_tables(t4_root)
        kwargs = {"version": request.version} if request.version else {}
        t4 = Tier4(str(t4_root), **kwargs)

    sample = find_sample_by_scene_and_index(t4, request.scenario_name, request.frame_index)

    _RENDER_PREFIX = "_render"

    with tempfile.TemporaryDirectory(prefix="t4render_") as tmpdir:
        visualize_static(
            t4,
            sample,
            cameras=request.cameras,
            show_annotations=request.show_annotations,
            save_dir=tmpdir,
            filename_prefix=_RENDER_PREFIX,
            target_objects=request.target_objects,
            crop_cameras=request.crop_cameras,
            crop_padding=request.crop_padding,
            crop_min_size=request.crop_min_size,
        )

        images: List[RenderImage] = []
        for png_path in sorted(_Path(tmpdir).glob("*.png")):
            stem = png_path.stem  # e.g. "_render_visualization" or "_render_CAM_FRONT_visualization_crop"
            if stem.endswith("_visualization"):
                label = "combined"
            elif "_visualization_crop" in stem:
                # "_render_CAM_FRONT_visualization_crop" → "CAM_FRONT"
                label = stem[len(_RENDER_PREFIX) + 1 : stem.index("_visualization_crop")]
            else:
                label = stem
            images.append(RenderImage(data=png_path.read_bytes(), label=label))

    return VisualizationResult(
        images=images,
        sample_token=sample.token,
        timestamp_us=sample.timestamp,
    )


def _fill_camera_axes(t4, sample, camera_channels, show_annotations, axes,
                      target_ann_tokens, target_objects):
    """Render camera images and detection overlays onto the given axes list."""
    import matplotlib.patches as patches
    from PIL import Image

    for idx, channel in enumerate(camera_channels):
        ax = axes[idx]
        token = sample.data.get(channel)
        if token is None:
            ax.set_visible(False)
            continue

        if show_annotations:
            data_path, boxes_2d, _ = t4.get_sample_data(
                token, as_3d=False, as_sensor_coord=True
            )
        else:
            data_path, _, _ = t4.get_sample_data(token, as_3d=False)
            boxes_2d = []

        img = Image.open(data_path)
        ax.imshow(img)
        ax.set_title(channel, fontsize=9)
        ax.axis("off")

        for box in boxes_2d:
            try:
                roi = box.roi  # (left, top, right, bottom)
                bx, by, w, h = roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]
                is_target = bool(target_ann_tokens) and _box_token(box) in target_ann_tokens
                color = "red" if is_target else "lime"
                lw = 2.5 if is_target else 1.0
                rect = patches.Rectangle(
                    (bx, by), w, h,
                    linewidth=lw, edgecolor=color, facecolor="none",
                    zorder=3 if is_target else 2,
                )
                ax.add_patch(rect)
                if is_target:
                    label_text = getattr(box, "label", None) or ""
                    ax.text(
                        bx, by - 3, f"[TARGET] {label_text}",
                        color="red", fontsize=7, fontweight="bold",
                        va="bottom", zorder=4,
                    )
            except Exception:
                pass

        if target_objects:
            img_w, img_h = img.size
            for obj in target_objects:
                # Try full 3D BBOX → 2D ROI first
                det_roi = _project_bbox_to_roi(t4, token, obj, img_w, img_h)
                if det_roi is not None:
                    u_min, v_min, u_max, v_max, *_ = det_roi
                    bw, bh = u_max - u_min, v_max - v_min
                    rect = patches.Rectangle(
                        (u_min, v_min), bw, bh,
                        linewidth=2.0, edgecolor="yellow", facecolor="none",
                        linestyle="--", zorder=7,
                    )
                    ax.add_patch(rect)
                    ax.text(
                        u_min, v_min - 3, f"det:{obj.label or '?'}",
                        color="yellow", fontsize=7, fontweight="bold",
                        va="bottom", zorder=7,
                    )
                    continue

                # Fallback: center point marker
                uv = _project_ego_to_cam(t4, token, [obj.x, obj.y, obj.z])
                if uv is None:
                    continue
                u, v = uv
                if not (0 <= u < img_w and 0 <= v < img_h):
                    continue
                ax.plot(u, v, "+", color="yellow", markersize=18,
                        markeredgewidth=2.5, zorder=7)
                circle = patches.Circle(
                    (u, v), radius=12,
                    fill=False, edgecolor="yellow", linewidth=2.0, zorder=7,
                )
                ax.add_patch(circle)
                ax.text(
                    u + 14, v, f"det:{obj.label or '?'}",
                    color="yellow", fontsize=7, fontweight="bold",
                    va="center", zorder=7,
                )

    # Hide unused axes
    for i in range(len(camera_channels), len(axes)):
        axes[i].set_visible(False)


def _fill_bev_ax(t4, sample, lidar_channel, show_annotations, ax,
                 target_ann_tokens, target_objects):
    """Render BEV point cloud and detection overlays onto the given axes."""
    import matplotlib.patches as patches

    token = sample.data.get(lidar_channel)
    if token is None:
        print(f"  WARNING: Channel {lidar_channel} not in sample.data")
        return

    if show_annotations:
        data_path, boxes_3d, _ = t4.get_sample_data(
            token, as_3d=True, as_sensor_coord=True
        )
    else:
        data_path, _, _ = t4.get_sample_data(token, as_3d=True)
        boxes_3d = []

    points = _load_pointcloud(data_path)
    if points is None or points.shape[0] == 0:
        print(f"  WARNING: Could not load point cloud from {data_path}")
        return

    x_all = points[:, 0]
    y_all = points[:, 1]
    intensity_all = points[:, 3] if points.shape[1] > 3 else np.ones(len(x_all))

    MARGIN = 3.0   # extra space around BBOX extent [m]

    # Compute view extent from all target BBOXes (or fall back to center point)
    view_xlim = None
    view_ylim = None
    if target_objects:
        import math as _math
        xs_min, xs_max, ys_min, ys_max = [], [], [], []
        for obj in target_objects:
            hw = obj.width  / 2.0 if obj.width  > 0 else 0.0
            hl = obj.length / 2.0 if obj.length > 0 else 0.0
            # AABB of the rotated box
            c, s = abs(_math.cos(obj.yaw)), abs(_math.sin(obj.yaw))
            half_x = c * hl + s * hw   # AABB half-extent along ego-x
            half_y = s * hl + c * hw   # AABB half-extent along ego-y
            xs_min.append(obj.x - half_x)
            xs_max.append(obj.x + half_x)
            ys_min.append(obj.y - half_y)
            ys_max.append(obj.y + half_y)
        x_lo = min(xs_min) - MARGIN
        x_hi = max(xs_max) + MARGIN
        y_lo = min(ys_min) - MARGIN
        y_hi = max(ys_max) + MARGIN
        view_xlim = (x_lo, x_hi)
        view_ylim = (y_lo, y_hi)
        # Crop point cloud to the view region (no need to load points outside)
        mask = (x_all >= x_lo) & (x_all <= x_hi) & (y_all >= y_lo) & (y_all <= y_hi)
        x, y = x_all[mask], y_all[mask]
        intensity = intensity_all[mask]
    else:
        x, y, intensity = x_all, y_all, intensity_all

    max_pts = 80_000
    if len(x) > max_pts:
        idx = np.random.choice(len(x), max_pts, replace=False)
        x, y, intensity = x[idx], y[idx], intensity[idx]

    sc = ax.scatter(x, y, c=intensity, s=0.5, cmap="viridis", vmin=0, vmax=255)
    ax.figure.colorbar(sc, ax=ax, label="Intensity", shrink=0.7)

    for box in boxes_3d:
        try:
            is_target = bool(target_ann_tokens) and _box_token(box) in target_ann_tokens
            _draw_box_bev(ax, box, highlight=is_target)
        except Exception:
            pass

    if target_objects:
        import math as _math
        for obj in target_objects:
            if obj.width > 0 and obj.length > 0:
                hl, hw = obj.length / 2.0, obj.width / 2.0
                c, s = _math.cos(obj.yaw), _math.sin(obj.yaw)
                # 4 corners in body frame (lx=length-axis, ly=width-axis)
                # mapped to ego frame: ego = R(yaw) * body
                corners = np.array([
                    [obj.x + lx * c - ly * s, obj.y + lx * s + ly * c]
                    for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
                ])
                poly = patches.Polygon(
                    corners, closed=True,
                    linewidth=2.0, edgecolor="yellow", facecolor="none",
                    linestyle="--", zorder=6,
                )
                ax.add_patch(poly)
                # Heading arrow: center → front midpoint
                front_mid = np.array([obj.x + hl * c, obj.y + hl * s])
                ax.annotate(
                    "",
                    xy=(front_mid[0], front_mid[1]),
                    xytext=(obj.x, obj.y),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                    zorder=7,
                )
            ax.annotate(
                f"{obj.label or 'target'}\n({obj.x:.1f},{obj.y:.1f})",
                (obj.x, obj.y),
                xytext=(6, 6), textcoords="offset points",
                color="yellow", fontsize=7, fontweight="bold", zorder=7,
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.set_title(f"LiDAR BEV — {lidar_channel}")

    if view_xlim is not None:
        ax.set_xlim(view_xlim)
        ax.set_ylim(view_ylim)


def _group_objects_by_camera(t4, sample, camera_channels, target_objects):
    """Map each target object to its best-scoring camera, then group by camera.

    For each object the best camera is chosen with the same (tier, area) score
    used previously in _find_largest_roi_camera.

    Returns a list of (channel, token, obj_roi_pairs, img_w, img_h) where
    obj_roi_pairs is [(obj, roi), ...] for all objects whose best camera is
    this channel.  Objects that are not visible in any camera are omitted.
    The list is ordered by descending total clipped area so the "most
    interesting" camera comes first.
    """
    from PIL import Image as PILImage

    # Cache (token, img_w, img_h) per channel to avoid re-opening images.
    cam_info: dict = {}
    for channel in camera_channels:
        token = sample.data.get(channel)
        if token is None:
            continue
        try:
            data_path, _, _ = t4.get_sample_data(token, as_3d=False)
            with PILImage.open(data_path) as img:
                img_w, img_h = img.size
            cam_info[channel] = (token, img_w, img_h)
        except Exception:
            continue

    # For each object find the best (channel, roi).
    obj_best: dict = {}  # obj_idx -> (channel, roi, tier, area)
    for obj_idx, obj in enumerate(target_objects):
        for channel, (token, img_w, img_h) in cam_info.items():
            roi = _project_bbox_to_roi(t4, token, obj, img_w, img_h)
            if roi is None:
                continue
            u_min, v_min, u_max, v_max, visibility = roi
            area = (u_max - u_min) * (v_max - v_min)
            tier = 1 if visibility >= _VISIBILITY_TIER_THRESHOLD else 0
            prev = obj_best.get(obj_idx)
            if prev is None or (tier, area) > (prev[2], prev[3]):
                obj_best[obj_idx] = (channel, roi, tier, area)

    # Group objects by their best camera.
    groups: dict = {}  # channel -> [(obj, roi), ...]
    for obj_idx, (channel, roi, _, _) in obj_best.items():
        groups.setdefault(channel, []).append((target_objects[obj_idx], roi))

    # Build result sorted by total clipped area descending.
    result = []
    for channel, obj_roi_pairs in groups.items():
        token, img_w, img_h = cam_info[channel]
        total_area = sum(
            (r[2] - r[0]) * (r[3] - r[1]) for _, r in obj_roi_pairs
        )
        result.append((channel, token, obj_roi_pairs, img_w, img_h, total_area))
    result.sort(key=lambda x: x[5], reverse=True)
    return [(ch, tok, pairs, w, h) for ch, tok, pairs, w, h, _ in result]


def _compute_crop_limits(roi, img_w, img_h, padding: int, min_size: int):
    """Expand ROI by padding, enforce min_size, clamp to image bounds.

    Returns (x0, x1, y0, y1) in image-pixel coordinates (y increases down).
    """
    u_min, v_min, u_max, v_max, *_ = roi

    u0 = max(0.0, u_min - padding)
    u1 = min(float(img_w), u_max + padding)
    v0 = max(0.0, v_min - padding)
    v1 = min(float(img_h), v_max + padding)

    # Enforce minimum width
    if u1 - u0 < min_size:
        cx = (u0 + u1) / 2.0
        u0 = max(0.0, cx - min_size / 2.0)
        u1 = min(float(img_w), u0 + min_size)
        u0 = max(0.0, u1 - min_size)   # re-clamp if hit right edge

    # Enforce minimum height
    if v1 - v0 < min_size:
        cy = (v0 + v1) / 2.0
        v0 = max(0.0, cy - min_size / 2.0)
        v1 = min(float(img_h), v0 + min_size)
        v0 = max(0.0, v1 - min_size)

    return u0, u1, v0, v1


def _plot_combined(t4, sample, camera_channels, lidar_channel, show_annotations,
                   save_dir, filename_prefix, target_ann_tokens, target_objects,
                   crop_cameras: bool = False,
                   crop_padding: int = 40,
                   crop_min_size: int = 300):
    """Single figure combining camera grid (left) and BEV point cloud (right).

    When crop_cameras=True the camera panel is replaced by a single cropped
    view of whichever camera shows the largest projected BBOX ROI.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    has_lidar = lidar_channel is not None
    ts_us = sample.timestamp

    # ------------------------------------------------------------------
    # Crop-camera mode: one [cropped camera | BEV] figure per camera group
    # ------------------------------------------------------------------
    if crop_cameras and target_objects:
        cam_groups = _group_objects_by_camera(
            t4, sample, camera_channels, target_objects
        )
        if cam_groups:
            prefix = filename_prefix if filename_prefix else str(ts_us)
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)

            for channel, token, obj_roi_pairs, img_w, img_h in cam_groups:
                # Merge all object ROIs for this camera into one crop region.
                merged_roi = (
                    min(r[0] for _, r in obj_roi_pairs),
                    min(r[1] for _, r in obj_roi_pairs),
                    max(r[2] for _, r in obj_roi_pairs),
                    max(r[3] for _, r in obj_roi_pairs),
                )
                x0, x1, y0, y1 = _compute_crop_limits(
                    merged_roi, img_w, img_h, crop_padding, crop_min_size
                )
                crop_w = x1 - x0
                crop_h = y1 - y0

                bev_size = 7.0
                cam_w_in = bev_size * (crop_w / crop_h)
                fig_w = cam_w_in + bev_size + 0.3
                fig_h = bev_size + 0.4

                fig = plt.figure(figsize=(fig_w, fig_h))
                gs = gridspec.GridSpec(
                    1, 2, figure=fig,
                    width_ratios=[cam_w_in, bev_size],
                    hspace=0.05, wspace=0.08,
                )
                cam_ax = fig.add_subplot(gs[0, 0])
                bev_ax = fig.add_subplot(gs[0, 1])

                _fill_camera_axes(
                    t4, sample, [channel], show_annotations,
                    [cam_ax], target_ann_tokens,
                    [obj for obj, _ in obj_roi_pairs],
                )
                cam_ax.set_xlim(x0, x1)
                cam_ax.set_ylim(y1, y0)
                cam_ax.set_title(f"{channel} [crop]", fontsize=9)

                if has_lidar:
                    _fill_bev_ax(t4, sample, lidar_channel, show_annotations,
                                 bev_ax, target_ann_tokens, target_objects)

                fig.suptitle(
                    f"timestamp: {ts_us} µs  ({ts_us / 1e6:.3f} s)    "
                    f"token: {sample.token}",
                    fontsize=9,
                )
                fig.tight_layout()

                if save_dir:
                    out = Path(save_dir) / f"{prefix}_{channel}_visualization_crop.png"
                    fig.savefig(out, dpi=150, bbox_inches="tight")
                    print(f"  Saved: {out}")
                plt.close(fig)
            return

        # No ROI found in any camera — fall through to standard layout.
        print("  WARNING: crop_cameras=True but no ROI visible; "
              "falling back to full camera grid.")

    # ------------------------------------------------------------------
    # Standard mode: [camera grid | BEV]
    # ------------------------------------------------------------------
    n_cams = len(camera_channels)
    cam_ncols = min(3, n_cams) if n_cams > 0 else 1
    cam_nrows = max(1, (n_cams + cam_ncols - 1) // cam_ncols)

    bev_col_w = 1.3
    fig_w = cam_ncols * 6 + (7 * bev_col_w if has_lidar else 0)
    fig_h = cam_nrows * 4 + 0.5

    if has_lidar:
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(
            cam_nrows, cam_ncols + 1, figure=fig,
            width_ratios=[1.0] * cam_ncols + [bev_col_w],
            hspace=0.05, wspace=0.05,
        )
        cam_axes = [fig.add_subplot(gs[r, c])
                    for r in range(cam_nrows) for c in range(cam_ncols)]
        bev_ax = fig.add_subplot(gs[:, cam_ncols])
    else:
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(cam_nrows, cam_ncols, figure=fig,
                               hspace=0.05, wspace=0.05)
        cam_axes = [fig.add_subplot(gs[r, c])
                    for r in range(cam_nrows) for c in range(cam_ncols)]
        bev_ax = None

    if n_cams > 0:
        _fill_camera_axes(t4, sample, camera_channels, show_annotations,
                          cam_axes, target_ann_tokens, target_objects)

    if has_lidar and bev_ax is not None:
        _fill_bev_ax(t4, sample, lidar_channel, show_annotations,
                     bev_ax, target_ann_tokens, target_objects)

    fig.suptitle(
        f"timestamp: {ts_us} µs  ({ts_us / 1e6:.3f} s)    token: {sample.token}",
        fontsize=9,
    )
    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        prefix = filename_prefix if filename_prefix else str(ts_us)
        out = Path(save_dir) / f"{prefix}_visualization.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def _draw_box_bev(ax, box, highlight: bool = False):
    """Draw the BEV footprint of a Box3D on a matplotlib Axes."""
    import matplotlib.patches as patches

    color = "yellow" if highlight else "cyan"
    lw = 2.5 if highlight else 1.0
    arrow_color = "red" if highlight else "orange"
    zorder = 4 if highlight else 2

    center = box.center
    size = box.size

    try:
        # t4-devkit Box3D.corners() is a METHOD returning shape (8, 3).
        # Corner layout (x=forward, y=left, z=up), pre-rotation:
        #   i=0: ( l/2,  w/2,  h/2)  front-left-top
        #   i=1: ( l/2, -w/2,  h/2)  front-right-top
        #   i=2: ( l/2, -w/2, -h/2)  front-right-bottom
        #   i=3: ( l/2,  w/2, -h/2)  front-left-bottom
        #   i=4: (-l/2,  w/2,  h/2)  rear-left-top
        #   i=5: (-l/2, -w/2,  h/2)  rear-right-top
        #   i=6: (-l/2, -w/2, -h/2)  rear-right-bottom
        #   i=7: (-l/2,  w/2, -h/2)  rear-left-bottom
        corners = box.corners()  # (8, 3)
        # BEV polygon: top-face corners in traversal order 0→1→5→4→0
        #   forms a proper closed rectangle in the xy-plane.
        poly_xy = corners[[0, 1, 5, 4], :2]  # (4, 2)
        xs = np.append(poly_xy[:, 0], poly_xy[0, 0])
        ys = np.append(poly_xy[:, 1], poly_xy[0, 1])
        ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder)
        # Heading arrow: box centre → midpoint of front edge (avg of 0 and 1)
        front_mid = (corners[0, :2] + corners[1, :2]) / 2
        ax.annotate(
            "",
            xy=(front_mid[0], front_mid[1]),
            xytext=(center[0], center[1]),
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=lw),
            zorder=zorder,
        )
        if highlight:
            ax.text(
                center[0], center[1], "[T]",
                color="yellow", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=5,
            )
    except (AttributeError, TypeError):
        # Fallback: axis-aligned rectangle (rotation ignored)
        w, l = float(size[0]), float(size[1])
        rect = patches.Rectangle(
            (center[0] - w / 2, center[1] - l / 2), w, l,
            linewidth=lw, edgecolor=color, facecolor="none", zorder=zorder,
        )
        ax.add_patch(rect)


def _load_pointcloud(data_path: str) -> Optional[np.ndarray]:
    """Load a point cloud file and return Nx4 (x,y,z,intensity) array."""
    path = Path(data_path)
    if not path.exists():
        print(f"  WARNING: Point cloud file not found: {data_path}")
        return None

    suffix = path.suffix.lower()
    if suffix == ".bin" or data_path.endswith(".pcd.bin"):
        # Binary float32 format: x, y, z, intensity[, ring, time, ...]
        raw = np.fromfile(data_path, dtype=np.float32)
        for ncols in (4, 5, 6, 3):
            if raw.size % ncols == 0:
                pts = raw.reshape(-1, ncols)
                # Always return Nx4 (pad intensity with zeros if only 3 cols)
                if ncols >= 4:
                    return pts[:, :4]
                else:
                    return np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
        print(f"  WARNING: Cannot determine point cloud stride for size {raw.size}: {data_path}")
        return None
    elif suffix == ".pcd":
        return _load_pcd(data_path)
    else:
        print(f"  WARNING: Unrecognized point cloud format: {data_path}")
        return None


def _load_pcd(filepath: str) -> Optional[np.ndarray]:
    """Minimal PCD (ASCII / binary) loader returning Nx4 array."""
    import struct

    header = {}
    fields = []
    size = []
    type_ = []
    count = []
    data_type = "ascii"
    header_end_byte = 0

    with open(filepath, "rb") as f:
        raw = f.read()

    lines = raw.split(b"\n")
    header_lines = []
    data_start_line = 0
    for i, line in enumerate(lines):
        decoded = line.decode("utf-8", errors="replace").strip()
        header_lines.append(decoded)
        if decoded.startswith("FIELDS"):
            fields = decoded.split()[1:]
        elif decoded.startswith("SIZE"):
            size = [int(v) for v in decoded.split()[1:]]
        elif decoded.startswith("TYPE"):
            type_ = decoded.split()[1:]
        elif decoded.startswith("COUNT"):
            count = [int(v) for v in decoded.split()[1:]]
        elif decoded.startswith("DATA"):
            data_type = decoded.split()[1]
            data_start_line = i + 1
            break

    # Byte offset to data
    offset = sum(len(l) + 1 for l in lines[:data_start_line])

    try:
        xi = fields.index("x")
        yi = fields.index("y")
        zi = fields.index("z")
        ii = fields.index("intensity") if "intensity" in fields else None
    except ValueError:
        return None

    if data_type == "ascii":
        pts_text = b"\n".join(lines[data_start_line:])
        arr = np.frombuffer(pts_text, dtype=np.uint8)
        # Parse float from text
        from io import StringIO
        pts = np.loadtxt(StringIO(pts_text.decode("utf-8", errors="replace")))
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]
        x = pts[:, xi]
        y = pts[:, yi]
        z = pts[:, zi]
        intensity = pts[:, ii] if ii is not None else np.zeros(len(x))

    elif data_type == "binary":
        row_size = sum(s * c for s, c in zip(size, count))
        data_bytes = raw[offset:]
        n_pts = len(data_bytes) // row_size
        pts_bytes = np.frombuffer(data_bytes[: n_pts * row_size], dtype=np.uint8)

        # Build structured dtype
        dt_map = {"F": "f", "I": "i", "U": "u"}
        dtype_fields = []
        for f_name, s, t, c in zip(fields, size, type_, count):
            np_type = f"{dt_map.get(t, 'f')}{s}"
            if c == 1:
                dtype_fields.append((f_name, np_type))
            else:
                dtype_fields.append((f_name, np_type, (c,)))
        dt = np.dtype(dtype_fields)
        structured = np.frombuffer(data_bytes[: n_pts * row_size], dtype=dt)

        x = structured["x"].astype(np.float32)
        y = structured["y"].astype(np.float32)
        z = structured["z"].astype(np.float32)
        intensity = (
            structured["intensity"].astype(np.float32)
            if ii is not None
            else np.zeros(len(x), dtype=np.float32)
        )
    else:
        print(f"  WARNING: PCD data type '{data_type}' not fully supported.")
        return None

    return np.column_stack([x, y, z, intensity])


# ---------------------------------------------------------------------------
# Rerun visualization (interactive)
# ---------------------------------------------------------------------------

def visualize_rerun(
    t4,
    sample,
    cameras: Optional[List[str]] = None,
    show_annotations: bool = True,
    save_dir: Optional[str] = None,
) -> None:
    """Use t4-devkit's built-in Rerun integration for interactive visualization.

    This renders the full scene via Rerun and then jumps to the closest
    timestamp. For Rerun, the native t4.render_scene() is the easiest path.

    Args:
        t4: Tier4 instance.
        sample: Closest Sample to the target timestamp.
        cameras: (unused — Rerun renders all cameras)
        show_annotations: (unused — Rerun always shows annotations)
        save_dir: Directory to save Rerun recording (.rrd) file.
    """
    ts_us = sample.timestamp
    ts_sec = ts_us / 1e6
    print(f"[visualize_rerun] Launching Rerun for timestamp {ts_us} µs ({ts_sec:.3f} s)")
    print(f"  Sample token: {sample.token}")
    print("  NOTE: Rerun will render the full scene; use the timeline to jump to the sample.")

    t4.render_scene(save_dir=save_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize T4 dataset images and point clouds at a specific timestamp.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset_path",
        help="Path to T4 dataset root directory (the UUID/scene directory).",
    )
    parser.add_argument(
        "timestamp_us",
        type=float,
        help=(
            "Target timestamp in microseconds (Unix time). "
            "The closest sample will be selected. "
            "Tip: use 0 to visualize the first sample."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="Use Rerun for interactive 3D visualization (requires t4-devkit with Rerun).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="Save visualization outputs to this directory instead of displaying interactively.",
    )
    parser.add_argument(
        "--version",
        default=None,
        metavar="VERSION",
        help="Dataset annotation version directory name (default: auto-detect latest).",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        metavar="CAM1,CAM2",
        help="Comma-separated camera channel names to display (default: all cameras).",
    )
    parser.add_argument(
        "--no-annotations",
        dest="show_annotations",
        action="store_false",
        default=True,
        help="Disable bounding box overlays.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output from Tier4 loader.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Import t4-devkit (must be installed)
    # ------------------------------------------------------------------
    try:
        from t4_devkit import Tier4
    except ImportError:
        print(
            "ERROR: t4_devkit is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/tier4/t4-devkit.git\n"
            "or clone and install locally:\n"
            "  git clone https://github.com/tier4/t4-devkit && cd t4-devkit && pip install -e ."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"Loading T4 dataset from: {dataset_path}")
    kwargs = {"verbose": args.verbose}
    if args.version:
        kwargs["version"] = args.version

    try:
        from t4_visualizer.downloader import find_t4_root, patch_missing_t4_tables
        t4_root = find_t4_root(dataset_path)
        if t4_root != dataset_path:
            print(f"Nested layout detected, using T4 root: {t4_root}")
        patch_missing_t4_tables(t4_root)
        t4 = Tier4(str(t4_root), **kwargs)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Find closest sample
    # ------------------------------------------------------------------
    timestamp_us = int(args.timestamp_us)
    print(f"Target timestamp: {timestamp_us} µs ({timestamp_us / 1e6:.6f} s)")

    sample = find_closest_sample(t4, timestamp_us)
    if sample is None:
        print("ERROR: No samples found in the dataset.")
        sys.exit(1)

    delta_ms = abs(sample.timestamp - timestamp_us) / 1e3
    print(
        f"Closest sample: token={sample.token}\n"
        f"  sample timestamp: {sample.timestamp} µs\n"
        f"  time delta: {delta_ms:.1f} ms"
    )

    # ------------------------------------------------------------------
    # Parse camera list
    # ------------------------------------------------------------------
    cameras = None
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

    # ------------------------------------------------------------------
    # Print available channels for this sample
    # ------------------------------------------------------------------
    all_cameras = list_camera_channels(t4, sample)
    all_lidars = list_lidar_channels(t4, sample)
    print(f"Available cameras : {all_cameras}")
    print(f"Available LiDAR   : {all_lidars}")

    # ------------------------------------------------------------------
    # Visualize
    # ------------------------------------------------------------------
    if args.rerun:
        visualize_rerun(
            t4,
            sample,
            cameras=cameras,
            show_annotations=args.show_annotations,
            save_dir=args.save_dir,
        )
    else:
        visualize_static(
            t4,
            sample,
            cameras=cameras,
            show_annotations=args.show_annotations,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
