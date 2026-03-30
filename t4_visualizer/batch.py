"""Batch visualization pipeline for T4 dataset scenes.

Reads a CSV or Parquet file whose rows each identify one annotated object,
groups them into unique frames, downloads the required datasets (with user
confirmation), and produces camera-image + LiDAR-BEV plots for every frame.

Required columns in the input file
------------------------------------
- ``t4dataset_id``   : dataset identifier used to download / locate the data
- ``scenario_name``  : name of the scene within the dataset
- ``frame_index``    : 0-based frame index within the scene

Optional columns (used when present)
--------------------------------------
- ``t4dataset_name`` : human-readable dataset name (shown in download prompt)
- ``status``         : group label (e.g. "degrade", "improved") — images are
                       placed in a sub-directory named after this value; rows
                       without a status (or with NaN) go into "unknown"
- ``cameras``        : comma-separated camera channel names to render
- ``description``    : free-text label added to plot titles
- ``label``          : string label for the target object (e.g. "car", "pedestrian");
                       when present, it is prepended to the output filename so that
                       results can be distinguished at a glance

One row = one annotated object.  Rows are grouped by
``(t4dataset_id, scenario_name, frame_index)`` to produce one visualization
per unique frame.

Output structure
-----------------
    <output_dir>/<status>/<label>_<t4dataset_id>_<scenario_name>_f<frame_index>_cameras.png
    <output_dir>/<status>/<label>_<t4dataset_id>_<scenario_name>_f<frame_index>_pointcloud.png
    <output_dir>/<status>/<label>_<t4dataset_id>_<scenario_name>_f<frame_index>_meta.txt

``<label>_`` is omitted when the ``label`` column is absent or empty.

Storage modes
--------------
``--temp``       Download datasets into a system temp directory that is
                 automatically deleted when the process exits.
``--data-dir``   Download / read datasets from a persistent directory (default:
                 ``./t4datasets/``). Already-downloaded datasets are reused.

Usage examples
--------------
    # Interactive confirmation prompt (default)
    python -m t4_visualizer.batch results.csv --output-dir out/

    # Skip confirmation (useful in scripts)
    python -m t4_visualizer.batch results.csv --output-dir out/ --yes

    # Temporary storage — data vanishes after the script finishes
    python -m t4_visualizer.batch results.csv --temp --output-dir out/ --yes

    # Custom data dir, parallel workers
    python -m t4_visualizer.batch results.csv --data-dir /mnt/t4data -o out/ -j 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from t4_visualizer.visualize import TargetObject, VisualizationRequest, render_frame


# ---------------------------------------------------------------------------
# Frame key type  (t4dataset_id, scenario_name, frame_index)
# ---------------------------------------------------------------------------

FrameKey = Tuple[str, str, int]


# ---------------------------------------------------------------------------
# Frame dataclass  (one unique frame to visualize)
# ---------------------------------------------------------------------------

@dataclass
class FrameRow:
    """One unique (dataset, scene, frame) combination to visualize."""
    t4dataset_id: str
    t4dataset_name: str          # "" when not available
    scenario_name: str
    frame_index: int
    status: Optional[str] = None
    cameras: Optional[List[str]] = field(default=None)
    description: str = ""
    target_objects: List[TargetObject] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class RowResult:
    frame: FrameRow
    success: bool
    output_dir: Optional[Path] = None
    error: str = ""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"t4dataset_id", "scenario_name", "frame_index"}


def load_input(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet file and validate required columns."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df["frame_index"] = df["frame_index"].astype("int64")
    return df


def df_to_frames(df: pd.DataFrame) -> List[FrameRow]:
    """Deduplicate rows into unique (t4dataset_id, scenario_name, frame_index) frames.

    When multiple rows share the same frame key (i.e. different annotated
    objects in the same frame), the first row's metadata (status, cameras,
    description) is used for the frame.
    """
    seen: Dict[FrameKey, FrameRow] = {}

    has_uuid = "uuid" in df.columns
    has_x = "x" in df.columns
    has_y = "y" in df.columns
    has_z = "z" in df.columns
    has_label = "label" in df.columns
    has_width = "width" in df.columns
    has_length = "length" in df.columns
    has_height = "height" in df.columns
    has_yaw = "yaw" in df.columns

    for _, r in df.iterrows():
        key: FrameKey = (
            str(r["t4dataset_id"]),
            str(r["scenario_name"]),
            int(r["frame_index"]),
        )
        if key not in seen:
            cameras: Optional[List[str]] = None
            if "cameras" in df.columns and pd.notna(r.get("cameras")):
                raw = str(r["cameras"]).strip()
                if raw:
                    cameras = [c.strip() for c in raw.split(",") if c.strip()]

            status: Optional[str] = None
            if "status" in df.columns:
                raw_status = r.get("status")
                if pd.notna(raw_status) and str(raw_status).strip():
                    status = str(raw_status).strip()

            t4dataset_name = ""
            if "t4dataset_name" in df.columns and pd.notna(r.get("t4dataset_name")):
                t4dataset_name = str(r["t4dataset_name"]).strip()

            description = ""
            if "description" in df.columns and pd.notna(r.get("description")):
                description = str(r["description"])

            seen[key] = FrameRow(
                t4dataset_id=key[0],
                t4dataset_name=t4dataset_name,
                scenario_name=key[1],
                frame_index=key[2],
                status=status,
                cameras=cameras,
                description=description,
            )

        # Collect target object for every row (multiple objects per frame supported)
        if has_uuid:
            raw_uuid = r.get("uuid")
            if pd.notna(raw_uuid) and str(raw_uuid).strip():
                seen[key].target_objects.append(TargetObject(
                    uuid=str(raw_uuid).strip(),
                    x=float(r["x"]) if has_x and pd.notna(r.get("x")) else 0.0,
                    y=float(r["y"]) if has_y and pd.notna(r.get("y")) else 0.0,
                    z=float(r["z"]) if has_z and pd.notna(r.get("z")) else 0.0,
                    label=str(r["label"]) if has_label and pd.notna(r.get("label")) else "",
                    width=float(r["width"]) if has_width and pd.notna(r.get("width")) else 0.0,
                    length=float(r["length"]) if has_length and pd.notna(r.get("length")) else 0.0,
                    height=float(r["height"]) if has_height and pd.notna(r.get("height")) else 0.0,
                    yaw=float(r["yaw"]) if has_yaw and pd.notna(r.get("yaw")) else 0.0,
                ))

    return list(seen.values())


# ---------------------------------------------------------------------------
# Download management
# ---------------------------------------------------------------------------

def find_dataset_in_dir(data_dir: Path, t4dataset_id: str, search_depth: int = 1) -> Optional[Path]:
    """Return the path to *t4dataset_id* under *data_dir*, or None if not found.

    Resolution order:

    1. Flat layout: ``data_dir/<id>/``
    2. Grouped webauto layout: ``data_dir/<group>/<id>/<version>/`` where *group*
       is a top-level folder (e.g. ``annotation_dataset``, ``j6gen6_3``, …). The
       T4 root is the *version* directory (often named ``0``).
    3. If *search_depth* >= 1: nested folders ``data_dir/*/<id>``, ``data_dir/*/*/<id>``,
       … up to *search_depth* intermediate directory levels (grouping folders).
    """
    from t4_visualizer.downloader import _find_webauto_nested

    data_dir = Path(data_dir)
    # Always check the flat layout first.
    flat = data_dir / t4dataset_id
    if flat.exists():
        return flat

    webauto = _find_webauto_nested(data_dir, t4dataset_id)
    if webauto is not None:
        return webauto

    if search_depth < 1:
        return None

    def _search_under(prefix: Path, levels_left: int) -> Optional[Path]:
        if levels_left < 1:
            return None
        try:
            entries = sorted(prefix.iterdir())
        except OSError:
            return None
        for subdir in entries:
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            candidate = subdir / t4dataset_id
            if candidate.exists():
                return candidate
            found = _search_under(subdir, levels_left - 1)
            if found is not None:
                return found
        return None

    return _search_under(data_dir, search_depth)


def _unique_datasets(frames: List[FrameRow]) -> List[FrameRow]:
    """Return one representative FrameRow per unique t4dataset_id (for prompts)."""
    seen: Dict[str, FrameRow] = {}
    for f in frames:
        if f.t4dataset_id not in seen:
            seen[f.t4dataset_id] = f
    return list(seen.values())


def confirm_downloads(
    frames: List[FrameRow],
    data_dir: Optional[Path] = None,
    search_depth: int = 1,
) -> bool:
    """Show datasets to be downloaded and ask the user for confirmation.

    If *data_dir* is given, already-present datasets are listed separately so
    the user can make an informed go/no-go decision.

    Returns True if the user approves (or nothing needs downloading), False otherwise.
    """
    unique = _unique_datasets(frames)
    total = len(unique)

    present: List[FrameRow] = []
    missing: List[FrameRow] = []
    if data_dir is not None and data_dir.exists():
        for f in unique:
            if find_dataset_in_dir(data_dir, f.t4dataset_id, search_depth) is not None:
                present.append(f)
            else:
                missing.append(f)
    else:
        missing = list(unique)

    print(f"\nDatasets required for visualization: {total}")

    if present:
        print(f"\n  Already in {data_dir}  ({len(present)}):")
        for f in present:
            label = f.t4dataset_name if f.t4dataset_name else f.t4dataset_id
            print(f"    [ok] {label}  (id: {f.t4dataset_id})")

    if missing:
        print(f"\n  Need to download  ({len(missing)}):")
        for f in missing:
            label = f.t4dataset_name if f.t4dataset_name else f.t4dataset_id
            print(f"    [dl] {label}  (id: {f.t4dataset_id})")

    print(f"\n  Summary: {total} total / {len(present)} present / {len(missing)} to download\n")

    if not missing:
        print("  All datasets already present — skipping download prompt.")
        return True

    try:
        answer = input("Proceed with download? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    return answer in ("y", "yes")


def resolve_dataset_path(
    t4dataset_id: str,
    data_dir: Path,
    do_download: bool,
    cache_limit: int = 0,
) -> Path:
    """Return the local path to a dataset, downloading it first if needed.

    When *cache_limit* > 0 a :class:`DatasetCache` is used so that the
    least-recently-used datasets are evicted once the limit is reached.
    """
    from t4_visualizer.downloader import DatasetCache, DownloadError, dataset_is_cached

    if do_download:
        cache = DatasetCache(data_dir, max_cached=cache_limit)
        return cache.ensure(t4dataset_id)
    else:
        path = data_dir / t4dataset_id
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path} and --no-download was specified."
            )
        # Still update the access timestamp so the cache knows it was used
        DatasetCache(data_dir, max_cached=0).touch(t4dataset_id)
        return path


# ---------------------------------------------------------------------------
# Single-frame visualization
# ---------------------------------------------------------------------------

def _status_dir(output_root: Path, status: Optional[str], has_status_column: bool) -> Path:
    if not has_status_column:
        return output_root
    return output_root / (status if status else "unknown")


def _filename_prefix(frame: FrameRow) -> str:
    """Build flat filename prefix.

    Format: <label>_<t4dataset_id>_<scenario_name>_f<frame_index>
    The <label>_ part is omitted when no target object carries a label.
    """
    safe = lambda s: str(s).replace("/", "-").replace(" ", "_")
    base = f"{safe(frame.t4dataset_id)}_{safe(frame.scenario_name)}_f{frame.frame_index:06d}"
    obj_label = next(
        (obj.label for obj in frame.target_objects if obj.label),
        "",
    )
    if obj_label:
        return f"{safe(obj_label)}_{base}"
    return base


def visualize_frame(
    frame: FrameRow,
    dataset_path: Path,
    output_root: Path,
    show_annotations: bool = True,
    version: Optional[str] = None,
    has_status_column: bool = False,
    crop_cameras: bool = False,
    crop_padding: int = 40,
    crop_min_size: int = 300,
) -> Path:
    """Visualize one frame. Returns the output directory used."""
    try:
        from t4_devkit import Tier4
    except ImportError:
        raise ImportError(
            "t4_devkit is not installed. "
            "Install with: pip install git+https://github.com/tier4/t4-devkit.git"
        )

    from t4_visualizer.visualize import (
        find_sample_by_scene_and_index,
        list_camera_channels,
        list_lidar_channels,
    )

    frame_out = _status_dir(output_root, frame.status, has_status_column)
    frame_out.mkdir(parents=True, exist_ok=True)
    prefix = _filename_prefix(frame)

    from t4_visualizer.downloader import find_t4_root, prepare_dataset_root, patch_missing_t4_tables
    t4_root = find_t4_root(dataset_path)
    if t4_root != dataset_path:
        print(f"  [batch] Nested layout detected, using T4 root: {t4_root}")
    t4_root = prepare_dataset_root(t4_root)
    patch_missing_t4_tables(t4_root)

    kwargs = {}
    if version:
        kwargs["version"] = version
    t4 = Tier4(str(t4_root), **kwargs)

    sample = find_sample_by_scene_and_index(t4, frame.scenario_name, frame.frame_index)

    print(
        f"  [{frame.scenario_name}] frame={frame.frame_index} "
        f"sample={sample.token}  ts={sample.timestamp}"
    )

    # Write metadata sidecar
    meta_path = frame_out / f"{prefix}_meta.txt"
    with open(meta_path, "w") as fh:
        fh.write(
            f"t4dataset_id  : {frame.t4dataset_id}\n"
            f"t4dataset_name: {frame.t4dataset_name}\n"
            f"scenario_name : {frame.scenario_name}\n"
            f"frame_index   : {frame.frame_index}\n"
            f"status        : {frame.status}\n"
            f"sample_token  : {sample.token}\n"
            f"sample_ts_us  : {sample.timestamp}\n"
            f"description   : {frame.description}\n"
            f"cameras       : {list_camera_channels(t4, sample)}\n"
            f"lidars        : {list_lidar_channels(t4, sample)}\n"
        )

    # Render via the programmatic API, passing the already-loaded Tier4
    # instance to avoid reloading the dataset.
    request = VisualizationRequest(
        dataset_path=t4_root,
        scenario_name=frame.scenario_name,
        frame_index=frame.frame_index,
        target_objects=frame.target_objects,
        cameras=frame.cameras,
        show_annotations=show_annotations,
        version=version,
        crop_cameras=crop_cameras,
        crop_padding=crop_padding,
        crop_min_size=crop_min_size,
    )
    result = render_frame(request, t4=t4)

    for img in result.images:
        if img.label == "combined":
            filename = f"{prefix}_visualization.png"
        else:
            filename = f"{prefix}_{img.label}_visualization_crop.png"
        out_path = frame_out / filename
        out_path.write_bytes(img.data)
        print(f"  Saved: {out_path}")

    return frame_out


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

@dataclass
class BatchConfig:
    input_path: str
    output_dir: Path
    data_dir: Optional[Path]
    use_temp: bool
    do_download: bool
    yes: bool                  # skip confirmation prompt
    show_annotations: bool
    version: Optional[str]
    workers: int
    fail_fast: bool
    crop_cameras: bool = False
    crop_padding: int = 40
    crop_min_size: int = 300
    cache_limit: int = 10


def run_batch(cfg: BatchConfig) -> List[RowResult]:
    """Main batch pipeline. Returns list of per-frame results."""
    print(f"Loading input: {cfg.input_path}")
    df = load_input(cfg.input_path)
    has_status_column = "status" in df.columns
    frames = df_to_frames(df)
    print(f"  {len(df)} rows → {len(frames)} unique frame(s) to visualize.")
    if has_status_column:
        counts = (
            df["status"].fillna("unknown").value_counts().to_dict()
        )
        print(f"  Status groups: {counts}")

    unique_ids = list(dict.fromkeys(f.t4dataset_id for f in frames))
    print(f"  {len(unique_ids)} unique dataset(s): {unique_ids}")

    # ----------------------------------------------------------------
    # User confirmation before downloading
    # ----------------------------------------------------------------
    if cfg.do_download:
        if cfg.yes:
            print("  --yes flag set, skipping confirmation.")
        else:
            approved = confirm_downloads(frames)
            if not approved:
                print("Download cancelled by user.")
                sys.exit(0)

    # ----------------------------------------------------------------
    # Setup data directory
    # ----------------------------------------------------------------
    _tmp_ctx = None
    if cfg.use_temp:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix="t4batch_")
        data_dir = Path(_tmp_ctx.name)
        print(f"  Using TEMP directory: {data_dir}")
    else:
        data_dir = cfg.data_dir or Path("t4datasets")
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Using persistent data directory: {data_dir.resolve()}")

    results: List[RowResult] = []

    try:
        # Download all required datasets first (sequential to avoid conflicts)
        dataset_paths: Dict[str, Path] = {}
        for did in unique_ids:
            try:
                path = resolve_dataset_path(did, data_dir, cfg.do_download,
                                            cache_limit=cfg.cache_limit)
                dataset_paths[did] = path
                print(f"  Dataset ready: {did} → {path}")
            except Exception as e:
                print(f"  ERROR downloading {did}: {e}")
                for fr in frames:
                    if fr.t4dataset_id == did:
                        results.append(RowResult(frame=fr, success=False, error=str(e)))
                if cfg.fail_fast:
                    raise

        ok_frames = [f for f in frames if f.t4dataset_id in dataset_paths]
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        def _process(frame: FrameRow) -> RowResult:
            dataset_path = dataset_paths[frame.t4dataset_id]
            try:
                out = visualize_frame(
                    frame,
                    dataset_path,
                    cfg.output_dir,
                    show_annotations=cfg.show_annotations,
                    version=cfg.version,
                    has_status_column=has_status_column,
                    crop_cameras=cfg.crop_cameras,
                    crop_padding=cfg.crop_padding,
                    crop_min_size=cfg.crop_min_size,
                )
                return RowResult(frame=frame, success=True, output_dir=out)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"  ERROR [{frame.scenario_name} / frame {frame.frame_index}]: {e}\n{tb}")
                return RowResult(frame=frame, success=False, error=str(e))

        if cfg.workers <= 1:
            for frame in ok_frames:
                print(
                    f"\n--- [{frame.t4dataset_id}] {frame.scenario_name} "
                    f"frame={frame.frame_index} ---"
                )
                result = _process(frame)
                results.append(result)
                if not result.success and cfg.fail_fast:
                    raise RuntimeError(result.error)
        else:
            print(f"\nProcessing {len(ok_frames)} frame(s) with {cfg.workers} workers...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                futures = {pool.submit(_process, frame): frame for frame in ok_frames}
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    status = "OK" if result.success else "FAIL"
                    print(
                        f"  [{status}] {result.frame.scenario_name} "
                        f"frame={result.frame.frame_index}"
                        + (f" → {result.output_dir}" if result.success else f": {result.error}")
                    )
                    if not result.success and cfg.fail_fast:
                        pool.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError(result.error)

    finally:
        if _tmp_ctx is not None:
            print(f"\nCleaning up temp directory: {_tmp_ctx.name}")
            _tmp_ctx.cleanup()

    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(results: List[RowResult], output_dir: Path) -> None:
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    print(f"\n{'='*60}")
    print(f"  Batch complete: {len(ok)} succeeded / {len(fail)} failed")
    print(f"  Output directory: {output_dir.resolve()}")
    print(f"{'='*60}")

    if fail:
        # Deduplicate by t4dataset_id for a concise dataset-level view
        failed_datasets: dict[str, list[str]] = {}
        for r in fail:
            did = r.frame.t4dataset_id
            entry = f"scenario={r.frame.scenario_name} frame={r.frame.frame_index}: {r.error}"
            failed_datasets.setdefault(did, []).append(entry)

        print(f"\nFailed datasets ({len(failed_datasets)} unique):")
        for did, entries in failed_datasets.items():
            print(f"  {did}")
            for e in entries:
                print(f"    {e}")

        # Write machine-readable list for easy re-runs / debugging
        failed_path = output_dir / "failed_datasets.txt"
        with open(failed_path, "w") as fh:
            for did, entries in failed_datasets.items():
                fh.write(f"{did}\n")
                for e in entries:
                    fh.write(f"  {e}\n")
                fh.write("\n")
        print(f"\n  Failed datasets list: {failed_path}")

    summary_path = output_dir / "batch_summary.csv"
    summary_df = pd.DataFrame([
        {
            "t4dataset_id": r.frame.t4dataset_id,
            "t4dataset_name": r.frame.t4dataset_name,
            "scenario_name": r.frame.scenario_name,
            "frame_index": r.frame.frame_index,
            "status": r.frame.status,
            "success": r.success,
            "output_dir": str(r.output_dir) if r.output_dir else "",
            "filename_prefix": _filename_prefix(r.frame) if r.success else "",
            "error": r.error,
        }
        for r in results
    ])
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary CSV: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch T4 dataset visualization pipeline.\n"
            "Reads a CSV/Parquet with columns "
            "[t4dataset_id, scenario_name, frame_index] "
            "and produces camera + LiDAR plots for each unique frame."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        help="CSV or Parquet file with columns: t4dataset_id, scenario_name, frame_index.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="batch_output",
        metavar="DIR",
        help="Root directory for visualization outputs (default: ./batch_output/).",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip the download confirmation prompt (non-interactive / scripted use).",
    )

    storage = parser.add_mutually_exclusive_group()
    storage.add_argument(
        "--temp",
        action="store_true",
        default=False,
        help="Download into a temporary directory that is deleted on exit.",
    )
    storage.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        help=(
            "Persistent directory for downloaded datasets (default: ./t4datasets/). "
            "Already-present datasets are reused without re-downloading."
        ),
    )

    parser.add_argument(
        "--no-download",
        action="store_true",
        default=False,
        help="Skip downloading; assume datasets already exist under --data-dir.",
    )
    parser.add_argument(
        "--no-annotations",
        dest="show_annotations",
        action="store_false",
        default=True,
        help="Disable bounding box overlays on images and point clouds.",
    )
    parser.add_argument(
        "--version",
        default=None,
        metavar="VERSION",
        help="Dataset annotation version directory name (default: auto-detect).",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel visualization workers (default: 1).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Abort the batch on the first error instead of continuing.",
    )
    parser.add_argument(
        "--cache-limit",
        type=int,
        default=10,
        metavar="N",
        dest="cache_limit",
        help=(
            "Maximum number of datasets to keep on disk (LRU eviction). "
            "0 disables eviction (default: 10)."
        ),
    )
    parser.add_argument(
        "--crop-view",
        action="store_true",
        default=False,
        dest="crop_cameras",
        help=(
            "Replace the full camera grid with a single cropped view of the "
            "camera showing the largest projected BBOX ROI. "
            "Output file: {prefix}_visualization_crop.png."
        ),
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=40,
        metavar="PX",
        help="Pixel padding around the ROI crop (default: 40).",
    )
    parser.add_argument(
        "--crop-min-size",
        type=int,
        default=300,
        metavar="PX",
        help="Minimum crop dimension in pixels (default: 300).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = BatchConfig(
        input_path=args.input,
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        use_temp=args.temp,
        do_download=not args.no_download,
        yes=args.yes,
        show_annotations=args.show_annotations,
        version=args.version,
        workers=args.workers,
        fail_fast=args.fail_fast,
        crop_cameras=args.crop_cameras,
        crop_padding=args.crop_padding,
        crop_min_size=args.crop_min_size,
        cache_limit=args.cache_limit,
    )

    results = run_batch(cfg)
    print_summary(results, cfg.output_dir)

    n_fail = sum(1 for r in results if not r.success)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    import sys as _sys
    if "--multi" in _sys.argv:
        _sys.argv.remove("--multi")
        multi_main()
    else:
        main()


# ===========================================================================
# Multi-CSV batch
# ===========================================================================

@dataclass
class RunSpec:
    """One CSV input with its output label."""
    label: str       # output subfolder name, e.g. "improve"
    input_path: str  # path to CSV / Parquet file


@dataclass
class MultiRunConfig:
    """Configuration for a multi-CSV batch run."""
    runs: List[RunSpec]
    output_dir: Path
    data_dir: Optional[Path]
    use_temp: bool
    do_download: bool
    yes: bool
    show_annotations: bool
    version: Optional[str]
    workers: int
    fail_fast: bool
    crop_cameras: bool = False
    crop_padding: int = 40
    crop_min_size: int = 300
    cache_limit: int = 10
    search_depth: int = 1


def _parse_run_spec(s: str) -> RunSpec:
    """Parse ``'label:path'`` or ``'path'`` (label = file stem)."""
    if ":" in s:
        label, _, path = s.partition(":")
        return RunSpec(label=label.strip(), input_path=path.strip())
    return RunSpec(label=Path(s).stem, input_path=s)


def multi_run(cfg: MultiRunConfig) -> Dict[str, List[RowResult]]:
    """Three-phase multi-CSV pipeline.

    Phase 1 — Load all CSVs and collect the union of required dataset IDs.
    Phase 2 — Download all datasets once (``DatasetCache.ensure_many``).
              Non-needed LRU entries are evicted first so the needed set is
              never sacrificed mid-run.
    Phase 3 — Visualize each CSV independently, routing output to
              ``output_dir / label /``.
    """
    # ------------------------------------------------------------------
    # Validate labels
    # ------------------------------------------------------------------
    labels = [spec.label for spec in cfg.runs]
    dupes = {l for l in labels if labels.count(l) > 1}
    if dupes:
        raise ValueError(f"Duplicate run labels: {sorted(dupes)}")

    # ------------------------------------------------------------------
    # Phase 1: load all CSVs
    # ------------------------------------------------------------------
    all_run_data: List[tuple] = []   # (label, frames, has_status)
    ids_ordered: List[str] = []      # unique IDs, insertion order
    seen_ids: set = set()

    for spec in cfg.runs:
        print(f"\nLoading [{spec.label}]: {spec.input_path}")
        df = load_input(spec.input_path)
        has_status = "status" in df.columns
        frames = df_to_frames(df)
        print(f"  {len(df)} rows → {len(frames)} frame(s)")
        all_run_data.append((spec.label, frames, has_status))
        for f in frames:
            if f.t4dataset_id not in seen_ids:
                ids_ordered.append(f.t4dataset_id)
                seen_ids.add(f.t4dataset_id)

    print(f"\n{len(ids_ordered)} unique dataset(s) across all runs:")
    for did in ids_ordered:
        print(f"  {did}")

    # ------------------------------------------------------------------
    # Phase 2: setup data dir + smart download
    # ------------------------------------------------------------------
    _tmp_ctx = None
    if cfg.use_temp:
        _tmp_ctx = tempfile.TemporaryDirectory(prefix="t4multi_")
        data_dir = Path(_tmp_ctx.name)
        print(f"\n  Using TEMP directory: {data_dir}")
    else:
        data_dir = cfg.data_dir or Path("t4datasets")
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Data directory: {data_dir.resolve()}")

    # ------------------------------------------------------------------
    # Confirmation (after data_dir is known so we can show present/missing)
    # ------------------------------------------------------------------
    if cfg.do_download and not cfg.yes:
        all_frames = [f for _, frames, _ in all_run_data for f in frames]
        # For temp dirs nothing is pre-existing, so pass None to skip the check.
        preview_dir = None if cfg.use_temp else data_dir
        if not confirm_downloads(all_frames, data_dir=preview_dir, search_depth=cfg.search_depth):
            print("Download cancelled by user.")
            sys.exit(0)

    try:
        from t4_visualizer.downloader import DatasetCache
        cache = DatasetCache(data_dir, cfg.cache_limit)
        dataset_paths: Dict[str, Path] = {}

        if cfg.do_download:
            try:
                dataset_paths = cache.ensure_many(ids_ordered)
            except Exception as e:
                print(f"  ERROR during batch download: {e}")
                if cfg.fail_fast:
                    raise
                # Fallback: try each ID individually
                for did in ids_ordered:
                    try:
                        dataset_paths[did] = cache.ensure(did)
                    except Exception as e2:
                        print(f"  ERROR downloading {did}: {e2}")
        else:
            found = {did: find_dataset_in_dir(data_dir, did, cfg.search_depth) for did in ids_ordered}
            present_ids = [did for did, p in found.items() if p is not None]
            missing_ids = [did for did, p in found.items() if p is None]
            print(
                f"\n  Dataset summary (--no-download): "
                f"{len(ids_ordered)} total / "
                f"{len(present_ids)} present / "
                f"{len(missing_ids)} missing"
            )
            for did in ids_ordered:
                path = found[did]
                if path is not None:
                    cache.touch(did)
                    dataset_paths[did] = path
                else:
                    msg = f"Dataset '{did}' not found under {data_dir} (search_depth={cfg.search_depth}, --no-download)"
                    print(f"  ERROR: {msg}")
                    if cfg.fail_fast:
                        raise FileNotFoundError(msg)

        # ------------------------------------------------------------------
        # Phase 3: visualize each run
        # ------------------------------------------------------------------
        all_results: Dict[str, List[RowResult]] = {}

        for label, frames, has_status in all_run_data:
            print(f"\n{'='*60}")
            print(f"  Visualizing [{label}]  ({len(frames)} frame(s))")
            print(f"{'='*60}")

            label_output = cfg.output_dir / label
            label_output.mkdir(parents=True, exist_ok=True)

            ok_frames = [f for f in frames if f.t4dataset_id in dataset_paths]
            skip_frames = [f for f in frames if f.t4dataset_id not in dataset_paths]

            def _process(frame: FrameRow) -> RowResult:
                dataset_path = dataset_paths[frame.t4dataset_id]
                try:
                    out = visualize_frame(
                        frame, dataset_path, label_output,
                        show_annotations=cfg.show_annotations,
                        version=cfg.version,
                        has_status_column=has_status,
                        crop_cameras=cfg.crop_cameras,
                        crop_padding=cfg.crop_padding,
                        crop_min_size=cfg.crop_min_size,
                    )
                    return RowResult(frame=frame, success=True, output_dir=out)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"  ERROR [{frame.scenario_name} / frame {frame.frame_index}]: "
                          f"{e}\n{tb}")
                    if cfg.fail_fast:
                        raise
                    return RowResult(frame=frame, success=False, error=str(e))

            if cfg.workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    results = list(pool.map(_process, ok_frames))
            else:
                results = [_process(f) for f in ok_frames]

            results += [
                RowResult(frame=f, success=False, error="dataset not available")
                for f in skip_frames
            ]
            all_results[label] = results
            print_summary(results, label_output)

        return all_results

    finally:
        if _tmp_ctx:
            _tmp_ctx.cleanup()


# ---------------------------------------------------------------------------
# t4-multi CLI
# ---------------------------------------------------------------------------

def _parse_multi_args():
    import argparse

    parser = argparse.ArgumentParser(
        prog="t4-multi",
        description=(
            "Visualize multiple CSV/Parquet result files in one run.\n"
            "Each CSV is assigned a label that becomes its output subfolder.\n\n"
            "Examples:\n"
            "  t4-multi improve:improve.csv degrade:degrade.csv -o ./viz\n"
            "  t4-multi improve.csv degrade.csv -o ./viz   # label = file stem"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "runs",
        nargs="+",
        metavar="[LABEL:]CSV",
        help=(
            "One or more CSV/Parquet files.  Prefix with 'label:' to set the "
            "output folder name; otherwise the file stem is used."
        ),
    )
    parser.add_argument("-o", "--output-dir", default="./output", metavar="DIR")
    parser.add_argument("--data-dir", default=None, metavar="PATH",
                        help="Persistent dataset cache directory (default: ./t4datasets).")
    parser.add_argument("--no-download", action="store_true", default=False)
    parser.add_argument(
        "--search-depth", type=int, default=1, metavar="N",
        help="How many sub-levels to search for datasets under --data-dir (default: 1).",
    )
    parser.add_argument("--temp", action="store_true", default=False)
    parser.add_argument("-y", "--yes", action="store_true", default=False)
    parser.add_argument("--no-annotations", action="store_false", dest="show_annotations",
                        default=True)
    parser.add_argument("--version", default=None, metavar="VERSION")
    parser.add_argument("--workers", "-j", type=int, default=1, metavar="N")
    parser.add_argument("--fail-fast", action="store_true", default=False)
    parser.add_argument("--cache-limit", type=int, default=10, metavar="N",
                        help="Max datasets on disk (LRU eviction). 0=unlimited.")
    parser.add_argument("--crop-view", action="store_true", default=False,
                        dest="crop_cameras")
    parser.add_argument("--crop-padding", type=int, default=40, metavar="PX")
    parser.add_argument("--crop-min-size", type=int, default=300, metavar="PX")

    return parser.parse_args()


def multi_main():
    """Entry point for the ``t4-multi`` command."""
    args = _parse_multi_args()

    runs = [_parse_run_spec(s) for s in args.runs]

    cfg = MultiRunConfig(
        runs=runs,
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir) if args.data_dir else None,
        use_temp=args.temp,
        do_download=not args.no_download,
        yes=args.yes,
        show_annotations=args.show_annotations,
        version=args.version,
        workers=args.workers,
        fail_fast=args.fail_fast,
        crop_cameras=args.crop_cameras,
        crop_padding=args.crop_padding,
        crop_min_size=args.crop_min_size,
        cache_limit=args.cache_limit,
        search_depth=args.search_depth,
    )

    all_results = multi_run(cfg)

    total_fail = sum(
        1 for results in all_results.values() for r in results if not r.success
    )
    sys.exit(1 if total_fail > 0 else 0)
