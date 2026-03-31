"""T4 dataset downloader using ``webauto data annotation-dataset pull``.

Default download command
------------------------
    webauto data annotation-dataset pull \\
        --project-id <project_id> \\
        --annotation-dataset-id <t4dataset_id> \\
        --asset-dir <dest_dir>/<t4dataset_id>

Configuration (in order of precedence)
---------------------------------------
1. Environment variable ``T4_DOWNLOAD_CMD`` (full shell template):

       T4_DOWNLOAD_CMD="webauto data annotation-dataset pull \\
           --project-id my_proj \\
           --annotation-dataset-id {t4dataset_id} \\
           --asset-dir {dataset_path}"

   Placeholders: ``{t4dataset_id}``, ``{dest_dir}``, ``{dataset_path}``
   (``{dataset_path}`` == ``{dest_dir}/{t4dataset_id}``)

2. Module-level constants ``WEBAUTO_PROJECT_ID`` / ``DOWNLOAD_CMD_TEMPLATE``.

3. Environment variable ``WEBAUTO_PROJECT_ID`` to override the project ID only.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration — edit these to match your environment
# ---------------------------------------------------------------------------

# webauto project ID.  Override with env var WEBAUTO_PROJECT_ID.
WEBAUTO_PROJECT_ID: str = os.environ.get("WEBAUTO_PROJECT_ID", "x2_dev")

# Full shell command template.  Overrides the default webauto command when set.
# Available placeholders: {t4dataset_id}, {dest_dir}, {dataset_path}
DOWNLOAD_CMD_TEMPLATE: Optional[str] = os.environ.get("T4_DOWNLOAD_CMD")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DownloadError(RuntimeError):
    """Raised when a dataset download fails."""


def download_dataset(t4dataset_id: str, dest_dir: Path) -> Path:
    """Download a T4 dataset by its ID into *dest_dir* and return the dataset path.

    The downloaded dataset is expected to end up at::

        dest_dir / <t4dataset_id> /

    Args:
        t4dataset_id: Dataset identifier string (UUID or name).
        dest_dir: Directory where the dataset should be placed.

    Returns:
        Path to the downloaded dataset root (``dest_dir / t4dataset_id``).

    Raises:
        DownloadError: If the download fails or the expected directory is not found.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected_path = dest_dir / t4dataset_id

    # Skip download if already present (flatten nested webauto layout first if needed)
    if expected_path.exists():
        if not _looks_like_t4dataset(expected_path):
            _try_flatten(expected_path, t4dataset_id)
        if _looks_like_t4dataset(expected_path):
            print(f"  [downloader] Dataset already exists, skipping download: {expected_path}")
            return expected_path

    print(f"  [downloader] Downloading {t4dataset_id} → {expected_path}")

    if DOWNLOAD_CMD_TEMPLATE:
        _run_cmd_template(DOWNLOAD_CMD_TEMPLATE, t4dataset_id, dest_dir, expected_path)
    else:
        _download_impl(t4dataset_id, dest_dir, expected_path)

    if not expected_path.exists():
        raise DownloadError(
            f"Download finished but expected path not found: {expected_path}\n"
            "Please check that your download logic places the dataset at "
            f"<dest_dir>/<t4dataset_id>/ (i.e. {expected_path})."
        )
    if not _looks_like_t4dataset(expected_path):
        raise DownloadError(
            f"Downloaded path exists but does not look like a T4 dataset: {expected_path}\n"
            "Expected an 'annotation' or 'v1.0-*' subdirectory with JSON files."
        )

    print(f"  [downloader] Ready: {expected_path}")
    return expected_path


def dataset_is_cached(t4dataset_id: str, dest_dir: Path) -> bool:
    """Return True if the dataset already exists and looks valid."""
    path = Path(dest_dir) / t4dataset_id
    return path.exists() and _looks_like_t4dataset(path)


# ---------------------------------------------------------------------------
# LRU dataset cache
# ---------------------------------------------------------------------------

class DatasetCache:
    """LRU disk cache for downloaded T4 datasets.

    Tracks last-access times in a JSON index file and evicts the least
    recently used dataset directories when the cache exceeds *max_cached*.

    Index file: ``{data_dir}/.cache_index.json``
    Format:     ``{ "<t4dataset_id>": "<ISO-8601 last_accessed UTC>" }``

    A file lock (``{data_dir}/.cache_lock``) serialises concurrent
    updates from parallel batch workers.

    Args:
        data_dir: Directory where datasets are stored.
        max_cached: Maximum number of datasets to keep on disk.
            ``0`` disables eviction entirely.
    """

    _INDEX = ".cache_index.json"
    _LOCK  = ".cache_lock"

    def __init__(self, data_dir: Path, max_cached: int = 10):
        self.data_dir = Path(data_dir)
        self.max_cached = max_cached

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ensure(self, t4dataset_id: str) -> Path:
        """Return the dataset path, downloading it first if necessary.

        Evicts LRU datasets before downloading a new one when the cache
        is at capacity.  Updates the last-accessed timestamp afterwards.
        """
        with self._lock():
            already_cached = self._on_disk(t4dataset_id)
            if not already_cached and self.max_cached > 0:
                self._evict_to(self.max_cached - 1)
            path = download_dataset(t4dataset_id, self.data_dir)
            self._touch(t4dataset_id)
        return path

    def touch(self, t4dataset_id: str) -> None:
        """Record that *t4dataset_id* was accessed right now."""
        with self._lock():
            self._touch(t4dataset_id)

    def ensure_many(self, dataset_ids: List[str]) -> Dict[str, Path]:
        """Download all *dataset_ids* with minimum evictions.

        Unlike calling ``ensure()`` N times, this method first pre-evicts
        LRU entries that are **not** in *dataset_ids* to free space, so
        no needed dataset is ever evicted mid-run.

        Steps:
        1. Deduplicate *dataset_ids* (preserves order).
        2. Under the lock: evict non-needed LRU entries to fit the whole
           set within *max_cached* (best-effort; warns if impossible).
        3. Download each missing dataset outside the lock.
        4. Touch every ID to mark it as recently used.

        Returns a ``{t4dataset_id: Path}`` mapping for every ID.
        """
        dataset_ids = list(dict.fromkeys(dataset_ids))  # dedup, keep order
        needed = set(dataset_ids)

        with self._lock():
            if self.max_cached > 0:
                already = sum(1 for did in needed if self._on_disk(did))
                new_count = len(needed) - already
                # Keep at most (max_cached - new_count) non-needed entries
                # so there is room for all new downloads.
                target = max(0, self.max_cached - new_count)
                self._evict_not_needed(needed, keep=target)

        paths: Dict[str, Path] = {}
        for did in dataset_ids:
            paths[did] = download_dataset(did, self.data_dir)
            with self._lock():
                self._touch(did)
        return paths

    def evict_lru(self, keep: int) -> List[str]:
        """Evict datasets until at most *keep* remain.  Returns evicted IDs."""
        with self._lock():
            return self._evict_to(keep)

    def clear(self) -> List[str]:
        """Delete all cached datasets.  Returns list of deleted IDs."""
        with self._lock():
            return self._evict_to(0)

    def status(self) -> List[dict]:
        """Return cache entries sorted from LRU to MRU.

        Each entry is a dict with keys:
        ``t4dataset_id``, ``last_accessed``, ``size_mb``, ``on_disk``.
        """
        with self._lock():
            index = self._read_index()

        rows = []
        for did, ts in sorted(index.items(), key=lambda x: x[1]):
            path = self.data_dir / did
            on_disk = path.exists()
            size_mb = _dir_size_mb(path) if on_disk else 0.0
            rows.append({
                "t4dataset_id": did,
                "last_accessed": ts,
                "size_mb": round(size_mb, 1),
                "on_disk": on_disk,
                "path": str(path),
            })
        return rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_disk(self, t4dataset_id: str) -> bool:
        path = self.data_dir / t4dataset_id
        return path.exists() and _looks_like_t4dataset(path)

    def _touch(self, t4dataset_id: str) -> None:
        index = self._read_index()
        index[t4dataset_id] = datetime.now(timezone.utc).isoformat()
        self._write_index(index)

    def _evict_to(self, keep: int) -> List[str]:
        """Evict LRU entries until len(on-disk entries) <= keep."""
        index = self._read_index()
        # Sync: keep only entries whose directory actually exists
        on_disk = {k: v for k, v in index.items()
                   if (self.data_dir / k).exists()}
        evicted = []
        while len(on_disk) > keep:
            lru_id = min(on_disk, key=lambda k: on_disk[k])
            ts = on_disk.pop(lru_id)
            index.pop(lru_id, None)
            self._delete(lru_id)
            evicted.append(lru_id)
            print(f"  [cache] Evicted {lru_id}  (last accessed: {ts})")
        self._write_index(index)
        return evicted

    def _evict_not_needed(self, needed: set, keep: int) -> List[str]:
        """Evict LRU entries that are *not* in *needed* until on-disk <= keep.

        If there are not enough evictable (non-needed) entries to reach
        *keep*, a warning is printed and the needed datasets are left alone.
        """
        index = self._read_index()
        on_disk = {k: v for k, v in index.items()
                   if (self.data_dir / k).exists()}
        evictable = {k: v for k, v in on_disk.items() if k not in needed}
        evicted = []
        while len(on_disk) > keep:
            if not evictable:
                over = len(on_disk) - keep
                print(f"  [cache] WARNING: {over} needed dataset(s) exceed "
                      f"cache_limit ({self.max_cached}); keeping them anyway.")
                break
            lru_id = min(evictable, key=lambda k: evictable[k])
            ts = evictable.pop(lru_id)
            on_disk.pop(lru_id)
            index.pop(lru_id, None)
            self._delete(lru_id)
            evicted.append(lru_id)
            print(f"  [cache] Evicted (pre-run) {lru_id}  (last accessed: {ts})")
        self._write_index(index)
        return evicted

    def _delete(self, t4dataset_id: str) -> None:
        path = self.data_dir / t4dataset_id
        if path.exists():
            shutil.rmtree(path)

    def _read_index(self) -> dict:
        index_path = self.data_dir / self._INDEX
        if not index_path.exists():
            return {}
        try:
            return json.loads(index_path.read_text())
        except Exception:
            return {}

    def _write_index(self, index: dict) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / self._INDEX).write_text(
            json.dumps(index, indent=2, sort_keys=True)
        )

    @contextmanager
    def _lock(self):
        """Exclusive file lock — serialises index access across processes."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.data_dir / self._LOCK
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# t4-cache CLI entry point
# ---------------------------------------------------------------------------

def cache_main() -> None:
    """Entry point for the ``t4-cache`` command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="t4-cache",
        description="Manage the local T4 dataset cache.",
    )
    parser.add_argument(
        "--data-dir",
        default="t4datasets",
        metavar="PATH",
        help="Dataset cache directory (default: ./t4datasets).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Cache size limit used for 'evict' (default: 10).",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show cached datasets (LRU first).")

    evict_p = sub.add_parser("evict", help="Evict LRU datasets until --limit remain.")
    evict_p.add_argument(
        "--keep",
        type=int,
        default=None,
        metavar="N",
        help="Keep this many datasets (overrides --limit).",
    )

    sub.add_parser("clear", help="Delete all cached datasets.")

    args = parser.parse_args()
    cache = DatasetCache(Path(args.data_dir), max_cached=args.limit)

    if args.cmd == "status":
        rows = cache.status()
        if not rows:
            print("Cache is empty.")
            return
        total_mb = sum(r["size_mb"] for r in rows)
        print(f"{'#':<4} {'t4dataset_id':<40} {'last_accessed':<28} {'size_mb':>8}  on_disk")
        print("-" * 90)
        for i, r in enumerate(rows, 1):
            flag = "yes" if r["on_disk"] else "MISSING"
            print(f"{i:<4} {r['t4dataset_id']:<40} {r['last_accessed']:<28} "
                  f"{r['size_mb']:>8.1f}  {flag}")
        print("-" * 90)
        print(f"Total: {len(rows)} datasets, {total_mb:.1f} MB")
        print(f"Limit: {cache.max_cached if cache.max_cached > 0 else 'unlimited'}")

    elif args.cmd == "evict":
        keep = args.keep if args.keep is not None else args.limit
        evicted = cache.evict_lru(keep)
        if evicted:
            print(f"Evicted {len(evicted)} dataset(s): {evicted}")
        else:
            print("Nothing to evict.")

    elif args.cmd == "clear":
        deleted = cache.clear()
        if deleted:
            print(f"Cleared {len(deleted)} dataset(s): {deleted}")
        else:
            print("Cache was already empty.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Directory names that look like RFC-4122 UUID strings (annotation-dataset ids).
_UUID_DIRNAME = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_uuid_shaped_dirname(name: str) -> bool:
    return bool(_UUID_DIRNAME.match(name))


def _find_webauto_nested(root: Path, t4dataset_id: str) -> Optional[Path]:
    """Return the versioned T4 root for webauto-style grouped layouts, if present.

    Data may live under any top-level group folder (not only ``annotation_dataset``)::

        root/<group>/<t4dataset_id>/<version>/

    Common *group* names include ``annotation_dataset``, project-specific names
    such as ``j6gen6_3``, etc.

    When several group folders contain the same *t4dataset_id*, the first match
    in sorted order by group folder name is returned. The *version* subdirectory
    chosen is the last in sorted order (same as before).
    """
    root = Path(root)
    try:
        groups = sorted(
            p for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    except OSError:
        return None
    for group in groups:
        uuid_dir = group / t4dataset_id
        if not uuid_dir.is_dir():
            continue
        versions = sorted(
            p for p in uuid_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        if versions:
            return versions[-1]
    return None


def list_webauto_annotation_dataset_ids(data_dir: Path) -> List[str]:
    """List annotation-dataset id strings by directory names only (fast).

    Collects names that match a UUID-shaped folder:

    - ``data_dir/<uuid>/`` (flat layout)
    - ``data_dir/<group>/<uuid>/`` (grouped webauto layout; *group* is any
      non-UUID top-level directory such as ``annotation_dataset`` or
      ``j6gen6_3``)

    Does not open files or validate T4 contents — intended for quick HTTP listing.
    """
    data_dir = Path(data_dir)
    seen: set[str] = set()
    out: List[str] = []
    try:
        top = sorted(
            p for p in data_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    except OSError:
        return []
    for p in top:
        if _is_uuid_shaped_dirname(p.name):
            if p.name not in seen:
                seen.add(p.name)
                out.append(p.name)
            continue
        try:
            for sub in p.iterdir():
                if not sub.is_dir() or sub.name.startswith("."):
                    continue
                if _is_uuid_shaped_dirname(sub.name) and sub.name not in seen:
                    seen.add(sub.name)
                    out.append(sub.name)
        except OSError:
            continue
    return sorted(out)


def _try_flatten(root: Path, t4dataset_id: str, dst: Optional[Path] = None) -> bool:
    """Move webauto's versioned directory to *dst*.

    webauto may place data under a grouped folder::

        root/<group>/<t4dataset_id>/<version>/

    This function moves that versioned directory to *dst* (default: *root*),
    then removes the now-empty ``<group>/<t4dataset_id>/`` wrapper when possible.

    Returns True if the directory was moved.
    """
    src = _find_webauto_nested(root, t4dataset_id)
    if src is None:
        return False
    if dst is None:
        dst = root
    print(f"  [downloader] Moving {src}  →  {dst}")
    shutil.move(str(src), str(dst))
    # Remove now-empty wrapper dirs (annotation_dataset/<t4dataset_id>/ etc.)
    for d in [src.parent, src.parent.parent]:
        try:
            d.rmdir()
        except OSError:
            break
    return True


def _is_t4_root(path: Path) -> bool:
    """Return True if *path* directly contains T4 annotation files."""
    for candidate in [
        path / "annotation",
        *sorted(path.glob("v1.0-*")),
        *sorted(path.glob("annotation/*")),
    ]:
        if (candidate / "sample.json").exists() or (candidate / "scene.json").exists():
            return True
    return (path / "sample.json").exists() or (path / "scene.json").exists()


def _looks_like_t4dataset(path: Path) -> bool:
    """Return True if *path* contains T4 annotation files (direct or one level nested).

    Handles two webauto download layouts:

    Normal layout::

        path/annotation/sample.json

    Nested layout (extra UUID subdirectory)::

        path/<uuid>/annotation/sample.json
        path/map/
    """
    if _is_t4_root(path):
        return True
    # Also accept if an immediate subdirectory is the T4 root (extra UUID nesting).
    try:
        for subdir in path.iterdir():
            if subdir.is_dir() and _is_t4_root(subdir):
                return True
    except OSError:
        pass
    return False


def find_t4_root(path: Path) -> Path:
    """Return the directory that should be passed to ``Tier4()``.

    Resolves two webauto download layouts:

    Normal layout — data directly in *path*::

        path/annotation/...   →  returns path

    Nested layout — extra UUID subdirectory::

        path/<uuid>/annotation/...
        path/map/             →  returns path/<uuid>/

    Falls back to *path* if neither layout is recognised (lets Tier4
    raise its own informative error).
    """
    if _is_t4_root(path):
        return path
    try:
        for subdir in sorted(path.iterdir()):
            if subdir.is_dir() and _is_t4_root(subdir):
                return subdir
    except OSError:
        pass
    return path


def _can_write_dir(path: Path) -> bool:
    """Return True if the current process can write to *path*."""
    return os.access(path, os.W_OK)


def _make_shadow(t4_root: Path) -> Path:
    """Return a writable local shadow of *t4_root*.

    The shadow contains:
    - A copy of every annotation JSON file (writable, patchable).
    - Symlinks for every non-JSON entry (images, point clouds, etc.)
      pointing back to the originals on the source filesystem.

    The shadow root lives under ~/.cache/t4_shadow/ and is keyed by the
    absolute real path of *t4_root*, so the same dataset always reuses
    the same shadow.
    """
    import hashlib
    import shutil

    key = hashlib.sha1(str(t4_root.resolve()).encode()).hexdigest()[:16]
    shadow_root = Path.home() / ".cache" / "t4_shadow" / key
    shadow_root.mkdir(parents=True, exist_ok=True)

    # Walk the source tree and mirror its structure locally.
    for src in t4_root.rglob("*"):
        rel = src.relative_to(t4_root)
        dst = shadow_root / rel

        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.suffix == ".json":
            # Copy JSON files so stubs can be added without touching the source.
            if not dst.exists():
                shutil.copy2(src, dst)
        else:
            # Symlink everything else (images, .pcd.bin, …) to avoid copies.
            if not dst.exists():
                dst.symlink_to(src.resolve())

    return shadow_root


def prepare_dataset_root(t4_root: Path) -> Path:
    """Return a writable root for *t4_root*, creating a local shadow if needed.

    This is a workaround for t4_devkit requiring stub files (``[]``) for
    optional tables that may be absent in some dataset exports.  When the
    source filesystem is read-only (e.g. a CIFS/NFS mount without write
    permission), stubs are written to a local shadow copy instead.

    TODO: Remove this function once t4_devkit handles missing optional tables
    gracefully without requiring the files to exist on disk.
    Upstream issue to file: t4_devkit should catch FileNotFoundError for
    optional tables (attribute, visibility, lidarseg) and return [] instead.
    """
    # Find the annotation directory to check write access.
    # Fall back to t4_root itself if no ann_dir is detected yet.
    probe_dir = t4_root
    for depth1 in [t4_root, *t4_root.iterdir()]:
        if depth1.is_dir() and (depth1 / "sample.json").exists():
            probe_dir = depth1
            break
        if depth1.is_dir():
            for depth2 in depth1.iterdir():
                if depth2.is_dir() and (depth2 / "sample.json").exists():
                    probe_dir = depth2
                    break

    if _can_write_dir(probe_dir):
        return t4_root  # Source is writable — no shadow needed.

    print(f"  [t4-shadow] Source not writable, creating local shadow for {t4_root.name}")
    return _make_shadow(t4_root)


def patch_missing_t4_tables(t4_root: Path) -> None:
    """Create empty JSON stubs for mandatory T4 tables that are absent on disk.

    Some webauto exports omit tables like ``attribute.json`` when the dataset
    contains no entries for that table.  ``t4_devkit`` still requires the file
    to exist (even if empty), so we create ``[]`` stubs on-the-fly without
    modifying the original data for tables that are known-safe to be empty.
    """
    # Locate annotation directories by finding `sample.json` anywhere under the
    # dataset root.  The previous two-level scan missed deeper layouts (e.g.
    # ``<uuid>/<ver>/annotation/sample.json``), so ``attribute.json`` was never
    # stubbed and t4_devkit raised ``FileNotFoundError: attribute is mandatory``.
    ann_dirs: list[Path] = []
    try:
        for sample_json in t4_root.rglob("sample.json"):
            parent = sample_json.parent
            if parent.is_dir():
                ann_dirs.append(parent)
    except OSError:
        pass

    # Deduplicate (same dir reachable via symlinks / multiple matches).
    seen: set[Path] = set()
    uniq_ann: list[Path] = []
    for d in ann_dirs:
        try:
            key = d.resolve()
        except OSError:
            key = d
        if key not in seen:
            seen.add(key)
            uniq_ann.append(d)

    if not uniq_ann:
        return

    # Tables that are safe to be empty (no entries = valid empty list).
    # Structural tables (sample, sensor, calibrated_sensor, …) are NOT listed
    # here because an empty stub would silently hide real data problems.
    SAFE_EMPTY = [
        "attribute.json",
        "visibility.json",
        "lidarseg.json",
    ]
    for ann_dir in uniq_ann:
        for name in SAFE_EMPTY:
            target = ann_dir / name
            if not target.exists():
                print(f"  [t4-patch] Creating empty stub: {target}")
                target.write_text("[]")


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory tree in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _run_cmd_template(
    template: str, t4dataset_id: str, dest_dir: Path, dataset_path: Path
) -> None:
    cmd = template.format(
        t4dataset_id=t4dataset_id,
        dest_dir=str(dest_dir),
        dataset_path=str(dataset_path),
    )
    print(f"  [downloader] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        raise DownloadError(f"Download command failed (exit {result.returncode}): {cmd}")


def _download_impl(t4dataset_id: str, dest_dir: Path, dataset_path: Path) -> None:
    """Download via ``webauto data annotation-dataset pull``.

    Passes ``dest_dir`` (not ``dest_dir/t4dataset_id``) as ``--asset-dir`` so
    that webauto places the dataset at::

        dest_dir/annotation_dataset/<t4dataset_id>/<version>/

    After the download completes the versioned directory is moved to
    ``dataset_path`` (= ``dest_dir / t4dataset_id``) so that the rest of the
    cache logic continues to work unchanged.
    """
    cmd = [
        "webauto", "data", "annotation-dataset", "pull",
        "--project-id", WEBAUTO_PROJECT_ID,
        "--annotation-dataset-id", t4dataset_id,
        "--asset-dir", str(dest_dir),
    ]
    print(f"  [downloader] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise DownloadError(
            f"webauto download failed (exit {result.returncode}) for dataset '{t4dataset_id}'.\n"
            f"Command: {' '.join(cmd)}\n"
            "Check that `webauto` is installed and you are logged in."
        )

    # webauto writes to dest_dir/annotation_dataset/<uuid>/<version>/
    # Flatten the versioned directory contents into dataset_path.
    _try_flatten(dest_dir, t4dataset_id, dst=dataset_path)

