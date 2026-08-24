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
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


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


def list_webauto_annotation_dataset_ids_multi(data_dirs: List[Path]) -> List[str]:
    """List annotation-dataset id strings from multiple directories.

    Combines results from all directories, removing duplicates.
    """
    seen: set[str] = set()
    out: List[str] = []
    for data_dir in data_dirs:
        ids = list_webauto_annotation_dataset_ids(data_dir)
        for id_ in ids:
            if id_ not in seen:
                seen.add(id_)
                out.append(id_)
    return sorted(out)


def parse_vehicle_catalog_url(
    catalog_url: str,
    *,
    fallback_project_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(project_id, vehicle_catalog_id)`` parsed from a catalog URL.

    Accepted URL shape example::

        https://evaluation.tier4.jp/evaluation/vehicle_catalogs/<catalog_id>?project_id=x2_dev

    Args:
        catalog_url: Full URL from the Evaluator UI.
        fallback_project_id: Used when query string has no ``project_id``.

    Raises:
        ValueError: If a UUID-like catalog id cannot be found or project_id is missing.
    """
    parsed = urlparse(str(catalog_url).strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("catalog_url must be a full URL.")

    m = re.search(
        r"/vehicle_catalogs/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        parsed.path,
        flags=re.IGNORECASE,
    )
    if not m:
        raise ValueError("Could not parse vehicle_catalog_id from catalog_url.")

    q = parse_qs(parsed.query)
    project_id = (q.get("project_id") or [""])[0].strip() or (fallback_project_id or "").strip()
    if not project_id:
        raise ValueError("project_id is required (query parameter or explicit argument).")
    return project_id, m.group(1)


def _extract_json_blob(text: str) -> Optional[Any]:
    """Best-effort JSON extraction from CLI stdout that may contain logs."""
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj >= 0 and last_obj > first_obj:
        snippet = raw[first_obj:last_obj + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr >= 0 and last_arr > first_arr:
        snippet = raw[first_arr:last_arr + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass
    return None


def _collect_catalog_dataset_ids(payload: Any) -> List[str]:
    """Recursively collect UUID-like dataset IDs from a catalog payload."""
    keys = {
        "t4_dataset_id",
        "t4DatasetId",
        "t4_dataset_ids",
        "t4DatasetIds",
        "annotation_dataset_id",
        "annotationDatasetId",
        "annotation_dataset_ids",
        "annotationDatasetIds",
        "dataset_id",
        "datasetId",
        "dataset_ids",
        "datasetIds",
    }
    seen: set[str] = set()
    out: List[str] = []

    def add_id(value: Any) -> None:
        if isinstance(value, str) and _is_uuid_shaped_dirname(value) and value not in seen:
            seen.add(value)
            out.append(value)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in keys:
                    if isinstance(v, list):
                        for item in v:
                            add_id(item)
                    elif isinstance(v, dict):
                        add_id(v.get("id"))
                    else:
                        add_id(v)
                visit(v)
            return
        if isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return out


def _make_webautoauth_session() -> Any:
    """Create an authenticated webauto HTTP session.

    Raises RuntimeError when webautoauth is unavailable or cannot initialize.
    """
    try:
        import webautoauth.requests
        from webautoauth.token import HttpService, TokenSource, load_config
    except ImportError as exc:
        raise RuntimeError(
            "webautoauth is required for suite-based catalog fallback. "
            f"python_executable={sys.executable}. "
            "Install in the same runtime: pip install webautoauth"
        ) from exc

    try:
        config = load_config()
        token_source = TokenSource(HttpService(config))
        return webautoauth.requests.make_session(token_source)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize webautoauth session: {exc}") from exc


def _http_get_json(session: Any, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resp = session.get(url, params=params or {}, headers={"accept": "application/json"})
    if getattr(resp, "status_code", None) != 200:
        body = getattr(resp, "text", "")
        raise RuntimeError(f"HTTP {getattr(resp, 'status_code', 'unknown')} for {url}: {body[:400]}")
    try:
        data = json.loads(resp.content)
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON response from {url}: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _get_vehicle_catalog_payload(project_id: str, vehicle_catalog_id: str) -> Tuple[Dict[str, Any], str]:
    """Fetch vehicle catalog JSON payload directly via authenticated APIs."""
    session = _make_webautoauth_session()
    urls = [
        f"https://evaluation.ci.web.auto/v3/projects/{project_id}/vehicle-catalogs/{vehicle_catalog_id}",
        f"https://evaluation.ci.web.auto/v3/projects/{project_id}/vehicle_catalogs/{vehicle_catalog_id}",
    ]
    errors: List[str] = []
    for url in urls:
        try:
            return _http_get_json(session, url), url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Failed to fetch vehicle catalog via API. " + " | ".join(errors))


def _list_catalog_suite_ids(project_id: str, vehicle_catalog_id: str) -> List[str]:
    """List suite IDs belonging to a vehicle catalog.

    Reuses list_catalog_suites (which handles the broken catalogId filter + attachments
    filtering) and returns only the IDs.
    """
    suites = list_catalog_suites(project_id, vehicle_catalog_id)
    return [str(s.get("id", "")).strip() for s in suites]


def get_suite_info(project_id: str, suite_id: str) -> Dict[str, Any]:
    """Fetch detailed information for a single suite.

    Returns a dict with keys: id, name, description, catalog_id, created_at,
    updated_at, specs (list of scenario refs), and any other fields from the API.
    """
    print(f"[downloader] Fetching suite {suite_id}...")
    session = _make_webautoauth_session()
    url = f"https://evaluation.ci.web.auto/v3/projects/{project_id}/suites/{suite_id}"
    data = _http_get_json(session, url)
    return data if isinstance(data, dict) else {}


def list_catalog_suites(project_id: str, vehicle_catalog_id: str) -> List[Dict[str, Any]]:
    """List all suites belonging to a vehicle catalog with their details.

    The /suites API does not filter by catalogId, so we fetch all suites and
    filter them by checking the attachments[*].catalog_id field.

    Args:
        project_id: The project ID (e.g., "x2_dev").
        vehicle_catalog_id: The vehicle catalog UUID.

    Returns:
        List of suite information dictionaries.
    """
    session = _make_webautoauth_session()
    base = f"https://evaluation.ci.web.auto/v3/projects/{project_id}/suites"

    # Fetch ALL suites first (catalogId filter is broken in the API)
    all_suites: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    next_token = ""
    loops = 0
    while True:
        loops += 1
        params: Dict[str, Any] = {"size": 100}
        if next_token:
            params["next_token"] = next_token
        print(f"[downloader] list_catalog_suites: fetching page {loops}, total fetched so far: {len(all_suites)}...")
        data = _http_get_json(session, base, params=params)
        suites = data.get("suites") or []
        for suite in suites:
            if not isinstance(suite, dict):
                continue
            sid = str(suite.get("id", "")).strip()
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                all_suites.append(suite)
        next_token = str(data.get("next_token", "") or "").strip()
        if not next_token or loops >= 200:
            break

    print(f"[downloader] list_catalog_suites: fetched {len(all_suites)} suites total, now filtering by catalog_id...")

    # Filter by checking attachments[*].catalog_id
    vehicle_catalog_id = str(vehicle_catalog_id).strip().lower()
    filtered: List[Dict[str, Any]] = []
    for suite in all_suites:
        atts = suite.get("attachments") or []
        if not isinstance(atts, list):
            atts = []
        for att in atts:
            att_cat_id = str(att.get("catalog_id", "")).strip().lower()
            if att_cat_id == vehicle_catalog_id:
                suite["catalog_id"] = vehicle_catalog_id
                filtered.append(suite)
                break

    print(f"[downloader] list_catalog_suites: filtered to {len(filtered)} suites in catalog {vehicle_catalog_id}")
    return filtered


def _list_suite_scenario_refs(project_id: str, suite_ids: List[str]) -> List[Tuple[str, Optional[int]]]:
    session = _make_webautoauth_session()
    refs: List[Tuple[str, Optional[int]]] = []
    seen: set[Tuple[str, Optional[int]]] = set()

    for suite_id in suite_ids:
        suite_url = f"https://evaluation.ci.web.auto/v3/projects/{project_id}/suites/{suite_id}"
        try:
            data = _http_get_json(session, suite_url)
        except Exception as exc:
            print(f"[downloader] warning: failed to describe suite {suite_id}: {exc}")
            continue
        specs = data.get("specs") or []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            scenario_id = str(spec.get("scenario_id", "")).strip()
            if not scenario_id:
                continue
            raw_ver = spec.get("scenario_version_id")
            if isinstance(raw_ver, int):
                scenario_version_id = raw_ver
            elif isinstance(raw_ver, str) and raw_ver.strip().isdigit():
                scenario_version_id = int(raw_ver.strip())
            else:
                scenario_version_id = None
            key = (scenario_id, scenario_version_id)
            if key not in seen:
                seen.add(key)
                refs.append(key)

    return refs


def _list_scenario_dataset_ids(project_id: str, scenario_refs: List[Tuple[str, Optional[int]]]) -> List[str]:
    session = _make_webautoauth_session()
    seen: set[str] = set()
    out: List[str] = []

    for scenario_id, scenario_version_id in scenario_refs:
        url = f"https://scenario.ci.web.auto/v1/projects/{project_id}/scenarios/{scenario_id}"
        params = {"scenario_version_id": scenario_version_id} if scenario_version_id is not None else None
        try:
            data = _http_get_json(session, url, params=params)
        except Exception as exc:
            print(
                f"[downloader] warning: failed to describe scenario {scenario_id}"
                f" (version={scenario_version_id}): {exc}"
            )
            continue
        ids = data.get("t4_dataset_ids") or []
        if isinstance(ids, list):
            for did in ids:
                if isinstance(did, str) and _is_uuid_shaped_dirname(did) and did not in seen:
                    seen.add(did)
                    out.append(did)

    return out


def get_scenario_info(project_id: str, scenario_id: str, scenario_version_id: Optional[int] = None) -> Dict[str, Any]:
    """Fetch detailed information for a single scenario.

    Returns a dict with all fields from the scenario API, including t4_dataset_ids.
    """
    print(f"[downloader] Fetching scenario {scenario_id} (version={scenario_version_id})...")
    session = _make_webautoauth_session()
    url = f"https://scenario.ci.web.auto/v1/projects/{project_id}/scenarios/{scenario_id}"
    params = {"scenario_version_id": scenario_version_id} if scenario_version_id is not None else None
    try:
        data = _http_get_json(session, url, params=params)
    except Exception as exc:
        print(f"[downloader] warning: failed to describe scenario {scenario_id} (version={scenario_version_id}): {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _list_vehicle_catalog_dataset_ids_via_suites(project_id: str, vehicle_catalog_id: str) -> List[str]:
    suite_ids = _list_catalog_suite_ids(project_id, vehicle_catalog_id)
    if not suite_ids:
        return []
    scenario_refs = _list_suite_scenario_refs(project_id, suite_ids)
    if not scenario_refs:
        return []
    return sorted(_list_scenario_dataset_ids(project_id, scenario_refs))


def list_vehicle_catalog_dataset_ids(project_id: str, vehicle_catalog_id: str) -> List[str]:
    """List T4 dataset IDs that appear in a vehicle catalog.

    Strategy:
    1) Try direct catalog API (webautoauth session).
    2) If no dataset IDs are present (common for suite-based catalogs),
       resolve suites -> scenarios -> t4_dataset_ids via authenticated APIs.
    """
    project_id = str(project_id).strip()
    vehicle_catalog_id = str(vehicle_catalog_id).strip()
    if not project_id or not vehicle_catalog_id:
        raise ValueError("project_id and vehicle_catalog_id are required.")

    direct_error: Optional[Exception] = None
    direct_ids: List[str] = []
    try:
        payload, _url = _get_vehicle_catalog_payload(project_id, vehicle_catalog_id)
        direct_ids = sorted(_collect_catalog_dataset_ids(payload))
        if direct_ids:
            return direct_ids
    except Exception as exc:
        direct_error = exc

    try:
        return _list_vehicle_catalog_dataset_ids_via_suites(project_id, vehicle_catalog_id)
    except Exception as suite_exc:
        if direct_error is not None:
            raise RuntimeError(
                "Catalog dataset lookup failed for both direct and suite-based paths. "
                f"direct_error={direct_error}; suite_error={suite_exc}"
            ) from suite_exc
        raise


def inspect_vehicle_catalog_dataset_lookup(project_id: str, vehicle_catalog_id: str) -> Dict[str, Any]:
    """Return step-by-step debug info for catalog→dataset resolution."""
    project_id = str(project_id).strip()
    vehicle_catalog_id = str(vehicle_catalog_id).strip()
    if not project_id or not vehicle_catalog_id:
        raise ValueError("project_id and vehicle_catalog_id are required.")

    out: Dict[str, Any] = {
        "project_id": project_id,
        "vehicle_catalog_id": vehicle_catalog_id,
        "steps": {},
        "dataset_ids": [],
        "source": "none",
    }

    direct_step: Dict[str, Any] = {
        "api_candidates": [
            f"https://evaluation.ci.web.auto/v3/projects/{project_id}/vehicle-catalogs/{vehicle_catalog_id}",
            f"https://evaluation.ci.web.auto/v3/projects/{project_id}/vehicle_catalogs/{vehicle_catalog_id}",
        ]
    }
    out["steps"]["direct_api"] = direct_step

    direct_ids: List[str] = []
    direct_error: Optional[str] = None
    try:
        payload, used_url = _get_vehicle_catalog_payload(project_id, vehicle_catalog_id)
        direct_step["used_url"] = used_url
        if isinstance(payload, dict):
            direct_step["top_keys"] = sorted(payload.keys())
        direct_ids = sorted(_collect_catalog_dataset_ids(payload))
        direct_step["dataset_ids_count"] = len(direct_ids)
        direct_step["dataset_ids_preview"] = direct_ids[:20]
    except Exception as exc:
        direct_error = str(exc)
        direct_step["error"] = direct_error

    if direct_ids:
        out["dataset_ids"] = direct_ids
        out["source"] = "direct"
        return out

    suite_step: Dict[str, Any] = {}
    out["steps"]["suite_fallback"] = suite_step
    try:
        suite_ids = _list_catalog_suite_ids(project_id, vehicle_catalog_id)
        suite_step["suite_count"] = len(suite_ids)
        suite_step["suite_ids_preview"] = suite_ids[:20]

        scenario_refs = _list_suite_scenario_refs(project_id, suite_ids)
        suite_step["scenario_ref_count"] = len(scenario_refs)
        suite_step["scenario_refs_preview"] = [
            {"scenario_id": sid, "scenario_version_id": ver}
            for sid, ver in scenario_refs[:20]
        ]

        suite_ids_final = sorted(_list_scenario_dataset_ids(project_id, scenario_refs))
        suite_step["dataset_ids_count"] = len(suite_ids_final)
        suite_step["dataset_ids_preview"] = suite_ids_final[:20]

        out["dataset_ids"] = suite_ids_final
        out["source"] = "suite"
    except Exception as exc:
        suite_step["error"] = str(exc)
        out["source"] = "none"
        if direct_error:
            out["error"] = (
                "Catalog dataset lookup failed for both direct and suite-based paths. "
                f"direct_error={direct_error}; suite_error={exc}"
            )
        else:
            out["error"] = f"Suite-based lookup failed: {exc}"

    return out


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

    Version-then-UUID layout — both levels present::

        path/0/<uuid>/annotation/...
        path/0/map/           →  returns path/0/<uuid>/

    The third shape is what some webauto downloads produce: the version
    directory holds another id-named directory instead of the tables, so
    neither of the first two checks matches and Tier4 was handed a directory
    with no annotations at all (a 500 with nothing in the log to explain it).

    Falls back to *path* if no layout is recognised (lets Tier4
    raise its own informative error).
    """
    if _is_t4_root(path):
        return path
    try:
        children = sorted(c for c in path.iterdir() if c.is_dir())
    except OSError:
        return path
    for subdir in children:
        if _is_t4_root(subdir):
            return subdir
    # One level deeper, e.g. a version directory wrapping the real root.
    for subdir in children:
        try:
            grandchildren = sorted(g for g in subdir.iterdir() if g.is_dir())
        except OSError:
            continue
        for candidate in grandchildren:
            if _is_t4_root(candidate):
                return candidate
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

    The shadow root lives under the first writable location among:
    - ``$T4_SHADOW_ROOT``
    - ``~/.cache/t4_shadow``
    - ``/tmp/t4_shadow``

    It is keyed by the absolute real path of *t4_root*, so the same dataset
    always reuses the same shadow.
    """
    import hashlib
    import shutil

    key = hashlib.sha1(str(t4_root.resolve()).encode()).hexdigest()[:16]
    candidates = []
    env_root = os.environ.get("T4_SHADOW_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path.home() / ".cache" / "t4_shadow")
    candidates.append(Path("/tmp/t4_shadow"))

    shadow_root = None
    last_exc = None
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".write_test"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            shadow_root = base / key
            shadow_root.mkdir(parents=True, exist_ok=True)
            break
        except OSError as exc:
            last_exc = exc
            continue
    if shadow_root is None:
        raise OSError(f"Unable to create T4 shadow directory: {last_exc}")

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
