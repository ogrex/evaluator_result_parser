"""Simple stepped load test CLI for the T4 FastAPI server.

The tool focuses on concurrency behavior and response-time changes as load
increases. It can either use an explicit JSON request body or discover a usable
render target from the server by calling ``/datasets`` and
``/datasets/{id}/scenarios``.

Supported benchmark modes:
- ``render``: ``POST /render``
- ``viewer-three-frame-bin``: repeated ``GET /viewer/three/frame.bin``
- ``viewer-three``: simulate the server-side request sequence used by ``/viewer/three``
- ``viewer-tlr``: simulate the server-side request sequence used by ``/viewer/tlr``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TIMEOUT_S = 300.0
DEFAULT_CONCURRENCY = (1, 2, 4, 8, 16)


class SampleResult:
    def __init__(
        self,
        *,
        ok: bool,
        status_code: Optional[int],
        client_elapsed_ms: float,
        server_elapsed_ms: Optional[float],
        server_render_ms: Optional[float],
        server_tier4_load_ms: Optional[float],
        error: str = "",
    ):
        self.ok = ok
        self.status_code = status_code
        self.client_elapsed_ms = client_elapsed_ms
        self.server_elapsed_ms = server_elapsed_ms
        self.server_render_ms = server_render_ms
        self.server_tier4_load_ms = server_tier4_load_ms
        self.error = error


class BenchmarkTarget:
    def __init__(self, *, label: str, request_context: Dict, request_steps: List[Dict]):
        self.label = label
        self.request_context = request_context
        self.request_steps = request_steps


def _http_json(
    url: str,
    *,
    method: str = "GET",
    json_body: Optional[Dict] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Tuple[Dict, Dict[str, str], int]:
    data = None
    headers = {"Accept": "application/json"}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
        payload = json.loads(body.decode("utf-8")) if body else {}
        return payload, dict(resp.headers.items()), resp.status


def _http_bytes(
    url: str,
    *,
    method: str = "GET",
    json_body: Optional[Dict] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Tuple[bytes, Dict[str, str], int]:
    data = None
    headers = {}
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read(), dict(resp.headers.items()), resp.status


def _parse_csv_ints(raw: str) -> List[int]:
    values: List[int] = []
    for part in str(raw).split(","):
        txt = part.strip()
        if not txt:
            continue
        value = int(txt)
        if value < 1:
            raise ValueError("Concurrency values must be >= 1.")
        values.append(value)
    if not values:
        raise ValueError("At least one concurrency value is required.")
    return values


def _format_ms(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.1f}"


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _extract_server_timing(headers: Dict[str, str], body: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    def parse_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    elapsed_ms = parse_float(headers.get("X-Server-Elapsed-Ms"))
    tier4_load_ms = parse_float(headers.get("X-Server-Tier4-Load-Ms"))
    render_ms = parse_float(headers.get("X-Server-Render-Ms"))

    if elapsed_ms is None:
        elapsed_ms = parse_float(body.get("elapsed_ms"))
    if tier4_load_ms is None:
        tier4_load_ms = parse_float(body.get("tier4_load_ms"))
    if render_ms is None:
        render_ms = parse_float(body.get("render_ms"))

    return elapsed_ms, render_ms, tier4_load_ms


def _pick_request_payload(args) -> Dict:
    if args.request_file:
        path = Path(args.request_file)
        return json.loads(path.read_text(encoding="utf-8"))

    dataset_id = args.dataset_id
    scenario_name = args.scenario_name
    frame_index = args.frame_index

    if dataset_id and scenario_name and frame_index is not None:
        return {
            "t4dataset_id": dataset_id,
            "scenario_name": scenario_name,
            "frame_index": frame_index,
            "cameras": _split_csv_optional(args.cameras),
            "show_annotations": args.show_annotations,
            "crop_cameras": args.crop_cameras,
            "crop_padding": args.crop_padding,
            "crop_min_size": args.crop_min_size,
            "version": args.version,
            "target_objects": [],
        }

    base_url = args.base_url.rstrip("/")
    datasets_url = f"{base_url}/datasets"
    datasets_payload, _, _ = _http_json(datasets_url, timeout_s=args.timeout)
    datasets = datasets_payload.get("datasets") or []
    if not datasets:
        raise RuntimeError(
            "No datasets were returned by /datasets. Pass --request-file or "
            "--dataset-id/--scenario-name/--frame-index explicitly."
        )

    dataset_id = dataset_id or datasets[0]
    scenarios_url = f"{base_url}/datasets/{urllib.parse.quote(dataset_id, safe='')}/scenarios"
    if args.version:
        scenarios_url += "?" + urllib.parse.urlencode({"version": args.version})
    scenarios_payload, _, _ = _http_json(scenarios_url, timeout_s=args.timeout)
    scenarios = scenarios_payload.get("scenarios") or []
    if not scenarios:
        raise RuntimeError(
            f"No scenarios were returned for dataset {dataset_id!r}. "
            "Pass an explicit request with --request-file if needed."
        )

    chosen = None
    if scenario_name:
        chosen = next((row for row in scenarios if row.get("name") == scenario_name), None)
        if chosen is None:
            raise RuntimeError(f"Scenario {scenario_name!r} was not found in dataset {dataset_id!r}.")
    else:
        chosen = max(scenarios, key=lambda row: int(row.get("nbr_samples") or 0))
        scenario_name = str(chosen.get("name") or "")

    nbr_samples = int(chosen.get("nbr_samples") or 1)
    if frame_index is None:
        frame_index = max(0, min(args.auto_frame_index, nbr_samples - 1))

    return {
        "t4dataset_id": dataset_id,
        "scenario_name": scenario_name,
        "frame_index": frame_index,
        "cameras": _split_csv_optional(args.cameras),
        "show_annotations": args.show_annotations,
        "crop_cameras": args.crop_cameras,
        "crop_padding": args.crop_padding,
        "crop_min_size": args.crop_min_size,
        "version": args.version,
        "target_objects": [],
    }


def _pick_viewer_context(args) -> Dict:
    payload = _pick_request_payload(args)
    return {
        "t4dataset_id": payload["t4dataset_id"],
        "scenario_name": payload["scenario_name"],
        "frame_index": payload["frame_index"],
        "version": payload.get("version"),
        "camera": args.camera,
    }


def _split_csv_optional(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    return values or None


def _build_url(base_url: str, path: str, query: Dict[str, object]) -> str:
    params = {}
    for key, value in query.items():
        if value is None:
            continue
        params[key] = str(value)
    qs = urllib.parse.urlencode(params)
    return f"{base_url.rstrip('/')}{path}" + (f"?{qs}" if qs else "")


def _build_target(args) -> BenchmarkTarget:
    mode = args.mode
    if mode == "render":
        payload = _pick_request_payload(args)
        return BenchmarkTarget(
            label="POST /render",
            request_context=payload,
            request_steps=[
                {
                    "name": "render_post",
                    "kind": "json",
                    "method": "POST",
                    "url": args.base_url.rstrip("/") + "/render",
                    "json_body": payload,
                }
            ],
        )

    context = _pick_viewer_context(args)
    dataset_id = context["t4dataset_id"]
    scenario_name = context["scenario_name"]
    frame_index = context["frame_index"]
    version = context.get("version")
    camera = context.get("camera")
    common = {
        "t4dataset_id": dataset_id,
        "scenario_name": scenario_name,
        "frame_index": frame_index,
    }
    if version:
        common["version"] = version

    if mode == "viewer-three-frame-bin":
        steps = [
            {
                "name": "viewer_frame_bin",
                "kind": "bytes",
                "method": "GET",
                "url": _build_url(args.base_url, "/viewer/three/frame.bin", common),
            },
        ]
        return BenchmarkTarget(
            label="3D viewer frame.bin only",
            request_context=context,
            request_steps=steps,
        )

    if mode == "viewer-three":
        steps: List[Dict] = [
            {
                "name": "viewer_page",
                "kind": "bytes",
                "method": "GET",
                "url": _build_url(
                    args.base_url,
                    "/viewer/three",
                    {
                        "t4dataset_id": dataset_id,
                        "scenario_name": scenario_name,
                        "frame_index": frame_index,
                        "version": version,
                    },
                ),
            },
            {
                "name": "viewer_meta",
                "kind": "json",
                "method": "GET",
                "url": _build_url(
                    args.base_url,
                    "/viewer/three/meta",
                    {
                        "t4dataset_id": dataset_id,
                        "scenario_name": scenario_name,
                        "version": version,
                    },
                ),
            },
            {
                "name": "viewer_frame_bin",
                "kind": "bytes",
                "method": "GET",
                "url": _build_url(args.base_url, "/viewer/three/frame.bin", common),
            },
        ]
        if args.viewer_three_include_camera_overlay:
            overlay_query = dict(common)
            overlay_query["show_annotations"] = "true" if args.show_annotations else "false"
            if args.viewer_three_all_cameras:
                overlay_query["all_cameras"] = "true"
            elif camera:
                overlay_query["camera"] = camera
            steps.append(
                {
                    "name": "viewer_camera_overlay",
                    "kind": "json",
                    "method": "GET",
                    "url": _build_url(args.base_url, "/viewer/three/camera-overlay", overlay_query),
                }
            )
        if args.viewer_three_include_lanelet:
            lanelet_query = dict(common)
            lanelet_query["max_segments"] = args.viewer_three_max_segments
            lanelet_query["clip_radius_m"] = args.viewer_three_clip_radius_m
            steps.append(
                {
                    "name": "viewer_lanelet_lines",
                    "kind": "json",
                    "method": "GET",
                    "url": _build_url(args.base_url, "/viewer/three/lanelet-lines", lanelet_query),
                }
            )
        return BenchmarkTarget(
            label="3D viewer session",
            request_context=context,
            request_steps=steps,
        )

    if mode == "viewer-tlr":
        page_query = {
            "t4dataset_id": dataset_id,
            "scenario_name": scenario_name,
            "frame_index": frame_index,
            "version": version,
        }
        if camera:
            page_query["camera"] = camera
        frame_query = dict(page_query)
        steps = [
            {
                "name": "tlr_page",
                "kind": "bytes",
                "method": "GET",
                "url": _build_url(args.base_url, "/viewer/tlr", page_query),
            },
            {
                "name": "tlr_frame",
                "kind": "json",
                "method": "GET",
                "url": _build_url(args.base_url, "/viewer/tlr/frame", frame_query),
            },
        ]
        return BenchmarkTarget(
            label="TLR viewer session",
            request_context=context,
            request_steps=steps,
        )

    raise RuntimeError(f"Unsupported mode: {mode}")


def _run_step(step: Dict, timeout_s: float) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    kind = step["kind"]
    method = step["method"]
    url = step["url"]
    body = step.get("json_body")
    if kind == "json":
        payload, headers, status_code = _http_json(
            url,
            method=method,
            json_body=body,
            timeout_s=timeout_s,
        )
        return (*_extract_server_timing(headers, payload), status_code)
    if kind == "bytes":
        _, headers, status_code = _http_bytes(
            url,
            method=method,
            json_body=body,
            timeout_s=timeout_s,
        )
        return (_parse_optional_float(headers.get("X-Server-Elapsed-Ms")), None, None, status_code)
    raise RuntimeError(f"Unsupported step kind: {kind}")


def _parse_optional_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_single_request(target: BenchmarkTarget, timeout_s: float) -> SampleResult:
    started = time.perf_counter()
    try:
        server_elapsed_values: List[float] = []
        server_render_values: List[float] = []
        server_tier4_values: List[float] = []
        status_code = 200
        for step in target.request_steps:
            step_server_elapsed_ms, step_server_render_ms, step_server_tier4_load_ms, status_code = _run_step(
                step,
                timeout_s,
            )
            if not (200 <= status_code < 300):
                raise urllib.error.HTTPError(step["url"], status_code, step["name"], hdrs=None, fp=None)
            if step_server_elapsed_ms is not None:
                server_elapsed_values.append(step_server_elapsed_ms)
            if step_server_render_ms is not None:
                server_render_values.append(step_server_render_ms)
            if step_server_tier4_load_ms is not None:
                server_tier4_values.append(step_server_tier4_load_ms)
        client_elapsed_ms = (time.perf_counter() - started) * 1000.0
        return SampleResult(
            ok=200 <= status_code < 300,
            status_code=status_code,
            client_elapsed_ms=client_elapsed_ms,
            server_elapsed_ms=sum(server_elapsed_values) if server_elapsed_values else None,
            server_render_ms=sum(server_render_values) if server_render_values else None,
            server_tier4_load_ms=sum(server_tier4_values) if server_tier4_values else None,
            error="" if 200 <= status_code < 300 else f"http_{status_code}",
        )
    except urllib.error.HTTPError as exc:
        return SampleResult(
            ok=False,
            status_code=exc.code,
            client_elapsed_ms=(time.perf_counter() - started) * 1000.0,
            server_elapsed_ms=None,
            server_render_ms=None,
            server_tier4_load_ms=None,
            error=f"http_{exc.code}: {exc.reason}",
        )
    except Exception as exc:
        return SampleResult(
            ok=False,
            status_code=None,
            client_elapsed_ms=(time.perf_counter() - started) * 1000.0,
            server_elapsed_ms=None,
            server_render_ms=None,
            server_tier4_load_ms=None,
            error=str(exc),
        )


def _summarize_results(concurrency: int, results: Sequence[SampleResult], wall_time_s: float) -> Dict[str, object]:
    ok_results = [r for r in results if r.ok]
    client_latencies = [r.client_elapsed_ms for r in ok_results]
    server_elapsed = [r.server_elapsed_ms for r in ok_results if r.server_elapsed_ms is not None]
    server_render = [r.server_render_ms for r in ok_results if r.server_render_ms is not None]
    server_tier4_load = [r.server_tier4_load_ms for r in ok_results if r.server_tier4_load_ms is not None]

    total = len(results)
    success = len(ok_results)
    failures = total - success
    success_rate = (success / total * 100.0) if total else 0.0
    throughput_rps = (success / wall_time_s) if wall_time_s > 0 else 0.0

    top_errors: Dict[str, int] = {}
    for item in results:
        if item.ok:
            continue
        key = item.error or f"http_{item.status_code}" or "unknown"
        top_errors[key] = top_errors.get(key, 0) + 1

    return {
        "concurrency": concurrency,
        "requests": total,
        "success": success,
        "failures": failures,
        "success_rate_pct": success_rate,
        "wall_time_s": wall_time_s,
        "throughput_rps": throughput_rps,
        "client_avg_ms": statistics.mean(client_latencies) if client_latencies else None,
        "client_p50_ms": _percentile(client_latencies, 0.50),
        "client_p90_ms": _percentile(client_latencies, 0.90),
        "client_p95_ms": _percentile(client_latencies, 0.95),
        "client_p99_ms": _percentile(client_latencies, 0.99),
        "client_max_ms": max(client_latencies) if client_latencies else None,
        "server_avg_ms": statistics.mean(server_elapsed) if server_elapsed else None,
        "server_p95_ms": _percentile(server_elapsed, 0.95),
        "server_render_avg_ms": statistics.mean(server_render) if server_render else None,
        "server_tier4_load_avg_ms": statistics.mean(server_tier4_load) if server_tier4_load else None,
        "top_errors": ", ".join(
            f"{name} x{count}"
            for name, count in sorted(top_errors.items(), key=lambda item: (-item[1], item[0]))[:3]
        ),
    }


def _run_level(target: BenchmarkTarget, *, concurrency: int, request_count: int, timeout_s: float) -> Dict[str, object]:
    started = time.perf_counter()
    results: List[SampleResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_run_single_request, target, timeout_s)
            for _ in range(request_count)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    wall_time_s = time.perf_counter() - started
    return _summarize_results(concurrency, results, wall_time_s)


def _has_any_server_timings(rows: Sequence[Dict[str, object]]) -> bool:
    for row in rows:
        if row.get("server_avg_ms") is not None:
            return True
        if row.get("server_render_avg_ms") is not None:
            return True
        if row.get("server_tier4_load_avg_ms") is not None:
            return True
    return False


def _render_columns(rows: Sequence[Dict[str, object]]) -> List[Tuple[str, str]]:
    columns = [
        ("concurrency", "concurrency"),
        ("requests", "requests"),
        ("success_rate_pct", "success%"),
        ("throughput_rps", "rps"),
        ("client_p50_ms", "client_p50_ms"),
        ("client_p95_ms", "client_p95_ms"),
        ("client_p99_ms", "client_p99_ms"),
    ]
    if _has_any_server_timings(rows):
        columns.extend(
            [
                ("server_avg_ms", "server_avg_ms"),
                ("server_render_avg_ms", "server_render_avg_ms"),
                ("server_tier4_load_avg_ms", "tier4_load_avg_ms"),
            ]
        )
    columns.append(("top_errors", "errors"))
    return columns


def _format_cell(key: str, row: Dict[str, object]) -> str:
    value = row.get(key)
    if key == "success_rate_pct":
        return f"{float(value):.1f}" if value is not None else "-"
    if key == "throughput_rps":
        return f"{float(value):.2f}" if value is not None else "-"
    if key.endswith("_ms"):
        return _format_ms(value if isinstance(value, (int, float)) or value is None else None)
    if key == "top_errors":
        return str(value or "-")
    return str(value)


def _print_summary_table(rows: Sequence[Dict[str, object]]) -> None:
    headers = [label for _, label in _render_columns(rows)]
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        columns = _render_columns(rows)
        print(
            "| "
            + " | ".join([_format_cell(key, row) for key, _ in columns])
            + " |"
        )


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = [
        "concurrency",
        "requests",
        "success",
        "failures",
        "success_rate_pct",
        "wall_time_s",
        "throughput_rps",
        "client_avg_ms",
        "client_p50_ms",
        "client_p90_ms",
        "client_p95_ms",
        "client_p99_ms",
        "client_max_ms",
        "server_avg_ms",
        "server_p95_ms",
        "server_render_avg_ms",
        "server_tier4_load_avg_ms",
        "top_errors",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _save_plot(path: Path, rows: Sequence[Dict[str, object]], target: BenchmarkTarget) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot-out. Install project dependencies first."
        ) from exc

    x = [int(row["concurrency"]) for row in rows]
    rps = [float(row["throughput_rps"]) for row in rows]
    p50 = [float(row["client_p50_ms"]) if row.get("client_p50_ms") is not None else math.nan for row in rows]
    p95 = [float(row["client_p95_ms"]) if row.get("client_p95_ms") is not None else math.nan for row in rows]
    p99 = [float(row["client_p99_ms"]) if row.get("client_p99_ms") is not None else math.nan for row in rows]
    success_rate = [float(row["success_rate_pct"]) for row in rows]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, 1.15]},
    )
    fig.patch.set_facecolor("#faf7f0")

    for ax in (ax1, ax2):
        ax.set_facecolor("#fffdf8")
        ax.grid(True, linestyle="--", linewidth=0.7, color="#ddcfb8", alpha=0.75)
        for spine in ax.spines.values():
            spine.set_color("#cdbfa8")

    ax1.plot(x, rps, marker="o", linewidth=2.8, color="#0f766e", label="Throughput (RPS)")
    ax1.set_title(f"{target.label} Concurrency Trend", fontsize=16, color="#3d2f1f", pad=12)
    ax1.set_ylabel("Requests / sec", color="#3d2f1f")
    ax1.set_xlabel("Concurrency", color="#3d2f1f")
    ax1.tick_params(colors="#4a3a28")

    ax1b = ax1.twinx()
    ax1b.plot(x, success_rate, marker="s", linewidth=1.8, color="#b45309", label="Success %")
    ax1b.set_ylabel("Success %", color="#7c2d12")
    ax1b.tick_params(colors="#7c2d12")
    ax1b.set_ylim(0, 105)

    ax2.plot(x, p50, marker="o", linewidth=2.0, color="#2563eb", label="p50")
    ax2.plot(x, p95, marker="o", linewidth=2.2, color="#d97706", label="p95")
    ax2.plot(x, p99, marker="o", linewidth=2.2, color="#dc2626", label="p99")
    ax2.set_ylabel("Latency (ms)", color="#3d2f1f")
    ax2.set_xlabel("Concurrency", color="#3d2f1f")
    ax2.tick_params(colors="#4a3a28")
    ax2.legend(loc="upper left", frameon=False)

    summary = (
        f"Dataset={target.request_context.get('t4dataset_id')}  "
        f"Scenario={target.request_context.get('scenario_name')}  "
        f"Frame={target.request_context.get('frame_index')}"
    )
    fig.text(0.01, 0.01, summary, fontsize=10, color="#6b5a45")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def _parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark T4 server concurrency and latency. "
            "Runs stepped load tests and reports how response time changes as concurrency increases."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("render", "viewer-three-frame-bin", "viewer-three", "viewer-tlr"),
        default="render",
        help="Benchmark target mode (default: render).",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server base URL.")
    parser.add_argument("--request-file", help="JSON file with an explicit POST /render request body.")
    parser.add_argument("--dataset-id", help="Dataset ID used when auto-building the render request.")
    parser.add_argument("--scenario-name", help="Scenario name used when auto-building the render request.")
    parser.add_argument("--frame-index", type=int, help="Frame index used when auto-building the render request.")
    parser.add_argument(
        "--auto-frame-index",
        type=int,
        default=0,
        help="Fallback frame index when auto-discovering a scenario (default: 0).",
    )
    parser.add_argument("--version", help="Optional dataset version passed through to /render.")
    parser.add_argument("--cameras", help="Comma-separated camera list to reduce load scope if desired.")
    parser.add_argument("--camera", help="Single camera used for viewer modes when applicable.")
    parser.add_argument("--crop-cameras", action="store_true", help="Enable crop_cameras in the request.")
    parser.add_argument(
        "--no-show-annotations",
        dest="show_annotations",
        action="store_false",
        help="Disable annotations in the generated render request.",
    )
    parser.set_defaults(show_annotations=True)
    parser.add_argument("--crop-padding", type=int, default=40, help="crop_padding value for auto-built requests.")
    parser.add_argument("--crop-min-size", type=int, default=300, help="crop_min_size value for auto-built requests.")
    parser.add_argument(
        "--concurrency",
        default="1,2,4,8,16",
        help="Comma-separated concurrency levels (default: 1,2,4,8,16).",
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=20,
        help="Total requests to send at each concurrency level (default: 20).",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=2,
        help="Warmup requests to send before measurements begin (default: 2).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_S:g}).",
    )
    parser.add_argument(
        "--viewer-three-include-camera-overlay",
        action="store_true",
        help="In viewer-three mode, include /viewer/three/camera-overlay in each synthetic session.",
    )
    parser.add_argument(
        "--viewer-three-all-cameras",
        action="store_true",
        help="In viewer-three mode with camera overlay enabled, request all cameras instead of one camera.",
    )
    parser.add_argument(
        "--viewer-three-include-lanelet",
        action="store_true",
        help="In viewer-three mode, include /viewer/three/lanelet-lines in each synthetic session.",
    )
    parser.add_argument(
        "--viewer-three-max-segments",
        type=int,
        default=90000,
        help="max_segments for viewer-three lanelet requests (default: 90000).",
    )
    parser.add_argument(
        "--viewer-three-clip-radius-m",
        type=float,
        default=170.0,
        help="clip_radius_m for viewer-three lanelet requests (default: 170).",
    )
    parser.add_argument("--csv-out", help="Optional CSV file path for the aggregated results.")
    parser.add_argument(
        "--plot-out",
        help="Optional PNG file path for a concurrency trend chart.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        concurrency_levels = _parse_csv_ints(args.concurrency)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        target = _build_target(args)
    except Exception as exc:
        print(f"Failed to prepare benchmark target: {exc}", file=sys.stderr)
        return 1

    print("Benchmark target")
    print(f"  Mode          : {args.mode}")
    print(f"  Label         : {target.label}")
    print(f"  Dataset       : {target.request_context.get('t4dataset_id')}")
    print(f"  Scenario      : {target.request_context.get('scenario_name')}")
    print(f"  Frame index   : {target.request_context.get('frame_index')}")
    if args.mode == "render":
        print(f"  Cameras       : {target.request_context.get('cameras')}")
        print(f"  Annotations   : {target.request_context.get('show_annotations')}")
        print(f"  Crop cameras  : {target.request_context.get('crop_cameras')}")
    if args.mode in ("viewer-three-frame-bin", "viewer-three"):
        print(f"  Camera overlay: {args.viewer_three_include_camera_overlay}")
        print(f"  All cameras   : {args.viewer_three_all_cameras}")
        print(f"  Lanelet       : {args.viewer_three_include_lanelet}")
    if args.mode == "viewer-tlr":
        print(f"  Camera        : {target.request_context.get('camera')}")
    print(f"  Warmup        : {args.warmup_requests}")
    print(f"  Levels        : {', '.join(str(v) for v in concurrency_levels)}")
    print(f"  Requests/level: {args.requests_per_level}")
    print("  Session steps :")
    for step in target.request_steps:
        print(f"    - {step['method']} {step['url']}")

    if args.warmup_requests > 0:
        print()
        print("Running warmup requests...")
        _run_level(
            target,
            concurrency=min(max(1, concurrency_levels[0]), args.warmup_requests),
            request_count=args.warmup_requests,
            timeout_s=args.timeout,
        )

    rows: List[Dict[str, object]] = []
    for concurrency in concurrency_levels:
        print()
        print(f"Running concurrency={concurrency} with {args.requests_per_level} total requests...")
        row = _run_level(
            target,
            concurrency=concurrency,
            request_count=args.requests_per_level,
            timeout_s=args.timeout,
        )
        rows.append(row)
        print(
            "  "
            f"success={row['success']}/{row['requests']} "
            f"success_rate={row['success_rate_pct']:.1f}% "
            f"rps={row['throughput_rps']:.2f} "
            f"client_p95={_format_ms(row['client_p95_ms'])} ms "
            f"client_p99={_format_ms(row['client_p99_ms'])} ms"
        )

    _print_summary_table(rows)
    if args.csv_out:
        out_path = Path(args.csv_out)
        _write_csv(out_path, rows)
        print()
        print(f"CSV written to: {out_path}")
    if args.plot_out:
        plot_path = Path(args.plot_out)
        try:
            _save_plot(plot_path, rows, target)
        except Exception as exc:
            print(f"Failed to write plot: {exc}", file=sys.stderr)
            return 1
        print(f"Plot written to: {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
