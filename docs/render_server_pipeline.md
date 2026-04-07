# T4 render server - request sequence and rendering pipeline

This document explains the detailed `/render` request flow implemented in `t4_visualizer/server.py` and `t4_visualizer/visualize.py`. It focuses on how a render request moves through dataset resolution, Tier4 caching, frame lookup, matplotlib rendering, and final JSON or HTML response generation.

**Related code**

- HTTP routes: `t4_visualizer/server.py`
- Render API: `t4_visualizer/visualize.py`
- Dataset discovery: `t4_visualizer/batch.py`
- Dataset preparation helpers: `t4_visualizer/downloader.py`

---

## 1. System context

```mermaid
flowchart LR
  C["Client\n(browser / curl / Python)"]
  S["FastAPI app\nserver.py"]
  PC["_DatasetPathCache\nTTL cache"]
  TC["_Tier4Cache\nLRU cache"]
  B["find_dataset_in_dir\nbatch.py"]
  D["Dataset root on disk"]
  P["prepare_dataset_root /\npatch_missing_t4_tables"]
  T["Tier4 instance\nt4_devkit"]
  R["render_frame\nvisualize.py"]
  M["matplotlib + tempfile PNGs"]

  C --> S
  S --> PC
  PC --> B
  B --> D
  S --> TC
  TC --> P
  P --> D
  TC --> T
  S --> R
  R --> T
  R --> M
  M --> S
  S --> C
```

---

## 2. `POST /render` detailed sequence

This is the most complete path because it supports `target_objects`.

```mermaid
sequenceDiagram
  autonumber
  participant Client
  participant FastAPI as FastAPI /render POST
  participant PathCache as _DatasetPathCache
  participant Finder as find_dataset_in_dir
  participant Tier4Cache as _Tier4Cache
  participant Prep as downloader helpers
  participant Tier4 as t4_devkit.Tier4
  participant Render as render_frame()
  participant Viz as visualize_static()
  participant MPL as matplotlib + tempfile

  Client->>FastAPI: POST /render {t4dataset_id, scenario_name, frame_index, target_objects, ...}
  FastAPI->>FastAPI: Validate body with RenderRequest
  FastAPI->>FastAPI: Log request parameters

  FastAPI->>PathCache: _resolve_dataset(t4dataset_id)
  alt dataset path cache hit
    PathCache-->>FastAPI: dataset_path
  else dataset path cache miss
    PathCache->>Finder: find_dataset_in_dir(data_dir, id, search_depth)
    Finder-->>PathCache: Path or None
    alt dataset not found
      PathCache-->>FastAPI: None
      FastAPI-->>Client: 404 dataset_not_found
    else found
      PathCache-->>FastAPI: dataset_path
    end
  end

  FastAPI->>FastAPI: Convert body.target_objects -> List[TargetObject]
  FastAPI->>FastAPI: Build VisualizationRequest fields

  FastAPI->>Tier4Cache: load(dataset_path, version)
  alt Tier4 cache hit
    Tier4Cache-->>FastAPI: cached Tier4 instance
  else Tier4 cache miss
    Tier4Cache->>Prep: find_t4_root(dataset_path)
    Prep->>Prep: prepare_dataset_root(...)
    Prep->>Prep: patch_missing_t4_tables(...)
    Prep-->>Tier4Cache: prepared dataset root
    Tier4Cache->>Tier4: Tier4(str(root), version=...)
    Tier4-->>Tier4Cache: loaded instance
    Tier4Cache-->>FastAPI: cached Tier4 instance
  end

  FastAPI->>Render: render_frame(request, t4=t4)
  Render->>Render: find_sample_by_scene_and_index(t4, scenario_name, frame_index)
  alt scenario missing
    Render-->>FastAPI: ValueError
    FastAPI-->>Client: 404 scenario_not_found
  else frame index too large
    Render-->>FastAPI: IndexError
    FastAPI-->>Client: 400 frame_index_out_of_range
  else sample resolved
    Render->>MPL: create TemporaryDirectory("t4render_*")
    Render->>Viz: visualize_static(t4, sample, ...)
    Viz->>Viz: list_camera_channels(sample)
    Viz->>Viz: list_lidar_channels(sample)
    alt crop_cameras=true and ROI visible
      Viz->>Viz: group target objects by best camera
      Viz->>MPL: render one cropped camera+BEV PNG per camera group
    else standard layout
      Viz->>MPL: render camera grid + optional BEV PNG
    end
    MPL-->>Render: PNG files in temp dir
    Render->>Render: read *.png bytes -> RenderImage[]
    Render-->>FastAPI: VisualizationResult(images, sample_token, timestamp_us)
  end

  FastAPI->>FastAPI: base64 encode PNG bytes
  FastAPI->>FastAPI: Compute elapsed_ms / tier4_load_ms / render_ms
  FastAPI-->>Client: JSON RenderResponse + timing headers
```

---

## 3. `GET /render` sequence with response-format branching

`GET /render` shares the same render core, but first decides whether the response should be JSON or HTML.

```mermaid
sequenceDiagram
  autonumber
  participant Client
  participant Route as GET /render
  participant Query as _render_get_query
  participant Format as _effective_render_format
  participant Core as _run_render
  participant HTML as _render_html_page

  Client->>Route: GET /render?...&format?
  Route->>Query: Parse query params
  Query-->>Route: namespace(q)
  Route->>Format: decide using explicit format, Sec-Fetch-Dest, Accept
  alt html
    Format-->>Route: "html"
  else json
    Format-->>Route: "json"
  end

  Route->>Core: _run_render(...)
  Core-->>Route: RenderResponse + timing headers

  alt format == html
    Route->>HTML: embed base64 PNGs in standalone page
    HTML-->>Route: html document
    Route-->>Client: text/html + timing headers
  else format == json
    Route-->>Client: JSON RenderResponse + timing headers
  end
```

`GET /render/view` and `GET /render/html` skip the content-negotiation step and always take the HTML branch.

Important boundary:

- `/render`, `/render/view`, and `/render/html` return rendered PNG output from `render_frame()`.
- They do not accept external `pred` / `gt` evaluation box payloads from an embedding dashboard.
- When a parent page needs to inject evaluation boxes into an embedded iframe, the relevant path is `/viewer/three`, not `/render/html`.

---

## 4. Embedded iframe flow with external `pred` / `gt` boxes

When the page is embedded inside an evaluation dashboard such as Streamlit and the parent wants to provide prediction / estimate boxes, the flow switches to the Three.js viewer endpoints in `server.py`.

```mermaid
sequenceDiagram
  autonumber
  participant Parent as Streamlit dashboard
  participant Iframe as /viewer/three iframe
  participant Server as FastAPI server.py
  participant PathCache as _DatasetPathCache
  participant Tier4Cache as _Tier4Cache
  participant Tier4 as t4_devkit.Tier4

  Parent->>Iframe: mount iframe /viewer/three?t4dataset_id&scenario_name&frame_index
  Iframe->>Server: GET /viewer/three
  Server-->>Iframe: viewer_three.html

  Iframe->>Server: GET /viewer/three/meta?dataset&scenario&version
  Server->>PathCache: _resolve_dataset(...)
  PathCache-->>Server: dataset_path
  Server->>Tier4Cache: load(dataset_path, version)
  Tier4Cache-->>Server: Tier4 instance
  Server-->>Iframe: scenario resolution + total_frames + frame URL template

  Iframe->>Server: GET /viewer/three/frame.bin?...&frame_index=N
  Server->>PathCache: _resolve_dataset(...)
  Server->>Tier4Cache: load(dataset_path, version)
  Tier4Cache-->>Server: Tier4 instance
  Server->>Tier4: sample lookup + LiDAR/3D boxes
  Server-->>Iframe: binary frame payload (dataset points + dataset 3D boxes)

  Parent->>Iframe: postMessage { type: "bbox_layers" | "bbox_layers_by_frame", gt, pred, ... }
  Iframe->>Iframe: setExternalLayerPayload(...) or applyEvalLayersForFrame(...)
  Iframe->>Iframe: render GT/EST in gtLayerGroup / predLayerGroup
  Iframe->>Server: optional GET /viewer/three/debug/message-received
  Server-->>Iframe: ack JSON

  alt camera viewport enabled and external boxes present
    Iframe->>Server: POST /viewer/three/camera-overlay?... Body: { gt, pred }
    Server->>PathCache: _resolve_dataset(...)
    Server->>Tier4Cache: load(dataset_path, version)
    Tier4Cache-->>Server: Tier4 instance
    Server->>Tier4: resolve scenario + sample + camera image + calibration
    Server->>Server: project dataset 3D boxes -> boxes_2d
    Server->>Server: project external gt/pred -> boxes_2d_eval_gt / boxes_2d_pred
    Server-->>Iframe: image_base64 + 2D overlay payload
    Iframe->>Iframe: draw camera image + magenta/green/blue overlays
  else no external boxes
    Iframe->>Server: GET /viewer/three/camera-overlay?... 
    Server-->>Iframe: image_base64 + dataset-only 2D boxes
  end
```

What lives where in this embedded-dashboard mode:

- Dataset LiDAR and dataset 3D annotations come from `/viewer/three/frame.bin`.
- External `pred` / `gt` boxes come from the parent page through `window.postMessage`.
- External boxes are not stored in the dataset and are not packed into `frame.bin`.
- Server-side projection of those external boxes happens only on `POST /viewer/three/camera-overlay`.

If you want the deeper viewer-specific pipeline, see [viewer_3d_pipeline.md](/home/leigu/evaluator_result_parser/docs/viewer_3d_pipeline.md).

---

## 5. `_run_render()` internals

`_run_render()` in `t4_visualizer/server.py` is the central bridge between HTTP and the pure render API.

```mermaid
sequenceDiagram
  autonumber
  participant Route
  participant Core as _run_render
  participant Cache as _Tier4Cache
  participant Render as render_frame

  Route->>Core: dataset_path, scenario_name, frame_index, cameras, crop options, targets
  Core->>Core: Build VisualizationRequest
  Core->>Core: t0 = perf_counter()
  Core->>Cache: load(dataset_path, version)
  Cache-->>Core: Tier4 instance
  Core->>Core: t1 = perf_counter()
  Core->>Render: render_frame(request, t4=t4)
  Render-->>Core: VisualizationResult
  Core->>Core: t2 = perf_counter()
  Core->>Core: elapsed_ms = t2 - t0
  Core->>Core: tier4_load_ms = t1 - t0
  Core->>Core: render_ms = t2 - t1
  Core->>Core: Encode image bytes to base64
  Core-->>Route: RenderResponse + X-Server-* + Server-Timing headers
```

---

## 6. `render_frame()` internals

The HTTP layer does not draw images itself. The actual rendering lives in `t4_visualizer/visualize.py`.

```mermaid
sequenceDiagram
  autonumber
  participant Server as _run_render
  participant RF as render_frame
  participant Scene as find_sample_by_scene_and_index
  participant Static as visualize_static
  participant Plot as _plot_combined
  participant Camera as _fill_camera_axes
  participant BEV as _fill_bev_ax
  participant Files as temp PNG files

  Server->>RF: render_frame(request, t4=cached_t4)
  RF->>Scene: find sample for scenario_name + frame_index
  Scene-->>RF: sample
  RF->>RF: create temp directory
  RF->>Static: visualize_static(t4, sample, ...)
  Static->>Static: choose camera channels
  Static->>Static: choose lidar channel
  Static->>Static: resolve target annotation tokens
  Static->>Plot: _plot_combined(...)

  alt crop camera mode with visible ROI
    Plot->>Plot: _group_objects_by_camera()
    Plot->>Camera: render selected camera crop(s)
    Plot->>BEV: render BEV for each output figure
  else normal combined view
    Plot->>Camera: render camera grid and 2D overlays
    Plot->>BEV: render LiDAR BEV and 3D boxes
  end

  Camera->>Camera: t4.get_sample_data(camera_token, as_3d=False[, as_sensor_coord=True])
  Camera->>Camera: image read + draw dataset / target overlays
  BEV->>BEV: t4.get_sample_data(lidar_token, as_3d=True[, as_sensor_coord=True])
  BEV->>BEV: _load_pointcloud(...)
  BEV->>BEV: draw point cloud + annotation boxes + target markers

  Plot-->>Files: save *_visualization*.png
  RF->>Files: read all PNG bytes
  RF-->>Server: VisualizationResult(images, sample_token, timestamp_us)
```

---

## 7. Error and timing behavior

```mermaid
flowchart TD
  A["Incoming render request"] --> B{"Dataset found?"}
  B -- no --> E1["404 dataset_not_found"]
  B -- yes --> C{"Scene/frame valid?"}
  C -- bad scene --> E2["404 scenario_not_found"]
  C -- bad frame --> E3["400 frame_index_out_of_range"]
  C -- yes --> D{"Unexpected render exception?"}
  D -- yes --> E4["500 render_failed\n(debug text only in debug visibility mode)"]
  D -- no --> F["200 OK\nJSON or HTML + X-Server-* + Server-Timing"]
```

Timing headers are generated only after a successful render:

- `X-Server-Elapsed-Ms`
- `X-Server-Tier4-Load-Ms`
- `X-Server-Render-Ms`
- `Server-Timing`

---

## 8. Mental model

- `_DatasetPathCache` avoids rescanning the data directory for repeated dataset IDs.
- `_Tier4Cache` avoids reloading `Tier4(...)` for repeated renders against the same dataset path.
- `_run_render()` is the HTTP-to-render adapter.
- `render_frame()` is the pure programmatic rendering entry point.
- `visualize_static()` and `_plot_combined()` do the actual matplotlib drawing.
- `POST /render` adds `target_objects`; `GET /render` is the same core flow but without that body payload.
- Embedded evaluation iframes with external `pred` / `gt` boxes use `/viewer/three` plus `postMessage`, and optionally `POST /viewer/three/camera-overlay` for 2D projection.
