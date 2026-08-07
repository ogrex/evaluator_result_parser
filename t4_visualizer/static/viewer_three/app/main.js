import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ColladaLoader } from "three/addons/loaders/ColladaLoader.js";
import { intensityToRgb, intensityToRgbRaw } from "./colormaps.js";
import { decodeBboxLayersBinaryV1 } from "./bbox_binary.js";
import {
  BOX_EDGE_PAIRS,
  boxCenterFromCornersArray,
  boxCornersFromPose,
  boxEdgeSegmentsFromCorners,
  boxSizeFromCornersArray,
} from "./box_geometry.js";
const params = new URLSearchParams(window.__T4_VIEWER_QS__ || window.location.search);
const dataset = params.get("t4dataset_id");
const scenario = params.get("scenario_name");
const startFrame = Number(params.get("frame_index") || "0");
const version = params.get("version");
const initialSessionId = params.get("session_id") || "";
const initialCompareViewMode = (() => {
  const raw = String(params.get("compare_view") || params.get("compare_mode") || "").toLowerCase();
  if (raw === "side_by_side" || raw === "side-by-side" || raw === "sidebyside") return "side_by_side";
  if (raw === "curtain") return "curtain";
  if (raw === "overlay") return "overlay";
  return "overlay";
})();
const initialHidePanels = (() => {
  const raw = String(params.get("hide_panels") || params.get("embed") || "").toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes" || raw === "on";
})();
const qv = version ? `&version=${encodeURIComponent(version)}` : "";
const metaUrl = `/viewer/three/meta?t4dataset_id=${encodeURIComponent(dataset)}&scenario_name=${encodeURIComponent(scenario)}${qv}`;
// Debug instrumentation (console traces, server ack round-trips) is opt-in via
// ?debug=1 — it must not run as a side effect of normal embedding.
const viewerDebugEnabled = (() => {
  const raw = String(params.get("debug") || "").toLowerCase();
  return raw === "1" || raw === "true" || raw === "yes";
})();
function dbg(...args){
  if (viewerDebugEnabled) console.info("[viewer-debug]", ...args);
}

/** GT/Pred from postMessage use center+size+yaw; dataset boxes use API corners. Extra yaw (rad) aligns typical eval exports with T4 ego (+x forward, +y left). Omit query param for +π/2; use `external_bbox_yaw_offset=0` if your JSON already matches T4. */
function getExternalBboxYawOffset(){
  if (!params.has("external_bbox_yaw_offset")) return Math.PI / 2;
  const n = Number(params.get("external_bbox_yaw_offset"));
  return Number.isFinite(n) ? n : Math.PI / 2;
}
/** Swap length↔width for center-based boxes (`external_bbox_swap_lw=1`) when exporter names dimensions opposite to T4. */
function getExternalBboxSwapLW(){
  const v = (params.get("external_bbox_swap_lw") || "").toLowerCase();
  return v === "1" || v === "true" || v === "yes";
}

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
// The scene ground follows the theme; without this the canvas clears to black.
renderer.setClearColor(TH.hex("sceneBg"), 1);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(TH.hex("fog"), 90, 230);
const perspCamera = new THREE.PerspectiveCamera(74, 1, 0.1, 1000);
const orthoCamera = new THREE.OrthographicCamera(-20, 20, 20, -20, 0.1, 1000);
let camera = perspCamera;
camera.position.set(-12, -8, 4.5);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0,0,0); controls.update();

function syncAdvRangeOutputs(){
  const setOut = (rangeId, outId, decimals) => {
    const r = document.getElementById(rangeId);
    const o = document.getElementById(outId);
    if (!r || !o) return;
    o.textContent = decimals != null ? Number(r.value).toFixed(decimals) : String(r.value);
  };
  setOut("fogNear", "fogNearOut", null);
  setOut("fogFar", "fogFarOut", null);
  setOut("camNear", "camNearOut", 2);
  setOut("camFar", "camFarOut", null);
}

function applyFogFromSettings(){
  const en = document.getElementById("advFogEnable");
  if (!en || !en.checked) {
    scene.fog = null;
    return;
  }
  const near = Number(document.getElementById("fogNear")?.value ?? 90);
  let far = Number(document.getElementById("fogFar")?.value ?? 230);
  if (far <= near + 1) far = near + 2;
  scene.fog = new THREE.Fog(TH.hex("fog"), near, far);
}

/**
 * Re-apply the theme to everything in the scene that caches a color.
 *
 * A theme flip changes the ground the scene is drawn on, and three.js bakes colors
 * into materials, grid geometry and canvas textures, so those have to be rebuilt --
 * CSS cannot reach any of it.
 */
function applySceneTheme(){
  syncDefaultColormap();
  renderer.setClearColor(TH.hex("sceneBg"), 1);
  applyFogFromSettings();
  if (typeof ambientLight !== "undefined" && ambientLight) {
    ambientLight.color.setHex(TH.hex("ambient"));
  }
  if (grid) {
    scene.remove(grid);
    grid.geometry.dispose();
    if (Array.isArray(grid.material)) grid.material.forEach((m) => m.dispose());
    else grid.material.dispose();
    grid = new THREE.GridHelper(180, 90, TH.hex("gridMajor"), TH.hex("gridMinor"));
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
  }
  // Label chips are cached per (text, color, theme); drop the stale-theme entries.
  refreshEvalStatusColors();
  if (typeof egoFallback !== "undefined" && egoFallback && egoFallback.material) {
    egoFallback.material.color.setHex(TH.hex("egoWire"));
  }
  labelSpriteCache.clear();
  // Rebuilds boxes, labels and per-point colormap colors for the current frame.
  const data = cache.get(frame);
  if (data) setFrameData(data);
  updateColorbar();
}

function applyCameraClipFromSettings(){
  const cn = Number(document.getElementById("camNear")?.value ?? 0.1);
  let cf = Number(document.getElementById("camFar")?.value ?? 1000);
  if (cf <= cn + 0.01) cf = cn + 1;
  perspCamera.near = cn;
  perspCamera.far = cf;
  perspCamera.updateProjectionMatrix();
  orthoCamera.near = cn;
  orthoCamera.far = cf;
  orthoCamera.updateProjectionMatrix();
}

function fxEnabled(id){
  const master = document.getElementById("fxEnableEmphasis");
  if (!master || !master.checked) return false;
  const el = document.getElementById(id);
  return !!(el && el.checked);
}

const ambientLight = new THREE.AmbientLight(TH.hex("ambient"), 0.34);
scene.add(ambientLight);
const dl = new THREE.DirectionalLight(0xffffff, 0.72); dl.position.set(12, -10, 30); scene.add(dl);
// Sky/ground gradient + opposite-side fill so the ego mesh reads as a solid form
// instead of a flat silhouette. Only the ego mesh is lit; everything else is
// MeshBasicMaterial and unaffected.
const egoHemiLight = new THREE.HemisphereLight(0xe6eef7, 0x30353c, 0.6);
scene.add(egoHemiLight);
const egoFillLight = new THREE.DirectionalLight(0xffffff, 0.3);
egoFillLight.position.set(-16, 12, 14);
scene.add(egoFillLight);
let grid = new THREE.GridHelper(180, 90, TH.hex("gridMajor"), TH.hex("gridMinor"));
grid.rotation.x = Math.PI / 2; // XY ground plane (z-up)
scene.add(grid);

const pointsGeom = new THREE.BufferGeometry();
// Opaque at full alpha so the cloud renders in the opaque queue and writes
// depth. As a transparent object it would be depth-sorted by its single
// centroid, which makes the whole cloud paint over the ghosted ego from far
// away and behave correctly only up close.
const pointsMat = new THREE.PointsMaterial({size:0.08, vertexColors:true, transparent:false, opacity:1.0});
/** Keep the transparent flag in step with alpha; opaque is the correct default. */
function applyPointOpacity(v){
  pointsMat.opacity = v;
  pointsMat.transparent = v < 0.99;
  pointsMat.depthWrite = true;
  pointsMat.needsUpdate = true;
}
const pointsObj = new THREE.Points(pointsGeom, pointsMat);
// If alpha is lowered the cloud rejoins the transparent queue; draw it first
// there too, so it can never paint over the ego regardless of centroid sorting.
pointsObj.renderOrder = -3;
scene.add(pointsObj);
const boxesGroup = new THREE.Group(); scene.add(boxesGroup);
const gtLayerGroup = new THREE.Group(); scene.add(gtLayerGroup);
const predLayerGroup = new THREE.Group(); scene.add(predLayerGroup);
const inspectOverlayGroup = new THREE.Group(); scene.add(inspectOverlayGroup);
const inspectTrailGroup = new THREE.Group(); scene.add(inspectTrailGroup);
gtLayerGroup.renderOrder = 0;
predLayerGroup.renderOrder = 1;
inspectOverlayGroup.renderOrder = 3;
inspectTrailGroup.renderOrder = 2;
const laneletGroup = new THREE.Group(); scene.add(laneletGroup);
const egoAxes = new THREE.AxesHelper(2.8); scene.add(egoAxes);
/** Ego vehicle root (base_link); mesh from sample_vehicle_description or wireframe fallback. */
const egoRoot = new THREE.Group();
scene.add(egoRoot);
const egoFallback = new THREE.Mesh(
  new THREE.BoxGeometry(4.6, 1.9, 1.6),
  new THREE.MeshBasicMaterial({
    color: TH.hex("egoWire"),
    wireframe: true,
    transparent: true,
    opacity: Number(document.getElementById("egoOpacity")?.value || "0.5"),
  })
);
egoFallback.position.set(0, 0, 0.8);
egoRoot.add(egoFallback);
const EGO_WHEEL_BASE = 2.79;
let egoContactShadow = null;
/** Current ego alpha. Translucent by default so the vehicle never hides the
 *  detections being evaluated (rviz uses 0.3; 0.5 keeps the form readable). */
function egoOpacityValue(){
  return Number(document.getElementById("egoOpacity")?.value || "0.5");
}
function applyEgoOpacity(opacity){
  const solid = opacity >= 0.99;
  egoRoot.traverse((o) => {
    if (!o.isMesh || !o.material) return;
    // The contact shadow keeps its own alpha and fades with the vehicle.
    if (o.userData && o.userData.isEgoShadow) {
      o.material.opacity = 0.6 * Math.min(1, opacity / 0.5);
      return;
    }
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      if (!m) continue;
      m.transparent = !solid;
      m.opacity = solid ? 1 : opacity;
      // Translucent ego must not depth-occlude: with depthWrite on, anything
      // drawn later behind the vehicle would vanish instead of showing through.
      if ("depthWrite" in m) m.depthWrite = solid;
    }
  });
}
/** Bundled sample (lexus) mesh descriptor; used if /viewer/assets/vehicle-model.json is unreachable. */
const EGO_MESH_FALLBACK = {
  source: "sample",
  url: "/viewer/assets/vehicle-mesh/lexus.dae",
  rotation: [-Math.PI / 2, 0, Math.PI],
  offset: [EGO_WHEEL_BASE * 0.5, 0, 0],
};
/** Ask the server which ego mesh to use (a confidential custom mesh may be installed). */
async function fetchEgoMeshDescriptor(){
  try {
    const res = await fetch("/viewer/assets/vehicle-model.json", {cache: "no-store"});
    if (!res.ok) throw new Error("HTTP " + res.status);
    const d = await res.json();
    if (!d || typeof d.url !== "string") throw new Error("malformed descriptor");
    return {
      source: d.source || "custom",
      url: d.url,
      rotation: Array.isArray(d.rotation) && d.rotation.length === 3 ? d.rotation : [0, 0, 0],
      offset: Array.isArray(d.offset) && d.offset.length === 3 ? d.offset : [0, 0, 0],
    };
  } catch (err) {
    console.warn("Ego mesh descriptor unavailable, using bundled sample mesh:", err);
    return EGO_MESH_FALLBACK;
  }
}
/** Per-part shading for meshes that ship no materials, matched on node name. */
const EGO_PART_STYLES = [
  // Match "wheel", not "tire": this file's geometry names are all `TIRE-FR.*`
  // regardless of the actual part, so "tire" would misclassify the body.
  {test: /wheel/i,                  color: 0x16181b, roughness: 0.94, metalness: 0.0},
  {test: /stoplight|rear_light/i,   color: 0x7e1c1c, roughness: 0.35, metalness: 0.0, emissive: 0x2c0707},
  {test: /turn(left|right)?_light/i, color: 0xa8621b, roughness: 0.35, metalness: 0.0, emissive: 0x2a1704},
  {test: /white_light|head_light/i, color: 0xdde3e8, roughness: 0.18, metalness: 0.15, emissive: 0x1d2124},
];
// Mid neutral rather than near-white: a white body has almost no value
// separation from the light theme's cream background.
const EGO_BODY_STYLE = {color: 0xc2c7cd, roughness: 0.46, metalness: 0.10};
/** Soft contact shadow so the vehicle sits on the ground plane instead of floating. */
function makeEgoContactShadow(){
  const size = 256;
  const cv = document.createElement("canvas");
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d");
  const g = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  g.addColorStop(0.0, "rgba(0,0,0,0.40)");
  g.addColorStop(0.5, "rgba(0,0,0,0.17)");
  g.addColorStop(1.0, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  // Scene is z-up and PlaneGeometry is already in XY, so no rotation needed.
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(9.6, 4.2),
    new THREE.MeshBasicMaterial({map: tex, transparent: true, depthWrite: false, opacity: 0.6})
  );
  mesh.position.set(2.1, 0, 0.02); // centred on the body, just above z=0
  mesh.renderOrder = -1;
  mesh.userData.isEgoShadow = true;
  return mesh;
}
/** Re-derive vertex normals with a crease angle.
 *
 * Low-poly exports often average normals across sharp edges, which shows up as
 * blotchy angular wedges on flat panels. Splitting normals at creases fixes it.
 * Optional: on failure the mesh just keeps its original normals.
 */
async function creaseEgoNormals(root){
  try {
    const {toCreasedNormals} = await import("three/addons/utils/BufferGeometryUtils.js");
    root.traverse((o) => {
      if (!o.isMesh || !o.geometry) return;
      const next = toCreasedNormals(o.geometry, Math.PI / 5); // 36°
      o.geometry.dispose();
      o.geometry = next;
    });
  } catch (err) {
    console.warn("Could not recompute creased normals for the ego mesh:", err);
  }
}
/** Pick a style from the mesh's own name plus its ancestors' names.
 *
 * ColladaLoader puts a node's name on the mesh itself when the node holds a
 * single geometry, and on an ancestor group otherwise, so check the whole chain.
 * Unmatched parts get the body style, which is the safe default.
 */
function egoPartStyle(obj){
  const names = [];
  for (let o = obj; o; o = o.parent) { if (o.name) names.push(o.name); }
  const joined = names.join("/");
  for (const s of EGO_PART_STYLES) { if (s.test.test(joined)) return s; }
  return EGO_BODY_STYLE;
}
/** Studio environment for reflections; optional, so a fetch failure just degrades. */
let egoEnvPromise = null;
function ensureEgoEnvironment(){
  if (egoEnvPromise) return egoEnvPromise;
  egoEnvPromise = (async () => {
    try {
      const {RoomEnvironment} = await import("three/addons/environments/RoomEnvironment.js");
      const pmrem = new THREE.PMREMGenerator(renderer);
      const env = pmrem.fromScene(new RoomEnvironment(), 0.04);
      // Only the ego mesh uses a lit material, so this affects nothing else.
      scene.environment = env.texture;
      pmrem.dispose();
    } catch (err) {
      console.warn("Ego environment map unavailable; using lights only:", err);
    }
  })();
  return egoEnvPromise;
}
function loadEgoVehicleMeshFrom(desc){
  const loader = new ColladaLoader();
  return new Promise((resolve, reject) => {
    loader.load(
      desc.url,
      (collada) => {
        const v0 = egoOpacityValue();
        if (egoFallback.parent) {
          egoRoot.remove(egoFallback);
          egoFallback.geometry.dispose();
          egoFallback.material.dispose();
        }
        const visual = new THREE.Group();
        visual.position.set(desc.offset[0], desc.offset[1], desc.offset[2]);
        const meshRoot = collada.scene;
        meshRoot.rotation.set(desc.rotation[0], desc.rotation[1], desc.rotation[2]);
        const solid = v0 >= 0.9;
        meshRoot.traverse((o) => {
          if (!o.isMesh || !o.material) return;
          const mats = Array.isArray(o.material) ? o.material : [o.material];
          const out = mats.map((m) => {
            // Textured mesh (the bundled sample): keep its authored material.
            if (m && m.map) {
              m.map.colorSpace = THREE.SRGBColorSpace;
              if ("color" in m) m.color.setRGB(1, 1, 1);
              return m;
            }
            // Untextured: the Collada may carry no effects at all, in which case
            // every part shares one flat default. Shade parts by node name instead.
            const st = egoPartStyle(o);
            const nm = new THREE.MeshStandardMaterial({
              color: st.color,
              roughness: st.roughness,
              metalness: st.metalness,
              emissive: st.emissive === undefined ? 0x000000 : st.emissive,
              envMapIntensity: 0.35, // restrained: high values exaggerate facets
            });
            if (m && m.dispose) m.dispose();
            return nm;
          });
          o.material = Array.isArray(o.material) ? out : out[0];
          for (const m of out) {
            if (!m) continue;
            m.transparent = !solid;
            m.opacity = solid ? 1 : v0;
            if ("depthWrite" in m) m.depthWrite = true;
          }
        });
        visual.add(meshRoot);
        egoRoot.add(visual);
        if (!egoContactShadow) {
          egoContactShadow = makeEgoContactShadow();
          egoRoot.add(egoContactShadow);
        }
        ensureEgoEnvironment();
        creaseEgoNormals(meshRoot);
        applyEgoOpacity(v0);
        resolve();
      },
      undefined,
      (err) => reject(err || new Error("Collada load failed"))
    );
  });
}
/** Load the custom ego mesh when installed, otherwise the bundled sample mesh. */
async function loadEgoVehicleMesh(){
  const desc = await fetchEgoMeshDescriptor();
  try {
    await loadEgoVehicleMeshFrom(desc);
  } catch (err) {
    if (desc.url === EGO_MESH_FALLBACK.url) throw err;
    console.warn("Custom ego mesh failed to load, falling back to sample mesh:", err);
    await loadEgoVehicleMeshFrom(EGO_MESH_FALLBACK);
  }
}
const cache = new Map(); const MAX_CACHE = 100;
// Frame cache is byte-budgeted as well as count-capped: dense LiDAR frames run
// to several MB each, so 100 frames of raw Float32Arrays could hold hundreds
// of MB of JS heap. Eviction is LRU (hits refresh recency in fetchFrame).
const CACHE_BYTE_BUDGET = 256 * 1024 * 1024;
let cacheBytes = 0;
const inflightFrames = new Map();  // frame index → Promise, dedups concurrent fetches
function frameByteSize(parsed){
  return ((parsed && parsed.pts && parsed.pts.byteLength) || 0)
    + ((parsed && parsed.boxes && parsed.boxes.byteLength) || 0)
    + 4096;  // header/labels/object overhead estimate
}
function cacheStoreFrame(i, parsed){
  if (cache.has(i)) cacheBytes -= frameByteSize(cache.get(i));
  cache.delete(i);
  cache.set(i, parsed);
  cacheBytes += frameByteSize(parsed);
  while (cache.size > 1 && (cache.size > MAX_CACHE || cacheBytes > CACHE_BYTE_BUDGET)) {
    const k = cache.keys().next().value;
    cacheBytes -= frameByteSize(cache.get(k));
    cache.delete(k);
  }
}
let totalFrames = 0; let playing = false; let frame = startFrame; let fps = 6;
let lastFrameTs = 0;
let followEgo = true;
let intensityGain = 1.0;
let inspectLockFocus = true;
let cameraViewportEnabled = false;
/** Last `/viewer/three/camera-overlay` JSON; used to redraw on resize / 2D toggle. */
let lastCameraPayload = null;
let lastCameraPanelCheckbox = false;
/** Bumped when the user turns the camera panel off or starts a new overlay fetch (drops stale async work). */
let cameraOverlayGeneration = 0;
const cameraCanvasHitRegions = new WeakMap();
const inspectRaycaster = new THREE.Raycaster();
inspectRaycaster.params.Line = { threshold: 0.5 };
const inspectPointer = new THREE.Vector2();
let pointerDownState = null;
let sceneSelectionCandidates = [];
let externalSelectionCandidates = [];
let selectedInspectState = null;
let inspectTrackGeneration = 0;
let spotlightFrames = [];
let spotlightIndex = -1;
let spotlightTouring = false;
let spotlightLastSwitchTs = 0;
let spotlightJumpInFlight = false;
let compareCurtainRatio = 0.5;
let compareRenderRunFilter = null;
let compareCurtainDragging = false;
const EVAL_FAST_RENDER_BOX_THRESHOLD = 450;
const SCENE_BOX_BATCH_THRESHOLD = 180;
const MAX_RENDER_POINTS = 180000;
let evalFastRenderMode = false;
/** Cap on how many eval-box text-label sprites we draw per frame (keeps the Labels toggle smooth on dense scenes). */
const LABEL_SPRITE_CAP = 220;
let labelBudget = 0;
let labelableCount = 0;
let hoverTipRaf = 0;
let hoverTipLastEvent = null;
const _hoverProjV = new THREE.Vector3();


/** Set once the operator picks a colormap themselves; suppresses the theme default. */
let pointColormapChosen = false;

/**
 * Follow the theme's default colormap until the operator picks one.
 *
 * Turbo reads well on the dark ground but is a rainbow whose mid-range goes pale,
 * so on paper the default becomes the single-hue "paper" ramp instead.
 */
function syncDefaultColormap(){
  if (pointColormapChosen) return false;
  const el = document.getElementById("pointColormap");
  if (!el) return false;
  const want = (window.TH && TH.isLight()) ? "paper" : "turbo";
  if (el.value === want) return false;
  el.value = want;
  return true;
}

function getPointColormapName(){
  const el = document.getElementById("pointColormap");
  return (el && el.value) ? el.value : "turbo";
}

function getPointIntensityNormalize(){
  const el = document.getElementById("pointIntensityNormalize");
  return !!(el && el.checked);
}
let laneletEnabled = false;
let laneletLoaded = false;
let laneletBounds = null;
let laneletFrame = -1;
let externalLayers = { gt: [], pred: [] };
/** When set, eval GT/pred layers are keyed by frame index (from parent postMessage). */
let bboxLayersByFrame = null;
let compareRunOrder = [];
/** Optional per-frame series from parent: `{ gt_tp, gt_fn, est_tp, est_fp }` or legacy `{ gt, pred, tpr }`. */
let metricsSeriesOverride = null;
let activeViewerSessionId = initialSessionId || "";
let activeViewerSessionSource = "";

function normalizeMetricsSeriesPayload(obj){
  const d = obj && typeof obj === "object" ? obj : {};
  return {
    gt_tp: Array.isArray(d.gt_tp) ? d.gt_tp : null,
    gt_fn: Array.isArray(d.gt_fn) ? d.gt_fn : null,
    est_tp: Array.isArray(d.est_tp) ? d.est_tp : null,
    est_fp: Array.isArray(d.est_fp) ? d.est_fp : null,
    tp_center_distance_mean: Array.isArray(d.tp_center_distance_mean) ? d.tp_center_distance_mean : null,
    tp_plane_distance_mean: Array.isArray(d.tp_plane_distance_mean) ? d.tp_plane_distance_mean : null,
    tp_yaw_error_abs_mean: Array.isArray(d.tp_yaw_error_abs_mean) ? d.tp_yaw_error_abs_mean : null,
    frame_severity_max: Array.isArray(d.frame_severity_max) ? d.frame_severity_max : null,
    gt: Array.isArray(d.gt) ? d.gt : [],
    pred: Array.isArray(d.pred) ? d.pred : [],
    tpr: Array.isArray(d.tpr) ? d.tpr : null,
  };
}


function looksLikeFrameMap(obj){
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return false;
  const keys = Object.keys(obj);
  if (!keys.length) return false;
  return keys.every((k) => /^\d+$/.test(String(k)));
}

function normalizeViewerSessionPayload(raw, sourceName){
  const root0 = raw && typeof raw === "object" ? raw : {};
  const root = root0.payload && typeof root0.payload === "object" ? root0.payload : root0;
  const out = {};
  let byFrame = null;
  if (root.type === "bbox_layers_by_frame" && root.frames && typeof root.frames === "object") byFrame = root.frames;
  else if (root.bbox_layers_by_frame && typeof root.bbox_layers_by_frame === "object") byFrame = root.bbox_layers_by_frame;
  else if (root.frames && typeof root.frames === "object" && !Array.isArray(root.frames)) byFrame = root.frames;
  else if (looksLikeFrameMap(root)) byFrame = root;
  if (byFrame) out.bbox_layers_by_frame = byFrame;

  let single = null;
  if (root.type === "bbox_layers") single = root;
  else if (root.bbox_layers && typeof root.bbox_layers === "object") single = root.bbox_layers;
  else if (Array.isArray(root.gt) || Array.isArray(root.pred)) single = root;
  if (single) {
    out.bbox_layers = {
      gt: Array.isArray(single.gt) ? single.gt : [],
      pred: Array.isArray(single.pred) ? single.pred : [],
    };
  }

  const metrics = root.type === "eval_metrics_series"
    ? root
    : (root.eval_metrics_series || root.metricsSeries || root.metrics_series || null);
  if (metrics && typeof metrics === "object") out.eval_metrics_series = normalizeMetricsSeriesPayload(metrics);
  if (Array.isArray(root.compare_runs)) out.compare_runs = root.compare_runs.map((v) => String(v || "").trim()).filter(Boolean);

  if (!Object.keys(out).length) {
    throw new Error("JSON must contain bbox_layers_by_frame, bbox_layers, eval_metrics_series, or a frame map.");
  }
  return {
    payload: out,
    source_name: String(root0.source_name || sourceName || "").trim() || null,
  };
}

function updateViewerSessionUrl(sessionId){
  if (sessionId) params.set("session_id", sessionId);
  else params.delete("session_id");
  const next = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, "", next);
}

function clearCurrentViewerLayers(opts){
  const doCameraRefresh = !opts || opts.refreshCamera !== false;
  const clearMetrics = !opts || opts.clearMetrics !== false;
  bboxLayersByFrame = null;
  compareRunOrder = [];
  if (clearMetrics) metricsSeriesOverride = null;
  externalLayers = { gt: [], pred: [] };
  renderExternalLayers();
  updateEvalHud();
  syncSpotlightFrames();
  rebuildAllMetricsPlots();
  if (doCameraRefresh && cameraViewportEnabled) refreshCameraOverlay().catch(() => {});
}

function applyViewerSessionPayload(payload, opts){
  const doCameraRefresh = !opts || opts.refreshCamera !== false;
  const norm = payload && typeof payload === "object" ? payload : {};
  compareRunOrder = Array.isArray(norm.compare_runs) ? norm.compare_runs.map((v) => String(v || "").trim()).filter(Boolean) : [];
  metricsSeriesOverride = norm.eval_metrics_series ? normalizeMetricsSeriesPayload(norm.eval_metrics_series) : null;
  if (norm.bbox_layers_by_frame && typeof norm.bbox_layers_by_frame === "object") {
    bboxLayersByFrame = norm.bbox_layers_by_frame;
    applyEvalLayersForFrame(frame);
  } else if (norm.bbox_layers && typeof norm.bbox_layers === "object") {
    bboxLayersByFrame = null;
    setExternalLayerPayload(norm.bbox_layers, { ack: false, refreshCamera: false });
  } else {
    bboxLayersByFrame = null;
    externalLayers = { gt: [], pred: [] };
    renderExternalLayers();
    updateEvalHud();
  }
  syncSpotlightFrames();
  rebuildAllMetricsPlots();
  if (doCameraRefresh && cameraViewportEnabled) refreshCameraOverlay().catch(() => {});
}

function buildCurrentViewerSessionPayload(){
  const payload = {};
  if (bboxLayersByFrame && typeof bboxLayersByFrame === "object" && Object.keys(bboxLayersByFrame).length) {
    payload.bbox_layers_by_frame = bboxLayersByFrame;
    if (compareRunOrder.length >= 2) payload.compare_runs = compareRunOrder;
  } else {
    const gt = Array.isArray(externalLayers.gt) ? externalLayers.gt : [];
    const pred = Array.isArray(externalLayers.pred) ? externalLayers.pred : [];
    if (gt.length || pred.length) payload.bbox_layers = { gt, pred };
  }
  if (metricsSeriesOverride && typeof metricsSeriesOverride === "object") {
    payload.eval_metrics_series = normalizeMetricsSeriesPayload(metricsSeriesOverride);
  }
  return Object.keys(payload).length ? payload : null;
}

function downloadViewerSessionJson(){
  const payload = buildCurrentViewerSessionPayload();
  if (!payload) {
    setStatus("save json: no imported or external overlay data to export");
    return;
  }
  const doc = {
    source_name: activeViewerSessionSource || null,
    exported_at: new Date().toISOString(),
    viewer_context: {
      t4dataset_id: dataset,
      scenario_name: scenario,
      frame_index: frame,
      version,
      session_id: activeViewerSessionId || null,
    },
    payload,
  };
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const safeDataset = String(dataset || "dataset").replace(/[^a-z0-9._-]+/gi, "_");
  const safeScenario = String(scenario || "scenario").replace(/[^a-z0-9._-]+/gi, "_");
  a.href = url;
  a.download = `${safeDataset}_${safeScenario}_frame${frame}_viewer_session.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
  setStatus(`saved json ${a.download}`);
}

async function parseJsonResponseOrThrow(res, fallback){
  const body = await res.json().catch(() => null);
  if (res.ok) return body;
  const msg = body && body.detail && body.detail.message
    ? body.detail.message
    : (body && body.message ? body.message : fallback);
  throw new Error(msg || fallback || `HTTP ${res.status}`);
}

async function loadViewerSessionById(sessionId){
  const sid = String(sessionId || "").trim();
  if (!sid) return null;
  const res = await fetch(`/viewer/three/session/${encodeURIComponent(sid)}`);
  const record = await parseJsonResponseOrThrow(res, `viewer session HTTP ${res.status}`);
  const normalized = normalizeViewerSessionPayload(record, record && record.source_name ? String(record.source_name) : "");
  activeViewerSessionId = String(record && record.session_id ? record.session_id : sid);
  activeViewerSessionSource = normalized.source_name || "";
  applyViewerSessionPayload(normalized.payload, { refreshCamera: false });
  setShareSessionState(`session ${activeViewerSessionId.slice(0, 8)}`);
  updateViewerSessionUrl(activeViewerSessionId);
  return record;
}

// One shared offscreen WebGL renderer draws all three metrics charts, blitted
// onto plain 2D canvases. Browsers cap live WebGL contexts (~8-16, oldest is
// silently lost); dedicating three contexts to 2D line charts risked killing
// the main scene's context. The chart scenes/cameras are unchanged.
const chartRendererCanvas = document.createElement("canvas");
const chartRenderer = new THREE.WebGLRenderer({ canvas: chartRendererCanvas, antialias: true, alpha: true });
chartRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
chartRenderer.setClearColor(0x000000, 0);
chartRenderer.outputColorSpace = THREE.SRGBColorSpace;
chartRenderer.toneMapping = THREE.NoToneMapping;
const chartSizes = new Map();  // target canvas → {w, h} CSS px, set by resizeMetricsRenderer
function renderChartToCanvas(scene, camera, targetCanvas){
  const size = chartSizes.get(targetCanvas);
  if (!size || !size.w || !size.h) return;
  chartRenderer.setSize(size.w, size.h, false);
  chartRenderer.render(scene, camera);
  if (targetCanvas.width !== chartRendererCanvas.width || targetCanvas.height !== chartRendererCanvas.height) {
    targetCanvas.width = chartRendererCanvas.width;
    targetCanvas.height = chartRendererCanvas.height;
  }
  const ctx = targetCanvas.getContext("2d");
  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.drawImage(chartRendererCanvas, 0, 0);
}

const metricsCanvas = document.getElementById("metricsCanvas");
const metricsWrapEl = document.getElementById("metricsWrap");
const metricsScene = new THREE.Scene();
metricsScene.background = null;
const metricsCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
metricsCamera.position.set(0, 0, 3);
let metricsChartGroup = new THREE.Group();
metricsScene.add(metricsChartGroup);
let metricsPlayhead = null;

const metricsRatesCanvas = document.getElementById("metricsRatesCanvas");
const metricsRatesWrapEl = document.getElementById("metricsRatesWrap");
const metricsRatesScene = new THREE.Scene();
metricsRatesScene.background = null;
const metricsRatesCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
metricsRatesCamera.position.set(0, 0, 3);
let metricsRatesChartGroup = new THREE.Group();
metricsRatesScene.add(metricsRatesChartGroup);
let metricsRatesPlayhead = null;

const metricsErrorCanvas = document.getElementById("metricsErrorCanvas");
const metricsErrorWrapEl = document.getElementById("metricsErrorWrap");
const metricsErrorScene = new THREE.Scene();
metricsErrorScene.background = null;
const metricsErrorCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 20);
metricsErrorCamera.position.set(0, 0, 3);
let metricsErrorChartGroup = new THREE.Group();
metricsErrorScene.add(metricsErrorChartGroup);
let metricsErrorPlayhead = null;
const metricsCompareWrapEl = document.getElementById("metricsCompareWrap");

const slider = document.getElementById("slider");
const frameTxt = document.getElementById("frameTxt");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const evalHudEl = document.getElementById("evalHud");
const appRootEl = document.getElementById("app");
const shareSessionStateEl = document.getElementById("shareSessionState");
const shareSessionUrlEl = document.getElementById("shareSessionUrl");
const viewerSessionFileInput = document.getElementById("viewerSessionFileInput");
const compareViewModeEl = document.getElementById("compareViewMode");
if (compareViewModeEl) compareViewModeEl.value = initialCompareViewMode;
if (initialHidePanels) {
  ["uiShowHud", "uiShowInspector", "uiShowMetricsCounts", "uiShowMetricsRates", "uiShowMetricsError", "uiShowCameraViewport"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.checked = false;
  });
}
function setStatus(t){ statusEl.textContent = t; }
function setFrameText(){ frameTxt.textContent = `frame ${frame}/${Math.max(0,totalFrames-1)}`; slider.value = String(frame); }
function setShareSessionState(t){ if (shareSessionStateEl) shareSessionStateEl.textContent = t; }
function revealShareSessionUrl(url){
  if (!shareSessionUrlEl) return;
  shareSessionUrlEl.hidden = false;
  shareSessionUrlEl.value = String(url || "");
  shareSessionUrlEl.title = String(url || "");
  shareSessionUrlEl.focus();
  shareSessionUrlEl.select();
}
if (shareSessionUrlEl) {
  shareSessionUrlEl.addEventListener("click", () => {
    shareSessionUrlEl.focus();
    shareSessionUrlEl.select();
  });
}
setShareSessionState(initialSessionId ? `session ${initialSessionId.slice(0, 8)}` : "session local-only");
function fmtNum(v, d){
  return Number.isFinite(v) ? Number(v).toFixed(d) : "—";
}
function fmtSigned(v, d){
  return Number.isFinite(v) ? `${Number(v) >= 0 ? "+" : ""}${Number(v).toFixed(d)}` : "—";
}
function safeNum(v){
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function clip01(v){
  return Math.max(0, Math.min(1, Number(v) || 0));
}
function fmtMaybe(v, suffix, digits){
  return Number.isFinite(v) ? `${Number(v).toFixed(digits != null ? digits : 2)}${suffix || ""}` : "—";
}
function fmtVec3(v, d){
  if (!v || v.length < 3) return "—";
  return `${fmtNum(v[0], d)}, ${fmtNum(v[1], d)}, ${fmtNum(v[2], d)}`;
}
function wrapAngleRad(v){
  if (!Number.isFinite(v)) return null;
  let out = Number(v);
  while (out > Math.PI) out -= Math.PI * 2;
  while (out < -Math.PI) out += Math.PI * 2;
  return out;
}
function severityBucket(score){
  if (!Number.isFinite(score)) return "unknown";
  if (score >= 0.82) return "critical";
  if (score >= 0.62) return "high";
  if (score >= 0.35) return "medium";
  return "low";
}
function deriveEvalSeverity(kind, status, metrics){
  const preset = safeNum(metrics && metrics.severity_score);
  if (preset != null) {
    return {
      score: clip01(preset),
      bucket: severityBucket(preset),
      reason: String((metrics && (metrics.severity_reason || metrics.severity_label)) || "provided"),
    };
  }
  const st = String(status || "TP").toUpperCase();
  if (st === "FN") return { score: 0.96, bucket: "critical", reason: "missed ground truth" };
  if (st === "FP") {
    const conf = safeNum(metrics && metrics.confidence);
    const fpScore = 0.66 + 0.18 * (conf != null ? conf : 0.35);
    return { score: clip01(fpScore), bucket: severityBucket(fpScore), reason: "false positive" };
  }
  const parts = [];
  const xyMag = safeNum(metrics && metrics.xy_error_mag);
  const zErr = safeNum(metrics && metrics.z_error);
  const yawErr = safeNum(metrics && metrics.yaw_error_abs);
  const centerD = safeNum(metrics && metrics.center_distance);
  const planeD = safeNum(metrics && metrics.plane_distance);
  const dt = safeNum(metrics && metrics.pair_dt_sec_abs);
  if (xyMag != null) parts.push(["xy error", Math.min(1, xyMag / 1.8) * 0.34]);
  if (zErr != null) parts.push(["z error", Math.min(1, Math.abs(zErr) / 1.0) * 0.16]);
  if (yawErr != null) parts.push(["yaw error", Math.min(1, yawErr / 0.8) * 0.24]);
  if (centerD != null) parts.push(["center distance", Math.min(1, centerD / 1.8) * 0.16]);
  if (planeD != null) parts.push(["plane distance", Math.min(1, planeD / 1.8) * 0.12]);
  if (dt != null) parts.push(["pair dt", Math.min(1, dt / 0.1) * 0.08]);
  if (!parts.length) return { score: kind === "GT" ? 0.08 : 0.12, bucket: "low", reason: "low TP error" };
  parts.sort((a, b) => b[1] - a[1]);
  const score = clip01(0.1 + parts.reduce((sum, p) => sum + p[1], 0));
  return { score, bucket: severityBucket(score), reason: String(parts[0][0]) };
}
function normalizeEvalBox(box, layerId, frameIndexOverride){
  if (!box || typeof box !== "object") return null;
  const kind = normalizeEvalKind(box, layerId);
  const status = normalizeEvalStatus(box);
  const pose = readEvalBoxPose(box);
  const rawCorners = Array.isArray(box.corners) && box.corners.length === 24
    ? box.corners.slice(0, 24).map((v) => Number(v || 0))
    : null;
  const derivedCenter = rawCorners ? boxCenterFromCornersArray(rawCorners) : null;
  const derivedSize = rawCorners ? boxSizeFromCornersArray(rawCorners) : null;
  const vx = safeNum(box.vx);
  const vy = safeNum(box.vy);
  const xErr = safeNum(box.x_error);
  const yErr = safeNum(box.y_error);
  const zErr = safeNum(box.z_error);
  const yawErr = wrapAngleRad(safeNum(box.yaw_error));
  const centerDistance = safeNum(box.center_distance);
  const planeDistance = safeNum(box.plane_distance);
  const pairDtSec = safeNum(box.pair_dt_sec);
  const xyErrorMag = (xErr != null || yErr != null) ? Math.hypot(xErr || 0, yErr || 0) : null;
  const speedMps = (vx != null || vy != null) ? Math.hypot(vx || 0, vy || 0) : null;
  const severity = deriveEvalSeverity(kind, status, {
    severity_score: box.severity_score,
    severity_reason: box.severity_reason,
    severity_label: box.severity_label,
    confidence: box.confidence,
    xy_error_mag: xyErrorMag,
    z_error: zErr,
    yaw_error_abs: yawErr != null ? Math.abs(yawErr) : null,
    center_distance: centerDistance,
    plane_distance: planeDistance,
    pair_dt_sec_abs: pairDtSec != null ? Math.abs(pairDtSec) : null,
  });
  const corners = rawCorners || boxCornersFromPose(pose.cx, pose.cy, pose.cz, pose.l, pose.w, pose.h, pose.yaw);
  return {
    ...box,
    _normalized_eval: true,
    label: String(box.label ?? box.name ?? kind),
    kind,
    status,
    source: String(layerId === "pred" ? "eval-pred" : "eval-gt"),
    layerId,
    uuid: String(box.uuid ?? box.id ?? box.track_id ?? box.object_id ?? box.instance_token ?? ""),
    pair_uuid: String(box.pair_uuid ?? ""),
    trackId: String(box.uuid ?? box.id ?? box.track_id ?? box.object_id ?? box.instance_token ?? ""),
    frameIndex: Number.isFinite(Number(box.frame_index)) ? Number(box.frame_index) : frameIndexOverride,
    frame_id: String(box.frame_id ?? ""),
    topic_name: String(box.topic_name ?? ""),
    suite_name: String(box.suite_name ?? ""),
    scenario_name: String(box.scenario_name ?? ""),
    run: String(box.run ?? ""),
    t4dataset_id: String(box.t4dataset_id ?? ""),
    t4dataset_name: String(box.t4dataset_name ?? ""),
    unix_time: safeNum(box.unix_time),
    confidence: safeNum(box.confidence),
    vx,
    vy,
    speed_mps: speedMps,
    x_error: xErr,
    y_error: yErr,
    z_error: zErr,
    yaw_error: yawErr,
    xy_error_mag: xyErrorMag,
    center_distance: centerDistance,
    plane_distance: planeDistance,
    pair_dt_sec: pairDtSec,
    severity_score: severity.score,
    severity_bucket: severity.bucket,
    severity_reason: severity.reason,
    center: derivedCenter || [pose.cx, pose.cy, pose.cz],
    size: derivedSize || [pose.l, pose.w, pose.h],
    yaw: pose.yaw,
    corners,
  };
}
function normalizeEvalEntry(entry, frameIndexOverride){
  const raw = entry && typeof entry === "object" ? entry : {};
  return {
    ...raw,
    gt: Array.isArray(raw.gt) ? raw.gt.map((box) => normalizeEvalBox(box, "gt", frameIndexOverride)).filter(Boolean) : [],
    pred: Array.isArray(raw.pred) ? raw.pred.map((box) => normalizeEvalBox(box, "pred", frameIndexOverride)).filter(Boolean) : [],
  };
}
function getInspectableBoxMeta(box, layerId){
  if (!box || typeof box !== "object") return null;
  if (box.center && box.size && box.corners) {
    return {
      center: Array.isArray(box.center) ? box.center.slice(0, 3) : null,
      size: Array.isArray(box.size) ? box.size.slice(0, 3) : null,
      corners: Array.isArray(box.corners) ? box.corners.slice(0, 24) : null,
      label: String(box.label ?? box.name ?? "box"),
      kind: normalizeEvalKind(box, layerId),
      status: normalizeEvalStatus(box),
      source: String(box.source || (layerId === "pred" ? "eval-pred" : "eval-gt")),
      trackId: String(box.uuid ?? box.id ?? box.track_id ?? box.object_id ?? box.instance_token ?? ""),
      pairId: String(box.pair_uuid ?? ""),
      meta: box,
    };
  }
  if (Array.isArray(box.corners) && box.corners.length === 24) {
    const center = boxCenterFromCornersArray(box.corners);
    const size = boxSizeFromCornersArray(box.corners);
    return {
      center,
      size,
      corners: box.corners.slice(0, 24),
      label: String(box.label ?? box.name ?? "box"),
      kind: normalizeEvalKind(box, layerId),
      status: normalizeEvalStatus(box),
      source: layerId === "pred" ? "eval-pred" : "eval-gt",
      trackId: String(box.uuid ?? box.id ?? box.track_id ?? box.object_id ?? box.instance_token ?? ""),
      pairId: String(box.pair_uuid ?? ""),
      meta: box,
    };
  }
  if (layerId === "gt" || layerId === "pred") {
    const pose = readEvalBoxPose(box);
    const corners = boxCornersFromPose(pose.cx, pose.cy, pose.cz, pose.l, pose.w, pose.h, pose.yaw);
    return {
      center: [pose.cx, pose.cy, pose.cz],
      size: [pose.l, pose.w, pose.h],
      corners,
      label: String(box.label ?? box.name ?? normalizeEvalKind(box, layerId)),
      kind: normalizeEvalKind(box, layerId),
      status: normalizeEvalStatus(box),
      source: layerId === "pred" ? "eval-pred" : "eval-gt",
      trackId: String(box.uuid ?? box.id ?? box.track_id ?? box.object_id ?? box.instance_token ?? ""),
      pairId: String(box.pair_uuid ?? ""),
      meta: box,
    };
  }
  return null;
}
function buildInspectCandidate(meta, extra){
  if (!meta || !meta.center || !meta.corners) return null;
  const c = meta.center;
  const size = Array.isArray(meta.size) ? meta.size : [0, 0, 0];
  return {
    center: [Number(c[0] || 0), Number(c[1] || 0), Number(c[2] || 0)],
    size: [Number(size[0] || 0), Number(size[1] || 0), Number(size[2] || 0)],
    corners: meta.corners.slice(0, 24).map((v) => Number(v || 0)),
    label: String(meta.label || "box"),
    kind: String(meta.kind || "SCENE"),
    status: String(meta.status || ""),
    source: String(meta.source || "scene"),
    trackId: String(meta.trackId || ""),
    pairId: String(meta.pairId || ""),
    frameIndex: frame,
    sampleToken: extra && extra.sampleToken ? String(extra.sampleToken) : "",
    meta: meta.meta || null,
  };
}
function disposeObjectDeep(obj, skipGeometry){
  // Sprites reuse the three.js-shared plane geometry and a material cached in
  // labelSpriteCache — the scene graph owns neither, so never dispose them.
  if (obj.isSprite) return;
  if (obj.geometry && !skipGeometry) obj.geometry.dispose?.();
  if (obj.material) {
    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    mats.forEach((m) => m && m.dispose && m.dispose());
  }
  // ArrowHelper children share module-level line/cone geometries (their
  // materials are per-arrow and safe to dispose).
  const childSkipGeometry = skipGeometry || !!(obj.userData && obj.userData.velocityVector);
  if (obj.children) {
    for (const c of obj.children) {
      if (c) disposeObjectDeep(c, childSkipGeometry);
    }
  }
}
function clearGroupDeep(g){
  while (g.children.length) {
    const child = g.children[0];
    if (!child) break;
    g.remove(child);
    disposeObjectDeep(child, false);
  }
}

function addEvalPulseMarker(parent, center, colorHex, opts){
  if (!parent || !center || center.length < 3) return null;
  const radius = Math.max(0.12, Number(opts && opts.radius) || 0.18);
  const opacity = Math.max(0.18, Number(opts && opts.opacity) || 0.42);
  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 18, 14),
    new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity,
      depthWrite: false,
    })
  );
  marker.position.set(Number(center[0] || 0), Number(center[1] || 0), Number(center[2] || 0));
  marker.userData.pulseMarker = true;
  marker.userData.pulseBaseScale = Number(opts && opts.baseScale) || 1;
  marker.userData.pulseScaleAmp = Number(opts && opts.scaleAmp) || 0.28;
  marker.userData.pulseOpacityBase = opacity;
  marker.userData.pulseOpacityAmp = Number(opts && opts.opacityAmp) || 0.22;
  marker.userData.pulsePhase = Number(opts && opts.phase) || 0;
  parent.add(marker);
  return marker;
}


function addEvalShockwaveRing(parent, center, colorHex, opts){
  if (!parent || !center || center.length < 3) return null;
  const radius = Math.max(0.4, Number(opts && opts.radius) || 1.4);
  const width = Math.max(0.08, Number(opts && opts.width) || 0.18);
  const opacity = Math.max(0.12, Number(opts && opts.opacity) || 0.34);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(Math.max(0.01, radius - width), radius, 48),
    new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
  );
  ring.position.set(Number(center[0] || 0), Number(center[1] || 0), Number(center[2] || 0));
  ring.userData.effectKind = "shockwave";
  ring.userData.effectBaseScale = Number(opts && opts.baseScale) || 1;
  ring.userData.effectScaleAmp = Number(opts && opts.scaleAmp) || 0.55;
  ring.userData.effectOpacityBase = opacity;
  ring.userData.effectOpacityAmp = Number(opts && opts.opacityAmp) || 0.18;
  ring.userData.effectPhase = Number(opts && opts.phase) || 0;
  parent.add(ring);
  return ring;
}

function addEvalLightPillar(parent, center, height, colorHex, opts){
  if (!parent || !center || center.length < 3) return null;
  const pillarHeight = Math.max(1.4, Number(height) || 2);
  const radius = Math.max(0.08, Number(opts && opts.radius) || 0.18);
  const opacity = Math.max(0.08, Number(opts && opts.opacity) || 0.16);
  const pillar = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius * 1.75, pillarHeight, 18, 1, true),
    new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
  );
  pillar.position.set(
    Number(center[0] || 0),
    Number(center[1] || 0),
    Number(center[2] || 0) + pillarHeight * 0.5
  );
  pillar.userData.effectKind = "pillar";
  pillar.userData.effectBaseScale = Number(opts && opts.baseScale) || 1;
  pillar.userData.effectScaleAmp = Number(opts && opts.scaleAmp) || 0.12;
  pillar.userData.effectOpacityBase = opacity;
  pillar.userData.effectOpacityAmp = Number(opts && opts.opacityAmp) || 0.08;
  pillar.userData.effectPhase = Number(opts && opts.phase) || 0;
  parent.add(pillar);
  return pillar;
}

/** Per-frame eval bucket counts for HUD and metrics (GT×TP|FN, EST×TP|FP). */
function countEvalBucketsFromBoxes(gtBoxes, predBoxes){
  const o = { gt_tp: 0, gt_fn: 0, est_tp: 0, est_fp: 0, polygon: 0 };
  for (const b of [...(gtBoxes || []), ...(predBoxes || [])]) {
    if (isEvalPolygonBox(b)) o.polygon++;
  }
  for (const b of gtBoxes || []) {
    const k = normalizeEvalKind(b, "gt");
    const st = normalizeEvalStatus(b);
    if (k === "GT" && st === "TP") o.gt_tp++;
    else if (k === "GT" && st === "FN") o.gt_fn++;
    else if (k === "GT") o.gt_fn++;
  }
  for (const b of predBoxes || []) {
    const k = normalizeEvalKind(b, "pred");
    const st = normalizeEvalStatus(b);
    if (k === "EST" && st === "TP") o.est_tp++;
    else if (k === "EST" && st === "FP") o.est_fp++;
    else if (k === "EST") o.est_fp++;
  }
  return o;
}

/**
 * When boxes lack status, infer buckets from legacy GT/pred lengths and optional matched_pairs
 * (same semantics as the old TPR chart: pairs = TP count on both sides).
 */
function metricsFromEntryLegacy(entry){
  let g = 0;
  let p = 0;
  let pairs = 0;
  if (entry && typeof entry === "object") {
    g = Array.isArray(entry.gt) ? entry.gt.length : 0;
    p = Array.isArray(entry.pred) ? entry.pred.length : 0;
    pairs = Array.isArray(entry.matched_pairs) ? entry.matched_pairs.length : 0;
    if (!pairs && Array.isArray(entry.matched)) pairs = Math.max(0, Math.floor(entry.matched.length / 2));
  }
  pairs = Math.min(pairs, g, p);
  return {
    gt_tp: pairs,
    gt_fn: Math.max(0, g - pairs),
    est_tp: pairs,
    est_fp: Math.max(0, p - pairs),
  };
}

function metricsFromEntry(entry){
  const fromBoxes = countEvalBucketsFromBoxes(
    entry && Array.isArray(entry.gt) ? entry.gt : [],
    entry && Array.isArray(entry.pred) ? entry.pred : []
  );
  const sum = fromBoxes.gt_tp + fromBoxes.gt_fn + fromBoxes.est_tp + fromBoxes.est_fp;
  if (sum > 0) return fromBoxes;
  return metricsFromEntryLegacy(entry);
}

function buildSpotlightFrames(){
  const series = computeMetricsSeries();
  if (!series || series.n <= 0) return [];
  const err = computeErrorSeries();
  const modeEl = document.getElementById("spotlightMode");
  const mode = modeEl && modeEl.value ? modeEl.value : "composite";
  const ranked = [];
  for (let i = 0; i < series.n; i++) {
    const fn = Number(series.gt_fn[i] || 0);
    const tp = Number(series.gt_tp[i] || 0);
    const gt = fn + tp;
    const fp = Number(series.est_fp[i] || 0);
    if (mode === "fn") {
      if (fn <= 0 || gt <= 0) continue;
      const rate = fn / gt;
      ranked.push({
        frameIndex: i,
        fnCount: fn,
        gtCount: gt,
        fnRate: rate,
        compositeSeverity: rate,
        score: rate * 1000 + fn * 10 + Math.min(20, gt),
      });
      continue;
    }
    const ctr = err ? Number(err.tp_center_distance_mean[i] || 0) : 0;
    const plane = err ? Number(err.tp_plane_distance_mean[i] || 0) : 0;
    const yaw = err ? Number(err.tp_yaw_error_abs_mean[i] || 0) : 0;
    const sev = err ? Number(err.frame_severity_max[i] || 0) : 0;
    const fnRate = gt > 0 ? fn / gt : 0;
    const score = sev * 1000 + fn * 95 + fp * 36 + ctr * 110 + plane * 80 + yaw * 65 + fnRate * 180;
    if (score <= 0.01) continue;
    ranked.push({
      frameIndex: i,
      fnCount: fn,
      gtCount: gt,
      fpCount: fp,
      fnRate,
      compositeSeverity: sev,
      score,
    });
  }
  ranked.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (b.fnCount !== a.fnCount) return b.fnCount - a.fnCount;
    return a.frameIndex - b.frameIndex;
  });
  const top = ranked.slice(0, 24);
  top.sort((a, b) => a.frameIndex - b.frameIndex);
  return top;
}

function syncSpotlightFrames(){
  const prevFrame = spotlightIndex >= 0 && spotlightIndex < spotlightFrames.length
    ? spotlightFrames[spotlightIndex].frameIndex
    : null;
  spotlightFrames = buildSpotlightFrames();
  if (!spotlightFrames.length) {
    spotlightIndex = -1;
    spotlightTouring = false;
    updateSpotlightHud();
    return;
  }
  const sameIdx = prevFrame != null ? spotlightFrames.findIndex((row) => row.frameIndex === prevFrame) : -1;
  if (sameIdx >= 0) spotlightIndex = sameIdx;
  else if (spotlightIndex < 0 || spotlightIndex >= spotlightFrames.length) spotlightIndex = 0;
  updateSpotlightHud();
}

function spotlightEntryForCurrentFrame(){
  return spotlightFrames.find((row) => row.frameIndex === frame) || null;
}

function currentFrameIsSpotlightHot(){
  return !!spotlightEntryForCurrentFrame();
}

function focusFirstFnBoxInFrame(){
  let best = null;
  let bestScore = -Infinity;
  for (const cand of externalSelectionCandidates || []) {
    if (!cand) continue;
    const meta = cand.meta || {};
    const sev = safeNum(meta.severity_score) || 0;
    const score = cand.status === "FN" ? 1 + sev : sev;
    if (score > bestScore) {
      best = cand;
      bestScore = score;
    }
  }
  if (best) {
    focusCameraOnCandidate(best, false, false);
    return best;
  }
  return null;
}

async function jumpToSpotlightIndex(nextIdx){
  if (!spotlightFrames.length) return;
  if (spotlightJumpInFlight) return;
  spotlightJumpInFlight = true;
  const n = spotlightFrames.length;
  try {
    spotlightIndex = ((nextIdx % n) + n) % n;
    const target = spotlightFrames[spotlightIndex];
    await showFrame(target.frameIndex);
    focusFirstFnBoxInFrame();
    updateSpotlightHud();
  } finally {
    spotlightJumpInFlight = false;
  }
}

function updateSpotlightHud(){
  const wrap = document.getElementById("spotlightHud");
  const chip = document.getElementById("spotlightChip");
  const cur = document.getElementById("spotlightCurrent");
  const rank = document.getElementById("spotlightRank");
  const rate = document.getElementById("spotlightRate");
  const count = document.getElementById("spotlightCount");
  const worst = document.getElementById("spotlightWorst");
  const reason = document.getElementById("spotlightReason");
  const note = document.getElementById("spotlightNote");
  const playBtn = document.getElementById("spotlightPlayBtn");
  if (!wrap || !chip || !cur || !rank || !rate || !count || !worst || !reason || !note || !playBtn) return;
  const hasAny = spotlightFrames.length > 0;
  wrap.hidden = !hasAny;
  if (!hasAny) return;
  const current = spotlightEntryForCurrentFrame();
  const idx = current ? spotlightFrames.findIndex((row) => row.frameIndex === current.frameIndex) : spotlightIndex;
  const row = current || (idx >= 0 ? spotlightFrames[idx] : null);
  const modeEl = document.getElementById("spotlightMode");
  const mode = modeEl && modeEl.value ? modeEl.value : "composite";
  chip.textContent = spotlightTouring ? "touring" : (current ? "live hotspot" : "ready");
  playBtn.textContent = spotlightTouring ? "Stop tour" : "Tour hotspots";
  if (!row) {
    cur.textContent = "No hotspot selected";
    rank.textContent = "—";
    rate.textContent = "—";
    count.textContent = "—";
    worst.textContent = "—";
    reason.textContent = "—";
    note.textContent = "Uses per-frame severity and eval counts to rank the hardest moments.";
    return;
  }
  const summary = spotlightSummaryForFrame(row.frameIndex);
  cur.textContent = `frame ${row.frameIndex}`;
  rank.textContent = `${Math.max(1, (idx >= 0 ? idx : 0) + 1)} / ${spotlightFrames.length}`;
  rate.textContent = mode === "fn"
    ? `${(100 * row.fnRate).toFixed(1)}% FN`
    : `${(100 * (row.compositeSeverity || 0)).toFixed(1)}% severity`;
  count.textContent = mode === "fn"
    ? `${row.fnCount} / ${row.gtCount}`
    : `FN ${row.fnCount} · FP ${row.fpCount || 0}`;
  worst.textContent = summary ? summary.worstLabel : "—";
  reason.textContent = summary ? summary.worstReason : "—";
  note.textContent = current
    ? "Current frame is a ranked hotspot. Spotlight keeps camera centering, but does not take over selection."
    : "Use Prev/Next or Tour to move through ranked hotspots in frame order.";
}

/** Update HUD for eval buckets when postMessage or per-frame eval data is present. */
function updateEvalHud(){
  if (!evalHudEl) return;
  const gt = Array.isArray(externalLayers.gt) ? externalLayers.gt.length : 0;
  const pred = Array.isArray(externalLayers.pred) ? externalLayers.pred.length : 0;
  const b = countEvalBucketsFromBoxes(externalLayers.gt, externalLayers.pred);
  const mapKeys = bboxLayersByFrame && typeof bboxLayersByFrame === "object"
    ? Object.keys(bboxLayersByFrame).length
    : 0;
  const hasFrameMap = mapKeys > 0;
  const hasAny = gt > 0 || pred > 0 || hasFrameMap || b.gt_tp + b.gt_fn + b.est_tp + b.est_fp > 0;
  if (!hasAny) {
    evalHudEl.hidden = true;
    const ids = ["cntGtTp","cntGtFn","cntEstTp","cntEstFp","cntPolygon"];
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) el.textContent = "0";
    }
    return;
  }
  const gTot = b.gt_tp + b.gt_fn;
  const pTot = b.est_tp + b.est_fp;
  const recall = gTot > 0 ? (b.gt_tp / gTot) : null;
  const precision = pTot > 0 ? (b.est_tp / pTot) : null;
  const rStr = recall != null ? recall.toFixed(3) : "—";
  const pStr = precision != null ? precision.toFixed(3) : "—";
  const cntGtTp = document.getElementById("cntGtTp");
  const cntGtFn = document.getElementById("cntGtFn");
  const cntEstTp = document.getElementById("cntEstTp");
  const cntEstFp = document.getElementById("cntEstFp");
  if (cntGtTp) cntGtTp.textContent = String(b.gt_tp);
  if (cntGtFn) cntGtFn.textContent = String(b.gt_fn);
  if (cntEstTp) cntEstTp.textContent = String(b.est_tp);
  if (cntEstFp) cntEstFp.textContent = String(b.est_fp);
  const cntPolygon = document.getElementById("cntPolygon");
  if (cntPolygon) cntPolygon.textContent = String(b.polygon || 0);
  evalHudEl.hidden = false;
  const mapCompact = hasFrameMap ? ` · map keys ${mapKeys}` : "";
  evalHudEl.innerHTML = `<div class="eval-rp">R ${rStr} · P ${pStr}${mapCompact}</div>`;
}
function syncFullscreenButton(){
  const btn = document.getElementById("toggleFullscreen");
  if (!btn) return;
  btn.textContent = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
}

function applyMainCameraAspect(aspect){
  const a = Math.max(0.05, Number(aspect) || 1);
  if (camera.isPerspectiveCamera) {
    camera.aspect = a;
  } else {
    const fr = 22;
    camera.left = -fr * a;
    camera.right = fr * a;
    camera.top = fr;
    camera.bottom = -fr;
  }
  camera.updateProjectionMatrix();
}

function resizeMainRenderer(){
  const wrap = document.getElementById("canvasWrap");
  const w = Math.max(1, (wrap && wrap.clientWidth) || canvas.clientWidth || window.innerWidth);
  const h = Math.max(1, (wrap && wrap.clientHeight) || canvas.clientHeight || (window.innerHeight - 64));
  renderer.setSize(w, h, false);
  applyMainCameraAspect(w / h);
  layoutMetricsStacks();
  resizeMetricsRenderer();
}

function layoutMetricsStacks(){
  const countOn = document.getElementById("uiShowMetricsCounts") && document.getElementById("uiShowMetricsCounts").checked;
  const ratesOn = document.getElementById("uiShowMetricsRates") && document.getElementById("uiShowMetricsRates").checked;
  const errorOn = document.getElementById("uiShowMetricsError") && document.getElementById("uiShowMetricsError").checked;
  const compareOn = metricsCompareWrapEl && !metricsCompareWrapEl.hidden;
  const gap = 8;
  const chartH = 168;
  const compareH = 172;
  let nextBottom = 12;
  if (metricsWrapEl && !metricsWrapEl.dataset.userPos) {
    metricsWrapEl.style.bottom = countOn ? `${nextBottom}px` : "";
  }
  if (countOn) nextBottom += chartH + gap;
  if (metricsCompareWrapEl && !metricsCompareWrapEl.dataset.userPos) {
    metricsCompareWrapEl.style.bottom = compareOn ? `${nextBottom}px` : "";
  }
  if (compareOn) nextBottom += compareH + gap;
  if (metricsRatesWrapEl && !metricsRatesWrapEl.dataset.userPos) {
    metricsRatesWrapEl.style.bottom = ratesOn ? `${nextBottom}px` : "";
  }
  if (ratesOn) nextBottom += chartH + gap;
  if (metricsErrorWrapEl && !metricsErrorWrapEl.dataset.userPos) {
    metricsErrorWrapEl.style.bottom = errorOn ? `${nextBottom}px` : "";
  }
}

function clampPanelToParent(panelEl, parentEl){
  const parent = parentEl || document.getElementById("canvasWrap");
  if (!panelEl || !parent || panelEl.hidden || !panelEl.dataset.userPos) return;
  const pr = parent.getBoundingClientRect();
  const ew = panelEl.offsetWidth;
  const eh = panelEl.offsetHeight;
  let l = parseFloat(panelEl.style.left);
  let t = parseFloat(panelEl.style.top);
  if (!Number.isFinite(l) || !Number.isFinite(t)) return;
  l = Math.max(0, Math.min(l, Math.max(0, pr.width - ew)));
  t = Math.max(0, Math.min(t, Math.max(0, pr.height - eh)));
  panelEl.style.left = `${l}px`;
  panelEl.style.top = `${t}px`;
}

function clampAllMovablePanels(){
  const parent = document.getElementById("canvasWrap");
  ["hud", "inspectPanel", "metricsWrap", "metricsCompareWrap", "metricsRatesWrap", "metricsErrorWrap", "cameraViewportWrap"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) clampPanelToParent(el, parent);
  });
}

function resetPanelPosition(panelEl){
  if (!panelEl) return;
  delete panelEl.dataset.userPos;
  panelEl.style.left = "";
  panelEl.style.top = "";
  panelEl.style.right = "";
  panelEl.style.bottom = "";
  layoutMetricsStacks();
}

/** Clear drag offsets on all overlay panels; one layoutMetricsStacks at end. */
function resetAllPanelPositions(){
  ["hud", "inspectPanel", "metricsWrap", "metricsCompareWrap", "metricsRatesWrap", "metricsErrorWrap", "cameraViewportWrap"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    delete el.dataset.userPos;
    el.style.left = "";
    el.style.top = "";
    el.style.right = "";
    el.style.bottom = "";
  });
  layoutMetricsStacks();
}

/**
 * Drag panel by handle (pointer); double-click handle resets to default CSS layout.
 * Uses #canvasWrap as bounding box.
 */
function makePanelDraggable(panelEl, handleEl, parentEl){
  const parent = parentEl || document.getElementById("canvasWrap");
  if (!panelEl || !handleEl || !parent) return;
  let dragging = false;
  let startX = 0;
  let startY = 0;
  let origLeft = 0;
  let origTop = 0;

  function panelToLeftTop(){
    const er = panelEl.getBoundingClientRect();
    const pr = parent.getBoundingClientRect();
    panelEl.style.left = `${er.left - pr.left}px`;
    panelEl.style.top = `${er.top - pr.top}px`;
    panelEl.style.right = "auto";
    panelEl.style.bottom = "auto";
    panelEl.dataset.userPos = "1";
  }

  handleEl.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    panelToLeftTop();
    dragging = true;
    startX = ev.clientX;
    startY = ev.clientY;
    origLeft = parseFloat(panelEl.style.left) || 0;
    origTop = parseFloat(panelEl.style.top) || 0;
    try {
      handleEl.setPointerCapture(ev.pointerId);
    } catch (_) {}
    handleEl.style.cursor = "grabbing";
  });

  handleEl.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - startX;
    const dy = ev.clientY - startY;
    panelEl.style.left = `${origLeft + dx}px`;
    panelEl.style.top = `${origTop + dy}px`;
    clampPanelToParent(panelEl, parent);
  });

  function endDrag(ev){
    if (!dragging) return;
    dragging = false;
    try {
      if (ev && ev.pointerId != null) handleEl.releasePointerCapture(ev.pointerId);
    } catch (_) {}
    handleEl.style.cursor = "grab";
  }

  handleEl.addEventListener("pointerup", endDrag);
  handleEl.addEventListener("pointercancel", endDrag);

  handleEl.addEventListener("dblclick", (ev) => {
    ev.preventDefault();
    resetPanelPosition(panelEl);
  });
}

function resizeMetricsRenderer(){
  if (metricsWrapEl && metricsCanvas && !metricsWrapEl.hidden) {
    const w = Math.max(120, metricsWrapEl.clientWidth || 280);
    const h = Math.max(80, metricsWrapEl.clientHeight || 120);
    chartSizes.set(metricsCanvas, { w, h });
  }
  if (metricsRatesWrapEl && metricsRatesCanvas && !metricsRatesWrapEl.hidden) {
    const w = Math.max(120, metricsRatesWrapEl.clientWidth || 280);
    const h = Math.max(80, metricsRatesWrapEl.clientHeight || 120);
    chartSizes.set(metricsRatesCanvas, { w, h });
  }
  if (metricsErrorWrapEl && metricsErrorCanvas && !metricsErrorWrapEl.hidden) {
    const w = Math.max(120, metricsErrorWrapEl.clientWidth || 280);
    const h = Math.max(80, metricsErrorWrapEl.clientHeight || 120);
    chartSizes.set(metricsErrorCanvas, { w, h });
  }
}

/** Four count series per frame from override or per-frame bbox map (GT·TP / GT·FN / EST·TP / EST·FP). */
function computeMetricsSeries(){
  const mo = metricsSeriesOverride;
  if (mo && typeof mo === "object") {
    const hasFour =
      Array.isArray(mo.gt_tp) || Array.isArray(mo.gt_fn) ||
      Array.isArray(mo.est_tp) || Array.isArray(mo.est_fp);
    if (hasFour) {
      const gt_tp = [];
      const gt_fn = [];
      const est_tp = [];
      const est_fp = [];
      for (let i = 0; i < totalFrames; i++) {
        gt_tp.push(Number((mo.gt_tp && mo.gt_tp[i]) || 0));
        gt_fn.push(Number((mo.gt_fn && mo.gt_fn[i]) || 0));
        est_tp.push(Number((mo.est_tp && mo.est_tp[i]) || 0));
        est_fp.push(Number((mo.est_fp && mo.est_fp[i]) || 0));
      }
      return { gt_tp, gt_fn, est_tp, est_fp, n: totalFrames };
    }
    if (Array.isArray(mo.gt) || Array.isArray(mo.pred) || Array.isArray(mo.tpr)) {
      const gIn = Array.isArray(mo.gt) ? mo.gt : [];
      const pIn = Array.isArray(mo.pred) ? mo.pred : [];
      const tIn = mo.tpr;
      const gt_tp = [];
      const gt_fn = [];
      const est_tp = [];
      const est_fp = [];
      for (let i = 0; i < totalFrames; i++) {
        const g = Number(gIn[i] || 0);
        const p = Number(pIn[i] || 0);
        const rec = Array.isArray(tIn) && i < tIn.length
          ? Math.min(1, Math.max(0, Number(tIn[i] || 0)))
          : (g > 0 ? Math.min(1, Math.min(g, p) / g) : 0);
        const gtp = Math.round(g * rec);
        const gfn = Math.max(0, g - gtp);
        const etp = Math.min(p, gtp);
        const efp = Math.max(0, p - etp);
        gt_tp.push(gtp);
        gt_fn.push(gfn);
        est_tp.push(etp);
        est_fp.push(efp);
      }
      return { gt_tp, gt_fn, est_tp, est_fp, n: totalFrames };
    }
  }
  if (!bboxLayersByFrame || totalFrames <= 0) return null;
  const gt_tp = [];
  const gt_fn = [];
  const est_tp = [];
  const est_fp = [];
  for (let fi = 0; fi < totalFrames; fi++) {
    const sk = String(fi);
    const entry = Object.prototype.hasOwnProperty.call(bboxLayersByFrame, sk)
      ? bboxLayersByFrame[sk]
      : bboxLayersByFrame[fi];
    const m = metricsFromEntry(entry && typeof entry === "object" ? entry : null);
    gt_tp.push(m.gt_tp);
    gt_fn.push(m.gt_fn);
    est_tp.push(m.est_tp);
    est_fp.push(m.est_fp);
  }
  return { gt_tp, gt_fn, est_tp, est_fp, n: totalFrames };
}

function emptyMetricsBuckets(){
  return { gt_tp: 0, gt_fn: 0, est_tp: 0, est_fp: 0 };
}

function countEvalBucketsForRun(gtBoxes, predBoxes, runName){
  const target = String(runName || "").trim();
  if (!target) return emptyMetricsBuckets();
  const gt = Array.isArray(gtBoxes) ? gtBoxes.filter((b) => evalRunNameFromMeta(b) === target) : [];
  const pred = Array.isArray(predBoxes) ? predBoxes.filter((b) => evalRunNameFromMeta(b) === target) : [];
  return countEvalBucketsFromBoxes(gt, pred);
}

function addMetricBuckets(a, b){
  a.gt_tp += Number(b && b.gt_tp || 0);
  a.gt_fn += Number(b && b.gt_fn || 0);
  a.est_tp += Number(b && b.est_tp || 0);
  a.est_fp += Number(b && b.est_fp || 0);
  return a;
}

function metricRatesFromBuckets(b){
  const gtTot = Number(b.gt_tp || 0) + Number(b.gt_fn || 0);
  const predTot = Number(b.est_tp || 0) + Number(b.est_fp || 0);
  return {
    recall: gtTot > 0 ? Number(b.gt_tp || 0) / gtTot : null,
    precision: predTot > 0 ? Number(b.est_tp || 0) / predTot : null,
  };
}

function frameEvalEntryForMetrics(i){
  if (bboxLayersByFrame && typeof bboxLayersByFrame === "object") {
    const sk = String(i);
    const entry = Object.prototype.hasOwnProperty.call(bboxLayersByFrame, sk) ? bboxLayersByFrame[sk] : bboxLayersByFrame[i];
    if (entry && typeof entry === "object") return normalizeEvalEntry(entry, i);
  }
  if (Number(i) === Number(frame)) return normalizeEvalEntry(externalLayers || {}, i);
  return null;
}

function computeCompareMetricsSummary(){
  const mode = compareModeValue();
  if (mode !== "side_by_side" && mode !== "curtain") return null;
  const runs = currentEvalRuns();
  if (!Array.isArray(runs) || runs.length < 2) return null;
  const runA = runs[0];
  const runB = runs[1];
  const n = Math.max(0, Number(totalFrames) || 0);
  const currentIndex = Math.min(Math.max(0, Number(frame) || 0), Math.max(0, n - 1));
  const totals = {
    a: emptyMetricsBuckets(),
    b: emptyMetricsBuckets(),
  };
  let current = null;
  let seen = 0;
  const maxN = n > 0 ? n : 1;
  for (let i = 0; i < maxN; i++) {
    const entry = frameEvalEntryForMetrics(i);
    if (!entry) continue;
    seen++;
    const a = countEvalBucketsForRun(entry.gt, entry.pred, runA);
    const b = countEvalBucketsForRun(entry.gt, entry.pred, runB);
    addMetricBuckets(totals.a, a);
    addMetricBuckets(totals.b, b);
    if (i === currentIndex) current = { a, b };
  }
  if (!seen) return null;
  if (!current) {
    const entry = frameEvalEntryForMetrics(currentIndex);
    current = entry
      ? { a: countEvalBucketsForRun(entry.gt, entry.pred, runA), b: countEvalBucketsForRun(entry.gt, entry.pred, runB) }
      : { a: emptyMetricsBuckets(), b: emptyMetricsBuckets() };
  }
  return { runA, runB, frameIndex: currentIndex, current, totals };
}

function computeCompareMetricsSeries(){
  const mode = compareModeValue();
  if (mode !== "side_by_side" && mode !== "curtain") return null;
  const runs = currentEvalRuns();
  if (!Array.isArray(runs) || runs.length < 2) return null;
  const runA = runs[0];
  const runB = runs[1];
  const n = Math.max(0, Number(totalFrames) || 0);
  if (n <= 0) return null;
  const a_tp = [];
  const a_fp = [];
  const b_tp = [];
  const b_fp = [];
  const a_fn = [];
  const b_fn = [];
  let seen = 0;
  for (let i = 0; i < n; i++) {
    const entry = frameEvalEntryForMetrics(i);
    if (entry) seen++;
    const a = entry ? countEvalBucketsForRun(entry.gt, entry.pred, runA) : emptyMetricsBuckets();
    const b = entry ? countEvalBucketsForRun(entry.gt, entry.pred, runB) : emptyMetricsBuckets();
    a_tp.push(a.est_tp);
    a_fp.push(a.est_fp);
    b_tp.push(b.est_tp);
    b_fp.push(b.est_fp);
    a_fn.push(a.gt_fn);
    b_fn.push(b.gt_fn);
  }
  if (!seen) return null;
  return { runA, runB, a_tp, a_fp, b_tp, b_fp, a_fn, b_fn, n };
}

/** Per-frame rates in [0,1]: TPR (recall), precision, F1, FDR = FP/(TP+FP) on detections. */
function computeRateSeries(){
  const s = computeMetricsSeries();
  if (!s || s.n <= 0) return null;
  const n = s.n;
  const tpr = [];
  const prec = [];
  const f1 = [];
  const fdr = [];
  for (let i = 0; i < n; i++) {
    const a = s.gt_tp[i];
    const b = s.gt_fn[i];
    const c = s.est_tp[i];
    const d = s.est_fp[i];
    const gTot = a + b;
    const pTot = c + d;
    const R = gTot > 0 ? a / gTot : 0;
    const P = pTot > 0 ? c / pTot : 0;
    const F1 = R + P > 1e-12 ? (2 * R * P) / (R + P) : 0;
    const FDR = pTot > 0 ? d / pTot : 0;
    tpr.push(R);
    prec.push(P);
    f1.push(F1);
    fdr.push(FDR);
  }
  return { tpr, prec, f1, fdr, n };
}

function computeErrorSeries(){
  const mo = metricsSeriesOverride;
  const hasOverride =
    mo && (
      Array.isArray(mo.tp_center_distance_mean) ||
      Array.isArray(mo.tp_plane_distance_mean) ||
      Array.isArray(mo.tp_yaw_error_abs_mean) ||
      Array.isArray(mo.frame_severity_max)
    );
  if (hasOverride) {
    const ctr = [];
    const plane = [];
    const yaw = [];
    const sev = [];
    for (let i = 0; i < totalFrames; i++) {
      ctr.push(Math.max(0, safeNum(mo.tp_center_distance_mean && mo.tp_center_distance_mean[i]) || 0));
      plane.push(Math.max(0, safeNum(mo.tp_plane_distance_mean && mo.tp_plane_distance_mean[i]) || 0));
      yaw.push(Math.max(0, safeNum(mo.tp_yaw_error_abs_mean && mo.tp_yaw_error_abs_mean[i]) || 0));
      sev.push(clip01(safeNum(mo.frame_severity_max && mo.frame_severity_max[i]) || 0));
    }
    return { tp_center_distance_mean: ctr, tp_plane_distance_mean: plane, tp_yaw_error_abs_mean: yaw, frame_severity_max: sev, n: totalFrames };
  }
  if (!bboxLayersByFrame || totalFrames <= 0) return null;
  const ctr = [];
  const plane = [];
  const yaw = [];
  const sev = [];
  for (let fi = 0; fi < totalFrames; fi++) {
    const sk = String(fi);
    const entry = Object.prototype.hasOwnProperty.call(bboxLayersByFrame, sk) ? bboxLayersByFrame[sk] : bboxLayersByFrame[fi];
    const norm = normalizeEvalEntry(entry, fi);
    const tp = [];
    for (const box of [...(norm.gt || []), ...(norm.pred || [])]) {
      if (!box || box.status !== "TP") continue;
      tp.push(box);
    }
    const ctrVals = tp.map((b) => safeNum(b.center_distance)).filter((v) => v != null);
    const planeVals = tp.map((b) => safeNum(b.plane_distance)).filter((v) => v != null);
    const yawVals = tp.map((b) => safeNum(b.yaw_error) != null ? Math.abs(Number(b.yaw_error)) : null).filter((v) => v != null);
    ctr.push(ctrVals.length ? ctrVals.reduce((a, b) => a + b, 0) / ctrVals.length : 0);
    plane.push(planeVals.length ? planeVals.reduce((a, b) => a + b, 0) / planeVals.length : 0);
    yaw.push(yawVals.length ? yawVals.reduce((a, b) => a + b, 0) / yawVals.length : 0);
    const sevVals = [...(norm.gt || []), ...(norm.pred || [])].map((b) => safeNum(b.severity_score)).filter((v) => v != null);
    sev.push(sevVals.length ? Math.max(...sevVals) : 0);
  }
  return { tp_center_distance_mean: ctr, tp_plane_distance_mean: plane, tp_yaw_error_abs_mean: yaw, frame_severity_max: sev, n: totalFrames };
}

function spotlightSummaryForFrame(frameIndex){
  if (!bboxLayersByFrame || totalFrames <= 0) return null;
  const sk = String(frameIndex);
  const entry = Object.prototype.hasOwnProperty.call(bboxLayersByFrame, sk) ? bboxLayersByFrame[sk] : bboxLayersByFrame[frameIndex];
  const norm = normalizeEvalEntry(entry, frameIndex);
  const all = [...(norm.gt || []), ...(norm.pred || [])];
  if (!all.length) return null;
  let worst = all[0];
  for (const box of all) {
    const a = safeNum(box && box.severity_score) || 0;
    const b = safeNum(worst && worst.severity_score) || 0;
    if (a > b) worst = box;
  }
  const counts = countEvalBucketsFromBoxes(norm.gt, norm.pred);
  return {
    worstLabel: `${String(worst.label || worst.kind || "box")} · ${String(worst.status || "TP")}`,
    worstReason: String(worst.severity_reason || "severity"),
    worstSeverity: safeNum(worst.severity_score) || 0,
    counts,
  };
}

function rebuildMetricsPlot(){
  while (metricsChartGroup.children.length) metricsChartGroup.remove(metricsChartGroup.children[0]);
  metricsPlayhead = null;
  const want = document.getElementById("uiShowMetricsCounts") && document.getElementById("uiShowMetricsCounts").checked;
  if (!metricsWrapEl || !want) {
    if (metricsWrapEl) metricsWrapEl.hidden = true;
    const cap0 = document.getElementById("metricsCaption");
    if (cap0) cap0.textContent = "";
    layoutMetricsStacks();
    return;
  }
  const compareSeries = computeCompareMetricsSeries();
  const series = compareSeries || computeMetricsSeries();
  if (!series || series.n <= 0) {
    metricsWrapEl.hidden = true;
    const cap1 = document.getElementById("metricsCaption");
    if (cap1) cap1.textContent = "";
    layoutMetricsStacks();
    return;
  }
  const chartSeries = compareSeries
    ? [
      { vals: compareSeries.a_tp, color: TH.hex("cmpATp"), label: "A TP" },
      { vals: compareSeries.a_fp, color: TH.hex("cmpAFp"), label: "A FP" },
      { vals: compareSeries.b_tp, color: TH.hex("cmpBTp"), label: "B TP" },
      { vals: compareSeries.b_fp, color: TH.hex("cmpBFp"), label: "B FP" },
    ]
    : [
      { vals: series.gt_tp, color: TH.hex("gtTp"), label: "GT/TP" },
      { vals: series.gt_fn, color: TH.hex("gtFn"), label: "GT/FN" },
      { vals: series.est_tp, color: TH.hex("estTp"), label: "EST/TP" },
      { vals: series.est_fp, color: TH.hex("estFp"), label: "EST/FP" },
    ];
  const n = series.n;
  let yMax = 1e-6;
  for (let i = 0; i < n; i++) {
    for (const row of chartSeries) yMax = Math.max(yMax, Number(row.vals[i] || 0), 1);
  }
  const x0 = -0.9;
  const x1 = 0.9;
  const y0 = -0.78;
  const y1 = 0.78;
  const span = x1 - x0;

  const bg = new THREE.Mesh(
    new THREE.PlaneGeometry(2.05, 1.85),
    new THREE.MeshBasicMaterial({ color: TH.hex("chartPanel"), transparent: true, opacity: 0.92 })
  );
  bg.position.z = -0.02;
  metricsChartGroup.add(bg);

  const starPos = [];
  const starCol = [];
  for (let s = 0; s < 64; s++) {
    starPos.push((Math.random() - 0.5) * 2, (Math.random() - 0.5) * 1.6, -0.01);
    const tw = 0.35 + Math.random() * 0.65;
    starCol.push(0.45 * tw, 0.55 * tw, 0.95 * tw);
  }
  const starG = new THREE.BufferGeometry();
  starG.setAttribute("position", new THREE.Float32BufferAttribute(starPos, 3));
  starG.setAttribute("color", new THREE.Float32BufferAttribute(starCol, 3));
  metricsChartGroup.add(new THREE.Points(starG, new THREE.PointsMaterial({ size: 0.012, vertexColors: true, transparent: true, opacity: 0.5 })));

  for (let g = 0; g <= 4; g++) {
    const t = g / 4;
    const y = y0 + (y1 - y0) * t;
    const gg = new THREE.BufferGeometry();
    gg.setAttribute("position", new THREE.Float32BufferAttribute([x0, y, 0, x1, y, 0], 3));
    metricsChartGroup.add(new THREE.Line(gg, new THREE.LineBasicMaterial({ color: TH.hex("chartGrid"), transparent: true, opacity: 0.55 })));
  }

  function lineStrip(vals, color){
    const pts = [];
    const denom = yMax;
    for (let i = 0; i < n; i++) {
      const xf = n <= 1 ? 0.5 : i / (n - 1);
      const x = x0 + span * xf;
      const v = vals[i] / denom;
      const y = y0 + (y1 - y0) * Math.max(0, Math.min(1, v));
      pts.push(x, y, 0.02);
    }
    if (pts.length === 3) {
      pts.push(pts[0], pts[1], pts[2]);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.Line(geom, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
  }
  for (const row of chartSeries) metricsChartGroup.add(lineStrip(row.vals, row.color));

  const title = metricsWrapEl.querySelector(".metrics-title");
  if (title) title.textContent = compareSeries ? "Compare Counts · TP / FP by run" : "Counts · GT/EST buckets";
  const legend = metricsWrapEl.querySelector(".metrics-legend");
  if (legend) {
    legend.classList.toggle("compare-legend", !!compareSeries);
    legend.innerHTML = compareSeries
      ? `<span class="c1">●</span>A TP <span class="c2">●</span>A FP <span class="c3">●</span>B TP <span class="c4">●</span>B FP`
      : `<span class="c1">●</span>GT/TP <span class="c2">●</span>GT/FN <span class="c3">●</span>EST/TP <span class="c4">●</span>EST/FP`;
  }

  const phGeom = new THREE.BufferGeometry();
  const phPos = new Float32Array([-0.9, -0.78, 0.04, -0.9, 0.78, 0.04]);
  phGeom.setAttribute("position", new THREE.BufferAttribute(phPos, 3));
  metricsPlayhead = new THREE.Group();
  metricsPlayhead.userData.geom = phGeom;
  metricsPlayhead.add(new THREE.Line(phGeom, new THREE.LineBasicMaterial({ color: TH.hex("playhead"), transparent: true, opacity: 1 })));
  const phGeom2 = new THREE.BufferGeometry();
  const phPos2 = new Float32Array(phPos);
  phGeom2.setAttribute("position", new THREE.BufferAttribute(phPos2, 3));
  metricsPlayhead.userData.geomGlow = phGeom2;
  metricsPlayhead.add(new THREE.Line(phGeom2, new THREE.LineBasicMaterial({ color: TH.hex("playheadGlow"), transparent: true, opacity: 0.4 })));
  metricsChartGroup.add(metricsPlayhead);

  metricsWrapEl.hidden = false;
  layoutMetricsStacks();
  resizeMetricsRenderer();
  updateMetricsPlayhead();
  updateMetricsFrameCaption();
}

function rebuildMetricsRatesPlot(){
  while (metricsRatesChartGroup.children.length) metricsRatesChartGroup.remove(metricsRatesChartGroup.children[0]);
  metricsRatesPlayhead = null;
  const want = document.getElementById("uiShowMetricsRates") && document.getElementById("uiShowMetricsRates").checked;
  if (!metricsRatesWrapEl || !want) {
    if (metricsRatesWrapEl) metricsRatesWrapEl.hidden = true;
    const c0 = document.getElementById("metricsRatesCaption");
    if (c0) c0.textContent = "";
    layoutMetricsStacks();
    return;
  }
  const rs = computeRateSeries();
  if (!rs || rs.n <= 0) {
    metricsRatesWrapEl.hidden = true;
    const c1 = document.getElementById("metricsRatesCaption");
    if (c1) c1.textContent = "";
    layoutMetricsStacks();
    return;
  }
  const { tpr, prec, f1, fdr, n } = rs;
  const x0 = -0.9;
  const x1 = 0.9;
  const y0 = -0.78;
  const y1 = 0.78;
  const span = x1 - x0;

  const bg = new THREE.Mesh(
    new THREE.PlaneGeometry(2.05, 1.85),
    new THREE.MeshBasicMaterial({ color: TH.hex("chartPanel"), transparent: true, opacity: 0.92 })
  );
  bg.position.z = -0.02;
  metricsRatesChartGroup.add(bg);

  for (let g = 0; g <= 4; g++) {
    const t = g / 4;
    const yy = y0 + (y1 - y0) * t;
    const gg = new THREE.BufferGeometry();
    gg.setAttribute("position", new THREE.Float32BufferAttribute([x0, yy, 0, x1, yy, 0], 3));
    metricsRatesChartGroup.add(new THREE.Line(gg, new THREE.LineBasicMaterial({ color: TH.hex("chartGrid"), transparent: true, opacity: 0.55 })));
  }

  function lineStrip01(vals, color){
    const pts = [];
    for (let i = 0; i < n; i++) {
      const xf = n <= 1 ? 0.5 : i / (n - 1);
      const x = x0 + span * xf;
      const vn = Math.max(0, Math.min(1, vals[i]));
      const y = y0 + (y1 - y0) * vn;
      pts.push(x, y, 0.02);
    }
    if (pts.length === 3) pts.push(pts[0], pts[1], pts[2]);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.Line(geom, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
  }
  metricsRatesChartGroup.add(lineStrip01(tpr, TH.hex("rate1")));
  metricsRatesChartGroup.add(lineStrip01(prec, TH.hex("estTp")));
  metricsRatesChartGroup.add(lineStrip01(f1, TH.hex("rate3")));
  metricsRatesChartGroup.add(lineStrip01(fdr, TH.hex("rate4")));

  const phGeom = new THREE.BufferGeometry();
  const phPos = new Float32Array([-0.9, -0.78, 0.04, -0.9, 0.78, 0.04]);
  phGeom.setAttribute("position", new THREE.BufferAttribute(phPos, 3));
  metricsRatesPlayhead = new THREE.Group();
  metricsRatesPlayhead.userData.geom = phGeom;
  metricsRatesPlayhead.add(new THREE.Line(phGeom, new THREE.LineBasicMaterial({ color: TH.hex("playhead"), transparent: true, opacity: 1 })));
  const phGeom2 = new THREE.BufferGeometry();
  const phPos2 = new Float32Array(phPos);
  phGeom2.setAttribute("position", new THREE.BufferAttribute(phPos2, 3));
  metricsRatesPlayhead.userData.geomGlow = phGeom2;
  metricsRatesPlayhead.add(new THREE.Line(phGeom2, new THREE.LineBasicMaterial({ color: TH.hex("playheadGlowRates"), transparent: true, opacity: 0.45 })));
  metricsRatesChartGroup.add(metricsRatesPlayhead);

  metricsRatesWrapEl.hidden = false;
  layoutMetricsStacks();
  resizeMetricsRenderer();
  updateMetricsRatesPlayhead();
  updateMetricsRatesCaption();
}

function rebuildMetricsErrorPlot(){
  while (metricsErrorChartGroup.children.length) metricsErrorChartGroup.remove(metricsErrorChartGroup.children[0]);
  metricsErrorPlayhead = null;
  const want = document.getElementById("uiShowMetricsError") && document.getElementById("uiShowMetricsError").checked;
  if (!metricsErrorWrapEl || !want) {
    if (metricsErrorWrapEl) metricsErrorWrapEl.hidden = true;
    const c0 = document.getElementById("metricsErrorCaption");
    if (c0) c0.textContent = "";
    layoutMetricsStacks();
    return;
  }
  const es = computeErrorSeries();
  if (!es || es.n <= 0) {
    metricsErrorWrapEl.hidden = true;
    const c1 = document.getElementById("metricsErrorCaption");
    if (c1) c1.textContent = "";
    layoutMetricsStacks();
    return;
  }
  const { tp_center_distance_mean: ctr, tp_plane_distance_mean: plane, tp_yaw_error_abs_mean: yaw, frame_severity_max: sev, n } = es;
  const x0 = -0.9;
  const x1 = 0.9;
  const y0 = -0.78;
  const y1 = 0.78;
  const span = x1 - x0;
  const yMax = Math.max(1e-6, ...ctr, ...plane, ...yaw, 1);
  const bg = new THREE.Mesh(
    new THREE.PlaneGeometry(2.05, 1.85),
    new THREE.MeshBasicMaterial({ color: TH.hex("chartPanelAlt"), transparent: true, opacity: 0.92 })
  );
  bg.position.z = -0.02;
  metricsErrorChartGroup.add(bg);
  for (let g = 0; g <= 4; g++) {
    const t = g / 4;
    const yy = y0 + (y1 - y0) * t;
    const gg = new THREE.BufferGeometry();
    gg.setAttribute("position", new THREE.Float32BufferAttribute([x0, yy, 0, x1, yy, 0], 3));
    metricsErrorChartGroup.add(new THREE.Line(gg, new THREE.LineBasicMaterial({ color: TH.hex("chartGridAlt"), transparent: true, opacity: 0.55 })));
  }
  function lineStripScaled(vals, color, denom, z){
    const pts = [];
    for (let i = 0; i < n; i++) {
      const xf = n <= 1 ? 0.5 : i / (n - 1);
      const x = x0 + span * xf;
      const vn = Math.max(0, Math.min(1, (vals[i] || 0) / Math.max(denom, 1e-6)));
      const yy = y0 + (y1 - y0) * vn;
      pts.push(x, yy, z || 0.02);
    }
    if (pts.length === 3) pts.push(pts[0], pts[1], pts[2]);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    return new THREE.Line(geom, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
  }
  metricsErrorChartGroup.add(lineStripScaled(ctr, TH.hex("err1"), yMax, 0.02));
  metricsErrorChartGroup.add(lineStripScaled(plane, TH.hex("err2"), yMax, 0.02));
  metricsErrorChartGroup.add(lineStripScaled(yaw, TH.hex("err3"), yMax, 0.02));
  metricsErrorChartGroup.add(lineStripScaled(sev, TH.hex("err4"), 1, 0.02));
  const phGeom = new THREE.BufferGeometry();
  const phPos = new Float32Array([-0.9, -0.78, 0.04, -0.9, 0.78, 0.04]);
  phGeom.setAttribute("position", new THREE.BufferAttribute(phPos, 3));
  metricsErrorPlayhead = new THREE.Group();
  metricsErrorPlayhead.userData.geom = phGeom;
  metricsErrorPlayhead.add(new THREE.Line(phGeom, new THREE.LineBasicMaterial({ color: TH.hex("playhead"), transparent: true, opacity: 1 })));
  const phGeom2 = new THREE.BufferGeometry();
  const phPos2 = new Float32Array(phPos);
  phGeom2.setAttribute("position", new THREE.BufferAttribute(phPos2, 3));
  metricsErrorPlayhead.userData.geomGlow = phGeom2;
  metricsErrorPlayhead.add(new THREE.Line(phGeom2, new THREE.LineBasicMaterial({ color: TH.hex("playheadGlowError"), transparent: true, opacity: 0.45 })));
  metricsErrorChartGroup.add(metricsErrorPlayhead);
  metricsErrorWrapEl.hidden = false;
  layoutMetricsStacks();
  resizeMetricsRenderer();
  updateMetricsErrorPlayhead();
  updateMetricsErrorCaption();
}

function rebuildAllMetricsPlots(){
  rebuildMetricsPlot();
  updateMetricsComparePanel();
  rebuildMetricsRatesPlot();
  rebuildMetricsErrorPlot();
  syncSpotlightFrames();
}

function shortRunName(name){
  const s = String(name || "").trim();
  if (!s) return "—";
  return s.length > 24 ? `${s.slice(0, 10)}…${s.slice(-10)}` : s;
}

function fmtRate(v){
  return v == null ? "—" : Number(v).toFixed(3);
}

function setTextById(id, text){
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function updateMetricsComparePanel(){
  if (!metricsCompareWrapEl) return;
  const summary = computeCompareMetricsSummary();
  if (!summary) {
    metricsCompareWrapEl.hidden = true;
    setTextById("metricsCompareCaption", "");
    layoutMetricsStacks();
    return;
  }
  const ca = summary.current.a;
  const cb = summary.current.b;
  const ta = summary.totals.a;
  const tb = summary.totals.b;
  const ra = metricRatesFromBuckets(ca);
  const rb = metricRatesFromBuckets(cb);
  const rta = metricRatesFromBuckets(ta);
  const rtb = metricRatesFromBuckets(tb);
  metricsCompareWrapEl.hidden = false;
  setTextById("metricsCompareCaption", `f${summary.frameIndex}: current frame · totals across loaded frames below`);
  setTextById("metricsCompareNameA", `A · ${shortRunName(summary.runA)}`);
  setTextById("metricsCompareNameB", `B · ${shortRunName(summary.runB)}`);
  setTextById("metricsCompareTpA", String(ca.est_tp));
  setTextById("metricsCompareFpA", String(ca.est_fp));
  setTextById("metricsCompareTpB", String(cb.est_tp));
  setTextById("metricsCompareFpB", String(cb.est_fp));
  setTextById("metricsCompareMiniA", `FN ${ca.gt_fn} · R ${fmtRate(ra.recall)} · P ${fmtRate(ra.precision)} · total TP ${ta.est_tp} FP ${ta.est_fp}`);
  setTextById("metricsCompareMiniB", `FN ${cb.gt_fn} · R ${fmtRate(rb.recall)} · P ${fmtRate(rb.precision)} · total TP ${tb.est_tp} FP ${tb.est_fp}`);
  const dTp = cb.est_tp - ca.est_tp;
  const dFp = cb.est_fp - ca.est_fp;
  const dTpTotal = tb.est_tp - ta.est_tp;
  const dFpTotal = tb.est_fp - ta.est_fp;
  const delta = document.getElementById("metricsCompareDelta");
  if (delta) {
    delta.innerHTML =
      `<span>ΔTP <b>${fmtSigned(dTp, 0)}</b> · total <b>${fmtSigned(dTpTotal, 0)}</b></span>` +
      `<span>ΔFP <b>${fmtSigned(dFp, 0)}</b> · total <b>${fmtSigned(dFpTotal, 0)}</b></span>`;
    delta.title = `Totals: A R ${fmtRate(rta.recall)} P ${fmtRate(rta.precision)} · B R ${fmtRate(rtb.recall)} P ${fmtRate(rtb.precision)}`;
  }
  layoutMetricsStacks();
}

function updateMetricsFrameCaption(){
  const cap = document.getElementById("metricsCaption");
  if (!cap) return;
  const cs = computeCompareMetricsSeries();
  if (cs && cs.n > 0) {
    const i = Math.min(Math.max(0, frame), cs.n - 1);
    cap.textContent =
      `f${i}: A TP=${cs.a_tp[i]} FP=${cs.a_fp[i]} FN=${cs.a_fn[i]}` +
      ` · B TP=${cs.b_tp[i]} FP=${cs.b_fp[i]} FN=${cs.b_fn[i]}` +
      ` · ΔTP=${fmtSigned(cs.b_tp[i] - cs.a_tp[i], 0)} ΔFP=${fmtSigned(cs.b_fp[i] - cs.a_fp[i], 0)}`;
    return;
  }
  const series = computeMetricsSeries();
  if (!series || series.n <= 0) {
    cap.textContent = "";
    return;
  }
  const i = Math.min(Math.max(0, frame), series.n - 1);
  const a = series.gt_tp[i];
  const b = series.gt_fn[i];
  const c = series.est_tp[i];
  const d = series.est_fp[i];
  const gTot = a + b;
  const pTot = c + d;
  const r = gTot > 0 ? a / gTot : null;
  const p = pTot > 0 ? c / pTot : null;
  cap.textContent =
    `f${i}: GT·TP=${a} GT·FN=${b} · EST·TP=${c} EST·FP=${d}` +
    ` · R=${r != null ? r.toFixed(3) : "—"} P=${p != null ? p.toFixed(3) : "—"}`;
}

function updateMetricsRatesCaption(){
  const cap = document.getElementById("metricsRatesCaption");
  if (!cap) return;
  const rs = computeRateSeries();
  if (!rs || rs.n <= 0) {
    cap.textContent = "";
    return;
  }
  const i = Math.min(Math.max(0, frame), rs.n - 1);
  cap.textContent =
    `f${i}: TPR=${rs.tpr[i].toFixed(3)} P=${rs.prec[i].toFixed(3)} F1=${rs.f1[i].toFixed(3)} FDR=${rs.fdr[i].toFixed(3)}`;
}

function updateMetricsErrorCaption(){
  const cap = document.getElementById("metricsErrorCaption");
  if (!cap) return;
  const es = computeErrorSeries();
  if (!es || es.n <= 0) {
    cap.textContent = "";
    return;
  }
  const i = Math.min(Math.max(0, frame), es.n - 1);
  cap.textContent =
    `f${i}: ctr=${es.tp_center_distance_mean[i].toFixed(3)} plane=${es.tp_plane_distance_mean[i].toFixed(3)}` +
    ` |yaw|=${es.tp_yaw_error_abs_mean[i].toFixed(3)} sev=${es.frame_severity_max[i].toFixed(3)}`;
}

function updateMetricsErrorPlayhead(){
  if (!metricsErrorPlayhead || !metricsErrorPlayhead.userData || !metricsErrorWrapEl || metricsErrorWrapEl.hidden) return;
  const es = computeErrorSeries();
  if (!es || es.n <= 0) return;
  const n = es.n;
  const xf = n <= 1 ? 0.5 : frame / Math.max(n - 1, 1);
  const x0 = -0.9;
  const x1 = 0.9;
  const x = x0 + (x1 - x0) * xf;
  const y0 = -0.78;
  const y1 = 0.78;
  const g0 = metricsErrorPlayhead.userData.geom;
  const g1 = metricsErrorPlayhead.userData.geomGlow;
  if (g0 && g0.attributes.position) {
    const a = g0.attributes.position.array;
    a[0] = x; a[1] = y0; a[3] = x; a[4] = y1;
    g0.attributes.position.needsUpdate = true;
  }
  if (g1 && g1.attributes.position) {
    const b = g1.attributes.position.array;
    b[0] = x; b[1] = y0; b[3] = x; b[4] = y1;
    g1.attributes.position.needsUpdate = true;
  }
  updateMetricsErrorCaption();
}

function updateMetricsRatesPlayhead(){
  if (!metricsRatesPlayhead || !metricsRatesPlayhead.userData || !metricsRatesWrapEl || metricsRatesWrapEl.hidden) return;
  const rs = computeRateSeries();
  if (!rs || rs.n <= 0) return;
  const n = rs.n;
  const xf = n <= 1 ? 0.5 : frame / Math.max(n - 1, 1);
  const x0 = -0.9;
  const x1 = 0.9;
  const x = x0 + (x1 - x0) * xf;
  const y0 = -0.78;
  const y1 = 0.78;
  const g0 = metricsRatesPlayhead.userData.geom;
  const g1 = metricsRatesPlayhead.userData.geomGlow;
  if (g0 && g0.attributes.position) {
    const a = g0.attributes.position.array;
    a[0] = x; a[1] = y0; a[3] = x; a[4] = y1;
    g0.attributes.position.needsUpdate = true;
  }
  if (g1 && g1.attributes.position) {
    const b = g1.attributes.position.array;
    b[0] = x; b[1] = y0; b[3] = x; b[4] = y1;
    g1.attributes.position.needsUpdate = true;
  }
  updateMetricsRatesCaption();
}

function updateMetricsPlayhead(){
  if (!metricsPlayhead || !metricsPlayhead.userData || metricsWrapEl.hidden) return;
  const series = computeMetricsSeries();
  if (!series || series.n <= 0) return;
  const n = series.n;
  const xf = n <= 1 ? 0.5 : frame / Math.max(n - 1, 1);
  const x0 = -0.9;
  const x1 = 0.9;
  const x = x0 + (x1 - x0) * xf;
  const y0 = -0.78;
  const y1 = 0.78;
  const g0 = metricsPlayhead.userData.geom;
  const g1 = metricsPlayhead.userData.geomGlow;
  if (g0 && g0.attributes.position) {
    const a = g0.attributes.position.array;
    a[0] = x; a[1] = y0; a[3] = x; a[4] = y1;
    g0.attributes.position.needsUpdate = true;
  }
  if (g1 && g1.attributes.position) {
    const b = g1.attributes.position.array;
    b[0] = x; b[1] = y0; b[3] = x; b[4] = y1;
    g1.attributes.position.needsUpdate = true;
  }
  updateMetricsFrameCaption();
  updateMetricsComparePanel();
  updateMetricsRatesPlayhead();
  updateMetricsErrorPlayhead();
}

function syncMetricsPlotVisibility(){
  const chk = document.getElementById("uiShowMetricsCounts");
  if (!chk || !chk.checked) {
    if (metricsWrapEl) metricsWrapEl.hidden = true;
    const cap = document.getElementById("metricsCaption");
    if (cap) cap.textContent = "";
    layoutMetricsStacks();
    resizeMetricsRenderer();
    return;
  }
  rebuildMetricsPlot();
}

function syncMetricsRatesPlotVisibility(){
  const chk = document.getElementById("uiShowMetricsRates");
  if (!chk || !chk.checked) {
    if (metricsRatesWrapEl) metricsRatesWrapEl.hidden = true;
    const cap = document.getElementById("metricsRatesCaption");
    if (cap) cap.textContent = "";
    layoutMetricsStacks();
    resizeMetricsRenderer();
    return;
  }
  rebuildMetricsRatesPlot();
}

function syncMetricsErrorPlotVisibility(){
  const chk = document.getElementById("uiShowMetricsError");
  if (!chk || !chk.checked) {
    if (metricsErrorWrapEl) metricsErrorWrapEl.hidden = true;
    const cap = document.getElementById("metricsErrorCaption");
    if (cap) cap.textContent = "";
    layoutMetricsStacks();
    resizeMetricsRenderer();
    return;
  }
  rebuildMetricsErrorPlot();
}

function applyPanelVisibility(){
  const hudEl = document.getElementById("hud");
  const hudOn = document.getElementById("uiShowHud") && document.getElementById("uiShowHud").checked;
  const fab = document.getElementById("hudRevealBtn");
  if (hudEl) hudEl.hidden = !hudOn;
  if (fab) fab.hidden = hudOn;
  const inspectEl = document.getElementById("inspectPanel");
  const inspectOn = document.getElementById("uiShowInspector") && document.getElementById("uiShowInspector").checked;
  const hasSelection = !!(selectedInspectState && selectedInspectState.current);
  if (inspectEl) inspectEl.hidden = !inspectOn || !hasSelection;
  const camWrap = document.getElementById("cameraViewportWrap");
  const camOn = document.getElementById("uiShowCameraViewport") && document.getElementById("uiShowCameraViewport").checked;
  if (camWrap) camWrap.hidden = !camOn;
  cameraViewportEnabled = !!camOn;
  const camBtn = document.getElementById("toggleCameraViewport");
  if (camBtn) {
    camBtn.textContent = camOn ? "Cameras: on" : "Cameras: off";
    camBtn.setAttribute("aria-pressed", camOn ? "true" : "false");
    camBtn.title = camOn
      ? "Camera images overlay (on) — click or uncheck Panels to hide"
      : "Camera images overlay (off) — click or enable in Panels to show";
  }
  if (!camOn) {
    cameraOverlayGeneration++;
    lastCameraPayload = null;
    clearCameraViewportCanvases();
  }
  if (camOn !== lastCameraPanelCheckbox) {
    lastCameraPanelCheckbox = camOn;
    if (camOn) refreshCameraOverlay().catch((e) => setStatus(`camera: ${e.message}`));
  }
  syncMetricsPlotVisibility();
  updateMetricsComparePanel();
  syncMetricsRatesPlotVisibility();
  syncMetricsErrorPlotVisibility();
}

(function wireHudHideAndAdvanced(){
  document.getElementById("advFogEnable")?.addEventListener("change", applyFogFromSettings);
  ["fogNear", "fogFar"].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", () => {
      syncAdvRangeOutputs();
      applyFogFromSettings();
    });
  });
  ["camNear", "camFar"].forEach((id) => {
    document.getElementById(id)?.addEventListener("input", () => {
      syncAdvRangeOutputs();
      applyCameraClipFromSettings();
    });
  });
  syncAdvRangeOutputs();
  applyFogFromSettings();
  applyCameraClipFromSettings();

  document.getElementById("hudHideBtn")?.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const c = document.getElementById("uiShowHud");
    if (c) c.checked = false;
    applyPanelVisibility();
  });
  document.getElementById("inspectHideBtn")?.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const c = document.getElementById("uiShowInspector");
    if (c) c.checked = false;
    applyPanelVisibility();
  });
})();
updateInspectHud();

/** The on-canvas viewport (CSS px, origin top-left of canvas) that the pointer sits in.
 *  side_by_side renders each run into a half-width viewport with its own aspect; overlay/curtain
 *  both render at the full-canvas aspect (curtain only scissors), so their pick viewport is the whole canvas. */
function compareViewportForPointer(clientX, rect){
  const localX = clientX - rect.left;
  if (compareModeValue() === "side_by_side") {
    const half = rect.width * 0.5;
    const onRight = localX >= half;
    return { vpX: onRight ? half : 0, vpY: 0, vpW: Math.max(1, half), vpH: rect.height };
  }
  return { vpX: 0, vpY: 0, vpW: rect.width, vpH: rect.height };
}

/** Pick the eval-box candidate under a screen point, correcting for the active compare viewport/aspect.
 *  Precise ray hits win; falls back to the nearest box center within a small screen radius (tiny point markers). */
function pickCandidateAtClientXY(clientX, clientY){
  const rect = canvas.getBoundingClientRect();
  if (!(rect.width > 0 && rect.height > 0)) return null;
  const vp = compareViewportForPointer(clientX, rect);
  const fullAspect = rect.width / rect.height;
  applyMainCameraAspect(vp.vpW / vp.vpH);  // match the projection the hovered viewport was rendered with
  try {
    inspectPointer.x = ((clientX - rect.left - vp.vpX) / vp.vpW) * 2 - 1;
    inspectPointer.y = -((clientY - rect.top - vp.vpY) / vp.vpH) * 2 + 1;
    inspectRaycaster.setFromCamera(inspectPointer, camera);
    const pickRoots = [];
    if (boxesGroup.visible) pickRoots.push(boxesGroup);
    if (gtLayerGroup.visible) pickRoots.push(gtLayerGroup);
    if (predLayerGroup.visible) pickRoots.push(predLayerGroup);
    if (!pickRoots.length) return null;
    const compareRunName = compareRunNameForCanvasClientX(clientX);
    const hits = inspectRaycaster.intersectObjects(pickRoots, true);
    const hit = hits.find((row) => {
      const c = getInspectCandidateFromObject(row.object);
      return !!c && candidateMatchesCompareRun(c, compareRunName);
    });
    let cand = hit ? getInspectCandidateFromObject(hit.object) : null;
    if (!cand) {
      const gtVis = gtLayerGroup.visible, prVis = predLayerGroup.visible;
      let bestPx = 26;
      for (const c of externalSelectionCandidates) {
        if (!c || !c.center) continue;
        const isPred = String(c.source || "").indexOf("pred") >= 0;
        if (isPred ? !prVis : !gtVis) continue;
        if (!candidateMatchesCompareRun(c, compareRunName)) continue;
        _hoverProjV.set(Number(c.center[0] || 0), Number(c.center[1] || 0), Number(c.center[2] || 0)).project(camera);
        if (_hoverProjV.z > 1) continue;  // behind the camera
        const sx = rect.left + vp.vpX + (_hoverProjV.x * 0.5 + 0.5) * vp.vpW;
        const sy = rect.top + vp.vpY + (-_hoverProjV.y * 0.5 + 0.5) * vp.vpH;
        const d = Math.hypot(sx - clientX, sy - clientY);
        if (d < bestPx) { bestPx = d; cand = c; }
      }
    }
    return cand;
  } finally {
    applyMainCameraAspect(fullAspect);  // restore; the render loop also resets this each frame
  }
}

function wireInspectPicking(){
  canvas.addEventListener("pointerdown", (ev) => {
    pointerDownState = { x: ev.clientX, y: ev.clientY, moved: false };
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!pointerDownState) return;
    if (Math.hypot(ev.clientX - pointerDownState.x, ev.clientY - pointerDownState.y) > 6) {
      pointerDownState.moved = true;
    }
  });
  canvas.addEventListener("pointerup", (ev) => {
    const pd = pointerDownState;
    pointerDownState = null;
    if (!pd || pd.moved) return;
    const cand = pickCandidateAtClientXY(ev.clientX, ev.clientY);
    if (cand) setSelectedInspectCandidate(cand, { focus: true });
  });
  // Hover-to-reveal label: cheap raycast throttled to one per animation frame; costs nothing while idle.
  canvas.addEventListener("pointermove", (ev) => {
    hoverTipLastEvent = ev;
    if (hoverTipRaf) return;
    hoverTipRaf = requestAnimationFrame(updateHoverTip);
  });
  canvas.addEventListener("pointerleave", hideHoverTip);
}

function hideHoverTip(){
  const tip = document.getElementById("hoverTip");
  if (tip) tip.classList.remove("visible");
  if (canvas) canvas.style.cursor = "";
}

/** rAF-driven: pick the box under the pointer and show its class label in a lightweight HTML chip. */
function updateHoverTip(){
  hoverTipRaf = 0;
  const ev = hoverTipLastEvent;
  const tip = document.getElementById("hoverTip");
  if (!ev || !tip) return;
  if (pointerDownState && pointerDownState.moved) { hideHoverTip(); return; }  // orbiting the camera — stay out of the way
  const cand = pickCandidateAtClientXY(ev.clientX, ev.clientY);
  if (!cand) { hideHoverTip(); return; }
  const meta = cand.meta || {};
  const conf = safeNum(meta.confidence);
  const isPoly = String(meta.shape_type || "").toLowerCase().indexOf("polygon") >= 0 || isEvalPointMarker(meta);
  const subParts = [String(cand.kind || "").toUpperCase(), String(cand.status || "").toUpperCase()].filter(Boolean);
  if (isPoly) subParts.push("polygon");
  let sub = subParts.join(" · ");
  if (conf != null) sub += `  ·  conf ${conf.toFixed(2)}`;
  const labelEl = document.createElement("div");
  labelEl.className = "tip-label";
  labelEl.textContent = String(cand.label || "object");
  const subEl = document.createElement("div");
  subEl.className = "tip-sub";
  subEl.textContent = sub;
  tip.replaceChildren(labelEl, subEl);
  const layerId = String(cand.source || "").indexOf("pred") >= 0 ? "pred" : "gt";
  const accent = evalLineColor(meta && Object.keys(meta).length ? meta : cand, layerId);
  tip.style.setProperty("--tip-accent", `rgba(${(accent >> 16) & 255},${(accent >> 8) & 255},${accent & 255},0.85)`);
  const wrapRect = (document.getElementById("canvasWrap") || canvas).getBoundingClientRect();
  tip.style.left = (ev.clientX - wrapRect.left) + "px";
  tip.style.top = (ev.clientY - wrapRect.top) + "px";
  tip.classList.add("visible");
  canvas.style.cursor = "pointer";
}

function readFloat32Copied(buf, off, count){
  const bytes = new Uint8Array(buf, off, count * 4);
  const copied = new Uint8Array(bytes.length);
  copied.set(bytes);
  return new Float32Array(copied.buffer);
}

function parseFrameBuffer(buf){
  const dv = new DataView(buf);
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 8));
  if (magic !== "T4V3D001" && magic !== "T4V3D002") throw new Error(`unexpected format ${magic}`);
  const frameIndex = dv.getUint32(12, true);
  const timestampUs = Number(dv.getBigUint64(16, true));
  const pointCount = dv.getUint32(24, true);
  const boxCount = dv.getUint32(28, true);
  const tokenLen = dv.getUint16(32, true);
  let off = 34;
  const sampleToken = new TextDecoder().decode(new Uint8Array(buf, off, tokenLen)); off += tokenLen;
  const pts = readFloat32Copied(buf, off, pointCount * 4); off += pointCount * 16;
  const boxStride = magic === "T4V3D002" ? 24 : 7;
  const boxes = readFloat32Copied(buf, off, boxCount * boxStride); off += boxCount * boxStride * 4;
  const labelLen = dv.getUint32(off, true); off += 4;
  const labels = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, off, labelLen)));
  return { frameIndex, timestampUs, sampleToken, pointCount, boxCount, pts, boxes, labels, format: magic };
}

function buildSceneCandidatesFromFrameData(data, frameIndex){
  const out = [];
  if (!data || !Number.isFinite(data.boxCount)) return out;
  if (data.format !== "T4V3D002") return out;
  for (let i = 0; i < data.boxCount; i++) {
    const base = i * 24;
    const corners = [];
    for (let j = 0; j < 24; j++) corners.push(Number(data.boxes[base + j] || 0));
    const center = boxCenterFromCornersArray(corners);
    const size = boxSizeFromCornersArray(corners);
    out.push({
      center,
      size,
      corners,
      label: String((Array.isArray(data.labels) ? data.labels[i] : null) || `box-${i}`),
      kind: "SCENE",
      status: "",
      source: "scene",
      trackId: "",
      frameIndex,
      sampleToken: String(data.sampleToken || ""),
      boxIndex: i,
    });
  }
  return out;
}

/**
 * Rotated box in ego frame (x forward, y left, z up), same as `visualize.py` TargetObject / `_project_bbox_to_roi`:
 * body-x = length (forward), body-y = width (lateral).
 */
function boxEdgesFromCenterWithOpts(box, opts){
  const yawOff = opts && Number.isFinite(opts.yawOffset) ? opts.yawOffset : 0;
  const swapLW = opts && opts.swapLW;
  const cx = Number(box.x ?? box.cx ?? 0);
  const cy = Number(box.y ?? box.cy ?? 0);
  const cz = Number(box.z ?? box.cz ?? 0);
  let l = Math.max(0.01, Number(box.length ?? box.l ?? 1.0));
  let w = Math.max(0.01, Number(box.width ?? box.w ?? 1.0));
  const h = Math.max(0.01, Number(box.height ?? box.h ?? 1.0));
  if (swapLW || box.swap_lw === true) {
    const t = l;
    l = w;
    w = t;
  }
  const yaw = Number(box.yaw ?? box.heading ?? 0.0) + yawOff;
  return boxEdgeSegmentsFromCorners(boxCornersFromPose(cx, cy, cz, l, w, h, yaw));
}

/** PostMessage GT/Pred wireframes: apply T4 alignment offset and optional length/width swap (see URL params). */
function boxEdgesFromCenterExternal(box){
  return boxEdgesFromCenterWithOpts(box, {
    yawOffset: getExternalBboxYawOffset(),
    swapLW: getExternalBboxSwapLW(),
  });
}

function boxEdgesFromCorners(box){
  const arr = Array.isArray(box.corners) ? box.corners : [];
  if (arr.length !== 24) return null;
  return boxEdgeSegmentsFromCorners(arr);
}

/** Aligns with common evaluator color_map: (GT|EST) × (TP|FN|FP).
 *  Rebuilt by `refreshEvalStatusColors()` because the palette is theme-dependent. */
let EVAL_STATUS_COLOR_HEX = evalStatusColorHex();

function evalStatusColorHex(){
  return {
    "GT:TP": TH.hex("boxGtTp"),
    "GT:FN": TH.hex("boxGtFn"),
    "EST:TP": TH.hex("boxEstTp"),
    "EST:FP": TH.hex("boxEstFp"),
  };
}

function refreshEvalStatusColors(){
  EVAL_STATUS_COLOR_HEX = evalStatusColorHex();
}

function normalizeEvalKind(box, layerId){
  const raw = box.kind ?? box.eval_kind ?? box.role ?? box.source_kind;
  if (raw != null && String(raw).trim() !== "") {
    const u = String(raw).toUpperCase();
    if (u === "EST" || u === "PRED" || u === "PREDICTION" || u === "ESTIMATE" || u === "DET") return "EST";
    if (u === "GT" || u === "GROUNDTRUTH" || u === "GTRUTH") return "GT";
  }
  if (layerId === "pred") return "EST";
  if (layerId === "gt") return "GT";
  return "GT";
}

function normalizeEvalStatus(box){
  const s = String(box.status ?? "TP").toUpperCase();
  if (s === "TP" || s === "FN" || s === "FP") return s;
  return "TP";
}

function isEvalPointMarker(box){
  const shape = String((box && box.shape_type) || "").toLowerCase();
  return shape === "invalid_polygon_marker" || shape === "point_marker";
}

function evalLineColor(box, layerId){
  const kind = normalizeEvalKind(box, layerId);
  const st = normalizeEvalStatus(box);
  const key = `${kind}:${st}`;
  if (Object.prototype.hasOwnProperty.call(EVAL_STATUS_COLOR_HEX, key)) {
    return EVAL_STATUS_COLOR_HEX[key];
  }
  return layerId === "pred" ? TH.hex("boxEstTp") : TH.hex("boxGtFallback");
}

/** Same pose math as wireframe path; used for GT solid boxes. */
function readEvalBoxPose(box){
  if (box && box._normalized_eval && Array.isArray(box.center) && Array.isArray(box.size)) {
    return {
      cx: Number(box.center[0] || 0),
      cy: Number(box.center[1] || 0),
      cz: Number(box.center[2] || 0),
      l: Math.max(0.01, Number(box.size[0] || 1.0)),
      w: Math.max(0.01, Number(box.size[1] || 1.0)),
      h: Math.max(0.01, Number(box.size[2] || 1.0)),
      yaw: Number(box.yaw || 0),
    };
  }
  const yawOff = getExternalBboxYawOffset();
  const swapLW = getExternalBboxSwapLW();
  const cx = Number(box.x ?? box.cx ?? 0);
  const cy = Number(box.y ?? box.cy ?? 0);
  const cz = Number(box.z ?? box.cz ?? 0);
  let l = Math.max(0.01, Number(box.length ?? box.l ?? 1.0));
  let w = Math.max(0.01, Number(box.width ?? box.w ?? 1.0));
  const h = Math.max(0.01, Number(box.height ?? box.h ?? 1.0));
  if (swapLW || box.swap_lw === true) {
    const t = l;
    l = w;
    w = t;
  }
  const yaw = Number(box.yaw ?? box.heading ?? 0.0) + yawOff;
  return { cx, cy, cz, l, w, h, yaw };
}

/** Mesh GT boxes need center+size; pure 24-float corners use wireframe fallback. */
function canBuildGtMeshFromBox(box){
  if (box && box.force_wireframe === true) return false;
  if (box && box._normalized_eval && Array.isArray(box.center) && Array.isArray(box.size)) {
    const l0 = Number(box.size[0] || 0);
    const w0 = Number(box.size[1] || 0);
    const h0 = Number(box.size[2] || 0);
    return l0 > 0 && w0 > 0 && h0 > 0;
  }
  if (Array.isArray(box.corners) && box.corners.length === 24) return false;
  const l = Number(box.length ?? box.l ?? 0);
  const wi = Number(box.width ?? box.w ?? 0);
  const hi = Number(box.height ?? box.h ?? 0);
  if (!(l > 0 && wi > 0 && hi > 0)) return false;
  return true;
}

function shouldRenderSeverityEffects(box){
  const severeOnly = !!(document.getElementById("highlightSevereOnly") && document.getElementById("highlightSevereOnly").checked);
  const sev = safeNum(box && box.severity_score);
  if (sev == null) return !severeOnly;
  if (!severeOnly) return true;
  return sev >= 0.62;
}

function velocityVectorsEnabled(){
  return !!(document.getElementById("advShowVelocityVectors") && document.getElementById("advShowVelocityVectors").checked);
}

function addVelocityVectorArrow(group, meta, color){
  if (!group || !meta || !velocityVectorsEnabled()) return null;
  const payload = meta.meta && typeof meta.meta === "object" ? meta.meta : meta;
  const vx = safeNum(payload.vx);
  const vy = safeNum(payload.vy);
  if (vx == null && vy == null) return null;
  const vel = new THREE.Vector3(vx || 0, vy || 0, 0);
  const speed = vel.length();
  if (!(speed > 0.05)) return null;
  const center = Array.isArray(meta.center) ? meta.center : null;
  const size = Array.isArray(meta.size) ? meta.size : [1, 1, 1];
  if (!center || center.length < 3) return null;
  const dir = vel.clone().normalize();
  const maxSize = Math.max(0.6, ...size.map((v) => Math.max(0, Number(v || 0))));
  const arrowLen = Math.min(5.5, Math.max(0.9, maxSize * 0.55, speed * 0.6));
  const lift = Math.max(0.24, 0.5 * Number(size[2] || 0) + 0.2);
  const origin = new THREE.Vector3(center[0], center[1], center[2] + lift);
  const headLength = Math.min(0.68, Math.max(0.24, arrowLen * 0.18));
  const headWidth = Math.min(0.36, Math.max(0.12, maxSize * 0.1));
  const alpha = Math.min(0.96, Math.max(0.28, Number(document.getElementById("boxOpacity")?.value || "0.9")));
  const arrow = new THREE.ArrowHelper(dir, origin, arrowLen, color, headLength, headWidth);
  const lineMat = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: alpha,
    depthWrite: false,
  });
  const coneMat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: alpha,
    depthWrite: false,
  });
  arrow.line.material = lineMat;
  arrow.line.userData.arrowOpacityTarget = true;
  if (arrow.cone) {
    arrow.cone.material = coneMat;
    arrow.cone.userData.arrowOpacityTarget = true;
  }
  arrow.renderOrder = 2;
  arrow.userData.velocityVector = true;
  group.add(arrow);
  return arrow;
}

function addTpQualityAura(parent, box, layerId){
  if (!box || normalizeEvalStatus(box) !== "TP") return;
  if (!fxEnabled("fxTpQuality")) return;
  if (!shouldRenderSeverityEffects(box)) return;
  const sev = clip01(safeNum(box.severity_score) || 0);
  if (sev < 0.22) return;
  const meta = getInspectableBoxMeta(box, layerId);
  if (!meta || !meta.center || !meta.size) return;
  const rel = selectionRelationForMeta(meta.meta || meta, selectedInspectState && selectedInspectState.current);
  if (selectedInspectState && selectedInspectState.current && rel === "none") return;
  const maxSize = Math.max(...(meta.size || [1, 1, 1]));
  const pose = readEvalBoxPose(box);
  const floorZ = (parent && parent.isMesh)
    ? (-Math.max(0.08, 0.5 * Number(meta.size[2] || 0.8)) + 0.035)
    : (Number(meta.center[2] || 0) - Math.max(0.08, 0.5 * Number(meta.size[2] || 0.8)) + 0.035);
  const hw = Math.max(0.05, Number(meta.size[1] || 0.8) * 0.5 + 0.08 + sev * 0.05);
  const hl = Math.max(0.05, Number(meta.size[0] || 0.8) * 0.5 + 0.08 + sev * 0.05);
  const yaw = Number(pose.yaw || 0);
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  const footprintPts = [];
  const corners = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]];
  for (const [bx, by] of corners) {
    if (parent && parent.isMesh) footprintPts.push(bx, by, floorZ);
    else footprintPts.push(meta.center[0] + bx * c - by * s, meta.center[1] + bx * s + by * c, floorZ);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(footprintPts, 3));
  const ring = new THREE.Line(
    geom,
    new THREE.LineBasicMaterial({ color: TH.hex("haloSev"), transparent: true, opacity: 0.2 + sev * 0.18 })
  );
  ring.userData.effectKind = "footprint";
  ring.userData.effectBaseScale = 1;
  ring.userData.effectScaleAmp = 0.04 + sev * 0.08;
  ring.userData.effectOpacityBase = 0.16 + sev * 0.08;
  ring.userData.effectOpacityAmp = 0.03 + sev * 0.03;
  ring.userData.effectPhase = 1.2;
  parent.add(ring);
  addEvalPulseMarker(parent, parent && parent.isMesh ? [0, 0, 0] : meta.center, TH.hex("pulse"), {
    radius: Math.max(0.12, Math.min(0.22, 0.06 * maxSize + sev * 0.05)),
    opacity: 0.14 + sev * 0.08,
    baseScale: 1,
    scaleAmp: 0.12 + sev * 0.12,
    opacityAmp: 0.05 + sev * 0.05,
    phase: 1.8,
  });
}

function evalFootprintVertices(box){
  const fp = box && Array.isArray(box.footprint) ? box.footprint : null;
  if (!fp || fp.length < 3) return null;
  const out = [];
  for (const pt of fp) {
    if (!Array.isArray(pt) || pt.length < 2) return null;
    const x = Number(pt[0]);
    const y = Number(pt[1]);
    const z = Number(pt.length >= 3 ? pt[2] : 0);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    out.push([x, y, z]);
  }
  if (out.length >= 2) {
    const a = out[0];
    const b = out[out.length - 1];
    if (Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-4) out.pop();
  }
  return out.length >= 3 ? out : null;
}

function addEvalFootprintPrism(group, box, layerId){
  const base = evalFootprintVertices(box);
  if (!base) return null;
  const col = evalLineColor(box, layerId);
  const v = Number(document.getElementById("boxOpacity").value || "0.9");
  const pose = readEvalBoxPose(box);
  const h = Math.max(0.15, Number(box.height ?? box.h ?? pose.h ?? 1.0));
  const zBase = base.reduce((sum, pt) => sum + Number(pt[2] || 0), 0) / base.length;
  const zTop = zBase + h;
  const groupObj = new THREE.Group();
  groupObj.userData.opacityFactor = 1;

  const faceVerts = [];
  for (let i = 1; i < base.length - 1; i++) {
    faceVerts.push(base[0][0], base[0][1], zBase, base[i][0], base[i][1], zBase, base[i + 1][0], base[i + 1][1], zBase);
    faceVerts.push(base[0][0], base[0][1], zTop, base[i + 1][0], base[i + 1][1], zTop, base[i][0], base[i][1], zTop);
  }
  for (let i = 0; i < base.length; i++) {
    const j = (i + 1) % base.length;
    faceVerts.push(base[i][0], base[i][1], zBase, base[j][0], base[j][1], zBase, base[j][0], base[j][1], zTop);
    faceVerts.push(base[i][0], base[i][1], zBase, base[j][0], base[j][1], zTop, base[i][0], base[i][1], zTop);
  }
  if (!evalFastRenderMode && faceVerts.length) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(faceVerts, 3));
    geom.computeVertexNormals();
    const faceOpacity = layerId === "pred" ? 0.08 : 0.13;
    const mesh = new THREE.Mesh(
      geom,
      new THREE.MeshBasicMaterial({
        color: col,
        transparent: true,
        opacity: v * faceOpacity,
        depthWrite: false,
        side: THREE.DoubleSide,
      })
    );
    mesh.userData.opacityFactor = faceOpacity;
    groupObj.add(mesh);
  }

  const edgeVerts = [];
  for (let i = 0; i < base.length; i++) {
    const j = (i + 1) % base.length;
    edgeVerts.push(base[i][0], base[i][1], zBase, base[j][0], base[j][1], zBase);
    edgeVerts.push(base[i][0], base[i][1], zTop, base[j][0], base[j][1], zTop);
    edgeVerts.push(base[i][0], base[i][1], zBase, base[i][0], base[i][1], zTop);
  }
  const edgeGeom = new THREE.BufferGeometry();
  edgeGeom.setAttribute("position", new THREE.Float32BufferAttribute(edgeVerts, 3));
  const edgeMul = layerId === "pred" && normalizeEvalStatus(box) === "FP" ? 0.68 : 1.0;
  const lines = new THREE.LineSegments(
    edgeGeom,
    new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: v * edgeMul })
  );
  lines.userData.opacityFactor = edgeMul;
  groupObj.add(lines);

  if (!evalFastRenderMode) addVelocityVectorArrow(group, getInspectableBoxMeta(box, layerId), col);
  if (!evalFastRenderMode) addTpQualityAura(groupObj, box, layerId);
  group.add(groupObj);
  return groupObj;
}

/**
 * GT: semi-transparent filled box + crisp edges (reference geometry).
 * FN: lighter fill so TP reads as “solid match”; edges stay visible.
 */
function addGtEvalBoxMesh(group, box, layerId){
  if (evalFootprintVertices(box)) return addEvalFootprintPrism(group, box, layerId);
  if (evalFastRenderMode) return addEvalWireframeLinesOnly(group, box, layerId);
  const pose = readEvalBoxPose(box);
  const col = evalLineColor(box, layerId);
  const st = normalizeEvalStatus(box);
  const isFn = st === "FN";
  const v = Number(document.getElementById("boxOpacity").value || "0.9");
  const sev = clip01(safeNum(box && box.severity_score) || 0);
  const faceMul = isFn ? 0.08 : (0.16 + 0.08 * sev);
  const edgeMul = isFn ? 0.82 : Math.min(1.18, 0.92 + 0.26 * sev);
  const geom = new THREE.BoxGeometry(pose.l, pose.w, pose.h);
  const faceMat = new THREE.MeshBasicMaterial({
    color: col,
    transparent: true,
    opacity: v * faceMul,
    depthWrite: false,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const mesh = new THREE.Mesh(geom, faceMat);
  mesh.position.set(pose.cx, pose.cy, pose.cz);
  mesh.rotation.z = pose.yaw;
  mesh.userData.opacityFactor = faceMul;
  const edgeGeom = new THREE.EdgesGeometry(geom, 22);
  const edgeMat = new THREE.LineBasicMaterial({
    color: col,
    transparent: true,
    opacity: v * edgeMul,
  });
  const outline = new THREE.LineSegments(edgeGeom, edgeMat);
  outline.userData.opacityFactor = edgeMul;
  mesh.add(outline);
  if (isFn && fxEnabled("fxFnRing")) {
    const hot = currentFrameIsSpotlightHot();
    const floorZ = -Math.max(0.05, pose.h * 0.5) + 0.03;
    addEvalShockwaveRing(mesh, [0, 0, floorZ], col, {
      radius: Math.max(0.85, 0.42 * Math.max(pose.l, pose.w)),
      width: 0.16,
      opacity: hot ? 0.34 : 0.26,
      baseScale: 1,
      scaleAmp: hot ? 0.82 : 0.68,
      opacityAmp: hot ? 0.2 : 0.16,
      phase: 0.6,
    });
    if (hot) {
      addEvalShockwaveRing(mesh, [0, 0, floorZ + 0.01], TH.hex("ring"), {
        radius: Math.max(0.56, 0.28 * Math.max(pose.l, pose.w)),
        width: 0.12,
        opacity: 0.22,
        baseScale: 1,
        scaleAmp: 0.56,
        opacityAmp: 0.12,
        phase: 2.0,
      });
    }
  }
  addVelocityVectorArrow(group, getInspectableBoxMeta(box, layerId), col);
  addTpQualityAura(mesh, box, layerId);
  group.add(mesh);
  return mesh;
}

/** EST: wireframe only; FP drawn fainter so TP/FP separate without new color names. */
function addEstEvalWireframe(group, box, layerId){
  if (evalFootprintVertices(box)) return addEvalFootprintPrism(group, box, layerId);
  if (isEvalPointMarker(box)) return addEvalPointMarker(group, box, layerId);
  const edges = boxEdgesFromCorners(box) || boxEdgesFromCenterExternal(box);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(edges, 3));
  const col = evalLineColor(box, layerId);
  const st = normalizeEvalStatus(box);
  const isFp = st === "FP";
  const v = Number(document.getElementById("boxOpacity").value || "0.9");
  const sev = clip01(safeNum(box && box.severity_score) || 0);
  const edgeMul = isFp ? (0.42 + 0.4 * sev) : Math.min(1.2, 0.9 + 0.28 * sev);
  const m = new THREE.LineBasicMaterial({
    color: col,
    transparent: true,
    opacity: v * edgeMul,
  });
  const seg = new THREE.LineSegments(g, m);
  seg.userData.opacityFactor = edgeMul;
  if (!evalFastRenderMode && isFp) {
    const meta = getInspectableBoxMeta(box, layerId);
    if (meta && meta.center && fxEnabled("fxFpPulse")) {
      if (selectedInspectState && selectedInspectState.current && selectionRelationForMeta(meta.meta || meta, selectedInspectState.current) === "none") {
        // Keep focus scenes cleaner when one object is selected.
      } else {
      addEvalPulseMarker(seg, meta.center, col, {
        radius: Math.max(0.2, Math.min(0.42, 0.12 * Math.max(...(meta.size || [1, 1, 1])))),
        opacity: 0.34,
        baseScale: 1,
        scaleAmp: 0.56,
        opacityAmp: 0.28,
        phase: 1.6,
      });
      }
    }
  }
  if (!evalFastRenderMode) addVelocityVectorArrow(group, getInspectableBoxMeta(box, layerId), col);
  if (!evalFastRenderMode) addTpQualityAura(seg, box, layerId);
  group.add(seg);
  return seg;
}

function labelsEnabled(){
  const el = document.getElementById("showLabels");
  return !!(el && el.checked);
}

/** Rasterized label chips are cached by (text, color): a scene of 200 "car" boxes rasterizes once, not 200×. */
const labelSpriteCache = new Map();
function labelSpriteAssets(label, colorHex){
  const key = `${label}|${colorHex}|${TH.current}`;
  const cached = labelSpriteCache.get(key);
  if (cached) return cached;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const padX = 12, fontPx = 30, baseH = 46;
  const measure = document.createElement("canvas").getContext("2d");
  measure.font = `700 ${fontPx}px DM Sans, sans-serif`;
  const textW = Math.ceil(measure.measureText(label).width);
  const w = textW + padX * 2, h = baseH;
  const cv = document.createElement("canvas");
  cv.width = Math.ceil(w * dpr);
  cv.height = Math.ceil(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const r = (colorHex >> 16) & 255, g = (colorHex >> 8) & 255, b = colorHex & 255;
  ctx.fillStyle = TH.css("labelChipBg");
  ctx.strokeStyle = `rgba(${r},${g},${b},0.9)`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.roundRect(1, 1, w - 2, h - 2, 9);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = TH.css("labelChipText");
  ctx.font = `700 ${fontPx}px DM Sans, sans-serif`;
  ctx.textBaseline = "middle";
  ctx.fillText(label, padX, h / 2 + 1);
  const tex = new THREE.CanvasTexture(cv);
  tex.anisotropy = 4;
  const material = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, depthTest: false });
  const assets = { material, aspect: w / h };
  labelSpriteCache.set(key, assets);
  return assets;
}

/** Compact class-label chip ("truck", "unknown", …) placed at world `pos`, above an eval box. Border color matches status. */
function makeEvalLabelSprite(text, colorHex, pos){
  const { material, aspect } = labelSpriteAssets(String(text || "object"), colorHex);
  const sprite = new THREE.Sprite(material);
  // Constant world height (~0.62 m) so text stays legible regardless of distance or label length.
  const worldH = 0.62;
  sprite.scale.set(worldH * aspect, worldH, 1);
  sprite.position.set(Number(pos[0] || 0), Number(pos[1] || 0), Number(pos[2] || 0));
  sprite.renderOrder = 4;
  sprite.userData.opacityFactor = 1;
  sprite.userData.keepOpacity = true;  // stay legible when the box-opacity slider dims wireframes
  return sprite;
}

function addEvalPointMarker(group, box, layerId){
  const pose = readEvalBoxPose(box);
  const col = evalLineColor(box, layerId);
  const v = Number(document.getElementById("boxOpacity").value || "0.9");
  const h = Math.max(0.15, Number(box.height ?? box.h ?? pose.h ?? 1.0));
  const radius = Math.max(0.14, Math.min(0.34, 0.12 + h * 0.04));
  const markerGroup = new THREE.Group();
  markerGroup.position.set(pose.cx, pose.cy, pose.cz);
  markerGroup.userData.opacityFactor = 1;

  const sphere = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 14, 10),
    new THREE.MeshBasicMaterial({
      color: col,
      transparent: true,
      opacity: Math.min(1, v * 0.92),
      depthWrite: false,
    })
  );
  sphere.userData.opacityFactor = 1;
  markerGroup.add(sphere);

  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([0, 0, -h * 0.5, 0, 0, h * 0.5], 3)
  );
  const line = new THREE.Line(
    lineGeom,
    new THREE.LineBasicMaterial({
      color: col,
      transparent: true,
      opacity: Math.min(1, v * 0.72),
      depthWrite: false,
    })
  );
  line.userData.opacityFactor = 0.8;
  markerGroup.add(line);

  if (!evalFastRenderMode && fxEnabled("fxFpPulse")) {
    addEvalPulseMarker(markerGroup, [0, 0, 0], col, {
      radius: Math.max(0.2, radius * 1.4),
      opacity: 0.3,
      baseScale: 1,
      scaleAmp: 0.52,
      opacityAmp: 0.24,
      phase: 1.6,
    });
  }
  group.add(markerGroup);
  return markerGroup;
}

function setInspectUserData(obj, candidate){
  if (!obj || !candidate) return;
  obj.userData.inspectCandidate = candidate;
  obj.userData.inspectSelectable = true;
}

function getInspectCandidateFromObject(obj){
  let cur = obj || null;
  while (cur) {
    if (cur.userData && cur.userData.inspectSelectable && cur.userData.inspectCandidate) {
      return cur.userData.inspectCandidate;
    }
    cur = cur.parent || null;
  }
  return null;
}

function candidateDistance(a, b){
  if (!a || !b || !a.center || !b.center) return Infinity;
  const dx = Number(a.center[0] || 0) - Number(b.center[0] || 0);
  const dy = Number(a.center[1] || 0) - Number(b.center[1] || 0);
  const dz = Number(a.center[2] || 0) - Number(b.center[2] || 0);
  return Math.hypot(dx, dy, dz);
}

function candidateSizeDelta(a, b){
  const as = Array.isArray(a && a.size) ? a.size : [];
  const bs = Array.isArray(b && b.size) ? b.size : [];
  if (as.length < 3 || bs.length < 3) return 0;
  return Math.abs(as[0] - bs[0]) + Math.abs(as[1] - bs[1]) + Math.abs(as[2] - bs[2]);
}

function candidateRunName(cand){
  return evalRunNameFromMeta(cand && (cand.meta || cand));
}

function candidateRunsCompatible(a, b){
  const ar = candidateRunName(a);
  const br = candidateRunName(b);
  return !ar || !br || ar === br;
}

function scoreCandidateMatch(sel, cand){
  if (!sel || !cand) return Infinity;
  if (!candidateRunsCompatible(sel, cand)) return Infinity;
  const sid = String(sel.trackId || "");
  const cid = String(cand.trackId || "");
  if (sid && cid && sid === cid && sel.source === cand.source) return 0;
  const sp = String(sel.pairId || "");
  const cp = String(cand.pairId || "");
  if (sp && cp && sp === cp && sel.source === cand.source) return 0.02;
  if (sel.source !== cand.source) return Infinity;
  if (sel.kind && cand.kind && sel.kind !== cand.kind) return Infinity;
  const selLabel = String(sel.label || "").toLowerCase();
  const candLabel = String(cand.label || "").toLowerCase();
  if (selLabel && candLabel && selLabel !== candLabel) return Infinity;
  return candidateDistance(sel, cand) + (0.35 * candidateSizeDelta(sel, cand));
}

function cameraRowSelectionRelation(row, sel){
  if (!row || !sel) return "none";
  const rowUuid = String(row.uuid || "");
  const rowPair = String(row.pair_uuid || "");
  const selUuid = String(sel.trackId || sel.uuid || "");
  const selPair = String(sel.pairId || "");
  const rowLayer = String(row.layer || row.kind || "").toLowerCase();
  const selLayer = String(sel.source || "").toLowerCase();
  if (rowUuid && selUuid && rowUuid === selUuid) return "self";
  if (rowPair && selPair && rowPair === selPair) {
    if ((rowLayer === "pred" && selLayer === "eval-pred") || (rowLayer === "gt" && selLayer === "eval-gt")) return "self";
    return "pair";
  }
  return "none";
}

function selectionRelationForMeta(meta, sel){
  if (!meta || !sel) return "none";
  const metaUuid = String(meta.uuid ?? meta.trackId ?? "");
  const metaPair = String(meta.pair_uuid ?? meta.pairId ?? "");
  const selUuid = String(sel.trackId || sel.uuid || "");
  const selPair = String(sel.pairId || "");
  const metaSource = String(meta.source || "").toLowerCase();
  const selSource = String(sel.source || "").toLowerCase();
  if (metaUuid && selUuid && metaUuid === selUuid && metaSource === selSource) return "self";
  if (metaPair && selPair && metaPair === selPair) return "pair";
  return "none";
}

function findBestCandidateForSelection(candidates, sel){
  let best = null;
  let bestScore = Infinity;
  for (const cand of candidates || []) {
    const score = scoreCandidateMatch(sel, cand);
    if (score < bestScore) {
      bestScore = score;
      best = cand;
    }
  }
  if (!best) return null;
  if (bestScore > 12) return null;
  return best;
}

function focusCameraOnCandidate(candidate, smooth, preserveOffset){
  if (!candidate || !candidate.center) return;
  const dst = new THREE.Vector3(candidate.center[0], candidate.center[1], candidate.center[2]);
  const keepOffset = preserveOffset !== false;
  const target = controls.target.clone();
  const delta = keepOffset
    ? camera.position.clone().sub(target)
    : new THREE.Vector3(-12, -8, 4.5);
  if (smooth) {
    controls.target.lerp(dst, 0.75);
    camera.position.copy(dst.clone().add(delta));
  } else {
    controls.target.copy(dst);
    camera.position.copy(dst.clone().add(delta));
  }
  controls.update();
}

function renderInspectSelection(){
  clearGroupDeep(inspectOverlayGroup);
  if (!selectedInspectState || !selectedInspectState.current || !selectedInspectState.current.corners) return;
  const current = selectedInspectState.current;
  const meta = current.meta || {};
  const corners = current.corners;
  const upAxis = new THREE.Vector3(0, 1, 0);
  const bracketMat = new THREE.MeshBasicMaterial({ color: TH.hex("bracket"), transparent: true, opacity: 0.98, depthWrite: false });
  const edgeMap = {
    0: [1, 3, 4], 1: [0, 2, 5], 2: [1, 3, 6], 3: [0, 2, 7],
    4: [0, 5, 7], 5: [1, 4, 6], 6: [2, 5, 7], 7: [3, 4, 6],
  };
  for (let i = 0; i < 8; i++) {
    const p0 = new THREE.Vector3(corners[i * 3], corners[i * 3 + 1], corners[i * 3 + 2]);
    for (const ni of edgeMap[i] || []) {
      if (ni < i) continue;
      const p1 = new THREE.Vector3(corners[ni * 3], corners[ni * 3 + 1], corners[ni * 3 + 2]);
      const dir = p1.clone().sub(p0);
      const len = dir.length();
      if (!(len > 1e-5)) continue;
      dir.normalize();
      const segLen = Math.min(len * 0.38, 0.62);
      const segments = [
        [p0, p0.clone().add(dir.clone().multiplyScalar(segLen))],
        [p1, p1.clone().add(dir.clone().multiplyScalar(-segLen))],
      ];
      for (const [a, b] of segments) {
        const delta = b.clone().sub(a);
        const segMid = a.clone().addScaledVector(delta, 0.5);
        const segLenActual = delta.length();
        const geom = new THREE.CylinderGeometry(0.028, 0.028, segLenActual, 10);
        const rod = new THREE.Mesh(geom, bracketMat);
        rod.position.copy(segMid);
        rod.quaternion.setFromUnitVectors(upAxis, delta.normalize());
        inspectOverlayGroup.add(rod);
      }
    }
  }
  const c = current.center;
  const color = current.status === "FP" ? TH.hex("inspectFp") : (current.status === "FN" ? TH.hex("inspectFn") : TH.hex("inspectTp"));
  const pulse = new THREE.Mesh(
    new THREE.SphereGeometry(0.34, 20, 16),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.92, depthWrite: false })
  );
  pulse.position.set(c[0], c[1], c[2]);
  pulse.userData.focusPulse = true;
  inspectOverlayGroup.add(pulse);
  const halo = new THREE.Mesh(
    new THREE.RingGeometry(0.44, 0.66, 44),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.44, side: THREE.DoubleSide, depthWrite: false })
  );
  halo.position.set(c[0], c[1], c[2] - Math.max(0.08, 0.5 * Number(current.size && current.size[2] || 0.8)) + 0.06);
  halo.userData.focusHalo = true;
  inspectOverlayGroup.add(halo);
  const spriteCanvas = document.createElement("canvas");
  spriteCanvas.width = 320;
  spriteCanvas.height = 62;
  const sctx = spriteCanvas.getContext("2d");
  if (sctx) {
    sctx.fillStyle = "rgba(6,10,20,0.84)";
    sctx.strokeStyle = current.status === "FP" ? "rgba(255,142,113,0.76)" : (current.status === "FN" ? "rgba(255,201,120,0.72)" : "rgba(146,224,255,0.76)");
    sctx.lineWidth = 2;
    sctx.beginPath();
    sctx.roundRect(1, 1, 318, 60, 12);
    sctx.fill();
    sctx.stroke();
    sctx.fillStyle = "#f4f8ff";
    sctx.font = "700 18px DM Sans";
    sctx.fillText(`${String(current.kind || "BOX")} · ${String(current.status || current.source || "").toUpperCase()}`, 14, 26);
    sctx.fillStyle = "#b7cbff";
    sctx.font = "600 15px JetBrains Mono";
    const conf = safeNum(meta.confidence);
    const confText = conf != null ? `conf ${Number(conf).toFixed(3)}` : "conf —";
    sctx.fillText(`${String(current.label || "object")}  |  ${confText}`, 14, 47);
  }
  const tex = new THREE.CanvasTexture(spriteCanvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.position.set(c[0], c[1], c[2] + Math.max(1.15, 0.76 * Math.max(...(current.size || [1, 1, 1]))));
  sprite.scale.set(4.2, 0.82, 1);
  inspectOverlayGroup.add(sprite);
}

function renderInspectTrail(points){
  clearGroupDeep(inspectTrailGroup);
  if (!Array.isArray(points) || points.length < 1) return;
  if (points.length >= 2) {
    const arr = [];
    for (const p of points) arr.push(p[0], p[1], p[2]);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
    inspectTrailGroup.add(new THREE.Line(
      g,
      new THREE.LineBasicMaterial({ color: TH.hex("trailLine"), transparent: true, opacity: 0.8 })
    ));
  }
  points.forEach((p, idx) => {
    const alpha = idx === points.length - 1 ? 0.9 : (0.22 + 0.5 * (idx / Math.max(1, points.length - 1)));
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(idx === points.length - 1 ? 0.16 : 0.11, 12, 10),
      new THREE.MeshBasicMaterial({ color: idx === points.length - 1 ? TH.hex("trailDotLast") : TH.hex("trailDot"), transparent: true, opacity: alpha })
    );
    m.position.set(p[0], p[1], p[2]);
    inspectTrailGroup.add(m);
  });
}

async function rebuildInspectTrack(){
  const sel = selectedInspectState;
  const gen = ++inspectTrackGeneration;
  if (!sel || !sel.current) {
    renderInspectTrail([]);
    updateInspectHud();
    return;
  }
  const radius = 8;
  const frames = [];
  for (let i = Math.max(0, frame - radius); i <= Math.min(totalFrames - 1, frame + radius); i++) frames.push(i);
  const resolved = [];
  for (const fi of frames) {
    try {
      let candidates = [];
      if (sel.current.source === "scene") {
        const data = await fetchFrame(fi);
        if (gen !== inspectTrackGeneration) return;
        candidates = buildSceneCandidatesFromFrameData(data, fi);
      } else {
        candidates = buildExternalCandidatesForFrameIndex(fi);
      }
      const hit = findBestCandidateForSelection(candidates, sel.current);
      if (hit) resolved.push(hit);
    } catch (_) {}
  }
  if (gen !== inspectTrackGeneration) return;
  sel.track = resolved;
  renderInspectTrail(resolved.map((r) => r.center));
  updateInspectHud();
}

function setSelectedInspectCandidate(candidate, opts){
  if (!candidate) {
    selectedInspectState = null;
    inspectTrackGeneration++;
    renderInspectSelection();
    renderInspectTrail([]);
    updateInspectHud();
    applyPanelVisibility();
    if (cameraViewportEnabled && lastCameraPayload) renderCameraViewport(lastCameraPayload, cameraOverlayGeneration).catch(() => {});
    return;
  }
  selectedInspectState = {
    current: candidate,
    track: [candidate],
  };
  renderInspectSelection();
  if (!opts || opts.focus !== false) focusCameraOnCandidate(candidate, false);
  updateInspectHud();
  applyPanelVisibility();
  rebuildInspectTrack().catch(() => {});
  if (cameraViewportEnabled && lastCameraPayload) renderCameraViewport(lastCameraPayload, cameraOverlayGeneration).catch(() => {});
}

function updateSelectedCandidateForFrame(){
  if (!selectedInspectState) return;
  const merged = [...sceneSelectionCandidates, ...externalSelectionCandidates];
  const hit = findBestCandidateForSelection(merged, selectedInspectState.current);
  if (!hit) {
    selectedInspectState.current.frameIndex = frame;
    renderInspectSelection();
    updateInspectHud();
    applyPanelVisibility();
    if (cameraViewportEnabled && lastCameraPayload) renderCameraViewport(lastCameraPayload, cameraOverlayGeneration).catch(() => {});
    return;
  }
  selectedInspectState.current = hit;
  renderInspectSelection();
  if (inspectLockFocus) focusCameraOnCandidate(hit, true);
  updateInspectHud();
  applyPanelVisibility();
  rebuildInspectTrack().catch(() => {});
  if (cameraViewportEnabled && lastCameraPayload) renderCameraViewport(lastCameraPayload, cameraOverlayGeneration).catch(() => {});
}

function renderInspectErrorGlyph(sel){
  const canvas = document.getElementById("inspectErrorGlyph");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const displayW = Math.max(620, Math.round(canvas.clientWidth || 680));
  const displayH = Math.max(320, Math.round(displayW * 0.5));
  if (canvas.width !== Math.round(displayW * dpr) || canvas.height !== Math.round(displayH * dpr)) {
    canvas.width = Math.round(displayW * dpr);
    canvas.height = Math.round(displayH * dpr);
    canvas.style.height = `${displayH}px`;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const labelTag = (text, x, y, opts) => {
    const padX = 8;
    const padY = 5;
    ctx.save();
    ctx.font = `${opts && opts.bold ? "700 " : "600 "}13px ${'JetBrains Mono'}, monospace`;
    const tw = ctx.measureText(text).width;
    const bw = tw + padX * 2;
    const bh = 22;
    const bx = Math.max(6, Math.min(w - bw - 6, x));
    const by = Math.max(6, Math.min(h - bh - 6, y));
    ctx.fillStyle = (opts && opts.bg) || "rgba(7,12,24,0.86)";
    ctx.strokeStyle = (opts && opts.stroke) || "rgba(146,168,228,0.32)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(bx, by, bw, bh, 8);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = (opts && opts.color) || "#eef4ff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, bx + padX, by + bh * 0.52);
    ctx.restore();
  };
  const w = displayW;
  const h = displayH;
  const stat = String(sel && sel.status || "");
  const theme = stat === "FP"
    ? { bg0: "#160909", bg1: "#3b1412", grid: "rgba(255,141,108,0.18)", accent: "#ff916f", accent2: "#ffd19b", text: "#fff2eb", soft: "#ffb29a" }
    : stat === "FN"
      ? { bg0: "#171005", bg1: "#412707", grid: "rgba(255,198,108,0.18)", accent: "#ffc76f", accent2: "#fff0b6", text: "#fff7eb", soft: "#ffd495" }
      : { bg0: "#09111d", bg1: "#0c2b3b", grid: "rgba(115,196,255,0.18)", accent: "#78deff", accent2: "#ffd89b", text: "#eef7ff", soft: "#a5dfff" };
  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, theme.bg1);
  bg.addColorStop(0.55, theme.bg0);
  bg.addColorStop(1, "#050814");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  const grad = ctx.createRadialGradient(w * 0.24, h * 0.22, 12, w * 0.34, h * 0.42, Math.max(w, h) * 0.76);
  grad.addColorStop(0, "rgba(255,255,255,0.08)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = 1;
  for (let x = 30; x < w - 24; x += 44) {
    ctx.beginPath();
    ctx.moveTo(x, 34);
    ctx.lineTo(x, h - 26);
    ctx.stroke();
  }
  for (let y = 36; y < h - 22; y += 42) {
    ctx.beginPath();
    ctx.moveTo(22, y);
    ctx.lineTo(w - 22, y);
    ctx.stroke();
  }
  if (!sel || !sel.meta) {
    labelTag("No pair error data", w * 0.5 - 72, h * 0.5 - 12, { color: theme.text, bg: "rgba(10,16,32,0.9)" });
    return;
  }
  const meta = sel.meta;
  const xErr = safeNum(meta.x_error) || 0;
  const yErr = safeNum(meta.y_error) || 0;
  const yawErr = safeNum(meta.yaw_error) || 0;
  const speed = safeNum(meta.speed_mps);
  const vx = safeNum(meta.vx);
  const vy = safeNum(meta.vy);
  const conf = safeNum(meta.confidence);
  const ctr = safeNum(meta.center_distance);
  const plane = safeNum(meta.plane_distance);
  const yaw = safeNum(meta.yaw);
  const drift = safeNum(meta.xy_error_mag);
  const hasPair = ctr != null || plane != null || drift != null || safeNum(meta.x_error) != null || safeNum(meta.y_error) != null || safeNum(meta.yaw_error) != null;
  const leftX = 26;
  const leftW = 154;
  const mapX = 204;
  const mapY = 56;
  const mapW = 270;
  const mapH = 160;
  const rightX = w - 182;
  const cx = mapX + mapW * 0.5;
  const cy = mapY + mapH * 0.5;
  const scale = 44;
  const tx = cx + xErr * scale;
  const ty = cy - yErr * scale;
  ctx.fillStyle = "rgba(7,11,22,0.34)";
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.beginPath();
  ctx.roundRect(leftX - 8, 18, leftW + 16, h - 36, 18);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.roundRect(mapX - 20, 18, mapW + 40, h - 36, 18);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.roundRect(rightX - 8, 18, 160, h - 36, 18);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = theme.soft;
  ctx.font = "700 13px DM Sans";
  ctx.fillText("SYSTEMS", leftX, 40);
  ctx.fillText("XY OFFSET MAP", mapX, 40);
  ctx.fillText("DYAW / MATCH", rightX, 40);
  const writeLeft = (label, value, y, accent) => {
    ctx.fillStyle = theme.soft;
    ctx.font = "700 11px DM Sans";
    ctx.fillText(label, leftX, y);
    ctx.fillStyle = accent || theme.text;
    ctx.font = "700 16px JetBrains Mono";
    ctx.fillText(value, leftX, y + 18);
  };
  writeLeft("CONFIDENCE", fmtMaybe(conf, "", 3), 78);
  writeLeft("SPEED", fmtMaybe(speed, " m/s", 2), 132);
  writeLeft("VELOCITY", `${fmtMaybe(vx, " m/s", 2)} / ${fmtMaybe(vy, " m/s", 2)}`, 186);
  writeLeft("YAW", fmtMaybe(yaw, " rad", 2), 240);
  if (!hasPair && sel.status !== "TP") {
    labelTag(sel.status === "FP" ? "Unmatched estimate" : "Missed ground truth", mapX + 40, 100, { color: theme.text, bg: "rgba(10,16,32,0.88)", stroke: theme.grid });
    labelTag("Pair metrics unavailable", mapX + 58, 136, { color: theme.soft, bg: "rgba(10,16,32,0.88)", stroke: theme.grid });
  } else {
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.beginPath();
    ctx.moveTo(cx, mapY + 4); ctx.lineTo(cx, mapY + mapH - 4);
    ctx.moveTo(mapX + 6, cy); ctx.lineTo(mapX + mapW - 6, cy);
    ctx.stroke();
    labelTag("GT REF", mapX + 2, mapY + 4, { color: theme.text, bg: "rgba(8,14,28,0.88)" });
    labelTag("EST", mapX + 94, mapY + 4, { color: theme.accent2, bg: "rgba(32,18,8,0.9)", stroke: "rgba(255,186,126,0.28)" });
    ctx.strokeStyle = theme.accent2;
    ctx.fillStyle = theme.accent2;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    const ang = Math.atan2(ty - cy, tx - cx);
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - 10 * Math.cos(ang - 0.42), ty - 10 * Math.sin(ang - 0.42));
    ctx.lineTo(tx - 10 * Math.cos(ang + 0.42), ty - 10 * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = theme.text;
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = theme.accent2;
    ctx.beginPath(); ctx.arc(tx, ty, 6.5, 0, Math.PI * 2); ctx.fill();
  }
  ctx.strokeStyle = theme.accent;
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.arc(rightX + 60, 126, 54, -Math.PI / 2, -Math.PI / 2 + yawErr, yawErr < 0);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(rightX + 60, 126, 54, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = theme.text;
  ctx.font = "700 12px JetBrains Mono";
  ctx.fillText("dyaw", rightX + 28, 112);
  ctx.font = "700 22px JetBrains Mono";
  ctx.fillText(fmtSigned(yawErr, 2), rightX + 10, 140);
  ctx.font = "600 11px DM Sans";
  ctx.fillStyle = theme.soft;
  ctx.fillText("rad", rightX + 116, 140);
  const writeRight = (label, value, y) => {
    ctx.fillStyle = theme.soft;
    ctx.font = "700 11px DM Sans";
    ctx.fillText(label, rightX, y);
    ctx.fillStyle = theme.text;
    ctx.font = "700 15px JetBrains Mono";
    ctx.fillText(value, rightX, y + 18);
  };
  writeRight("CENTER", fmtMaybe(ctr, " m", 3), 198);
  writeRight("PLANE", fmtMaybe(plane, " m", 3), 236);
  writeRight("XY DRIFT", fmtMaybe(drift, " m", 3), 274);
  labelTag(`dx ${fmtSigned(xErr, 2)} m`, mapX + 6, h - 42, { color: theme.text, bg: "rgba(10,16,32,0.88)", stroke: theme.grid });
  labelTag(`dy ${fmtSigned(yErr, 2)} m`, mapX + 132, h - 42, { color: theme.text, bg: "rgba(10,16,32,0.88)", stroke: theme.grid });
}

function updateInspectHud(){
  const wrap = document.getElementById("inspectHud");
  const empty = document.getElementById("inspectEmpty");
  const body = document.getElementById("inspectBody");
  const chip = document.getElementById("inspectTypeChip");
  if (!wrap || !empty || !body || !chip) return;
  wrap.hidden = false;
  const sel = selectedInspectState && selectedInspectState.current ? selectedInspectState.current : null;
  if (!sel) {
    chip.textContent = "none";
    wrap.dataset.status = "";
    empty.hidden = false;
    body.hidden = true;
    renderInspectErrorGlyph(null);
    const panel = document.getElementById("inspectPanel");
    if (panel) panel.hidden = true;
    return;
  }
  const meta = sel.meta || {};
  const track = Array.isArray(selectedInspectState.track) ? selectedInspectState.track : [];
  const setText = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  };
  empty.hidden = true;
  body.hidden = false;
  wrap.dataset.status = String(sel.status || sel.kind || "");
  chip.textContent = `${sel.kind || "BOX"} · ${sel.status || sel.source}`;
  setText("inspectLabel", sel.label || "box");
  setText("inspectFrame", `${sel.frameIndex}/${Math.max(0, totalFrames - 1)}`);
  setText("inspectPos", fmtVec3(sel.center, 2));
  setText("inspectSize", fmtVec3(sel.size, 2));
  setText("inspectYaw", fmtMaybe(safeNum(meta.yaw), " rad", 2));
  setText("inspectVelocity", `${fmtMaybe(safeNum(meta.vx), " m/s", 2)} · ${fmtMaybe(safeNum(meta.vy), " m/s", 2)}`);
  setText("inspectSpeed", fmtMaybe(safeNum(meta.speed_mps), " m/s", 2));
  setText("inspectCenterDistance", fmtMaybe(safeNum(meta.center_distance), " m", 3));
  setText("inspectPlaneDistance", fmtMaybe(safeNum(meta.plane_distance), " m", 3));
  setText("inspectPairDt", fmtMaybe(safeNum(meta.pair_dt_sec), " s", 4));
  setText("inspectErrorXY", `${fmtSigned(safeNum(meta.x_error), 2)} / ${fmtSigned(safeNum(meta.y_error), 2)}`);
  setText("inspectErrorZYaw", `${fmtSigned(safeNum(meta.z_error), 2)} / ${fmtSigned(safeNum(meta.yaw_error), 2)}`);
  setText("inspectErrorMag", fmtMaybe(safeNum(meta.xy_error_mag), " m", 3));
  setText("inspectTopic", meta.topic_name ? String(meta.topic_name) : "—");
  setText("inspectFrameId", meta.frame_id ? String(meta.frame_id) : "—");
  setText("inspectRun", meta.run ? String(meta.run) : "—");
  setText("inspectDataset", meta.t4dataset_name ? String(meta.t4dataset_name) : (meta.t4dataset_id ? String(meta.t4dataset_id) : "—"));
  setText("inspectScenario", meta.scenario_name ? String(meta.scenario_name) : "—");
  setText("inspectKpiCenter", fmtMaybe(safeNum(meta.center_distance), " m", 3));
  setText("inspectKpiPlane", fmtMaybe(safeNum(meta.plane_distance), " m", 3));
  setText("inspectKpiYaw", fmtMaybe(safeNum(meta.yaw_error) != null ? Math.abs(Number(meta.yaw_error)) : null, " rad", 3));
  setText("inspectKpiXY", fmtMaybe(safeNum(meta.xy_error_mag), " m", 3));
  setText("inspectStatusChip", `${sel.status || "status"}${meta.pair_uuid ? ` · pair ${String(meta.pair_uuid).slice(0, 8)}` : ""}`);
  setText("inspectConfidenceChip", `confidence ${fmtMaybe(safeNum(meta.confidence), "", 3)}`);
  setText("inspectSeverityChip", `score ${fmtMaybe(safeNum(meta.severity_score), "", 3)}`);
  setText("inspectFocusChip", inspectLockFocus ? "focus locked" : "focus free");
  setText("inspectSourceChip", String(sel.source || "scene"));
  setText("inspectConfidenceHero", fmtMaybe(safeNum(meta.confidence), "", 3));
  setText("inspectConfidenceNote", meta.confidence != null ? `${sel.status || sel.kind} confidence signal` : "No confidence data");
  setText("inspectSeverityBucket", `Severity ${String(meta.severity_bucket || "low")}`);
  setText("inspectSeverityScore", `Score ${fmtMaybe(safeNum(meta.severity_score), "", 3)}`);
  const confFill = document.getElementById("inspectConfidenceFill");
  if (confFill) confFill.style.width = `${Math.max(0, Math.min(100, (safeNum(meta.confidence) || 0) * 100))}%`;
  let motionText = "stationary";
  if (track.length >= 2) {
    const first = track[0].center;
    const last = track[track.length - 1].center;
    motionText = `${fmtNum(last[0] - first[0], 2)}m x, ${fmtNum(last[1] - first[1], 2)}m y`;
  }
  setText("inspectMotion", motionText);
  const summaryEl = document.getElementById("inspectSummary");
  if (summaryEl) {
    if (sel.status === "FP") summaryEl.textContent = `Confidence is ${fmtMaybe(safeNum(meta.confidence), "", 3)} for this unmatched estimate. Severity reflects false-positive confidence, clutter, and any available motion cues.`;
    else if (sel.status === "FN") summaryEl.textContent = `Missed ground truth near ${fmtVec3(sel.center, 1)}. Severity stays high because the object was not recovered by the estimate layer.`;
    else summaryEl.textContent = `Matched object with dominant issue: ${String(meta.severity_reason || "low TP error")}. Confidence remains primary, while dyaw, drift, and pair distances explain why this TP still deserves attention.`;
  }
  renderInspectErrorGlyph(sel);
  const noteEl = document.getElementById("inspectTrailNote");
  if (noteEl) noteEl.textContent = track.length > 1
    ? `Trail frames ${track[0].frameIndex} → ${track[track.length - 1].frameIndex} (${track.length} matched)`
    : "Trail is building from nearby frames.";
  const panel = document.getElementById("inspectPanel");
  const showInspector = !!(document.getElementById("uiShowInspector") && document.getElementById("uiShowInspector").checked);
  if (panel) panel.hidden = !showInspector;
}

function animatePulseMarkers(root, tsMs){
  if (!root) return;
  root.traverse((obj) => {
    if (!obj || !obj.userData) return;
    const isPulse = !!obj.userData.pulseMarker;
    const kind = String(obj.userData.effectKind || "");
    if (!isPulse && !kind) return;
    const phase = Number(isPulse ? obj.userData.pulsePhase : obj.userData.effectPhase || 0);
    const u = 0.5 + 0.5 * Math.sin((tsMs || 0) * 0.006 + phase);
    const baseScale = Number(isPulse ? obj.userData.pulseBaseScale : obj.userData.effectBaseScale || 1);
    const scaleAmp = Number(isPulse ? obj.userData.pulseScaleAmp : obj.userData.effectScaleAmp || 0.25);
    const opacityBase = Number(isPulse ? obj.userData.pulseOpacityBase : obj.userData.effectOpacityBase || 0.4);
    const opacityAmp = Number(isPulse ? obj.userData.pulseOpacityAmp : obj.userData.effectOpacityAmp || 0.2);
    if (kind === "pillar") {
      obj.scale.set(1 + scaleAmp * u, 1, 1 + scaleAmp * u);
    } else {
      const s = baseScale + scaleAmp * u;
      obj.scale.setScalar(s);
    }
    if (obj.material) {
      const alpha = kind === "ghost"
        ? Math.max(0.02, opacityBase + opacityAmp * (0.35 + 0.65 * u))
        : opacityBase + opacityAmp * u;
      obj.material.opacity = alpha;
    }
  });
}

function addEvalWireframeLinesOnly(group, box, layerId){
  if (evalFootprintVertices(box)) return addEvalFootprintPrism(group, box, layerId);
  const edges = boxEdgesFromCorners(box) || boxEdgesFromCenterExternal(box);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(edges, 3));
  const col = evalLineColor(box, layerId);
  const v = Number(document.getElementById("boxOpacity").value || "0.9");
  const m = new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: v });
  const seg = new THREE.LineSegments(g, m);
  seg.userData.opacityFactor = 1;
  const st = normalizeEvalStatus(box);
  if (!evalFastRenderMode && layerId === "gt" && st === "FN" && fxEnabled("fxFnRing")) {
    const hot = currentFrameIsSpotlightHot();
    const meta = getInspectableBoxMeta(box, layerId);
    if (meta && meta.center) {
      const floorZ = Number(meta.center[2] || 0) - Math.max(0.08, 0.5 * Number(meta.size && meta.size[2] || 0.8)) + 0.03;
      addEvalShockwaveRing(seg, [meta.center[0], meta.center[1], floorZ], col, {
        radius: Math.max(0.85, 0.42 * Math.max(...(meta.size || [1, 1, 1]))),
        width: 0.16,
        opacity: hot ? 0.34 : 0.24,
        baseScale: 1,
        scaleAmp: hot ? 0.82 : 0.68,
        opacityAmp: hot ? 0.2 : 0.16,
        phase: 0.6,
      });
      if (hot) {
        addEvalShockwaveRing(seg, [meta.center[0], meta.center[1], floorZ + 0.01], TH.hex("ring"), {
          radius: Math.max(0.56, 0.28 * Math.max(...(meta.size || [1, 1, 1]))),
          width: 0.12,
          opacity: 0.22,
          baseScale: 1,
          scaleAmp: 0.56,
          opacityAmp: 0.12,
          phase: 2.0,
        });
      }
    }
  }
  if (!evalFastRenderMode) addTpQualityAura(seg, box, layerId);
  group.add(seg);
  if (!evalFastRenderMode) addVelocityVectorArrow(group, getInspectableBoxMeta(box, layerId), col);
  return seg;
}

/** Polygon-shaped rows. The evaluator marks a polygon whose box dimensions are zero as
 * "invalid_polygon_marker", which is most of them, so both names mean polygon here. */
function isEvalPolygonBox(box){
  const shape = String((box && box.shape_type) || "").toLowerCase();
  return shape === "polygon" || shape === "invalid_polygon_marker";
}
function evalPolygonsVisible(){
  const el = document.getElementById("showPolygons");
  return !el || !!el.checked;
}

function isEvalTupleVisible(box, layerId){
  // Polygons cut across the four tuples rather than forming a fifth one: a polygon is
  // still a GT or EST row, so this is a second gate, not another bucket.
  if (isEvalPolygonBox(box) && !evalPolygonsVisible()) return false;
  const kind = normalizeEvalKind(box, layerId);
  const st = normalizeEvalStatus(box);
  const showGtTp = !!(document.getElementById("showGtTp") && document.getElementById("showGtTp").checked);
  const showGtFn = !!(document.getElementById("showGtFn") && document.getElementById("showGtFn").checked);
  const showEstTp = !!(document.getElementById("showEstTp") && document.getElementById("showEstTp").checked);
  const showEstFp = !!(document.getElementById("showEstFp") && document.getElementById("showEstFp").checked);
  if (kind === "GT" && st === "TP") return showGtTp;
  if (kind === "GT" && st === "FN") return showGtFn;
  if (kind === "EST" && st === "TP") return showEstTp;
  if (kind === "EST" && st === "FP") return showEstFp;
  if (layerId === "gt") return showGtTp || showGtFn;
  return showEstTp || showEstFp;
}

function countVisibleEvalBoxes(boxes, layerId){
  if (!Array.isArray(boxes)) return 0;
  let n = 0;
  for (const b of boxes) {
    if (isEvalTupleVisible(b, layerId)) n++;
  }
  return n;
}

function drawLayerBoxes(group, boxes, layerId){
  clearGroupDeep(group);
  const nextCandidates = [];
  if (!Array.isArray(boxes)) {
    if (layerId === "gt") externalSelectionCandidates = externalSelectionCandidates.filter((c) => c && c.source === "eval-pred");
    else externalSelectionCandidates = externalSelectionCandidates.filter((c) => c && c.source === "eval-gt");
    return;
  }
  for (const b of boxes){
    if (!isEvalTupleVisible(b, layerId)) continue;
    labelableCount += 1;
    const candidate = buildInspectCandidate(getInspectableBoxMeta(b, layerId), null);
    if (layerId === "gt") {
      const obj = canBuildGtMeshFromBox(b) ? addGtEvalBoxMesh(group, b, layerId) : addEvalWireframeLinesOnly(group, b, layerId);
      if (obj && obj.userData) obj.userData.compareRun = evalRunNameFromMeta(b);
      setInspectUserData(obj, candidate);
    } else {
      const obj = addEstEvalWireframe(group, b, layerId);
      if (obj && obj.userData) obj.userData.compareRun = evalRunNameFromMeta(b);
      setInspectUserData(obj, candidate);
    }
    if (candidate && labelsEnabled() && labelBudget > 0) {
      labelBudget -= 1;
      const c = candidate.center || [0, 0, 0];
      const s = candidate.size || [0, 0, 0];
      const zTop = Number(c[2] || 0) + Math.max(0, Number(s[2] || 0)) * 0.5 + 0.55;
      const spr = makeEvalLabelSprite(candidate.label, evalLineColor(b, layerId), [c[0], c[1], zTop]);
      spr.userData.compareRun = evalRunNameFromMeta(b);
      group.add(spr);
    }
    if (candidate) nextCandidates.push(candidate);
  }
  if (layerId === "gt") {
    externalSelectionCandidates = [
      ...nextCandidates,
      ...externalSelectionCandidates.filter((c) => c && c.source === "eval-pred"),
    ];
  } else {
    externalSelectionCandidates = [
      ...externalSelectionCandidates.filter((c) => c && c.source === "eval-gt"),
      ...nextCandidates,
    ];
  }
}

function renderExternalLayers(){
  const visibleEvalCount = countVisibleEvalBoxes(externalLayers.gt, "gt") + countVisibleEvalBoxes(externalLayers.pred, "pred");
  evalFastRenderMode = visibleEvalCount > EVAL_FAST_RENDER_BOX_THRESHOLD;
  labelBudget = LABEL_SPRITE_CAP;
  labelableCount = 0;
  drawLayerBoxes(gtLayerGroup, externalLayers.gt, "gt");
  drawLayerBoxes(predLayerGroup, externalLayers.pred, "pred");
  const labelCntEl = document.getElementById("cntLabels");
  if (labelCntEl) {
    const capped = labelsEnabled() && labelableCount > LABEL_SPRITE_CAP;
    labelCntEl.textContent = capped ? `${LABEL_SPRITE_CAP}/${labelableCount}` : String(labelableCount);
    labelCntEl.title = capped
      ? `Showing ${LABEL_SPRITE_CAP} of ${labelableCount} labels (capped for performance). Hover any box for its label.`
      : `${labelableCount} eval boxes in this frame`;
  }
  const gtOn = !!(document.getElementById("showGtTp") && document.getElementById("showGtTp").checked)
    || !!(document.getElementById("showGtFn") && document.getElementById("showGtFn").checked);
  const prOn = !!(document.getElementById("showEstTp") && document.getElementById("showEstTp").checked)
    || !!(document.getElementById("showEstFp") && document.getElementById("showEstFp").checked);
  gtLayerGroup.visible = gtOn;
  predLayerGroup.visible = prOn;
}

function evalRunNameFromMeta(meta){
  const text = String((meta && (meta.run ?? meta.meta?.run)) ?? "").trim();
  return text && text.toLowerCase() !== "nan" && text.toLowerCase() !== "none" ? text : "";
}

function currentEvalRuns(){
  if (Array.isArray(compareRunOrder) && compareRunOrder.length >= 2) return compareRunOrder.slice();
  const seen = new Set();
  const out = [];
  const addRows = (rows) => {
    if (!Array.isArray(rows)) return;
    for (const row of rows) {
      const name = evalRunNameFromMeta(row);
      if (name && !seen.has(name)) {
        seen.add(name);
        out.push(name);
      }
    }
  };
  addRows(externalLayers.gt);
  addRows(externalLayers.pred);
  return out;
}

function compareModeValue(){
  const el = document.getElementById("compareViewMode");
  return el ? String(el.value || "overlay") : "overlay";
}

function compareRunNameForCanvasClientX(clientX){
  const mode = compareModeValue();
  if (mode !== "side_by_side" && mode !== "curtain") return "";
  const runs = currentEvalRuns();
  if (!Array.isArray(runs) || runs.length < 2) return "";
  const rect = canvas.getBoundingClientRect();
  if (!(rect.width > 0)) return "";
  const x = clientX - rect.left;
  if (mode === "side_by_side") return x < rect.width * 0.5 ? runs[0] : runs[1];
  const split = Math.max(0.06, Math.min(0.94, compareCurtainRatio));
  return x < rect.width * split ? runs[0] : runs[1];
}

function candidateMatchesCompareRun(candidate, runName){
  if (!runName || !candidate) return true;
  const candRun = candidateRunName(candidate);
  return !candRun || candRun === runName;
}

function setCompareLabels(runA, runB, mode){
  const la = document.getElementById("compareLabelA");
  const lb = document.getElementById("compareLabelB");
  const handle = document.getElementById("compareCurtainHandle");
  const active = mode === "side_by_side" || mode === "curtain";
  if (la) {
    la.textContent = runA ? `A · ${runA}` : "A";
    la.classList.toggle("visible", active);
  }
  if (lb) {
    lb.textContent = runB ? `B · ${runB}` : "B";
    lb.classList.toggle("visible", active);
  }
  if (handle) {
    handle.classList.toggle("visible", active);
    handle.classList.toggle("is-curtain", mode === "curtain");
    const x = mode === "curtain" ? compareCurtainRatio : 0.5;
    handle.style.left = `${Math.round(x * 10000) / 100}%`;
  }
}

function setCompareRunVisibilityForGroup(group, runName){
  if (!group) return;
  const taggedRuns = [];
  for (const child of group.children || []) {
    const cand = getInspectCandidateFromObject(child);
    const boxRun = String((child.userData && child.userData.compareRun) || evalRunNameFromMeta(cand && cand.meta) || "").trim();
    if (boxRun) taggedRuns.push(boxRun);
  }
  const hasRunTags = taggedRuns.length > 0;
  for (const child of group.children || []) {
    const cand = getInspectCandidateFromObject(child);
    const boxRun = String((child.userData && child.userData.compareRun) || evalRunNameFromMeta(cand && cand.meta) || "").trim();
    child.visible = !runName || !hasRunTags || boxRun === runName;
  }
}

function setCompareRunVisibility(runName){
  compareRenderRunFilter = runName || null;
  setCompareRunVisibilityForGroup(gtLayerGroup, compareRenderRunFilter);
  setCompareRunVisibilityForGroup(predLayerGroup, compareRenderRunFilter);
}

function renderSceneViewport(x, y, w, h, runName){
  const vw = Math.max(1, Math.floor(w));
  const vh = Math.max(1, Math.floor(h));
  renderer.setViewport(Math.floor(x), Math.floor(y), vw, vh);
  renderer.setScissor(Math.floor(x), Math.floor(y), vw, vh);
  applyMainCameraAspect(vw / vh);
  setCompareRunVisibility(runName);
  renderer.render(scene, camera);
}

function renderSceneFullViewportClipped(clipX, clipY, clipW, clipH, fullW, fullH, runName){
  const fw = Math.max(1, Math.floor(fullW));
  const fh = Math.max(1, Math.floor(fullH));
  const sx = Math.floor(clipX);
  const sy = Math.floor(clipY);
  const sw = Math.max(1, Math.floor(clipW));
  const sh = Math.max(1, Math.floor(clipH));
  renderer.setViewport(0, 0, fw, fh);
  renderer.setScissor(sx, sy, sw, sh);
  applyMainCameraAspect(fw / fh);
  setCompareRunVisibility(runName);
  renderer.render(scene, camera);
}

function renderMainScene(){
  const size = renderer.getSize(new THREE.Vector2());
  const w = Math.max(1, Math.floor(size.x));
  const h = Math.max(1, Math.floor(size.y));
  const mode = compareModeValue();
  const runs = currentEvalRuns();
  const runA = runs[0] || "A";
  const runB = runs[1] || "B";
  const activeMode = mode === "side_by_side" || mode === "curtain" ? mode : "overlay";
  setCompareLabels(runA, runB, activeMode);

  if (activeMode === "side_by_side") {
    const leftW = Math.max(1, Math.round(w * 0.5));
    const rightW = Math.max(1, w - leftW);
    renderer.setScissorTest(true);
    renderer.setViewport(0, 0, w, h);
    renderer.setScissor(0, 0, w, h);
    renderer.clear();
    renderSceneViewport(0, 0, leftW, h, runA);
    renderSceneViewport(leftW, 0, rightW, h, runB);
    setCompareRunVisibility(null);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    applyMainCameraAspect(w / h);
    return;
  }

  if (activeMode === "curtain") {
    const split = Math.max(0.06, Math.min(0.94, compareCurtainRatio));
    const curtainX = Math.max(1, Math.min(w - 1, Math.round(w * split)));
    const rightW = Math.max(1, w - curtainX);
    renderer.setScissorTest(true);
    renderer.setViewport(0, 0, w, h);
    renderer.setScissor(0, 0, w, h);
    renderer.clear();
    renderSceneFullViewportClipped(0, 0, w, h, w, h, runA);
    renderSceneFullViewportClipped(curtainX, 0, rightW, h, w, h, runB);
    setCompareRunVisibility(null);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, w, h);
    applyMainCameraAspect(w / h);
    return;
  }

  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, w, h);
  applyMainCameraAspect(w / h);
  setCompareRunVisibility(null);
  renderer.render(scene, camera);
}

function buildExternalCandidatesForFrameIndex(i){
  const out = [];
  const pushFrom = (rows, layerId) => {
    if (!Array.isArray(rows)) return;
    for (const row of rows) {
      if (!isEvalTupleVisible(row, layerId)) continue;
      const cand = buildInspectCandidate(getInspectableBoxMeta(row, layerId), null);
      if (cand) {
        cand.frameIndex = i;
        out.push(cand);
      }
    }
  };
  if (!bboxLayersByFrame || typeof bboxLayersByFrame !== "object") return out;
  const sk = String(i);
  const entry = Object.prototype.hasOwnProperty.call(bboxLayersByFrame, sk) ? bboxLayersByFrame[sk] : bboxLayersByFrame[i];
  if (!entry || typeof entry !== "object") return out;
  const norm = normalizeEvalEntry(entry, i);
  pushFrom(norm.gt, "gt");
  pushFrom(norm.pred, "pred");
  return out;
}

function normalizeExternalLayerPayload(payload, frameIndexOverride){
  const raw = payload && typeof payload === "object" ? payload : {};
  const norm = normalizeEvalEntry(raw, frameIndexOverride != null ? frameIndexOverride : frame);
  return { ...raw, gt: norm.gt, pred: norm.pred };
}

function setExternalLayerPayload(payload, opts){
  const doAck = !opts || opts.ack !== false;
  const doCameraRefresh = !opts || opts.refreshCamera !== false;
  const norm = normalizeExternalLayerPayload(payload, frame);
  const gt = Array.isArray(norm.gt) ? norm.gt : [];
  const pred = Array.isArray(norm.pred) ? norm.pred : [];
  if (gt.length && viewerDebugEnabled) {
    const b0 = gt[0];
    window.__t4EvalDebugFirstGt = {
      frame,
      uuid: b0.uuid,
      status: b0.status,
      force_wireframe: b0.force_wireframe === true,
      size: b0.size,
      yaw: b0.yaw,
      corners0: Array.isArray(b0.corners) ? b0.corners.slice(0, 6) : null,
    };
    dbg("first normalized GT", window.__t4EvalDebugFirstGt);
  }
  externalLayers = { gt, pred };
  renderExternalLayers();
  updateEvalHud();
  syncSpotlightFrames();
  const b = countEvalBucketsFromBoxes(gt, pred);
  setStatus(`ready · cached=${cache.size} · ext(GT·TP=${b.gt_tp},GT·FN=${b.gt_fn},EST·TP=${b.est_tp},EST·FP=${b.est_fp})`);
  if (doCameraRefresh && cameraViewportEnabled) refreshCameraOverlay().catch(() => {});
  if (!doAck || !viewerDebugEnabled) return;
  const ack = `/viewer/three/debug/message-received?t4dataset_id=${encodeURIComponent(dataset)}&scenario_name=${encodeURIComponent(scenario)}&frame_index=${frame}&gt_count=${gt.length}&pred_count=${pred.length}&matched_count=0`;
  fetch(ack)
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((body) => {
      dbg("bbox_layers ack ok", body);
    })
    .catch((err) => {
      dbg("bbox_layers ack failed", err);
    });
}

/** Apply eval parquet layers for frame index *i* when bboxLayersByFrame is set (parent sent bbox_layers_by_frame). */
function applyEvalLayersForFrame(i){
  if (!bboxLayersByFrame) return;
  const sk = String(i);
  const entry = Object.prototype.hasOwnProperty.call(bboxLayersByFrame, sk)
    ? bboxLayersByFrame[sk]
    : bboxLayersByFrame[i];
  if (entry && typeof entry === "object"){
    setExternalLayerPayload(entry, { ack: false, refreshCamera: false });
  } else {
    externalLayers = { gt: [], pred: [] };
    renderExternalLayers();
    updateEvalHud();
    setStatus(`ready · cached=${cache.size} · ext(empty)`);
  }
}

async function fetchFrame(i){
  if (cache.has(i)) {
    // Refresh recency so ping-pong scrubbing doesn't evict the frames in use.
    const v = cache.get(i);
    cache.delete(i);
    cache.set(i, v);
    return v;
  }
  // Dedup: showFrame and prefetchAround racing on the same index share one
  // request instead of double-fetching and double-parsing.
  if (inflightFrames.has(i)) return inflightFrames.get(i);
  const p = (async () => {
    const url = `/viewer/three/frame.bin?t4dataset_id=${encodeURIComponent(dataset)}&scenario_name=${encodeURIComponent(scenario)}&frame_index=${i}${qv}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`frame ${i}: HTTP ${res.status}`);
    const parsed = parseFrameBuffer(await res.arrayBuffer());
    cacheStoreFrame(i, parsed);
    return parsed;
  })();
  inflightFrames.set(i, p);
  try {
    return await p;
  } finally {
    inflightFrames.delete(i);
  }
}

async function loadLanelet(){
  if (laneletLoaded && laneletFrame === frame) return;
  setStatus("loading lanelet...");
  const url = `/viewer/three/lanelet-lines?t4dataset_id=${encodeURIComponent(dataset)}&scenario_name=${encodeURIComponent(scenario)}&frame_index=${frame}${qv}&max_segments=90000&clip_radius_m=170`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`lanelet HTTP ${res.status}`);
  const data = await res.json();
  if (!data.available) {
    setStatus(`lanelet unavailable: ${data.reason || "not found"}`);
    laneletLoaded = true;
    return;
  }
  const segs = Array.isArray(data.segments) ? data.segments : [];
  const roleColors = { left: TH.hex("laneLeft"), right: TH.hex("laneRight"), centerline: TH.hex("laneCenter"), unknown: TH.hex("laneUnknown") };
  const byRole = { left: [], right: [], centerline: [], unknown: [] };
  let xmin=Infinity, xmax=-Infinity, ymin=Infinity, ymax=-Infinity;
  for (const s of segs){
    const role = byRole[s.role] ? s.role : "unknown";
    byRole[role].push(s.x0, s.y0, s.z0 || 0, s.x1, s.y1, s.z1 || 0);
    xmin = Math.min(xmin, s.x0, s.x1); xmax = Math.max(xmax, s.x0, s.x1);
    ymin = Math.min(ymin, s.y0, s.y1); ymax = Math.max(ymax, s.y0, s.y1);
  }
  laneletBounds = isFinite(xmin) ? { xmin, xmax, ymin, ymax } : null;
  while (laneletGroup.children.length) laneletGroup.remove(laneletGroup.children[0]);
  for (const role of Object.keys(byRole)){
    const arr = byRole[role];
    if (!arr.length) continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
    const m = new THREE.LineBasicMaterial({ color: roleColors[role], transparent: true, opacity: role === "centerline" ? 0.95 : 0.78 });
    laneletGroup.add(new THREE.LineSegments(g, m));
  }
  laneletGroup.visible = laneletEnabled;
  laneletLoaded = true;
  laneletFrame = frame;
  setStatus(`lanelet loaded: ${segs.length} segments`);
}

async function prefetchAround(i){
  const tasks = [];
  for (let d = 1; d <= 3; d++) {
    for (const j of [i + d, i - d]) {
      if (j >= 0 && j < totalFrames && !cache.has(j)) {
        tasks.push(fetchFrame(j).catch(() => null));
      }
    }
  }
  await Promise.all(tasks);
}

const RING_CAMERA_ORDER = [
  "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
  "CAM_BACK_LEFT",  "CAM_BACK", "CAM_BACK_RIGHT",
];

function shortCamLabel(name){
  const s = String(name || "");
  const parts = s.split(/[/\\]/);
  return parts[parts.length - 1] || s;
}

/** Order channel names for a stable 2×3 grid (front ring, then extras). */
function orderedCamerasForGrid(list){
  const L = Array.isArray(list) ? list.map(String) : [];
  const out = [];
  const norm = (x) => x.replace(/[^a-z0-9]+/gi, "").toUpperCase();
  for (const pref of RING_CAMERA_ORDER) {
    const hit = L.find((c) => norm(c) === norm(pref));
    if (hit && !out.includes(hit)) out.push(hit);
  }
  const rest = L.filter((c) => !out.includes(c)).sort();
  for (const c of rest) out.push(c);
  return out.slice(0, 6);
}

/** Map `cameras_payload` rows into six slots (row-major 3×2). */
function buildCameraSlots(rows){
  const byCam = {};
  for (const r of rows || []) {
    if (!r || r.camera == null) continue;
    byCam[String(r.camera)] = r;
  }
  const order = orderedCamerasForGrid(Object.keys(byCam));
  const slots = new Array(6).fill(null);
  for (let i = 0; i < Math.min(6, order.length); i++) slots[i] = byCam[order[i]] || null;
  return slots;
}

function ensureCameraGridSlots(){
  const grid = document.getElementById("cameraGridWrap");
  if (!grid || grid.children.length >= 6) return;
  grid.innerHTML = "";
  for (let i = 0; i < 6; i++) {
    const slot = document.createElement("div");
    slot.className = "cam-slot";
    const c = document.createElement("canvas");
    const lbl = document.createElement("span");
    lbl.className = "cam-slot-label";
    slot.appendChild(c);
    slot.appendChild(lbl);
    grid.appendChild(slot);
  }
}

/** Single vs 2×3: toggles classes so CSS hides the inactive branch (fixes `hidden` lost to `.cam-grid6{display:grid}`). */
function applyCameraViewportModeFromUi(){
  const sel = document.getElementById("cameraLayoutMode");
  const layout = sel && sel.value ? sel.value : "single";
  const wantGrid = layout === "grid6";
  const wrap = document.getElementById("cameraViewportWrap");
  const singleWrap = document.getElementById("cameraSingleWrap");
  const gridWrap = document.getElementById("cameraGridWrap");
  if (wrap) {
    wrap.classList.toggle("cam-mode-grid", wantGrid);
    wrap.classList.toggle("cam-mode-single", !wantGrid);
  }
  if (wantGrid) {
    if (singleWrap) singleWrap.hidden = true;
    if (gridWrap) {
      gridWrap.hidden = false;
      ensureCameraGridSlots();
    }
  } else {
    if (singleWrap) singleWrap.hidden = false;
    if (gridWrap) gridWrap.hidden = true;
  }
}

function currentCameraViewportLayout(){
  const sel = document.getElementById("cameraLayoutMode");
  return sel && sel.value ? sel.value : "single";
}

function cameraPayloadAspectForLayout(payload, layout){
  const fallback = 16 / 9;
  if (!payload || typeof payload !== "object") return fallback;
  const rows = Array.isArray(payload.cameras_payload) ? payload.cameras_payload.filter(Boolean) : [];
  const base = rows[0] || payload;
  const iw = Math.max(1, Number(base && base.width ? base.width : 0));
  const ih = Math.max(1, Number(base && base.height ? base.height : 0));
  if (!(iw > 0 && ih > 0)) return fallback;
  const oneAspect = iw / ih;
  return layout === "grid6" ? (3 * iw) / (2 * ih) : oneAspect;
}

function autoSizeCameraViewport(payload){
  const wrap = document.getElementById("cameraViewportWrap");
  if (!wrap) return;
  const layout = currentCameraViewportLayout();
  const aspect = cameraPayloadAspectForLayout(payload, layout);
  const chromeH = 86;
  const minW = 260;
  const minBodyH = layout === "grid6" ? 220 : 140;
  const minH = chromeH + minBodyH;
  const maxW = Math.max(minW, Math.min(window.innerWidth - 24, layout === "grid6" ? 1200 : 980));
  const maxH = Math.max(minH, Math.min(window.innerHeight - 48, layout === "grid6" ? 820 : 760));

  let width = maxW;
  let bodyH = Math.round(width / Math.max(0.2, aspect));
  let height = chromeH + bodyH;
  if (height > maxH) {
    height = maxH;
    bodyH = Math.max(minBodyH, height - chromeH);
    width = Math.round(bodyH * aspect);
  }
  width = Math.max(minW, Math.min(width, maxW));
  height = Math.max(minH, Math.min(height, maxH));

  wrap.style.width = `${Math.round(width)}px`;
  wrap.style.height = `${Math.round(height)}px`;
}

function resetCameraViewportUserSize(){
  const wrap = document.getElementById("cameraViewportWrap");
  if (!wrap) return;
  delete wrap.dataset.userSize;
  try {
    localStorage.removeItem(CAM_VP_SIZE_STORAGE_KEY);
  } catch (_) {}
}

function ensureOverlayCameraOptions(payload){
  const sel = document.getElementById("overlayCamera");
  if (!sel || !payload) return;
  const cams = Array.isArray(payload.available_cameras) ? payload.available_cameras : [];
  const prev = sel.value;
  sel.innerHTML = "";
  for (const c of cams) {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = shortCamLabel(c);
    sel.appendChild(o);
  }
  if (cams.length && prev && cams.includes(prev)) sel.value = prev;
  else if (payload.camera && cams.includes(payload.camera)) sel.value = payload.camera;
  else if (cams.length) sel.value = cams[0];
}

function updateCameraToolbarMode(){
  const layout = document.getElementById("cameraLayoutMode") && document.getElementById("cameraLayoutMode").value;
  const sel = document.getElementById("overlayCamera");
  const lab = sel && sel.closest("label");
  const grid6 = layout === "grid6";
  if (lab) lab.style.opacity = grid6 ? "0.5" : "";
  if (sel) sel.disabled = !!grid6;
  applyCameraViewportModeFromUi();
}

async function fetchCameraOverlay(i){
  const layout = document.getElementById("cameraLayoutMode") && document.getElementById("cameraLayoutMode").value;
  const allMode = layout === "grid6";
  const camSel = document.getElementById("overlayCamera");
  const selected = camSel && camSel.value ? camSel.value : "";
  const camParam = !allMode && selected ? `&camera=${encodeURIComponent(selected)}` : "";
  const allFlag = allMode ? "&all_cameras=true" : "";
  const showAnn = document.getElementById("showOverlayAnnotations") && document.getElementById("showOverlayAnnotations").checked;
  const rangeEl = document.getElementById("cameraOverlayMaxRange");
  const maxBoxEl = document.getElementById("cameraOverlayMaxBoxes");
  let maxRange = rangeEl ? Number(rangeEl.value) : 120;
  let maxBoxes = maxBoxEl ? Number(maxBoxEl.value) : 128;
  if (!Number.isFinite(maxRange) || maxRange < 5) maxRange = 120;
  if (!Number.isFinite(maxBoxes) || maxBoxes < 1) maxBoxes = 128;
  const rangeParam = `&max_range_m=${encodeURIComponent(String(maxRange))}`;
  const maxBoxParam = `&max_scene_boxes=${encodeURIComponent(String(Math.min(500, Math.max(1, Math.floor(maxBoxes)))))}`;
  const yawOff = getExternalBboxYawOffset();
  const swapLw = getExternalBboxSwapLW();
  const extAlignParam =
    `&external_bbox_yaw_offset=${encodeURIComponent(String(yawOff))}` +
    `&external_bbox_swap_lw=${swapLw ? "true" : "false"}`;
  const qs =
    `t4dataset_id=${encodeURIComponent(dataset)}` +
    `&scenario_name=${encodeURIComponent(scenario)}&frame_index=${i}${qv}` +
    `${camParam}${allFlag}&show_annotations=${showAnn ? "true" : "false"}${rangeParam}${maxBoxParam}${extAlignParam}`;
  const predN = Array.isArray(externalLayers.pred) ? externalLayers.pred.length : 0;
  const gtN = Array.isArray(externalLayers.gt) ? externalLayers.gt.length : 0;
  const hasExt = predN > 0 || gtN > 0;
  let res;
  if (hasExt) {
    res = await fetch(`/viewer/three/camera-overlay?${qs}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pred: Array.isArray(externalLayers.pred) ? externalLayers.pred : [],
        gt: Array.isArray(externalLayers.gt) ? externalLayers.gt : [],
      }),
    });
  } else {
    res = await fetch(`/viewer/three/camera-overlay?${qs}`);
  }
  if (!res.ok) throw new Error(`camera overlay HTTP ${res.status}`);
  return await res.json();
}

/** Stroke 2D rects (pixel coords in full-image space) onto a scaled canvas. */
function strokeBoxes2dOnCanvas(ctx, bxs, iw, ih, cw, ch, strokeStyle, lineWidth){
  if (!ctx || !Array.isArray(bxs) || !bxs.length) return;
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  for (const b of bxs) {
    const x0 = (Number(b.x0) / iw) * cw;
    const y0 = (Number(b.y0) / ih) * ch;
    const x1 = (Number(b.x1) / iw) * cw;
    const y1 = (Number(b.y1) / ih) * ch;
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  }
}

function cameraOverlayStyleForRow(row, kind){
  const st = normalizeCameraOverlayStatus(row);
  if (kind === "GT" && st === "FN") return { stroke: "rgba(255, 153, 51, 0.95)", fill: "rgba(255, 153, 51, 0.10)" };
  if (kind === "GT") return { stroke: "rgba(75, 208, 141, 0.95)", fill: "rgba(75, 208, 141, 0.10)" };
  if (kind === "EST" && st === "FP") return { stroke: "rgba(255, 102, 102, 0.82)", fill: "rgba(255, 102, 102, 0.09)" };
  return { stroke: "rgba(102, 179, 255, 0.96)", fill: "rgba(102, 179, 255, 0.08)" };
}

function normalizeCameraOverlayStatus(b){
  if (!b || typeof b !== "object") return "TP";
  const raw = (b.status != null ? String(b.status) : "").trim().toUpperCase();
  if (!raw) return "TP";
  if (raw === "TRUE_POSITIVE") return "TP";
  if (raw === "FALSE_NEGATIVE") return "FN";
  if (raw === "FALSE_POSITIVE") return "FP";
  return raw;
}

function isCameraOverlayEvalVisible(kind, status, row){
  if (isEvalPolygonBox(row) && !evalPolygonsVisible()) return false;
  const st = String(status || "").toUpperCase();
  const showGtTp = !!(document.getElementById("showGtTp") && document.getElementById("showGtTp").checked);
  const showGtFn = !!(document.getElementById("showGtFn") && document.getElementById("showGtFn").checked);
  const showEstTp = !!(document.getElementById("showEstTp") && document.getElementById("showEstTp").checked);
  const showEstFp = !!(document.getElementById("showEstFp") && document.getElementById("showEstFp").checked);
  if (kind === "GT" && st === "TP") return showGtTp;
  if (kind === "GT" && st === "FN") return showGtFn;
  if (kind === "EST" && st === "TP") return showEstTp;
  if (kind === "EST" && st === "FP") return showEstFp;
  if (kind === "GT") return showGtTp || showGtFn;
  return showEstTp || showEstFp;
}

function drawCameraOverlayEvalRows(ctx, canvas, rows, kind, iw, ih, cw, ch){
  if (!ctx || !canvas || !Array.isArray(rows) || !rows.length) {
    if (canvas && !cameraCanvasHitRegions.has(canvas)) cameraCanvasHitRegions.set(canvas, []);
    return;
  }
  const sel = selectedInspectState && selectedInspectState.current ? selectedInspectState.current : null;
  const selectionOnly = !!(document.getElementById("cameraSelectionOnly") && document.getElementById("cameraSelectionOnly").checked && sel);
  const hitRegions = Array.isArray(cameraCanvasHitRegions.get(canvas)) ? cameraCanvasHitRegions.get(canvas).slice() : [];
  for (const r of rows) {
    const st = normalizeCameraOverlayStatus(r);
    if (!isCameraOverlayEvalVisible(kind, st, r)) continue;
    const relation = cameraRowSelectionRelation(r, sel);
    if (selectionOnly && relation === "none") continue;
    const style = cameraOverlayStyleForRow(r, kind);
    const x0 = (Number(r.x0) / iw) * cw;
    const y0 = (Number(r.y0) / ih) * ch;
    const x1 = (Number(r.x1) / iw) * cw;
    const y1 = (Number(r.y1) / ih) * ch;
    const w = x1 - x0;
    const h = y1 - y0;
    const sev = clip01(safeNum(r.severity_score) || 0);
    let alpha = relation === "self" ? 1 : (relation === "pair" ? 0.92 : 0.68);
    if (sel && relation === "none") alpha = selectionOnly ? 0 : 0.2;
    if (alpha <= 0) continue;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = style.fill;
    ctx.fillRect(x0, y0, w, h);
    ctx.strokeStyle = relation === "self" ? "#ffffff" : (relation === "pair" ? "#ffd38f" : style.stroke);
    ctx.lineWidth = relation === "self" ? 3 : (1.6 + sev * 1.3);
    ctx.strokeRect(x0, y0, w, h);
    if (w > 54 && h > 18) {
      const label = `${st}${r.confidence != null ? ` ${Number(r.confidence).toFixed(2)}` : ""}`;
      ctx.font = "10px JetBrains Mono, monospace";
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(x0, Math.max(0, y0 - 13), tw + 8, 12);
      ctx.fillStyle = relation === "self" ? "#ffffff" : "#dce7ff";
      ctx.fillText(label, x0 + 4, Math.max(9, y0 - 3));
    }
    ctx.restore();
    hitRegions.push({ x0, y0, x1, y1, row: { ...r, kind } });
  }
  cameraCanvasHitRegions.set(canvas, hitRegions);
}

function candidateFromCameraRow(row){
  if (!row) return null;
  const merged = [...externalSelectionCandidates, ...(sceneSelectionCandidates || [])];
  const rowUuid = String(row.uuid || "");
  const rowPair = String(row.pair_uuid || "");
  const rowRun = evalRunNameFromMeta(row);
  let best = null;
  for (const cand of merged) {
    if (!cand) continue;
    const candRun = candidateRunName(cand);
    if (rowRun && candRun && rowRun !== candRun) continue;
    if (rowUuid && String(cand.trackId || "") === rowUuid) return cand;
    if (!best && rowPair && String(cand.pairId || "") === rowPair) best = cand;
  }
  return best;
}

function clearCameraViewportCanvases(){
  const c = document.getElementById("overlayCanvasSingle");
  if (c) {
    c.width = 1;
    c.height = 1;
    const ctx = c.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, 1, 1);
    cameraCanvasHitRegions.set(c, []);
  }
  const grid = document.getElementById("cameraGridWrap");
  if (grid) {
    grid.querySelectorAll("canvas").forEach((cv) => {
      cv.width = 1;
      cv.height = 1;
      cameraCanvasHitRegions.set(cv, []);
    });
  }
}

function wireCameraOverlayPicking(){
  function handleCanvasPick(canvas, ev){
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return;
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const hits = Array.isArray(cameraCanvasHitRegions.get(canvas)) ? cameraCanvasHitRegions.get(canvas) : [];
    const hit = hits.find((r) => x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1);
    if (!hit) return;
    const cand = candidateFromCameraRow(hit.row);
    if (cand) setSelectedInspectCandidate(cand, { focus: true });
  }
  const single = document.getElementById("overlayCanvasSingle");
  if (single) {
    single.addEventListener("pointerup", (ev) => handleCanvasPick(single, ev));
  }
  const grid = document.getElementById("cameraGridWrap");
  if (grid) {
    grid.addEventListener("pointerup", (ev) => {
      const canvas = ev.target && ev.target.closest ? ev.target.closest("canvas") : null;
      if (canvas) handleCanvasPick(canvas, ev);
    });
  }
}

/**
 * Paint one camera payload. Does not resize/clear the canvas until the image has decoded,
 * so the previous frame stays visible while scrubbing (avoids dark flash).
 */
function drawPayloadOnCanvas(canvas, payload, labelEl, gen){
  return new Promise((resolve) => {
    if (!canvas) {
      resolve();
      return;
    }
    if (gen != null && gen !== cameraOverlayGeneration) {
      resolve();
      return;
    }
    const iw = Math.max(1, Number(payload.width || 1));
    const ih = Math.max(1, Number(payload.height || 1));
    const wrap = canvas.parentElement;
    const maxW = (wrap && wrap.clientWidth) ? wrap.clientWidth : 640;
    const scale = Math.min(1, Math.max(0.05, maxW / iw));
    const cw = Math.max(1, Math.floor(iw * scale));
    const ch = Math.max(1, Math.floor(ih * scale));
    const b64 = payload.image_base64;
    if (!b64) {
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#0a0e18";
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = "#5a6a8a";
      ctx.font = "12px sans-serif";
      ctx.fillText("No image", 8, 22);
      cameraCanvasHitRegions.set(canvas, []);
      if (labelEl) labelEl.textContent = "";
      resolve();
      return;
    }
    const fmt = String(payload.image_format || "jpeg").toLowerCase();
    const mime = fmt === "png" ? "image/png" : "image/jpeg";
    const img = new Image();
    img.onload = () => {
      if (gen != null && gen !== cameraOverlayGeneration) {
        resolve();
        return;
      }
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, cw, ch);
      cameraCanvasHitRegions.set(canvas, []);
      const ann = document.getElementById("showOverlayAnnotations");
      if (ann && ann.checked) {
        const ann3d = document.getElementById("show3dAnnotations");
        const wantScene3d = !ann3d || !!ann3d.checked;
        strokeBoxes2dOnCanvas(
          ctx,
          wantScene3d && Array.isArray(payload.boxes_2d) ? payload.boxes_2d : [],
          iw,
          ih,
          cw,
          ch,
          "rgba(255, 94, 200, 0.95)",
          2,
        );
        drawCameraOverlayEvalRows(
          ctx,
          canvas,
          Array.isArray(payload.boxes_2d_eval_gt) ? payload.boxes_2d_eval_gt : [],
          "GT",
          iw,
          ih,
          cw,
          ch,
        );
        drawCameraOverlayEvalRows(
          ctx,
          canvas,
          Array.isArray(payload.boxes_2d_pred) ? payload.boxes_2d_pred : [],
          "EST",
          iw,
          ih,
          cw,
          ch,
        );
      } else {
        cameraCanvasHitRegions.set(canvas, []);
      }
      if (labelEl) labelEl.textContent = shortCamLabel(payload.camera || "");
      resolve();
    };
    img.onerror = () => {
      if (gen != null && gen !== cameraOverlayGeneration) {
        resolve();
        return;
      }
      canvas.width = cw;
      canvas.height = ch;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#0a0e18";
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = "#8a5a7a";
      ctx.font = "12px sans-serif";
      ctx.fillText("Load error", 8, 22);
      cameraCanvasHitRegions.set(canvas, []);
      resolve();
    };
    img.src = `data:${mime};base64,${b64}`;
  });
}

async function renderCameraViewport(payload, gen){
  if (!payload) return;
  if (gen != null && gen !== cameraOverlayGeneration) return;
  applyCameraViewportModeFromUi();
  const mode = document.getElementById("cameraLayoutMode") && document.getElementById("cameraLayoutMode").value;
  const canvasSingle = document.getElementById("overlayCanvasSingle");
  const gridWrap = document.getElementById("cameraGridWrap");
  const rows = Array.isArray(payload.cameras_payload) ? payload.cameras_payload : [];
  if (mode === "grid6") {
    ensureCameraGridSlots();
    const slots = buildCameraSlots(rows);
    const slotEls = gridWrap ? gridWrap.querySelectorAll(".cam-slot") : [];
    for (let i = 0; i < 6; i++) {
      if (gen != null && gen !== cameraOverlayGeneration) return;
      const p = slots[i];
      const slot = slotEls[i];
      if (!slot) continue;
      const c = slot.querySelector("canvas");
      const lbl = slot.querySelector(".cam-slot-label");
      if (!p || !p.image_base64) {
        if (c) {
          c.width = 160;
          c.height = 90;
          const ctx = c.getContext("2d");
          ctx.fillStyle = "#121620";
          ctx.fillRect(0, 0, c.width, c.height);
          ctx.fillStyle = "#5a6a8a";
          ctx.font = "11px sans-serif";
          ctx.fillText("—", 72, 52);
          cameraCanvasHitRegions.set(c, []);
        }
        if (lbl) lbl.textContent = "";
        continue;
      }
      await drawPayloadOnCanvas(c, p, lbl, gen);
    }
  } else {
    await drawPayloadOnCanvas(canvasSingle, payload, null, gen);
  }
}

async function refreshCameraOverlay(){
  if (!cameraViewportEnabled) return;
  const gen = ++cameraOverlayGeneration;
  try {
    const payload = await fetchCameraOverlay(frame);
    if (gen !== cameraOverlayGeneration) return;
    lastCameraPayload = payload;
    if (payload) {
      ensureOverlayCameraOptions(payload);
      const wrap = document.getElementById("cameraViewportWrap");
      if (wrap && !wrap.dataset.userSize) autoSizeCameraViewport(payload);
      await renderCameraViewport(payload, gen);
    }
  } catch (e) {
    if (gen === cameraOverlayGeneration) setStatus(`camera: ${e.message}`);
  }
}

function resizeCameraViewport(){
  if (!cameraViewportEnabled || !lastCameraPayload) return;
  const g = cameraOverlayGeneration;
  renderCameraViewport(lastCameraPayload, g).catch(() => {});
}

const CAM_VP_SIZE_STORAGE_KEY = "t4viewer.cameraViewportSize";
const INSPECT_PANEL_SIZE_STORAGE_KEY = "t4viewer.inspectPanelSize";

function wireCameraViewportResize(){
  const wrap = document.getElementById("cameraViewportWrap");
  const handle = document.getElementById("cameraViewportResize");
  if (!wrap || !handle) return;
  try {
    const raw = localStorage.getItem(CAM_VP_SIZE_STORAGE_KEY);
    if (raw) {
      const o = JSON.parse(raw);
      const w = Number(o.w);
      const h = Number(o.h != null ? o.h : o.maxH);
      if (Number.isFinite(w) && Number.isFinite(h)) {
        wrap.style.width = `${Math.round(w)}px`;
        wrap.style.height = `${Math.round(h)}px`;
        wrap.dataset.userSize = "1";
      }
    }
  } catch (_) {}

  let active = false;
  let startX = 0;
  let startY = 0;
  let startW = 0;
  let startH = 0;

  handle.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    ev.stopPropagation();
    active = true;
    startX = ev.clientX;
    startY = ev.clientY;
    const r = wrap.getBoundingClientRect();
    startW = r.width;
    startH = r.height;
    try {
      handle.setPointerCapture(ev.pointerId);
    } catch (_) {}
  });

  handle.addEventListener("pointermove", (ev) => {
    if (!active) return;
    const dx = ev.clientX - startX;
    const dy = ev.clientY - startY;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    let nw = Math.round(startW + dx);
    let nh = Math.round(startH + dy);
    nw = Math.max(260, Math.min(nw, vw - 8));
    nh = Math.max(140, Math.min(nh, vh - 48));
    wrap.style.width = `${nw}px`;
    wrap.style.height = `${nh}px`;
    wrap.dataset.userSize = "1";
    resizeCameraViewport();
  });

  function endResize(ev){
    if (!active) return;
    active = false;
    try {
      if (ev && ev.pointerId != null) handle.releasePointerCapture(ev.pointerId);
    } catch (_) {}
    const r = wrap.getBoundingClientRect();
    try {
      localStorage.setItem(CAM_VP_SIZE_STORAGE_KEY, JSON.stringify({ w: r.width, h: r.height }));
    } catch (_) {}
  }

  handle.addEventListener("pointerup", endResize);
  handle.addEventListener("pointercancel", endResize);

  handle.addEventListener("dblclick", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    wrap.style.width = "";
    wrap.style.height = "";
    delete wrap.dataset.userSize;
    try {
      localStorage.removeItem(CAM_VP_SIZE_STORAGE_KEY);
    } catch (_) {}
    resizeCameraViewport();
  });
}

function wireInspectPanelResize(){
  const wrap = document.getElementById("inspectPanel");
  const handle = document.getElementById("inspectPanelResize");
  if (!wrap || !handle) return;
  try {
    const raw = localStorage.getItem(INSPECT_PANEL_SIZE_STORAGE_KEY);
    if (raw) {
      const o = JSON.parse(raw);
      const w = Number(o.w);
      const h = Number(o.h);
      if (Number.isFinite(w) && Number.isFinite(h)) {
        wrap.style.width = `${Math.round(w)}px`;
        wrap.style.height = `${Math.round(h)}px`;
        wrap.dataset.userSize = "1";
      }
    }
  } catch (_) {}

  let active = false;
  let startX = 0;
  let startY = 0;
  let startW = 0;
  let startH = 0;

  handle.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    ev.preventDefault();
    ev.stopPropagation();
    active = true;
    startX = ev.clientX;
    startY = ev.clientY;
    const r = wrap.getBoundingClientRect();
    startW = r.width;
    startH = r.height;
    try {
      handle.setPointerCapture(ev.pointerId);
    } catch (_) {}
  });

  handle.addEventListener("pointermove", (ev) => {
    if (!active) return;
    const dx = ev.clientX - startX;
    const dy = ev.clientY - startY;
    let nw = Math.round(startW + dx);
    let nh = Math.round(startH + dy);
    nw = Math.max(460, Math.min(nw, window.innerWidth - 12));
    nh = Math.max(340, Math.min(nh, window.innerHeight - 20));
    wrap.style.width = `${nw}px`;
    wrap.style.height = `${nh}px`;
    wrap.dataset.userSize = "1";
    clampPanelToParent(wrap);
  });

  function endResize(ev){
    if (!active) return;
    active = false;
    try {
      if (ev && ev.pointerId != null) handle.releasePointerCapture(ev.pointerId);
    } catch (_) {}
    const r = wrap.getBoundingClientRect();
    try {
      localStorage.setItem(INSPECT_PANEL_SIZE_STORAGE_KEY, JSON.stringify({ w: r.width, h: r.height }));
    } catch (_) {}
  }

  handle.addEventListener("pointerup", endResize);
  handle.addEventListener("pointercancel", endResize);

  handle.addEventListener("dblclick", (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    wrap.style.width = "";
    wrap.style.height = "";
    delete wrap.dataset.userSize;
    try {
      localStorage.removeItem(INSPECT_PANEL_SIZE_STORAGE_KEY);
    } catch (_) {}
    clampPanelToParent(wrap);
  });
}

function setFrameData(data){
  sceneSelectionCandidates = buildSceneCandidatesFromFrameData(data, frame);
  const n = data.pointCount;
  const pointStep = n > MAX_RENDER_POINTS ? Math.ceil(n / MAX_RENDER_POINTS) : 1;
  const renderPointCount = Math.ceil(n / pointStep);
  const pos = new Float32Array(renderPointCount * 3);
  const col = new Float32Array(renderPointCount * 3);
  const mapName = getPointColormapName();
  const normalize = getPointIntensityNormalize();
  let minI = Infinity;
  let maxI = -Infinity;
  if (normalize && n > 0) {
    for (let i = 0; i < n; i++) {
      const s = data.pts[i * 4 + 3] * intensityGain;
      if (s < minI) minI = s;
      if (s > maxI) maxI = s;
    }
  }
  const rng = maxI - minI;
  const eps = 1e-9;
  let outPointIndex = 0;
  for (let i = 0; i < n; i += pointStep) {
    const x = data.pts[i * 4];
    const y = data.pts[i * 4 + 1];
    const z = data.pts[i * 4 + 2];
    const inten = data.pts[i * 4 + 3];
    const outBase = outPointIndex * 3;
    pos[outBase] = x;
    pos[outBase + 1] = y;
    pos[outBase + 2] = z;
    const scaled = inten * intensityGain;
    let tNorm;
    if (normalize && n > 0) {
      if (rng <= eps) tNorm = 0.5;
      else tNorm = (scaled - minI) / rng;
    } else {
      tNorm = Math.max(0, Math.min(1, scaled));
    }
    const rgb = intensityToRgb(tNorm, mapName);
    col[outBase] = rgb[0];
    col[outBase + 1] = rgb[1];
    col[outBase + 2] = rgb[2];
    outPointIndex++;
  }
  pointsGeom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  pointsGeom.setAttribute("color", new THREE.BufferAttribute(col, 3));
  pointsGeom.computeBoundingSphere();
  clearGroupDeep(boxesGroup);
  const batchSceneBoxes = data.boxCount > SCENE_BOX_BATCH_THRESHOLD;
  const batchedLinePts = batchSceneBoxes ? [] : null;
  for (let i=0;i<data.boxCount;i++) {
    const candidate = sceneSelectionCandidates[i] || null;
    if (data.format === "T4V3D002") {
      const base = i * 24;
      const linePts = boxEdgeSegmentsFromCorners(data.boxes.subarray(base, base + 24));
      if (batchSceneBoxes) {
        batchedLinePts.push(...linePts);
        continue;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(linePts, 3));
      const m = new THREE.LineBasicMaterial({
        color:TH.hex("boxAnnotation"), transparent:true,
        opacity:Number(document.getElementById("boxOpacity").value || "0.9")
      });
      const lines = new THREE.LineSegments(g, m);
      setInspectUserData(lines, candidate);
      boxesGroup.add(lines);
    } else {
      const cx=data.boxes[i*7], cy=data.boxes[i*7+1], cz=data.boxes[i*7+2];
      const sx=data.boxes[i*7+3], sy=data.boxes[i*7+4], sz=data.boxes[i*7+5], yaw=data.boxes[i*7+6];
      if (batchSceneBoxes) {
        batchedLinePts.push(...boxEdgeSegmentsFromCorners(boxCornersFromPose(cx, cy, cz, sx, sy, sz, yaw)));
        continue;
      }
      const g = new THREE.BoxGeometry(sx, sy, sz);
      const m = new THREE.MeshBasicMaterial({color:TH.hex("boxAnnotation"), wireframe:true, transparent:true, opacity:Number(document.getElementById("boxOpacity").value || "0.9")});
      const box = new THREE.Mesh(g, m);
      box.position.set(cx, cy, cz);
      box.rotation.set(0, 0, yaw);
      setInspectUserData(box, candidate);
      boxesGroup.add(box);
    }
  }
  if (batchSceneBoxes && batchedLinePts && batchedLinePts.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(batchedLinePts, 3));
    const m = new THREE.LineBasicMaterial({
      color: TH.hex("boxAnnotation"),
      transparent: true,
      opacity: Number(document.getElementById("boxOpacity").value || "0.9"),
    });
    const lines = new THREE.LineSegments(g, m);
    lines.userData.inspectSelectable = false;
    boxesGroup.add(lines);
  }
  const ann3d = document.getElementById("show3dAnnotations");
  boxesGroup.visible = !ann3d || !!ann3d.checked;
  const pointMeta = pointStep > 1 ? `${renderPointCount}/${data.pointCount}` : `${data.pointCount}`;
  const boxMeta = batchSceneBoxes ? `${data.boxCount} batched` : `${data.boxCount}`;
  metaEl.innerHTML = `<div>sample=<code>${data.sampleToken}</code></div><div>timestamp_us=${data.timestampUs}</div><div>points=${pointMeta} boxes=${boxMeta}</div>`;
}

let showFrameGeneration = 0;
async function showFrame(i){
  // Generation token: while a fetch is in flight the user may scrub further;
  // a stale response must never overwrite the scene of a newer frame.
  const gen = ++showFrameGeneration;
  frame = Math.max(0, Math.min(totalFrames-1, i));
  setFrameText();
  setStatus(`loading frame ${frame} ...`);
  const data = await fetchFrame(frame);
  if (gen !== showFrameGeneration) return;
  setFrameData(data);
  updateColorbar();
  if (followEgo) {
    controls.target.set(0, 0, 0);
    controls.update();
  }
  if (laneletEnabled) {
    await loadLanelet();
    if (gen !== showFrameGeneration) return;
    laneletGroup.visible = true;
  }
  prefetchAround(frame);
  if (bboxLayersByFrame) {
    applyEvalLayersForFrame(frame);
  } else {
    updateEvalHud();
    setStatus(`ready · cached=${cache.size}`);
  }
  updateSpotlightHud();
  updateSelectedCandidateForFrame();
  if (cameraViewportEnabled) {
    await refreshCameraOverlay();
    if (gen !== showFrameGeneration) return;
  }
  updateMetricsPlayhead();
  updateMetricsRatesPlayhead();
  updateMetricsErrorPlayhead();
}

document.getElementById("playBtn").addEventListener("click", () => {
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "Pause" : "Play";
});
document.getElementById("speed").addEventListener("change", (e) => {
  fps = 6 * Number(e.target.value || "1");
});
slider.addEventListener("input", () => {
  showFrame(Number(slider.value)).catch((e) => setStatus(`error: ${e.message}`));
});
document.getElementById("camReset").addEventListener("click", () => {
  followEgo = false;
  setSelectedInspectCandidate(null);
  camera.position.set(-12,-8,4.5);
  controls.target.set(0,0,0);
  controls.update();
});
document.getElementById("camTop").addEventListener("click", () => {
  followEgo = false;
  camera.position.set(0,0,60); controls.target.set(0,0,0); controls.update();
});
document.getElementById("camFollow").addEventListener("click", () => {
  followEgo = true;
  camera.position.set(-10,-2,3.8); controls.target.set(0,0,0); controls.update();
  setStatus("follow ego: on");
});
document.getElementById("inspectFocusBtn")?.addEventListener("click", () => {
  if (selectedInspectState && selectedInspectState.current) focusCameraOnCandidate(selectedInspectState.current, false);
});
document.getElementById("inspectLockBtn")?.addEventListener("click", () => {
  inspectLockFocus = !inspectLockFocus;
  const btn = document.getElementById("inspectLockBtn");
  if (btn) {
    btn.textContent = inspectLockFocus ? "Lock focus: on" : "Lock focus: off";
    btn.setAttribute("aria-pressed", inspectLockFocus ? "true" : "false");
  }
});
document.getElementById("inspectClearBtn")?.addEventListener("click", () => {
  setSelectedInspectCandidate(null);
});
document.getElementById("spotlightPrevBtn")?.addEventListener("click", () => {
  if (!spotlightFrames.length) return;
  const base = spotlightIndex >= 0 ? spotlightIndex : 0;
  jumpToSpotlightIndex(base - 1).catch((e) => setStatus(`spotlight: ${e.message}`));
});
document.getElementById("spotlightNextBtn")?.addEventListener("click", () => {
  if (!spotlightFrames.length) return;
  const base = spotlightIndex >= 0 ? spotlightIndex : -1;
  jumpToSpotlightIndex(base + 1).catch((e) => setStatus(`spotlight: ${e.message}`));
});
document.getElementById("spotlightPlayBtn")?.addEventListener("click", () => {
  if (!spotlightFrames.length) return;
  spotlightTouring = !spotlightTouring;
  spotlightLastSwitchTs = 0;
  updateSpotlightHud();
});
document.getElementById("toggleLanelet").addEventListener("click", async () => {
  laneletEnabled = !laneletEnabled;
  const btn = document.getElementById("toggleLanelet");
  btn.textContent = laneletEnabled ? "Lanelet ON" : "Lanelet OFF";
  if (laneletEnabled && !laneletLoaded) {
    try {
      await loadLanelet();
    } catch (err) {
      setStatus(`lanelet error: ${err.message}`);
      laneletEnabled = false;
      btn.textContent = "Lanelet OFF";
    }
  }
  if (laneletEnabled && laneletLoaded && laneletFrame !== frame) {
    try { await loadLanelet(); } catch (_) {}
  }
  laneletGroup.visible = laneletEnabled;
});
document.getElementById("show3dAnnotations").addEventListener("change", (e) => {
  boxesGroup.visible = !!e.target.checked;
  refreshCameraOverlay().catch(() => {});
});
["uiShowHud", "uiShowInspector", "uiShowMetricsCounts", "uiShowMetricsRates", "uiShowMetricsError", "uiShowCameraViewport"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => applyPanelVisibility());
});
{
  const hudFab = document.getElementById("hudRevealBtn");
  if (hudFab) {
    hudFab.addEventListener("click", () => {
      const c = document.getElementById("uiShowHud");
      if (c) {
        c.checked = true;
        applyPanelVisibility();
      }
    });
  }
}
document.getElementById("pointSize").addEventListener("input", (e) => {
  pointsMat.size = Number(e.target.value || "0.08");
});
document.getElementById("pointOpacity").addEventListener("input", (e) => {
  applyPointOpacity(Number(e.target.value || "1.0"));
});
document.getElementById("intensityGain").addEventListener("input", async (e) => {
  intensityGain = Number(e.target.value || "1.0");
  const data = cache.get(frame);
  if (data) setFrameData(data);
  updateColorbar();
});
document.getElementById("pointColormap").addEventListener("change", () => {
  pointColormapChosen = true;   // stop tracking the theme default from here on
  const data = cache.get(frame);
  if (data) setFrameData(data);
  updateColorbar();
});
document.getElementById("pointIntensityNormalize").addEventListener("change", () => {
  const data = cache.get(frame);
  if (data) setFrameData(data);
  updateColorbar();
});
document.getElementById("showColorbar").addEventListener("change", () => {
  updateColorbar();
});

/** Draw a vertical colorbar showing the current intensity map. */
function updateColorbar(){
  const wrap = document.getElementById("colorbarWrap");
  const canvas = document.getElementById("colorbarCanvas");
  if (!wrap || !canvas) return;
  const show = document.getElementById("showColorbar") && document.getElementById("showColorbar").checked;
  if (!show) { wrap.classList.remove("visible"); return; }
  const mapName = getPointColormapName();
  const W = 28, H = 140;
  canvas.width = W; canvas.height = H;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  const data = cache.get(frame);
  const normalize = getPointIntensityNormalize();
  let minI = 0, maxI = 1, rng = 1;
  if (data && normalize) {
    const n = data.pointCount;
    minI = Infinity; maxI = -Infinity;
    for (let i = 0; i < n; i++) {
      const s = data.pts[i * 4 + 3] * intensityGain;
      if (s < minI) minI = s;
      if (s > maxI) maxI = s;
    }
    rng = maxI - minI || 1e-9;
  }
  const img = ctx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    const t = 1 - y / (H - 1);
    const rgb = intensityToRgb(t, mapName);
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      img.data[idx] = Math.round(rgb[0] * 255);
      img.data[idx + 1] = Math.round(rgb[1] * 255);
      img.data[idx + 2] = Math.round(rgb[2] * 255);
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  // Labels
  ctx.fillStyle = TH.css("colorbarText"); ctx.font = "9px JetBrains Mono, monospace";
  ctx.textAlign = "right";
  const labelMin = minI === Infinity ? "0.00" : minI.toFixed(2);
  const labelMax = maxI === -Infinity ? "1.00" : maxI.toFixed(2);
  ctx.fillText(labelMax, W - 3, 10);
  ctx.fillText(labelMin, W - 3, H - 2);
  wrap.classList.add("visible");
}

/** GT meshes + nested edge outlines; EST line segments — scale by userData.opacityFactor. */
function applyOpacityToEvalLayerGroup(grp, v){
  grp.traverse((o) => {
    if (!o.material) return;
    if (o.userData && o.userData.keepOpacity) return;
    const f = o.userData.opacityFactor != null ? o.userData.opacityFactor : 1;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      if (!m) continue;
      m.transparent = true;
      m.opacity = v * f;
    }
  });
}
document.getElementById("boxOpacity").addEventListener("input", () => {
  const v = Number(document.getElementById("boxOpacity").value || "0.9");
  boxesGroup.traverse((child) => {
    if (child.material) child.material.opacity = v;
  });
  applyOpacityToEvalLayerGroup(gtLayerGroup, v);
  applyOpacityToEvalLayerGroup(predLayerGroup, v);
});
document.getElementById("egoOpacity")?.addEventListener("input", () => {
  applyEgoOpacity(egoOpacityValue());
});
["showGtTp", "showGtFn", "showEstTp", "showEstFp", "showPolygons", "showLabels"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => {
    renderExternalLayers();
    refreshCameraOverlay().catch(() => {});
  });
});
["fxEnableEmphasis", "fxFnRing", "fxFpPulse", "fxTpQuality", "highlightSevereOnly"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => {
    renderExternalLayers();
    renderInspectSelection();
    if (cameraViewportEnabled && lastCameraPayload) renderCameraViewport(lastCameraPayload, cameraOverlayGeneration).catch(() => {});
  });
});
document.getElementById("advShowVelocityVectors")?.addEventListener("change", () => {
  const data = cache.get(frame);
  if (data) setFrameData(data);
  renderExternalLayers();
});
document.getElementById("cameraSelectionOnly")?.addEventListener("change", () => {
  if (cameraViewportEnabled && lastCameraPayload) renderCameraViewport(lastCameraPayload, cameraOverlayGeneration).catch(() => {});
});
document.getElementById("spotlightMode")?.addEventListener("change", () => {
  syncSpotlightFrames();
  rebuildMetricsErrorPlot();
});
document.getElementById("projMode").addEventListener("click", () => {
  const wasPersp = camera === perspCamera;
  const next = wasPersp ? orthoCamera : perspCamera;
  next.position.copy(camera.position);
  next.up.set(0, 0, 1);
  if (wasPersp) {
    const fr = 22;
    orthoCamera.left = -fr; orthoCamera.right = fr; orthoCamera.top = fr; orthoCamera.bottom = -fr;
    orthoCamera.updateProjectionMatrix();
    document.getElementById("projMode").textContent = "orthographic";
  } else {
    perspCamera.fov = 74;
    perspCamera.updateProjectionMatrix();
    document.getElementById("projMode").textContent = "perspective";
  }
  controls.object = next;
  camera = next;
  controls.update();
});
window.addEventListener("message", (ev) => {
  // Only trust windows we have a relationship with: the embedding parent
  // (cross-origin dashboards included), the opener, or this window itself.
  // Arbitrary pages holding a reference to this window are ignored.
  if (!ev.source || (ev.source !== window.parent && ev.source !== window && ev.source !== window.opener)) {
    dbg("message dropped: untrusted source", { origin: ev.origin || "null" });
    return;
  }
  const d = ev && ev.data ? ev.data : null;
  if (!d || typeof d !== "object") return;
  dbg("message received", { type: d.type, origin: ev.origin || "null" });
  if (d.type === "bbox_layers_binary_v1") {
    try {
      const decoded = decodeBboxLayersBinaryV1(d.buffer);
      applyViewerSessionPayload(decoded, { refreshCamera: false });
      const n = decoded.bbox_layers_by_frame && typeof decoded.bbox_layers_by_frame === "object"
        ? Object.keys(decoded.bbox_layers_by_frame).length
        : 0;
      dbg("bbox_layers_binary_v1 decoded", { frames: n });
      if (cameraViewportEnabled) refreshCameraOverlay().catch(() => {});
    } catch (err) {
      dbg("bbox_layers_binary_v1 decode failed", err);
      setStatus(`bbox binary decode failed: ${err && err.message ? err.message : String(err)}`);
    }
  } else if (d.type === "bbox_layers") {
    bboxLayersByFrame = null;
    compareRunOrder = Array.isArray(d.compare_runs) ? d.compare_runs.map((v) => String(v || "").trim()).filter(Boolean) : [];
    setExternalLayerPayload(d);
  } else if (d.type === "bbox_layers_by_frame") {
    const frames = d.frames;
    if (frames && typeof frames === "object"){
      bboxLayersByFrame = frames;
      compareRunOrder = Array.isArray(d.compare_runs) ? d.compare_runs.map((v) => String(v || "").trim()).filter(Boolean) : [];
      const n = Object.keys(frames).length;
      if (viewerDebugEnabled) {
        const ack = `/viewer/three/debug/message-received?t4dataset_id=${encodeURIComponent(dataset)}&scenario_name=${encodeURIComponent(scenario)}&frame_index=${frame}&gt_count=0&pred_count=0&matched_count=0&message_type=bbox_layers_by_frame&frames_count=${n}`;
        fetch(ack)
          .then((r) => r.ok ? r.json() : Promise.reject(new Error(String(r.status))))
          .then((body) => dbg("bbox_layers_by_frame ack ok", body))
          .catch((err) => dbg("bbox_layers_by_frame ack failed", err));
      }
      applyEvalLayersForFrame(frame);
      rebuildAllMetricsPlots();
      if (cameraViewportEnabled) refreshCameraOverlay().catch(() => {});
    }
  } else if (d.type === "eval_metrics_series") {
    metricsSeriesOverride = normalizeMetricsSeriesPayload(d);
    rebuildAllMetricsPlots();
  } else if (d.type === "bbox_layers_clear") {
    clearCurrentViewerLayers();
  }
});
window.T4ViewerAPI = {
  setLayers(payload){ bboxLayersByFrame = null; setExternalLayerPayload(payload || {}); },
  clearLayers(){ clearCurrentViewerLayers({ clearMetrics: false }); },
  debugState(){
    const firstGt = Array.isArray(externalLayers.gt) && externalLayers.gt.length ? externalLayers.gt[0] : null;
    const firstPred = Array.isArray(externalLayers.pred) && externalLayers.pred.length ? externalLayers.pred[0] : null;
    const summarize = (b) => b ? ({
      uuid: b.uuid,
      status: b.status,
      force_wireframe: b.force_wireframe === true,
      length: b.length,
      width: b.width,
      height: b.height,
      size: b.size,
      yaw: b.yaw,
      meta_length: b.meta && b.meta.length,
      meta_width: b.meta && b.meta.width,
      meta_yaw: b.meta && b.meta.yaw,
      corners0: Array.isArray(b.corners) ? b.corners.slice(0, 6) : null,
    }) : null;
    return {
      frame,
      params: {
        external_bbox_yaw_offset: params.get("external_bbox_yaw_offset"),
        external_bbox_swap_lw: params.get("external_bbox_swap_lw"),
        external_bbox_alignment_version: params.get("external_bbox_alignment_version"),
      },
      bboxLayersByFrameCount: bboxLayersByFrame && typeof bboxLayersByFrame === "object" ? Object.keys(bboxLayersByFrame).length : 0,
      externalGtCount: Array.isArray(externalLayers.gt) ? externalLayers.gt.length : 0,
      externalPredCount: Array.isArray(externalLayers.pred) ? externalLayers.pred.length : 0,
      gtLayerChildren: gtLayerGroup.children.length,
      predLayerChildren: predLayerGroup.children.length,
      firstGt: summarize(firstGt),
      firstPred: summarize(firstPred),
    };
  },
  setMetricsSeries(obj){
    if (!obj || typeof obj !== "object") return;
    metricsSeriesOverride = normalizeMetricsSeriesPayload(obj);
    rebuildAllMetricsPlots();
  },
  clearMetricsSeries(){ metricsSeriesOverride = null; rebuildAllMetricsPlots(); },
  getFrameContext(){ return { dataset, scenario, frame, totalFrames }; },
};
document.getElementById("importSessionBtn").addEventListener("click", () => {
  if (viewerSessionFileInput) viewerSessionFileInput.click();
});
document.getElementById("exportSessionBtn").addEventListener("click", () => {
  downloadViewerSessionJson();
});
if (viewerSessionFileInput) {
  viewerSessionFileInput.addEventListener("change", async (ev) => {
    const input = ev.currentTarget;
    const file = input && input.files && input.files[0] ? input.files[0] : null;
    if (!file) return;
    try {
      const raw = JSON.parse(await file.text());
      const normalized = normalizeViewerSessionPayload(raw, file.name);
      applyViewerSessionPayload(normalized.payload);
      activeViewerSessionId = "";
      activeViewerSessionSource = normalized.source_name || file.name;
      updateViewerSessionUrl("");
      setShareSessionState(`imported ${String(file.name).slice(0, 18)}`);
      setStatus(`imported ${file.name}`);
    } catch (err) {
      setStatus(`import error: ${err && err.message ? err.message : String(err)}`);
    } finally {
      input.value = "";
    }
  });
}
document.getElementById("shareSessionBtn").addEventListener("click", async () => {
  const payload = buildCurrentViewerSessionPayload();
  if (!payload) {
    setStatus("share: no imported or external overlay data to save");
    return;
  }
  try {
    setStatus("saving viewer session...");
    const res = await fetch("/viewer/three/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        payload,
        source_name: activeViewerSessionSource || null,
        t4dataset_id: dataset,
        scenario_name: scenario,
        frame_index: frame,
        version,
      }),
    });
    const body = await parseJsonResponseOrThrow(res, `viewer session HTTP ${res.status}`);
    activeViewerSessionId = String(body && body.session_id ? body.session_id : "");
    if (!activeViewerSessionId) throw new Error("Server did not return a session id.");
    updateViewerSessionUrl(activeViewerSessionId);
    const shareUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
    setShareSessionState(`session ${activeViewerSessionId.slice(0, 8)}`);
    revealShareSessionUrl(shareUrl);
    let copied = false;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(shareUrl);
        copied = true;
      } catch (_) {}
    }
    if (!copied && shareSessionUrlEl) {
      try {
        shareSessionUrlEl.focus();
        shareSessionUrlEl.select();
        copied = document.execCommand && document.execCommand("copy");
      } catch (_) {}
    }
    setStatus(copied ? `share link copied · ${activeViewerSessionId.slice(0, 8)}` : `share link ready in URL field · ${activeViewerSessionId.slice(0, 8)}`);
  } catch (err) {
    setStatus(`share error: ${err && err.message ? err.message : String(err)}`);
  }
});
document.getElementById("toggleFullscreen").addEventListener("click", async () => {
  const fsEl = appRootEl || document.documentElement;
  try {
    if (!document.fullscreenElement) {
      await fsEl.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch (err) {
    setStatus(`fullscreen: ${err && err.message ? err.message : String(err)}`);
  }
});
document.addEventListener("fullscreenchange", () => {
  syncFullscreenButton();
  resetAllPanelPositions();
  resizeMainRenderer();
  resizeMetricsRenderer();
});
document.getElementById("framePrev").addEventListener("click", () => {
  showFrame(frame - 1).catch((e) => setStatus(`error: ${e.message}`));
});
document.getElementById("frameNext").addEventListener("click", () => {
  showFrame(frame + 1).catch((e) => setStatus(`error: ${e.message}`));
});
document.getElementById("toggleCameraViewport")?.addEventListener("click", () => {
  const c = document.getElementById("uiShowCameraViewport");
  if (c) c.checked = !c.checked;
  applyPanelVisibility();
});
document.getElementById("cameraViewportHideBtn")?.addEventListener("click", () => {
  const c = document.getElementById("uiShowCameraViewport");
  if (c) c.checked = false;
  applyPanelVisibility();
});
document.getElementById("cameraLayoutMode")?.addEventListener("change", () => {
  resetCameraViewportUserSize();
  updateCameraToolbarMode();
  if (lastCameraPayload) autoSizeCameraViewport(lastCameraPayload);
  refreshCameraOverlay().catch((e) => setStatus(`camera: ${e.message}`));
});
document.getElementById("overlayCamera")?.addEventListener("change", () => {
  refreshCameraOverlay().catch((e) => setStatus(`camera: ${e.message}`));
});
document.getElementById("showOverlayAnnotations")?.addEventListener("change", () => {
  refreshCameraOverlay().catch((e) => setStatus(`camera: ${e.message}`));
});
document.getElementById("cameraOverlayMaxRange")?.addEventListener("change", () => {
  refreshCameraOverlay().catch((e) => setStatus(`camera: ${e.message}`));
});
document.getElementById("cameraOverlayMaxBoxes")?.addEventListener("change", () => {
  refreshCameraOverlay().catch((e) => setStatus(`camera: ${e.message}`));
});
document.getElementById("compareViewMode")?.addEventListener("change", () => {
  const mode = compareModeValue();
  const runs = currentEvalRuns();
  setCompareLabels(runs[0] || "", runs[1] || "", runs.length >= 2 ? mode : "overlay");
  rebuildMetricsPlot();
  updateMetricsComparePanel();
});
function updateCompareCurtainFromClientX(clientX){
  const wrap = document.getElementById("canvasWrap");
  if (!wrap) return;
  const rect = wrap.getBoundingClientRect();
  if (!(rect.width > 0)) return;
  compareCurtainRatio = Math.max(0.06, Math.min(0.94, (Number(clientX) - rect.left) / rect.width));
  const handle = document.getElementById("compareCurtainHandle");
  if (handle) handle.style.left = `${Math.round(compareCurtainRatio * 10000) / 100}%`;
}
document.getElementById("compareCurtainHandle")?.addEventListener("pointerdown", (ev) => {
  if (compareModeValue() !== "curtain") return;
  compareCurtainDragging = true;
  ev.currentTarget.setPointerCapture?.(ev.pointerId);
  updateCompareCurtainFromClientX(ev.clientX);
  ev.preventDefault();
});
window.addEventListener("pointermove", (ev) => {
  if (!compareCurtainDragging) return;
  updateCompareCurtainFromClientX(ev.clientX);
});
window.addEventListener("pointerup", () => {
  compareCurtainDragging = false;
});
document.getElementById("canvasWrap")?.addEventListener("dblclick", (ev) => {
  if (compareModeValue() !== "curtain") return;
  updateCompareCurtainFromClientX(ev.clientX);
});
window.addEventListener("resize", () => {
  resizeMainRenderer();
  resizeMetricsRenderer();
  resizeCameraViewport();
  clampAllMovablePanels();
});
window.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && selectedInspectState) {
    setSelectedInspectCandidate(null);
  } else if ((ev.key === "f" || ev.key === "F") && selectedInspectState && selectedInspectState.current) {
    focusCameraOnCandidate(selectedInspectState.current, false);
  }
});

function animate(ts){
  requestAnimationFrame(animate);
  if (playing && totalFrames > 0 && ts - lastFrameTs > (1000 / Math.max(1, fps))) {
    lastFrameTs = ts;
    const nxt = frame + 1 >= totalFrames ? 0 : frame + 1;
    showFrame(nxt).catch((err) => setStatus(`error: ${err.message}`));
  }
  if (spotlightTouring && spotlightFrames.length > 0 && !spotlightJumpInFlight) {
    if (!spotlightLastSwitchTs || ts - spotlightLastSwitchTs > 1800) {
      spotlightLastSwitchTs = ts;
      const base = spotlightIndex >= 0 ? spotlightIndex : -1;
      jumpToSpotlightIndex(base + 1).catch((err) => setStatus(`spotlight: ${err.message}`));
    }
  }
  controls.update();
  animatePulseMarkers(gtLayerGroup, ts);
  animatePulseMarkers(predLayerGroup, ts);
  inspectOverlayGroup.traverse((obj) => {
    if (!obj || !obj.userData) return;
    const u = 0.5 + 0.5 * Math.sin((ts || 0) * 0.006);
    if (obj.userData.focusPulse) {
      const s = 0.92 + 0.24 * u;
      obj.scale.setScalar(s);
      if (obj.material) obj.material.opacity = 0.5 + 0.34 * u;
    } else if (obj.userData.focusHalo) {
      const s = 1 + 0.06 * u;
      obj.scale.setScalar(s);
      if (obj.material) obj.material.opacity = 0.22 + 0.18 * u;
    }
  });
  renderMainScene();
  const cntOn = document.getElementById("uiShowMetricsCounts") && document.getElementById("uiShowMetricsCounts").checked;
  const rateOn = document.getElementById("uiShowMetricsRates") && document.getElementById("uiShowMetricsRates").checked;
  const errOn = document.getElementById("uiShowMetricsError") && document.getElementById("uiShowMetricsError").checked;
  if (cntOn && metricsWrapEl && !metricsWrapEl.hidden && metricsChartGroup.children.length) {
    renderChartToCanvas(metricsScene, metricsCamera, metricsCanvas);
  }
  if (rateOn && metricsRatesWrapEl && !metricsRatesWrapEl.hidden && metricsRatesChartGroup.children.length) {
    renderChartToCanvas(metricsRatesScene, metricsRatesCamera, metricsRatesCanvas);
  }
  if (errOn && metricsErrorWrapEl && !metricsErrorWrapEl.hidden && metricsErrorChartGroup.children.length) {
    renderChartToCanvas(metricsErrorScene, metricsErrorCamera, metricsErrorCanvas);
  }
}

(async function boot(){
  setStatus("bootstrapping...");
  wireCameraViewportResize();
  wireInspectPanelResize();
  wireCameraOverlayPicking();
  wireInspectPicking();
  const egoMeshPromise = loadEgoVehicleMesh().catch((err) => {
    console.warn("[viewer] ego vehicle mesh unavailable, using box fallback", err);
  });
  const metaRes = await fetch(metaUrl);
  if (!metaRes.ok) throw new Error(`meta HTTP ${metaRes.status}`);
  const meta = await metaRes.json();
  totalFrames = Number(meta.total_frames || 0);
  slider.max = String(Math.max(0, totalFrames - 1));
  camera = perspCamera;
  document.getElementById("projMode").textContent = "perspective";
  camera.position.set(-10, -2, 3.8);
  controls.target.set(0,0,0);
  controls.update();
  // Defer first size sync to next frame so layout is stable.
  requestAnimationFrame(() => resizeMainRenderer());
  setTimeout(() => resizeMainRenderer(), 0);
  await showFrame(Math.min(startFrame, Math.max(0, totalFrames - 1)));
  if (initialSessionId) {
    const record = await loadViewerSessionById(initialSessionId);
    const src = record && record.source_name ? ` · ${record.source_name}` : "";
    setStatus(`loaded shared session ${initialSessionId.slice(0, 8)}${src}`);
  }
  updateCameraToolbarMode();
  applyPanelVisibility();
  const hudEl = document.getElementById("hud");
  const hudHandle = document.getElementById("hudDragHandle");
  if (hudEl && hudHandle) makePanelDraggable(hudEl, hudHandle);
  const inspectPanelEl = document.getElementById("inspectPanel");
  const inspectHandle = document.getElementById("inspectDragHandle");
  if (inspectPanelEl && inspectHandle) makePanelDraggable(inspectPanelEl, inspectHandle);
  if (metricsWrapEl) {
    const h = metricsWrapEl.querySelector(".panel-drag-handle");
    if (h) makePanelDraggable(metricsWrapEl, h);
  }
  if (metricsCompareWrapEl) {
    const h = metricsCompareWrapEl.querySelector(".panel-drag-handle");
    if (h) makePanelDraggable(metricsCompareWrapEl, h);
  }
  if (metricsRatesWrapEl) {
    const h = metricsRatesWrapEl.querySelector(".panel-drag-handle");
    if (h) makePanelDraggable(metricsRatesWrapEl, h);
  }
  if (metricsErrorWrapEl) {
    const h = metricsErrorWrapEl.querySelector(".panel-drag-handle");
    if (h) makePanelDraggable(metricsErrorWrapEl, h);
  }
  const camVp = document.getElementById("cameraViewportWrap");
  const camHandle = document.getElementById("cameraViewportDragHandle");
  if (camVp && camHandle) {
    makePanelDraggable(camVp, camHandle);
  }
  egoMeshPromise.catch(() => {});
  animate(0);
})().catch((err) => {
  setStatus(`boot error: ${err.message}`);
});

// Top level on purpose: the theme toggle must work even when boot fails (missing
// dataset, fetch error), so it cannot live inside the boot IIFE. The chrome is
// CSS-driven; the WebGL metrics charts bake colors into materials and need a rebuild.
if (window.TH) {
  syncDefaultColormap();
  TH.bindToggle(document.getElementById("themeToggleBtn"));
  TH.onChange(() => {
    try {
      applySceneTheme();
    } catch (err) {
      /* scene not booted yet (no dataset loaded) */
    }
    try {
      rebuildMetricsPlot();
      rebuildMetricsRatesPlot();
      rebuildMetricsErrorPlot();
    } catch (err) {
      /* charts not built yet */
    }
  });
}
