/** Decoder for the T4BBOX1 packed eval-layers postMessage payload (bbox_layers_binary_v1). */
const BBOX_BINARY_OPTIONAL_NUMERIC_FIELDS = [
  "vx","vy","confidence","pointcloud_num","x_error","y_error","z_error","yaw_error","vx_error","vy_error",
  "speed_error","center_distance","plane_distance","pair_dt_sec","dx_min","dy_min","unix_time","frame_index"
];
const BBOX_BINARY_TEXT_FIELDS = [
  "uuid","label","status","frame_id","shape_type","visibility","pair_uuid","topic_name","t4dataset_id",
  "suite_name","t4dataset_name","scenario_name","run","source"
];

export function decodeBboxLayersBinaryV1(buffer){
  const raw = buffer instanceof ArrayBuffer ? buffer : (buffer && buffer.buffer instanceof ArrayBuffer ? buffer.buffer : null);
  if (!raw) throw new Error("bbox binary payload missing ArrayBuffer");
  const dv = new DataView(raw);
  let off = 0;
  const need = (n) => {
    if (off + n > dv.byteLength) throw new Error("bbox binary payload truncated");
  };
  const u8 = () => { need(1); const v = dv.getUint8(off); off += 1; return v; };
  const u32 = () => { need(4); const v = dv.getUint32(off, true); off += 4; return v; };
  const i32 = () => { need(4); const v = dv.getInt32(off, true); off += 4; return v; };
  const f32 = () => { need(4); const v = dv.getFloat32(off, true); off += 4; return v; };
  need(8);
  const magic = String.fromCharCode(...new Uint8Array(raw, off, 8));
  off += 8;
  if (magic !== "T4BBOX1\0") throw new Error(`unknown bbox binary magic ${JSON.stringify(magic)}`);
  const frameCount = u32();
  const stringCount = u32();
  const boxCount = u32();
  const pairCount = u32();
  const runCount = u32();
  const strings = [];
  const decoder = new TextDecoder();
  for (let i = 0; i < stringCount; i++) {
    const n = u32();
    need(n);
    strings.push(n ? decoder.decode(new Uint8Array(raw, off, n)) : "");
    off += n;
  }
  const str = (id) => strings[id] || "";
  const compare_runs = [];
  for (let i = 0; i < runCount; i++) {
    const value = str(u32());
    if (value) compare_runs.push(value);
  }
  const frameRows = [];
  for (let i = 0; i < frameCount; i++) {
    frameRows.push({
      frame_index: i32(),
      gt_start: u32(),
      gt_count: u32(),
      pred_start: u32(),
      pred_count: u32(),
      pair_start: u32(),
      pair_count: u32(),
    });
  }
  const boxes = [];
  for (let i = 0; i < boxCount; i++) {
    const box = {
      x: f32(),
      y: f32(),
      z: f32(),
      width: f32(),
      length: f32(),
      height: f32(),
      yaw: f32(),
    };
    for (const field of BBOX_BINARY_OPTIONAL_NUMERIC_FIELDS) {
      const value = f32();
      if (Number.isFinite(value)) box[field] = value;
    }
    const corners = [];
    let hasCorners = true;
    for (let j = 0; j < 24; j++) {
      const value = f32();
      if (!Number.isFinite(value)) hasCorners = false;
      corners.push(value);
    }
    if (hasCorners) box.corners = corners;
    for (const field of BBOX_BINARY_TEXT_FIELDS) {
      const value = str(u32());
      if (value) box[field] = value;
    }
    if (u8() === 1) box.force_wireframe = true;
    boxes.push(box);
  }
  const pairs = [];
  for (let i = 0; i < pairCount; i++) {
    pairs.push({ gt_idx: u32(), pred_idx: u32(), pair_uuid: str(u32()) });
  }
  if (off < dv.byteLength) {
    for (const box of boxes) {
      if (off >= dv.byteLength) break;
      const pointCount = u32();
      if (pointCount > 0) {
        const footprint = [];
        for (let j = 0; j < pointCount; j++) {
          footprint.push([f32(), f32(), f32()]);
        }
        if (footprint.length >= 3) box.footprint = footprint;
      }
    }
  }
  const bbox_layers_by_frame = {};
  for (const row of frameRows) {
    bbox_layers_by_frame[String(row.frame_index)] = {
      gt: boxes.slice(row.gt_start, row.gt_start + row.gt_count),
      pred: boxes.slice(row.pred_start, row.pred_start + row.pred_count),
      matched_pairs: pairs.slice(row.pair_start, row.pair_start + row.pair_count),
    };
  }
  const out = { bbox_layers_by_frame };
  if (compare_runs.length >= 2) out.compare_runs = compare_runs;
  return out;
}
