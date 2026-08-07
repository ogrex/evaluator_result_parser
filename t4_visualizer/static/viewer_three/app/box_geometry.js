/** Pure cuboid geometry helpers shared by every wireframe-box builder. */
import * as THREE from "three";

export function boxCenterFromCornersArray(arr){
  if (!Array.isArray(arr) || arr.length < 24) return null;
  let sx = 0, sy = 0, sz = 0;
  for (let i = 0; i < 8; i++) {
    sx += Number(arr[i * 3] || 0);
    sy += Number(arr[i * 3 + 1] || 0);
    sz += Number(arr[i * 3 + 2] || 0);
  }
  return [sx / 8, sy / 8, sz / 8];
}

export function boxSizeFromCornersArray(arr){
  if (!Array.isArray(arr) || arr.length < 24) return null;
  const p0 = new THREE.Vector3(arr[0], arr[1], arr[2]);
  const p1 = new THREE.Vector3(arr[3], arr[4], arr[5]);
  const p3 = new THREE.Vector3(arr[9], arr[10], arr[11]);
  const p4 = new THREE.Vector3(arr[12], arr[13], arr[14]);
  return [p0.distanceTo(p4), p0.distanceTo(p3), p0.distanceTo(p1)];
}

export function boxCornersFromPose(cx, cy, cz, l, w, h, yaw){
  const hl = Math.max(0.01, l) * 0.5;
  const hw = Math.max(0.01, w) * 0.5;
  const hh = Math.max(0.01, h) * 0.5;
  const c = Math.cos(yaw || 0);
  const s = Math.sin(yaw || 0);
  const body = [
    [ hl,  hw,  hh], [ hl, -hw,  hh], [ hl, -hw, -hh], [ hl,  hw, -hh],
    [-hl,  hw,  hh], [-hl, -hw,  hh], [-hl, -hw, -hh], [-hl,  hw, -hh],
  ];
  const out = [];
  for (const [bx, by, bz] of body) {
    out.push(cx + bx * c - by * s, cy + bx * s + by * c, cz + bz);
  }
  return out;
}

/** The 12 edges of a cuboid, as index pairs into an 8-corner list (two quads + verticals).
 *  Single source of truth for every wireframe-box builder in this file. */
export const BOX_EDGE_PAIRS = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

/** corners: flat [x0,y0,z0,...] of 8 corners (Array or Float32Array view) → flat line-segment vertices. */
export function boxEdgeSegmentsFromCorners(corners){
  if (!corners || corners.length < 24) return [];
  const out = [];
  for (const [a, b] of BOX_EDGE_PAIRS) {
    const a0 = a * 3;
    const b0 = b * 3;
    out.push(
      corners[a0], corners[a0 + 1], corners[a0 + 2],
      corners[b0], corners[b0 + 1], corners[b0 + 2]
    );
  }
  return out;
}
