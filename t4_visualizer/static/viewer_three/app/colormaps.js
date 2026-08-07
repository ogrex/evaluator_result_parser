/** Point-intensity colormaps. Reads the page theme via the global `TH` (t4_theme.js). */
function lerp(a, b, u){ return a + (b - a) * u; }

/** Piecewise linear colormap: stops sorted by t in [0,1], RGB in [0,1]. */
function lerpColorStops(t, stops){
  t = Math.max(0, Math.min(1, t));
  if (!stops || !stops.length) return [1, 1, 1];
  if (t <= stops[0].t) return [stops[0].r, stops[0].g, stops[0].b];
  const last = stops[stops.length - 1];
  if (t >= last.t) return [last.r, last.g, last.b];
  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (t >= a.t && t <= b.t) {
      const u = (t - a.t) / (b.t - a.t + 1e-20);
      return [lerp(a.r, b.r, u), lerp(a.g, b.g, u), lerp(a.b, b.b, u)];
    }
  }
  return [last.r, last.g, last.b];
}

const TURBO_STOPS = [
  { t: 0, r: 0.19, g: 0.07, b: 0.48 },
  { t: 0.25, r: 0.0, g: 0.53, b: 0.99 },
  { t: 0.5, r: 0.25, g: 0.99, b: 0.51 },
  { t: 0.75, r: 0.99, g: 0.9, b: 0.25 },
  { t: 1, r: 0.9, g: 0.15, b: 0.0 },
];
const VIRIDIS_STOPS = [
  { t: 0, r: 0.267004, g: 0.004874, b: 0.329415 },
  { t: 0.25, r: 0.282623, g: 0.140956, b: 0.457517 },
  { t: 0.5, r: 0.253935, g: 0.265254, b: 0.529983 },
  { t: 0.75, r: 0.206756, g: 0.371758, b: 0.553123 },
  { t: 1, r: 0.993248, g: 0.906157, b: 0.143936 },
];
/**
 * Single-hue blue ramp, light -> dark, for use over a light ground.
 *
 * Turbo and jet are rainbows and viridis/plasma run dark -> light, so on paper
 * their bright end lands at the ground's own luminance and those points vanish.
 * This ramp instead darkens monotonically with intensity, putting ink where the
 * signal is. It starts at blue step 300 rather than the lighter steps: a heatmap
 * cell may recede into its surface, but a 1px point that does is simply gone
 * (step 300 = 2.15:1 against the scene ground, the lightest step clearing 2:1).
 */
const PAPER_STOPS = [
  { t: 0,    r: 0.427451, g: 0.654902, b: 0.925490 },  /* #6da7ec  blue 300 */
  { t: 0.25, r: 0.223529, g: 0.529412, b: 0.898039 },  /* #3987e5  blue 400 */
  { t: 0.5,  r: 0.145098, g: 0.415686, b: 0.749020 },  /* #256abf  blue 500 */
  { t: 0.75, r: 0.094118, g: 0.309804, b: 0.584314 },  /* #184f95  blue 600 */
  { t: 1,    r: 0.050980, g: 0.211765, b: 0.419608 },  /* #0d366b  blue 700 */
];
const PLASMA_STOPS = [
  { t: 0, r: 0.050383, g: 0.029803, b: 0.527975 },
  { t: 0.25, r: 0.493733, g: 0.011988, b: 0.657865 },
  { t: 0.5, r: 0.798216, g: 0.280197, b: 0.469538 },
  { t: 0.75, r: 0.987053, g: 0.633744, b: 0.279089 },
  { t: 1, r: 0.940015, g: 0.975158, b: 0.131326 },
];

function colormapJet(t){
  t = Math.max(0, Math.min(1, t));
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
  return [r, g, b];
}

/** Previous viewer mapping (blue ↔ warm). */
function colormapClassic(t){
  const u = Math.max(0, Math.min(1, t));
  return [0.15 + 0.85 * u, 0.35 + 0.5 * (1 - u), 1.0 - 0.6 * u];
}

export function intensityToRgb(tNorm, mapName){
  const rgb = intensityToRgbRaw(tNorm, mapName);
  // "paper" is already built for a light ground; the others need toning down.
  const scale = mapName === "paper" ? 1 : TH.num("pointScale", 1);
  if (scale === 1) return rgb;
  // Light ground: darken so the bright end of every map stays visible.
  return [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale];
}

/** Raw colormap lookup, before any theme adjustment. */
export function intensityToRgbRaw(tNorm, mapName){
  switch (mapName) {
    case "viridis": return lerpColorStops(tNorm, VIRIDIS_STOPS);
    case "plasma": return lerpColorStops(tNorm, PLASMA_STOPS);
    case "jet": return colormapJet(tNorm);
    case "paper": return lerpColorStops(tNorm, PAPER_STOPS);
    case "grayscale": {
      let g = Math.max(0, Math.min(1, tNorm));
      // Light ground: bright grays vanish, so run the ramp dark-on-light.
      if (TH.num("grayscaleInvert", 0)) g = 1 - g;
      return [g, g, g];
    }
    case "classic": return colormapClassic(tNorm);
    case "turbo":
    default: return lerpColorStops(tNorm, TURBO_STOPS);
  }
}
