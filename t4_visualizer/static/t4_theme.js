/**
 * Shared light/dark theme for the T4 viewer templates.
 *
 * The chrome (top bar, HUD panel, metrics panels, tooltips) is driven by the CSS
 * tokens under :root[data-theme="..."]. The 3D scene is WebGL and cannot read CSS,
 * so its palette is mirrored here.
 *
 *   TH.hex(name)  -> 0xRRGGBB for three.js materials
 *   TH.css(name)  -> CSS color string for canvas-2D label textures
 *   TH.num(name)  -> tuning scalar (e.g. point-cloud darkening in light mode)
 *
 * Point-cloud intensity colormaps were designed against a near-black ground: on
 * paper their bright end washes out and grayscale disappears entirely, so light
 * mode darkens every mapped color and inverts grayscale.
 */
(function () {
  const STORAGE_KEY = "t4Theme";
  const THEMES = ["dark", "light"];

  const PALETTE = {
    dark: {
      // metrics chart furniture
      chartPanel: 0x06091c,
      chartPanelAlt: 0x090b1f,
      chartGrid: 0x2a3f60,
      chartGridAlt: 0x31425f,
      playhead: 0xffffff,
      playheadGlow: 0x6ea8ff,
      playheadGlowRates: 0xffcc66,
      playheadGlowError: 0xffbe78,
      // series, grouped as the charts use them
      gtTp: 0x00cc66,
      gtFn: 0xff9933,
      estTp: 0x66b3ff,
      estFp: 0xff6666,
      cmpATp: 0x42d6ff,
      cmpAFp: 0xff7d7d,
      cmpBTp: 0xb7e46d,
      cmpBFp: 0xffbd68,
      rate1: 0x5ee4a8,
      rate2: 0x66b3ff,
      rate3: 0xffcc66,
      rate4: 0xff7799,
      err1: 0x6fe2ff,
      err2: 0x8fb1ff,
      err3: 0xffbe78,
      err4: 0xff739d,

      // --- 3D scene ---
      sceneBg: 0x000000,
      fog: 0x070b16,
      gridMajor: 0x2b3a6f,
      gridMinor: 0x1a2342,
      ambient: 0x8aa6ff,
      egoWire: 0x6ea8ff,
      // eval box status colors (also shown by the HUD layer swatches)
      boxGtTp: 0x00cc66,
      boxGtFn: 0xff9933,
      boxEstTp: 0x66b3ff,
      boxEstFp: 0xff6666,
      boxGtFallback: 0x4bd08d,
      boxAnnotation: 0xffd166,
      // emphasis effects
      haloSev: 0x8de4ff,
      pulse: 0x9ce6ff,
      ring: 0xffe1ae,
      bracket: 0xffffff,
      inspectFp: 0xff8f74,
      inspectFn: 0xffcf82,
      inspectTp: 0x97e6ff,
      trailLine: 0x86d4ff,
      trailDot: 0x7cb3ff,
      trailDotLast: 0xffffff,
      // lanelet roles
      laneLeft: 0x4bd08d,
      laneRight: 0xff8f5a,
      laneCenter: 0x7aa7ff,
      laneUnknown: 0xd0d7ff,
      // canvas-2D textures / overlays
      labelChipBg: "rgba(6,10,20,0.82)",
      labelChipText: "#f4f8ff",
      colorbarText: "#94a3c8",
      // point-cloud colormap tuning
      pointScale: 1,
      grayscaleInvert: 0
    },
    light: {
      // Paper panels, ink grid; series darkened to hold contrast on white.
      chartPanel: 0xffffff,
      chartPanelAlt: 0xfaf9f5,
      chartGrid: 0xd8d3c6,
      chartGridAlt: 0xd8d3c6,
      playhead: 0x1b1916,
      playheadGlow: 0xd97757,
      playheadGlowRates: 0x96700a,
      playheadGlowError: 0xa8720c,
      gtTp: 0x177a4a,
      gtFn: 0xc26a10,
      estTp: 0x2868ad,
      estFp: 0xc0392b,
      cmpATp: 0x1f6f86,
      cmpAFp: 0xb8453a,
      cmpBTp: 0x5c7a1f,
      cmpBFp: 0xa8720c,
      rate1: 0x2c7d59,
      rate2: 0x2868ad,
      rate3: 0x96700a,
      rate4: 0xb03a58,
      err1: 0x1f6f86,
      err2: 0x3a5f9e,
      err3: 0xa8720c,
      err4: 0xb03a58,

      // --- 3D scene ---
      // Slightly deeper than the paper panels so those still read as raised.
      sceneBg: 0xf0eee6,
      fog: 0xf0eee6,
      gridMajor: 0xc9c3b5,
      gridMinor: 0xdcd7ca,
      ambient: 0xfff4e8,
      egoWire: 0x2f5f8f,
      // Match the CSS --s-* series tokens so a swatch, a chart line and a box in
      // the scene all read as the same category.
      boxGtTp: 0x177a4a,
      boxGtFn: 0xc26a10,
      boxEstTp: 0x2868ad,
      boxEstFp: 0xc0392b,
      boxGtFallback: 0x2f6647,
      boxAnnotation: 0xa8720c,
      haloSev: 0x1f6f86,
      pulse: 0x1f6f86,
      ring: 0xa8791a,
      bracket: 0x1b1916,
      inspectFp: 0xb8453a,
      inspectFn: 0xa8720c,
      inspectTp: 0x2868ad,
      trailLine: 0x2868ad,
      trailDot: 0x3a6ea8,
      trailDotLast: 0x1b1916,
      laneLeft: 0x2f6647,
      laneRight: 0xb8543a,
      laneCenter: 0x3a6ea8,
      laneUnknown: 0x6b6862,
      labelChipBg: "rgba(255,255,255,0.92)",
      labelChipText: "#1b1916",
      colorbarText: "#6b6862",
      pointScale: 0.72,
      grayscaleInvert: 1
    }
  };

  function normalize(name) {
    return THEMES.indexOf(name) >= 0 ? name : null;
  }

  function stored() {
    try {
      return normalize(window.localStorage.getItem(STORAGE_KEY));
    } catch (err) {
      return null;
    }
  }

  function fromQuery() {
    try {
      return normalize(new URLSearchParams(window.location.search).get("theme"));
    } catch (err) {
      return null;
    }
  }

  function fromSystem() {
    try {
      return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    } catch (err) {
      return "dark";
    }
  }

  // Query param wins (parent page -> embedded viewer), then the saved choice, then the OS.
  let current = fromQuery() || stored() || fromSystem();
  let explicit = Boolean(fromQuery() || stored());
  const listeners = [];

  const TH = {
    THEMES: THEMES,
    get current() {
      return current;
    },
    /** 0xRRGGBB for three.js materials. */
    hex: function (name) {
      const p = PALETTE[current] || PALETTE.dark;
      return name in p ? p[name] : 0xffffff;
    },
    /** CSS color string, for colors baked into canvas-2D textures. */
    css: function (name) {
      const p = PALETTE[current] || PALETTE.dark;
      return name in p ? p[name] : "#000";
    },
    /** Tuning scalar. */
    num: function (name, fallback) {
      const p = PALETTE[current] || PALETTE.dark;
      return name in p ? Number(p[name]) : Number(fallback || 0);
    },
    isLight: function () {
      return current === "light";
    },
    set: function (name, opts) {
      const next = normalize(name);
      if (!next || next === current) {
        if (next) apply();
        return current;
      }
      current = next;
      if (!opts || opts.persist !== false) {
        explicit = true;
        try {
          window.localStorage.setItem(STORAGE_KEY, next);
        } catch (err) {
          /* private mode: session-only */
        }
      }
      apply();
      return current;
    },
    toggle: function () {
      return TH.set(current === "dark" ? "light" : "dark");
    },
    /** Register a redraw callback; WebGL charts must be rebuilt on change. */
    onChange: function (fn) {
      if (typeof fn === "function") listeners.push(fn);
    },
    /** Wire a button as a theme toggle and keep its glyph in sync. */
    bindToggle: function (el) {
      if (!el) return;
      const sync = function () {
        const light = current === "light";
        el.textContent = light ? "☾" : "☀";
        el.title = light ? "Switch to dark theme" : "Switch to light theme";
        el.setAttribute("aria-label", el.title);
        el.setAttribute("aria-pressed", String(light));
      };
      el.addEventListener("click", function () {
        TH.toggle();
      });
      TH.onChange(sync);
      sync();
    }
  };

  function apply() {
    const root = document.documentElement;
    root.setAttribute("data-theme", current);
    root.style.colorScheme = current;
    for (let i = 0; i < listeners.length; i++) {
      try {
        listeners[i](current);
      } catch (err) {
        console.error("theme listener failed", err);
      }
    }
  }

  // The viewer is commonly embedded in an iframe; follow the parent's theme.
  window.addEventListener("message", function (ev) {
    const data = ev && ev.data;
    if (data && data.type === "t4-theme" && normalize(data.theme)) {
      TH.set(data.theme, {persist: false});
    }
  });

  try {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", function (ev) {
      if (!explicit) TH.set(ev.matches ? "light" : "dark", {persist: false});
    });
  } catch (err) {
    /* older browsers: no live OS updates */
  }

  window.T4Theme = TH;
  window.TH = TH;
  apply();
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", apply, {once: true});
  }
})();
