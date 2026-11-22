// Synthetic ground-truth generator for validating the keyer.
//
// Produces, for each scene, the exact linear-light layers:
//   fg      : straight foreground colour (linear RGB), valid everywhere
//   alpha   : coverage in [0,1] (anti-aliased / translucent edges included)
//   screen  : screen colour actually present behind the subject (linear RGB),
//             including light falloff, vignette and cast shadows
//   mixed   : composited input = alpha*fg + (1-alpha)*screen, then encoded to
//             8-bit sRGB and optionally JPEG-compressed
//
// The collider is a signed distance field blended with a per-pixel coverage
// ramp, so edge pixels really are mixtures of foreground and screen (no
// precomputed mask is used anywhere in the key path).

import { PNG } from 'pngjs';
import jpeg from 'jpeg-js';
import { linearToSrgb8, srgb8ToLinear, clamp01 } from '../src/color.js';

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const smooth = (e0, e1, x) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};

// --- geometry: signed distance (positive inside) --------------------------
function sdRoundRect(x, y, cx, cy, hw, hh, r) {
  const qx = Math.abs(x - cx) - (hw - r);
  const qy = Math.abs(y - cy) - (hh - r);
  const ax = Math.max(qx, 0), ay = Math.max(qy, 0);
  return -(Math.sqrt(ax * ax + ay * ay) + Math.min(Math.max(qx, qy), 0) - r);
}

function sdCircle(x, y, cx, cy, r) {
  return r - Math.hypot(x - cx, y - cy);
}

function sdCapsule(x, y, ax, ay, bx, by, r) {
  const pax = x - ax, pay = y - ay, bax = bx - ax, bay = by - ay;
  const t = clamp01((pax * bax + pay * bay) / (bax * bax + bay * bay));
  return r - Math.hypot(pax - bax * t, pay - bay * t);
}

// Positive-inside SDFs: union is smooth-max, subtraction is min(-shape).
function sunion(a, b, k) {
  const h = clamp01(0.5 + 0.5 * (a - b) / k);
  return b * (1 - h) + a * h + k * h * (1 - h);
}

// Definition of every scene: subject SDF + foreground colour function, plus
// screen setup.  The same definition provides ground truth and mixed image.
export const SCENES = {
  presenter: {
    width: 480, height: 360,
    screen: { kind: 'green', base: [0.045, 0.36, 0.075], falloff: 0.42, vignette: 0.30, seed: 11 },
    shadows: [{ ax: 120, ay: 300, bx: 360, by: 330, r: 60, strength: 0.45 }],
    spill: { amount: 0.22 },
    subject: {
      // head + body + arm + a genuinely green scarf and a green mug (both must
      // survive spill correction and must not be keyed out)
      sdf(x, y) {
        let d = sdCircle(x, y, 240, 92, 40);           // head
        d = sunion(d, sdRoundRect(x, y, 240, 250, 78, 120, 42), 18); // torso
        d = sunion(d, sdCapsule(x, y, 178, 200, 128, 300, 26), 14);  // left arm
        d = sunion(d, sdCapsule(x, y, 302, 210, 356, 260, 24), 12);  // right arm
        d = sunion(d, sdRoundRect(x, y, 372, 240, 26, 24, 8), 6);    // mug
        d = Math.min(d, -sdRoundRect(x, y, 240, 175, 26, 14, 8));    // collar notch (hole)
        return d;
      },
      // linear colour of the true subject
      color(x, y) {
        // skin: warm, not key-coloured
        if (y < 138) return [0.52, 0.30, 0.20];
        // scarf band: genuinely green object, but a distinctly different green
        // (teal-leaning) so it is recoverable without a blanket despill
        if (y > 150 && y < 186) return [0.06, 0.32, 0.21];
        // mug: bright lime-green, very different from the screen green
        if (x > 344 && y > 212) return [0.40, 0.46, 0.09];
        // shirt: neutral grey with texture
        const n = 0.5 + 0.06 * Math.sin(x * 0.21) * Math.cos(y * 0.17);
        return [n * 0.72, n * 0.74, n * 0.78];
      },
    },
    compression: { quality: 82 },
  },

  blueProduct: {
    width: 480, height: 360,
    screen: { kind: 'blue', base: [0.055, 0.10, 0.44], falloff: 0.34, vignette: 0.26, seed: 23 },
    shadows: [{ ax: 180, ay: 320, bx: 330, by: 335, r: 52, strength: 0.35 }],
    spill: { amount: 0.18 },
    // A saturated blue object under blue light: the hard case where a naive
    // "reduce the blue channel" spill fix destroys the product.
    subject: {
      sdf(x, y) {
        let d = sdRoundRect(x, y, 240, 250, 108, 86, 26);          // box
        d = sunion(d, sdCapsule(x, y, 240, 150, 240, 205, 20), 10); // neck
        d = sunion(d, sdRoundRect(x, y, 240, 120, 58, 38, 18), 10); // top
        return d;
      },
      color(x, y) {
        if (y < 168) return [0.10, 0.42, 0.62];  // cyan-blue lid (distinct from screen)
        if (x > 240) return [0.30, 0.12, 0.52];  // violet-blue panel
        return [0.34, 0.36, 0.40];                // neutral grey panel
      },
    },
    compression: { quality: 85 },
  },

  despill: {
    // A large NEUTRAL object under strong, fairly uniform green spill.  The
    // object is matte grey (skin/fabric-like) so its own colour has no green
    // dominance; any green left in the recovered foreground is spill.  A
    // saturated green object sits alongside so the despill can be checked for
    // NOT destroying legitimate hue.  This is the dedicated test for stage 5.
    width: 480, height: 360,
    screen: { kind: 'green', base: [0.05, 0.34, 0.075], falloff: 0.10, vignette: 0.10, seed: 51 },
    shadows: [],
    spill: { amount: 0.28 },
    subject: {
      sdf(x, y) {
        let d = sdRoundRect(x, y, 160, 190, 90, 130, 18);   // large neutral slab
        d = sunion(d, sdRoundRect(x, y, 340, 190, 70, 100, 16), 8); // green object
        return d;
      },
      color(x, y) {
        if (x > 268) return [0.10, 0.30, 0.22];  // genuinely green (teal) object
        // neutral slab with slight texture, no green dominance
        const n = 0.42 + 0.04 * Math.sin(x * 0.3) * Math.cos(y * 0.25);
        return [n, n * 1.02, n * 1.06];          // faintly cool grey
      },
    },
    compression: { quality: 90 },
  },

  fineDetail: {
    width: 480, height: 360,
    screen: { kind: 'green', base: [0.05, 0.30, 0.075], falloff: 0.38, vignette: 0.34, seed: 37 },
    shadows: [],
    // Thin struts, a ring, and fine parallel lines: sub-pixel coverage would be
    // lost by any binary matte.
    subject: {
      sdf(x, y) {
        // plate with a genuine screen hole (the ring)
        let d = sdRoundRect(x, y, 150, 260, 96, 96, 10);
        d = Math.min(d, -sdCircle(x, y, 150, 260, 44));
        // thin vertical struts
        for (let i = 0; i < 9; i++) {
          const xx = 276 + i * 20;
          d = sunion(d, sdCapsule(x, y, xx, 80, xx, 300, 1.6), 1.2);
        }
        // horizontal bracket
        d = sunion(d, sdCapsule(x, y, 268, 190, 448, 190, 2.2), 1.2);
        // small disc
        d = sunion(d, sdCircle(x, y, 470, 66, 6), 1.5);
        return d;
      },
      color(x, y) {
        if (x > 262) return [0.72, 0.70, 0.66];
        if (Math.hypot(x - 150, y - 260) < 52) return [0.20, 0.22, 0.26];
        return [0.55, 0.57, 0.60];
      },
    },
    compression: { quality: 90 },
  },
};

function screenAt(scene, x, y) {
  const s = scene.screen;
  const { width: W, height: H } = scene;
  let base = s.base;
  // multiplicative light falloff (darker towards bottom-right) + vignette,
  // the classic unevenly-lit cyclorama.  Both are pure scalings.
  const fx = x / W, fy = y / H;
  let scale = 1 - s.falloff * (0.55 * fx + 0.45 * fy);
  const vx = (fx - 0.5) * 2, vy = (fy - 0.5) * 2;
  scale *= 1 - s.vignette * (vx * vx * 0.6 + vy * vy * 0.6);
  // gentle low-frequency lamp ripple
  scale *= 1 + 0.035 * Math.sin(x * 0.017 + 0.7) * Math.cos(y * 0.013);
  let r = base[0] * scale, g = base[1] * scale, b = base[2] * scale;

  // soft cast shadows: also multiplicative on the screen
  for (const sh of scene.shadows || []) {
    const ax = sh.ax, ay = sh.ay, bx = sh.bx, by = sh.by;
    const pax = x - ax, pay = y - ay, bax = bx - ax, bay = by - ay;
    const t = clamp01((pax * bax + pay * bay) / (bax * bax + bay * bay));
    const dist = Math.hypot(pax - bax * t, pay - bay * t);
    const k = 1 - sh.strength * smooth(sh.r, sh.r * 0.3, dist);
    r *= k; g *= k; b *= k;
  }
  return [r, g, b];
}

// Build the full ground-truth set for a scene.  `params` lets the caller add
// noise / compression variants.
export function buildScene(name, opts = {}) {
  const scene = SCENES[name];
  if (!scene) throw new Error(`unknown scene ${name}`);
  const W = scene.width, H = scene.height, n = W * H;
  const rnd = mulberry32((scene.screen.seed || 1) + (opts.seedOffset || 0));
  const noise = opts.noise ?? 0.012;       // gaussian in linear light, multiplicative
  const edgePx = opts.edgePx ?? 1.4;        // anti-alias ramp in pixels

  const fgL = new Float32Array(n * 3);       // true foreground incl. spill
  const fgCleanL = new Float32Array(n * 3);  // true foreground WITHOUT spill (despill target)
  const alpha = new Float32Array(n);
  const screenL = new Float32Array(n * 3);
  const mixedL = new Float32Array(n * 3);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const d = scene.subject.sdf(x + 0.5, y + 0.5);
      // coverage from SDF: exact sub-pixel coverage, not a binary mask
      const a = smooth(-edgePx * 0.5, edgePx * 0.5, d);
      alpha[i] = a;
      const col = scene.subject.color(x + 0.5, y + 0.5);
      fgCleanL[i * 3] = col[0]; fgCleanL[i * 3 + 1] = col[1]; fgCleanL[i * 3 + 2] = col[2];
      const scr = screenAt(scene, x + 0.5, y + 0.5);
      // additive neutral ambient bounce onto the screen (real sets have it)
      const amb = 0.006;
      const sr = scr[0] + amb, sg = scr[1] + amb, sb = scr[2] + amb;
      screenL[i * 3] = sr; screenL[i * 3 + 1] = sg; screenL[i * 3 + 2] = sb;

      // Foreground light: key-coloured bounce light spills onto the subject.
      // This is *in the true foreground* (a real camera records it), so a
      // correct key must remove it from F without touching alpha and without
      // destroying a genuinely green/blue object's own hue.  Spill is added
      // along the screen's chroma direction (the screen colour minus its
      // neutral component), strongest near the silhouette boundary.
      const spillAmt = scene.spill ? scene.spill.amount : 0;
      let fr = col[0], fgc = col[1], fb = col[2];
      if (spillAmt > 0) {
        // a large soft source spills fairly uniformly, stronger at grazing
        // edges; keep a substantial base so interiors are genuinely tinted
        const edgeProx = 1 - smooth(-6, 60, d);   // 1 at boundary, ->0 deep inside
        const k = spillAmt * (0.55 + 0.45 * edgeProx);
        // screen chroma vector = screen colour projected off the neutral axis
        const sw = (sr + sg + sb) / 3;
        const cr = sr - sw, cg = sg - sw, cb = sb - sw;
        fr += k * cr; fgc += k * cg; fb += k * cb;
      }
      fgL[i * 3] = fr; fgL[i * 3 + 1] = fgc; fgL[i * 3 + 2] = fb;

      for (let c = 0; c < 3; c++) {
        const m = a * fgL[i * 3 + c] + (1 - a) * [sr, sg, sb][c];
        mixedL[i * 3 + c] = m;
      }
    }
  }

  // sensor/compression noise in linear light (multiplicative, so the darkest
  // screen pixels do not turn neutral)
  for (let i = 0; i < n * 3; i++) {
    const m = mixedL[i];
    mixedL[i] = Math.max(0, m * (1 + noise * gauss(rnd)));
  }

  const mixed8 = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    mixed8[i * 4] = linearToSrgb8(mixedL[i * 3]);
    mixed8[i * 4 + 1] = linearToSrgb8(mixedL[i * 3 + 1]);
    mixed8[i * 4 + 2] = linearToSrgb8(mixedL[i * 3 + 2]);
    mixed8[i * 4 + 3] = 255;
  }

  let input8 = mixed8;
  let jpegBuffer = null;
  if (opts.jpeg !== false && scene.compression) {
    const enc = jpeg.encode({ data: Buffer.from(mixed8), width: W, height: H }, scene.compression.quality);
    jpegBuffer = enc.data;
    const dec = jpeg.decode(jpegBuffer, { useTArray: true });
    input8 = new Uint8ClampedArray(dec.data);
  }

  return { name, width: W, height: H, fgL, fgCleanL, alpha, screenL, mixedL, mixed8, input8, jpegBuffer, scene };
}

function gauss(rnd) {
  let u = 0, v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Encode an 8-bit RGBA buffer to PNG.
export function encodePng(rgba, width, height) {
  const png = new PNG({ width, height });
  png.data = Buffer.from(rgba.buffer ? rgba.buffer : rgba, rgba.byteOffset || 0, rgba.byteLength);
  return PNG.sync.write(png);
}

export { linearToSrgb8, srgb8ToLinear };
