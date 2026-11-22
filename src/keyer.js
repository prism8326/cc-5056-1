// Chroma-key foreground kernel.
//
// COLOUR SPACE
// ------------
// Input is 8-bit sRGB, decoded once to linear light.  All mixing, unmixing,
// screen-model fitting and compositing happen in linear light; output is
// re-encoded to sRGB only at the boundaries.
//
// Chroma is decomposed on an orthonormal basis:
//   e0 = (1,1,1)/sqrt(3)   neutral / brightness axis
//   e1 = screen chroma direction, perpendicular to e0
//   e2 = e0 x e1           remaining (off-axis) chroma axis
// A screen colour B = w*e0 + p*e1 + q*e2.  Multiplicative shading (vignette,
// soft shadow, uneven lamps) scales (w,p) without rotating the vector, so the
// local screen model is fitted as a smooth brightness field.  A *foreground*
// object generally has off-axis chroma q != 0 and a different p/w ratio; that
// is what distinguishes a real green object from a darker green screen.
//
// PREMULTIPLIED RELATIONS
// -----------------------
//   C = alpha*F + (1-alpha)*B          (linear light)
// F is straight foreground colour, P = alpha*F is premultiplied foreground.
// We solve for alpha against a spatially coherent foreground model Fs:
//   alpha = clamp( dot(C-B, Fs-B) / |Fs-B|^2 )
// and recover F = (C-(1-alpha)*B)/alpha.  Compositing over a new background is
// P + (1-alpha)*Bg.  PNG export is straight-alpha (F, alpha); background is
// never baked in.
//
// PIPELINE (stages are separate; alpha and colour edits never share a knob)
//   1. fit local screen model B(x,y)                 (screen fields)
//   2. initial alpha from chroma distance            (no colour edits)
//   3. joint alpha / foreground refinement with a
//      colour-coherent foreground model              (alpha separation)
//   4. adjudication: exact-screen lock, manual mask  (no blind transparency)
//   5. spill correction on F only, protected by off-axis chroma  -> alpha fixed
//   6. diagnostics + failure feedback
//
// A pixel merely *near* the screen colour is not automatically transparent:
// alpha is decided by the mixing model.  Spill correction never just lowers
// the green/blue channel; it removes screen-axis chroma only where the colour
// carries no off-axis hue of its own, so genuine green/blue objects survive.

import {
  linearToSrgb8, srgb8ToLinear, clamp01, smoothstep,
  chromaBasis,
} from './color.js';

export const DEFAULT_PARAMS = {
  keyLinear: null,
  // initial chroma alpha
  alphaFloor: 0.04,
  // screen model
  screenSigma: 10,
  screenPrior: 0.02,
  screenQThresh: [0.06, 0.20],
  screenPMin: 0.004,
  screenRatioTol: 0.35,
  // alpha / foreground refinement
  refineStrength: 1.0,
  coreDistThreshold: 0.22,
  coreRadius: 44,
  mixMinFrac: 0.10,
  bilateralRadius: 7,
  bilateralSigmaColor: 0.09,
  bilateralSigmaSpace: 3.2,
  sourceAlphaMin: 0.22,
  coreBlend: 1.0,            // weight of the nearest-core F vs the bilateral F
  // adjudication
  strictScreenQ: 0.03,
  manualMask: null,          // Uint8Array, 0 none / 1 protect / 2 force-key
  manualPriority: 0.85,
  // spill
  spillStrength: 0.8,
  spillOffAxisLo: 0.06,
  spillOffAxisHi: 0.16,
  alphaExportFloor: 0.004,
  quality: 1,                // spatial subsample for the bilateral (1 = full)
};

function prepareParams(p = {}) {
  const q = { ...DEFAULT_PARAMS, ...p };
  if (!q.keyLinear || q.keyLinear.length !== 3) throw new Error('keyLinear is required');
  if (!(q.keyLinear[0] + q.keyLinear[1] + q.keyLinear[2] > 1e-6)) throw new Error('Key colour is black');
  q.screenSigma = Math.max(1, q.screenSigma | 0);
  q.bilateralRadius = Math.max(1, q.bilateralRadius | 0);
  q.quality = Math.min(4, Math.max(1, q.quality | 0));
  return q;
}

// --- separable Gaussian blur (used only for the smooth screen model) --------
function gaussKernel(sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  const inv = 1 / (2 * sigma * sigma);
  let sum = 0;
  for (let i = -r; i <= r; i++) { const v = Math.exp(-i * i * inv); k[i + r] = v; sum += v; }
  for (let i = 0; i < k.length; i++) k[i] /= sum;
  return { k, r };
}

function blurField(src, w, h, sigma, out) {
  const { k, r } = gaussKernel(sigma);
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let xx = x + i; if (xx < 0) xx = 0; else if (xx >= w) xx = w - 1;
        acc += src[row + xx] * k[i + r];
      }
      tmp[row + x] = acc;
    }
  }
  const dst = out || new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let i = -r; i <= r; i++) {
        let yy = y + i; if (yy < 0) yy = 0; else if (yy >= h) yy = h - 1;
        acc += tmp[yy * w + x] * k[i + r];
      }
      dst[y * w + x] = acc;
    }
  }
  return dst;
}

function normConv1(values, w, h, weights, sigma, prior, priorWeight) {
  const den = blurField(weights, w, h, sigma);
  const tmp = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) tmp[i] = values[i] * weights[i];
  const num = blurField(tmp, w, h, sigma);
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = (num[i] + prior * priorWeight) / (den[i] + priorWeight);
  return { field: out, support: den };
}

// Cross-bilateral foreground model.  Sources are pixels carrying a given
// weight (0..1) that says "this pixel is definitely foreground".  The source
// value is the *raw observed* colour C, which equals the true straight
// foreground wherever coverage is 1.  The colour term keeps the model
// coherent inside an object and sharp across object boundaries, so a green
// scarf keeps its own green instead of averaging into a grey shirt.  A screen
// pixel nearby still receives the foreground colour as its model; projection
// against that model then correctly yields alpha ~= 0.
function coherentForeground(C, srcW, w, h, p) {
  const n = w * h;
  const r = p.bilateralRadius;
  const step = p.quality;
  const sS = p.bilateralSigmaSpace;
  const sC = p.bilateralSigmaColor;
  const invS2 = 1 / (2 * sS * sS);
  const invC2 = 1 / (2 * sC * sC);
  const out = new Float32Array(n * 3);
  const support = new Float32Array(n);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const cr = C[i * 3], cg = C[i * 3 + 1], cb = C[i * 3 + 2];
      let wr = 0, wg = 0, wb = 0, ws = 0;
      const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
      const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
      for (let yy = y0; yy <= y1; yy += step) {
        const dy = yy - y; const row = yy * w;
        for (let xx = x0; xx <= x1; xx += step) {
          const j = row + xx;
          const sw = srcW[j];
          if (sw <= 0.01) continue;
          const dx = xx - x;
          const d2 = dx * dx + dy * dy;
          if (d2 > r * r) continue;
          const dc2 = (C[j * 3] - cr) ** 2 + (C[j * 3 + 1] - cg) ** 2 + (C[j * 3 + 2] - cb) ** 2;
          const wgt = sw * Math.exp(-d2 * invS2) * Math.exp(-dc2 * invC2);
          if (wgt < 1e-5) continue;
          wr += wgt * C[j * 3]; wg += wgt * C[j * 3 + 1]; wb += wgt * C[j * 3 + 2]; ws += wgt;
        }
      }
      if (ws > 1e-4) {
        out[i * 3] = wr / ws; out[i * 3 + 1] = wg / ws; out[i * 3 + 2] = wb / ws;
      } else {
        // No confident foreground nearby: signal *no model*, never fall back to
        // the pixel's own colour (that would make every screen pixel look like
        // its own foreground and project to alpha = 1).
        out[i * 3] = 0; out[i * 3 + 1] = 0; out[i * 3 + 2] = 0;
      }
      support[i] = ws;
    }
  }
  return { fg: out, support };
}

// Nearest confidently-opaque "core" pixel.  Core pixels sit well off the
// screen ray, so their observed colour equals the true straight foreground.
// For every pixel we copy the core colour from the nearest core pixel within
// a radius (spatial nearest, not an average), which is what keeps thin
// structures and sharp colour boundaries from being polluted by screen or by
// a neighbouring differently-coloured object.  Implemented as a bounded
// vector-propagation (jump-flood approximation).
function nearestCore(C, dInit, B, w, h, p) {
  const n = w * h;
  const fg = new Float32Array(n * 3);
  const valid = new Uint8Array(n);
  const srcX = new Int32Array(n).fill(-1);
  const srcY = new Int32Array(n).fill(-1);
  const thr = p.coreDistThreshold;
  // Erode by one pixel: a genuine edge pixel is a *mix* and therefore also has
  // high ray distance, so "far from ray" alone would let 1px anti-aliased edges
  // seed themselves (self-reference -> alpha pinned to 1).  A core pixel must
  // be far from the ray AND surrounded by pixels that are also far from it,
  // i.e. an interior pixel of a real object, not a boundary.
  const eroded = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let mn = dInit[y * w + x];
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= w) continue;
          const v = dInit[yy * w + xx];
          if (v < mn) mn = v;
        }
      }
      eroded[y * w + x] = mn;
    }
  }
  for (let i = 0; i < n; i++) {
    if (eroded[i] >= thr) { srcX[i] = i % w; srcY[i] = (i / w) | 0; valid[i] = 1; }
  }
  const maxR = p.coreRadius;
  const passes = [1, 2, 4, 8].filter((s) => s <= maxR);
  const dist2 = (x, y, i) => {
    if (srcX[i] < 0) return Infinity;
    const dx = srcX[i] - x, dy = srcY[i] - y;
    return dx * dx + dy * dy;
  };
  for (const step of passes) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let bx = srcX[i], by = srcY[i], bd = dist2(x, y, i);
        const cand = [
          [x - step, y], [x + step, y], [x, y - step], [x, y + step],
          [x - step, y - step], [x + step, y - step], [x - step, y + step], [x + step, y + step],
        ];
        for (const [cx, cy] of cand) {
          if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
          const j = cy * w + cx;
          if (srcX[j] < 0) continue;
          const dx = srcX[j] - x, dy = srcY[j] - y, d = dx * dx + dy * dy;
          if (d < bd) { bd = d; bx = srcX[j]; by = srcY[j]; }
        }
        srcX[i] = bx; srcY[i] = by;
      }
    }
    // backward pass improves propagation from the other direction
    for (let y = h - 1; y >= 0; y--) {
      for (let x = w - 1; x >= 0; x--) {
        const i = y * w + x;
        let bx = srcX[i], by = srcY[i], bd = dist2(x, y, i);
        const cand = [
          [x - step, y], [x + step, y], [x, y - step], [x, y + step],
          [x - step, y - step], [x + step, y - step], [x - step, y + step], [x + step, y + step],
        ];
        for (const [cx, cy] of cand) {
          if (cx < 0 || cy < 0 || cx >= w || cy >= h) continue;
          const j = cy * w + cx;
          if (srcX[j] < 0) continue;
          const dx = srcX[j] - x, dy = srcY[j] - y, d = dx * dx + dy * dy;
          if (d < bd) { bd = d; bx = srcX[j]; by = srcY[j]; }
        }
        srcX[i] = bx; srcY[i] = by;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    if (srcX[i] < 0) { valid[i] = 0; continue; }
    const x = i % w, y = (i / w) | 0;
    const ddx = srcX[i] - x, ddy = srcY[i] - y;
    if (ddx * ddx + ddy * ddy > maxR * maxR) { valid[i] = 0; continue; }
    const j = srcY[i] * w + srcX[i];
    fg[i * 3] = C[j * 3]; fg[i * 3 + 1] = C[j * 3 + 1]; fg[i * 3 + 2] = C[j * 3 + 2];
    valid[i] = 1;
  }
  return { fg, valid };
}

export function keyImage(input, width, height, userParams) {
  const p = prepareParams(userParams);
  const w = width, h = height, n = w * h;

  // ---- decode to linear ---------------------------------------------------
  let C;
  if (input.length === n * 3) {
    C = input instanceof Float32Array ? input : Float32Array.from(input);
  } else {
    C = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      C[i * 3] = srgb8ToLinear(input[i * 4]);
      C[i * 3 + 1] = srgb8ToLinear(input[i * 4 + 1]);
      C[i * 3 + 2] = srgb8ToLinear(input[i * 4 + 2]);
    }
  }

  const basis = chromaBasis(p.keyLinear);
  const { e0, e1, e2, pK } = basis;

  // pixel chroma coordinates
  const cw = new Float32Array(n), cp = new Float32Array(n), cq = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = C[i * 3], g = C[i * 3 + 1], b = C[i * 3 + 2];
    cw[i] = r * e0[0] + g * e0[1] + b * e0[2];
    cp[i] = r * e1[0] + g * e1[1] + b * e1[2];
    cq[i] = r * e2[0] + g * e2[1] + b * e2[2];
  }

  // ---- stage 1: local screen model ---------------------------------------
  // A pixel looks like screen only if BOTH hold:
  //   (a) its screen-axis chroma ratio p/w matches the screen's own ratio
  //       (multiplicative shading preserves p/w; additive neutral ambient
  //       shifts it only slightly), and
  //   (b) its off-axis chroma q/w is near zero.
  // Using p/w and q/w (not raw p, q) is what stops neutral grey objects -
  // which have p ~= q ~= 0 in *every* basis - from being mistaken for screen.
  const conf = new Float32Array(n);
  let confSum = 0, wpSum = 0, wwSum = 0, wqSum = 0;
  const pKRatio = pK;
  for (let i = 0; i < n; i++) {
    const wi = cw[i];
    const pn = cp[i] / (wi + 1e-6);
    const qn = cq[i] / (wi + 1e-6);
    const mag = smoothstep(p.screenPMin, p.screenPMin * 6, cp[i]);
    const ratioMatch = 1 - smoothstep(p.screenRatioTol * 0.5, p.screenRatioTol, Math.abs(pn - pKRatio));
    const onAxis = 1 - smoothstep(p.screenQThresh[0], p.screenQThresh[1], Math.abs(qn));
    const sign = cp[i] > 0 ? 1 : 0;
    conf[i] = mag * ratioMatch * onAxis * sign;
    confSum += conf[i]; wpSum += conf[i] * cp[i]; wwSum += conf[i] * cw[i]; wqSum += conf[i] * cq[i];
  }
  const coverage = confSum / n;
  const priorP = confSum > 1e-6 ? wpSum / confSum : pK * 0.5;
  const priorW = confSum > 1e-6 ? wwSum / confSum : 0.5;
  const priorQ = confSum > 1e-6 ? wqSum / confSum : 0;

  const screenW = normConv1(cw, w, h, conf, p.screenSigma, priorW, p.screenPrior).field;
  const screenP = normConv1(cp, w, h, conf, p.screenSigma, priorP, p.screenPrior).field;
  const screenQ = normConv1(cq, w, h, conf, p.screenSigma, priorQ, p.screenPrior).field;
  const modelSupport = normConv1(conf, w, h, conf, p.screenSigma, 0, p.screenPrior).support;

  const B = new Float32Array(n * 3);
  const surfW = new Float32Array(n), surfP = new Float32Array(n), surfQ = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sw = screenW[i], sp = screenP[i], sq = screenQ[i];
    if (sp < 1e-4) { sw = priorW; sp = priorP; sq = priorQ; }
    surfW[i] = sw; surfP[i] = sp; surfQ[i] = sq;
    B[i * 3] = sw * e0[0] + sp * e1[0] + sq * e2[0];
    B[i * 3 + 1] = sw * e0[1] + sp * e1[1] + sq * e2[1];
    B[i * 3 + 2] = sw * e0[2] + sp * e1[2] + sq * e2[2];
  }

  // ---- stage 2: initial alpha -------------------------------------------
  // A *shaded* screen traces a ray from the origin: B_s = s * u, where u is
  // the key's RGB direction.  A mixture C = alpha*F + (1-alpha)*B leaves that
  // ray.  distance(C, ray) grows with alpha; measured in units of |B| it is
  // roughly alpha * sin(angle between F and the screen direction).  We invert
  // that with a conservative, normalised measure that ignores brightness, so
  // bright and shadowed screen areas behave the same.
  const alpha = new Float32Array(n);
  const dInit = new Float32Array(n);
  const uLen = Math.hypot(p.keyLinear[0], p.keyLinear[1], p.keyLinear[2]);
  const u = [p.keyLinear[0] / uLen, p.keyLinear[1] / uLen, p.keyLinear[2] / uLen];
  for (let i = 0; i < n; i++) {
    const cr = C[i * 3], cg = C[i * 3 + 1], cb = C[i * 3 + 2];
    const t = cr * u[0] + cg * u[1] + cb * u[2];       // projection onto screen ray
    const pr = cr - t * u[0], pg = cg - t * u[1], pb = cb - t * u[2];
    const perp = Math.hypot(pr, pg, pb);                // distance from screen ray
    const blen = Math.hypot(B[i * 3], B[i * 3 + 1], B[i * 3 + 2]) + 1e-6;
    // sin of the angle between this colour and the screen ray, brightness free
    const sinAng = perp / (Math.hypot(cr, cg, cb) + 1e-6);
    dInit[i] = sinAng;
    alpha[i] = clamp01((sinAng - p.alphaFloor * 0.5) / (0.5 - p.alphaFloor * 0.5));
    // clamp scale for the special case where |B| dominates (pure screen has
    // sinAng ~ 0; any real object with chroma off the screen ray has larger)
    if (perp < blen * 0.02) alpha[i] *= 0.5;
  }

  // Interiority: how far a pixel is from the silhouette boundary (a
  // confidently-keyed pixel).  Used to keep the foreground-rescue step off
  // genuine anti-aliased edges: at an edge the true alpha is a mix, and
  // forcing it opaque destroys the very sub-pixel coverage we want.  A rescue
  // should only act on enclosed regions (a green object that the key took for
  // screen), which are away from the boundary.
  const interiority = new Float32Array(n);
  {
    const bg = new Uint8Array(n);
    for (let i = 0; i < n; i++) bg[i] = alpha[i] < 0.12 ? 1 : 0;
    // two-pass chamfer distance (in pixels) from a background/edge pixel
    const INF = 1e6;
    const dist = new Float32Array(n).fill(INF);
    for (let i = 0; i < n; i++) if (bg[i]) dist[i] = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let d = dist[i];
      if (y > 0) d = Math.min(d, dist[i - w] + 1);
      if (x > 0) d = Math.min(d, dist[i - 1] + 1);
      if (y > 0 && x > 0) d = Math.min(d, dist[i - w - 1] + 1.4142);
      if (y > 0 && x < w - 1) d = Math.min(d, dist[i - w + 1] + 1.4142);
      dist[i] = d;
    }
    for (let y = h - 1; y >= 0; y--) for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      let d = dist[i];
      if (y < h - 1) d = Math.min(d, dist[i + w] + 1);
      if (x < w - 1) d = Math.min(d, dist[i + 1] + 1);
      if (y < h - 1 && x < w - 1) d = Math.min(d, dist[i + w + 1] + 1.4142);
      if (y < h - 1 && x > 0) d = Math.min(d, dist[i + w - 1] + 1.4142);
      dist[i] = d;
    }
    for (let i = 0; i < n; i++) interiority[i] = Math.min(dist[i], 20);
  }

  // ---- stage 3: foreground model + projection ---------------------------
  // The projection alpha below is exact when F and B are known:
  //   alpha = dot(C - B, F - B) / |F - B|^2.
  // The difficulty is estimating F at an edge pixel, whose own colour is a
  // mixture and therefore useless as an F estimate.  We estimate F from the
  // nearest confidently-opaque "core" pixel (a pixel well off the screen ray),
  // which is exactly right for thin structures where a symmetric blur would
  // average in screen colour.  A colour-coherent bilateral supplies F where a
  // broader object interior is available and is averaged with the core value.
  const fg = new Float32Array(n * 3);
  const fgModel = new Float32Array(n * 3);
  const fgSupport = new Float32Array(n);

  const unmix = (out, a) => {
    for (let i = 0; i < n; i++) {
      const ai = a[i];
      const b0 = B[i * 3], b1 = B[i * 3 + 1], b2 = B[i * 3 + 2];
      if (ai > 0.10) {
        out[i * 3] = (C[i * 3] - (1 - ai) * b0) / ai;
        out[i * 3 + 1] = (C[i * 3 + 1] - (1 - ai) * b1) / ai;
        out[i * 3 + 2] = (C[i * 3 + 2] - (1 - ai) * b2) / ai;
      } else {
        const m = (C[i * 3] + C[i * 3 + 1] + C[i * 3 + 2]) / 3;
        out[i * 3] = m; out[i * 3 + 1] = m; out[i * 3 + 2] = m;
      }
    }
  };

  // Core = pixels well off the screen ray.  Their straight colour ~= C.
  const coreDist = nearestCore(C, dInit, B, w, h, p, fgModel);
  // Blend a coherent bilateral foreground estimate for broad interiors.
  const srcW = new Float32Array(n);
  for (let i = 0; i < n; i++) srcW[i] = smoothstep(p.sourceAlphaMin, Math.min(0.4, p.sourceAlphaMin + 0.18), dInit[i]);
  const cf = coherentForeground(C, srcW, w, h, p);
  for (let i = 0; i < n; i++) {
    const cm = coreDist.fg;
    const haveCore = coreDist.valid[i];
    const haveBi = cf.support[i] > 1e-3;
    fgSupport[i] = haveCore ? 1 : 0;
    for (let c = 0; c < 3; c++) {
      if (haveCore && haveBi) fgModel[i * 3 + c] = p.coreBlend * cm[i * 3 + c] + (1 - p.coreBlend) * cf.fg[i * 3 + c];
      else if (haveCore) fgModel[i * 3 + c] = cm[i * 3 + c];
      else if (haveBi) { fgModel[i * 3 + c] = cf.fg[i * 3 + c]; fgSupport[i] = Math.min(1, cf.support[i]); }
      else fgModel[i * 3 + c] = 0;
    }
  }

  // Projection alpha, blended with the initial ray-distance alpha where the
  // model is weak so we never invent an edge without evidence.
  const cMinFrac = p.mixMinFrac; // |C-B| below this fraction of |B| => screen
  for (let i = 0; i < n; i++) {
    if (fgSupport[i] < 1e-3) continue;
    const cx = C[i * 3] - B[i * 3], cy = C[i * 3 + 1] - B[i * 3 + 1], cz = C[i * 3 + 2] - B[i * 3 + 2];
    const cbLen = Math.hypot(cx, cy, cz);
    const bLen = Math.hypot(B[i * 3], B[i * 3 + 1], B[i * 3 + 2]) + 1e-6;
    const dx = fgModel[i * 3] - B[i * 3];
    const dy = fgModel[i * 3 + 1] - B[i * 3 + 1];
    const dz = fgModel[i * 3 + 2] - B[i * 3 + 2];
    const denom = dx * dx + dy * dy + dz * dz;
    if (denom < 1e-5) continue;
    const aLs = clamp01((cx * dx + cy * dy + cz * dz) / denom);
    // Physical floor: a mixture with coverage alpha must move C at least
    // ~alpha*|F-B| off B.  If C has barely left the screen colour, no model
    // (however confident) can justify a large alpha.  Scale the projection by
    // how far C has actually departed from B, measured against |C-B| that an
    // alpha-sized mix would produce.
    const fMinusB = cMinFrac * bLen;         // required departure for alpha=1
    const departure = smoothstep(fMinusB * 0.15, fMinusB, cbLen);
    const aGated = aLs * departure;
    const conf = Math.min(1, fgSupport[i]);
    alpha[i] = clamp01(alpha[i] * (1 - conf * p.refineStrength) + aGated * conf * p.refineStrength);
  }
  unmix(fg, alpha);

  // ---- stage 4: adjudication --------------------------------------------
  // 4a. A pixel that still lies essentially on the screen ray (no off-axis
  // chroma, brightness consistent with a shaded screen) has no evidence of
  // foreground.  Lock to transparent.
  const protectedFlag = new Uint8Array(n);
  const conflictFlag = new Uint8Array(n);
  let strictLocked = 0, autoProtected = 0;
  for (let i = 0; i < n; i++) {
    if (dInit[i] < p.strictScreenQ * 0.5) {
      if (alpha[i] < 0.6) { alpha[i] *= 0.2; strictLocked++; }
    }
  }

  // 4b. Foreground rescue via the local colour model.  This applies ONLY where
  // the mixing model does not explain the pixel: a true edge pixel lies on the
  // line between B and F, so its mix residual is near zero and projection is
  // trusted exactly.  A pixel with a large residual (model does not fit) that
  // nevertheless sits far from the screen ray and close to the local foreground
  // colour is the ambiguous case ("key-like object" vs "screen") and is lifted.
  for (let i = 0; i < n; i++) {
    if (alpha[i] >= 0.9 || fgSupport[i] < 1e-3) continue;
    // Only rescue enclosed regions, never a genuine silhouette edge.
    const interior = smoothstep(1.5, 4.0, interiority[i]);
    if (interior <= 0) continue;
    const rayFar = smoothstep(0.04, 0.14, dInit[i]);
    if (rayFar <= 0) continue;
    // residual of the best mixing explanation: |C - (B + alpha*(F-B))|
    const ex = B[i * 3] + alpha[i] * (fgModel[i * 3] - B[i * 3]);
    const ey = B[i * 3 + 1] + alpha[i] * (fgModel[i * 3 + 1] - B[i * 3 + 1]);
    const ez = B[i * 3 + 2] + alpha[i] * (fgModel[i * 3 + 2] - B[i * 3 + 2]);
    const mixResid = Math.hypot(C[i * 3] - ex, C[i * 3 + 1] - ey, C[i * 3 + 2] - ez);
    const cfLen = Math.hypot(C[i * 3] - B[i * 3], C[i * 3 + 1] - B[i * 3 + 1], C[i * 3 + 2] - B[i * 3 + 2]) + 1e-6;
    const misfit = smoothstep(0.12, 0.35, mixResid / cfLen);
    if (misfit <= 0) continue;
    const df = Math.hypot(C[i * 3] - fgModel[i * 3], C[i * 3 + 1] - fgModel[i * 3 + 1], C[i * 3 + 2] - fgModel[i * 3 + 2])
      / (Math.hypot(fgModel[i * 3], fgModel[i * 3 + 1], fgModel[i * 3 + 2]) + 0.02);
    const fgLike = smoothstep(0.28, 0.10, df);
    const raise = (1 - alpha[i]) * fgLike * rayFar * misfit * interior * 0.85;
    if (raise > 0.02 && !p._noRescue) { alpha[i] = clamp01(alpha[i] + raise); protectedFlag[i] = 1; autoProtected++; }
  }

  // 4c. Manual protection / force-key overrides everything.
  let manualConflicts = 0, manualApplied = 0;
  if (p.manualMask) {
    const m = p.manualMask;
    for (let i = 0; i < n; i++) {
      const v = m[i]; if (!v) continue;
      const target = v === 1 ? 1 : 0;
      const keyWants = v === 1 ? alpha[i] < 0.5 : alpha[i] > 0.5;
      if (keyWants) { conflictFlag[i] = 1; manualConflicts++; }
      alpha[i] = clamp01(alpha[i] * (1 - p.manualPriority) + target * p.manualPriority);
      manualApplied++;
    }
  }
  for (let i = 0; i < n; i++) {
    if (alpha[i] > 0.995) alpha[i] = 1; else if (alpha[i] < p.alphaExportFloor) alpha[i] = 0;
  }

  // final straight foreground from the final alpha
  unmix(fg, alpha);
  // keep a raw copy for spill diagnostics
  const fgRaw = fg.slice();

  // ---- stage 5: spill correction (F only; alpha never changes) -----------
  // Model: observed foreground = clean foreground + s*screenChroma, s>=0.
  // Screen light carries the screen's chroma, which in this basis is almost
  // entirely along e1 (on-axis), with negligible e2.  So spill shows up as a
  // *positive* e1 component in F.  A neutral object's own colour has e1 ~ 0,
  // so all of that e1 is spill.  A genuinely coloured object carries its own
  // chroma, which is what e2 (off-axis) measures: the more off-axis hue an
  // object has, the more of its e1 we must keep, because part of that e1 is
  // the object's real tint rather than bounce light.
  //
  //   keep = preserveHue = f(|e2| / brightness)
  //   target_e1 = keep * current_e1
  //   remove = spillStrength * (current_e1 - target_e1)   (only when >0)
  //
  // e0 (brightness) and e2 (off-axis hue) are never modified.  This is not a
  // channel clamp: it removes one scalar along the screen's chroma axis and is
  // gated by the object's own off-axis hue, so the presenter's green scarf and
  // the blue product survive while a neutral slab's green cast is removed.
  // Per-pixel bookkeeping for the UI's residue view (what was removed, and what
  // removable spill is still left) — this is diagnostics, not part of the key.
  const spillRemoved = new Float32Array(n);
  const spillResidualPx = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    if (a <= 0.02) continue;
    if (p.manualMask && p.manualMask[i] === 1) continue;
    const fw = fg[i * 3] * e0[0] + fg[i * 3 + 1] * e0[1] + fg[i * 3 + 2] * e0[2];
    const fp = fg[i * 3] * e1[0] + fg[i * 3 + 1] * e1[1] + fg[i * 3 + 2] * e1[2];
    const fq = fg[i * 3] * e2[0] + fg[i * 3 + 1] * e2[1] + fg[i * 3 + 2] * e2[2];
    if (fw <= 1e-5) continue;
    if (fp <= 0) continue; // e1 below neutral is the object's own warm tint, not spill

    // Stable off-axis hue fraction (relative to brightness, not to e1).
    const offAxis = Math.abs(fq) / (fw + 1e-6);
    const keep = smoothstep(p.spillOffAxisLo, p.spillOffAxisHi, offAxis);
    // Local foreground model nudges keep upward where a coherent coloured
    // region surrounds the pixel (protects the interior of a green object).
    let keepLocal = keep;
    if (fgSupport[i] > 1e-3) {
      const mr = fgModel[i * 3], mg = fgModel[i * 3 + 1], mb = fgModel[i * 3 + 2];
      const mw = mr * e0[0] + mg * e0[1] + mb * e0[2];
      const mq = mr * e2[0] + mg * e2[1] + mb * e2[2];
      if (mw > 1e-4) {
        const localOff = Math.abs(mq) / mw;
        keepLocal = Math.max(keep, smoothstep(p.spillOffAxisLo, p.spillOffAxisHi, localOff) * 0.9);
      }
    }
    const targetP = keepLocal * fp;
    const remove = p.spillStrength * Math.max(0, fp - targetP);
    spillRemoved[i] = Math.max(0, fp - targetP);
    // What a full-strength corrector would still call removable (unremoved
    // because spillStrength < 1, or because protection spared it, it is 0).
    spillResidualPx[i] = Math.max(0, fp - targetP) - remove;
    if (remove <= 0) continue;
    const np = fp - remove;
    fg[i * 3] = fw * e0[0] + np * e1[0] + fq * e2[0];
    fg[i * 3 + 1] = fw * e0[1] + np * e1[1] + fq * e2[1];
    fg[i * 3 + 2] = fw * e0[2] + np * e1[2] + fq * e2[2];
  }

  // ---- stage 6: diagnostics ---------------------------------------------
  const residual = new Float32Array(n);
  let edgeCount = 0, resSum = 0;
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    if (a <= 0.05 || a >= 0.95) continue;
    // re-composite against the coherent model and local screen
    const predR = a * fg[i * 3] + (1 - a) * B[i * 3];
    const predG = a * fg[i * 3 + 1] + (1 - a) * B[i * 3 + 1];
    const predB = a * fg[i * 3 + 2] + (1 - a) * B[i * 3 + 2];
    // compare to a smoothed observed value to ignore sensor noise
    residual[i] = Math.hypot(C[i * 3] - predR, C[i * 3 + 1] - predG, C[i * 3 + 2] - predB);
    resSum += residual[i]; edgeCount++;
  }

  let spillBefore = 0, spillAfter = 0, visible = 0, protectedPixels = 0;
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    if (a <= 0.05) continue;
    visible++;
    if (protectedFlag[i]) protectedPixels++;
    const fw0 = fgRaw[i * 3] * e0[0] + fgRaw[i * 3 + 1] * e0[1] + fgRaw[i * 3 + 2] * e0[2];
    const fp0 = fgRaw[i * 3] * e1[0] + fgRaw[i * 3 + 1] * e1[1] + fgRaw[i * 3 + 2] * e1[2];
    const fw1 = fg[i * 3] * e0[0] + fg[i * 3 + 1] * e0[1] + fg[i * 3 + 2] * e0[2];
    const fp1 = fg[i * 3] * e1[0] + fg[i * 3 + 1] * e1[1] + fg[i * 3 + 2] * e1[2];
    spillBefore += Math.max(0, fp0 - 0.12 * fw0);
    spillAfter += Math.max(0, fp1 - 0.12 * fw1);
  }

  let meanAlpha = 0;
  for (let i = 0; i < n; i++) meanAlpha += alpha[i];
  meanAlpha /= n;

  const warnings = [];
  const k = p.keyLinear;
  const kmax = Math.max(k[0], k[1], k[2]), kmin = Math.min(k[0], k[1], k[2]);
  if (kmax < 0.02) warnings.push({ code: 'key-black', text: 'Sampled key colour is almost black; sample a lit part of the screen.' });
  else if ((kmax - kmin) / (kmax + 1e-6) < 0.15) warnings.push({ code: 'key-neutral', text: 'Sampled key colour is nearly neutral; colour separation will be weak.' });
  if (coverage < 0.02) warnings.push({ code: 'low-screen-coverage', text: 'Very few pixels match the screen colour. Resample the key or loosen the chroma thresholds.' });
  let lowConf = 0; for (let i = 0; i < n; i++) if (modelSupport[i] < 0.02) lowConf++;
  if (lowConf / n > 0.25) warnings.push({ code: 'low-screen-model', text: 'The local screen model has low support over much of the image; screen lighting may vary too strongly.' });
  if (meanAlpha < 0.01) warnings.push({ code: 'alpha-low', text: 'Almost everything was keyed out. Check the sampled key and threshold ramp.' });
  if (meanAlpha > 0.985) warnings.push({ code: 'alpha-high', text: 'Almost nothing was keyed out. The key colour or thresholds may not match this screen.' });
  if (manualConflicts > 0) warnings.push({ code: 'manual-conflict', text: `${manualConflicts} manual-mask pixels disagree with the local key model; manual priority resolved them.` });
  if (strictLocked > 0) warnings.push({ code: 'adjudicated-screen', text: `${strictLocked} foreground-like pixels matched the screen too exactly to auto-protect; use the protection brush if they are real objects.` });
  if (spillBefore > 0.01 && spillAfter > spillBefore * 0.9) warnings.push({ code: 'spill-residual', text: 'Spill correction left most key-coloured contamination in place; check spill strength / protection ratio.' });

  const stats = {
    coverage, modelSupportMean: modelSupport.reduce((a, b) => a + b, 0) / n,
    meanAlpha, visiblePixels: visible, autoProtected, protectedPixels,
    strictRejected: strictLocked, strictLocked, manualApplied, manualConflicts,
    edgePixels: edgeCount, edgeResidualMean: edgeCount ? resSum / edgeCount : 0,
    spillBefore: visible ? spillBefore / visible : 0,
    spillAfter: visible ? spillAfter / visible : 0,
    keyLinear: [...p.keyLinear],
    basis: { e0, e1, e2, pK: basis.pK, kcLen: basis.kcLen },
  };

  return {
    width: w, height: h,
    alpha, fg, fgRaw, residual,
    spillRemoved, spillResidual: spillResidualPx,
    screen: { r: B, w: surfW, p: surfP, q: surfQ },
    screenSupport: modelSupport, fgSupport,
    fgModel: fg,
    dField: dInit,
    protectedFlag, conflictFlag,
    stats, warnings,
  };
}

// Auto-pick the dominant saturated border colour (linear).
export function pickAutoKey(rgba, width, height) {
  const ring = Math.max(2, Math.floor(Math.min(width, height) * 0.03));
  const buckets = new Map();
  const add = (r, g, b) => {
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    if (mx < 30) return;
    const sat = (mx - mn) / mx;
    if (sat < 0.15) return;
    const dom = r === mx ? 'r' : g === mx ? 'g' : 'b';
    const key = `${dom}:${Math.round((mx - mn) / 32)}:${Math.round(sat * 8)}`;
    if (!buckets.has(key)) buckets.set(key, { n: 0, r: 0, g: 0, b: 0 });
    const q = buckets.get(key); q.n++; q.r += r; q.g += g; q.b += b;
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (x >= ring && x < width - ring && y >= ring && y < height - ring) continue;
      const i = (y * width + x) * 4;
      add(rgba[i], rgba[i + 1], rgba[i + 2]);
    }
  }
  let best = null;
  for (const v of buckets.values()) if (!best || v.n > best.n) best = v;
  if (!best) return null;
  return [srgb8ToLinear(Math.round(best.r / best.n)), srgb8ToLinear(Math.round(best.g / best.n)), srgb8ToLinear(Math.round(best.b / best.n))];
}

export function compositePixels(res, bgRgb8, outRgba8) {
  const { width: w, height: h, alpha, fg } = res;
  const n = w * h;
  const out = outRgba8 || new Uint8ClampedArray(n * 4);
  const b0 = srgb8ToLinear(bgRgb8[0]), b1 = srgb8ToLinear(bgRgb8[1]), b2 = srgb8ToLinear(bgRgb8[2]);
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    out[i * 4] = linearToSrgb8(Math.max(0, fg[i * 3] * a + b0 * (1 - a)));
    out[i * 4 + 1] = linearToSrgb8(Math.max(0, fg[i * 3 + 1] * a + b1 * (1 - a)));
    out[i * 4 + 2] = linearToSrgb8(Math.max(0, fg[i * 3 + 2] * a + b2 * (1 - a)));
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function alphaRgba(res) {
  const n = res.width * res.height;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const a = Math.round(clamp01(res.alpha[i]) * 255);
    out[i * 4] = a; out[i * 4 + 1] = a; out[i * 4 + 2] = a; out[i * 4 + 3] = 255;
  }
  return out;
}

export function straightRgba(res) {
  const { width: w, height: h, alpha, fg } = res;
  const n = w * h;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    let r = fg[i * 3], g = fg[i * 3 + 1], b = fg[i * 3 + 2];
    if (a <= 0.004) { r = 0.5; g = 0.5; b = 0.5; }
    out[i * 4] = linearToSrgb8(clamp01(r));
    out[i * 4 + 1] = linearToSrgb8(clamp01(g));
    out[i * 4 + 2] = linearToSrgb8(clamp01(b));
    out[i * 4 + 3] = Math.round(clamp01(a) * 255);
  }
  return out;
}

// Residue view: RED = removable screen-axis spill still left in the corrected
// colour (should be near zero after a full-strength despill), GREEN = alpha
// edge residual from the re-composite check (model does not fit the mix), and a
// dim BLUE wash = spill that was successfully removed (context).  Genuine
// green/blue object colour is not "removable spill", so it does NOT light up
// red here — that is the whole point of the two-step separation.
export function spillOverlay(res, outRgba8) {
  const { width: w, height: h, alpha, residual } = res;
  const n = w * h;
  const out = outRgba8 || new Uint8ClampedArray(n * 4);
  const removed = res.spillRemoved;
  const left = res.spillResidual;
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    const red = clamp01((left ? left[i] : 0) * 6) * a;
    const green = clamp01((residual ? residual[i] : 0) * 3) * (1 - a);
    const blue = clamp01((removed ? removed[i] : 0) * 4) * a * 0.7;
    const base = 0.20 + 0.30 * a;
    out[i * 4] = linearToSrgb8(clamp01(base + red * 0.95));
    out[i * 4 + 1] = linearToSrgb8(clamp01(base + green * 0.9));
    out[i * 4 + 2] = linearToSrgb8(clamp01(base + blue * 0.9));
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function toRgba8FromLinear(img, width, height) {
  const n = width * height;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = linearToSrgb8(clamp01(img[i * 3]));
    out[i * 4 + 1] = linearToSrgb8(clamp01(img[i * 3 + 1]));
    out[i * 4 + 2] = linearToSrgb8(clamp01(img[i * 3 + 2]));
    out[i * 4 + 3] = 255;
  }
  return out;
}
