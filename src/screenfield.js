// Non-uniform screen (backdrop) modelling for the chroma-key kernel.
//
// What lives here:
//   - "uniform": a single screen colour for the whole frame (the fixed-colour
//     key baseline used for comparison).
//   - "field": one shared low-frequency illumination field.  The whole screen
//     shares a single chroma ratio; brightness lives on a coarse bilinear
//     grid.  The grid has BOUNDED degrees of freedom (capped in the kernel),
//     so the model can follow lamp falloff, vignette and cast shadows but can
//     never wrap itself around a foreground object to absorb it.
//   - "regions": the frame is partitioned into tiles; each tile re-estimates
//     its own screen colour from local evidence and is blended with the
//     shared field by how much evidence the tile actually has.  A tile with
//     no evidence falls back to the field instead of inventing a colour.
//     User sample regions only feed the tiles they intersect, so deleting a
//     bad sample re-fits exactly the tiles it supported and nothing else.
//   - sample-region contamination diagnostics: a declared background sample
//     that does not look like screen — or that disagrees with the estimate
//     supported by the remaining (clean) samples — is flagged and
//     down-weighted before it can poison the fit.
//   - a temporal update for video: the estimate is an exponential moving
//     average whose per-node/per-tile adoption weight collapses where the
//     current frame stops looking like screen (e.g. the subject walks into a
//     sample region), so the next frame does not inherit the subject as
//     "screen".
//
// All coordinates are the kernel's chroma coordinates (w = neutral
// brightness, p = on-axis screen chroma, q = off-axis chroma); the caller
// turns the returned (w,p,q) fields back into RGB with the chroma basis.

import { clamp01, smoothstep } from './color.js';

// ---------------------------------------------------------------------------
// background connectivity gate
// ---------------------------------------------------------------------------
// The backdrop is, by definition, the screen-coloured region connected to the
// frame border or to a declared clean sample.  A screen-HUED blob enclosed by
// foreground (a green panel, a mug on the set) is not connected to the
// backdrop; its pixels must not feed the background estimate.  Non-connected
// evidence is not deleted, only down-weighted, so a frame whose screen
// genuinely never touches the border still degrades softly.
function connectivityGate(conf, w, h, seedRects) {
  const n = w * h;
  const thresh = 0.25;
  const bg = new Uint8Array(n);
  const stack = [];
  const push = (i) => { if (!bg[i] && conf[i] > thresh) { bg[i] = 1; stack.push(i); } };
  for (let x = 0; x < w; x++) { push(x); push((h - 1) * w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  for (const r of seedRects) {
    const x0 = Math.max(0, Math.floor(r.x)), y0 = Math.max(0, Math.floor(r.y));
    const x1 = Math.min(w, Math.ceil(r.x + r.w)), y1 = Math.min(h, Math.ceil(r.y + r.h));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) push(y * w + x);
  }
  while (stack.length) {
    const i = stack.pop();
    const x = i % w, y = (i / w) | 0;
    if (x > 0) push(i - 1);
    if (x < w - 1) push(i + 1);
    if (y > 0) push(i - w);
    if (y < h - 1) push(i + w);
    if (x > 0 && y > 0) push(i - w - 1);
    if (x < w - 1 && y > 0) push(i - w + 1);
    if (x > 0 && y < h - 1) push(i + w - 1);
    if (x < w - 1 && y < h - 1) push(i + w + 1);
  }
  return bg;
}

function applyGate(conf, bg, n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = bg[i] ? conf[i] : conf[i] * 0.12;
  return out;
}

// ---------------------------------------------------------------------------
// robust re-fit (absorption guard)
// ---------------------------------------------------------------------------
// A same-hue foreground object is *connected* to the screen through its own
// anti-aliased edge (a green panel's edge pixels are green too), so
// connectivity alone cannot keep it out of the model.  What separates it is
// that its colour is inconsistent with the smooth field supported by the
// overwhelming majority of screen pixels.  So the field is fitted, every
// pixel's residual against the fit is measured, inconsistent pixels are
// down-weighted, and the field is re-fitted.  Shadows and wrinkles (small
// residuals against a followable field) survive; a compact object that
// disagrees with the field by ~2x does not.
function rejectionWeights(cw, cp, cq, conf, w, h, field) {
  const n = w * h;
  const out = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (conf[i] <= 1e-6) { out[i] = 0; continue; }
      const g = evalField(field, x, y, w, h);
      const pw = field.ratio * g;
      const num = Math.hypot(cw[i] - g, cp[i] - pw, cq[i] - field.q);
      const rel = num / (Math.abs(g) + 1e-6);
      // soft rejection: full weight to ~25% residual, gone by ~60%
      out[i] = 1 - smoothstep(0.25, 0.6, rel);
    }
  }
  return out;
}

function fitFieldRobust(cw, cp, cq, conf, w, h, gx, gy, prior, rounds) {
  let confCur = conf;
  let field = fitFieldShared(cw, cp, cq, confCur, w, h, gx, gy, prior);
  let reject = null;
  for (let r = 0; r < rounds; r++) {
    reject = rejectionWeights(cw, cp, cq, confCur, w, h, field);
    const confR = new Float32Array(conf.length);
    for (let i = 0; i < conf.length; i++) confR[i] = conf[i] * reject[i];
    field = fitFieldShared(cw, cp, cq, confR, w, h, gx, gy, prior);
    confCur = confR;
  }
  return { field, conf: confCur, reject };
}

// ---------------------------------------------------------------------------
// small dense linear solve (Gaussian elimination with partial pivoting)
// ---------------------------------------------------------------------------
function solveLinear(A, b, n) {
  const M = A.slice();
  const x = b.slice();
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r * n + col]) > Math.abs(M[piv * n + col])) piv = r;
    }
    if (Math.abs(M[piv * n + col]) < 1e-12) continue;
    if (piv !== col) {
      for (let c = 0; c < n; c++) {
        const t = M[col * n + c]; M[col * n + c] = M[piv * n + c]; M[piv * n + c] = t;
      }
      const t = x[col]; x[col] = x[piv]; x[piv] = t;
    }
    const inv = 1 / M[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = M[r * n + col] * inv;
      if (f === 0) continue;
      for (let c = col; c < n; c++) M[r * n + c] -= f * M[col * n + c];
      x[r] -= f * x[col];
    }
  }
  const out = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = x[r];
    for (let c = r + 1; c < n; c++) s -= M[r * n + c] * out[c];
    const d = M[r * n + r];
    out[r] = s / (Math.abs(d) < 1e-12 ? 1e-12 : d);
  }
  return out;
}

// 1-D hat weights for a node axis of g nodes spanning [0,1].
function axisWeights(t, g) {
  if (g <= 1) return [[0, 1]];
  const u = clamp01(t) * (g - 1);
  let k0 = Math.floor(u);
  if (k0 > g - 2) k0 = g - 2;
  const f = u - k0;
  return [[k0, 1 - f], [k0 + 1, f]];
}

// ---------------------------------------------------------------------------
// shared low-frequency illumination field
// ---------------------------------------------------------------------------
// Brightness g(x) on a gx*gy bilinear grid, fitted by weighted least squares
// in LOG space: multiplicative shading (falloff, vignette, cast shadows) is
// additive in log brightness, so a coarse low-frequency grid represents it
// faithfully instead of averaging steep gradients into a bright bias.
// One shared chroma ratio r (p = r*w) and one shared off-axis mean q serve
// the whole screen.  A regulariser pulls every node toward the global
// weighted mean so unsupported nodes fall back to the shared estimate
// instead of extrapolating.
export function fitFieldShared(cw, cp, cq, conf, w, h, gx, gy, prior, extraObs) {
  const n = w * h;
  const nn = gx * gy;
  const A = new Float64Array(nn * nn);
  const bv = new Float64Array(nn);
  const sup = new Float64Array(nn);
  let cSum = 0, wSum = 0, qSum = 0, lwSum = 0;
  for (let i = 0; i < n; i++) {
    const c = conf[i];
    if (c <= 0) continue;
    cSum += c; wSum += c * cw[i]; qSum += c * cq[i];
    lwSum += c * Math.log(Math.max(1e-4, cw[i]));
  }
  const gMean = cSum > 1e-9 ? wSum / cSum : 0.5;
  const qBar = cSum > 1e-9 ? qSum / cSum : 0;
  const lMean = cSum > 1e-9 ? lwSum / cSum : Math.log(0.5);

  const accumulate = (x, y, cwgt0, lw) => {
    const wy = axisWeights(h <= 1 ? 0 : y / (h - 1), gy);
    const wx = axisWeights(w <= 1 ? 0 : x / (w - 1), gx);
    for (const [ky, py] of wy) {
      for (const [kx, px] of wx) {
        const k = ky * gx + kx;
        const cwgt = cwgt0 * px * py;
        bv[k] += cwgt * lw;
        sup[k] += cwgt;
        for (const [ly, py2] of wy) {
          for (const [lx, px2] of wx) {
            A[k * nn + ly * gx + lx] += cwgt * px2 * py2;
          }
        }
      }
    }
  };

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = conf[y * w + x];
      if (c <= 1e-6) continue;
      accumulate(x, y, c, Math.log(Math.max(1e-4, cw[y * w + x])));
    }
  }
  // user-declared clean samples act as direct robust observations of the
  // background at their location (see fitScreenModel)
  if (extraObs) {
    for (const o of extraObs) {
      accumulate((o.x0 + o.x1 - 1) / 2, (o.y0 + o.y1 - 1) / 2, o.weight, Math.log(Math.max(1e-4, o.w)));
      cSum += o.weight;
    }
  }
  const mu = prior * (cSum / Math.max(1, nn) + 1e-9);
  for (let k = 0; k < nn; k++) { A[k * nn + k] += mu; bv[k] += mu * lMean; }
  const lg = solveLinear(A, bv, nn);
  const g = new Float64Array(nn);
  for (let k = 0; k < nn; k++) g[k] = Math.exp(lg[k]);

  // shared chroma ratio: p_i ~= r * g(x_i)
  let rNum = 0, rDen = 0;
  for (let y = 0; y < h; y++) {
    const wy = axisWeights(h <= 1 ? 0 : y / (h - 1), gy);
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const c = conf[i];
      if (c <= 1e-6) continue;
      const wx = axisWeights(w <= 1 ? 0 : x / (w - 1), gx);
      let gi = 0;
      for (const [ky, py] of wy) for (const [kx, px] of wx) gi += g[ky * gx + kx] * px * py;
      rNum += c * cp[i] * gi;
      rDen += c * gi * gi;
    }
  }
  if (extraObs) {
    for (const o of extraObs) {
      const wy = axisWeights(h <= 1 ? 0 : ((o.y0 + o.y1 - 1) / 2) / (h - 1), gy);
      const wx = axisWeights(w <= 1 ? 0 : ((o.x0 + o.x1 - 1) / 2) / (w - 1), gx);
      let gi = 0;
      for (const [ky, py] of wy) for (const [kx, px] of wx) gi += g[ky * gx + kx] * px * py;
      rNum += o.weight * o.p * gi;
      rDen += o.weight * gi * gi;
    }
  }
  const ratio = rDen > 1e-9 ? rNum / rDen : 0;
  return { gx, gy, g, ratio, q: qBar, support: sup, sRef: cSum / Math.max(1, nn), mean: gMean };
}

export function evalField(field, x, y, w, h) {
  const wy = axisWeights(h <= 1 ? 0 : y / (h - 1), field.gy);
  const wx = axisWeights(w <= 1 ? 0 : x / (w - 1), field.gx);
  let v = 0;
  for (const [ky, py] of wy) for (const [kx, px] of wx) v += field.g[ky * field.gx + kx] * px * py;
  return v;
}

function fieldSupportAt(field, x, y, w, h) {
  const wy = axisWeights(h <= 1 ? 0 : y / (h - 1), field.gy);
  const wx = axisWeights(w <= 1 ? 0 : x / (w - 1), field.gx);
  let v = 0;
  for (const [ky, py] of wy) for (const [kx, px] of wx) v += field.support[ky * field.gx + kx] * px * py;
  return v;
}

// ---------------------------------------------------------------------------
// partitioned (regional) re-estimation
// ---------------------------------------------------------------------------
// Per-tile screen estimate from local evidence, blended with the shared field
// by evidence mass.  `conf` here already includes any (clean) sample boosts;
// because a sample only boosts the pixels inside it, it only influences the
// tiles it intersects — deleting it re-fits exactly those tiles.
export function fitRegions(cw, cp, cq, conf, w, h, T, priorFrac, field) {
  const tx = Math.max(1, Math.ceil(w / T));
  const ty = Math.max(1, Math.ceil(h / T));
  const nt = tx * ty;
  const E = new Float64Array(nt);
  const SW = new Float64Array(nt);
  const SP = new Float64Array(nt);
  const SQ = new Float64Array(nt);
  let cSum = 0;
  for (let y = 0; y < h; y++) {
    const tY = Math.min(ty - 1, (y / T) | 0);
    for (let x = 0; x < w; x++) {
      const c = conf[y * w + x];
      if (c <= 1e-6) continue;
      const t = tY * tx + Math.min(tx - 1, (x / T) | 0);
      E[t] += c; SW[t] += c * cw[y * w + x]; SP[t] += c * cp[y * w + x]; SQ[t] += c * cq[y * w + x];
      cSum += c;
    }
  }
  const confMean = cSum / (w * h);
  const pi = priorFrac * T * T * confMean;
  const W = new Float64Array(nt);
  const R = new Float64Array(nt);
  const Q = new Float64Array(nt);
  const S = new Float64Array(nt);
  for (let tY = 0; tY < ty; tY++) {
    for (let tX = 0; tX < tx; tX++) {
      const t = tY * tx + tX;
      const cx = Math.min(w - 1, (tX + 0.5) * T);
      const cy = Math.min(h - 1, (tY + 0.5) * T);
      const gF = evalField(field, cx, cy, w, h);
      const den = E[t] + pi;
      const wDen = SW[t] + pi * gF + 1e-12;
      W[t] = (SW[t] + pi * gF) / den;
      R[t] = (SP[t] + pi * gF * field.ratio) / wDen;
      Q[t] = (SQ[t] + pi * field.q) / den;
      S[t] = E[t] / den;
    }
  }
  return { tx, ty, T, W, R, Q, S, E, prior: pi };
}

// Bilinear interpolation of a per-tile field over tile centres.
function evalTiles(arr, tx, ty, x, y, w, h, T) {
  const u = Math.min(Math.max((x + 0.5) / T - 0.5, 0), tx - 1);
  const v = Math.min(Math.max((y + 0.5) / T - 0.5, 0), ty - 1);
  const x0 = Math.min(tx - 1, Math.floor(u));
  const y0 = Math.min(ty - 1, Math.floor(v));
  const x1 = Math.min(tx - 1, x0 + 1);
  const y1 = Math.min(ty - 1, y0 + 1);
  const fx = u - x0, fy = v - y0;
  const a = arr[y0 * tx + x0], b = arr[y0 * tx + x1];
  const c = arr[y1 * tx + x0], d = arr[y1 * tx + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// ---------------------------------------------------------------------------
// sample-region contamination diagnostics
// ---------------------------------------------------------------------------
// Each declared sample is checked two ways:
//   onFrac    — fraction of its pixels that look like screen at all;
//   deviation — median chroma-ratio distance |pn - ratio, qn - q| of its
//               pixels against the shared chroma estimate fitted WITHOUT any
//               sample boosts (i.e. supported by the raw evidence and the
//               other, clean samples).  The check is brightness-free on
//               purpose: a shadow band or wrinkle is DARKER than the lit
//               field prediction but keeps the screen's chroma direction, so
//               it must pass; a subject or prop disagrees in chroma and is
//               flagged.  This is the cross-check against uncontaminated
//               evidence.
function diagnoseSamples(samples, cw, cp, cq, conf, w, h, ratioQ, opts) {
  const out = [];
  const norm = Math.abs(ratioQ.ratio) + 0.15;
  for (let index = 0; index < samples.length; index++) {
    const s = samples[index];
    const x0 = Math.max(0, Math.floor(s.x));
    const y0 = Math.max(0, Math.floor(s.y));
    const x1 = Math.min(w, Math.ceil(s.x + s.w));
    const y1 = Math.min(h, Math.ceil(s.y + s.h));
    const area = (x1 - x0) * (y1 - y0);
    if (area < 16) {
      out.push({ index, x: s.x, y: s.y, w: s.w, h: s.h, onFrac: 0, deviation: 0, status: 'empty', boost: 0 });
      continue;
    }
    let on = 0;
    const devs = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = y * w + x;
        if (conf[i] > 0.5) on++;
        const pn = cp[i] / (cw[i] + 1e-6);
        const qn = cq[i] / (cw[i] + 1e-6);
        devs.push(Math.hypot(pn - ratioQ.ratio, qn - ratioQ.q) / norm);
      }
    }
    devs.sort((a, b) => a - b);
    const median = devs[devs.length >> 1];
    const onFrac = on / area;
    const contaminated = onFrac < opts.contamOnFrac || median > opts.contamDev;
    out.push({
      index, x: s.x, y: s.y, w: s.w, h: s.h,
      onFrac, deviation: median,
      status: contaminated ? 'contaminated' : 'clean',
      boost: contaminated ? 0 : 1,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// temporal update (video)
// ---------------------------------------------------------------------------
// Exponential moving average with per-element adoption weights.  Where the
// current frame offers little screen evidence (subject walked into a sample,
// light failed), the adoption weight collapses and the previous estimate is
// kept — the subject is not inherited as "screen" on the next frame.
// `health` (optional) further scales adoption element-wise: nodes/tiles
// covered by a declared sample adopt only as fast as that sample is healthy
// (screen-like) this frame.
function temporalMerge1D(fresh, freshSupport, sRef, prev, rate, health) {
  const n = fresh.length;
  const out = new Float64Array(n);
  const adoption = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let lam = prev ? rate * Math.min(1, freshSupport[k] / (sRef + 1e-9)) : 1;
    if (health) lam *= health[k];
    adoption[k] = lam;
    out[k] = prev ? prev[k] + lam * (fresh[k] - prev[k]) : fresh[k];
  }
  return { values: out, adoption };
}

// ---------------------------------------------------------------------------
// main entry
// ---------------------------------------------------------------------------
// opts: {
//   w, h, cw, cp, cq, conf,            // chroma coords + per-pixel screen confidence
//   mode,                              // 'uniform' | 'field' | 'regions'
//   samples,                           // [{x,y,w,h}] declared background samples (may be null)
//   params,                            // kernel params (thresholds, grid, rates)
//   prevState,                         // screenState from the previous frame (or null)
// }
export function fitScreenModel(opts) {
  const { w, h, cw, cp, cq, conf, mode, params } = opts;
  const n = w * h;
  const samples = (opts.samples || []).filter((s) => s && s.w > 0 && s.h > 0);
  const prevState = opts.prevState && opts.prevState.mode === mode ? opts.prevState : null;
  const rate = params.temporalRate;

  // -- strict chroma-signature evidence -------------------------------------
  // The background model is built only from pixels whose chroma DIRECTION
  // matches the screen (q/w ~ 0): multiplicative shading (shadows, wrinkles,
  // falloff) scales a screen pixel but never rotates its hue, so genuine
  // screen survives at any brightness.  A same-hue foreground object sits
  // off the screen's chroma axis; no matter how large or how connected it
  // is, it contributes nothing to the background estimate.  This is the
  // primary absorption guard.
  const confS = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const qn = Math.abs(cq[i] / (cw[i] + 1e-6));
    confS[i] = conf[i] * (1 - smoothstep(0.07, 0.13, qn));
  }

  // -- connectivity gate, round 0: border-seeded only.  The preliminary base
  // fit from gated evidence classifies the declared samples; clean samples
  // then become additional background anchors for the final gate.
  const gate0 = connectivityGate(confS, w, h, []);
  const confG0 = applyGate(confS, gate0, n);
  const rob0 = fitFieldRobust(cw, cp, cq, confG0, w, h, params.fieldGridX, params.fieldGridY, params.fieldPrior, 1);
  const fit0 = rob0.field;
  const diag0 = samples.length
    ? diagnoseSamples(samples, cw, cp, cq, conf, w, h, { ratio: fit0.ratio, q: fit0.q }, params)
    : [];
  const cleanRects = samples.filter((s, i) => diag0[i] && diag0[i].status === 'clean');

  // -- final gate + robust base fit from raw (gated) evidence.  This base
  // fit is the reference every declared sample is cross-checked against, so
  // a contaminated sample is judged by clean evidence, never by itself.
  const gate1 = connectivityGate(confS, w, h, cleanRects);
  const confG1 = applyGate(confS, gate1, n);
  const rob1 = fitFieldRobust(cw, cp, cq, confG1, w, h, params.fieldGridX, params.fieldGridY, params.fieldPrior, 2);
  const baseField = rob1.field;
  const confBase = rob1.conf;   // gated + robustness-weighted evidence

  // -- contamination diagnostics -------------------------------------------
  const diag = samples.length
    ? diagnoseSamples(samples, cw, cp, cq, conf, w, h, { ratio: baseField.ratio, q: baseField.q }, params)
    : [];

  // -- boosted confidence: clean samples strengthen their own pixels;
  // contaminated ones are left at their natural (usually low) confidence.
  let confFit = confBase;
  if (samples.length) {
    confFit = new Float32Array(confBase);
    for (const d of diag) {
      if (!d.boost) continue;
      const x0 = Math.max(0, Math.floor(d.x));
      const y0 = Math.max(0, Math.floor(d.y));
      const x1 = Math.min(w, Math.ceil(d.x + d.w));
      const y1 = Math.min(h, Math.ceil(d.y + d.h));
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = y * w + x;
          confFit[i] = Math.min(1, confFit[i] * params.sampleBoost + 0.25 * params.sampleBoost * 0.25);
        }
      }
    }
  }

  // -- clean samples as direct observations ----------------------------------
  // A clean (cross-checked) sample is a user-declared reliable measurement of
  // the background at its location.  Beyond boosting pixel confidence it
  // contributes its robust per-channel median as a pseudo-observation, so the
  // estimate is anchored there even where pixel confidence is already high —
  // and deleting the sample re-fits exactly the nodes/tiles it supported.
  const sampleObs = [];
  for (const d of diag) {
    if (!d.boost) continue;
    const x0 = Math.max(0, Math.floor(d.x));
    const y0 = Math.max(0, Math.floor(d.y));
    const x1 = Math.min(w, Math.ceil(d.x + d.w));
    const y1 = Math.min(h, Math.ceil(d.y + d.h));
    const ws = [], ps = [], qs = [];
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = y * w + x;
        if (conf[i] > 0.25) { ws.push(cw[i]); ps.push(cp[i]); qs.push(cq[i]); }
      }
    }
    if (ws.length < 16) continue;
    ws.sort((a, b) => a - b); ps.sort((a, b) => a - b); qs.sort((a, b) => a - b);
    const mid = ws.length >> 1;
    sampleObs.push({
      x0, y0, x1, y1,
      w: ws[mid], p: ps[mid], q: qs[mid],
      weight: params.sampleBoost * 0.05 * ws.length,
    });
  }

  // -- sample health per position: 1 outside declared samples; inside a
  // sample it tracks how screen-like the sample still is this frame.  The
  // temporal adoption of nearby nodes/tiles is scaled by it, so a subject
  // walking into a sample slows the update there instead of being learned.
  const healthAt = (x, y) => {
    let hh = 1;
    for (const d of diag) {
      if (x >= d.x && x <= d.x + d.w && y >= d.y && y <= d.y + d.h) {
        hh = Math.min(hh, smoothstep(params.contamOnFrac, 0.95, d.onFrac));
      }
    }
    return hh;
  };

  // -- final fit per mode ----------------------------------------------------
  const wArr = new Float32Array(n);
  const pArr = new Float32Array(n);
  const qArr = new Float32Array(n);
  const support = new Float32Array(n);
  let state = { mode };
  let adoptionMean = 1;

  if (mode === 'uniform') {
    let cS = 0, wS = 0, pS = 0, qS = 0;
    for (let i = 0; i < n; i++) {
      const c = confFit[i];
      if (c <= 0) continue;
      cS += c; wS += c * cw[i]; pS += c * cp[i]; qS += c * cq[i];
    }
    for (const o of sampleObs) {
      cS += o.weight; wS += o.weight * o.w; pS += o.weight * o.p; qS += o.weight * o.q;
    }
    let W = cS > 1e-9 ? wS / cS : baseField.mean;
    let P = cS > 1e-9 ? pS / cS : baseField.ratio * baseField.mean;
    let Q = cS > 1e-9 ? qS / cS : baseField.q;
    let adoption = null;
    if (rate > 0) {
      const prev = prevState && prevState.uniform;
      const sup = [cS / n];
      const mW = temporalMerge1D([W], sup, 0.08, prev && [prev.w], rate, null);
      const mP = temporalMerge1D([P], sup, 0.08, prev && [prev.p], rate, null);
      const mQ = temporalMerge1D([Q], sup, 0.08, prev && [prev.q], rate, null);
      W = mW.values[0]; P = mP.values[0]; Q = mQ.values[0];
      adoption = mW.adoption;
      adoptionMean = adoption[0];
    }
    for (let i = 0; i < n; i++) {
      wArr[i] = W; pArr[i] = P; qArr[i] = Q;
      support[i] = clamp01((cS / n) * 2);
    }
    state.uniform = { w: W, p: P, q: Q };
    state.adoption = adoption;
  } else if (mode === 'field') {
    const fresh = fitFieldShared(cw, cp, cq, confFit, w, h, params.fieldGridX, params.fieldGridY, params.fieldPrior, sampleObs);
    let g = fresh.g, ratio = fresh.ratio, q = fresh.q;
    let adoption = null;
    if (rate > 0) {
      const prev = prevState && prevState.field;
      const health = new Float32Array(fresh.gx * fresh.gy);
      for (let ky = 0; ky < fresh.gy; ky++) {
        for (let kx = 0; kx < fresh.gx; kx++) {
          health[ky * fresh.gx + kx] = healthAt(
            kx * (w - 1) / (fresh.gx - 1),
            ky * (h - 1) / (fresh.gy - 1),
          );
        }
      }
      const mg = temporalMerge1D(fresh.g, fresh.support, fresh.sRef, prev && prev.g, rate, health);
      g = mg.values;
      adoption = mg.adoption;
      const lam = adoption.reduce((a, b) => a + b, 0) / adoption.length;
      ratio = prev ? prev.ratio + lam * (fresh.ratio - prev.ratio) : fresh.ratio;
      q = prev ? prev.q + lam * (fresh.q - prev.q) : fresh.q;
      adoptionMean = lam;
    }
    const merged = { ...fresh, g, ratio, q };
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const gi = evalField(merged, x, y, w, h);
        wArr[i] = gi;
        pArr[i] = ratio * gi;
        qArr[i] = q;
        support[i] = clamp01(fieldSupportAt(fresh, x, y, w, h) / (fresh.sRef + 1e-9));
      }
    }
    state.field = { gx: fresh.gx, gy: fresh.gy, g, ratio, q };
    state.adoption = adoption;
    state.grid = [fresh.gx, fresh.gy];
  } else {
    // regions: tiles fitted from (boosted) local evidence, prior = base field
    // fitted from raw evidence.  The prior never sees the sample boosts, so a
    // bad sample cannot leak into tiles it does not intersect.
    const fresh = fitRegions(cw, cp, cq, confFit, w, h, params.regionSize, params.regionPrior, baseField);
    // blend each clean sample's robust observation into exactly the tiles it
    // overlaps — deleting the sample re-fits those tiles and nothing else
    for (const o of sampleObs) {
      const tx0 = Math.floor(o.x0 / fresh.T), tx1 = Math.floor((o.x1 - 1) / fresh.T);
      const ty0 = Math.floor(o.y0 / fresh.T), ty1 = Math.floor((o.y1 - 1) / fresh.T);
      for (let ty = ty0; ty <= ty1; ty++) {
        for (let tx = tx0; tx <= tx1; tx++) {
          const overlap = Math.max(0, Math.min(o.x1, (tx + 1) * fresh.T) - Math.max(o.x0, tx * fresh.T))
            * Math.max(0, Math.min(o.y1, (ty + 1) * fresh.T) - Math.max(o.y0, ty * fresh.T));
          if (!overlap) continue;
          const t = ty * fresh.tx + tx;
          const bW = params.sampleBoost * 0.25 * overlap;
          const E0 = fresh.E[t];
          fresh.W[t] = (fresh.W[t] * E0 + o.w * bW) / (E0 + bW);
          fresh.R[t] = (fresh.R[t] * E0 + o.p * bW) / (E0 + bW);
          fresh.Q[t] = (fresh.Q[t] * E0 + o.q * bW) / (E0 + bW);
          fresh.E[t] = E0 + bW;
        }
      }
    }
    let W = fresh.W, R = fresh.R, Q = fresh.Q;
    let adoption = null;
    if (rate > 0) {
      const prev = prevState && prevState.tiles;
      const sRef = fresh.prior * 2;
      const health = new Float32Array(fresh.tx * fresh.ty);
      for (let ty = 0; ty < fresh.ty; ty++) {
        for (let tx = 0; tx < fresh.tx; tx++) {
          health[ty * fresh.tx + tx] = healthAt(
            Math.min(w - 1, (tx + 0.5) * fresh.T),
            Math.min(h - 1, (ty + 0.5) * fresh.T),
          );
        }
      }
      const mW = temporalMerge1D(fresh.W, fresh.E, sRef, prev && prev.W, rate, health);
      // adoption is driven by local evidence mass; reuse the same weights for R/Q
      const mR = temporalMerge1D(fresh.R, fresh.E, sRef, prev && prev.R, rate, health);
      const mQ = temporalMerge1D(fresh.Q, fresh.E, sRef, prev && prev.Q, rate, health);
      W = mW.values; R = mR.values; Q = mQ.values;
      adoption = mW.adoption;
      adoptionMean = adoption.reduce((a, b) => a + b, 0) / adoption.length;
    }
    const { tx, ty, T } = fresh;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const wi = evalTiles(W, tx, ty, x, y, w, h, T);
        wArr[i] = wi;
        pArr[i] = evalTiles(R, tx, ty, x, y, w, h, T) * wi;
        qArr[i] = evalTiles(Q, tx, ty, x, y, w, h, T);
        support[i] = clamp01(evalTiles(fresh.S, tx, ty, x, y, w, h, T));
      }
    }
    state.tiles = { tx, ty, T, W, R, Q };
    state.adoption = adoption;
    state.grid = [tx, ty];
  }

  return { wArr, pArr, qArr, support, diag, state, adoptionMean };
}
