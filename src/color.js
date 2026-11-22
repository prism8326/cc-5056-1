// Colour-space utilities shared by the keyer and the UI.
// All image maths is done in linear light; sRGB encoding happens at the
// boundaries (decode input, encode output) only.

const SRGB_TO_LINEAR_LUT = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR_LUT[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

export function srgb8ToLinear(v) {
  return SRGB_TO_LINEAR_LUT[v & 255];
}

export function linearToSrgb(x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

export function linearToSrgb8(x) {
  return Math.round(linearToSrgb(x) * 255);
}

export function clamp01(x) {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export function smoothstep(edge0, edge1, x) {
  if (edge1 <= edge0) return x >= edge1 ? 1 : 0;
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

export function dot3(a, b, ao = 0, bo = 0) {
  return a[ao] * b[bo] + a[ao + 1] * b[bo + 1] + a[ao + 2] * b[bo + 2];
}

const INV_SQRT3 = 1 / Math.sqrt(3);

// Orthonormal basis for chroma decomposition:
//   e0 = neutral axis (1,1,1)/sqrt(3)
//   e1 = chroma direction of the screen colour, inside the plane normal to e0
//   e2 = e0 x e1
// Multiplicative shading of the screen scales its chroma vector without
// rotating it, so e1/e2 coordinates are shading invariant by construction.
export function chromaBasis(keyLinear) {
  const e0 = [INV_SQRT3, INV_SQRT3, INV_SQRT3];
  const wK = dot3(keyLinear, e0);
  const kc = [
    keyLinear[0] - wK * e0[0],
    keyLinear[1] - wK * e0[1],
    keyLinear[2] - wK * e0[2],
  ];
  const kcLen = Math.hypot(kc[0], kc[1], kc[2]);
  if (kcLen < 1e-6) throw new Error('Key colour is neutral; sample a saturated screen colour.');
  const e1 = [kc[0] / kcLen, kc[1] / kcLen, kc[2] / kcLen];
  // e2 = e0 x e1
  const e2 = [
    e0[1] * e1[2] - e0[2] * e1[1],
    e0[2] * e1[0] - e0[0] * e1[2],
    e0[0] * e1[1] - e0[1] * e1[0],
  ];
  const e2Len = Math.hypot(e2[0], e2[1], e2[2]);
  e2[0] /= e2Len; e2[1] /= e2Len; e2[2] /= e2Len;
  return { e0, e1, e2, wK, kcLen, pK: kcLen / Math.max(wK, 1e-6) };
}

export function keyUnitDirection(keyLinear) {
  const len = Math.hypot(keyLinear[0], keyLinear[1], keyLinear[2]);
  return [keyLinear[0] / len, keyLinear[1] / len, keyLinear[2] / len];
}

// Bilinear sample of a scalar field.
export function sampleBilinear(field, w, h, x, y) {
  const x0 = Math.max(0, Math.min(w - 1, Math.floor(x)));
  const y0 = Math.max(0, Math.min(h - 1, Math.floor(y)));
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const fx = Math.max(0, Math.min(1, x - x0));
  const fy = Math.max(0, Math.min(1, y - y0));
  const a = field[y0 * w + x0], b = field[y0 * w + x1];
  const c = field[y1 * w + x0], d = field[y1 * w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}
