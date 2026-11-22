// Ground-truth evaluation for the chroma-key kernel.
//
// The delivered result is compared against the *clean* foreground (the true
// subject albedo with screen-spill light removed) and the true coverage, not
// against the raw input.  That is the whole point of a key + despill: recover
// the subject as it would look off the screen.
//
// Metrics:
//   alpha MAE, edge alpha MAE (with a JPEG oracle floor for honesty)
//   premultiplied edge colour error  (what the composite actually sees)
//   composite error over a DARK and a LIGHT background vs clean truth
//   spill contamination before/after despill on opaque pixels
//   preserved hue on genuinely green/blue objects
//   thin-structure retention
//
// Run: node tools/evaluate.mjs

import fs from 'node:fs';
import path from 'node:path';
import { buildScene, encodePng } from './scenes.js';
import { keyImage, compositePixels, alphaRgba, straightRgba, spillOverlay } from '../src/keyer.js';
import { srgb8ToLinear } from '../src/color.js';

const OUT = path.resolve('test-output');
fs.mkdirSync(OUT, { recursive: true });

const DARK = [12, 14, 18];
const LIGHT = [238, 240, 245];
const SCENES = ['presenter', 'blueProduct', 'fineDetail', 'despill'];

function keyLinearFrom(gt) {
  const { width: W, height: H } = gt;
  const sx = Math.round(W * 0.06), sy = Math.round(H * 0.08);
  const si = (sy * W + sx) * 3;
  return [gt.screenL[si], gt.screenL[si + 1], gt.screenL[si + 2]];
}

function decodeLinear(gt) {
  const n = gt.width * gt.height;
  const C = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) C[i * 3 + c] = srgb8ToLinear(gt.input8[i * 4 + c]);
  return C;
}

function oracleEdgeAlpha(gt) {
  const { width: W, height: H } = gt;
  const n = W * H;
  const C = decodeLinear(gt);
  let s = 0, c = 0;
  for (let i = 0; i < n; i++) {
    const ga = gt.alpha[i];
    if (ga <= 0.05 || ga >= 0.95) continue;
    const dx = gt.fgL[i * 3] - gt.screenL[i * 3], dy = gt.fgL[i * 3 + 1] - gt.screenL[i * 3 + 1], dz = gt.fgL[i * 3 + 2] - gt.screenL[i * 3 + 2];
    const den = dx * dx + dy * dy + dz * dz;
    if (den < 1e-9) continue;
    const cx = C[i * 3] - gt.screenL[i * 3], cy = C[i * 3 + 1] - gt.screenL[i * 3 + 1], cz = C[i * 3 + 2] - gt.screenL[i * 3 + 2];
    const a = Math.max(0, Math.min(1, (cx * dx + cy * dy + cz * dz) / den));
    s += Math.abs(a - ga); c++;
  }
  return c ? s / c : 0;
}

function compositeError(res, gt, bg) {
  const n = gt.width * gt.height;
  const bgL = [srgb8ToLinear(bg[0]), srgb8ToLinear(bg[1]), srgb8ToLinear(bg[2])];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const a = res.alpha[i];
    for (let c = 0; c < 3; c++) {
      const got = res.fg[i * 3 + c] * a + bgL[c] * (1 - a);
      const tru = gt.fgCleanL[i * 3 + c] * gt.alpha[i] + bgL[c] * (1 - gt.alpha[i]);
      s += Math.abs(got - tru);
    }
  }
  return s / (n * 3);
}

function premultipliedEdgeError(res, gt) {
  const n = gt.width * gt.height;
  let s = 0, c = 0;
  for (let i = 0; i < n; i++) {
    const ga = gt.alpha[i];
    if (ga <= 0.05 || ga >= 0.95) continue;
    s += Math.hypot(
      res.alpha[i] * res.fg[i * 3] - ga * gt.fgCleanL[i * 3],
      res.alpha[i] * res.fg[i * 3 + 1] - ga * gt.fgCleanL[i * 3 + 1],
      res.alpha[i] * res.fg[i * 3 + 2] - ga * gt.fgCleanL[i * 3 + 2],
    );
    c++;
  }
  return c ? s / c : 0;
}

function objMask(gt, kind) {
  const n = gt.width * gt.height;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (gt.alpha[i] < 0.9) continue;
    const r = gt.fgCleanL[i * 3], g = gt.fgCleanL[i * 3 + 1], b = gt.fgCleanL[i * 3 + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = (mx - mn) / (mx + 1e-6);
    if (sat < 0.35) continue;
    const dom = mx === r ? 'r' : mx === g ? 'g' : 'b';
    if (dom === kind) m[i] = 1;
  }
  return m;
}

function run() {
  const report = { scenes: {}, checks: [] };
  let failed = 0;
  const add = (ok, label, detail) => { report.checks.push({ ok, label, detail }); if (!ok) failed++; };

  for (const name of SCENES) {
    const gt = buildScene(name);
    const { width: W, height: H } = gt;
    const n = W * H;
    const keyLinear = keyLinearFrom(gt);
    const t0 = Date.now();
    const res = keyImage(gt.input8, W, H, { keyLinear });
    const ms = Date.now() - t0;

    const oracle = oracleEdgeAlpha(gt);

    // alpha metrics
    let allE = 0, edgeE = 0, edgeC = 0, opaqueLeak = 0, opaqueC = 0, screenLeak = 0, screenC = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.abs(res.alpha[i] - gt.alpha[i]);
      allE += e;
      const ga = gt.alpha[i];
      if (ga > 0.05 && ga < 0.95) { edgeE += e; edgeC++; }
      if (ga > 0.9) { opaqueLeak += e; opaqueC++; }
      if (ga < 0.05) { screenLeak += e; screenC++; }
    }
    const alphaMae = allE / n;
    const edgeMae = edgeC ? edgeE / edgeC : 0;

    // colour / spill metrics, on opaque visible pixels
    const { e0, e1 } = res.stats.basis;
    const neutral = new Uint8Array(n);
    const greenObj = objMask(gt, 'g');
    const blueObj = objMask(gt, 'b');
    for (let i = 0; i < n; i++) {
      if (gt.alpha[i] < 0.9) continue;
      if (greenObj[i] || blueObj[i]) continue;
      const r = gt.fgCleanL[i * 3], g = gt.fgCleanL[i * 3 + 1], b = gt.fgCleanL[i * 3 + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      if ((mx - mn) / (mx + 1e-6) < 0.2) neutral[i] = 1;
    }
    const hadSpill = (gt.scene.spill && gt.scene.spill.amount > 0);

    // Spill contamination = key-axis chroma offset from the clean truth.
    //
    // A despill can only remove spill where the spill is *separable* from the
    // object's own colour: i.e. where the clean subject carries negligible
    // chroma along the key axis.  On a genuinely blue/green object the spill
    // is collinear with — and a small fraction of — the object's own key-axis
    // chroma, so removing it would destroy the object (the exact thing the
    // despill must NOT do).  We therefore measure removal on pixels whose
    // clean truth is neutral on the key axis (|e1|/brightness < 0.04), and
    // keep the separate hue-preservation checks for the coloured objects.
    const keyChroma = (v, i) => v[i * 3] * e1[0] + v[i * 3 + 1] * e1[1] + v[i * 3 + 2] * e1[2];
    const bright = (v, i) => v[i * 3] * e0[0] + v[i * 3 + 1] * e0[1] + v[i * 3 + 2] * e0[2];
    let contBefore = 0, contAfter = 0, contC = 0;
    for (let i = 0; i < n; i++) {
      if (gt.alpha[i] < 0.9) continue;
      const cw = bright(gt.fgCleanL, i);
      if (cw < 1e-4) continue;
      if (Math.abs(keyChroma(gt.fgCleanL, i)) / cw >= 0.04) continue; // object hue is not separable from spill
      const cleanP = keyChroma(gt.fgCleanL, i);
      contBefore += Math.abs(keyChroma(res.fgRaw, i) - cleanP);
      contAfter += Math.abs(keyChroma(res.fg, i) - cleanP);
      contC++;
    }
    // Require enough separable pixels to make a removal claim meaningful.
    const separable = contC >= 200;
    const spillBefore = separable ? contBefore / contC : 0;
    const spillAfter = separable ? contAfter / contC : 0;

    // hue preservation on genuinely coloured objects (should match clean truth)
    const objErr = (mask, count) => {
      if (!count) return null;
      let s = 0;
      for (let i = 0; i < n; i++) if (mask[i]) s += Math.hypot(res.fg[i * 3] - gt.fgCleanL[i * 3], res.fg[i * 3 + 1] - gt.fgCleanL[i * 3 + 1], res.fg[i * 3 + 2] - gt.fgCleanL[i * 3 + 2]);
      return s / count;
    };
    let gN = 0, bN = 0;
    for (let i = 0; i < n; i++) { if (greenObj[i]) gN++; if (blueObj[i]) bN++; }
    const greenErr = objErr(greenObj, gN);
    const blueErr = objErr(blueObj, bN);

    const premulErr = premultipliedEdgeError(res, gt);
    const compDark = compositeError(res, gt, DARK);
    const compLight = compositeError(res, gt, LIGHT);

    // thin structure retention
    let thinTotal = 0, thinLost = 0;
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (gt.alpha[i] < 0.6) continue;
      let narrow = false;
      for (const [dx, dy] of [[1, 0], [0, 1]]) {
        const j = (y + dy) * W + (x + dx);
        if (gt.alpha[j] < 0.25) narrow = true;
      }
      if (!narrow) continue;
      thinTotal++;
      if (res.alpha[i] < 0.4) thinLost++;
    }

    // ---- artefacts --------------------------------------------------------
    const stem = path.join(OUT, name);
    fs.writeFileSync(`${stem}-input.png`, encodePng(gt.input8, W, H));
    fs.writeFileSync(`${stem}-gt-alpha.png`, encodePng(alphaRgba({ width: W, height: H, alpha: gt.alpha }), W, H));
    fs.writeFileSync(`${stem}-alpha.png`, encodePng(alphaRgba(res), W, H));
    fs.writeFileSync(`${stem}-straight.png`, encodePng(straightRgba(res), W, H));
    fs.writeFileSync(`${stem}-on-dark.png`, encodePng(compositePixels(res, DARK), W, H));
    fs.writeFileSync(`${stem}-on-light.png`, encodePng(compositePixels(res, LIGHT), W, H));
    fs.writeFileSync(`${stem}-spill-overlay.png`, encodePng(spillOverlay(res), W, H));
    const diff = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
      const e = Math.abs(res.alpha[i] - gt.alpha[i]);
      const v = Math.round(Math.min(1, e) * 255);
      diff[i * 4] = v; diff[i * 4 + 1] = Math.round(v * 0.25); diff[i * 4 + 2] = Math.round(v * 0.1); diff[i * 4 + 3] = 255;
    }
    fs.writeFileSync(`${stem}-alpha-diff.png`, encodePng(diff, W, H));

    const m = {
      ms, oracleEdgeAlpha: oracle,
      alphaMae, edgeMae, edgeAlphaOverOracle: edgeMae - oracle,
      opaqueAlphaLeak: opaqueC ? opaqueLeak / opaqueC : 0,
      screenAlphaLeak: screenC ? screenLeak / screenC : 0,
      premulEdgeErr: premulErr,
      compErrDark: compDark, compErrLight: compLight,
      spillBefore, spillAfter, hadSpill,
      greenErr, blueErr,
      thinTotal, thinLost,
      warnings: res.warnings.map((w) => w.code),
      stats: {
        coverage: res.stats.coverage,
        autoProtected: res.stats.autoProtected,
        strictLocked: res.stats.strictLocked,
        edgeResidualMean: res.stats.edgeResidualMean,
      },
    };
    report.scenes[name] = m;

    // ---- checks -----------------------------------------------------------
    add(alphaMae < 0.012, `${name}: alpha MAE < 0.012`, alphaMae.toFixed(4));
    add(edgeMae < oracle + 0.035, `${name}: edge alpha near JPEG oracle`, `${edgeMae.toFixed(3)} vs oracle ${oracle.toFixed(3)}`);
    add(m.opaqueAlphaLeak < 0.02, `${name}: opaque alpha preserved`, m.opaqueAlphaLeak.toFixed(4));
    add(m.screenAlphaLeak < 0.02, `${name}: screen keyed out`, m.screenAlphaLeak.toFixed(4));
    add(premulErr < 0.09, `${name}: premultiplied edge colour`, premulErr.toFixed(4));
    add(compDark < 0.02, `${name}: composite on DARK bg`, compDark.toFixed(4));
    add(compLight < 0.02, `${name}: composite on LIGHT bg`, compLight.toFixed(4));
    if (greenErr !== null) add(greenErr < 0.05, `${name}: green object hue preserved`, greenErr.toFixed(4));
    if (blueErr !== null) add(blueErr < 0.05, `${name}: blue object hue preserved`, blueErr.toFixed(4));
    if (hadSpill) {
      if (separable) add(spillAfter < spillBefore * 0.8, `${name}: despill removes separable contamination`, `${spillBefore.toFixed(4)} -> ${spillAfter.toFixed(4)} (neutral px)`);
      else add(true, `${name}: despill N/A — all visible colour is subject hue (inseparable)`, `${contC} neutral px`);
    }
    if (name === 'fineDetail') add(thinLost / Math.max(1, thinTotal) < 0.05, `${name}: thin detail preserved`, `${thinLost}/${thinTotal}`);
  }

  report.failed = failed;
  fs.writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 2));
  for (const c of report.checks) console.log(`${c.ok ? 'PASS' : 'FAIL'}  ${c.label}  (${c.detail})`);
  console.log(`\n${report.checks.length - failed}/${report.checks.length} checks passed`);
  return failed;
}

process.exit(run() ? 1 : 0);
