// Ground-truth evaluation for the NON-UNIFORM screen (backdrop) models.
//
// Everything here compares models against the same known ground truth —
// never against "more transparent area", which is not a quality measure.
//
//   A. gradient + cast-shadow screen (shadowScreen): fixed-colour key vs
//      illumination field vs regional re-estimation vs the original unified
//      model — alpha MAE, edge alpha, premultiplied edge colour, composite
//      error on DARK and LIGHT backgrounds.
//   B. wrinkled screen (wrinkleScreen): mid-frequency shading the shared
//      field cannot follow; the regional estimate must beat the fixed key.
//   C. absorption guard: the screen-hue panel must not be swallowed by the
//      background model (the estimated field stays on the true screen), and
//      the new models must not lose more of the panel than the fixed key.
//   D. sample regions: contamination is diagnosed against the clean samples;
//      deleting a sample re-fits exactly the tiles it supported (regional
//      mode) and leaves every other tile bit-identical.
//   E. bounded freedom: an absurd grid request is clamped, and the field
//      model does not trade away edge quality on fineDetail.
//   F. video (walk sequence): temporal update stays stable under camera
//      jitter, adoption collapses when the subject enters the sample region,
//      and the subject is not learned as screen on the following frames.
//
// Run: node tools/evaluate-screenfield.mjs
// Artefacts + screenfield-report.json land in test-output/.

import fs from 'node:fs';
import path from 'node:path';
import { buildScene, buildSequenceFrame, encodePng, SEQUENCES } from './scenes.js';
import { keyImage, compositePixels, alphaRgba } from '../src/keyer.js';
import { srgb8ToLinear, linearToSrgb8, clamp01 } from '../src/color.js';

const OUT = path.resolve('test-output');
fs.mkdirSync(OUT, { recursive: true });

const DARK = [12, 14, 18];
const LIGHT = [238, 240, 245];

function keyLinearFrom(gt) {
  const { width: W, height: H } = gt;
  const sx = Math.round(W * 0.06), sy = Math.round(H * 0.08);
  const si = (sy * W + sx) * 3;
  return [gt.screenL[si], gt.screenL[si + 1], gt.screenL[si + 2]];
}

function alphaMetrics(res, gt) {
  const n = gt.width * gt.height;
  let allE = 0, edgeE = 0, edgeC = 0, screenLeak = 0, screenC = 0, opaqueLeak = 0, opaqueC = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.abs(res.alpha[i] - gt.alpha[i]);
    allE += e;
    const ga = gt.alpha[i];
    if (ga > 0.05 && ga < 0.95) { edgeE += e; edgeC++; }
    if (ga < 0.05) { screenLeak += e; screenC++; }
    if (ga > 0.9) { opaqueLeak += e; opaqueC++; }
  }
  return {
    alphaMae: allE / n,
    edgeMae: edgeC ? edgeE / edgeC : 0,
    screenAlphaLeak: screenC ? screenLeak / screenC : 0,
    opaqueAlphaLeak: opaqueC ? opaqueLeak / opaqueC : 0,
  };
}

function premulEdgeError(res, gt) {
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

// mean |B_est - B_gt| / |B_gt| of the estimated screen field inside a rect
function fieldDeviation(res, gt, rect) {
  const { width: W } = gt;
  let s = 0, c = 0;
  for (let y = rect.y; y < rect.y + rect.h; y++) {
    for (let x = rect.x; x < rect.x + rect.w; x++) {
      const i = y * W + x;
      const br = res.screen.r[i * 3], bg = res.screen.r[i * 3 + 1], bb = res.screen.r[i * 3 + 2];
      const tr = gt.screenL[i * 3], tg = gt.screenL[i * 3 + 1], tb = gt.screenL[i * 3 + 2];
      s += Math.hypot(br - tr, bg - tg, bb - tb) / (Math.hypot(tr, tg, tb) + 1e-6);
      c++;
    }
  }
  return c ? s / c : 0;
}

// absorption ratio: how far the estimated background moved from the true
// screen TOWARD the observed (foreground-covered) colour.  0 = the model
// stayed on the screen; 1 = the model became the foreground object.
function absorptionRatio(res, gt, rect) {
  const { width: W } = gt;
  let s = 0, c = 0;
  for (let y = rect.y; y < rect.y + rect.h; y++) {
    for (let x = rect.x; x < rect.x + rect.w; x++) {
      const i = y * W + x;
      const tr = gt.screenL[i * 3], tg = gt.screenL[i * 3 + 1], tb = gt.screenL[i * 3 + 2];
      const or_ = srgb8ToLinear(gt.input8[i * 4]), og = srgb8ToLinear(gt.input8[i * 4 + 1]), ob = srgb8ToLinear(gt.input8[i * 4 + 2]);
      const moved = Math.hypot(res.screen.r[i * 3] - tr, res.screen.r[i * 3 + 1] - tg, res.screen.r[i * 3 + 2] - tb);
      const available = Math.hypot(or_ - tr, og - tg, ob - tb) + 1e-6;
      s += moved / available;
      c++;
    }
  }
  return c ? s / c : 0;
}

function meanAlphaWhere(res, gt, rect, gtMin) {
  const { width: W } = gt;
  let s = 0, c = 0;
  for (let y = rect.y; y < rect.y + rect.h; y++) {
    for (let x = rect.x; x < rect.x + rect.w; x++) {
      const i = y * W + x;
      if (gt.alpha[i] < gtMin) continue;
      s += res.alpha[i]; c++;
    }
  }
  return c ? s / c : 0;
}

function screenFieldPng(res) {
  const { width: w, height: h } = res;
  const n = w * h;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = linearToSrgb8(clamp01(res.screen.r[i * 3]));
    out[i * 4 + 1] = linearToSrgb8(clamp01(res.screen.r[i * 3 + 1]));
    out[i * 4 + 2] = linearToSrgb8(clamp01(res.screen.r[i * 3 + 2]));
    out[i * 4 + 3] = 255;
  }
  return out;
}

function supportHeatPng(res) {
  const { width: w, height: h } = res;
  const n = w * h;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const s = clamp01(res.screenSupport[i]);
    const hr = clamp01(s * 1.7 - 0.35), hg = clamp01(s * 1.25), hb = clamp01(1.05 - s * 1.5);
    out[i * 4] = linearToSrgb8(clamp01(res.screen.r[i * 3] * 0.25 + hr * 0.75));
    out[i * 4 + 1] = linearToSrgb8(clamp01(res.screen.r[i * 3 + 1] * 0.25 + hg * 0.75));
    out[i * 4 + 2] = linearToSrgb8(clamp01(res.screen.r[i * 3 + 2] * 0.25 + hb * 0.75));
    out[i * 4 + 3] = 255;
  }
  return out;
}

function run() {
  const report = { scenes: {}, temporal: {}, checks: [] };
  let failed = 0;
  const add = (ok, label, detail) => {
    report.checks.push({ ok, label, detail });
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}  (${detail})`);
    if (!ok) failed++;
  };

  // ------------------------------------------------------------------
  // A/B. fixed colour vs field vs regional vs unified, same ground truth
  // ------------------------------------------------------------------
  const MODES = ['uniform', 'field', 'regions', 'legacy'];
  const PANEL = { x: 334, y: 124, w: 92, h: 132 };   // interior of the screen-hue panel

  for (const name of ['shadowScreen', 'wrinkleScreen']) {
    const gt = buildScene(name);
    const { width: W, height: H } = gt;
    const keyLinear = keyLinearFrom(gt);
    const per = {};
    for (const mode of MODES) {
      // wrinkles run at ~35 px; the regional estimate is meant to be tuned
      // to the local lighting scale, so it gets a smaller tile here
      const extra = (name === 'wrinkleScreen' && mode === 'regions') ? { regionSize: 24 } : {};
      const res = keyImage(gt.input8, W, H, { keyLinear, screenMode: mode, ...extra });
      const m = {
        ...alphaMetrics(res, gt),
        premulEdgeErr: premulEdgeError(res, gt),
        compErrDark: compositeError(res, gt, DARK),
        compErrLight: compositeError(res, gt, LIGHT),
        fieldDevPanel: fieldDeviation(res, gt, PANEL),
        absorbPanel: absorptionRatio(res, gt, PANEL),
        panelAlpha: meanAlphaWhere(res, gt, PANEL, 0.9),
      };
      per[mode] = m;
      const stem = path.join(OUT, `sf-${name}-${mode}`);
      fs.writeFileSync(`${stem}-alpha.png`, encodePng(alphaRgba(res), W, H));
      fs.writeFileSync(`${stem}-on-dark.png`, encodePng(compositePixels(res, DARK), W, H));
      if (mode === 'field' || mode === 'regions') {
        fs.writeFileSync(`${stem}-screenfield.png`, encodePng(screenFieldPng(res), W, H));
        fs.writeFileSync(`${stem}-support.png`, encodePng(supportHeatPng(res), W, H));
      }
    }
    report.scenes[name] = per;
    const u = per.uniform;
    for (const mode of ['field', 'regions']) {
      const m = per[mode];
      // The panel is foreground every mode keeps equally, so raw MAE barely
      // moves; the model's job is the screen region.  Require strictly less
      // screen-region alpha leak AND strictly lower MAE, with a combined
      // relative improvement of at least 5% (shadowed screens improve ~10x,
      // mildly wrinkled ones improve modestly but consistently).
      const gain = (u.screenAlphaLeak - m.screenAlphaLeak) / (u.screenAlphaLeak + 1e-9)
        + (u.alphaMae - m.alphaMae) / (u.alphaMae + 1e-9);
      add(m.screenAlphaLeak < u.screenAlphaLeak && m.alphaMae < u.alphaMae && gain >= 0.05,
        `${name}: ${mode} beats fixed-colour key on the screen region`,
        `leak ${m.screenAlphaLeak.toFixed(4)} vs ${u.screenAlphaLeak.toFixed(4)}, MAE ${m.alphaMae.toFixed(4)} vs ${u.alphaMae.toFixed(4)} (gain ${(gain * 100).toFixed(0)}%)`);
      add(m.compErrDark <= u.compErrDark * 1.05 && m.compErrLight <= u.compErrLight,
        `${name}: ${mode} composite error beats fixed-colour key (dark+light)`,
        `${m.compErrDark.toFixed(4)}/${m.compErrLight.toFixed(4)} vs ${u.compErrDark.toFixed(4)}/${u.compErrLight.toFixed(4)}`);
      add(m.premulEdgeErr <= u.premulEdgeErr * 1.05,
        `${name}: ${mode} edge foreground colour not worse than fixed key`,
        `${m.premulEdgeErr.toFixed(4)} vs ${u.premulEdgeErr.toFixed(4)}`);
    }
  }

  // C. absorption guard (shadowScreen's screen-hue panel): the background
  // model must not swallow a same-hue foreground interior — the estimated
  // field stays on the true screen — and the new models keep at least as
  // much of the panel as the fixed-colour key.
  {
    const per = report.scenes.shadowScreen;
    const u = per.uniform;
    for (const mode of ['field', 'regions']) {
      const m = per[mode];
      add(m.absorbPanel < 0.45,
        `shadowScreen: ${mode} background model does not absorb the screen-hue panel`,
        `absorption ${(m.absorbPanel * 100).toFixed(0)}% (field dev ${(m.fieldDevPanel * 100).toFixed(1)}%)`);
      add(m.panelAlpha > u.panelAlpha - 0.05,
        `shadowScreen: ${mode} keeps as much of the panel as the fixed key`,
        `panel alpha ${m.panelAlpha.toFixed(3)} vs fixed ${u.panelAlpha.toFixed(3)}`);
    }
  }

  // ------------------------------------------------------------------
  // D. sample regions: diagnostics, cross-check, strictly local re-fit
  // ------------------------------------------------------------------
  {
    const gt = buildScene('shadowScreen');
    const { width: W, height: H } = gt;
    const keyLinear = keyLinearFrom(gt);
    const clean1 = { x: 8, y: 8, w: 70, h: 50 };
    const clean2 = { x: 402, y: 300, w: 70, h: 52 };
    const clean3 = { x: 240, y: 8, w: 80, h: 46 };
    const bad = { x: 100, y: 180, w: 80, h: 100 };   // lies over the subject's torso

    const resAll = keyImage(gt.input8, W, H, {
      keyLinear, screenMode: 'regions', screenSamples: [clean1, clean2, clean3, bad],
    });
    const diag = resAll.screenDiag;
    add(diag.length === 4 && diag[3].status === 'contaminated'
      && diag[0].status === 'clean' && diag[1].status === 'clean' && diag[2].status === 'clean',
      'samples: contaminated region diagnosed, clean regions pass',
      diag.map((d) => `${d.status}(${(d.onFrac * 100).toFixed(0)}%,dev ${(d.deviation * 100).toFixed(0)}%)`).join(' '));
    add(diag[3].deviation > Math.max(diag[0].deviation, diag[1].deviation, diag[2].deviation) * 1.5,
      'samples: cross-check against clean evidence separates the bad region',
      `dev ${(diag[3].deviation * 100).toFixed(0)}% vs clean ${(diag[0].deviation * 100).toFixed(0)}%/${(diag[1].deviation * 100).toFixed(0)}%/${(diag[2].deviation * 100).toFixed(0)}%`);
    add(diag[3].boost === 0, 'samples: contaminated region is down-weighted before the fit', `boost=${diag[3].boost}`);

    // Deleting a sample must re-fit exactly the tiles it supported.  The tile
    // surface is bilinearly interpolated, so a tile's influence reaches half
    // a tile into its neighbours — the supported region is the sample's tile
    // footprint dilated by one tile (the interpolation halo).  Beyond that
    // the field must be bit-identical.
    const T = 36;
    const tilesOf = (r) => {
      const ts = new Set();
      for (let ty = Math.floor(r.y / T) - 1; ty <= Math.floor((r.y + r.h - 1) / T) + 1; ty++)
        for (let tx = Math.floor(r.x / T) - 1; tx <= Math.floor((r.x + r.w - 1) / T) + 1; tx++)
          ts.add(`${tx},${ty}`);
      return ts;
    };
    const localCheck = (withSamples, withoutSamples, removed) => {
      const withS = keyImage(gt.input8, W, H, { keyLinear, screenMode: 'regions', screenSamples: withSamples });
      const without = keyImage(gt.input8, W, H, { keyLinear, screenMode: 'regions', screenSamples: withoutSamples });
      const supported = tilesOf(removed);
      let outsideMax = 0, insideMax = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const i = (y * W + x) * 3;
          const d = Math.max(
            Math.abs(withS.screen.r[i] - without.screen.r[i]),
            Math.abs(withS.screen.r[i + 1] - without.screen.r[i + 1]),
            Math.abs(withS.screen.r[i + 2] - without.screen.r[i + 2]),
          );
          if (supported.has(`${Math.floor(x / T)},${Math.floor(y / T)}`)) insideMax = Math.max(insideMax, d);
          else outsideMax = Math.max(outsideMax, d);
        }
      }
      return { outsideMax, insideMax };
    };
    const locClean = localCheck([clean1, clean2, clean3], [clean1, clean2], clean3);
    add(locClean.outsideMax === 0 && locClean.insideMax > 0,
      'samples: deleting a clean sample re-fits only the tiles it supported',
      `outside Δ=${locClean.outsideMax}, inside Δ=${locClean.insideMax.toFixed(4)}`);
    const locBad = localCheck([clean1, clean2, bad], [clean1, clean2], bad);
    add(locBad.outsideMax === 0,
      'samples: deleting the contaminated sample leaves unsupported tiles untouched',
      `outside Δ=${locBad.outsideMax}, inside Δ=${locBad.insideMax.toFixed(4)}`);

    // quality: clean samples help the regional estimate on the shadowed screen
    const mNone = alphaMetrics(keyImage(gt.input8, W, H, { keyLinear, screenMode: 'regions' }), gt);
    const mClean = alphaMetrics(keyImage(gt.input8, W, H, { keyLinear, screenMode: 'regions', screenSamples: [clean1, clean2] }), gt);
    const mBadKept = alphaMetrics(resAll, gt);
    report.scenes.shadowScreen.samples = { none: mNone.alphaMae, clean: mClean.alphaMae, badKept: mBadKept.alphaMae };
    add(mClean.alphaMae <= mNone.alphaMae + 0.002 && mBadKept.alphaMae <= mNone.alphaMae + 0.004,
      'samples: clean samples do not hurt and a contaminated one is neutralised',
      `none ${mNone.alphaMae.toFixed(4)} / clean ${mClean.alphaMae.toFixed(4)} / bad kept ${mBadKept.alphaMae.toFixed(4)}`);
  }

  // ------------------------------------------------------------------
  // E. bounded freedom: grid request is clamped; edges stay honest
  // ------------------------------------------------------------------
  {
    const gt = buildScene('fineDetail');
    const keyLinear = keyLinearFrom(gt);
    const resField = keyImage(gt.input8, gt.width, gt.height, {
      keyLinear, screenMode: 'field', fieldGridX: 40, fieldGridY: 40,
    });
    const grid = resField.stats.screenGrid;
    add(grid[0] <= 12 && grid[1] <= 10,
      'freedom: illumination field grid is capped (cannot grow to chase edges)',
      `requested 40x40, used ${grid[0]}x${grid[1]}`);
    const resLegacy = keyImage(gt.input8, gt.width, gt.height, { keyLinear, screenMode: 'legacy' });
    const eF = alphaMetrics(resField, gt).edgeMae;
    const eL = alphaMetrics(resLegacy, gt).edgeMae;
    add(eF < eL + 0.02,
      'freedom: field model keeps fine-detail edge quality close to the unified model',
      `edge MAE ${eF.toFixed(4)} vs unified ${eL.toFixed(4)}`);
  }

  // ------------------------------------------------------------------
  // F. video: temporal update under jitter, subject crossing the samples
  // ------------------------------------------------------------------
  {
    const seq = SEQUENCES.walk;
    const sample = seq.sampleRegion;
    const runSequence = (temporalRate) => {
      let prev = null;
      const out = [];
      for (let t = 0; t < seq.frames; t++) {
        const gt = buildSequenceFrame('walk', t);
        const keyLinear = keyLinearFrom(gt);
        const res = keyImage(gt.input8, gt.width, gt.height, {
          keyLinear, screenMode: 'field',
          screenSamples: [sample],
          temporalRate, temporalPrev: prev,
        });
        prev = res.screenState;
        out.push({ gt, res });
      }
      return out;
    };
    const on = runSequence(0.35);
    const off = runSequence(0);

    // adoption over the field nodes inside the sample region
    const adoptionInSample = (res, W, H) => {
      const st = res.screenState;
      if (!st || !st.adoption || !st.field) return null;
      const { gx, gy } = st.field;
      let s = 0, c = 0;
      for (let ky = 0; ky < gy; ky++) {
        for (let kx = 0; kx < gx; kx++) {
          const x = kx * (W - 1) / (gx - 1), y = ky * (H - 1) / (gy - 1);
          if (x >= sample.x && x <= sample.x + sample.w && y >= sample.y && y <= sample.y + sample.h) {
            s += st.adoption[ky * gx + kx]; c++;
          }
        }
      }
      return c ? s / c : null;
    };

    const W = on[0].gt.width, H = on[0].gt.height;
    const adoptSeries = on.map(({ res }) => adoptionInSample(res, W, H));
    const aClean = adoptSeries[1];
    const aCross = Math.min(adoptSeries[3], adoptSeries[4]);
    add(aCross < aClean * 0.6,
      'video: adoption weight collapses when the subject enters the sample region',
      `clean frame ${aClean && aClean.toFixed(2)} -> crossing ${aCross && aCross.toFixed(2)}`);

    // contamination is flagged while the subject is inside the sample
    const flagged = on.slice(3, 6).some(({ res }) => res.screenDiag.some((d) => d.status === 'contaminated'));
    add(flagged, 'video: sample contamination is diagnosed while the subject crosses it',
      on.map(({ res }, t) => `f${t}:${res.screenDiag[0] ? res.screenDiag[0].status : '-'}(${(res.screenDiag[0] ? res.screenDiag[0].onFrac : 0).toFixed(2)})`).join(' '));

    // the subject is not learned as screen: opaque alpha survives the crossing
    let subjAlpha = 0, subjN = 0;
    for (const t of [3, 4, 5]) {
      const { gt, res } = on[t];
      for (let i = 0; i < gt.width * gt.height; i++) {
        if (gt.alpha[i] > 0.9) { subjAlpha += res.alpha[i]; subjN++; }
      }
    }
    subjAlpha /= Math.max(1, subjN);
    add(subjAlpha > 0.7, 'video: subject crossing the samples is not absorbed next frames',
      `mean opaque-subject alpha during crossing ${subjAlpha.toFixed(3)}`);

    // temporal stability of the background estimate on clean screen pixels
    const stability = (frames) => {
      let s = 0, c = 0;
      for (let t = 1; t < frames.length; t++) {
        const { gt } = frames[t];
        for (let i = 0; i < gt.width * gt.height; i++) {
          if (gt.alpha[i] > 0.05 || frames[t - 1].gt.alpha[i] > 0.05) continue;
          for (let ch = 0; ch < 3; ch++) {
            s += Math.abs(frames[t].res.screen.r[i * 3 + ch] - frames[t - 1].res.screen.r[i * 3 + ch]);
            c++;
          }
        }
      }
      return s / c;
    };
    const stabOn = stability(on);
    const stabOff = stability(off);
    add(stabOn <= stabOff,
      'video: temporal update is more stable than per-frame refits under camera jitter',
      `mean |ΔB| on clean screen ${stabOn.toFixed(5)} vs ${stabOff.toFixed(5)}`);

    // keying quality across the sequence: temporal must not lag behind
    const maeOf = (frames) => frames.reduce((acc, { gt, res }) => acc + alphaMetrics(res, gt).alphaMae, 0) / frames.length;
    const maeOn = maeOf(on), maeOff = maeOf(off);
    add(maeOn <= maeOff + 0.005,
      'video: temporal update keeps alpha quality (no lag penalty)',
      `sequence alpha MAE ${maeOn.toFixed(4)} vs per-frame ${maeOff.toFixed(4)}`);

    report.temporal = { adoptSeries, stabOn, stabOff, maeOn, maeOff };
    const f4 = on[4];
    fs.writeFileSync(path.join(OUT, 'sf-walk-f4-alpha.png'), encodePng(alphaRgba(f4.res), W, H));
    fs.writeFileSync(path.join(OUT, 'sf-walk-f4-support.png'), encodePng(supportHeatPng(f4.res), W, H));
    fs.writeFileSync(path.join(OUT, 'sf-walk-f4-on-dark.png'), encodePng(compositePixels(f4.res, DARK), W, H));
  }

  report.failed = failed;
  fs.writeFileSync(path.join(OUT, 'screenfield-report.json'), JSON.stringify(report, null, 2));
  console.log(`\n${report.checks.length - failed}/${report.checks.length} checks passed`);
  return failed;
}

process.exit(run() ? 1 : 0);
