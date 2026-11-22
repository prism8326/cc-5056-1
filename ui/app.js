// Chroma-Key Studio — UI shell around the shared kernel.
//
// Everything visual here is fed by src/keyer.js (the same module the ground-
// truth evaluation uses).  The UI never synthesises a mask or a result: it
// hands the decoded input pixels and the current parameter set to keyImage()
// and paints whatever comes back.  That is what makes the controls "drive the
// chroma-key foreground" rather than sit beside a canned result.

import {
  keyImage, pickAutoKey, DEFAULT_PARAMS,
  compositePixels, alphaRgba, straightRgba, spillOverlay,
} from '../src/keyer.js';
import { srgb8ToLinear, linearToSrgb, clamp01 } from '../src/color.js';

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const DARK = [12, 14, 18];
const LIGHT = [238, 240, 245];

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  name: 'presenter',
  width: 0,
  height: 0,
  imageData: null,        // ImageData of the decoded input (8-bit sRGB, opaque)
  input8: null,           // Uint8ClampedArray RGBA
  keyLinear: null,
  params: {
    ...DEFAULT_PARAMS,
    screenQThresh0: DEFAULT_PARAMS.screenQThresh[0],
    screenQThresh1: DEFAULT_PARAMS.screenQThresh[1],
  },
  manualMask: null,       // Uint8Array, 1=protect 2=force-key
  result: null,
  view: 'comp',
  bg: { kind: 'dark', custom: [74, 85, 104] },
  brush: 'off',
  brushSize: 18,
  eyedrop: false,
  samples: [],            // declared reliable background regions [{x,y,w,h}]
  deletedSamples: [],     // undo stack for deleted samples
  sampleDraw: false,      // sample-drawing tool active
  dragRect: null,         // in-progress sample rect (image coords)
  sequence: null,         // { name, frames, frame } when a video sequence is loaded
  temporalPrev: null,     // screenState handed to the next frame's kernel run
  playing: false,
  dirty: false,
  busy: false,
  pendingTimer: null,
  lastPaint: null,        // { kind, imageData } for eyedropper lookup
  computeSeq: 0,          // increments each completed kernel run (test hook)
};

// Ranges for the editable parameters.  Each slider is wired straight to a key
// in DEFAULT_PARAMS so the UI cannot claim an effect the kernel ignores.
const PARAM_SPEC = [
  { group: 'Alpha ramp', items: [
    { key: 'alphaFloor', label: 'Alpha floor', min: 0, max: 0.3, step: 0.005,
      fmt: (v) => v.toFixed(3), hint: 'Distance from the screen ray below which a pixel is treated as pure screen.' },
  ]},
  { group: 'Screen model', items: [
    { key: 'screenSigma', label: 'Screen smoothing σ', min: 2, max: 40, step: 1,
      fmt: (v) => `${v | 0} px`, hint: 'Spatial scale of the local screen-lighting estimate. Raise for gradual falloff, lower near shadows.' },
    { key: 'screenQThresh0', label: 'On-axis tolerance (lo)', min: 0, max: 0.4, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'Off-axis chroma below this is treated as screen.' },
    { key: 'screenQThresh1', label: 'On-axis tolerance (hi)', min: 0.02, max: 0.6, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'Off-axis chroma above this is never screen.' },
    { key: 'screenRatioTol', label: 'Chroma ratio tolerance', min: 0.05, max: 0.8, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'How close the p/w ratio must match the key before a shaded pixel counts as screen.' },
  ]},
  { group: 'Backdrop field (non-uniform screen)', items: [
    { key: 'fieldGridX', label: 'Field grid columns', min: 2, max: 12, step: 1,
      fmt: (v) => `${v | 0}`, hint: 'Low-frequency illumination field resolution (bounded — the model cannot grow extra freedom to chase edges).' },
    { key: 'fieldGridY', label: 'Field grid rows', min: 2, max: 10, step: 1,
      fmt: (v) => `${v | 0}`, hint: 'Rows of the shared illumination field.' },
    { key: 'fieldPrior', label: 'Field smoothing prior', min: 0, max: 0.6, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'Pull of field nodes toward the global mean where evidence is thin.' },
    { key: 'regionSize', label: 'Region tile size', min: 16, max: 96, step: 2,
      fmt: (v) => `${v | 0} px`, hint: 'Tile size for regional re-estimation; match it to the local lighting scale.' },
    { key: 'regionPrior', label: 'Region prior strength', min: 0, max: 0.6, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'How strongly tiles fall back to the shared field where local evidence is missing.' },
  ]},
  { group: 'Samples & temporal', items: [
    { key: 'sampleBoost', label: 'Sample trust', min: 1, max: 6, step: 0.25,
      fmt: (v) => v.toFixed(2), hint: 'How strongly a clean declared sample anchors the estimate at its location.' },
    { key: 'contamOnFrac', label: 'Contamination: screen fraction', min: 0.3, max: 0.95, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'A sample with fewer screen-like pixels than this is diagnosed as contaminated.' },
    { key: 'contamDev', label: 'Contamination: chroma deviation', min: 0.05, max: 0.5, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'A sample whose chroma direction disagrees with the clean-evidence estimate by more than this is contaminated (brightness differences like shadows are not contamination).' },
    { key: 'temporalRate', label: 'Temporal adoption rate', min: 0, max: 0.9, step: 0.05,
      fmt: (v) => v.toFixed(2), hint: '0 = re-fit every frame; higher keeps the previous frame\'s background estimate and adopts new evidence at this rate — collapsed automatically where the subject covers a sample.' },
  ]},
  { group: 'Refinement', items: [
    { key: 'refineStrength', label: 'Projection refine', min: 0, max: 1, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'How strongly the foreground/screen projection replaces the raw chroma alpha.' },
    { key: 'coreDistThreshold', label: 'Foreground core threshold', min: 0.05, max: 0.6, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'Ray distance above which a pixel seeds the local foreground colour model.' },
    { key: 'coreRadius', label: 'Core search radius', min: 8, max: 96, step: 2,
      fmt: (v) => `${v | 0} px`, hint: 'How far a pixel may look for a clean interior sample.' },
    { key: 'coreBlend', label: 'Core vs bilateral F', min: 0, max: 1, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: '1 = nearest clean-core colour only (best for thin structures); lower blends a spatial average that can smear edges.' },
    { key: 'mixMinFrac', label: 'Departure gate', min: 0.02, max: 0.3, step: 0.005,
      fmt: (v) => v.toFixed(3), hint: 'Minimum |C−B| as a fraction of |B| before projection trusts an alpha.' },
  ]},
  { group: 'Adjudication', items: [
    { key: 'strictScreenQ', label: 'Strict screen lock', min: 0, max: 0.12, step: 0.002,
      fmt: (v) => v.toFixed(3), hint: 'Pixels this close to the screen ray are forced transparent even if bright.' },
    { key: 'manualPriority', label: 'Manual brush priority', min: 0, max: 1, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'How hard brush strokes override the local key model.' },
  ]},
  { group: 'Despill (separate step)', items: [
    { key: 'spillStrength', label: 'Spill strength', min: 0, max: 1, step: 0.01,
      fmt: (v) => v.toFixed(2), hint: 'Fraction of excess screen-axis chroma removed from the foreground colour.' },
    { key: 'spillOffAxisLo', label: 'Protect hue below', min: 0, max: 0.4, step: 0.005,
      fmt: (v) => v.toFixed(3), hint: 'Off-axis hue ratio below this is treated as neutral → fully despilled.' },
    { key: 'spillOffAxisHi', label: 'Protect hue above', min: 0.02, max: 0.6, step: 0.005,
      fmt: (v) => v.toFixed(3), hint: 'Off-axis hue ratio above this is treated as a real colour → spared.' },
  ]},
];

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
function hexOfLinear(lin) {
  return '#' + lin.map((v) => Math.round(clamp01(linearToSrgb(v)) * 255).toString(16).padStart(2, '0')).join('');
}
function setStatus(text, busy) {
  const el = $('#status');
  el.textContent = text;
  el.classList.toggle('busy', !!busy);
}
function showError(msg) {
  const el = $('#errbox');
  el.textContent = msg; el.classList.add('on');
  clearTimeout(showError._t);
  showError._t = setTimeout(() => el.classList.remove('on'), 6000);
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------
async function loadScene(name) {
  state.name = name;
  const url = `/scene/${name}.png`;
  await loadImageUrl(url, `scene:${name}`);
  $$('#sceneGrid .btn').forEach((b) => b.classList.toggle('active', b.dataset.scene === name));
}

async function loadImageUrl(url, label) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  await new Promise((res, rej) => { img.onload = res; img.onerror = () => rej(new Error('image load failed')); img.src = url; });
  const w = img.naturalWidth, h = img.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, w, h);
  adoptImage(data, w, h);
  if (label) setStatus(label, false);
}

function adoptImage(imageData, w, h) {
  state.width = w; state.height = h;
  state.imageData = imageData;
  state.input8 = imageData.data;
  state.srcCanvas = document.createElement('canvas');
  state.srcCanvas.width = w; state.srcCanvas.height = h;
  state.srcCtx = state.srcCanvas.getContext('2d', { willReadFrequently: true });
  state.srcCtx.putImageData(imageData, 0, 0);
  state.manualMask = new Uint8Array(w * h);
  state.result = null;
  // Auto-pick a key from the border; the user can resample or eyedrop.
  state.keyLinear = pickAutoKey(state.input8, w, h);
  if (!state.keyLinear) {
    // Fall back to the historically dominant border colour even if unsaturated.
    state.keyLinear = [0.05, 0.36, 0.08];
  }
  // The main canvas matches source resolution; display size is CSS-driven.
  const cv = $('#cv');
  cv.width = w; cv.height = h;
  $('#cvDark').width = w; $('#cvDark').height = h;
  $('#cvLight').width = w; $('#cvLight').height = h;
  updateKeySwatch();
  updateMaskCount();
  markDirty(true);
}

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------
function markDirty(immediate) {
  state.dirty = true;
  clearTimeout(state.pendingTimer);
  const delay = immediate ? 0 : 180;
  state.pendingTimer = setTimeout(recompute, delay);
}

function recompute() {
  if (!state.input8) return;
  if (state.busy) { state.dirty = true; return; }
  state.busy = true;
  state.dirty = false;
  setStatus('keying…', true);
  // Yield a frame so the "keying…" label paints before the heavy sync work.
  requestAnimationFrame(() => {
    const t0 = performance.now();
    try {
      const params = buildKernelParams();
      const res = keyImage(state.input8, state.width, state.height, params);
      state.result = res;
      state.computeSeq++;
      // Chain the background estimate for the next frame (temporal update).
      if (params.temporalPrev !== null || (params.screenMode !== 'legacy' && params.temporalRate > 0)) {
        state.temporalPrev = res.screenState || null;
      }
      const ms = Math.round(performance.now() - t0);
      paintAll(ms);
      renderWarnings(res);
      renderStats(res, ms);
      renderSampleList(res);
      setStatus(`${state.width}×${state.height} · keyed in ${ms} ms · ${res.stats.visiblePixels} visible px`, false);
    } catch (e) {
      showError(e.message || String(e));
      setStatus('error: ' + (e.message || e), false);
    } finally {
      state.busy = false;
      if (state.dirty) markDirty(true);
    }
  });
}

function buildKernelParams() {
  const p = { ...state.params };
  // Slider-only keys need folding back into the kernel's real parameter names.
  p.screenQThresh = [Number(state.params.screenQThresh0), Number(state.params.screenQThresh1)];
  delete p.screenQThresh0; delete p.screenQThresh1;
  p.keyLinear = state.keyLinear;
  p.manualMask = state.manualMask && state.manualMask.some((v) => v) ? state.manualMask : null;
  // Non-uniform backdrop modelling: declared samples + temporal chain.
  p.screenSamples = p.screenMode === 'legacy' ? null
    : state.samples.map((s) => ({ ...s }));
  p.temporalPrev = (p.screenMode !== 'legacy' && p.temporalRate > 0) ? state.temporalPrev : null;
  return p;
}
// ---------------------------------------------------------------------------
// Backdrop model mode + background sample regions
// ---------------------------------------------------------------------------
const MODE_HINTS = {
  legacy: 'Original unified background: one smoothed screen colour shared across the frame.',
  uniform: 'Single re-fit: one background colour, but estimated with the absorption-guarded evidence pipeline.',
  field: 'Shared low-frequency illumination field: brightness varies on a bounded grid, chroma ratio is shared. Handles falloff, shadows and wrinkles.',
  regions: 'Regional re-estimation: the frame is partitioned into tiles; each tile re-estimates the background from local evidence with the shared field as prior.',
};

function setScreenMode(mode) {
  state.params.screenMode = mode;
  state.temporalPrev = null;   // state is per-mode; do not chain across modes
  $$('#modeSeg .btn').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  $('#modeHint').textContent = MODE_HINTS[mode];
  markDirty();
}
$$('#modeSeg .btn').forEach((b) => b.addEventListener('click', () => setScreenMode(b.dataset.mode)));

function setSampleDraw(on) {
  state.sampleDraw = on;
  if (on) { setEyedrop(false); setBrush('off'); }
  $('#sampleDrawBtn').classList.toggle('active', on);
  updateBrushCursor();
}
$('#sampleDrawBtn').addEventListener('click', () => setSampleDraw(!state.sampleDraw));

function deleteSample(index) {
  if (index < 0 || index >= state.samples.length) return;
  state.deletedSamples.push(state.samples[index]);
  state.samples.splice(index, 1);
  renderSampleList(state.result);
  markDirty();
}

function undoDeleteSample() {
  const s = state.deletedSamples.pop();
  if (!s) return;
  state.samples.push(s);
  renderSampleList(state.result);
  markDirty();
}
$('#undoSampleBtn').addEventListener('click', undoDeleteSample);

function renderSampleList(res) {
  const host = $('#sampleList');
  host.innerHTML = '';
  $('#undoSampleBtn').disabled = state.deletedSamples.length === 0;
  if (!state.samples.length) {
    host.innerHTML = '<div class="sample-empty-note">No background samples declared. The model relies on frame-wide evidence; declare samples where you know the backdrop is clean.</div>';
    return;
  }
  const diag = res && res.screenDiag;
  for (let i = 0; i < state.samples.length; i++) {
    const d = diag && diag.find((dd) => dd.index === i);
    const status = d ? d.status : 'pending';
    const row = document.createElement('div');
    row.className = `sample-row ${status}`;
    const meta = d
      ? `screen ${(d.onFrac * 100).toFixed(0)}% · dev ${(d.deviation * 100).toFixed(0)}% · ${status}`
      : 'not yet evaluated';
    row.innerHTML = `<span class="dot"></span><span>#${i + 1}</span><span class="meta">${meta}</span>`;
    const del = document.createElement('button');
    del.className = 'btn small del';
    del.textContent = 'Delete';
    del.title = 'Remove this sample — only the region it supported is re-fitted';
    del.addEventListener('click', () => deleteSample(i));
    row.appendChild(del);
    host.appendChild(row);
  }
}

// ---------------------------------------------------------------------------
// Video sequences (temporal update)
// ---------------------------------------------------------------------------
const SEQUENCES = [
  ['walk', 'Walk (subject crosses a sample)'],
];

function buildSequenceButtons() {
  const host = $('#seqGrid');
  for (const [id, label] of SEQUENCES) {
    const b = document.createElement('button');
    b.className = 'btn small';
    b.dataset.seq = id;
    b.textContent = label;
    b.addEventListener('click', () => loadSequence(id));
    host.appendChild(b);
  }
}

async function loadSequence(name) {
  stopPlayback();
  const meta = await (await fetch(`/seq/${name}/meta.json`)).json();
  state.sequence = { name, frames: meta.frames, frame: 0, sampleRegion: meta.sampleRegion };
  state.temporalPrev = null;
  // A sequence is a temporal problem: make sure the temporal update is
  // actually engaged (field model + adoption rate) so the guard is visible.
  if (state.params.screenMode === 'legacy') setScreenMode('field');
  if (state.params.temporalRate === 0) setParamSlider('temporalRate', 0.35);
  // Seed the declared sample the sequence was designed around, if the user
  // has not declared any yet — the temporal guard is exercised through it.
  if (!state.samples.length && meta.sampleRegion) {
    state.samples = [{ ...meta.sampleRegion }];
  }
  $('#seqControls').style.display = '';
  $('#seqOffRow').style.display = '';
  $$('#seqGrid .btn').forEach((b) => b.classList.toggle('active', b.dataset.seq === name));
  await loadSeqFrame(0);
}

async function loadSeqFrame(t) {
  const seq = state.sequence;
  if (!seq) return;
  seq.frame = Math.max(0, Math.min(seq.frames - 1, t));
  const keepKey = state.keyLinear;   // one key for the whole shot, not per-frame
  await loadImageUrl(`/seq/${seq.name}/${seq.frame}.png`, null);
  state.keyLinear = keepKey;
  updateKeySwatch();
  $('#seqLabel').textContent = `frame ${seq.frame + 1}/${seq.frames}`;
  markDirty(true);
}

function leaveSequence() {
  stopPlayback();
  state.sequence = null;
  state.temporalPrev = null;
  $('#seqControls').style.display = 'none';
  $('#seqOffRow').style.display = 'none';
  $$('#seqGrid .btn').forEach((b) => b.classList.remove('active'));
  const still = SCENES.some(([id]) => id === state.name) ? state.name : 'presenter';
  loadScene(still);
}

function stopPlayback() {
  if (state.playing) {
    clearInterval(state.playing);
    state.playing = null;
    $('#seqPlay').textContent = 'Play';
  }
}

$('#seqPrev').addEventListener('click', () => state.sequence && loadSeqFrame(state.sequence.frame - 1));
$('#seqNext').addEventListener('click', () => state.sequence && loadSeqFrame(state.sequence.frame + 1));
$('#seqOff').addEventListener('click', leaveSequence);
$('#seqPlay').addEventListener('click', () => {
  if (!state.sequence) return;
  if (state.playing) { stopPlayback(); return; }
  $('#seqPlay').textContent = 'Pause';
  state.playing = setInterval(() => {
    if (!state.sequence) return stopPlayback();
    const next = state.sequence.frame + 1;
    if (next >= state.sequence.frames) return stopPlayback();
    loadSeqFrame(next);
  }, 700);
});

// ---------------------------------------------------------------------------
// Painting
// ---------------------------------------------------------------------------

const paintCache = { input: null };

function paintAll(ms) {
  void ms;
  const res = state.result;
  if (!res) return;
  const w = state.width, h = state.height;

  const dual = state.view === 'dual';
  $('#single-wrap').classList.toggle('on', !dual);
  $('#dual-wrap').classList.toggle('on', dual);

  if (dual) {
    const dd = compositePixels(res, DARK);
    putImage($('#cvDark'), dd, w, h);
    const ll = compositePixels(res, LIGHT);
    putImage($('#cvLight'), ll, w, h);
    return;
  }

  const cv = $('#cv');
  const ctx = cv.getContext('2d');
  let rgba, cap;
  switch (state.view) {
    case 'input':
      rgba = state.input8; cap = 'Input (as captured) — pick a screen colour here';
      break;
    case 'alpha':
      rgba = alphaRgba(res); cap = 'Transparency (alpha) alone';
      break;
    case 'fg':
      rgba = correctedOverGrey(res); cap = 'Corrected colour alone (despilled foreground over grey)';
      break;
    case 'spill':
      rgba = spillOverlay(res); cap = 'Residue — red = removable spill left, green = edge residual, blue = spill removed';
      break;
    case 'field':
      rgba = screenFieldRgba(res); cap = 'Estimated backdrop field (the actual background the kernel subtracts)';
      break;
    case 'support':
      rgba = supportHeatRgba(res); cap = 'Backdrop confidence range — warm = supported by evidence, cold = extrapolated';
      break;
    case 'comp':
    default: {
      const bg = resolveBg();
      if (bg.checker) {
        paintCompositeChecker(res);
        state.lastPaint = { kind: 'comp', imageData: null };
        $('#cap').textContent = bg.name;
        drawSampleOverlay(cv.getContext('2d'));
        updateBrushCursor();
        return;
      }
      rgba = compositePixels(res, bg.rgb);
      cap = bg.name;
      break;
    }
  }
  const img = new ImageData(new Uint8ClampedArray(rgba), w, h);
  ctx.putImageData(img, 0, 0);
  state.lastPaint = { kind: state.view, imageData: img };
  $('#cap').textContent = cap;
  drawSampleOverlay(ctx);
  updateBrushCursor();
}

// The estimated background field B(x) the kernel actually subtracts, in sRGB.
function screenFieldRgba(res) {
  const { width: w, height: h, screen } = res;
  const n = w * h;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = Math.round(clamp01(linearToSrgb(screen.r[i * 3])) * 255);
    out[i * 4 + 1] = Math.round(clamp01(linearToSrgb(screen.r[i * 3 + 1])) * 255);
    out[i * 4 + 2] = Math.round(clamp01(linearToSrgb(screen.r[i * 3 + 2])) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

// Confidence range of the background estimate as a heat map.
const HEAT_STOPS = [
  [0.0, 32, 34, 74], [0.25, 35, 100, 170], [0.5, 47, 163, 107],
  [0.75, 224, 195, 65], [1.0, 214, 69, 69],
];
function supportHeatRgba(res) {
  const { width: w, height: h, screenSupport } = res;
  const n = w * h;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const t = clamp01(screenSupport[i]);
    let a = HEAT_STOPS[0], b = HEAT_STOPS[HEAT_STOPS.length - 1];
    for (let s = 0; s < HEAT_STOPS.length - 1; s++) {
      if (t >= HEAT_STOPS[s][0] && t <= HEAT_STOPS[s + 1][0]) { a = HEAT_STOPS[s]; b = HEAT_STOPS[s + 1]; break; }
    }
    const f = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
    out[i * 4] = Math.round(a[1] + (b[1] - a[1]) * f);
    out[i * 4 + 1] = Math.round(a[2] + (b[2] - a[2]) * f);
    out[i * 4 + 2] = Math.round(a[3] + (b[3] - a[3]) * f);
    out[i * 4 + 3] = 255;
  }
  return out;
}

// Declared background samples drawn over the current view, coloured by the
// kernel's own diagnosis (green = clean, red = contaminated, grey = empty).
function drawSampleOverlay(ctx) {
  if (!ctx) return;
  const diag = state.result && state.result.screenDiag;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.font = '11px ui-monospace, monospace';
  const drawRect = (s, color, label) => {
    ctx.strokeStyle = color;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(s.x + 0.5, s.y + 0.5, s.w, s.h);
    ctx.setLineDash([]);
    if (label) {
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = '#000a';
      ctx.fillRect(s.x + 2, s.y + 2, tw + 8, 15);
      ctx.fillStyle = color;
      ctx.fillText(label, s.x + 6, s.y + 13);
    }
  };
  for (let i = 0; i < state.samples.length; i++) {
    const s = state.samples[i];
    const d = diag && diag.find((dd) => dd.index === i);
    const color = !d ? '#98a0ad' : d.status === 'clean' ? '#7ee0a0' : d.status === 'contaminated' ? '#ff6b6b' : '#98a0ad';
    drawRect(s, color, `#${i + 1} ${d ? d.status : ''}`);
  }
  if (state.dragRect) drawRect(state.dragRect, '#57b6ff', 'new sample');
  ctx.restore();
}

function putImage(cv, rgba, w, h) {
  const ctx = cv.getContext('2d');
  const img = new ImageData(new Uint8ClampedArray(rgba), w, h);
  ctx.putImageData(img, 0, 0);
  state.lastPaint = { kind: 'dual', imageData: img };
}

function correctedOverGrey(res) {
  // Straight foreground colour over a neutral mid grey so the corrected colour
  // is visible on its own (not hidden behind a black background).
  const { width: w, height: h, alpha, fg } = res;
  const n = w * h;
  const out = new Uint8ClampedArray(n * 4);
  const gL = srgb8ToLinear(128);
  for (let i = 0; i < n; i++) {
    const a = alpha[i];
    const r = fg[i * 3] * a + gL * (1 - a);
    const g = fg[i * 3 + 1] * a + gL * (1 - a);
    const b = fg[i * 3 + 2] * a + gL * (1 - a);
    out[i * 4] = Math.round(clamp01(linearToSrgb(r)) * 255);
    out[i * 4 + 1] = Math.round(clamp01(linearToSrgb(g)) * 255);
    out[i * 4 + 2] = Math.round(clamp01(linearToSrgb(b)) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

function resolveBg() {
  if (state.bg.kind === 'dark') return { rgb: DARK, name: 'Composite on DARK [12,14,18]' };
  if (state.bg.kind === 'light') return { rgb: LIGHT, name: 'Composite on LIGHT [238,240,245]' };
  if (state.bg.kind === 'checker') {
    return { rgb: [90, 92, 96], name: 'Composite on CHECKER (transparency cue)', checker: true };
  }
  const c = state.bg.custom;
  return { rgb: c, name: `Composite on custom rgb(${c.join(',')})` };
}

// For "Composite" view with checker, draw the checker plate then blend the
// premultiplied foreground on top so transparency is obvious.
function paintCompositeChecker(res) {
  const w = state.width, h = state.height;
  const cv = $('#cv');
  const ctx = cv.getContext('2d');
  const cell = 12;
  const img = ctx.createImageData(w, h);
  const { alpha, fg } = res;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const a = alpha[i];
      const shade = ((Math.floor(x / cell) + Math.floor(y / cell)) & 1) ? 190 : 140;
      const bgL = srgb8ToLinear(shade);
      img.data[i * 4] = Math.round(clamp01(linearToSrgb(fg[i * 3] * a + bgL * (1 - a))) * 255);
      img.data[i * 4 + 1] = Math.round(clamp01(linearToSrgb(fg[i * 3 + 1] * a + bgL * (1 - a))) * 255);
      img.data[i * 4 + 2] = Math.round(clamp01(linearToSrgb(fg[i * 3 + 2] * a + bgL * (1 - a))) * 255);
      img.data[i * 4 + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ---------------------------------------------------------------------------
// Warnings (failure feedback back into the key)
// ---------------------------------------------------------------------------
const WARN_PARAM_HINT = {
  'key-black': ['eyedrop', null],
  'key-neutral': ['eyedrop', null],
  'low-screen-coverage': ['param', 'screenQThresh1'],
  'low-screen-model': ['param', 'screenSigma'],
  'alpha-low': ['param', 'alphaFloor'],
  'alpha-high': ['param', 'alphaFloor'],
  'manual-conflict': ['brush', 'protect'],
  'adjudicated-screen': ['brush', 'protect'],
  'spill-residual': ['param', 'spillStrength'],
  'sample-contaminated': ['samples', null],
  'sample-empty': ['samples', null],
  'field-low-support': ['samples', null],
  'temporal-adoption-low': ['param', 'temporalRate'],
};

function renderWarnings(res) {
  const box = $('#warnings');
  box.innerHTML = '';
  if (!res.warnings.length) {
    const d = document.createElement('div');
    d.className = 'ok-note';
    d.textContent = 'No kernel warnings. Alpha and spill are within the expected range.';
    box.appendChild(d);
    return;
  }
  for (const w of res.warnings) {
    const el = document.createElement('div');
    el.className = 'warn-item' + (w.code.includes('black') || w.code.includes('low') ? ' err' : '');
    el.innerHTML = `<div class="code">${w.code}</div><div>${w.text}</div>`;
    el.addEventListener('click', () => actOnWarning(w.code));
    box.appendChild(el);
  }
}

function actOnWarning(code) {
  const hint = WARN_PARAM_HINT[code];
  if (!hint) return;
  if (hint[0] === 'eyedrop') { setEyedrop(true); return; }
  if (hint[0] === 'brush') { setBrush(hint[1]); return; }
  if (hint[0] === 'samples') {
    $('#sampleList').scrollIntoView({ block: 'center', behavior: 'smooth' });
    setSampleDraw(true);
    return;
  }
  if (hint[0] === 'param') { flashParam(hint[1]); }
}

function flashParam(key) {
  const wrap = document.querySelector(`[data-param="${key}"]`);
  if (!wrap) return;
  wrap.classList.add('flash');
  wrap.scrollIntoView({ block: 'center', behavior: 'smooth' });
  setTimeout(() => wrap.classList.remove('flash'), 1600);
}

function renderStats(res, ms) {
  const s = res.stats;
  const rows = [
    ['screen support', (s.modelSupportMean * 100).toFixed(1) + '%'],
    ['coverage', (s.coverage * 100).toFixed(1) + '%'],
    ['mean alpha', s.meanAlpha.toFixed(3)],
    ['backdrop model', s.screenMode + (s.screenGrid ? ` ${s.screenGrid[0]}×${s.screenGrid[1]}` : '')],
    ['samples clean/contam.', `${s.samples.filter((d) => d.status === 'clean').length} / ${s.samplesContaminated}`],
    ['auto-protected px', String(s.autoProtected)],
    ['strict screen-locked', String(s.strictLocked)],
    ['brush applied px', String(s.manualApplied)],
    ['brush conflicts', String(s.manualConflicts)],
    ['spill in→out', `${s.spillBefore.toFixed(3)} → ${s.spillAfter.toFixed(3)}`],
    ['edge residual', s.edgeResidualMean.toFixed(4)],
    ['kernel time', `${ms} ms`],
  ];
  if (s.adoptionMean !== null && s.adoptionMean !== undefined) {
    rows.splice(5, 0, ['temporal adoption', s.adoptionMean.toFixed(2)]);
  }
  $('#stats').innerHTML = rows.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join('');
}

// ---------------------------------------------------------------------------
// Key colour UI
// ---------------------------------------------------------------------------
function updateKeySwatch() {
  if (!state.keyLinear) return;
  $('#keySwatch').style.background = hexOfLinear(state.keyLinear);
  $('#keyHex').textContent = 'linear ' + state.keyLinear.map((v) => v.toFixed(3)).join(', ');
}

function setEyedrop(on) {
  state.eyedrop = on;
  if (on) setSampleDraw(false);
  $('#eyedropBtn').classList.toggle('active', on);
  updateBrushCursor();
}

$('#eyedropBtn').addEventListener('click', () => setEyedrop(!state.eyedrop));
$('#autoKeyBtn').addEventListener('click', () => {
  const k = pickAutoKey(state.input8, state.width, state.height);
  if (k) { state.keyLinear = k; updateKeySwatch(); markDirty(); }
});

// ---------------------------------------------------------------------------
// Canvas interaction: eyedropper + brush
// ---------------------------------------------------------------------------
const cv = $('#cv');
function canvasPixel(ev, canvas) {
  const r = canvas.getBoundingClientRect();
  const x = Math.floor((ev.clientX - r.left) / r.width * state.width);
  const y = Math.floor((ev.clientY - r.top) / r.height * state.height);
  return { x: Math.max(0, Math.min(state.width - 1, x)), y: Math.max(0, Math.min(state.height - 1, y)) };
}
function updateBrushCursor() {
  cv.classList.toggle('eyedrop', state.eyedrop);
  cv.classList.toggle('brush', !state.eyedrop && !state.sampleDraw && state.brush !== 'off');
  cv.classList.toggle('sampling', state.sampleDraw);
}

cv.addEventListener('pointerdown', (ev) => {
  if (!state.input8) return;
  if (state.sampleDraw) {
    const { x, y } = canvasPixel(ev, cv);
    state.dragRect = { x, y, w: 0, h: 0 };
    cv.setPointerCapture(ev.pointerId);
    return;
  }
  if (state.eyedrop) {
    const { x, y } = canvasPixel(ev, cv);
    // Sample from the current painted image if it is the input; otherwise use
    // the true input pixels (never the result), so the key comes from source.
    const i = (y * state.width + x) * 4;
    const r = state.input8[i], g = state.input8[i + 1], b = state.input8[i + 2];
    state.keyLinear = [srgb8ToLinear(r), srgb8ToLinear(g), srgb8ToLinear(b)];
    updateKeySwatch();
    setEyedrop(false);
    markDirty();
    return;
  }
  if (state.brush !== 'off') {
    cv.setPointerCapture(ev.pointerId);
    paintBrush(ev);
    cv._brushing = true;
  }
});
cv.addEventListener('pointermove', (ev) => {
  if (state.dragRect) {
    const { x, y } = canvasPixel(ev, cv);
    const r = state.dragRect;
    r.w = x - r.x; r.h = y - r.y;
    paintAll(0);   // repaints current view + in-progress rect
    return;
  }
  if (cv._brushing) paintBrush(ev);
});
cv.addEventListener('pointerup', () => {
  if (state.dragRect) {
    let { x, y, w, h } = state.dragRect;
    state.dragRect = null;
    if (w < 0) { x += w; w = -w; }
    if (h < 0) { y += h; h = -h; }
    if (w >= 4 && h >= 4) {
      state.samples.push({ x, y, w, h });
      state.deletedSamples.length = 0;  // a new declaration invalidates the undo stack
      // Declaring a sample only makes sense with a non-uniform model —
      // switch out of the legacy unified mode so the sample takes effect.
      if (state.params.screenMode === 'legacy') setScreenMode('field');
      renderSampleList(state.result);
      markDirty();
    } else {
      paintAll(0);
    }
    return;
  }
  if (cv._brushing) { cv._brushing = false; markDirty(); }
});

function paintBrush(ev) {
  const { x, y } = canvasPixel(ev, cv);
  const r = state.brushSize;
  const val = state.brush === 'protect' ? 1 : state.brush === 'force' ? 2 : 0;
  const m = state.manualMask;
  const w = state.width, h = state.height;
  const r2 = r * r;
  for (let yy = Math.max(0, y - r); yy <= Math.min(h - 1, y + r); yy++) {
    for (let xx = Math.max(0, x - r); xx <= Math.min(w - 1, x + r); xx++) {
      const dx = xx - x, dy = yy - y;
      if (dx * dx + dy * dy <= r2) m[yy * w + xx] = val;
    }
  }
  updateMaskCount();
  // Live preview: overlay brush strokes subtly on the current view.
  paintBrushOverlay(x, y, r, val);
}

function paintBrushOverlay(x, y, r, val) {
  if (state.view === 'dual') return;
  const ctx = cv.getContext('2d');
  ctx.save();
  ctx.globalAlpha = 0.55;
  ctx.beginPath();
  const sx = cv.width / state.width, sy = cv.height / state.height;
  ctx.arc(x * sx, y * sy, r * sx, 0, Math.PI * 2);
  ctx.strokeStyle = val === 1 ? '#7ee0a0' : val === 2 ? '#ff6b6b' : '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function updateMaskCount() {
  let c = 0;
  if (state.manualMask) for (let i = 0; i < state.manualMask.length; i++) if (state.manualMask[i]) c++;
  $('#maskCount').textContent = c ? `${c} px marked` : 'no strokes';
}

$('#clearMaskBtn').addEventListener('click', () => {
  if (state.manualMask) state.manualMask.fill(0);
  updateMaskCount();
  markDirty();
});

// ---------------------------------------------------------------------------
// Brush / view / bg button groups
// ---------------------------------------------------------------------------
function setBrush(v) {
  state.brush = v;
  if (v !== 'off') setSampleDraw(false);
  $$('#brushSeg .btn').forEach((b) => b.classList.toggle('active', b.dataset.brush === v));
  updateBrushCursor();
}
function setView(v) {
  state.view = v;
  $$('#viewSeg .btn').forEach((b) => b.classList.toggle('active', b.dataset.view === v));
  paintAll(0);
}
function setBg(v) {
  state.bg.kind = v;
  $$('#bgSeg .btn').forEach((b) => b.classList.toggle('active', b.dataset.bg === v));
  paintAll(0);
}

$$('#brushSeg .btn').forEach((b) => b.addEventListener('click', () => setBrush(b.dataset.brush)));
$$('#viewSeg .btn').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
$$('#bgSeg .btn').forEach((b) => b.addEventListener('click', () => setBg(b.dataset.bg)));
$('#bgColor').addEventListener('input', (e) => {
  const h = e.target.value;
  state.bg.kind = 'custom';
  state.bg.custom = [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  $$('#bgSeg .btn').forEach((b) => b.classList.remove('active'));
  paintAll(0);
});

$('#brushSize').addEventListener('input', (e) => {
  state.brushSize = Number(e.target.value);
  $('#brushSizeVal').textContent = `${state.brushSize} px`;
});

$('#recomputeBtn').addEventListener('click', () => markDirty(true));

// ---------------------------------------------------------------------------
// Parameter panel
// ---------------------------------------------------------------------------
// Set a slider-backed parameter from code (sequence auto-setup, tests) and
// reflect it in the panel.
function setParamSlider(key, value) {
  state.params[key] = value;
  const wrap = document.querySelector(`[data-param="${key}"]`);
  if (wrap) {
    const input = wrap.querySelector('input');
    input.value = String(value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

function buildParamPanel() {
  const host = $('#paramGroups');
  for (const g of PARAM_SPEC) {
    const sec = document.createElement('div');
    sec.style.marginBottom = '10px';
    const h = document.createElement('div');
    h.style.cssText = 'font-size:11.5px;color:var(--muted);margin:6px 0 2px;font-weight:600;';
    h.textContent = g.group;
    sec.appendChild(h);
    for (const it of g.items) {
      const wrap = document.createElement('div');
      wrap.className = 'slider';
      wrap.dataset.param = it.key;
      let cur = state.params[it.key];
      if (it.key === 'screenQThresh0') cur = state.params.screenQThresh[0];
      if (it.key === 'screenQThresh1') cur = state.params.screenQThresh[1];
      wrap.innerHTML = `
        <div class="lab"><span class="name">${it.label}</span><span class="val" data-val></span></div>
        <input type="range" min="${it.min}" max="${it.max}" step="${it.step}" value="${cur}">
        <div class="hint">${it.hint}</div>`;
      const input = wrap.querySelector('input');
      const valEl = wrap.querySelector('[data-val]');
      valEl.textContent = it.fmt(Number(input.value));
      input.addEventListener('input', () => {
        const v = Number(input.value);
        valEl.textContent = it.fmt(v);
        state.params[it.key] = v;
        markDirty();  // debounced; kernel recompute follows
      });
      sec.appendChild(wrap);
    }
    host.appendChild(sec);
  }
}

// ---------------------------------------------------------------------------
// Scene buttons + file loading
// ---------------------------------------------------------------------------
const SCENES = [
  ['presenter', 'Presenter (green + green objects)'],
  ['blueProduct', 'Blue product (blue object)'],
  ['fineDetail', 'Fine detail (struts)'],
  ['despill', 'Despill stress'],
];
function buildSceneButtons() {
  const host = $('#sceneGrid');
  for (const [id, label] of SCENES) {
    const b = document.createElement('button');
    b.className = 'btn';
    b.dataset.scene = id;
    b.textContent = label;
    b.addEventListener('click', () => loadScene(id).then(recompute));
    host.appendChild(b);
  }
}

$('#fileInput').addEventListener('change', async (e) => {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  try {
    await loadImageUrl(url, `file:${f.name}`);
    setStatus(`${f.name} · ${state.width}×${state.height}`, false);
    recompute();
  } catch (err) {
    showError('Could not load that image: ' + err.message);
  } finally {
    URL.revokeObjectURL(url);
  }
});

// ---------------------------------------------------------------------------
// Export real pixels
// ---------------------------------------------------------------------------
function download(canvas, filename) {
  canvas.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, 'image/png');
}
function rgbaToCanvas(rgba, w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
  return c;
}
function exportRgba(rgba, kind) {
  const c = rgbaToCanvas(rgba, state.width, state.height);
  download(c, `key-${state.name}-${kind}.png`);
}
$('#exportStraight').addEventListener('click', () => {
  if (!state.result) return showError('Key an image first.');
  exportRgba(straightRgba(state.result), 'straight');
});
$('#exportAlpha').addEventListener('click', () => {
  if (!state.result) return showError('Key an image first.');
  exportRgba(alphaRgba(state.result), 'alpha');
});
$('#exportComp').addEventListener('click', () => {
  if (!state.result) return showError('Key an image first.');
  exportRgba(compositePixels(state.result, resolveBg().rgb), 'composite');
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function boot() {
  buildSceneButtons();
  buildSequenceButtons();
  buildParamPanel();
  setView('comp');
  setBg('dark');
  setBrush('off');
  setScreenMode(state.params.screenMode || 'legacy');
  renderSampleList(null);
  loadScene('presenter').then(recompute).catch((e) => {
    showError('Could not load the demo scene. Run `node serve.mjs` and open http://localhost:8080/ — ' + e.message);
    setStatus('no image', false);
  });
}
boot();

// Expose a tiny hook for automated UI verification (no effect on normal use).
window.__keyStudio = {
  state, recompute, loadScene, setView, setBg, setBrush,
  setScreenMode, setSampleDraw, deleteSample, undoDeleteSample,
  loadSequence, loadSeqFrame, leaveSequence,
};
