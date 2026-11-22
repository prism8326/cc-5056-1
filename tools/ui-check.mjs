// Automated UI verification for Chroma-Key Studio.
//
// Starts the static server, drives the real page in headless Chromium, and
// asserts that the controls actually change the kernel output (not just the
// DOM): the eyedropper changes the key colour, a threshold slider changes the
// alpha field, the brush changes adjudicated alpha, both bright/dark
// composites render, and a real PNG is produced on export.
//
// Run: node tools/ui-check.mjs
// Screenshots land in test-output/ui-*.png

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const PORT = 8091;
const BASE = `http://localhost:${PORT}`;
const OUT = path.resolve('test-output');
fs.mkdirSync(OUT, { recursive: true });

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function seqOf(page) {
  return page.evaluate(() => window.__keyStudio.state.computeSeq);
}
async function waitForNewKernel(page, minSeq, timeout = 20000) {
  await page.waitForFunction(
    (m) => window.__keyStudio.state.computeSeq >= m,
    minSeq, { timeout },
  );
}

async function alphaSignature(page) {
  return page.evaluate(() => {
    const st = window.__keyStudio.state;
    if (!st.result) return null;
    const a = st.result.alpha;
    let s = 0, n = 0;
    for (let i = 0; i < a.length; i += 97) { s += a[i]; n++; }
    return { mean: s / n, visible: st.result.stats.visiblePixels, spill: st.result.stats.spillAfter };
  });
}

async function main() {
  const server = spawn(process.execPath, ['serve.mjs'], {
    cwd: path.resolve('.'), env: { ...process.env, PORT: String(PORT) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logs = [];
  server.stdout.on('data', (d) => logs.push(d.toString()));
  server.stderr.on('data', (d) => logs.push(d.toString()));
  await wait(600);

  const checks = [];
  const add = (ok, label, detail = '') => { checks.push({ ok, label, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? '  (' + detail + ')' : ''}`); };

  let browser;
  try {
    // Some minimal containers lack the system libs Chromium links against.
    // If a local prefix was extracted (see README), expose it to the browser.
    const extraLib = process.env.CHROMIUM_EXTRA_LIBS || '/tmp/chroot/usr/lib/aarch64-linux-gnu:/tmp/chroot/lib/aarch64-linux-gnu';
    const launchEnv = { ...process.env };
    if (fs.existsSync(extraLib.split(':')[0])) {
      launchEnv.LD_LIBRARY_PATH = [extraLib, process.env.LD_LIBRARY_PATH].filter(Boolean).join(':');
    }
    browser = await chromium.launch({ args: ['--no-sandbox'], env: launchEnv });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const consoleErrors = [];
    page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
    page.on('pageerror', (e) => consoleErrors.push('pageerror: ' + e.message));

    await page.goto(BASE + '/', { waitUntil: 'load' });
    await waitForNewKernel(page, 1);
    add(true, 'page loads and kernel runs');

    // --- initial state -----------------------------------------------------
    const init = await alphaSignature(page);
    add(init && init.visible > 1000, 'foreground detected in demo scene', init && `visible=${init.visible}`);
    await page.screenshot({ path: path.join(OUT, 'ui-01-composite-dark.png') });

    // --- views render distinct pixels -------------------------------------
    const grab = async () => page.evaluate(() => {
      const cv = document.querySelector('#cv');
      const d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
      let h = 0; for (let i = 0; i < d.length; i += 997) h = (h * 31 + d[i]) >>> 0;
      return h;
    });
    await page.click('#viewSeg [data-view="alpha"]'); await wait(120);
    const hAlpha = await grab();
    await page.screenshot({ path: path.join(OUT, 'ui-02-alpha.png') });
    await page.click('#viewSeg [data-view="fg"]'); await wait(120);
    const hFg = await grab();
    await page.screenshot({ path: path.join(OUT, 'ui-03-corrected-colour.png') });
    await page.click('#viewSeg [data-view="spill"]'); await wait(120);
    const hSpill = await grab();
    await page.screenshot({ path: path.join(OUT, 'ui-04-spill.png') });
    add(hAlpha !== hFg && hFg !== hSpill, 'alpha / colour / spill views differ', `${hAlpha},${hFg},${hSpill}`);

    // --- dual bright/dark view --------------------------------------------
    await page.click('#viewSeg [data-view="dual"]'); await wait(150);
    const dualShown = await page.evaluate(() => document.querySelector('#dual-wrap').classList.contains('on'));
    const dualDiff = await page.evaluate(() => {
      const a = document.querySelector('#cvDark').getContext('2d').getImageData(0, 0, 480, 360).data;
      const b = document.querySelector('#cvLight').getContext('2d').getImageData(0, 0, 480, 360).data;
      let s = 0; for (let i = 0; i < a.length; i += 400) s += Math.abs(a[i] - b[i]);
      return s;
    });
    add(dualShown && dualDiff > 0, 'dual dark/light composites render', `diff=${dualDiff}`);
    await page.screenshot({ path: path.join(OUT, 'ui-05-dual-dark-light.png') });

    // --- eyedropper changes the key ---------------------------------------
    await page.click('#viewSeg [data-view="comp"]'); await wait(120);
    const keyBefore = await page.evaluate(() => document.querySelector('#keyHex').textContent);
    let seq = await seqOf(page);
    await page.click('#eyedropBtn');
    const cvBox = await page.locator('#cv').boundingBox();
    // click near a screen-only corner (top-left area) of the image
    await page.mouse.click(cvBox.x + cvBox.width * 0.08, cvBox.y + cvBox.height * 0.12);
    await waitForNewKernel(page, seq + 1);
    const keyAfter = await page.evaluate(() => document.querySelector('#keyHex').textContent);
    add(keyBefore !== keyAfter, 'eyedropper samples a new key colour', `${keyBefore} -> ${keyAfter}`);

    // --- threshold slider changes the kernel output -----------------------
    const beforeSlider = await alphaSignature(page);
    seq = await seqOf(page);
    await page.evaluate(() => {
      const rng = document.querySelector('[data-param="alphaFloor"] input');
      rng.value = '0.20';
      rng.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForNewKernel(page, seq + 1);
    const afterSlider = await alphaSignature(page);
    add(Math.abs(afterSlider.mean - beforeSlider.mean) > 1e-4,
      'alpha threshold slider drives the kernel',
      `${beforeSlider.mean.toFixed(4)} -> ${afterSlider.mean.toFixed(4)}`);
    // restore
    seq = await seqOf(page);
    await page.evaluate(() => {
      const rng = document.querySelector('[data-param="alphaFloor"] input');
      rng.value = '0.04';
      rng.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForNewKernel(page, seq + 1);

    // --- despill slider affects corrected colour (F only) -----------------
    await page.click('#viewSeg [data-view="fg"]'); await wait(100);
    // Snapshot the corrected-colour canvas, drop spill strength to 0, and
    // measure how many visible pixels' colour actually changed.
    const fgHash = await grab();
    seq = await seqOf(page);
    await page.evaluate(() => {
      const rng = document.querySelector('[data-param="spillStrength"] input');
      rng.value = '0.0';
      rng.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForNewKernel(page, seq + 1);
    const fgHash0 = await grab();
    const alphaAt0 = await alphaSignature(page);
    const spillPixelChange = await page.evaluate(() => {
      const st = window.__keyStudio.state;
      return st.result.stats.spillAfter;
    });
    seq = await seqOf(page);
    await page.evaluate(() => {
      const rng = document.querySelector('[data-param="spillStrength"] input');
      rng.value = '0.8';
      rng.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForNewKernel(page, seq + 1);
    add(fgHash !== fgHash0, 'despill strength changes corrected colour (alpha untouched)',
      `corrected-colour canvas differs; spillAfter@0=${spillPixelChange.toFixed(4)}`);
    // Check alpha was not modified by the despill pass.
    const alphaAt1 = await alphaSignature(page);
    add(Math.abs(alphaAt1.mean - alphaAt0.mean) < 1e-9, 'despill leaves alpha unchanged (separation)',
      `mean alpha ${alphaAt0.mean.toFixed(6)} vs ${alphaAt1.mean.toFixed(6)}`);

    // --- manual protection brush overrides the key model ------------------
    // Force-key a patch of foreground, then confirm adjudicated alpha drops
    // and the kernel reports brush pixels applied.
    await page.click('#viewSeg [data-view="comp"]'); await wait(100);
    const alphaPatch = await page.evaluate(() => {
      const st = window.__keyStudio.state; const a = st.result.alpha;
      const x = Math.round(st.width * 0.5), y = Math.round(st.height * 0.7);
      return a[y * st.width + x];
    });
    await page.click('#brushSeg [data-brush="force"]');
    await page.evaluate(() => { const r = document.querySelector('#brushSize'); r.value = '26'; r.dispatchEvent(new Event('input', { bubbles: true })); });
    const cb = await page.locator('#cv').boundingBox();
    seq = await seqOf(page);
    await page.mouse.move(cb.x + cb.width * 0.5, cb.y + cb.height * 0.7);
    await page.mouse.down();
    await page.mouse.move(cb.x + cb.width * 0.52, cb.y + cb.height * 0.71, { steps: 4 });
    await page.mouse.up();
    await waitForNewKernel(page, seq + 1);
    const brushStats = await page.evaluate(() => {
      const st = window.__keyStudio.state;
      const a = st.result.alpha;
      const x = Math.round(st.width * 0.5), y = Math.round(st.height * 0.7);
      let marks = 0; for (let i = 0; i < st.manualMask.length; i++) if (st.manualMask[i]) marks++;
      return { applied: st.result.stats.manualApplied, marks, patchAlpha: a[y * st.width + x] };
    });
    add(brushStats.applied > 0 && brushStats.patchAlpha < alphaPatch,
      'force-key brush overrides the kernel at the stroke',
      `px=${brushStats.applied}, marks=${brushStats.marks}, alpha ${alphaPatch.toFixed(2)} -> ${brushStats.patchAlpha.toFixed(2)}`);
    seq = await seqOf(page);
    await page.click('#clearMaskBtn'); await waitForNewKernel(page, seq + 1);

    // --- export produces a real PNG ---------------------------------------
    const dl = page.waitForEvent('download', { timeout: 8000 });
    await page.click('#exportStraight');
    const download = await dl;
    const tmp = path.join(OUT, 'ui-export-straight.png');
    await download.saveAs(tmp);
    const size = fs.statSync(tmp).size;
    const magic = fs.readFileSync(tmp).subarray(0, 8);
    const isPng = magic[0] === 0x89 && magic[1] === 0x50 && magic[2] === 0x4e && magic[3] === 0x47;
    add(isPng && size > 2000, 'straight RGBA PNG exports real pixels', `${size} bytes`);

    // --- composite bg selectors -------------------------------------------
    await page.click('#bgSeg [data-bg="light"]'); await wait(120);
    const hLight = await grab();
    await page.screenshot({ path: path.join(OUT, 'ui-06-composite-light.png') });
    await page.click('#bgSeg [data-bg="dark"]'); await wait(120);
    const hDark = await grab();
    add(hLight !== hDark, 'dark and light background selectors differ', `${hDark} vs ${hLight}`);

    // --- warnings surface failure feedback --------------------------------
    // Set an absurd alpha floor so almost everything keys out → warning.
    seq = await seqOf(page);
    await page.evaluate(() => {
      const rng = document.querySelector('[data-param="alphaFloor"] input');
      rng.value = '0.29';
      rng.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await waitForNewKernel(page, seq + 1);
    const warnCount = await page.evaluate(() => document.querySelectorAll('#warnings .warn-item').length);
    add(warnCount > 0, 'kernel failure feedback appears in the UI', `${warnCount} warning(s)`);
    await page.screenshot({ path: path.join(OUT, 'ui-07-warning-feedback.png') });

    // --- other scenes compute --------------------------------------------
    for (const sc of ['blueProduct', 'fineDetail', 'despill']) {
      seq = await seqOf(page);
      await page.evaluate((s) => window.__keyStudio.loadScene(s), sc);
      await waitForNewKernel(page, seq + 1);
      const sig = await alphaSignature(page);
      add(sig && sig.visible > 1000, `scene "${sc}" keys successfully`, `visible=${sig && sig.visible}`);
    }
    await page.screenshot({ path: path.join(OUT, 'ui-08-despill-scene.png') });

    add(consoleErrors.length === 0, 'no browser console errors', consoleErrors.slice(0, 3).join(' | '));
  } catch (e) {
    add(false, 'UI run completed', e.message);
    console.error(e);
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
  }

  const failed = checks.filter((c) => !c.ok).length;
  console.log(`\n${checks.length - failed}/${checks.length} UI checks passed`);
  if (logs.length) fs.writeFileSync(path.join(OUT, 'ui-server.log'), logs.join(''));
  process.exit(failed ? 1 : 0);
}

main();
