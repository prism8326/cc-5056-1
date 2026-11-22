// Minimal dependency-free static server for the chroma-key studio UI.
//
//   node serve.mjs            # http://localhost:8080
//   PORT=3000 node serve.mjs
//
// It serves the repository root (so ES modules under ./src and ./ui load) and
// additionally synthesises the test scenes on demand at /scene/<name>.png.
// Those scenes are generated from tools/scenes.js exactly like the evaluation
// harness does — the UI receives only the *mixed* input image, never a mask.

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildScene, buildSequenceFrame, encodePng, SCENES, SEQUENCES } from './tools/scenes.js';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 8080;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
};

const sceneCache = new Map();
function scenePng(name) {
  if (sceneCache.has(name)) return sceneCache.get(name);
  if (!SCENES[name]) return null;
  const gt = buildScene(name);
  const buf = encodePng(gt.input8, gt.width, gt.height);
  sceneCache.set(name, buf);
  return buf;
}

function sequenceMeta(name) {
  const seq = SEQUENCES[name];
  if (!seq) return null;
  return { frames: seq.frames, width: seq.width, height: seq.height, sampleRegion: seq.sampleRegion };
}

function sequencePng(name, frame) {
  const key = `${name}/${frame}`;
  if (sceneCache.has(key)) return sceneCache.get(key);
  const seq = SEQUENCES[name];
  if (!seq || frame < 0 || frame >= seq.frames) return null;
  const gt = buildSequenceFrame(name, frame);
  const buf = encodePng(gt.input8, gt.width, gt.height);
  sceneCache.set(key, buf);
  return buf;
}

const server = http.createServer((req, res) => {
  let urlPath;
  try {
    urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    res.writeHead(400); return res.end('bad request');
  }

  const sceneMatch = urlPath.match(/^\/scene\/([A-Za-z0-9_-]+)\.png$/);
  if (sceneMatch) {
    const buf = scenePng(sceneMatch[1]);
    if (!buf) { res.writeHead(404); return res.end('unknown scene'); }
    res.writeHead(200, { 'Content-Type': 'image/png', 'Cache-Control': 'no-cache' });
    return res.end(buf);
  }

  const seqMetaMatch = urlPath.match(/^\/seq\/([A-Za-z0-9_-]+)\/meta\.json$/);
  if (seqMetaMatch) {
    const meta = sequenceMeta(seqMetaMatch[1]);
    if (!meta) { res.writeHead(404); return res.end('unknown sequence'); }
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' });
    return res.end(JSON.stringify(meta));
  }

  const seqMatch = urlPath.match(/^\/seq\/([A-Za-z0-9_-]+)\/(\d+)\.png$/);
  if (seqMatch) {
    const buf = sequencePng(seqMatch[1], Number(seqMatch[2]));
    if (!buf) { res.writeHead(404); return res.end('unknown sequence frame'); }
    res.writeHead(200, { 'Content-Type': 'image/png', 'Cache-Control': 'no-cache' });
    return res.end(buf);
  }

  if (urlPath === '/') urlPath = '/index.html';

  // Resolve and confine to ROOT.
  const filePath = path.join(ROOT, urlPath);
  if (!filePath.startsWith(ROOT + path.sep)) { res.writeHead(403); return res.end('forbidden'); }

  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404, { 'Content-Type': 'text/plain' }); return res.end('not found'); }
    res.writeHead(200, {
      'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Chroma-key studio running at http://localhost:${PORT}/`);
  console.log('Press Ctrl+C to stop.');
});
