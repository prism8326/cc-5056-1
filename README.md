# Chroma-Key Studio

A studio background-replacement tool for green/blue-screen footage. It estimates
per-pixel foreground **transparency**, recovers the foreground **colour** with the
screen's spill light removed, and composites the result onto any new background —
checked on both a dark and a light background, not just the flattering one.

The interactive canvas and the ground-truth evaluation share **one kernel**
(`src/keyer.js`). The UI controls are wired straight into that kernel; nothing is
pre-masked, and every slider re-runs the real key on the real pixels.

---

## Run the interface

```bash
npm install        # once (pngjs, jpeg-js for the test images; playwright for tests)
node serve.mjs     # or: npm start
```

Then open **http://localhost:8080/** in a browser. Pick a test scene from the
left, or **Load image…** to open your own green/blue-screen frame.

`serve.mjs` is a small dependency-free static server. It serves the repo root and
generates the four synthetic test scenes on demand at `/scene/<name>.png` — the
browser only ever receives the *mixed* input image, never a mask.

The UI is a set of ES modules (`ui/app.js` → `src/keyer.js` → `src/color.js`), so
it must be served over HTTP (opening `index.html` from `file://` will not load the
modules). Any static server works; `serve.mjs` is provided so you don't need one.

### What you can do in the UI

| Control | What it does |
|---|---|
| **Eyedropper** | Sample the screen colour from the input image (linear RGB internally). |
| **Auto (border)** | Pick the dominant saturated border colour automatically. |
| **Backdrop model** | Choose the background estimate: *Unified (original)* — the original single smoothed colour; *Single re-fit* — one colour from the guarded evidence pipeline; *Illumination field* — a shared low-frequency brightness field on a bounded grid; *Regional re-fit* — per-tile re-estimation with the field as prior. |
| **Draw background sample** | Drag a rectangle over a region you *know* is clean backdrop. The kernel cross-checks it against clean evidence, down-weights it if contaminated, and anchors the estimate there if clean. Deleting a sample re-fits only the region it supported; **Undo delete** restores it. |
| **View** | *Input*, *Alpha* (transparency alone), *Corrected colour* (despilled foreground over grey), *Composite*, *Spill / residue*, *Backdrop field* (the actual estimated background the kernel subtracts), *Confidence* (where that estimate is supported by evidence vs extrapolated), and *Dark ⇄ Light* side-by-side. |
| **Video sequence** | Load the synthetic walk sequence and step/play frames. With a temporal adoption rate > 0 the background estimate is carried across frames; where the subject covers a declared sample, adoption collapses so the subject is not learned as screen. |
| **New background** | Dark `[12,14,18]`, Light `[238,240,245]`, a checkerboard transparency cue, or any custom colour. |
| **Repair brush** | *Protect* forces foreground, *Force key* forces transparent, *Erase* clears marks. Overrides are adjudicated against the local key model and reported in Diagnostics. |
| **Parameters** | Every slider is bound to a real kernel parameter (grouped by stage). Moving one re-runs the key. |
| **Export** | Straight-alpha RGBA PNG (despilled colour + per-pixel alpha, composites anywhere), composite PNG, or alpha PNG — real pixels, not a preview. |
| **Diagnostics** | Kernel warnings (failure feedback) plus live stats. Click a warning to jump to the relevant control. |

The **Alpha** view is the transparency separated from colour; **Corrected colour**
is the colour separated from transparency; **Spill / residue** shows, per pixel,
removable spill still left (red), edge re-composite residual (green), and spill
that was actually removed (blue). Genuinely green/blue objects stay dark there
because their colour is not removable spill — which is how you can see that the
corrector did not eat the object.

---

## Why the two steps are separate

**Transparency separation** and **spill correction** are different problems and are
never collapsed into one:

1. **Matte (alpha).** How much of the pixel is foreground vs screen.
2. **Colour (despill).** The camera recorded `C = α·F + (1−α)·S`, where the real
   foreground `F` already includes key-coloured light bouncing onto the subject.
   Even at `α = 1` (fully opaque) `F` is tinted by the screen; that tint must be
   removed from `F` *without* touching `α`.

A correct tool must therefore **not** make every near-screen-colour pixel
transparent, and must **not** simply pull down the green/blue channel — both would
delete genuinely green/blue objects. This tool keys on geometry (distance from the
screen's *chroma ray*) for the matte, and only then removes screen-axis chroma from
the foreground colour, gated by how much off-axis hue the object genuinely carries.

## Colour space and premultiplication

- All maths is in **linear light**. 8-bit sRGB input is decoded once
  (`srgb8ToLinear`); output is re-encoded once (`linearToSrgb8`).
- An orthonormal basis is built around the sampled key colour:
  - `e0 = (1,1,1)/√3` — the neutral/brightness axis,
  - `e1` — the screen's own chroma direction (in the plane ⊥ `e0`),
  - `e2 = e0 × e1` — off-axis chroma.
  Multiplicative shading (lamp falloff, vignette, cast shadows) scales the screen's
  colour **without rotating it**, so coordinates on `e1`/`e2` are shading-invariant
  and the same thresholds work across a lit-and-shadowed cyclorama.
- The foreground/screen mixture is recovered by **projection**:
  `α = ⟨C − B, F − B⟩ / |F − B|²`, which is exact when `F` and `B` are known.
- Compositing uses the premultiplied form `C' = α·F + (1−α)·B_new`. The straight
  RGBA export keeps `(F, α)` so the result re-composites anywhere; the composite
  export is the flattened `C'`.

## Pipeline (stages in `src/keyer.js`)

1. **Screen model** — the local background colour `B(x)`. Four modes, all
   estimating the same quantity:
   - *Unified (original)* — confidence-weighted, spatially smoothed screen
     colour shared across the frame (normalized convolution).
   - *Single re-fit* — one global colour from the guarded evidence pipeline.
   - *Illumination field* — brightness `g(x)` on a bounded bilinear grid
     (≤ 12×10 nodes, fitted in log space so multiplicative shading is
     low-frequency), one shared chroma ratio for the whole screen.
   - *Regional re-fit* — the frame is partitioned into tiles; each tile
     re-estimates `(w, p, q)` from local evidence with the field as prior.
   The non-uniform modes share the machinery in `src/screenfield.js`:
   - **Absorption guard** — only pixels whose chroma *direction* matches the
     screen (`q/w ≈ 0`) build the background model. Shading scales a screen
     pixel but never rotates its hue, so genuine screen survives at any
     brightness while a same-hue foreground object contributes nothing. A
     border-connectivity gate and a robust (IRLS) refit add further
     protection. The model's degrees of freedom are capped, so it cannot
     grow extra freedom to chase edges.
   - **Declared samples** — user-marked reliable background regions. Each is
     diagnosed (screen-like fraction + chroma deviation against the
     clean-evidence estimate — brightness differences such as shadows are
     *not* contamination). Clean samples anchor the estimate locally as
     direct observations; contaminated ones are down-weighted and reported.
     Deleting a sample re-fits exactly the tiles/nodes it supported.
   - **Temporal update** — for video, the previous frame's estimate is kept
     and new evidence is adopted at a configurable rate; where a declared
     sample stops looking like screen (the subject walked into it), adoption
     collapses so the subject is not learned as background.
2. **Initial alpha** — brightness-free distance of each pixel from the screen's
   chroma ray. Screen pixels sit on the ray; real objects leave it.
3. **Foreground model + projection** — the clean foreground colour `F` is taken
   from the nearest confidently-opaque *core* (rim-eroded so edge mixes cannot seed
   themselves), then alpha is the exact projection of `C` onto `B→F`, gated by how
   far `C` has actually departed from `B`.
4. **Adjudication** — where the screen brightness overlaps a foreground-like colour,
   a strict screen-ray lock and a residual-gated, interiority-gated rescue decide
   between the local model and manual protection; the repair brush overrides last
   and conflicts are reported.
5. **Despill (F only)** — removes positive screen-axis chroma from `F`, sparing
   genuine off-axis hue (green scarf, blue product). `α` is never modified here.
6. **Diagnostics** — edge re-composite residual and spill before/after, surfaced as
   warnings in the UI.

## Ground-truth evaluation

The test scenes are generated with **known** foreground, screen, and semi-transparent
edges (signed-distance coverage, not a binary mask), plus shadows, noise, and JPEG
compression. The evaluator compares the recovered result against the *clean* truth
(subject albedo with spill removed) — not against the mixed input.

```bash
node tools/evaluate.mjs    # or: npm run evaluate
```

It reports, per scene: alpha MAE, edge alpha vs a JPEG "oracle" floor, opaque and
screen alpha leak, premultiplied edge colour, composite error on a **dark** and a
**light** background, spill removal on genuinely-neutral pixels, hue preservation on
genuine green/blue objects, and thin-structure retention. Artefacts (input, gt-alpha,
alpha, straight, on-dark, on-light, spill overlay, alpha-diff) and `report.json` land
in `test-output/`.

The `blueProduct` scene is the deliberate hard case: a saturated blue object under
blue light. Its spill is collinear with — and only ~15% of the magnitude of — its
own hue, so it is *inseparable* from the product's colour; the evaluator records
that honestly rather than demanding a removal that would destroy the object.

### Non-uniform backdrop evaluation

```bash
node tools/evaluate-screenfield.mjs    # or: npm run evaluate:screenfield
```

A second harness compares the fixed-colour key against the illumination-field and
regional models **under the same ground truth**, on a shadowed screen (falloff,
cast shadow bands, a screen-hue foreground panel) and a wrinkled screen. It checks:
screen-region alpha leak and MAE vs the fixed key, composite error on dark/light,
edge foreground colour, absorption of the same-hue panel (must stay low), sample
diagnostics (contaminated region flagged, clean ones pass, cross-check separation),
strictly local re-fit when a sample is deleted, the field's bounded degrees of
freedom, and temporal stability on the 8-frame walk sequence (adoption collapse
while the subject crosses a sample, no subject absorption, lower per-frame jitter
than independent refits). Artefacts and `screenfield-report.json` land in
`test-output/`.

```bash
node tools/ui-check.mjs    # or: npm run test:ui
```

Drives the real page in headless Chromium and asserts the controls actually change
the kernel output (eyedropper, threshold slider, despill slider leaving alpha
untouched, brush override, dark/light composites, PNG export, warning feedback).

> In a minimal container Chromium may be missing system libraries. The UI check
> looks for a local prefix in `$CHROMIUM_EXTRA_LIBS` (default
> `/tmp/chroot/usr/lib/aarch64-linux-gnu`) so the browser can start without root.
