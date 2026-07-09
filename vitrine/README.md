# vitrine

The atelier, staged: a live, interactive demonstration of the Namer
(a miniature CLIP), the Sculptor (a miniature I-JEPA) and their
Handshake, running entirely in the visitor's browser through ONNX
Runtime Web. No backend, no server inference, no Python at runtime.

The research lives in the parent repository; this is the glass in
front of it. Every number on the page is read from the shipped results
JSONs or computed live by the exported models; nothing is hardcoded.

## Running it

```bash
# once, in the atelier repo root: export the crates
.venv/bin/python -m export.for_the_vitrine

# then, in vitrine/
npm install
npm test            # gates P and W, plus the probe arithmetic gate
npm run build       # gathers crates into public/ and exports to out/
npx serve out       # or any static file server
```

Deploy to Cloudflare Pages: build command `npm run build`, output
directory `out`. The `public/_headers` file sets immutable caching for
the models and the WASM runtime. No functions, no bindings.

## The gates

- **Gate P (parity)**: every exported ONNX model, run on 32 fixed
  seed-7 scenes, matches its PyTorch output within 1e-4. Measured:
  gaze 5.0e-6, tongue 1.3e-6, eye below 1e-4 veiled and unveiled,
  imagination 1.6e-6, grip and both bridges below 1e-4.
- **Gate W (world)**: PRNG parity with numpy's PCG64 was not attempted;
  the TypeScript world uses mulberry32, documented in lib/world.ts.
  The chosen gate instead: (a) factor-replayed renders match Python's
  pixels (worst per-scene mean absolute difference 0.035 of 255),
  (b) 300 TS captions parse back to their factors exactly with a
  closed vocabulary, and (c) the ONNX gaze retrieves TS-rendered
  scenes at 0.6045 retrieval@1 versus the Python reference 0.583 over
  1024 scenes, within the 3-point gate.
- **Probe arithmetic**: the TypeScript linear, attentive and bilinear
  readouts match the dumped Python outputs within 1e-3 on the parity
  scenes (long fp32 dot products re-summed in float64).

## Measured latencies (Apple M4 Pro, WASM, single thread)

- tongue forward, one caption: about 2 ms
- eye plus imagination, one veiled canvas: 9 to 15 ms
- barman step, nine candidate futures through the eye: 47 to 61 ms

All far under the 150 ms interaction budget; the barman's step
animation hides its batch entirely.

## What is where

```
app/            the play: one page, prologue, three labs, ledger
components/     one folder per act; stage/ holds rail, curtain, canvas
lib/            world (TS twin), tokenizer, inference, probes, results
public/         models (onnx), data (weights, bank, map), results (json)
tests/          the gates
scripts/        gather.mjs, copies the crates verbatim at build time
```

Honest notes, deviations and their interpretations live in
DEMO_NOTES.md.
