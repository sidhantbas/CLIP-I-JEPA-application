# demo notes

A running record of measured latencies and every deviation between the
live computation and the recorded results, one sentence of
interpretation each. Deviations are reported, never hidden.

## Phase A: export and Gate P

- The sculptor checkpoint holds only the eye; the imagination
  (predictor) was never saved by the frozen training code. It was
  regrown in export/readouts.py against the frozen eye standing in for
  its own EMA memory, original objective and budget, final regret
  0.071, matching the recorded final raw regret (0.122) in order of
  magnitude; the eye received no gradient. The probes, barman cost
  heads, attentive grip and attentive bridge were likewise regrown,
  since Phases 4, 8 and 9 measured and discarded them.
- The export entry point is one script, for_the_vitrine.py, with three
  private helper modules (onnx_bodies, readouts, exhibits) so every
  file honors the 250-line rule.
- The token glance bilinear table (1,048,576 floats) and the patch bank
  ship as raw float32/uint8 binaries, not JSON, purely for page weight;
  probes and the attentive heads are plain JSON as specified.
- Gate P (onnxruntime-node vs PyTorch cpu fp32, 32 seed-7 scenes):
  gaze 5.0e-6, tongue 1.3e-6, eye full and veiled below 1e-4,
  imagination 1.6e-6, grip and both bridges below 1e-4. Tolerance 1e-4,
  all green. One lesson: parity truths must be computed on uint8
  quantized canvases, because that is the only world the browser sees.
- Gate P runs on onnxruntime-node in the test harness; the site runs
  onnxruntime-web (WASM). Both execute the identical graphs with fp32
  CPU kernels.

## Phase B: the world in TypeScript and Gate W

- PRNG parity with numpy PCG64 was not attempted; the TS world uses
  mulberry32, documented in lib/world.ts. Rendering given factors is
  the same float64 arithmetic in both languages, so the gate chosen is:
  factor-replayed pixels, caption faithfulness, retrieval proximity.
- Gate W results: worst per-scene pixel mean absolute difference
  0.035 of 255 on the 32 parity scenes (essentially exact, residue is
  uint8 rounding ties); 300 TS captions parse back to factors exactly
  with a closed vocabulary; TS-scene retrieval@1 through the ONNX gaze
  is 0.6045 vs the Python reference 0.583 over 1024 scenes, a gap of
  2.2 points, within the 3-point gate and attributable to sampling
  differences between the two generators.
- The probe arithmetic gate: TS linear, attentive and bilinear cost
  math matches the dumped Python outputs on parity scenes within 1e-3
  (the stated tolerance for 8192-term fp32 dot products re-summed in
  float64).

## Phases C, D, E: the labs, verified live

- Measured latencies in the production WASM build (M4 Pro, one
  thread): tongue 2 ms per caption; eye plus imagination 9 to 15 ms
  per veiled canvas; barman step (nine futures through the eye) 47 to
  61 ms. All under the 150 ms budget; the barman's 240 ms step
  animation hides its batch entirely.
- Live rerank sanity: composing "a small violet star rests in the
  center" ranks a violet star first at cosine 0.760; facing chips move
  rankings visibly less than color and form chips, which is the lab's
  intended lesson and matches the Phase 4 finding.
- One real defect found and fixed by automated interaction testing:
  ONNX Runtime's WASM backend cannot run two inferences concurrently,
  so bursts of veil toggles crashed with "Session already started".
  All inference now passes through a single promise turnstile in
  lib/inference.ts, and the veil collapses bursts into one final
  encore pass.
- One layout defect found and fixed: the prologue's ambient stream
  changed caption heights each pulse, shifting the page under the
  visitor; caption blocks now reserve fixed height, and the page opts
  out of scroll anchoring, which fed back against the curtains'
  sticky instruction boxes.
- The barman's live session tallies will wander from the recorded
  500-episode rates at small counts; the ledger says so in plain text.

## Phase F: the polish

- Copy revised twice at the visitor's request: first anchored to the
  real models (miniature CLIP, miniature I-JEPA), then fully moved to
  a professional register for an academic presentation. All fable
  vocabulary (Namer, Sculptor, gaze, tongue, veil, barman) removed
  from user-facing text; sections are now framed as three
  applications: CLIP retrieval, I-JEPA prediction, and combined
  language-conditioned control. Internal code names are unchanged,
  and no results or mechanisms were altered.
- Reduced motion: useReducedMotion gates the typing caret, the layout
  glides, the bar animations and the step cadence; keyboard: every
  interaction is a real button, the veil toggles with Enter or Space,
  and all canvases carry text alternatives.
- No em dashes anywhere in copy, comments or code; verified by grep.
- Deploy: static export to out/, Cloudflare Pages ready, immutable
  cache headers for models and the WASM runtime in public/_headers.
