"use client";

/** The two I-JEPA demos, side by side with the architecture above them
 *  on the page: first the mask-and-predict panel, where the visitor
 *  hides a region and watches the model fill it with its nearest
 *  latents; then the probing figure, where two toggles reveal that the
 *  encoder kept position (hidden by pooling) but truly lost
 *  orientation. Neither is gated: this is a page, not a guided tour. */

import { useEffect } from "react";
import InterrogationFigure from "@/components/sculptor/interrogation";
import VeilPainter from "@/components/sculptor/veil";
import { prepare } from "@/lib/inference";

export default function IJepaLab() {
  useEffect(() => { prepare("eye", "imagination"); }, []);

  return (
    <>
      <section id="predict" aria-labelledby="predict-title"
               className="mt-16 border-t hairline pt-12">
        <h2 id="predict-title" className="text-2xl">
          Live demo: mask a region, predict its representation
        </h2>
        <p className="mb-8 mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
          Drag across the grid to hide patches, then release. The encoder
          runs on the visible patches only; the predictor produces a
          latent for each hidden patch. Since the model has no pixel
          decoder, each prediction is shown as its nearest match among
          8,192 reference patches, which is why a good prediction fills
          the gap with the right color and a continuing contour.
        </p>
        <VeilPainter onImagined={() => {}} />
      </section>

      <section id="probe" aria-labelledby="probe-title"
               className="mt-16 border-t hairline pt-12">
        <h2 id="probe-title" className="text-2xl">
          Live demo: what did the representation actually keep?
        </h2>
        <p className="mb-8 mt-3 max-w-xl leading-relaxed text-[var(--ink-soft)]">
          Both frozen encoders are asked to classify the largest shape&apos;s
          factors from their features. Two probe capacities (a single
          linear layer versus a small MLP) and two feature granularities
          (one averaged vector versus all 64 patch tokens). Flip both
          switches and watch position and orientation behave in opposite
          ways.
        </p>
        <InterrogationFigure onBothFlipped={() => {}} />
      </section>
    </>
  );
}
