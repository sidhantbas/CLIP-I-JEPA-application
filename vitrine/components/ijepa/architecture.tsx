"use client";

/** The I-JEPA architecture as its two paths: the online path that sees
 *  a masked image, and the target path that sees everything but trains
 *  nothing. Shapes and counts come from atelier/sculptor.py. Press any
 *  block to read what it does and why it is there. */

import { useState } from "react";
import { DetailCard, Stage, StageColumn } from "@/components/stage/pipeline";

const ONLINE: Stage[] = [
  {
    id: "jepa-in", label: "input image", output: "3 x 64 x 64", params: "0",
    detail: "One rendered scene. No captions, no labels: nothing but the " +
      "image itself enters this model, in training or at inference.",
  },
  {
    id: "patchify", label: "patchify + 2D positions", output: "64 tokens x 128",
    params: "27k",
    detail: "A single 8x8 convolution with stride 8 turns the canvas into " +
      "an 8x8 grid of tokens. Position enters as the sum of a learned row " +
      "embedding and a learned column embedding, so every token knows " +
      "where it sits before any attention happens.",
  },
  {
    id: "mask", label: "mask a contiguous block", output: "33 to 45 visible tokens",
    params: "0",
    detail: "A contiguous rectangular block covering 30 to 50 percent of " +
      "the tokens is removed from the sequence. Removed, not noised and " +
      "not replaced: the encoder below never attends to a masked " +
      "position. Contiguity matters, because completing a large missing " +
      "region requires understanding objects, not interpolating texture.",
  },
  {
    id: "context", label: "context encoder, 4 layers", output: "visible x 128",
    params: "1.35M",
    detail: "A transformer encoder (4 layers, 4 heads, feed-forward width " +
      "1024, final LayerNorm) over the visible tokens only. This is the " +
      "representation everything downstream uses: the probes, the " +
      "retrieval bridge and the planner all read this encoder's output, " +
      "with the mask removed.",
  },
  {
    id: "predictor", label: "predictor, 2 layers", output: "masked x 128",
    params: "0.42M",
    detail: "The visible latents are joined by one learned mask token per " +
      "hidden position, each carrying that position's embedding. Two " +
      "transformer layers let the mask tokens gather evidence from the " +
      "context, and their outputs are the predictions. The predictor is " +
      "discarded after training; it exists only to force the encoder to " +
      "learn predictable structure.",
  },
];

const TARGET: Stage[] = [
  {
    id: "full", label: "the same image, unmasked", output: "3 x 64 x 64",
    params: "0",
    detail: "The target path sees everything. It provides the answer key " +
      "for the online path's fill-in-the-blank exam.",
  },
  {
    id: "ema", label: "target encoder (EMA copy)", output: "64 tokens x 128",
    params: "0 trained",
    detail: "Architecturally identical to the context encoder, but its " +
      "weights are an exponential moving average of the context " +
      "encoder's, momentum 0.996, and it receives no gradients. The slow " +
      "average keeps the targets stable and, together with the final " +
      "LayerNorm, prevents the collapse where both encoders map " +
      "everything to a constant and score perfectly.",
  },
  {
    id: "targets", label: "targets at masked positions", output: "masked x 128",
    params: "0",
    detail: "The target encoder's latents at exactly the positions the " +
      "online path could not see. These vectors, not pixels, are what " +
      "the predictor must match.",
  },
];

const LOSS: Stage = {
  id: "loss", label: "latent regression loss",
  output: "one scalar loss", params: "0",
  detail: "The loss is the regression error between predicted and target " +
    "latents at the masked positions, and nothing else. Because the " +
    "target is a representation rather than pixels, the model may ignore " +
    "detail that does not help prediction. That freedom is the method's " +
    "strength and its documented blind spot here: orientation, carried " +
    "by a few pixels of notch, is dropped entirely, while position, " +
    "color and shape survive.",
};

export default function IJepaArchitecture() {
  const [selected, setSelected] = useState<string | null>(null);
  const all = [...ONLINE, ...TARGET, LOSS];
  const stage = all.find((s) => s.id === selected) ?? null;

  return (
    <div>
      <div className="grid gap-8 sm:grid-cols-2">
        <StageColumn title="online path: sees a masked image" stages={ONLINE}
                     selected={selected} onSelect={setSelected} />
        <div>
          <StageColumn title="target path: sees everything, trains nothing"
                       stages={TARGET}
                       selected={selected} onSelect={setSelected} />
          <p className="figure-number ui-sans mt-3 border-l-2 pl-3 text-[10px] leading-relaxed text-[var(--ink-soft)]"
             style={{ borderColor: "var(--hairline)" }}>
            weights flow one way: after every step the target encoder
            drifts toward the context encoder, w_target = 0.996 w_target
            + 0.004 w_context
          </p>
        </div>
      </div>

      <div className="ui-sans mt-4">
        <button type="button" onClick={() => setSelected(LOSS.id)}
          aria-pressed={selected === LOSS.id}
          className={`flex w-full items-baseline justify-between gap-3 border px-3 py-2 text-left text-xs transition-colors ${
            selected === LOSS.id
              ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
              : "hairline bg-[var(--paper)] hover:border-[var(--ink-soft)]"}`}>
          <span>{LOSS.label}: prediction meets target, in latent space only</span>
          <span className={`figure-number text-[10px] ${
            selected === LOSS.id
              ? "text-[var(--paper)]/70" : "text-[var(--ink-soft)]"}`}>
            no pixels reconstructed
          </span>
        </button>
      </div>

      <div className="mt-4">
        <DetailCard stage={stage}
          hint="Press any block to see its output shape, its parameter count and its role. The key design choice: the prediction target is a representation, not an image." />
      </div>

      <p className="figure-number ui-sans mt-4 text-xs text-[var(--ink-soft)]">
        training: 3,000 steps, batch 256, 20,000 scenes, AdamW, learning
        rate 3e-4, weight decay 0.05. The predictor used in this page was
        regrown against the frozen encoder (documented in the repository);
        the encoder itself is the original checkpoint.
      </p>
    </div>
  );
}
