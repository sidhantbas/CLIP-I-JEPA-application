"use client";

/** The CLIP architecture, block by block, with real tensor shapes and
 *  parameter counts taken from the training code (atelier/namer.py).
 *  Press any block to read what it does and why it is there. */

import { useState } from "react";
import { DetailCard, Stage, StageColumn } from "@/components/stage/pipeline";

const IMAGE: Stage[] = [
  {
    id: "image-in", label: "input image", output: "3 x 64 x 64", params: "0",
    detail: "One rendered scene, RGB in [0, 1]. No augmentation and no " +
      "cropping: the environment is controlled, so the model sees scenes " +
      "exactly as the dataset produces them.",
  },
  {
    id: "conv1", label: "conv stage 1", output: "64 x 32 x 32", params: "39k",
    detail: "Two 3x3 convolutions, the first with stride 2, each followed " +
      "by GroupNorm and GELU. Striding rather than pooling halves the " +
      "resolution while the channel width grows.",
  },
  {
    id: "conv2", label: "conv stage 2", output: "128 x 16 x 16", params: "222k",
    detail: "The same block at doubled width. By this depth the receptive " +
      "field spans a typical shape, so filters can respond to whole forms, " +
      "not only edges and corners.",
  },
  {
    id: "conv3", label: "conv stage 3", output: "192 x 8 x 8", params: "554k",
    detail: "Third stride-2 block. The 8x8 grid roughly matches the " +
      "granularity at which captions describe position, a 3x3 region " +
      "vocabulary over the canvas.",
  },
  {
    id: "conv4", label: "conv stage 4", output: "256 x 4 x 4", params: "1.03M",
    detail: "Final stride-2 block, the widest and most expensive stage. " +
      "Its 4x4 map is the last point where spatial layout is explicit.",
  },
  {
    id: "pool", label: "global average pool", output: "256", params: "0",
    detail: "The 4x4 map is averaged to a single 256-d vector. Layout is " +
      "deliberately discarded here; whatever position information survives " +
      "must already be encoded in the channels. The probing experiments " +
      "show it is, because captions supervise it.",
  },
  {
    id: "img-proj", label: "linear projection", output: "128", params: "33k",
    detail: "Maps the pooled features into the shared 128-d embedding " +
      "space where images and captions are compared. The embedding is " +
      "L2-normalized, so only direction carries meaning.",
  },
];

const TEXT: Stage[] = [
  {
    id: "tokens", label: "input caption", output: "up to 24 tokens", params: "0",
    detail: "A caption over the closed 26-word vocabulary, padded to " +
      "length 24. The grammar is templated, so every query composed in " +
      "the demo below is a sentence the model saw in training.",
  },
  {
    id: "embed", label: "token + position embeddings", output: "24 x 128",
    params: "6.4k",
    detail: "Each word becomes a learned 128-d vector; a learned position " +
      "vector is added so word order is visible to the layers above.",
  },
  {
    id: "attn", label: "transformer encoder, 2 layers", output: "24 x 128",
    params: "397k",
    detail: "Two pre-norm encoder layers, 4 attention heads, feed-forward " +
      "width 512. Self-attention binds modifiers to their nouns, so " +
      "'small violet star' is one description rather than three words.",
  },
  {
    id: "last", label: "last-token readout", output: "128", params: "0",
    detail: "The hidden state at the last non-pad token stands for the " +
      "whole caption. With bidirectional attention over short templated " +
      "sentences, that state can see and summarize every word.",
  },
  {
    id: "txt-proj", label: "linear projection", output: "128", params: "16.5k",
    detail: "Projects the caption summary into the same 128-d space as the " +
      "images, again L2-normalized before comparison.",
  },
];

const OBJECTIVE: Stage = {
  id: "objective", label: "contrastive objective (InfoNCE)",
  output: "one scalar loss", params: "1",
  detail: "In every batch of 256 pairs, all 256 x 256 cosine similarities " +
    "are computed and scaled by a learnable temperature (initialized at " +
    "0.07). Symmetric cross-entropy then demands that each image ranks " +
    "its own caption first among 256, and each caption its own image. " +
    "Nothing else is supervised: every capability the demo shows falls " +
    "out of this single matching game.",
};

export default function ClipArchitecture() {
  const [selected, setSelected] = useState<string | null>(null);
  const all = [...IMAGE, ...TEXT, OBJECTIVE];
  const stage = all.find((s) => s.id === selected) ?? null;

  return (
    <div>
      <div className="grid gap-8 sm:grid-cols-2">
        <StageColumn title="image encoder, 1.88M parameters" stages={IMAGE}
                     selected={selected} onSelect={setSelected} />
        <StageColumn title="text encoder, 0.42M parameters" stages={TEXT}
                     selected={selected} onSelect={setSelected} />
      </div>

      <div className="ui-sans mt-4">
        <button type="button" onClick={() => setSelected(OBJECTIVE.id)}
          aria-pressed={selected === OBJECTIVE.id}
          className={`flex w-full items-baseline justify-between gap-3 border px-3 py-2 text-left text-xs transition-colors ${
            selected === OBJECTIVE.id
              ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
              : "hairline bg-[var(--paper)] hover:border-[var(--ink-soft)]"}`}>
          <span>{OBJECTIVE.label}: both towers meet in one 128-d space</span>
          <span className={`figure-number text-[10px] ${
            selected === OBJECTIVE.id
              ? "text-[var(--paper)]/70" : "text-[var(--ink-soft)]"}`}>
            temperature, learned
          </span>
        </button>
      </div>

      <div className="mt-4">
        <DetailCard stage={stage}
          hint="Press any block to see its output shape, its parameter count and its role. The two towers never share weights; they meet only in the final embedding space." />
      </div>

      <p className="figure-number ui-sans mt-4 text-xs text-[var(--ink-soft)]">
        training: 3,000 steps, batch 256, 20,000 scenes, AdamW, learning
        rate 3e-4, weight decay 0.05. Verified in-browser against the
        PyTorch reference to within 1e-4.
      </p>
    </div>
  );
}
