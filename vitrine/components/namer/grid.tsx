"use client";

/** The reranking grid: sixteen fixed scenes whose gaze embeddings are
 *  computed once at load; every caption edit runs the tongue live and
 *  the grid resorts by cosine, scenes gliding to their new ranks. Click
 *  a scene to flip the direction: eight candidate captions rerank
 *  against that scene's gaze embedding instead. */

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import SceneCanvas from "@/components/stage/scene-canvas";
import { runGaze, runTongue } from "@/lib/inference";
import { dot, normalize } from "@/lib/probes";
import { Tokenizer } from "@/lib/tokenizer";
import { theTokenizerSpec } from "@/lib/results";
import { Scene, conjure, mulberry32, render, toModelInput } from "@/lib/world";

const N = 16;

export interface GridHandle { scenes: Scene[] }

export default function RerankGrid({
  caption, flipped, onFlip, onSims, onTongueRan,
}: {
  caption: string | null;
  flipped: number | null;              // index of the clicked scene, or null
  onFlip: (index: number | null) => void;
  onSims?: (sims: number[] | null) => void;
  onTongueRan?: (ms: number) => void;
}) {
  const still = useReducedMotion() ?? false;
  const scenes = useMemo(() => {
    const rng = mulberry32(7 + 21);
    return Array.from({ length: N }, () => conjure(rng));
  }, []);
  const [sights, setSights] = useState<Float64Array[] | null>(null);
  const [speeches, setSpeeches] = useState<Float64Array[] | null>(null);
  const [sims, setSims] = useState<number[] | null>(null);
  const tokenizer = useRef<Tokenizer | null>(null);
  const debounce = useRef<number>(0);

  useEffect(() => {                    // the gaze looks at the grid once
    let live = true;
    (async () => {
      const spec = await theTokenizerSpec();
      tokenizer.current = new Tokenizer(spec);
      const canvases = new Float32Array(N * 3 * 64 * 64);
      scenes.forEach((s, i) =>
        canvases.set(toModelInput(render(s.factors)), i * 3 * 64 * 64));
      const flat = await runGaze(canvases, N);
      if (!live) return;
      setSights(Array.from({ length: N }, (_, i) =>
        normalize(flat.subarray(i * 128, (i + 1) * 128))));
      const spoken = await runTongue(
        new Tokenizer(spec).encodeBatch(scenes.map((s) => s.caption)), N);
      if (!live) return;
      setSpeeches(Array.from({ length: N }, (_, i) =>
        normalize(spoken.subarray(i * 128, (i + 1) * 128))));
    })();
    return () => { live = false; };
  }, [scenes]);

  useEffect(() => {                    // the tongue speaks per edit, debounced
    if (!caption || !sights || !tokenizer.current) {
      setSims(null);
      onSims?.(null);
      return;
    }
    window.clearTimeout(debounce.current);
    debounce.current = window.setTimeout(async () => {
      const started = performance.now();
      const speech = await runTongue(tokenizer.current!.encode(caption), 1);
      performance.measure("vitrine:tongue",
        { start: started, duration: performance.now() - started });
      onTongueRan?.(performance.now() - started);
      const unit = normalize(speech);
      const next = sights.map((sight) => dot(sight, unit));
      setSims(next);
      onSims?.(next);
    }, 150);
  }, [caption, sights, onSims, onTongueRan]);

  const order = useMemo(() => {
    const ranked = Array.from({ length: N }, (_, i) => i);
    if (sims) ranked.sort((a, b) => sims[b] - sims[a]);
    return ranked;
  }, [sims]);

  if (flipped !== null && sights && speeches) {
    const sight = sights[flipped];
    const ranked = scenes
      .map((s, j) => ({ caption: s.caption, sim: dot(sight, speeches[j]), truth: j === flipped }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, 8);
    return (
      <div className="space-y-4">
        <div className="flex items-start gap-5">
          <SceneCanvas factors={scenes[flipped].factors}
                       caption={scenes[flipped].caption} scale={2}
                       className="border hairline" />
          <button type="button" onClick={() => onFlip(null)}
                  className="ui-sans text-xs underline decoration-dotted underline-offset-4">
            back to text-to-image retrieval
          </button>
        </div>
        <ol className="space-y-1.5">
          {ranked.map((row) => (
            <li key={row.caption}
                className={`flex items-baseline justify-between gap-4 border-b hairline pb-1.5 text-sm ${row.truth ? "" : "text-[var(--ink-soft)]"}`}>
              <span>{row.caption}{row.truth ? "  (ground-truth caption)" : ""}</span>
              <span className="figure-number ui-sans text-xs">{row.sim.toFixed(3)}</span>
            </li>
          ))}
        </ol>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-4 gap-4" role="list"
         aria-label="sixteen scenes, ranked by similarity to your query">
      {order.map((i, rank) => (
        <motion.button
          key={i}
          layout={!still}
          transition={{ type: "spring", stiffness: 240, damping: 28 }}
          type="button"
          role="listitem"
          onClick={() => onFlip(i)}
          aria-label={`scene ranked ${rank + 1}: ${scenes[i].caption}; click to reverse the retrieval direction`}
          className="group space-y-1 text-left"
        >
          <SceneCanvas factors={scenes[i].factors} caption={scenes[i].caption}
                       scale={1.5}
                       className={`border transition-opacity ${sims && rank < 3 ? "border-[var(--ink)]" : "hairline"} ${sims && rank > 9 ? "opacity-45" : ""}`} />
          <span className="figure-number ui-sans block text-[10px] text-[var(--ink-soft)]">
            {sims ? sims[i].toFixed(3) : " "}
          </span>
        </motion.button>
      ))}
    </div>
  );
}
