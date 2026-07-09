"use client";

/** The bridge: a composed goal caption travels tongue, then bridge,
 *  then lands among 512 held-out scenes in Sculptor space. The map is
 *  a fixed 2D projection computed at export time and labeled as such.
 *  One toggle switches the target between mean-pooled and attentively
 *  pooled space; the retrieval readout beside it comes from the JSONs. */

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { runBridge, runTongue } from "@/lib/inference";
import { normalize } from "@/lib/probes";
import { Tokenizer } from "@/lib/tokenizer";
import { theConstellation, theHandshake, theTokenizerSpec } from "@/lib/results";
import type { Constellation, HandshakeUnpooled } from "@/lib/results";

type Space = "pooled" | "attentive";

function project(unit: ArrayLike<number>, space: Constellation["spaces"][string]): [number, number] {
  let x = 0, y = 0;
  for (let k = 0; k < 128; k++) {
    const v = unit[k] - space.center[k];
    x += v * space.axes[k][0];
    y += v * space.axes[k][1];
  }
  return [x, y];
}

export default function Bridge({ caption }: { caption: string | null }) {
  const still = useReducedMotion() ?? false;
  const [map, setMap] = useState<Constellation | null>(null);
  const [record, setRecord] = useState<HandshakeUnpooled | null>(null);
  const [space, setSpace] = useState<Space>("pooled");
  const [landed, setLanded] = useState<[number, number] | null>(null);

  useEffect(() => {
    theConstellation().then(setMap);
    theHandshake().then(setRecord);
  }, []);

  useEffect(() => {
    let live = true;
    if (!caption || !map) { setLanded(null); return; }
    (async () => {
      const spec = await theTokenizerSpec();
      const speech = await runTongue(new Tokenizer(spec).encode(caption), 1);
      const reached = await runBridge(space, speech, 1);
      if (!live) return;
      setLanded(project(normalize(reached), map.spaces[space]));
    })();
    return () => { live = false; };
  }, [caption, space, map]);

  const frame = useMemo(() => {
    if (!map) return null;
    const xs = map.points.map((p) => p[space][0]);
    const ys = map.points.map((p) => p[space][1]);
    const pad = 0.08;
    const lo = [Math.min(...xs) - pad, Math.min(...ys) - pad];
    const hi = [Math.max(...xs) + pad, Math.max(...ys) + pad];
    const sx = (x: number) => ((x - lo[0]) / (hi[0] - lo[0])) * 300 + 10;
    const sy = (y: number) => ((y - lo[1]) / (hi[1] - lo[1])) * 240 + 10;
    return { sx, sy };
  }, [map, space]);

  const at5 = record && (space === "pooled" ? record["pooled@5"] : record["attentive@5"]);

  return (
    <div>
      <div className="ui-sans flex flex-wrap items-end gap-6">
        <div className="inline-flex border hairline">
          {(["pooled", "attentive"] as Space[]).map((s) => (
            <button key={s} type="button" aria-pressed={space === s}
              onClick={() => setSpace(s)}
              className={`px-3 py-1.5 text-xs ${space === s
                ? "bg-[var(--ink)] text-[var(--paper)]"
                : "text-[var(--ink-soft)] hover:text-[var(--ink)]"}`}>
              {s === "pooled" ? "mean-pooled targets" : "attention-pooled targets"}
            </button>
          ))}
        </div>
        {record && (
          <p className="text-xs text-[var(--ink-soft)]">
            recorded retrieval@5 among {record.candidates}:{" "}
            <span className="figure-number text-sm text-[var(--ink)]">
              {(at5! * 100).toFixed(1)}%
            </span>
            <span className="ml-2">
              ({(record["pooled@5"] * 100).toFixed(1)} pooled, {(record["attentive@5"] * 100).toFixed(1)} attentive)
            </span>
          </p>
        )}
      </div>

      <svg viewBox="0 0 320 260" role="img" className="mt-5 w-full max-w-md border hairline bg-[var(--wash)]/40"
        aria-label={`a fixed 2D projection of 512 scenes in I-JEPA's ${space === "pooled" ? "mean-pooled" : "attention-pooled"} representation space${landed ? "; your query has landed among them" : ""}`}>
        {map && frame && map.points.map((p, i) => (
          <circle key={i} cx={frame.sx(p[space][0])} cy={frame.sy(p[space][1])}
            r={1.6} fill="var(--ink-soft)" opacity={0.5}>
            <title>{p.caption}</title>
          </circle>
        ))}
        {landed && frame && (
          <motion.g
            initial={still ? false : { opacity: 0, scale: 0.3 }}
            animate={{ opacity: 1, scale: 1 }}
            key={`${landed[0]}-${space}`}
            transition={{ duration: still ? 0 : 0.45, ease: "easeOut" }}>
            <circle cx={frame.sx(landed[0])} cy={frame.sy(landed[1])} r={5}
              fill="none" stroke="var(--ink)" strokeWidth={1.4} />
            <circle cx={frame.sx(landed[0])} cy={frame.sy(landed[1])} r={1.8}
              fill="var(--ink)" />
          </motion.g>
        )}
        <text x={12} y={252} fontSize={8} className="ui-sans" fill="var(--ink-soft)">
          a projection: two principal axes of this space, fixed at export time
        </text>
      </svg>

      <p className="mt-4 max-w-md text-sm leading-relaxed text-[var(--ink-soft)]">
        The same bridge, retrained toward attention-pooled targets on the
        same budget, nearly doubles retrieval. The bottleneck was the
        pooling of the target space, not the representation itself.
      </p>
    </div>
  );
}
