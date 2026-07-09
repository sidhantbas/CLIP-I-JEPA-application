"use client";

/** The barman, live. The visitor's composed caption stands as the
 *  order; a matching shape lands somewhere else on the canvas. Pick a
 *  cost head, then step the plan or pour all twelve moves: every step
 *  renders the nine candidate futures in TypeScript, encodes them with
 *  the ONNX eye, and scores them with the chosen head. The session
 *  tally sits beside the recorded 500-episode rates, labeled yours vs
 *  the record. */

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import SceneCanvas from "@/components/stage/scene-canvas";
import { runBridge, runEye, runTongue } from "@/lib/inference";
import { theCostHeads } from "@/lib/heads";
import type { CostHeads } from "@/lib/heads";
import {
  attentiveGlanceCost, normalize, pooledGlanceCost, tokenGlanceCost,
} from "@/lib/probes";
import { theBarman, theTokenizerSpec } from "@/lib/results";
import type { BarmanUnpooled } from "@/lib/results";
import { Tokenizer } from "@/lib/tokenizer";
import {
  COMPASS_8, GRID_NAMES, ShapeFactors, cellOf, mulberry32, render,
  toModelInput,
} from "@/lib/world";
import type { Composition } from "@/components/namer/composer";

const MOVES: Array<[number, number]> = [[0, 0],
  ...Array.from({ length: 8 }, (_, k): [number, number] => [
    Math.round(8 * Math.cos((45 * k * Math.PI) / 180)),
    Math.round(-8 * Math.sin((45 * k * Math.PI) / 180)),
  ])];
type Head = "pooled glance" | "token glance" | "attentive";
const HEADS: Head[] = ["pooled glance", "token glance", "attentive"];
const RECORD_KEY: Record<Head, string> = {
  "pooled glance": "glance_pooled", "token glance": "token_glance",
  attentive: "attend",
};
const HEAD_SHOWN: Record<Head, string> = {
  "pooled glance": "cosine on pooled features",
  "token glance": "bilinear on all tokens",
  attentive: "attention-based",
};

function nudge(s: ShapeFactors, move: [number, number]): ShapeFactors {
  const lo = s.size + 1, hi = 63 - s.size;
  const x = Math.min(hi, Math.max(lo, s.x + move[0]));
  const y = Math.min(hi, Math.max(lo, s.y + move[1]));
  return { ...s, x, y, grid: cellOf(x, y) };
}

export default function Barman({ order }: { order: Composition }) {
  const still = useReducedMotion() ?? false;
  const rng = useRef(mulberry32(7 + 41));
  const [shape, setShape] = useState<ShapeFactors | null>(null);
  const [head, setHead] = useState<Head>("pooled glance");
  const [stepCount, setStepCount] = useState(0);
  const [status, setStatus] = useState<"idle" | "busy" | "won" | "lost">("idle");
  const [tally, setTally] = useState<Record<Head, { wins: number; plays: number }>>(
    { "pooled glance": { wins: 0, plays: 0 }, "token glance": { wins: 0, plays: 0 },
      attentive: { wins: 0, plays: 0 } });
  const [record, setRecord] = useState<BarmanUnpooled | null>(null);
  const goal = useRef<Float64Array | null>(null);
  const heads = useRef<CostHeads | null>(null);

  const ready = order.size && order.color && order.form && order.region;
  const caption = ready
    ? `a ${order.size} ${order.color} ${order.form} rests in the ${order.region}` : null;

  useEffect(() => { theBarman().then(setRecord); void theCostHeads(); }, []);

  useEffect(() => {                    // a fresh order takes the floor
    if (!ready) { setShape(null); return; }
    const r = rng.current;
    const size = order.size === "small" ? 8 + r() * 3.5 : 11.5 + r() * 3.5;
    let x = 0, y = 0;
    do {
      const lo = size + 1, hi = 63 - size;
      x = lo + r() * (hi - lo);
      y = lo + r() * (hi - lo);
    } while (cellOf(x, y) === order.region);
    const angle = r() * 360;
    setShape({ form: order.form!, color: order.color!, x, y,
               grid: cellOf(x, y), angle,
               facing: COMPASS_8[Math.floor(((angle + 22.5) % 360) / 45)], size });
    setStepCount(0);
    setStatus("idle");
    goal.current = null;
  }, [order, ready]);

  async function ensureGoal(): Promise<Float64Array> {
    if (goal.current) return goal.current;
    const spec = await theTokenizerSpec();
    const speech = await runTongue(new Tokenizer(spec).encode(caption!), 1);
    goal.current = normalize(await runBridge("pooled", speech, 1));
    return goal.current;
  }

  async function stepOnce(current: ShapeFactors): Promise<ShapeFactors> {
    const g = await ensureGoal();
    if (!heads.current) heads.current = await theCostHeads();
    const futures = MOVES.map((m) => nudge(current, m));
    const canvases = new Float32Array(9 * 3 * 64 * 64);
    futures.forEach((f, i) =>
      canvases.set(toModelInput(render([f])), i * 3 * 64 * 64));
    const started = performance.now();
    const tokens = await runEye(canvases, 9,
      Array.from({ length: 64 }, (_, i) => i));
    performance.measure("vitrine:barman-eye",
      { start: started, duration: performance.now() - started });
    if (process.env.NODE_ENV === "development") {
      console.log(`[vitrine] barman step: eye ${(performance.now() - started).toFixed(0)}ms`);
    }
    let best = Infinity, arg = 0;
    for (let i = 0; i < 9; i++) {
      const flat = tokens.subarray(i * 64 * 128, (i + 1) * 64 * 128);
      let cost: number;
      if (head === "pooled glance") {
        const pooled = new Float64Array(128);
        for (let t = 0; t < 64; t++) {
          for (let k = 0; k < 128; k++) pooled[k] += flat[t * 128 + k] / 64;
        }
        cost = pooledGlanceCost(pooled, g);
      } else if (head === "token glance") {
        cost = tokenGlanceCost(flat, g, heads.current.tokenGlance);
      } else {
        cost = attentiveGlanceCost(flat, 64, 128, g, heads.current.attentive);
      }
      if (cost < best) { best = cost; arg = i; }
    }
    return futures[arg];
  }

  async function run(all: boolean) {
    if (!shape || status === "busy") return;
    setStatus("busy");
    let current = shape, taken = stepCount;
    do {
      current = await stepOnce(current);
      taken += 1;
      setShape(current);
      setStepCount(taken);
      if (all && !still) await new Promise((rest) => setTimeout(rest, 240));
    } while (all && current.grid !== order.region && taken < 12);
    if (current.grid === order.region) {
      setStatus("won");
      setTally((t) => ({ ...t, [head]: { wins: t[head].wins + 1, plays: t[head].plays + 1 } }));
    } else if (taken >= 12) {
      setStatus("lost");
      setTally((t) => ({ ...t, [head]: { ...t[head], plays: t[head].plays + 1 } }));
    } else setStatus("idle");
  }

  if (!caption || !shape) {
    return (
      <p className="max-w-md text-sm italic text-[var(--ink-soft)]">
        Compose a full goal above and the planner will place a matching
        shape on the canvas.
      </p>
    );
  }
  const goalIndex = (GRID_NAMES.flat() as string[]).indexOf(order.region!);
  return (
    <div className="space-y-4">
      <p className="text-sm italic">goal: {caption}</p>
      <div className="relative inline-block">
        <SceneCanvas factors={[shape]} scale={3} className="border hairline"
          caption={`the ${shape.color} ${shape.form} is in the ${shape.grid}; the goal is ${order.region}`} />
        <div aria-hidden className="pointer-events-none absolute border-2 border-dashed border-[var(--ink)]/50"
          style={{ left: (goalIndex % 3) * 64, top: Math.floor(goalIndex / 3) * 64,
                   width: 64, height: 64 }} />
      </div>
      <div className="ui-sans flex flex-wrap items-center gap-2 text-xs">
        {HEADS.map((h) => (
          <button key={h} type="button" aria-pressed={head === h}
            onClick={() => setHead(h)} disabled={status === "busy"}
            className={`border px-3 py-1.5 ${head === h
              ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
              : "hairline text-[var(--ink-soft)]"}`}>
            {HEAD_SHOWN[h]}
          </button>
        ))}
        <span className="mx-2 text-[var(--ink-soft)]">
          step {stepCount} of 12 {status === "won" && "· goal reached"}
          {status === "lost" && "· failed within 12 steps"}
        </span>
        <button type="button" onClick={() => void run(false)}
          disabled={status !== "idle" || stepCount >= 12}
          className="border hairline px-3 py-1.5 disabled:opacity-40">
          one step
        </button>
        <button type="button" onClick={() => void run(true)}
          disabled={status !== "idle"}
          className="border border-[var(--ink)] px-3 py-1.5 disabled:opacity-40">
          run all 12
        </button>
      </div>
      {record && (
        <table className="ui-sans w-full max-w-md text-left text-xs">
          <caption className="sr-only">your session success rate against the recorded 500-episode rates</caption>
          <thead>
            <tr className="border-b hairline text-[var(--ink-soft)]">
              <th className="py-1 font-normal">cost head</th>
              <th className="font-normal">your session</th>
              <th className="font-normal">recorded ({record.episodes} episodes)</th>
            </tr>
          </thead>
          <tbody>
            {HEADS.map((h) => (
              <tr key={h} className="border-b hairline">
                <td className="py-1">{HEAD_SHOWN[h]}</td>
                <td className="figure-number">
                  {tally[h].plays === 0 ? "no episodes yet"
                    : `${tally[h].wins}/${tally[h].plays}`}
                </td>
                <td className="figure-number">
                  {(record.sculptor[RECORD_KEY[h]] * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
