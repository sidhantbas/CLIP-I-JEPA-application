"use client";

/** The living diagram: gaze and tongue drawn as their true blocks,
 *  meeting at the accord. Whichever path just carried a signal
 *  illuminates; the accord cell grid mirrors the 4x4 scene grid,
 *  shading by the current similarities. */

import { motion, useReducedMotion } from "framer-motion";

function Blocks({ x, labels, lit }: { x: number; labels: string[]; lit: boolean }) {
  return (
    <>
      {labels.map((label, i) => (
        <g key={label + i}>
          <rect x={x} y={30 + i * 34} width={86} height={24} rx={2}
                fill="none" stroke={lit ? "var(--ink)" : "var(--hairline)"}
                strokeWidth={1.2} />
          <text x={x + 43} y={30 + i * 34 + 16} textAnchor="middle"
                className="ui-sans" fontSize={9}
                fill={lit ? "var(--ink)" : "var(--ink-soft)"}>{label}</text>
        </g>
      ))}
    </>
  );
}

export default function AccordDiagram({ sims, tongueBusy }: {
  sims: number[] | null;
  tongueBusy: boolean;
}) {
  const still = useReducedMotion() ?? false;
  const lo = sims ? Math.min(...sims) : 0;
  const hi = sims ? Math.max(...sims) : 1;
  return (
    <svg viewBox="0 0 320 220" role="img" className="w-full max-w-sm"
         aria-label={`the architecture: a convolutional image encoder and a transformer text encoder meeting in a shared embedding space${sims ? "; the matrix shows your query's similarity to each scene" : ""}`}>
      <text x={43} y={18} textAnchor="middle" className="ui-sans" fontSize={10}
            fill="var(--ink)">image encoder</text>
      <Blocks x={0} labels={["conv 64", "conv 128", "conv 192", "conv 256"]}
              lit={sims !== null} />
      <text x={277} y={18} textAnchor="middle" className="ui-sans" fontSize={10}
            fill="var(--ink)">text encoder</text>
      <Blocks x={234} labels={["embed 26 words", "attend", "attend", "project"]}
              lit={tongueBusy || sims !== null} />
      <motion.path
        d="M 86 96 C 130 96, 130 130, 148 130" fill="none"
        stroke={sims ? "var(--ink)" : "var(--hairline)"} strokeWidth={1.2}
        animate={still ? undefined : { pathLength: [0.9, 1] }} />
      <motion.path
        d="M 234 96 C 190 96, 190 130, 172 130" fill="none"
        stroke={tongueBusy || sims ? "var(--ink)" : "var(--hairline)"}
        strokeWidth={1.2} />
      <text x={160} y={120} textAnchor="middle" className="ui-sans"
            fontSize={9} fill="var(--ink-soft)">similarity</text>
      {Array.from({ length: 16 }, (_, i) => {
        const strength = sims
          ? (sims[i] - lo) / Math.max(hi - lo, 1e-6)
          : 0;
        return (
          <rect key={i}
                x={140 + (i % 4) * 11} y={128 + Math.floor(i / 4) * 11}
                width={9} height={9}
                fill={sims ? `rgba(25, 24, 23, ${0.08 + 0.8 * strength})` : "var(--wash)"} />
        );
      })}
      <text x={160} y={196} textAnchor="middle" className="ui-sans"
            fontSize={8} fill="var(--ink-soft)">
        {sims ? "your query against the sixteen scenes" : "waiting for a query"}
      </text>
    </svg>
  );
}
