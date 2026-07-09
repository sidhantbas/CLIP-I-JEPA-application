"use client";

/** The interrogation figure: Phase 4 and Phase 8 rebuilt as one chart
 *  with two physical toggles, glance/scrutiny and pooled/unpooled.
 *  Every number is read from the shipped JSONs. The accent color fires
 *  exactly once in the whole vitrine: the moment the Sculptor's
 *  position bar climbs past the pixel floor when unpooled. Orientation
 *  refuses to move under every combination; the copy says why. */

import { motion, useReducedMotion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { theInterrogation, theUnpooled } from "@/lib/results";
import type { Interrogation, Unpooled } from "@/lib/results";

const FACTORS = ["form", "color", "position", "orientation"];

function Toggle({ label, options, value, onChange }: {
  label: string; options: Array<{ value: string; shown: string }>;
  value: string; onChange: (v: string) => void;
}) {
  return (
    <fieldset className="ui-sans">
      <legend className="mb-1 text-[10px] uppercase tracking-[0.2em] text-[var(--ink-soft)]">
        {label}
      </legend>
      <div className="inline-flex border hairline">
        {options.map((option) => (
          <button key={option.value} type="button"
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
            className={`px-3 py-1.5 text-xs transition-colors ${
              value === option.value
                ? "bg-[var(--ink)] text-[var(--paper)]"
                : "text-[var(--ink-soft)] hover:text-[var(--ink)]"}`}>
            {option.shown}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export default function InterrogationFigure({ onBothFlipped }: {
  onBothFlipped: () => void;
}) {
  const still = useReducedMotion() ?? false;
  const [pooled4, setPooled4] = useState<Interrogation | null>(null);
  const [unpooled8, setUnpooled8] = useState<Unpooled | null>(null);
  const [readout, setReadout] = useState<"glance" | "scrutiny">("glance");
  const [pooling, setPooling] = useState<"pooled" | "unpooled">("pooled");
  const [flippedReadout, setFlippedReadout] = useState(false);
  const [flippedPooling, setFlippedPooling] = useState(false);

  useEffect(() => {
    theInterrogation().then(setPooled4);
    theUnpooled().then(setUnpooled8);
  }, []);
  useEffect(() => {
    if (flippedReadout && flippedPooling) onBothFlipped();
  }, [flippedReadout, flippedPooling, onBothFlipped]);

  const bars = useMemo(() => {
    if (!pooled4 || !unpooled8) return null;
    return FACTORS.map((factor) => {
      const geometric = factor === "position" || factor === "orientation";
      if (pooling === "pooled" || !geometric) {
        return {
          factor, faded: pooling === "unpooled",
          namer: pooled4.namer[factor][readout].mean,
          sculptor: pooled4.sculptor[factor][readout].mean,
          floor: pooled4.pixels[factor][readout].mean,
          chance: pooled4.chance[factor],
        };
      }
      const key = readout === "glance" ? "glance_tokens" : "scrutiny_tokens";
      return {
        factor, faded: false,
        namer: unpooled8.namer[factor][key].mean,
        sculptor: unpooled8.sculptor[factor][key].mean,
        floor: unpooled8.pixels[factor].glance_tokens.mean,
        chance: unpooled8.chance[factor],
      };
    });
  }, [pooled4, unpooled8, readout, pooling]);

  const recovered = pooling === "unpooled" && bars !== null &&
    bars[2].sculptor > bars[2].floor;

  return (
    <div>
      <div className="flex flex-wrap gap-6">
        <Toggle label="probe capacity"
          options={[{ value: "glance", shown: "linear probe" },
                    { value: "scrutiny", shown: "MLP probe" }]}
          value={readout}
          onChange={(v) => { setReadout(v as "glance"); setFlippedReadout(true); }} />
        <Toggle label="feature granularity"
          options={[{ value: "pooled", shown: "pooled (one vector)" },
                    { value: "unpooled", shown: "token-level (64 tokens)" }]}
          value={pooling}
          onChange={(v) => { setPooling(v as "pooled"); setFlippedPooling(true); }} />
      </div>

      <svg viewBox="0 0 640 240" role="img" className="mt-8 w-full max-w-2xl"
        aria-label={bars ? `probe accuracy, ${readout === "glance" ? "linear" : "MLP"} probe on ${pooling === "pooled" ? "pooled" : "token-level"} features: ` +
          bars.map((b) => `${b.factor}: CLIP ${b.namer.toFixed(2)}, I-JEPA ${b.sculptor.toFixed(2)}, raw-pixel baseline ${b.floor.toFixed(2)}`).join("; ")
          : "loading the probe results"}>
        {bars?.map((bar, i) => {
          const x0 = 20 + i * 155;
          const h = (v: number) => v * 170;
          const isPosition = bar.factor === "position";
          return (
            <g key={bar.factor} opacity={bar.faded ? 0.3 : 1}>
              {(["namer", "sculptor"] as const).map((who, j) => (
                <g key={who}>
                  <motion.rect
                    x={x0 + j * 56} width={40}
                    animate={{ y: 190 - h(bar[who]), height: h(bar[who]) }}
                    transition={{ duration: still ? 0 : 0.5, ease: "easeOut" }}
                    fill={who === "sculptor" && isPosition && recovered
                      ? "var(--accent)"
                      : who === "namer" ? "var(--ink-soft)" : "var(--ink)"} />
                  <text x={x0 + j * 56 + 20} y={186 - h(bar[who])}
                    textAnchor="middle" fontSize={10}
                    className="figure-number ui-sans" fill="var(--ink)">
                    {bar[who].toFixed(2)}
                  </text>
                </g>
              ))}
              <line x1={x0 - 6} x2={x0 + 118} y1={190 - h(bar.floor)}
                y2={190 - h(bar.floor)} stroke="var(--ink)"
                strokeWidth={1} strokeDasharray="5 3" />
              <line x1={x0 - 6} x2={x0 + 118} y1={190 - h(bar.chance)}
                y2={190 - h(bar.chance)} stroke="var(--ink-soft)"
                strokeWidth={0.8} strokeDasharray="1.5 3" />
              <text x={x0 + 56} y={212} textAnchor="middle" fontSize={11}
                className="ui-sans" fill="var(--ink)">{bar.factor}</text>
              {bar.faded && (
                <text x={x0 + 56} y={228} textAnchor="middle" fontSize={8}
                  className="ui-sans" fill="var(--ink-soft)">
                  not probed token-level: already saturated
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <p className="ui-sans mt-1 text-[10px] text-[var(--ink-soft)]">
        gray bar: CLIP. black bar: I-JEPA. long dashes: raw-pixel
        baseline. short dashes: chance.
      </p>

      <p className="mt-4 max-w-xl text-sm leading-relaxed text-[var(--ink-soft)]">
        {recovered ? (
          <span>
            <span style={{ color: "var(--accent)" }}>Position was encoded
            all along</span>: at token level, even a linear probe exceeds
            the raw-pixel baseline. The averaged evaluation destroyed the
            information; the objective had learned it.
          </span>
        ) : (
          "Try both settings. Watch what happens to position when the features are read token by token instead of averaged, and watch what orientation does."
        )}
        {" "}Orientation stays at chance under every setting: a genuine
        blind spot of the objective, not of the measurement.
      </p>
    </div>
  );
}
