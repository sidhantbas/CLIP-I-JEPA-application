"use client";

/** The dataset explorer: a sample of generated scenes on the left, the
 *  selected scene dissected on the right. Hovering or focusing a row of
 *  the factor table highlights the caption words that factor produced,
 *  making the image-to-caption mapping literal. When a caption omits
 *  orientation, the omission is called out, because that 50% gap is
 *  the dataset's central design choice. */

import { useMemo, useState } from "react";
import SceneCanvas from "@/components/stage/scene-canvas";
import { Scene, conjure, mulberry32, sizeWord } from "@/lib/world";

const SAMPLE = 12;

type FactorKey = "size" | "color" | "form" | "position" | "orientation";
interface Span { text: string; factor?: FactorKey; shape?: number }
interface Highlight { factor: FactorKey; shape: number }

/** Rebuilds the caption as annotated spans from the ground truth. */
function annotate(scene: Scene): Span[] {
  const p = scene.factors[0];
  const spans: Span[] = [
    { text: "a " },
    { text: sizeWord(p), factor: "size", shape: 0 },
    { text: " " },
    { text: p.color, factor: "color", shape: 0 },
    { text: " " },
    { text: p.form, factor: "form", shape: 0 },
    { text: " rests in the " },
    { text: p.grid, factor: "position", shape: 0 },
  ];
  if (scene.caption.includes(", facing ")) {
    spans.push({ text: ", facing " },
               { text: p.facing, factor: "orientation", shape: 0 });
  }
  scene.factors.slice(1).forEach((o, k) => {
    spans.push({ text: ", beside a " },
      { text: sizeWord(o), factor: "size", shape: k + 1 },
      { text: " " },
      { text: o.color, factor: "color", shape: k + 1 },
      { text: " " },
      { text: o.form, factor: "form", shape: k + 1 });
  });
  return spans;
}

function FactorRow({ name, truth, inCaption, active, onHover }: {
  name: string; truth: string; inCaption: string;
  active: boolean; onHover: (on: boolean) => void;
}) {
  return (
    <tr tabIndex={0}
        onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)}
        onFocus={() => onHover(true)} onBlur={() => onHover(false)}
        className={`cursor-default transition-colors ${
          active ? "bg-[var(--wash)]" : ""}`}>
      <td className="ui-sans py-1.5 pr-4 text-[10px] uppercase tracking-[0.15em] text-[var(--ink-soft)]">
        {name}
      </td>
      <td className="figure-number py-1.5 pr-4 text-xs">{truth}</td>
      <td className={`py-1.5 text-xs italic ${
        inCaption === "not mentioned" ? "text-[var(--accent)]" : "text-[var(--ink-soft)]"}`}>
        {inCaption}
      </td>
    </tr>
  );
}

export default function DatasetExplorer() {
  const [seed, setSeed] = useState(7);
  const [picked, setPicked] = useState(0);
  const [mark, setMark] = useState<Highlight | null>(null);
  const [grid, setGrid] = useState(false);

  const scenes = useMemo(() => {
    const rng = mulberry32(seed);
    return Array.from({ length: SAMPLE }, () => conjure(rng));
  }, [seed]);

  const scene = scenes[picked];
  const spans = useMemo(() => annotate(scene), [scene]);
  const facingSpoken = scene.caption.includes(", facing ");

  const hover = (factor: FactorKey, shape: number) => (on: boolean) =>
    setMark(on ? { factor, shape } : null);

  return (
    <div className="grid gap-10 lg:grid-cols-[auto_1fr]">
      <div>
        <div className="grid w-fit grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-3">
          {scenes.map((s, i) => (
            <button key={s.caption + i} type="button"
              onClick={() => { setPicked(i); setMark(null); }}
              aria-pressed={picked === i}
              aria-label={`inspect scene ${i + 1}: ${s.caption}`}
              className={`border-2 transition-colors ${
                picked === i ? "border-[var(--ink)]" : "border-transparent hover:border-[var(--ink-soft)]"}`}>
              <SceneCanvas factors={s.factors} caption="" scale={1.25} />
            </button>
          ))}
        </div>
        <div className="ui-sans mt-4 flex items-center gap-4 text-xs">
          <button type="button"
            onClick={() => { setSeed((s) => s + 1); setPicked(0); setMark(null); }}
            className="underline decoration-dotted underline-offset-4">
            draw {SAMPLE} new scenes
          </button>
          <span className="figure-number text-[var(--ink-soft)]">seed {seed}</span>
        </div>
      </div>

      <div>
        <div className="flex flex-wrap items-start gap-6">
          <div className="relative" style={{ width: 192, height: 192 }}>
            <SceneCanvas factors={scene.factors} caption={scene.caption}
                         scale={3} className="border hairline" />
            {grid && (
              <div aria-hidden className="pointer-events-none absolute inset-0 grid grid-cols-3 grid-rows-3">
                {Array.from({ length: 9 }, (_, i) => (
                  <div key={i} className="border border-[var(--paper)]/60" />
                ))}
              </div>
            )}
          </div>
          <div className="max-w-xs">
            <p className="text-base italic leading-relaxed" aria-label={scene.caption}>
              {spans.map((span, i) =>
                span.factor ? (
                  <mark key={i} className={`transition-colors ${
                    mark && mark.factor === span.factor && mark.shape === span.shape
                      ? "bg-[var(--ink)] text-[var(--paper)]"
                      : "bg-transparent text-inherit underline decoration-dotted underline-offset-4"}`}>
                    {span.text}
                  </mark>
                ) : (
                  <span key={i}>{span.text}</span>
                ))}
            </p>
            {!facingSpoken && (
              <p className="ui-sans mt-3 text-xs leading-relaxed" style={{ color: "var(--accent)" }}>
                This caption omits orientation. The true facing is{" "}
                {scene.factors[0].facing}; half of all captions leave it
                unsaid, by design.
              </p>
            )}
            <label className="ui-sans mt-4 flex w-fit cursor-pointer items-center gap-2 text-xs text-[var(--ink-soft)]">
              <input type="checkbox" checked={grid}
                     onChange={(e) => setGrid(e.target.checked)} />
              show the 3x3 region grid captions use
            </label>
          </div>
        </div>

        {scene.factors.map((f, shape) => (
          <table key={shape} className="mt-6 w-full max-w-lg border-t hairline">
            <caption className="ui-sans py-2 text-left text-[10px] uppercase tracking-[0.2em] text-[var(--ink-soft)]">
              {shape === 0 ? "primary shape (largest)" : `companion shape ${shape}`}
              , ground truth against caption
            </caption>
            <tbody>
              <FactorRow name="form" truth={f.form}
                inCaption={`"${f.form}"`}
                active={mark?.factor === "form" && mark.shape === shape}
                onHover={hover("form", shape)} />
              <FactorRow name="color" truth={f.color}
                inCaption={`"${f.color}"`}
                active={mark?.factor === "color" && mark.shape === shape}
                onHover={hover("color", shape)} />
              <FactorRow name="position"
                truth={`x ${f.x.toFixed(1)}, y ${f.y.toFixed(1)}`}
                inCaption={shape === 0 ? `"${f.grid}" (3x3 region)` : "not mentioned"}
                active={mark?.factor === "position" && mark.shape === shape}
                onHover={hover("position", shape)} />
              <FactorRow name="orientation"
                truth={`${f.angle.toFixed(1)} deg, bin "${f.facing}"`}
                inCaption={shape === 0 && facingSpoken ? `"facing ${f.facing}"` : "not mentioned"}
                active={mark?.factor === "orientation" && mark.shape === shape}
                onHover={hover("orientation", shape)} />
              <FactorRow name="size"
                truth={`radius ${f.size.toFixed(1)} px`}
                inCaption={`"${sizeWord(f)}" (threshold 11.5 px)`}
                active={mark?.factor === "size" && mark.shape === shape}
                onHover={hover("size", shape)} />
            </tbody>
          </table>
        ))}
        <p className="ui-sans mt-4 max-w-lg text-xs leading-relaxed text-[var(--ink-soft)]">
          Hover a row to see which caption words it produced. Orientation
          is visible in the image as a small notch toward the facing
          direction; without it, a circle would have no readable
          orientation at all.
        </p>
      </div>
    </div>
  );
}
