"use client";

/** Latent Space: The Architect's Dilemma.
 *
 *  A deduction game whose mechanics are the I-JEPA architecture. The
 *  world is a row of semantic blocks; one is redacted (a Void Zone).
 *  The player reads the visible Context Blocks, deduces the redacted
 *  block's semantic embedding (geometry, material, dynamics), and
 *  submits it at the Predictor Terminal. The Alignment Engine scores the
 *  prediction by L2 distance to the Ghost (the target encoder's ground
 *  truth); a distance under epsilon is alignment. The Ghost's stability
 *  tracks player mastery via an exponential moving average, and correct
 *  alignments yield compute energy.
 *
 *  No pixels are evaluated: state lives entirely in the abstract
 *  embedding vector. The "encoders" here are behavioral stand-ins for
 *  the exported I-JEPA network, not that network executing inference. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// ---- semantic embedding (Module A) -----------------------------------------
const GEOMS = ["cube", "cylinder", "sphere", "pyramid"] as const;
const MATS = ["matte", "specular", "void"] as const;
type Geom = (typeof GEOMS)[number];
type Mat = (typeof MATS)[number];
interface Dynamics { offset_x: number; velocity: number }
interface Embedding { geometry: Geom; material: Mat; dynamics: Dynamics }

const GEOM_W = 0.6, MAT_W = 0.4, DYN_W = 0.5;
// dense_latent_encoder: one-hot categoricals + scaled dynamics
function toVector(e: Embedding): number[] {
  const g = GEOMS.map((x) => (x === e.geometry ? GEOM_W : 0));
  const m = MATS.map((x) => (x === e.material ? MAT_W : 0));
  return [...g, ...m, (e.dynamics.offset_x / 8) * DYN_W, e.dynamics.velocity * DYN_W];
}
function l2(a: number[], b: number[]): number {
  return a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0);
}
const EPSILON = 0.15;

// ---- puzzle generation: deducible context -> unique target ------------------
interface Puzzle {
  clue: string;
  context: (Embedding | null)[];   // 4 blocks; one is null (the Void Zone)
  voidIndex: number;
  answer: Embedding;
  needsOffset: boolean;
}
const obj = (geometry: Geom, material: Mat, dynamics: Dynamics = { offset_x: 0, velocity: 0 }): Embedding =>
  ({ geometry, material, dynamics });
const pick = <T,>(a: readonly T[]): T => a[Math.floor(Math.random() * a.length)];
function pickDiff<T>(a: readonly T[], not: T): T { let x: T; do { x = pick(a); } while (x === not); return x; }

const GENERATORS: (() => Puzzle)[] = [
  () => { const g = pick(GEOMS), m = pick(MATS);
    return { clue: "Observation: this row is uniform in geometry and material.",
      context: [obj(g, m), obj(g, m), null, obj(g, m)], voidIndex: 2, answer: obj(g, m), needsOffset: false }; },
  () => { const a = pick(GEOMS), b = pickDiff(GEOMS, a), m = pick(MATS);
    return { clue: "Observation: geometry alternates in a repeating two-step sequence.",
      context: [obj(a, m), obj(b, m), obj(a, m), null], voidIndex: 3, answer: obj(b, m), needsOffset: false }; },
  () => { const a = pick(GEOMS), b = pickDiff(GEOMS, a), m = pick(MATS);
    return { clue: "Observation: the row is mirror-symmetric about its center.",
      context: [obj(a, m), obj(b, m), null, obj(a, m)], voidIndex: 2, answer: obj(b, m), needsOffset: false }; },
  () => { const g = pick(GEOMS);
    return { clue: "Observation: neighbors share geometry and material; the redacted block is the sole void-material anomaly.",
      context: [obj(g, "matte"), obj(g, "matte"), null, obj(g, "matte")], voidIndex: 2, answer: obj(g, "void"), needsOffset: false }; },
  () => { const g = pick(GEOMS), m = pickDiff(MATS, "void"); const off = pick([2, 4, 6]);
    return { clue: `Observation: uniform geometry and material; the redacted block is displaced by ${off} units along x. Tune the offset to align.`,
      context: [obj(g, m), obj(g, m), null, obj(g, m)], voidIndex: 2, answer: obj(g, m, { offset_x: off, velocity: 0 }), needsOffset: true }; },
];
function newPuzzle(): Puzzle { return GENERATORS[Math.floor(Math.random() * GENERATORS.length)](); }

// ---- vector rendering: blueprint glyphs ------------------------------------
const INK = "#141419";
function BlockGlyph({ e, mode, size = 64 }: { e: Embedding | null; mode: "context" | "void" | "aligned"; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const staticPhase = useRef(0);
  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const ctx = cv.getContext("2d"); if (!ctx) return;
    let raf = 0;
    const line = mode === "aligned" ? "#f07b47" : mode === "context" ? "#7fa8d8" : "#4a4a55";
    const draw = () => {
      ctx.clearRect(0, 0, size, size);
      if (mode === "void" && !e) {
        // entropy static field: informational entropy, unpredictable
        for (let i = 0; i < 90; i++) {
          const x = Math.random() * size, y = Math.random() * size;
          ctx.fillStyle = `rgba(150,140,170,${0.05 + Math.random() * 0.25})`;
          ctx.fillRect(x, y, 2, 2);
        }
        raf = requestAnimationFrame(draw);
        return;
      }
      if (!e) return;
      ctx.strokeStyle = line; ctx.fillStyle = "transparent";
      ctx.lineWidth = mode === "aligned" ? 2 : 1.4;
      if (mode === "aligned") { ctx.shadowColor = line; ctx.shadowBlur = 10; }
      drawWire(ctx, e.geometry, size);
      ctx.shadowBlur = 0;
      // material hint: specular = double stroke, void = dashed, matte = solid
      if (e.material === "specular") { ctx.globalAlpha = 0.4; drawWire(ctx, e.geometry, size, 4); ctx.globalAlpha = 1; }
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [e, mode, size]);
  return <canvas ref={ref} width={size} height={size} className="block" aria-hidden />;
}
function drawWire(ctx: CanvasRenderingContext2D, g: Geom, s: number, inset = 0) {
  const p = 12 + inset, q = s - 12 - inset, c = s / 2, r = (q - p) / 2;
  ctx.beginPath();
  if (g === "cube") {
    const d = r * 0.5;
    ctx.rect(p, p + d, q - p - d, q - p - d);
    ctx.moveTo(p, p + d); ctx.lineTo(p + d, p); ctx.lineTo(q, p); ctx.lineTo(q, q - d);
    ctx.lineTo(q - d, q); ctx.moveTo(q, p); ctx.lineTo(q - d, p + d);
  } else if (g === "cylinder") {
    ctx.ellipse(c, p + r * 0.4, r, r * 0.35, 0, 0, Math.PI * 2);
    ctx.moveTo(p, p + r * 0.4); ctx.lineTo(p, q - r * 0.4);
    ctx.moveTo(q, p + r * 0.4); ctx.lineTo(q, q - r * 0.4);
    ctx.ellipse(c, q - r * 0.4, r, r * 0.35, 0, 0, Math.PI);
  } else if (g === "sphere") {
    ctx.arc(c, c, r, 0, Math.PI * 2);
    ctx.moveTo(p, c); ctx.ellipse(c, c, r, r * 0.35, 0, 0, Math.PI * 2);
    ctx.moveTo(c, p); ctx.ellipse(c, c, r * 0.35, r, 0, 0, Math.PI * 2);
  } else {
    ctx.moveTo(c, p); ctx.lineTo(q, q); ctx.lineTo(p, q); ctx.closePath();
    ctx.moveTo(c, p); ctx.lineTo(c, q);
  }
  ctx.stroke();
}

const MATERIAL_HINT: Record<Mat, string> = {
  matte: "solid wireframe", specular: "doubled wireframe", void: "no return signal",
};

type Phase = "deducing" | "aligned" | "dissonant";

export default function LatentGame() {
  const [puzzle, setPuzzle] = useState<Puzzle>(() => newPuzzle());
  const [geom, setGeom] = useState<Geom | null>(null);
  const [mat, setMat] = useState<Mat | null>(null);
  const [offset, setOffset] = useState(0);
  const [phase, setPhase] = useState<Phase>("deducing");
  const [energy, setEnergy] = useState(150);
  const [ghostWeight, setGhostWeight] = useState(1.0);
  const [lastLoss, setLastLoss] = useState<number | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [solved, setSolved] = useState(0);

  const truthVec = useMemo(() => toVector(puzzle.answer), [puzzle]);

  // live L2 as the player composes their prediction (the Alignment Engine)
  const liveLoss = useMemo(() => {
    if (!geom || !mat) return null;
    const guess = toVector(obj(geom, mat, { offset_x: puzzle.needsOffset ? offset : 0, velocity: 0 }));
    // scale ground truth by the Ghost's EMA weight, per spec
    const scaled = truthVec.map((v) => v * ghostWeight);
    return l2(guess, scaled);
  }, [geom, mat, offset, puzzle, truthVec, ghostWeight]);

  const submit = useCallback(() => {
    if (!geom || !mat || phase === "aligned") return;
    const guess = toVector(obj(geom, mat, { offset_x: puzzle.needsOffset ? offset : 0, velocity: 0 }));
    const scaled = truthVec.map((v) => v * ghostWeight);
    const loss = l2(guess, scaled);
    setLastLoss(loss);
    if (loss <= EPSILON) {
      const reward = Math.round(100 * (1 - loss));
      setEnergy((e) => e + reward);
      setGhostWeight((w) => 0.99 * w + 0.01 * 1.0);
      setPhase("aligned");
      setSolved((s) => s + 1);
      setLog((l) => [`ALIGNMENT SUCCESS · L2 ${loss.toFixed(3)} ≤ ε · +${reward} compute`, ...l].slice(0, 6));
    } else {
      setEnergy((e) => Math.max(0, e - 50));
      setGhostWeight((w) => 0.99 * w + 0.01 * 0.0);
      setPhase("dissonant");
      setLog((l) => [`ALIGNMENT FAILURE · L2 ${loss.toFixed(3)} > ε · −50 compute`, ...l].slice(0, 6));
    }
  }, [geom, mat, offset, phase, puzzle, truthVec, ghostWeight]);

  const next = useCallback(() => {
    setPuzzle(newPuzzle()); setGeom(null); setMat(null); setOffset(0);
    setPhase("deducing"); setLastLoss(null);
  }, []);

  return (
    <div>
      {/* HUD */}
      <div className="ui-sans mt-8 flex flex-wrap gap-x-8 gap-y-3 rounded-lg bg-[var(--wash)] px-5 py-4 tabular-nums">
        <Stat label="compute energy" value={String(energy)} tone={energy > 100 ? "good" : energy > 0 ? "ink" : "warn"} />
        <Stat label="ghost stability (EMA)" value={ghostWeight.toFixed(3)} tone="ink" />
        <Stat label="alignment threshold ε" value={EPSILON.toFixed(2)} tone="ink" />
        <Stat label="zones aligned" value={String(solved)} tone="accent" />
      </div>

      {/* the world row */}
      <div className="mt-8">
        <p className="ui-sans text-[10px] uppercase tracking-[0.16em] text-[var(--ink-soft)]">
          observation module · context blocks (z&#8339;)
        </p>
        <p className="mt-2 max-w-2xl leading-relaxed">{puzzle.clue}</p>
        <div className="mt-4 flex flex-wrap gap-3">
          {puzzle.context.map((block, i) => {
            const isVoid = i === puzzle.voidIndex;
            const showAligned = isVoid && phase === "aligned";
            return (
              <figure key={i} className="m-0">
                <div className={`rounded-md border ${
                  isVoid ? (showAligned ? "border-[var(--accent)]" : "border-dashed border-[var(--ink-soft)]/50") : "hairline"}`}
                  style={{ background: INK, padding: 8 }}>
                  <BlockGlyph
                    e={isVoid ? (showAligned ? puzzle.answer : null) : block}
                    mode={isVoid ? (showAligned ? "aligned" : "void") : "context"} />
                </div>
                <figcaption className="ui-sans mt-1.5 text-center text-[10px] uppercase tracking-[0.08em] text-[var(--ink-soft)]">
                  {isVoid ? (showAligned ? "aligned" : "void zone") : `${block!.geometry}`}
                </figcaption>
              </figure>
            );
          })}
        </div>
      </div>

      {/* predictor terminal */}
      <div className="mt-10 grid gap-8 lg:grid-cols-[1.3fr_1fr]">
        <div className="rounded-lg border hairline p-5">
          <p className="ui-sans text-[10px] uppercase tracking-[0.16em] text-[var(--ink-soft)]">
            predictor terminal · compose ẑ&#8339;
          </p>

          <Field label="geometry">
            {GEOMS.map((g) => (
              <Choice key={g} active={geom === g} onClick={() => { setGeom(g); if (phase !== "deducing") setPhase("deducing"); }}>{g}</Choice>
            ))}
          </Field>
          <Field label="material">
            {MATS.map((m) => (
              <Choice key={m} active={mat === m} onClick={() => { setMat(m); if (phase !== "deducing") setPhase("deducing"); }}>{m}</Choice>
            ))}
          </Field>
          {puzzle.needsOffset && (
            <div className="mt-4">
              <p className="ui-sans mb-1 text-[10px] uppercase tracking-[0.14em] text-[var(--ink-soft)]">
                positional offset Δy (x) · <span className="tabular-nums">{offset}</span>
              </p>
              <input type="range" min={0} max={8} step={1} value={offset}
                onChange={(e) => { setOffset(Number(e.target.value)); if (phase !== "deducing") setPhase("deducing"); }}
                aria-label="positional offset along x"
                className="w-full accent-[var(--accent)]" />
            </div>
          )}

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button type="button" onClick={submit} disabled={!geom || !mat || phase === "aligned"}
              className="ui-sans rounded-lg border border-[var(--ink)] bg-[var(--ink)] px-5 py-2.5 text-sm font-semibold text-[var(--paper)] transition-opacity hover:opacity-85 disabled:opacity-40">
              Execute prediction
            </button>
            {phase === "aligned" ? (
              <button type="button" onClick={next}
                className="ui-sans rounded-lg border border-[var(--accent)] bg-transparent px-4 py-2.5 text-sm font-semibold text-[var(--accent)] transition-opacity hover:opacity-80">
                Next zone &rarr;
              </button>
            ) : (
              <button type="button" onClick={next}
                className="ui-sans rounded-lg border border-[var(--ink)] bg-transparent px-4 py-2.5 text-sm text-[var(--ink)] transition-opacity hover:opacity-70">
                Skip zone
              </button>
            )}
          </div>
        </div>

        {/* alignment engine readout */}
        <div className="rounded-lg border hairline p-5">
          <p className="ui-sans text-[10px] uppercase tracking-[0.16em] text-[var(--ink-soft)]">
            alignment engine · L2(ẑ&#8339;, z&#8339;)
          </p>
          <div className="mt-4">
            <div className="ui-sans flex items-baseline justify-between text-sm">
              <span className="text-[var(--ink-soft)]">structural distance</span>
              <span className="tabular-nums text-2xl font-semibold" style={{
                color: liveLoss == null ? "var(--ink-soft)" : liveLoss <= EPSILON ? "var(--good,#1a8a4a)" : "var(--accent)" }}>
                {liveLoss == null ? "—" : liveLoss.toFixed(3)}
              </span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-[var(--hairline)]">
              <div className="h-full rounded-full transition-all" style={{
                width: liveLoss == null ? "0%" : `${Math.max(3, Math.min(100, (1 - Math.min(liveLoss, 1)) * 100))}%`,
                background: liveLoss != null && liveLoss <= EPSILON ? "var(--good,#1a8a4a)" : "var(--accent)" }} />
            </div>
            <p className="ui-sans mt-2 text-[11px] text-[var(--ink-soft)]">
              {liveLoss == null ? "compose a prediction to read the live distance"
                : liveLoss <= EPSILON ? "within threshold — execute to lock alignment"
                : "structural dissonance — refine the deduction"}
            </p>
          </div>
          <p className="ui-sans mt-5 text-[11px] leading-relaxed text-[var(--ink-soft)]">
            material signatures — matte: {MATERIAL_HINT.matte}; specular: {MATERIAL_HINT.specular}; void: {MATERIAL_HINT.void}.
          </p>
          {log.length > 0 && (
            <ul className="ui-sans mt-4 space-y-1 border-t hairline pt-3 text-[11px] tabular-nums">
              {log.map((l, i) => (
                <li key={i} className={l.startsWith("ALIGNMENT SUCCESS") ? "text-[var(--good,#1a8a4a)]" : "text-[var(--ink-soft)]"}>{l}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {phase === "aligned" && lastLoss != null && (
        <div className="mt-6 rounded-lg bg-[var(--wash)] px-5 py-4 text-[15.5px] leading-relaxed">
          <b className="text-[var(--accent)]">Alignment locked.</b> The static dissipated and the pristine wireframe snapped into place: you mapped the redacted zone&rsquo;s semantic truth at L2 distance {lastLoss.toFixed(3)}, inside the threshold. The Ghost&rsquo;s stability rose toward you via EMA, and compute energy was awarded in proportion to the alignment.
        </div>
      )}
      {phase === "dissonant" && lastLoss != null && (
        <div className="mt-6 rounded-lg bg-[var(--wash)] px-5 py-4 text-[15.5px] leading-relaxed ring-1 ring-[var(--ink-soft)]/25">
          <b className="text-[var(--accent)]">Structural dissonance.</b> The prediction sat at L2 distance {lastLoss.toFixed(3)}, beyond the threshold ε = {EPSILON.toFixed(2)} — compute energy was spent. A wrong geometry or material moves the embedding a fixed distance away; the deduction has to be exact. Re-read the context and refine.
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone: "good" | "warn" | "ink" | "accent" }) {
  const color = tone === "good" ? "var(--good, #1a8a4a)" : tone === "warn" ? "var(--accent)"
    : tone === "accent" ? "var(--accent)" : "var(--ink)";
  return (
    <div>
      <div className="text-xl font-semibold" style={{ color }}>{value}</div>
      <div className="text-[11px] text-[var(--ink-soft)]">{label}</div>
    </div>
  );
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-4">
      <p className="ui-sans mb-1.5 text-[10px] uppercase tracking-[0.14em] text-[var(--ink-soft)]">{label}</p>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}
function Choice({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} aria-pressed={active}
      className={`ui-sans rounded-full border px-3 py-1.5 text-[13px] transition-colors ${
        active ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
               : "hairline bg-[var(--paper)] text-[var(--ink-soft)] hover:border-[var(--ink-soft)] hover:text-[var(--ink)]"}`}>
      {children}
    </button>
  );
}
