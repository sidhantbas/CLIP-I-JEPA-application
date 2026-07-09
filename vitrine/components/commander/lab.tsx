"use client";

/** Instruction-conditioned control over a 4x4 canvas.
 *
 *  A natural-language instruction is parsed into a structured layout
 *  specification -- a color, a shape and a spatial pattern -- and a
 *  compiler realizes it exactly on the grid. Instructions may either
 *  replace the canvas or, in additive mode, stack onto it and draw in
 *  cell by cell, in the manner of a sequential predictor completing a
 *  scene. The pattern vocabulary spans geometric regions (diagonals,
 *  halves, rows, columns, border, checkerboard, corners, center) and
 *  letter-shapes (L, H, T, X, O, plus). Color and shape are free over
 *  the full inventory.
 *
 *  The parsing, compilation and rendering run live; no ground-truth
 *  layout is consulted -- the canvas is constructed toward the stated
 *  specification. The text matcher and layout compiler are behavioral
 *  stand-ins for the exported CLIP and I-JEPA networks, not those
 *  networks executing inference. */

import { useCallback, useEffect, useRef, useState } from "react";

const N = 4;
const CANVAS_BG = "#141419";
const COLORS: Record<string, string> = {
  red: "#e61f1f", blue: "#2659eb", green: "#1abf4d",
  yellow: "#e6c435", violet: "#9e38d9",
};
const COLOR_KEYS = Object.keys(COLORS);
const FORMS = ["triangle", "square", "circle", "star"];

interface Cell { form: string; color: string }
type Grid = (Cell | null)[][];   // [y][x]

function emptyGrid(): Grid {
  return Array.from({ length: N }, () => Array.from({ length: N }, () => null as Cell | null));
}
function cloneGrid(g: Grid): Grid { return g.map((row) => row.map((c) => (c ? { ...c } : null))); }

// ---- spatial patterns: each maps to the set of [x,y] cells it occupies -----
const cellsWhere = (pred: (x: number, y: number) => boolean): [number, number][] => {
  const out: [number, number][] = [];
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (pred(x, y)) out.push([x, y]);
  return out;
};
interface Pattern { label: string; words: string[]; cells: () => [number, number][] }
const PATTERNS: Record<string, Pattern> = {
  full: { label: "the whole canvas", words: ["whole", "entire", "everywhere", "all", "fill"],
    cells: () => cellsWhere(() => true) },
  diagonal: { label: "the main diagonal", words: ["diagonal", "diag"],
    cells: () => Array.from({ length: N }, (_, i): [number, number] => [i, i]) },
  antidiagonal: { label: "the anti-diagonal", words: ["anti-diagonal", "antidiagonal", "counter-diagonal", "other diagonal", "reverse diagonal"],
    cells: () => Array.from({ length: N }, (_, i): [number, number] => [i, N - 1 - i]) },
  tophalf: { label: "the top half", words: ["top half", "top", "upper"],
    cells: () => cellsWhere((_, y) => y < N / 2) },
  bottomhalf: { label: "the bottom half", words: ["bottom half", "bottom", "lower"],
    cells: () => cellsWhere((_, y) => y >= N / 2) },
  lefthalf: { label: "the left half", words: ["left half", "left"],
    cells: () => cellsWhere((x) => x < N / 2) },
  righthalf: { label: "the right half", words: ["right half", "right"],
    cells: () => cellsWhere((x) => x >= N / 2) },
  border: { label: "the border", words: ["border", "edge", "frame", "perimeter", "outline"],
    cells: () => cellsWhere((x, y) => x === 0 || y === 0 || x === N - 1 || y === N - 1) },
  center: { label: "the center", words: ["center", "centre", "middle"],
    cells: () => cellsWhere((x, y) => x >= 1 && x <= 2 && y >= 1 && y <= 2) },
  checkerboard: { label: "a checkerboard", words: ["checkerboard", "checker", "chequer", "checkered"],
    cells: () => cellsWhere((x, y) => (x + y) % 2 === 0) },
  corners: { label: "the corners", words: ["corners", "corner"],
    cells: () => [[0, 0], [N - 1, 0], [0, N - 1], [N - 1, N - 1]] },
  L: { label: "an L shape", words: ["l shape", "l-shape", "letter l", "an l", " l "],
    cells: () => [[0, 0], [0, 1], [0, 2], [0, 3], [1, 3], [2, 3], [3, 3]] },
  H: { label: "an H shape", words: ["h shape", "h-shape", "letter h", "an h", " h "],
    cells: () => cellsWhere((x, y) => x === 0 || x === 3 || ((y === 1 || y === 2))) },
  T: { label: "a T shape", words: ["t shape", "t-shape", "letter t", " t "],
    cells: () => cellsWhere((x, y) => y === 0 || x === 1 || x === 2) },
  X: { label: "an X shape", words: ["x shape", "x-shape", "letter x", "cross diagonal", " x "],
    cells: () => [...Array.from({ length: N }, (_, i): [number, number] => [i, i]),
                  ...Array.from({ length: N }, (_, i): [number, number] => [i, N - 1 - i])] },
  O: { label: "an O shape", words: ["o shape", "o-shape", "letter o", "ring", " o "],
    cells: () => cellsWhere((x, y) => x === 0 || y === 0 || x === N - 1 || y === N - 1) },
  plus: { label: "a plus shape", words: ["plus", "cross", "+"],
    cells: () => cellsWhere((x, y) => x === 1 || x === 2 || y === 1 || y === 2) },
  row0: { label: "the top row", words: ["row 1", "first row", "row one"], cells: () => cellsWhere((_, y) => y === 0) },
  row3: { label: "the bottom row", words: ["row 4", "last row", "bottom row"], cells: () => cellsWhere((_, y) => y === N - 1) },
  col0: { label: "the left column", words: ["column 1", "first column", "left column"], cells: () => cellsWhere((x) => x === 0) },
  col3: { label: "the right column", words: ["column 4", "last column", "right column"], cells: () => cellsWhere((x) => x === N - 1) },
};

const SHAPE_SYNONYMS: Record<string, string> = {
  circle: "circle", circles: "circle", disc: "circle", discs: "circle",
  disk: "circle", disks: "circle", dot: "circle", dots: "circle",
  square: "square", squares: "square", box: "square", boxes: "square",
  triangle: "triangle", triangles: "triangle", tri: "triangle", tris: "triangle",
  star: "star", stars: "star",
};
const ADD_WORDS = ["add", "serially", "serial", "then", "also", "append", "overlay", "stack", "onto"];
const CLEAR_WORDS = ["clear", "reset", "empty", "wipe", "start over", "blank"];

// ---- the parser: instruction -> {color, shape, pattern, additive} ----------
interface Spec { color: string; shape: string; pattern: string; additive: boolean }
interface ParseResult { spec: Spec | null; conf: number; clear: boolean }

function parse(text: string): ParseResult {
  const t = " " + text.toLowerCase().replace(/[^a-z0-9+ -]/g, " ").replace(/\s+/g, " ") + " ";
  if (CLEAR_WORDS.some((w) => t.includes(w))) return { spec: null, conf: 1, clear: true };

  let color: string | null = null;
  for (const c of COLOR_KEYS) if (new RegExp("\\b" + c + "\\b").test(t)) { color = c; break; }
  let shape: string | null = null;
  for (const k of Object.keys(SHAPE_SYNONYMS)) if (new RegExp("\\b" + k + "\\b").test(t)) { shape = SHAPE_SYNONYMS[k]; break; }

  // find the best-matching pattern by longest matching phrase
  let pattern: string | null = null, patScore = 0;
  for (const [key, p] of Object.entries(PATTERNS)) {
    for (const w of p.words) {
      const needle = w.trim();
      if (needle && t.includes(w) && needle.length > patScore) { patScore = needle.length; pattern = key; }
    }
  }

  const additive = ADD_WORDS.some((w) => new RegExp("\\b" + w + "\\b").test(t));

  if (!color && !shape && !pattern) return { spec: null, conf: 0, clear: false };
  const chosen: Spec = {
    color: color ?? "red",
    shape: shape ?? "square",
    pattern: pattern ?? "full",
    additive,
  };
  // confidence rises with how much of the spec was explicit
  const explicit = (color ? 1 : 0) + (shape ? 1 : 0) + (pattern ? 1 : 0);
  return { spec: chosen, conf: Math.min(0.98, 0.5 + 0.16 * explicit), clear: false };
}

function specLabel(s: Spec): string {
  const p = PATTERNS[s.pattern];
  return `place a ${s.color} ${s.shape} on ${p.label}`;
}

// apply a spec to a grid, returning the new grid and the ordered list of
// cells that changed (for step-through rendering)
function compile(base: Grid, spec: Spec): { grid: Grid; changed: [number, number][] } {
  const grid = spec.additive ? cloneGrid(base) : emptyGrid();
  const cells = PATTERNS[spec.pattern].cells();
  const changed: [number, number][] = [];
  cells.forEach(([x, y]) => {
    grid[y][x] = { form: spec.shape, color: spec.color };
    changed.push([x, y]);
  });
  return { grid, changed };
}

// ---- rendering -------------------------------------------------------------
function drawGrid(canvas: HTMLCanvasElement, grid: Grid, reveal?: Set<string>) {
  const ctx = canvas.getContext("2d"); if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = CANVAS_BG; ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#2a2a32"; ctx.lineWidth = 1;
  for (let g = 1; g < N; g++) {
    const p = Math.round(g * W / N) + 0.5;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(W, p); ctx.stroke();
  }
  const cell = W / N, r = cell * 0.30;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const c = grid[y][x];
    if (!c) continue;
    if (reveal && !reveal.has(x + "," + y)) continue;
    const cx = (x + 0.5) * cell, cy = (y + 0.5) * cell;
    ctx.save(); ctx.translate(cx, cy); ctx.fillStyle = COLORS[c.color]; ctx.beginPath();
    if (c.form === "circle") { ctx.arc(0, 0, r, 0, Math.PI * 2); }
    else {
      const count = c.form === "star" ? 10 : c.form === "square" ? 4 : 3;
      const step = c.form === "star" ? 36 : c.form === "square" ? 90 : 120;
      const off = c.form === "square" ? 45 : c.form === "triangle" ? 90 : 0;
      for (let k = 0; k < count; k++) {
        const ang = (off + k * step) * Math.PI / 180;
        const rr = (c.form === "star" && k % 2 === 1) ? r * 0.45 : r;
        const px = Math.cos(ang) * rr, py = -Math.sin(ang) * rr;
        k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.closePath();
    }
    ctx.fill(); ctx.restore();
  }
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const EXAMPLES = [
  "fill the whole canvas with a red circle",
  "put a blue square on the diagonal",
  "green triangle on the top half",
  "serially add a violet star in an L shape",
  "add a yellow circle in an H shape",
  "red square on the border",
  "checkerboard of blue triangles",
  "add a green star on the anti-diagonal",
];

type Status = null
  | { kind: "ok"; label: string; count: number; additive: boolean }
  | { kind: "clear" }
  | { kind: "none" };

export default function CommanderLab() {
  const [grid, setGrid] = useState<Grid>(() => emptyGrid());
  const [cmd, setCmd] = useState("");
  const [parsed, setParsed] = useState<ParseResult | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [status, setStatus] = useState<Status>(null);
  const [narrate, setNarrate] = useState(
    "Describe a color, a shape and a region. Prefix with “add” or “serially add” to stack onto the current canvas instead of replacing it.");
  const [busy, setBusy] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => { if (canvasRef.current) drawGrid(canvasRef.current, grid); }, [grid]);

  const clearCanvas = useCallback(() => {
    if (busy) return;
    setGrid(emptyGrid()); setHistory([]); setStatus({ kind: "clear" });
    setNarrate("Canvas cleared. Issue a new instruction to begin.");
  }, [busy]);

  const run = useCallback(async () => {
    if (busy) return;
    const res = parse(cmd);
    setParsed(res);
    if (res.clear) { clearCanvas(); return; }
    if (!res.spec) { setStatus({ kind: "none" }); return; }
    setBusy(true);
    const spec = res.spec;
    const { grid: target, changed } = compile(grid, spec);

    setNarrate(`Parsed: ${specLabel(spec)}${spec.additive ? ", stacking onto the current canvas" : ", replacing the canvas"}. Compiling ${changed.length} cell${changed.length > 1 ? "s" : ""}…`);
    await sleep(450);

    // step-through: reveal cells one at a time on top of the prior canvas
    const shown = new Set<string>();
    // start from the additive base already on screen; for replace, clear first
    const baseGrid = spec.additive ? grid : emptyGrid();
    changed.forEach(([x, y]) => { /* mark for reveal */ void x; void y; });
    // draw the merged grid but reveal only the base + progressively the new cells
    const baseKeys = new Set<string>();
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) if (baseGrid[y][x]) baseKeys.add(x + "," + y);

    const drawStep = () => {
      if (canvasRef.current) drawGrid(canvasRef.current, target, new Set([...baseKeys, ...shown]));
    };
    drawStep();
    const per = changed.length > 10 ? 40 : changed.length > 6 ? 70 : 110;
    for (const [x, y] of changed) {
      shown.add(x + "," + y);
      drawStep();
      await sleep(per);
    }

    setGrid(target);
    setHistory((h) => [...h, specLabel(spec) + (spec.additive ? " (stacked)" : "")]);
    setStatus({ kind: "ok", label: specLabel(spec), count: changed.length, additive: spec.additive });
    setNarrate(`${spec.additive ? "Stacked" : "Rendered"} ${changed.length} cell${changed.length > 1 ? "s" : ""} at exact positions. ${spec.additive ? "The canvas retains everything placed before." : "The canvas was rebuilt from this instruction."}`);
    setBusy(false);
  }, [busy, cmd, grid, clearCanvas]);

  const onInput = (v: string) => { setCmd(v); setParsed(parse(v)); };
  const pickExample = (e: string) => { setCmd(e); setParsed(parse(e)); };

  return (
    <div>
      {/* command */}
      <div className="mt-8 flex flex-wrap gap-3">
        <label className="flex flex-1 items-center gap-3 rounded-lg border-[1.5px] border-[var(--ink)] bg-[var(--paper)] px-4 focus-within:border-[var(--accent)]"
               style={{ minWidth: "300px" }}>
          <span aria-hidden className="ui-sans font-bold text-[var(--accent)]">&rsaquo;</span>
          <input value={cmd} onChange={(e) => onInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); run(); } }}
            placeholder="e.g. serially add a blue square in an L shape"
            aria-label="natural-language layout instruction"
            autoComplete="off" spellCheck={false}
            className="flex-1 bg-transparent py-3 text-base italic text-[var(--ink)] outline-none placeholder:not-italic placeholder:text-[var(--ink-soft)]" />
        </label>
        <button type="button" onClick={run} disabled={busy}
          className="ui-sans rounded-lg border border-[var(--ink)] bg-[var(--ink)] px-5 py-3 text-sm font-semibold text-[var(--paper)] transition-opacity hover:opacity-85 disabled:opacity-40">
          Issue instruction
        </button>
        <button type="button" onClick={clearCanvas} disabled={busy}
          className="ui-sans rounded-lg border border-[var(--ink)] bg-transparent px-4 py-3 text-sm text-[var(--ink)] transition-opacity hover:opacity-70 disabled:opacity-40">
          Clear
        </button>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {EXAMPLES.map((e) => (
          <button key={e} type="button" onClick={() => pickExample(e)}
            className="ui-sans rounded-full border hairline bg-[var(--paper)] px-3 py-1.5 text-[12.5px] text-[var(--ink-soft)] transition-colors hover:border-[var(--ink-soft)] hover:text-[var(--ink)]">
            {e}
          </button>
        ))}
      </div>

      {/* parse read-back */}
      <div aria-live="polite"
           className="ui-sans mt-4 flex min-h-[48px] flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-[var(--wash)] px-4 py-3 text-[13.5px]">
        <span className="text-[10px] uppercase tracking-[0.14em] text-[var(--ink-soft)]">text encoder parses</span>
        {parsed?.clear ? (
          <span className="font-semibold text-[var(--ink)]">clear the canvas</span>
        ) : parsed?.spec ? (
          <>
            <span className="font-semibold text-[var(--ink)]">&ldquo;{specLabel(parsed.spec)}&rdquo;</span>
            <span className="text-[var(--ink-soft)]">&middot; {parsed.spec.additive ? "additive (stacks)" : "replaces canvas"}</span>
            <span className="ml-auto tabular-nums text-[var(--ink-soft)]">confidence <b className="text-[var(--accent)]">{parsed.conf.toFixed(2)}</b></span>
          </>
        ) : cmd ? (
          <span className="text-[var(--ink-soft)]">no color, shape or region recognized; try an example below</span>
        ) : (
          <span className="text-[var(--ink-soft)]">awaiting an instruction&hellip;</span>
        )}
      </div>

      {/* canvas + history */}
      <div className="mt-10 flex flex-wrap items-start gap-10">
        <figure className="m-0 flex flex-col items-center gap-2">
          <canvas ref={canvasRef} width={256} height={256} role="img"
            aria-label="the 4 by 4 layout canvas"
            className="rounded-md" style={{ width: 256, height: 256 }} />
          <figcaption className="ui-sans text-[10px] uppercase tracking-[0.1em] text-[var(--ink-soft)]">the 4 &times; 4 canvas</figcaption>
        </figure>

        <div className="min-w-[220px] flex-1">
          <h3 className="text-lg">Instruction stack</h3>
          <p className="ui-sans mt-1 mb-3 text-xs text-[var(--ink-soft)]">
            Additive instructions accumulate here; each is applied in order.
          </p>
          {history.length === 0 ? (
            <p className="ui-sans text-sm text-[var(--ink-soft)]">No instructions issued yet.</p>
          ) : (
            <ol className="ui-sans space-y-1.5 text-sm tabular-nums">
              {history.map((h, i) => (
                <li key={i} className="flex gap-2">
                  <span className="text-[var(--ink-soft)]">{i + 1}.</span>
                  <span>{h}</span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>

      <p className="ui-sans mt-6 min-h-[1.4em] text-[13px] leading-relaxed text-[var(--ink-soft)]">
        {narrate}
      </p>

      {status?.kind === "ok" && (
        <div className="mt-4 rounded-lg bg-[var(--wash)] px-5 py-4 text-[15.5px] leading-relaxed">
          <b className="text-[var(--accent)]">{status.additive ? "Layer added." : "Layout rendered."}</b>{" "}
          &ldquo;{status.label}&rdquo; was compiled to {status.count} cell{status.count > 1 ? "s" : ""} at exact grid positions
          {status.additive ? ", stacked onto the existing canvas without disturbing prior placements." : "."}{" "}
          The instruction was grounded in geometry: a color, a shape and a spatial region, bound together and realized precisely.
        </div>
      )}

      {/* vocabulary reference */}
      <div className="mt-10 rounded-lg border hairline p-5">
        <h3 className="ui-sans text-[11px] uppercase tracking-[0.15em] text-[var(--ink-soft)]">The instruction language</h3>
        <div className="mt-4 grid gap-6 sm:grid-cols-3">
          <Vocab title="colors" items={COLOR_KEYS} />
          <Vocab title="shapes" items={FORMS} />
          <Vocab title="regions & shapes" items={[
            "whole canvas", "diagonal", "anti-diagonal", "top / bottom half",
            "left / right half", "border", "center", "checkerboard", "corners",
            "L, H, T, X, O, plus", "top / bottom row", "left / right column",
          ]} />
        </div>
        <p className="ui-sans mt-5 max-w-2xl text-xs leading-relaxed text-[var(--ink-soft)]">
          Combine any color, any shape and any region in one sentence.
          Prefix with <b className="text-[var(--ink)]">add</b> or{" "}
          <b className="text-[var(--ink)]">serially add</b> to stack onto the
          canvas and watch it draw in cell by cell; omit it to replace the
          canvas. Say <b className="text-[var(--ink)]">clear</b> to reset.
        </p>
      </div>
    </div>
  );
}

function Vocab({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <p className="ui-sans text-[10px] uppercase tracking-[0.14em] text-[var(--ink-soft)]">{title}</p>
      <ul className="ui-sans mt-2 flex flex-wrap gap-1.5">
        {items.map((it) => (
          <li key={it} className="rounded border hairline px-2 py-0.5 text-[12px] text-[var(--ink)]">{it}</li>
        ))}
      </ul>
    </div>
  );
}
