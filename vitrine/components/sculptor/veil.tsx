"use client";

/** The veil painter. The visitor drags across the 8x8 patch grid to
 *  hide patches; on release the eye encodes only what remains, the
 *  imagination predicts the hidden latents, and each hidden cell fills
 *  with the nearest patch from the bank of eight thousand remembered
 *  patches, ringed by its cosine similarity. It imagines in thought,
 *  not in pixels; these are the closest thoughts it has. */

import { useEffect, useMemo, useRef, useState } from "react";
import { runEye, runImagination } from "@/lib/inference";
import { loadF32, loadU8 } from "@/lib/results";
import { Scene, conjure, mulberry32, render, toModelInput } from "@/lib/world";

const CELL = 34;                       // on-screen pixels per patch cell
interface Guess { bank: number; sim: number }

function nearest(guess: Float32Array, latents: Float32Array,
                 norms: Float32Array): Guess {
  let gn = 0;
  for (const v of guess) gn += v * v;
  gn = Math.sqrt(gn);
  let best = -Infinity, arg = 0;
  for (let b = 0; b < norms.length; b++) {
    let d = 0;
    const at = b * 128;
    for (let k = 0; k < 128; k++) d += latents[at + k] * guess[k];
    d /= norms[b] * gn;
    if (d > best) { best = d; arg = b; }
  }
  return { bank: arg, sim: best };
}

export default function VeilPainter({ onImagined }: { onImagined: () => void }) {
  const rng = useRef(mulberry32(7 + 31));
  const [scene, setScene] = useState<Scene | null>(null);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  const [guesses, setGuesses] = useState<Map<number, Guess>>(new Map());
  const [thinking, setThinking] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bank = useRef<{ latents: Float32Array; patches: Uint8Array;
                        norms: Float32Array } | null>(null);
  const drag = useRef<{ active: boolean; hide: boolean }>({ active: false, hide: true });
  const veiled = useRef<Set<number>>(new Set());   // mirror: drags outrun renders
  const busy = useRef(false);
  const encore = useRef(false);

  useEffect(() => {
    setScene(conjure(rng.current));
    Promise.all([loadF32("/data/bank_latents.f32"),
                 loadU8("/data/bank_patches.u8")]).then(([latents, patches]) => {
      const norms = new Float32Array(latents.length / 128);
      for (let b = 0; b < norms.length; b++) {
        let s = 0;
        for (let k = 0; k < 128; k++) s += latents[b * 128 + k] ** 2;
        norms[b] = Math.sqrt(s);
      }
      bank.current = { latents, patches, norms };
    });
  }, []);

  const pixels = useMemo(() => scene && render(scene.factors), [scene]);

  useEffect(() => {                    // compose the display canvas
    const canvas = canvasRef.current;
    if (!canvas || !pixels) return;
    const image = new ImageData(64, 64);
    for (let i = 0; i < 64 * 64; i++) {
      const cell = Math.floor(i / 64 / 8) * 8 + Math.floor((i % 64) / 8);
      const veiled = hidden.has(cell);
      const guess = guesses.get(cell);
      let r = 241, g = 239, b = 233;   // the wash, where nothing is known
      if (!veiled) {
        r = Math.round(pixels[i * 3] * 255);
        g = Math.round(pixels[i * 3 + 1] * 255);
        b = Math.round(pixels[i * 3 + 2] * 255);
      } else if (guess && bank.current) {
        const py = Math.floor(i / 64) % 8, px = i % 8;
        const at = guess.bank * 192 + (py * 8 + px) * 3;
        [r, g, b] = bank.current.patches.subarray(at, at + 3);
      }
      image.data.set([r, g, b, 255], i * 4);
    }
    canvas.getContext("2d")?.putImageData(image, 0, 0);
  }, [pixels, hidden, guesses]);

  async function imagine(veiledCells: Set<number>) {
    if (!pixels || !bank.current || veiledCells.size === 0) return;
    if (busy.current) { encore.current = true; return; }  // collapse bursts
    busy.current = true;
    const hiddenIdx = [...veiledCells].sort((a, b) => a - b);
    const visible = Array.from({ length: 64 }, (_, i) => i)
      .filter((i) => !veiledCells.has(i));
    setThinking(true);
    const started = performance.now();
    const context = await runEye(toModelInput(pixels), 1, visible);
    const predicted = await runImagination(context, 1, visible.length, hiddenIdx);
    const ms = performance.now() - started;
    performance.measure("vitrine:imagine", { start: started, duration: ms });
    if (process.env.NODE_ENV === "development") {
      console.log(`[vitrine] eye+imagination ${ms.toFixed(1)}ms for ${hiddenIdx.length} hidden`);
    }
    const found = new Map<number, Guess>();
    hiddenIdx.forEach((cell, i) => {
      found.set(cell, nearest(predicted.subarray(i * 128, (i + 1) * 128),
                              bank.current!.latents, bank.current!.norms));
    });
    setGuesses(found);
    setThinking(false);
    onImagined();
    busy.current = false;
    if (encore.current) {              // one more pass with the final veil
      encore.current = false;
      void imagine(veiled.current);
    }
  }

  const toggle = (cell: number, forceHide?: boolean) => {
    setGuesses(new Map());
    const next = new Set(veiled.current);
    const hide = forceHide ?? !next.has(cell);
    if (hide && next.size < 44) next.add(cell);
    if (!hide) next.delete(cell);
    veiled.current = next;
    setHidden(next);
  };

  if (!scene) return null;
  return (
    <div className="space-y-4">
      <div
        className="relative touch-none select-none"
        style={{ width: 8 * CELL, height: 8 * CELL }}
        onPointerUp={() => { drag.current.active = false; void imagine(veiled.current); }}
        onPointerLeave={() => { drag.current.active = false; }}
      >
        <canvas ref={canvasRef} width={64} height={64} aria-hidden
                className="pixelated absolute inset-0 h-full w-full border hairline" />
        <div role="group" aria-label="the mask: an 8 by 8 grid of patches; press a cell to hide or reveal it, then the model predicts the hidden representations"
             className="absolute inset-0 grid grid-cols-8">
          {Array.from({ length: 64 }, (_, cell) => {
            const guess = guesses.get(cell);
            return (
              <button key={cell} type="button" aria-pressed={hidden.has(cell)}
                aria-label={`patch ${cell}${hidden.has(cell) ? ", masked" : ""}${guess ? `, predicted with cosine similarity ${guess.sim.toFixed(2)}` : ""}`}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggle(cell);
                    void imagine(veiled.current);
                  }
                }}
                onPointerDown={(e) => {
                  e.preventDefault();
                  drag.current = { active: true, hide: !hidden.has(cell) };
                  toggle(cell, drag.current.hide);
                }}
                onPointerEnter={() => {
                  if (drag.current.active) toggle(cell, drag.current.hide);
                }}
                className="relative border border-transparent hover:border-[var(--ink-soft)]/40"
                style={guess ? {
                  boxShadow: `inset 0 0 0 2px rgba(25,24,23,${Math.max(0, (guess.sim - 0.3) / 0.7).toFixed(2)})`,
                } : undefined}
              />
            );
          })}
        </div>
        {thinking && (
          <p className="ui-sans absolute -bottom-6 left-0 text-xs text-[var(--ink-soft)]" role="status">
            predicting the hidden representations
          </p>
        )}
      </div>
      <div className="ui-sans flex gap-4 pt-4 text-xs">
        <button type="button" className="underline decoration-dotted underline-offset-4"
                onClick={() => { setScene(conjure(rng.current)); veiled.current = new Set();
                                 setHidden(new Set()); setGuesses(new Map()); }}>
          new image
        </button>
        <button type="button" className="underline decoration-dotted underline-offset-4"
                onClick={() => { veiled.current = new Set();
                                 setHidden(new Set()); setGuesses(new Map()); }}>
          clear the mask
        </button>
      </div>
      <p className="max-w-md text-sm leading-relaxed text-[var(--ink-soft)]">
        The model predicts the internal representation of every masked
        patch, never its pixels. Each masked cell shows its nearest
        neighbor among 8,192 reference patches; ring darkness marks the
        cosine similarity of the match.
      </p>
    </div>
  );
}
