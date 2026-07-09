/** The world, reborn in the browser.
 *
 * A faithful port of atelier/world.py: same factors, same signed
 * distance rendering (float64 in both languages), same nose notch that
 * gives every form a visible heading, same caption templates over the
 * same closed vocabulary. One documented difference: the PRNG is
 * mulberry32, not numpy's PCG64, so sampled scenes differ from Python
 * draws; rendering given factors is bit-comparable, and Gate W tests
 * factor-replayed pixels plus caption faithfulness plus retrieval. */

export const FORMS = ["triangle", "square", "circle", "star"] as const;
export const COLORS: Record<string, [number, number, number]> = {
  red: [0.9, 0.12, 0.12], blue: [0.15, 0.35, 0.92],
  green: [0.1, 0.75, 0.3], yellow: [0.95, 0.85, 0.12],
  violet: [0.62, 0.22, 0.85],
};
export const COMPASS_8 = ["east", "north-east", "north", "north-west",
  "west", "south-west", "south", "south-east"] as const;
export const GRID_NAMES = [
  ["north-west", "north", "north-east"],
  ["west", "center", "east"],
  ["south-west", "south", "south-east"],
] as const;
const BACKGROUND: [number, number, number] = [0.08, 0.08, 0.1];
export const SIZE = 64;

export interface ShapeFactors {
  form: string; color: string; x: number; y: number; grid: string;
  angle: number; facing: string; size: number;
}
export interface Scene { factors: ShapeFactors[]; caption: string }

export const sizeWord = (f: ShapeFactors): string =>
  f.size < 11.5 ? "small" : "large";

/** Deterministic 32-bit PRNG; the documented stand-in for PCG64. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Signed distance from one pixel center to a polygon, negative inside;
 *  the same winding-sign algorithm as the Python, per pixel. */
function polygonSdf(px: number, py: number, verts: number[][]): number {
  let d = Infinity;
  let sign = 1;
  const n = verts.length;
  for (let i = 0; i < n; i++) {
    const j = (i - 1 + n) % n;
    const ex = verts[j][0] - verts[i][0], ey = verts[j][1] - verts[i][1];
    const wx = px - verts[i][0], wy = py - verts[i][1];
    const t = Math.min(1, Math.max(0, (wx * ex + wy * ey) / (ex * ex + ey * ey)));
    const bx = wx - ex * t, by = wy - ey * t;
    d = Math.min(d, bx * bx + by * by);
    const c1 = py >= verts[i][1], c2 = py < verts[j][1], c3 = ex * wy > ey * wx;
    if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) sign = -sign;
  }
  return sign * Math.sqrt(d);
}

function shapeVerts(f: ShapeFactors): number[][] {
  const a = (f.angle * Math.PI) / 180;
  const step = { triangle: 120, square: 90, star: 36 }[f.form] ?? 120;
  const count = f.form === "star" ? 10 : f.form === "square" ? 4 : 3;
  const offset = f.form === "square" ? Math.PI / 4 : 0;
  const verts: number[][] = [];
  for (let k = 0; k < count; k++) {
    const angle = a + offset + (k * step * Math.PI) / 180;
    const r = f.form === "star" && k % 2 === 1 ? f.size * 0.45 : f.size;
    verts.push([f.x + r * Math.cos(angle), f.y - r * Math.sin(angle)]);
  }
  return verts;
}

/** Anti-aliased coverage of one shape at one pixel, nose punched out. */
function shapeAlpha(px: number, py: number, f: ShapeFactors,
                    verts: number[][] | null): number {
  const a = (f.angle * Math.PI) / 180;
  const sdf = f.form === "circle"
    ? Math.hypot(px - f.x, py - f.y) - f.size
    : polygonSdf(px, py, verts as number[][]);
  const alpha = Math.min(1, Math.max(0, 0.5 - sdf));
  const nx = f.x + 0.55 * f.size * Math.cos(a);
  const ny = f.y - 0.55 * f.size * Math.sin(a);
  const noseSdf = Math.hypot(px - nx, py - ny) - 0.22 * f.size;
  return alpha * (1 - Math.min(1, Math.max(0, 0.5 - noseSdf)));
}

/** Renders factors to RGB float64 in [0,1], HWC row-major, 64x64x3.
 *  Shapes drawn smallest first so the primary is never occluded. */
export function render(factors: ShapeFactors[]): Float64Array {
  const out = new Float64Array(SIZE * SIZE * 3);
  for (let i = 0; i < out.length; i += 3) out.set(BACKGROUND, i);
  const order = [...factors].sort((p, q) => p.size - q.size);
  for (const f of order) {
    const verts = f.form === "circle" ? null : shapeVerts(f);
    const color = COLORS[f.color];
    for (let row = 0; row < SIZE; row++) {
      for (let col = 0; col < SIZE; col++) {
        const alpha = shapeAlpha(col + 0.5, row + 0.5, f, verts);
        if (alpha <= 0) continue;
        const at = (row * SIZE + col) * 3;
        for (let c = 0; c < 3; c++) {
          out[at + c] = out[at + c] * (1 - alpha) + color[c] * alpha;
        }
      }
    }
  }
  return out;
}

export function cellOf(x: number, y: number): string {
  const cell = SIZE / 3;
  return GRID_NAMES[Math.min(Math.floor(y / cell), 2)][Math.min(Math.floor(x / cell), 2)];
}

export function birthShape(rng: () => number): ShapeFactors {
  const size = 8 + rng() * 7;
  const lo = size + 1, hi = SIZE - size - 1;
  const x = lo + rng() * (hi - lo), y = lo + rng() * (hi - lo);
  const angle = rng() * 360;
  return {
    form: FORMS[Math.floor(rng() * 4)],
    color: Object.keys(COLORS)[Math.floor(rng() * 5)],
    x, y, grid: cellOf(x, y), angle,
    facing: COMPASS_8[Math.floor(((angle + 22.5) % 360) / 45)], size,
  };
}

/** The caption: primary with place, sometimes facing, then companions. */
export function speak(factors: ShapeFactors[], mentionFacing: boolean): string {
  const p = factors[0];
  let caption = `a ${sizeWord(p)} ${p.color} ${p.form} rests in the ${p.grid}`;
  if (mentionFacing) caption += `, facing ${p.facing}`;
  for (const o of factors.slice(1)) {
    caption += `, beside a ${sizeWord(o)} ${o.color} ${o.form}`;
  }
  return caption;
}

/** One scene: 1 to 3 shapes sorted largest first, captioned. */
export function conjure(rng: () => number, orientationP = 0.5): Scene {
  const count = 1 + Math.floor(rng() * 3);
  const factors = Array.from({ length: count }, () => birthShape(rng))
    .sort((p, q) => q.size - p.size);
  return { factors, caption: speak(factors, rng() < orientationP) };
}

/** HWC float64 -> CHW float32 in [0,1], the models' plate. */
export function toModelInput(rgb: Float64Array): Float32Array {
  const out = new Float32Array(3 * SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    for (let c = 0; c < 3; c++) out[c * SIZE * SIZE + i] = rgb[i * 3 + c];
  }
  return out;
}
