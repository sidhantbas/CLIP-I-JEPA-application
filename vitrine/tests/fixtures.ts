/** Shared scaffolding for the gates: the parity manifest, ONNX sessions
 *  through onnxruntime-node, and deviation arithmetic. Tests read the
 *  crates straight from ../export so they run before any gathering. */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-node";

const CRATES = join(__dirname, "..", "..", "export");

export const parity = JSON.parse(
  readFileSync(join(CRATES, "parity.json"), "utf8"));

export function crateJson(name: string) {
  return JSON.parse(readFileSync(join(CRATES, "out", "data", name), "utf8"));
}

export function crateF32(name: string): Float32Array {
  const raw = readFileSync(join(CRATES, "out", "data", name));
  return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
}

export async function session(name: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(join(CRATES, "out", "models", `${name}.onnx`));
}

export const f32 = (data: Float32Array | number[], dims: number[]) =>
  new ort.Tensor("float32", Float32Array.from(data as number[]), dims);
export const i64 = (data: BigInt64Array | number[], dims: number[]) =>
  new ort.Tensor("int64",
    data instanceof BigInt64Array ? data : BigInt64Array.from(data.map(BigInt)),
    dims);

/** base64 HWC uint8 canvas -> CHW float32 in [0,1]. */
export function canvasCHW(b64: string): Float32Array {
  const bytes = Buffer.from(b64, "base64");
  const out = new Float32Array(3 * 64 * 64);
  for (let i = 0; i < 64 * 64; i++) {
    for (let c = 0; c < 3; c++) out[c * 64 * 64 + i] = bytes[i * 3 + c] / 255;
  }
  return out;
}

/** base64 HWC uint8 canvas -> plain byte array, for pixel comparison. */
export function canvasBytes(b64: string): Uint8Array {
  return Uint8Array.from(Buffer.from(b64, "base64"));
}

const flatten = (x: unknown): number[] =>
  Array.isArray(x) ? x.flatMap(flatten) : [x as number];

/** Largest absolute elementwise gap between truth (nested lists from
 *  the manifest) and the live output. */
export function maxAbsDev(truth: unknown, live: ArrayLike<number>): number {
  const flat = flatten(truth);
  if (flat.length !== live.length) {
    throw new Error(`shape mismatch: ${flat.length} vs ${live.length}`);
  }
  let worst = 0;
  for (let i = 0; i < flat.length; i++) {
    worst = Math.max(worst, Math.abs(flat[i] - live[i]));
  }
  return worst;
}

export function stack(scenes: Array<{ canvas_b64: string }>): Float32Array {
  const out = new Float32Array(scenes.length * 3 * 64 * 64);
  scenes.forEach((s, i) => out.set(canvasCHW(s.canvas_b64), i * 3 * 64 * 64));
  return out;
}
