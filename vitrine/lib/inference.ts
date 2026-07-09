/** One loader for every model: onnxruntime-web sessions, created lazily
 *  the first time a lab asks and cached forever after. WASM binaries
 *  are served from /ort/ so the static site owns its runtime. All
 *  helpers speak plain Float32Array in, Float32Array out; shapes are
 *  the caller's contract, stated per function. */

import * as ort from "onnxruntime-web";

let configured = false;
const sessions = new Map<string, Promise<ort.InferenceSession>>();

/** The WASM backend runs one inference at a time; every call passes
 *  through this single turnstile, so bursts of interaction queue up
 *  instead of colliding with "Session already started". */
let turnstile: Promise<unknown> = Promise.resolve();
function inTurn<T>(job: () => Promise<T>): Promise<T> {
  const turn = turnstile.then(job, job);
  turnstile = turn.catch(() => undefined);
  return turn;
}

function summon(name: string): Promise<ort.InferenceSession> {
  if (!configured) {
    ort.env.wasm.wasmPaths = "/ort/";
    ort.env.wasm.numThreads = 1;   // single-threaded: no COOP/COEP headers needed
    configured = true;
  }
  let session = sessions.get(name);
  if (!session) {
    session = ort.InferenceSession.create(`/models/${name}.onnx`);
    sessions.set(name, session);
  }
  return session;
}

/** Warm a lab's models ahead of first interaction. */
export function prepare(...names: string[]): void {
  for (const name of names) void summon(name);
}

const f32 = (data: Float32Array, dims: number[]) =>
  new ort.Tensor("float32", data, dims);
const i64 = (values: ArrayLike<number> | BigInt64Array, dims: number[]) =>
  new ort.Tensor("int64",
    values instanceof BigInt64Array
      ? values
      : BigInt64Array.from(Array.from(values as ArrayLike<number>, BigInt)),
    dims);

/** canvases (n * 3*64*64) -> gaze embeddings (n * 128). */
export async function runGaze(canvases: Float32Array, n: number): Promise<Float32Array> {
  return inTurn(async () => {
    const out = await (await summon("gaze")).run(
      { canvas: f32(canvases, [n, 3, 64, 64]) });
    return out.embedding.data as Float32Array;
  });
}

/** token ids (n * 24) -> tongue embeddings (n * 128). */
export async function runTongue(tokens: BigInt64Array, n: number): Promise<Float32Array> {
  return inTurn(async () => {
    const out = await (await summon("tongue")).run({ tokens: i64(tokens, [n, 24]) });
    return out.embedding.data as Float32Array;
  });
}

/** canvases (n * 3*64*64) with kept patch indices -> eye tokens
 *  (n * keep.length * 128). Pass all 64 indices for the full canvas. */
export async function runEye(canvases: Float32Array, n: number,
                             keep: number[]): Promise<Float32Array> {
  return inTurn(async () => {
    const out = await (await summon("eye")).run({
      canvas: f32(canvases, [n, 3, 64, 64]),
      keep: i64(keep, [keep.length]),
    });
    return out.tokens.data as Float32Array;
  });
}

/** context (n * nv * 128) plus hidden indices -> predicted latents at
 *  the hidden positions only, (n * hidden.length * 128); the model
 *  returns the full sequence and the slice happens here. */
export async function runImagination(context: Float32Array, n: number,
                                     nv: number, hidden: number[]): Promise<Float32Array> {
  return inTurn(async () => {
    const out = await (await summon("imagination")).run({
      context: f32(context, [n, nv, 128]),
      hidden: i64(hidden, [hidden.length]),
    });
    const seq = out.sequence.data as Float32Array;
    const nh = hidden.length;
    const guesses = new Float32Array(n * nh * 128);
    for (let b = 0; b < n; b++) {
      guesses.set(seq.subarray((b * (nv + nh) + nv) * 128,
                               (b + 1) * (nv + nh) * 128), b * nh * 128);
    }
    return guesses;
  });
}

/** eye tokens (n * 64 * 128) -> attentive summaries (n * 128). */
export async function runGrip(tokens: Float32Array, n: number): Promise<Float32Array> {
  return inTurn(async () => {
    const out = await (await summon("grip")).run(
      { tokens: f32(tokens, [n, 64, 128]) });
    return out.summary.data as Float32Array;
  });
}

/** tongue embeddings (n * 128) -> Sculptor-space points (n * 128),
 *  through the pooled or the attentive bridge. */
export async function runBridge(kind: "pooled" | "attentive",
                                speech: Float32Array, n: number): Promise<Float32Array> {
  return inTurn(async () => {
    const out = await (await summon(`bridge_${kind}`)).run(
      { speech: f32(speech, [n, 128]) });
    return out.reached.data as Float32Array;
  });
}
