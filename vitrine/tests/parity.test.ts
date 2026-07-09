/** Gate P: every exported ONNX model, run on the 32 fixed seed-7
 *  scenes, must match its PyTorch output within 1e-4. The imagination,
 *  grip and bridges are fed the manifest's own upstream outputs so
 *  each graph is judged alone, not through its neighbors' noise. */

import { describe, expect, it } from "vitest";
import { f32, i64, maxAbsDev, parity, session, stack } from "./fixtures";

const scenes = parity.scenes as Array<{ canvas_b64: string; tokens: number[] }>;
const truth = parity.outputs;
const TOL = parity.tolerance as number;
const B = scenes.length;

const flat = (rows: number[][]) => rows.flat();

describe("Gate P: the browser runs the same models", () => {
  it("gaze", async () => {
    const out = await (await session("gaze")).run(
      { canvas: f32(stack(scenes), [B, 3, 64, 64]) });
    const dev = maxAbsDev(truth.gaze, out.embedding.data as Float32Array);
    console.log(`gaze max deviation ${dev.toExponential(2)}`);
    expect(dev).toBeLessThan(TOL);
  });

  it("tongue", async () => {
    const tokens = scenes.flatMap((s) => s.tokens);
    const out = await (await session("tongue")).run(
      { tokens: i64(tokens, [B, 24]) });
    const dev = maxAbsDev(truth.tongue, out.embedding.data as Float32Array);
    console.log(`tongue max deviation ${dev.toExponential(2)}`);
    expect(dev).toBeLessThan(TOL);
  });

  it("eye, unveiled and veiled", async () => {
    const eye = await session("eye");
    const canvases = f32(stack(scenes), [B, 3, 64, 64]);
    const all = await eye.run({ canvas: canvases,
      keep: i64(Array.from({ length: 64 }, (_, i) => i), [64]) });
    const full = maxAbsDev(truth.eye, all.tokens.data as Float32Array);
    const veiled = await eye.run({ canvas: canvases,
      keep: i64(parity.veil.visible, [parity.veil.visible.length]) });
    const ctx = maxAbsDev(truth.context, veiled.tokens.data as Float32Array);
    console.log(`eye full ${full.toExponential(2)}, veiled ${ctx.toExponential(2)}`);
    expect(full).toBeLessThan(TOL);
    expect(ctx).toBeLessThan(TOL);
  });

  it("imagination", async () => {
    const nv = parity.veil.visible.length;
    const nh = parity.veil.hidden.length;
    const out = await (await session("imagination")).run({
      context: f32(flat(truth.context.map(flat)), [B, nv, 128]),
      hidden: i64(parity.veil.hidden, [nh]),
    });
    const seq = out.sequence.data as Float32Array;
    const guesses = new Float32Array(B * nh * 128);
    for (let b = 0; b < B; b++) {
      guesses.set(seq.subarray((b * (nv + nh) + nv) * 128,
                               (b + 1) * (nv + nh) * 128), b * nh * 128);
    }
    const dev = maxAbsDev(truth.imagination, guesses);
    console.log(`imagination max deviation ${dev.toExponential(2)}`);
    expect(dev).toBeLessThan(TOL);
  });

  it("grip and both bridges", async () => {
    const eyeTokens = f32(flat(truth.eye.map(flat)), [B, 64, 128]);
    const speech = f32(flat(truth.tongue), [B, 128]);
    const grip = await (await session("grip")).run({ tokens: eyeTokens });
    const pooled = await (await session("bridge_pooled")).run({ speech });
    const attentive = await (await session("bridge_attentive")).run({ speech });
    const devs = {
      grip: maxAbsDev(truth.grip, grip.summary.data as Float32Array),
      pooled: maxAbsDev(truth.bridge_pooled, pooled.reached.data as Float32Array),
      attentive: maxAbsDev(truth.bridge_attentive,
                           attentive.reached.data as Float32Array),
    };
    console.log(`grip ${devs.grip.toExponential(2)}, bridges ` +
      `${devs.pooled.toExponential(2)} / ${devs.attentive.toExponential(2)}`);
    for (const dev of Object.values(devs)) expect(dev).toBeLessThan(TOL);
  });
});
