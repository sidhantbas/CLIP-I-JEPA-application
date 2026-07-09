/** Gate W: the TypeScript world is the same world. The PRNG is not
 *  numpy's, so the gate is (a) factor-replayed renders match Python's
 *  pixels, (b) TS captions parse back to TS factors exactly, and
 *  (c) the ONNX gaze retrieves captions for TS-rendered scenes within
 *  3 points of the Python retrieval reference. Chosen gate and result
 *  are recorded in the vitrine README. */

import { describe, expect, it } from "vitest";
import { Tokenizer } from "../lib/tokenizer";
import {
  COMPASS_8, ShapeFactors, SIZE, conjure, mulberry32, render, sizeWord,
  toModelInput,
} from "../lib/world";
import { canvasBytes, crateJson, f32, i64, parity, session } from "./fixtures";

/** Parses a caption back to its stated facts, mirroring recite_back. */
function reciteBack(caption: string) {
  const clauses = caption.split(", ");
  const head = clauses[0].split(" ");
  const fact = { sizeWord: head[1], color: head[2], form: head[3],
                 grid: head.slice(7).join(" "), facing: "",
                 beside: [] as string[][] };
  for (const clause of clauses.slice(1)) {
    const part = clause.split(" ");
    if (part[0] === "facing") fact.facing = part[1];
    else fact.beside.push([part[2], part[3], part[4]]);
  }
  return fact;
}

describe("Gate W: the world is the same world", () => {
  it("factor-replayed renders match Python pixels", () => {
    let worstMad = 0;
    for (const scene of parity.scenes) {
      const factors = scene.factors as ShapeFactors[];
      const mine = render(factors);
      const theirs = canvasBytes(scene.canvas_b64);
      let total = 0;
      for (let i = 0; i < theirs.length; i++) {
        total += Math.abs(Math.round(mine[i] * 255) - theirs[i]);
      }
      worstMad = Math.max(worstMad, total / theirs.length);
    }
    console.log(`worst per-scene pixel MAD ${worstMad.toFixed(4)} (of 255)`);
    expect(worstMad).toBeLessThan(1.0);
  });

  it("captions parse back to factors, vocabulary stays closed", () => {
    const rng = mulberry32(7);
    const spec = crateJson("tokenizer.json");
    const known = new Set(spec.vocabulary as string[]);
    for (let i = 0; i < 300; i++) {
      const scene = conjure(rng);
      const fact = reciteBack(scene.caption);
      const primary = scene.factors[0];
      expect(fact.form).toBe(primary.form);
      expect(fact.color).toBe(primary.color);
      expect(fact.grid).toBe(primary.grid);
      expect(fact.sizeWord).toBe(sizeWord(primary));
      if (fact.facing) expect(fact.facing).toBe(primary.facing);
      expect(fact.beside.length).toBe(scene.factors.length - 1);
      expect(COMPASS_8).toContain(primary.facing);
      for (const word of scene.caption.replaceAll(",", "").split(" ")) {
        expect(known.has(word)).toBe(true);
      }
    }
  });

  it("the gaze retrieves TS scenes within 3 points of Python", async () => {
    const n = 1024;
    const rng = mulberry32(7 + 6);
    const scenes = Array.from({ length: n }, () => conjure(rng));
    const canvases = new Float32Array(n * 3 * SIZE * SIZE);
    scenes.forEach((s, i) =>
      canvases.set(toModelInput(render(s.factors)), i * 3 * SIZE * SIZE));
    const tokenizer = new Tokenizer(crateJson("tokenizer.json"));
    const tokens = tokenizer.encodeBatch(scenes.map((s) => s.caption));
    const sight = (await (await session("gaze")).run(
      { canvas: f32(canvases, [n, 3, 64, 64]) })).embedding.data as Float32Array;
    const speech = (await (await session("tongue")).run(
      { tokens: i64(tokens, [n, 24]) })).embedding.data as Float32Array;
    const unit = (v: Float32Array) => {
      let s = 0;
      for (const x of v) s += x * x;
      return v.map((x) => x / Math.sqrt(s));
    };
    let hits = 0;
    const speeches = Array.from({ length: n }, (_, j) =>
      unit(speech.slice(j * 128, (j + 1) * 128)));
    for (let i = 0; i < n; i++) {
      const s = unit(sight.slice(i * 128, (i + 1) * 128));
      let best = -Infinity, arg = -1;
      for (let j = 0; j < n; j++) {
        let sim = 0;
        for (let k = 0; k < 128; k++) sim += s[k] * speeches[j][k];
        if (sim > best) { best = sim; arg = j; }
      }
      if (arg === i) hits += 1;
    }
    const mine = hits / n;
    const reference = parity.world_reference.python_retrieval_at_1;
    console.log(`TS retrieval@1 ${mine.toFixed(4)} vs Python ${reference}`);
    expect(Math.abs(mine - reference)).toBeLessThan(0.03);
  });
});
