/** The readout arithmetic gate: lib/probes.ts, fed the dumped weights,
 *  must reproduce the Python probe logits and cost-head values on the
 *  parity scenes. Features are rebuilt from the manifest's own eye and
 *  gaze outputs, the exact tensors the Python probes saw. */

import { describe, expect, it } from "vitest";
import {
  AttendWeights, attendForward, attentiveGlanceCost, linear, normalize,
  standardize, tokenGlanceCost,
} from "../lib/probes";
import { crateF32, crateJson, maxAbsDev, parity } from "./fixtures";

const truth = parity.outputs;
const probes = crateJson("probes.json");
const heads = crateJson("heads.json");
const TOL = 1e-3;   // long f32 dot products drift past 1e-4; stated honestly

const eyeTokens = (i: number): number[] => (truth.eye[i] as number[][]).flat();
const pooledSculptor = (i: number): number[] => {
  const out = new Array(128).fill(0);
  for (const token of truth.eye[i] as number[][]) {
    token.forEach((v, k) => { out[k] += v / 64; });
  }
  return out;
};

describe("probe arithmetic matches the dumped Python outputs", () => {
  it("linear glances, pooled and token-wide", () => {
    for (const name of Object.keys(probes)) {
      const spec = probes[name];
      if (spec.kind !== "glance") continue;
      for (let i = 0; i < 8; i++) {
        const feats = name.includes("glance_tokens") ? eyeTokens(i)
          : name.startsWith("namer") ? truth.gaze[i] : pooledSculptor(i);
        const logits = linear(standardize(feats, spec.mean, spec.std), spec);
        const dev = maxAbsDev(truth.probes[name][i], logits);
        expect(dev, name).toBeLessThan(TOL);
      }
    }
  });

  it("the attentive probe", () => {
    for (const factor of ["position", "orientation"]) {
      const name = `sculptor.${factor}.attend`;
      const spec = probes[name];
      const mean = (spec.mean as number[][]).flat();   // trained on (64, 128)
      const std = (spec.std as number[][]).flat();
      for (let i = 0; i < 8; i++) {
        const z = standardize(eyeTokens(i), mean, std);
        const logits = attendForward(z, 64, 128, spec as AttendWeights);
        expect(maxAbsDev(truth.probes[name][i], logits), name).toBeLessThan(TOL);
      }
    }
  });

  it("the barman cost heads", () => {
    const mean = crateF32(heads.press_mean), std = crateF32(heads.press_std);
    const tokenGlance = {
      pair: crateF32(heads.token_glance.pair),
      pairBias: heads.token_glance.pair_bias[0],
      state: crateF32(heads.token_glance.state),
      stateBias: heads.token_glance.state_bias[0],
      goal: heads.token_glance.goal, mean, std,
    };
    const attentive = {
      query: heads.attend.query, key: heads.attend.key,
      value: heads.attend.value, pair: heads.attend.pair,
      pairBias: heads.attend.pair_bias[0], state: heads.attend.state,
      goal: heads.attend.goal, mean, std,
    };
    for (let i = 0; i < 8; i++) {
      const goal = normalize(truth.bridge_pooled[i]);
      const flat = eyeTokens(i);
      const tg = tokenGlanceCost(flat, goal, tokenGlance);
      const at = attentiveGlanceCost(flat, 64, 128, goal, attentive);
      expect(Math.abs(tg - truth.heads.token_glance[i]), "token_glance")
        .toBeLessThan(TOL);
      expect(Math.abs(at - truth.heads.attend[i]), "attend").toBeLessThan(TOL);
    }
  });
});
