/** The barman's cost heads, assembled from the shipped crates: the
 *  manifest JSON for the small tensors, raw float32 files for the
 *  bilinear table and the standardization constants. Loaded once,
 *  cached, and handed to the pure math in probes.ts. */

import { AttentiveGlanceHead, TokenGlanceHead } from "@/lib/probes";
import { loadF32, loadJson } from "@/lib/results";

interface HeadsManifest {
  press_mean: string; press_std: string;
  token_glance: {
    pair: string; pair_bias: number[]; state: string; state_bias: number[];
    goal: { weight: number[][]; bias: number[] };
  };
  attend: {
    query: number[]; key: { weight: number[][]; bias: number[] };
    value: { weight: number[][]; bias: number[] };
    pair: number[][]; pair_bias: number[];
    state: { weight: number[][]; bias: number[] };
    goal: { weight: number[][]; bias: number[] };
  };
}

export interface CostHeads {
  tokenGlance: TokenGlanceHead;
  attentive: AttentiveGlanceHead;
}

let cached: Promise<CostHeads> | null = null;

export function theCostHeads(): Promise<CostHeads> {
  if (!cached) {
    cached = (async () => {
      const m = await loadJson<HeadsManifest>("/data/heads.json");
      const [mean, std, pair, state] = await Promise.all([
        loadF32(`/data/${m.press_mean}`), loadF32(`/data/${m.press_std}`),
        loadF32(`/data/${m.token_glance.pair}`),
        loadF32(`/data/${m.token_glance.state}`),
      ]);
      return {
        tokenGlance: {
          pair, pairBias: m.token_glance.pair_bias[0],
          state, stateBias: m.token_glance.state_bias[0],
          goal: m.token_glance.goal, mean, std,
        },
        attentive: {
          query: m.attend.query, key: m.attend.key, value: m.attend.value,
          pair: m.attend.pair, pairBias: m.attend.pair_bias[0],
          state: m.attend.state, goal: m.attend.goal, mean, std,
        },
      };
    })();
  }
  return cached;
}
