/** The readouts, as arithmetic: linear glances, the attentive probe and
 *  the barman's cost heads, re-expressed as plain loops over Float32
 *  and Float64 arrays. Every function mirrors one dumped Python module
 *  and is held to it by probes.test.ts on the parity scenes. */

export interface LinearWeights { weight: number[][]; bias: number[] }
export interface AttendWeights {
  query: number[]; key: LinearWeights; value: LinearWeights;
  head: LinearWeights;
}

export type Vec = ArrayLike<number>;

export function standardize(x: Vec, mean: Vec, std: Vec): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - mean[i]) / std[i];
  return out;
}

/** y = W x + b, with W as (rows, cols) nested lists. */
export function linear(x: Vec, w: LinearWeights): Float64Array {
  const out = new Float64Array(w.bias.length);
  for (let r = 0; r < w.weight.length; r++) {
    let sum = w.bias[r];
    const row = w.weight[r];
    for (let c = 0; c < row.length; c++) sum += row[c] * x[c];
    out[r] = sum;
  }
  return out;
}

export function dot(a: Vec, b: Vec): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function normalize(x: Vec): Float64Array {
  const scale = 1 / Math.sqrt(dot(x, x));
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale;
  return out;
}

function softmax(scores: Float64Array): Float64Array {
  const top = Math.max(...scores);
  let total = 0;
  const out = scores.map((s) => { const e = Math.exp(s - top); total += e; return e; });
  return out.map((e) => e / total);
}

/** One learnable query attends over tokens (n, d) flat; the summary is
 *  the attention-weighted value projection, then the head classifies.
 *  Exactly interrogation_unpooled.Attend. */
export function attendForward(tokens: Vec, n: number, d: number,
                              w: AttendWeights): Float64Array {
  const scores = new Float64Array(n);
  const values: Float64Array[] = [];
  for (let t = 0; t < n; t++) {
    const token = Array.prototype.slice.call(tokens, t * d, (t + 1) * d);
    scores[t] = dot(linear(token, w.key), w.query) / Math.sqrt(d);
    values.push(linear(token, w.value));
  }
  const weights = softmax(scores);
  const summary = new Float64Array(d);
  for (let t = 0; t < n; t++) {
    for (let i = 0; i < d; i++) summary[i] += weights[t] * values[t][i];
  }
  return linear(summary, w.head);
}

export interface TokenGlanceHead {
  pair: Float32Array;        // (wide * d) bilinear table, row-major by state dim
  pairBias: number; state: Float32Array; stateBias: number;
  goal: LinearWeights; mean: Float32Array; std: Float32Array;
}

/** cost = z W g + w_s z + w_g g + biases, z the standardized flat tokens.
 *  Exactly barman_heads.TokenGlance. */
export function tokenGlanceCost(tokensFlat: Vec, goal: Vec,
                                head: TokenGlanceHead): number {
  const d = goal.length;
  let cost = head.pairBias + head.stateBias + linear(goal, head.goal)[0];
  for (let i = 0; i < tokensFlat.length; i++) {
    const z = (tokensFlat[i] - head.mean[i]) / head.std[i];
    if (z === 0) continue;
    let inner = head.state[i];
    const row = i * d;
    for (let j = 0; j < d; j++) inner += head.pair[row + j] * goal[j];
    cost += z * inner;
  }
  return cost;
}

export interface AttentiveGlanceHead {
  query: number[]; key: LinearWeights; value: LinearWeights;
  pair: number[][]; pairBias: number;
  state: LinearWeights; goal: LinearWeights;
  mean: Float32Array; std: Float32Array;
}

/** Attentive pooling over standardized tokens, then bilinear with the
 *  goal. Exactly barman_heads.AttentiveGlance. */
export function attentiveGlanceCost(tokensFlat: Vec, n: number, d: number,
                                    goal: Vec, head: AttentiveGlanceHead): number {
  const scores = new Float64Array(n);
  const values: Float64Array[] = [];
  for (let t = 0; t < n; t++) {
    const z = new Float64Array(d);
    for (let i = 0; i < d; i++) {
      const at = t * d + i;
      z[i] = (tokensFlat[at] - head.mean[at]) / head.std[at];
    }
    scores[t] = dot(linear(z, head.key), head.query) / Math.sqrt(d);
    values.push(linear(z, head.value));
  }
  const weights = softmax(scores);
  const summary = new Float64Array(d);
  for (let t = 0; t < n; t++) {
    for (let i = 0; i < d; i++) summary[i] += weights[t] * values[t][i];
  }
  let cost = head.pairBias + linear(summary, head.state)[0]
    + linear(goal, head.goal)[0];
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) cost += summary[i] * head.pair[i][j] * goal[j];
  }
  return cost;
}

/** The pooled glance cost: one minus cosine, no weights at all. */
export function pooledGlanceCost(feature: Vec, goal: Vec): number {
  return 1 - dot(normalize(feature), normalize(goal));
}
