/** Typed loaders for the shipped truths: the atelier results JSONs, the
 *  exhibit data and the raw binary crates, fetched once and cached.
 *  Nothing here invents a number; it only carries them. */

export interface Cell { mean: number; std: number }
export interface Interrogation {
  chance: Record<string, number>;
  namer: Record<string, { glance: Cell; scrutiny: Cell }>;
  sculptor: Record<string, { glance: Cell; scrutiny: Cell }>;
  pixels: Record<string, { glance: Cell; scrutiny: Cell }>;
}
export interface Unpooled {
  chance: Record<string, number>;
  namer: Record<string, Record<string, Cell>>;
  sculptor: Record<string, Record<string, Cell>>;
  pixels: Record<string, Record<string, Cell>>;
}
export interface BarmanUnpooled {
  namer: { glance: number; scrutiny: number };
  sculptor: Record<string, number>;
  episodes: number;
  verdict: string;
}
export interface HandshakeUnpooled {
  candidates: number;
  "pooled@1": number; "pooled@5": number;
  "attentive@1": number; "attentive@5": number;
  attentive_retrieval_at_k: number[];
  bottleneck: string;
}
export interface Constellation {
  spaces: Record<string, { center: number[]; axes: number[][] }>;
  points: Array<{ pooled: number[]; attentive: number[]; caption: string;
                  form: string; color: string; grid: string }>;
}
export interface TokenizerSpecFile {
  vocabulary: string[]; max_len: number; pad: number; unk: number;
}

const jsonCache = new Map<string, Promise<unknown>>();
const bufferCache = new Map<string, Promise<ArrayBuffer>>();

export function loadJson<T>(path: string): Promise<T> {
  let hit = jsonCache.get(path);
  if (!hit) {
    hit = fetch(path).then((r) => {
      if (!r.ok) throw new Error(`missing crate: ${path}`);
      return r.json();
    });
    jsonCache.set(path, hit);
  }
  return hit as Promise<T>;
}

async function loadBuffer(path: string): Promise<ArrayBuffer> {
  let hit = bufferCache.get(path);
  if (!hit) {
    hit = fetch(path).then((r) => {
      if (!r.ok) throw new Error(`missing crate: ${path}`);
      return r.arrayBuffer();
    });
    bufferCache.set(path, hit);
  }
  return hit;
}

export const loadF32 = async (path: string): Promise<Float32Array> =>
  new Float32Array(await loadBuffer(path));
export const loadU8 = async (path: string): Promise<Uint8Array> =>
  new Uint8Array(await loadBuffer(path));

export const theInterrogation = () =>
  loadJson<Interrogation>("/results/interrogation.json");
export const theUnpooled = () =>
  loadJson<Unpooled>("/results/interrogation_unpooled.json");
export const theBarman = () =>
  loadJson<BarmanUnpooled>("/results/barman_unpooled.json");
export const theHandshake = () =>
  loadJson<HandshakeUnpooled>("/results/handshake_unpooled.json");
export const theConstellation = () =>
  loadJson<Constellation>("/data/constellation.json");
export const theTokenizerSpec = () =>
  loadJson<TokenizerSpecFile>("/data/tokenizer.json");
