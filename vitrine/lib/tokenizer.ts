/** The closed language's ledger: word-level ids over the shipped
 *  vocabulary, identical rules to atelier's Tokenizer: strip commas,
 *  split on spaces, ids are 2 + sorted-vocabulary index, 0 pads,
 *  1 marks the unknown, fixed length 24. Models take int64, so the
 *  encoder speaks BigInt64Array. */

export interface TokenizerSpec {
  vocabulary: string[];
  max_len: number;
  pad: number;
  unk: number;
}

export class Tokenizer {
  readonly maxLen: number;
  readonly pad: number;
  readonly unk: number;
  readonly vocabulary: string[];
  private ids = new Map<string, number>();

  constructor(spec: TokenizerSpec) {
    this.maxLen = spec.max_len;
    this.pad = spec.pad;
    this.unk = spec.unk;
    this.vocabulary = spec.vocabulary;
    spec.vocabulary.forEach((word, i) => this.ids.set(word, i + 2));
  }

  /** One caption -> (max_len,) int64 ids, padded with zeros. */
  encode(caption: string): BigInt64Array {
    const out = new BigInt64Array(this.maxLen).fill(BigInt(this.pad));
    caption.replaceAll(",", "").split(/\s+/).filter(Boolean)
      .slice(0, this.maxLen)
      .forEach((word, i) => {
        out[i] = BigInt(this.ids.get(word) ?? this.unk);
      });
    return out;
  }

  /** Many captions -> flat (n * max_len,) int64, batch-major. */
  encodeBatch(captions: string[]): BigInt64Array {
    const out = new BigInt64Array(captions.length * this.maxLen);
    captions.forEach((caption, i) =>
      out.set(this.encode(caption), i * this.maxLen));
    return out;
  }
}
