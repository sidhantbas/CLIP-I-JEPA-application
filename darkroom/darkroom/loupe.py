"""Act III: the loupe. How good are the prints, and says who?

Two judges look at each print. BLEU compares its n-grams against the
five human sentences, the classic (and word-bound) verdict. The accord
asks the frozen CLIP eye itself: does the print's text embedding still
point at the photograph it came from? Human captions set the ceiling
for the accord, and shuffled pairings set its floor.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from math import exp, log

import torch

from .config import DarkroomConfig
from .contact_sheet import load_negatives, read_split

MAX_ORDER = 4


def words_of(sentence: str) -> list[str]:
    """Lowercased word tokens; punctuation falls away, apostrophes stay."""
    return re.findall(r"[a-z0-9']+", sentence.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(prints: list[list[str]],
                references: list[list[list[str]]]) -> dict[str, float]:
    """Corpus-level BLEU-1..4, the standard recipe: modified n-gram
    precision clipped against the best reference, geometric mean over
    orders, brevity penalty from the closest reference length."""
    matched = [0] * MAX_ORDER
    possible = [0] * MAX_ORDER
    print_len, ref_len = 0, 0
    for hyp, refs in zip(prints, references):
        print_len += len(hyp)
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
        for n in range(1, MAX_ORDER + 1):
            counts = _ngrams(hyp, n)
            ceiling: Counter = Counter()
            for r in refs:
                ceiling |= _ngrams(r, n)
            matched[n - 1] += sum(min(c, ceiling[g]) for g, c in counts.items())
            possible[n - 1] += max(0, len(hyp) - n + 1)

    brevity = 1.0 if print_len > ref_len else exp(1 - ref_len / max(1, print_len))
    scores: dict[str, float] = {}
    for n in range(1, MAX_ORDER + 1):
        precisions = [matched[k] / possible[k] if possible[k] else 0.0
                      for k in range(n)]
        if min(precisions) > 0:
            geometric = exp(sum(log(p) for p in precisions) / n)
        else:
            geometric = 0.0
        scores[f"bleu{n}"] = round(brevity * geometric, 4)
    return scores


def measure_accord(cfg: DarkroomConfig, photos: list[str],
                   sentences: list[str]) -> torch.Tensor:
    """Cosine between each sentence's CLIP text embedding and its
    photograph's negative. Returns one cosine per pair, on cpu."""
    from .giants import summon_eye

    eye, processor = summon_eye(cfg)
    names, negatives = load_negatives(cfg)
    index = {n: i for i, n in enumerate(names)}
    mine = negatives[torch.tensor([index[p] for p in photos])]

    cosines = []
    with torch.no_grad():
        for i in range(0, len(sentences), cfg.clip_batch):
            batch = processor.tokenizer(sentences[i:i + cfg.clip_batch],
                                        padding=True, truncation=True,
                                        return_tensors="pt").to(cfg.device)
            speech = eye.get_text_features(**batch).pooler_output
            speech = torch.nn.functional.normalize(speech, dim=-1).cpu()
            cosines.append((speech * mine[i:i + cfg.clip_batch]).sum(-1))
    return torch.cat(cosines)


def perform(cfg: DarkroomConfig) -> dict:
    """Judges results/prints.json and files the verdict to results/loupe.json."""
    sheet = json.loads((cfg.results_dir / "prints.json").read_text())
    assert [row["photo"] for row in sheet] == read_split(cfg, "test")

    hyps = [words_of(row["print"]) for row in sheet]
    refs = [[words_of(r) for r in row["references"]] for row in sheet]
    bleu = corpus_bleu(hyps, refs)

    photos = [row["photo"] for row in sheet]
    prints_accord = measure_accord(cfg, photos, [row["print"] for row in sheet])
    humans_accord = measure_accord(cfg, photos,
                                   [row["references"][0] for row in sheet])
    shuffled_accord = measure_accord(cfg, photos[1:] + photos[:1],
                                     [row["print"] for row in sheet])

    verdict = {
        "bleu": bleu,
        "accord": {
            "prints": round(prints_accord.mean().item(), 4),
            "humans": round(humans_accord.mean().item(), 4),
            "shuffled": round(shuffled_accord.mean().item(), 4),
        },
    }
    cfg.results_dir.mkdir(exist_ok=True)
    with open(cfg.results_dir / "loupe.json", "w") as f:
        json.dump(verdict, f, indent=1)

    return {
        **bleu,
        "accord (prints / humans / shuffled)":
            f'{verdict["accord"]["prints"]} / {verdict["accord"]["humans"]}'
            f' / {verdict["accord"]["shuffled"]}',
        "verdict": ("the prints hold" if bleu["bleu4"] > 0.10
                    else "BLURRY PRINTS, inspect the developer"),
    }
