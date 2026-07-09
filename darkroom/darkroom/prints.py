"""Act II: the prints. A thousand negatives become a thousand sentences.

Each test photograph's negative is developed into its prefix strip, and
the frozen tongue continues it under a beam search: five drafts kept
alive at once, the best finished one framed. No reference caption is in
the room; the prints are made blind and judged later, under the loupe.
"""

from __future__ import annotations

import json

import torch
from tqdm import tqdm

from .config import DarkroomConfig
from .contact_sheet import load_negatives, read_captions, read_split
from .developer import resurrect


def make_prints(cfg: DarkroomConfig, negatives: torch.Tensor,
                tongue, tokenizer) -> list[str]:
    """Develops each negative and lets the tongue speak, beam_width drafts
    at a time. Returns one sentence per negative."""
    developer = resurrect(cfg)
    eos = tokenizer.eos_token_id
    sentences: list[str] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(negatives), cfg.generate_batch), desc="printing"):
            prefix = developer(negatives[i:i + cfg.generate_batch].to(cfg.device))
            drafts = tongue.generate(
                inputs_embeds=prefix,
                attention_mask=torch.ones(prefix.shape[:2], dtype=torch.long,
                                          device=cfg.device),
                max_new_tokens=cfg.max_generate,
                num_beams=cfg.beam_width,
                do_sample=False,
                early_stopping=True,
                eos_token_id=eos,
                pad_token_id=eos,
            )
            for line in tokenizer.batch_decode(drafts, skip_special_tokens=True):
                sentences.append(" ".join(line.split()))
    return sentences


def perform(cfg: DarkroomConfig) -> dict:
    """Prints every test photograph and files the results.

    results/prints.json holds, per photo: the print and its five human
    references, so the loupe (and the gallery) never need the models."""
    from .giants import summon_tongue

    names, negatives = load_negatives(cfg)
    index = {n: i for i, n in enumerate(names)}
    captions = read_captions(cfg)
    test = read_split(cfg, "test")
    tongue, tokenizer = summon_tongue(cfg)

    order = torch.tensor([index[n] for n in test])
    sentences = make_prints(cfg, negatives[order], tongue, tokenizer)

    sheet = [{"photo": n, "print": s, "references": captions[n]}
             for n, s in zip(test, sentences)]
    cfg.results_dir.mkdir(exist_ok=True)
    with open(cfg.results_dir / "prints.json", "w") as f:
        json.dump(sheet, f, indent=1)

    words = [len(s.split()) for s in sentences]
    return {
        "prints made": len(sheet),
        "mean words per print": round(sum(words) / len(words), 1),
        "first print": f'{sheet[0]["photo"]}: "{sheet[0]["print"]}"',
        "verdict": "prints hung to dry",
    }
