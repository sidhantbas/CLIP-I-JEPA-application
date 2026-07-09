"""Act 0: the contact sheet. Every photograph, exposed once.

Flickr8k arrives as 8k photographs, five sentences each, and three
split lists. This act reads the sentences, then runs every photograph
through the frozen CLIP eye exactly once. What remains is the negative:
a unit-norm 512-d vector per photo, cached to results/negatives.pt.
Every later act works from the negatives; no photograph is looked at
twice until the gallery wants portraits.
"""

from __future__ import annotations

import torch
from PIL import Image
from tqdm import tqdm

from .config import DarkroomConfig

SPLITS = ("train", "dev", "test")


def read_captions(cfg: DarkroomConfig) -> dict[str, list[str]]:
    """Parses Flickr8k.token.txt into {photo filename: five sentences}.

    Lines look like `1000268201_693b08cb0e.jpg#0\tA child in a pink...`.
    Sentences keep their casing; the trailing ` .` is tidied to `.`."""
    sheet: dict[str, list[str]] = {}
    for line in (cfg.data_dir / "Flickr8k.token.txt").read_text().splitlines():
        if not line.strip():
            continue
        tag, sentence = line.split("\t", 1)
        name = tag.split("#", 1)[0]
        sentence = " ".join(sentence.split())
        if sentence.endswith(" ."):
            sentence = sentence[:-2] + "."
        sheet.setdefault(name, []).append(sentence)
    return sheet


def read_split(cfg: DarkroomConfig, split: str) -> list[str]:
    """Reads one of the official split lists, keeping only photos that
    actually exist in the mirror (a handful are listed but absent)."""
    listing = cfg.data_dir / f"Flickr_8k.{split}Images.txt"
    names = [n for n in listing.read_text().split() if n]
    return [n for n in names if (cfg.image_dir / n).exists()]


def load_negatives(cfg: DarkroomConfig) -> tuple[list[str], torch.Tensor]:
    """Loads the cached negatives: (names, unit-norm float tensor (N, 512))."""
    relic = torch.load(cfg.results_dir / "negatives.pt", map_location="cpu")
    return relic["names"], relic["negatives"]


def expose(cfg: DarkroomConfig, names: list[str]) -> torch.Tensor:
    """Runs the frozen eye over the named photographs, returns (N, 512)
    unit-norm features on cpu."""
    from .giants import summon_eye

    eye, processor = summon_eye(cfg)
    negatives = []
    with torch.no_grad():
        for i in tqdm(range(0, len(names), cfg.clip_batch), desc="exposing"):
            photos = [Image.open(cfg.image_dir / n).convert("RGB")
                      for n in names[i:i + cfg.clip_batch]]
            pixels = processor(images=photos, return_tensors="pt")["pixel_values"]
            # transformers 5 returns the vision outputs; the projected
            # 512-d features ride in pooler_output.
            features = eye.get_image_features(
                pixel_values=pixels.to(cfg.device)).pooler_output
            negatives.append(torch.nn.functional.normalize(features, dim=-1).cpu())
    return torch.cat(negatives).float()


def perform(cfg: DarkroomConfig) -> dict:
    """Exposes every photograph in the three splits and caches the negatives.

    The exposure is deterministic (the eye is frozen), so an existing,
    complete cache is trusted rather than re-shot."""
    captions = read_captions(cfg)
    splits = {s: read_split(cfg, s) for s in SPLITS}
    names = sorted({n for split in splits.values() for n in split})

    cache = cfg.results_dir / "negatives.pt"
    if cache.exists():
        cached_names, negatives = load_negatives(cfg)
        if cached_names == names:
            return {
                "photographs": len(names),
                "sentences": sum(len(captions[n]) for n in names),
                "splits": {s: len(v) for s, v in splits.items()},
                "verdict": "negatives already exposed, cache trusted",
            }

    negatives = expose(cfg, names)
    cfg.results_dir.mkdir(exist_ok=True)
    torch.save({"names": names, "negatives": negatives}, cache)
    return {
        "photographs": len(names),
        "sentences": sum(len(captions[n]) for n in names),
        "splits": {s: len(v) for s, v in splits.items()},
        "negative dim": negatives.shape[1],
        "verdict": "contact sheet exposed",
    }
