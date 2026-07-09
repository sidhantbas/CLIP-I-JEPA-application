"""The frozen giants. Neither takes a single gradient step.

The eye is CLIP ViT-B/32: it looks at a photograph once and leaves a
512-d negative behind. The tongue is GPT-2 (124M): it can continue any
sequence of 768-d token embeddings, and never learns a new word here.
Everything the darkroom teaches lives in the small developer between
them; the giants only lend their light.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
)

from .config import DarkroomConfig


def _freeze(model: torch.nn.Module, device: str) -> torch.nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device).eval()


def summon_eye(cfg: DarkroomConfig):
    """Returns (clip_model, clip_processor), frozen, on cfg.device.

    The processor bundles the image pipeline (resize, crop, normalize)
    and the CLIP text tokenizer; the loupe uses both sides."""
    eye = _freeze(CLIPModel.from_pretrained(cfg.clip_name), cfg.device)
    processor = AutoProcessor.from_pretrained(cfg.clip_name)
    return eye, processor


def summon_tongue(cfg: DarkroomConfig):
    """Returns (gpt2_lm, gpt2_tokenizer), frozen, on cfg.device.

    GPT-2 has no pad token of its own; we let eos stand in for it and
    mask attention ourselves, so the borrowed vocabulary stays intact."""
    tongue = _freeze(AutoModelForCausalLM.from_pretrained(cfg.gpt2_name), cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.gpt2_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tongue, tokenizer
