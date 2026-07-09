"""Every knob in one dataclass, one seed for the run."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def summon_determinism(seed: int) -> None:
    """Seeds python, numpy and torch RNGs so one integer reproduces the run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DarkroomConfig:
    seed: int = 7
    device: str = field(default_factory=pick_device)

    # the frozen giants
    clip_name: str = "openai/clip-vit-base-patch32"   # 512-d image/text embeddings
    gpt2_name: str = "gpt2"                           # 124M, 768-d token embeddings
    clip_dim: int = 512
    gpt2_dim: int = 768

    # the developer (the only trainable part)
    prefix_len: int = 10          # how many pseudo-tokens one photo becomes
    mapper_hidden: int = 3840     # MLP waist, (prefix_len * gpt2_dim) // 2

    # training
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_caption_tokens: int = 48  # GPT-2 BPE tokens, incl. the closing eos
    log_every: int = 50

    # inference
    beam_width: int = 5
    max_generate: int = 40
    generate_batch: int = 32      # photos printed per generate() call
    clip_batch: int = 128         # photos exposed per CLIP forward

    # places
    data_dir: Path = ROOT / "data"
    image_dir: Path = ROOT / "data" / "Flicker8k_Dataset"   # the mirror's own spelling
    results_dir: Path = ROOT / "results"

    def banner(self) -> str:
        """Renders the config as a readable block for the console."""
        lines = ["darkroom config"]
        for name, value in self.__dict__.items():
            lines.append(f"  {name:24s} {value}")
        return "\n".join(lines)
