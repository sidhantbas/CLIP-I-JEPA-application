"""Every knob in one dataclass, one seed to rule the run.

The config is the score of the whole play. Each act reads its part from
here and nowhere else, so a single seed reproduces the entire pipeline:
world generation, both trainings, probes, bridge, planner, figures.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def resolve_device() -> str:
    """Returns 'cuda' if available, else 'mps', else 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def summon_determinism(seed: int) -> None:
    """Seeds python, numpy and torch RNGs so one integer reproduces the run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AtelierConfig:
    """All hyperparameters for every act, with the single master seed."""

    seed: int = 7
    device: str = field(default_factory=resolve_device)
    results_dir: Path = Path("results")

    # Act 0, the world and its speech.
    canvas_size: int = 64
    min_shapes: int = 1
    max_shapes: int = 3
    caption_mentions_orientation_p: float = 0.5
    max_caption_len: int = 24

    # Shared width of every latent space in the play.
    latent_dim: int = 128

    # Act I, the Namer (mini CLIP).
    namer_scenes: int = 20_000
    namer_steps: int = 3_000
    namer_batch: int = 256
    namer_lr: float = 3e-4
    namer_weight_decay: float = 0.05
    namer_temperature_init: float = 0.07
    namer_log_every: int = 100

    # Act II, the Sculptor (mini JEPA).
    patch_size: int = 8
    sculptor_scenes: int = 20_000
    sculptor_steps: int = 3_000
    sculptor_batch: int = 256
    sculptor_lr: float = 3e-4
    sculptor_weight_decay: float = 0.05
    sculptor_ema_momentum: float = 0.996
    sculptor_mask_ratio: tuple[float, float] = (0.30, 0.50)
    sculptor_depth: int = 4
    sculptor_predictor_depth: int = 2
    sculptor_heads: int = 4
    sculptor_collapse_floor: float = 0.01
    sculptor_log_every: int = 100

    # Act III, the interrogation (probes).
    probe_train_scenes: int = 10_000
    probe_test_scenes: int = 2_000
    probe_seeds: int = 3
    probe_epochs: int = 40
    probe_batch: int = 512
    probe_lr: float = 1e-2
    scrutiny_hidden: int = 256
    pixel_pca_dim: int = 128

    # Act IV, the handshake (text to Sculptor space).
    handshake_hidden: int = 256
    handshake_steps: int = 1_500
    handshake_batch: int = 256
    handshake_lr: float = 1e-3
    handshake_candidates: int = 512

    # Act V, the barman (toy planner).
    barman_episodes: int = 500
    barman_max_steps: int = 12
    barman_step_px: int = 8
    barman_rollouts: int = 24_000
    barman_scorer_epochs: int = 40

    def banner(self) -> str:
        """Renders the config as a readable block for the console."""
        lines = ["atelier config"]
        for name, value in self.__dict__.items():
            lines.append(f"  {name:34s} {value}")
        return "\n".join(lines)
