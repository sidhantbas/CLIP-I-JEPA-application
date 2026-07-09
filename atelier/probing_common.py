"""The interrogation room's shared tools.

Everything a probing act needs regardless of what it probes: the factor
tables, the ground-truth labeler, the PCA floor, the probe bodies, and
the one training loop every probe passes through. Both sittings of the
interrogation (pooled, Act III, and unpooled, Act III second sitting)
import from here, so their numbers are comparable by construction.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

from .config import AtelierConfig
from .world import COLORS, COMPASS_8, FORMS, GRID_NAMES, Scene

GRID_FLAT: list[str] = [name for row in GRID_NAMES for name in row]
FACTORS: dict[str, int] = {"form": 4, "color": 5, "position": 9, "orientation": 8}
CHANCE: dict[str, float] = {"form": 1 / 4, "color": 1 / 5,
                            "position": 1 / 9, "orientation": 1 / 8}


def testify(scenes: list[Scene], factor: str) -> Tensor:
    """Ground-truth class labels of the primary (largest) shape, (N,) long."""
    primary = [s.factors[0] for s in scenes]
    if factor == "form":
        ids = [FORMS.index(p.form) for p in primary]
    elif factor == "color":
        ids = [list(COLORS).index(p.color) for p in primary]
    elif factor == "position":
        ids = [GRID_FLAT.index(p.grid) for p in primary]
    else:
        ids = [COMPASS_8.index(p.facing) for p in primary]
    return torch.tensor(ids, dtype=torch.long)


def flatten_to_pca(train: Tensor, test: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """The floor baseline: flattens uint8 canvases and projects both splits
    onto the top principal components fit on the train split alone."""
    x_tr = train.flatten(1).float() / 255.0            # (N, 12288)
    x_te = test.flatten(1).float() / 255.0
    mean = x_tr.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x_tr - mean, q=dim, niter=4)  # v: (12288, dim)
    return (x_tr - mean) @ v, (x_te - mean) @ v


def build_probe(kind: str, dim: int, n_classes: int, hidden: int) -> nn.Module:
    """A glance is one linear layer; a scrutiny is a 2-hidden-layer GELU MLP."""
    if kind == "glance":
        return nn.Linear(dim, n_classes)
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                         nn.Linear(hidden, hidden), nn.GELU(),
                         nn.Linear(hidden, n_classes))


def question(features: tuple[Tensor, Tensor], labels: tuple[Tensor, Tensor],
             forge: Callable[[], nn.Module], seed: int,
             cfg: AtelierConfig) -> float:
    """Trains one probe (built by forge) on frozen features and returns
    test accuracy.

    Features may be (B, F) or (B, N, D); they are standardized
    elementwise by train statistics so every probe faces the same
    conditioning. AdamW (weight decay 0.01), cosine schedule, fixed
    epochs. The seed fixes both the probe's initialization and the
    batch order.
    """
    device = cfg.device
    x_tr, x_te = features
    y_tr, y_te = labels
    mean, std = x_tr.mean(dim=0), x_tr.std(dim=0) + 1e-6
    x_tr = ((x_tr - mean) / std).to(device)
    x_te = ((x_te - mean) / std).to(device)
    y_tr_d, y_te_d = y_tr.to(device), y_te.to(device)

    torch.manual_seed(seed)
    probe = forge().to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr)
    total = cfg.probe_epochs * (len(x_tr) // cfg.probe_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    picker = torch.Generator().manual_seed(seed)

    probe.train()
    for _ in range(cfg.probe_epochs):
        order = torch.randperm(len(x_tr), generator=picker).to(device)
        for i in range(0, len(order) - cfg.probe_batch + 1, cfg.probe_batch):
            pick = order[i:i + cfg.probe_batch]
            loss = nn.functional.cross_entropy(probe(x_tr[pick]), y_tr_d[pick])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    probe.eval()
    with torch.no_grad():
        hits = (probe(x_te).argmax(dim=1) == y_te_d).float().mean().item()
    return hits


def cross_examine(features: tuple[Tensor, Tensor], labels: tuple[Tensor, Tensor],
                  forge: Callable[[], nn.Module], cfg: AtelierConfig) -> dict:
    """Repeats one question across probe seeds, returns mean and std."""
    scores = [question(features, labels, forge, cfg.seed + s, cfg)
              for s in range(cfg.probe_seeds)]
    t = torch.tensor(scores)
    return {"mean": round(t.mean().item(), 4),
            "std": round(t.std(unbiased=False).item(), 4)}
