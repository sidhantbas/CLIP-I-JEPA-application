"""Act V: the barman. A planner that reads only what a glance affords.

One shape on the canvas, one order: a caption naming the cell where it
should rest. The barman nudges the shape by small displacements, always
greedily taking the candidate next state whose feature scores best
against the goal embedding. The experimental variable is the cost head:
a plain cosine distance (the glance) or a small trained scorer (the
scrutiny). The world itself re-renders every move, so nothing drifts:
only the readout is on trial. Success should track accessibility, not
presence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from .config import AtelierConfig
from .handshake import resurrect as resurrect_handshake
from .namer import resurrect as resurrect_namer
from .sculptor import resurrect as resurrect_sculptor
from .world import (BACKGROUND, COLORS, COMPASS_8, FORMS, GRID_NAMES,
                    ShapeFactors, shape_alpha)

GRID_FLAT: list[str] = [name for row in GRID_NAMES for name in row]
MOVES: list[tuple[int, int]] = [(0, 0)] + [
    (round(8 * np.cos(np.deg2rad(45 * k))), round(-8 * np.sin(np.deg2rad(45 * k))))
    for k in range(8)
]                                          # stay plus 8 compass steps of 8 px

_YS, _XS = np.mgrid[0:64, 0:64]
_PX, _PY = _XS + 0.5, _YS + 0.5


@dataclass
class Order:
    """One episode: a shape to move and the cell where it should rest."""

    shape: ShapeFactors
    goal_grid: str

    @property
    def caption(self) -> str:
        return (f"a {self.shape.size_word} {self.shape.color} {self.shape.form}"
                f" rests in the {self.goal_grid}")

    @property
    def goal_center(self) -> tuple[float, float]:
        i = GRID_FLAT.index(self.goal_grid)
        return (i % 3) * 64 / 3 + 64 / 6, (i // 3) * 64 / 3 + 64 / 6


def pour(shape: ShapeFactors) -> Tensor:
    """Renders the single-shape canvas at the shape's current position.
    Returns (3, 64, 64) float in [0, 1]."""
    canvas = np.empty((64, 64, 3))
    canvas[:] = BACKGROUND
    alpha = shape_alpha(_PX, _PY, shape)[..., None]           # (64, 64, 1)
    canvas = canvas * (1.0 - alpha) + np.array(COLORS[shape.color]) * alpha
    return torch.from_numpy(canvas.transpose(2, 0, 1)).float()


def cell_of(x: float, y: float) -> str:
    return GRID_FLAT[min(int(y / (64 / 3)), 2) * 3 + min(int(x / (64 / 3)), 2)]


def take_order(rng: np.random.Generator) -> Order:
    """Samples a shape at a random legal position and a goal cell that is
    not the cell it starts in."""
    size = float(rng.uniform(8.0, 15.0))
    lo, hi = size + 1.0, 63.0 - size
    x, y = float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))
    angle = float(rng.uniform(0.0, 360.0))
    shape = ShapeFactors(
        form=str(rng.choice(FORMS)), color=str(rng.choice(list(COLORS))),
        x=x, y=y, grid=cell_of(x, y), angle=angle,
        facing=COMPASS_8[int(((angle + 22.5) % 360.0) // 45.0)], size=size)
    goals = [g for g in GRID_FLAT if g != shape.grid]
    return Order(shape=shape, goal_grid=str(rng.choice(goals)))


def nudge(shape: ShapeFactors, move: tuple[int, int]) -> ShapeFactors:
    """The shape displaced by one move, clamped inside the canvas."""
    lo, hi = shape.size + 1.0, 63.0 - shape.size
    x = float(np.clip(shape.x + move[0], lo, hi))
    y = float(np.clip(shape.y + move[1], lo, hi))
    return replace(shape, x=x, y=y, grid=cell_of(x, y))


class Scrutiny(nn.Module):
    """The trained cost head: concat(state feature, goal embedding) through
    a small MLP predicting normalized distance to the goal cell center."""

    def __init__(self, dim: int = 128, hidden: int = 256):
        super().__init__()
        self.judge = nn.Sequential(nn.Linear(2 * dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))

    def forward(self, features: Tensor, goals: Tensor) -> Tensor:
        return self.judge(torch.cat([features, goals], dim=-1)).squeeze(-1)  # (B,)


def random_rollouts(cfg: AtelierConfig, rng: np.random.Generator
                    ) -> tuple[Tensor, list[Order], Tensor]:
    """Random-policy rollouts for scorer training: canvases (N, 3, 64, 64),
    the order each state belongs to, and distance-to-goal targets (N,)."""
    canvases, owners, dists = [], [], []
    n_rollouts = cfg.barman_rollouts // cfg.barman_max_steps
    for _ in range(n_rollouts):
        order = take_order(rng)
        shape = order.shape
        gx, gy = order.goal_center
        for _ in range(cfg.barman_max_steps):
            shape = nudge(shape, MOVES[int(rng.integers(len(MOVES)))])
            canvases.append(pour(shape))
            owners.append(order)
            dists.append(np.hypot(shape.x - gx, shape.y - gy) / 64.0)
    return torch.stack(canvases), owners, torch.tensor(dists).float()


def train_scrutiny(features: Tensor, goals: Tensor, dists: Tensor,
                   cfg: AtelierConfig) -> Scrutiny:
    """Fits the scorer by MSE on (feature, goal embedding, distance) triples."""
    device = cfg.device
    scorer = Scrutiny(cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
    picker = torch.Generator().manual_seed(cfg.seed)
    x, g, y = features.to(device), goals.to(device), dists.to(device)
    for _ in range(cfg.barman_scorer_epochs):
        order = torch.randperm(len(x), generator=picker).to(device)
        for i in range(0, len(order) - 255, 256):
            pick = order[i:i + 256]
            loss = nn.functional.mse_loss(scorer(x[pick], g[pick]), y[pick])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return scorer.eval()


def serve(order: Order, goal: Tensor, encode, cost, cfg: AtelierConfig) -> bool:
    """Plays one greedy episode: at each step, renders all candidate next
    states, encodes them, and takes the cheapest. True if the shape ends
    in the goal cell within barman_max_steps."""
    shape = order.shape
    for _ in range(cfg.barman_max_steps):
        futures = [nudge(shape, m) for m in MOVES]
        batch = torch.stack([pour(f) for f in futures]).to(cfg.device)
        costs = cost(encode(batch), goal.expand(len(MOVES), -1))    # (9,)
        shape = futures[int(costs.argmin())]
        if shape.grid == order.goal_grid:
            return True
    return False


def perform(cfg: AtelierConfig) -> dict:
    """Runs the 2x2 experiment: {Namer, Sculptor} x {glance, scrutiny}.

    Goal embeddings: captions through the tongue, and through the
    handshake as well for the Sculptor's space. The same episodes are
    served in all four cells. Success rates go to results/barman.json.
    """
    device = cfg.device
    gaze, tongue, tokenizer = resurrect_namer(cfg)
    eye = resurrect_sculptor(cfg)
    handshake = resurrect_handshake(cfg)

    def encode_namer(batch: Tensor) -> Tensor:
        return nn.functional.normalize(gaze(batch), dim=-1)

    def encode_sculptor(batch: Tensor) -> Tensor:
        return nn.functional.normalize(eye(batch).mean(dim=1), dim=-1)

    @torch.no_grad()
    def goal_embed(orders: list[Order], space: str) -> Tensor:
        tokens = tokenizer.encode_batch([o.caption for o in orders]).to(device)
        speech = tongue(tokens)
        if space == "sculptor":
            speech = handshake(speech)
        return nn.functional.normalize(speech, dim=-1)          # (N, D)

    encoders = {"namer": encode_namer, "sculptor": encode_sculptor}

    rng = np.random.default_rng(cfg.seed + 3)
    canvases, owners, dists = random_rollouts(cfg, rng)
    scorers: dict[str, Scrutiny] = {}
    for name, encode in encoders.items():
        with torch.no_grad():
            feats = torch.cat([encode(canvases[i:i + 512].to(device)).cpu()
                               for i in range(0, len(canvases), 512)])
            goals = goal_embed(owners, name).cpu()
        scorers[name] = train_scrutiny(feats, goals, dists, cfg)

    episodes = [take_order(rng) for _ in range(cfg.barman_episodes)]
    tab: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, encode in encoders.items():
            goals = goal_embed(episodes, name)
            costs = {
                "glance": lambda f, g: 1.0 - (f * g).sum(dim=-1),
                "scrutiny": lambda f, g, s=scorers[name]: s(f, g),
            }
            tab[name] = {}
            for kind, cost in costs.items():
                wins = sum(
                    serve(o, goals[i], encode, cost, cfg)
                    for i, o in enumerate(tqdm(episodes, desc=f"{name}/{kind}",
                                               disable=None)))
                tab[name][kind] = round(wins / len(episodes), 4)

    tab["episodes"] = cfg.barman_episodes
    with open(cfg.results_dir / "barman.json", "w") as f:
        json.dump(tab, f, indent=1)
    return {name: tab[name] for name in encoders}
