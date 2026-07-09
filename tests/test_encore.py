"""Gate B for the second pour: the token-glance cost head must plan on
what is trivially there. With the state cell planted one-hot into fake
tokens and the goal one-hot as the goal embedding, distance-to-goal is
a 9x9 table that a bilinear form can represent exactly, so the trained
head must rank candidate moves near-perfectly."""

import torch

from atelier.barman_heads import AttentiveGlance, TokenGlance, train_head
from atelier.config import AtelierConfig

CENTERS = torch.tensor([[c % 3, c // 3] for c in range(9)]).float()


def _cfg() -> AtelierConfig:
    cfg = AtelierConfig()
    cfg.device = "cpu"
    cfg.barman_scorer_epochs = 30
    return cfg


def _planted(n: int, g: torch.Generator):
    state = torch.randint(0, 9, (n,), generator=g)
    goal = torch.randint(0, 9, (n,), generator=g)
    tokens = 0.05 * torch.randn(n, 9, 8, generator=g)
    tokens[torch.arange(n), state, 0] += 2.0
    dists = (CENTERS[state] - CENTERS[goal]).norm(dim=1) / 3.0
    return tokens, torch.eye(9)[goal], dists


def test_token_glance_plans_on_planted_signal():
    g = torch.Generator().manual_seed(0)
    tokens, goals, dists = _planted(6000, g)
    flat = tokens.flatten(1)
    head = TokenGlance(72, 9, flat.mean(dim=0), flat.std(dim=0) + 1e-6)
    train_head(head, tokens, goals, dists, _cfg())
    hits = 0
    with torch.no_grad():
        for _ in range(200):
            goal = int(torch.randint(0, 9, (1,), generator=g))
            candidates = 0.05 * torch.randn(9, 9, 8, generator=g)
            candidates[torch.arange(9), torch.arange(9), 0] += 2.0
            costs = head(candidates, torch.eye(9)[goal].repeat(9, 1))
            hits += int(costs.argmin()) == goal
    assert hits / 200 >= 0.95        # Gate B: near-perfect candidate ranking


def test_attentive_glance_shapes():
    mean, std = torch.zeros(64 * 128), torch.ones(64 * 128)
    head = AttentiveGlance(64, 128, mean, std)
    costs = head(torch.randn(5, 64, 128), torch.randn(5, 128))
    assert costs.shape == (5,)
