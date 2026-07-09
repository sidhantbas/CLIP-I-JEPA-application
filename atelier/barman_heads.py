"""Act V, the second pour: the barman looks again.

Phase 6 scored the Sculptor's planning on mean-pooled features and
concluded that no readout rescues absent signal. Phase 8 then proved
the position signal was present per token and dying in the pool. This
act reruns exactly the Sculptor cells with cost heads that reach the
tokens: environment, goals, episode stream and metric byte identical
to Phase 6, Namer cells copied, not rerun. Cost heads are bilinear in
(state, goal), never concatenated: an additive linear form cannot
express the state-goal interaction that distance-to-goal is."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from .barman import Order, random_rollouts, serve, take_order
from .config import AtelierConfig
from .handshake import resurrect as resurrect_handshake, second_grip
from .namer import resurrect as resurrect_namer
from .sculptor import Eye, resurrect as resurrect_sculptor

CELLS = ("glance_pooled", "scrutiny_pooled", "token_glance", "attend",
         "token_scrutiny")


def press(tokens: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Flattens (B, N, D) -> (B, N*D), standardized by rollout statistics."""
    return (tokens.flatten(1) - mean) / std


class TokenGlance(nn.Module):
    """The linear reach: distance is bilinear(state, goal) plus linear
    terms, so for any fixed goal it is linear in the tokens, the planning
    analogue of Phase 8's token glance."""

    def __init__(self, wide: int, dim: int, mean: Tensor, std: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.pair = nn.Bilinear(wide, dim, 1)
        self.state, self.goal = nn.Linear(wide, 1), nn.Linear(dim, 1)

    def forward(self, tokens: Tensor, goals: Tensor) -> Tensor:
        z = press(tokens, self.mean, self.std)                # (B, wide)
        return (self.pair(z, goals) + self.state(z)
                + self.goal(goals)).squeeze(-1)               # (B,)


class AttentiveGlance(nn.Module):
    """The affordable reach: one learnable query attends over standardized
    tokens; the summary meets the goal in a bilinear form."""

    def __init__(self, n_tokens: int, dim: int, mean: Tensor, std: Tensor):
        super().__init__()
        self.n_tokens, self.dim = n_tokens, dim
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.query = nn.Parameter(0.02 * torch.randn(dim))
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.pair = nn.Bilinear(dim, dim, 1)
        self.state, self.goal = nn.Linear(dim, 1), nn.Linear(dim, 1)

    def forward(self, tokens: Tensor, goals: Tensor) -> Tensor:
        z = press(tokens, self.mean, self.std).reshape(
            -1, self.n_tokens, self.dim)                      # (B, N, D)
        scores = self.key(z) @ self.query / self.dim ** 0.5   # (B, N)
        summary = (scores.softmax(dim=1).unsqueeze(-1)
                   * self.value(z)).sum(dim=1)                # (B, D)
        return (self.pair(summary, goals) + self.state(summary)
                + self.goal(goals)).squeeze(-1)


class TokenScrutiny(nn.Module):
    """The ceiling Phase 8 warned about: an MLP over concatenated
    standardized tokens and goal, budget limited by construction."""

    def __init__(self, wide: int, dim: int, hidden: int, mean: Tensor,
                 std: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.judge = nn.Sequential(nn.Linear(wide + dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))

    def forward(self, tokens: Tensor, goals: Tensor) -> Tensor:
        z = press(tokens, self.mean, self.std)
        return self.judge(torch.cat([z, goals], dim=-1)).squeeze(-1)


def train_head(head: nn.Module, tokens: Tensor, goals: Tensor, dists: Tensor,
               cfg: AtelierConfig) -> nn.Module:
    """Fits a cost head by MSE on (token state, goal, distance) triples,
    the Phase 6 scorer budget: AdamW 1e-3, batch 256, barman_scorer_epochs
    epochs. Token states are large: they stay on cpu, batches cross alone."""
    device = cfg.device
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    picker = torch.Generator().manual_seed(cfg.seed)
    for _ in range(cfg.barman_scorer_epochs):
        order = torch.randperm(len(tokens), generator=picker)
        for i in range(0, len(order) - 255, 256):
            pick = order[i:i + 256]
            loss = nn.functional.mse_loss(
                head(tokens[pick].to(device), goals[pick].to(device)),
                dists[pick].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return head.eval()


def tokens_of(eye: Eye, canvases: Tensor, device: str,
              batch: int = 256) -> Tensor:
    """Eye token states for float canvases (N, 3, 64, 64) -> (N, 64, D) cpu."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(canvases), batch):
            outs.append(eye(canvases[i:i + batch].to(device)).cpu())
    return torch.cat(outs)


def perform(cfg: AtelierConfig) -> dict:
    """Reruns the Sculptor planning cells with token-aware cost heads on
    the exact Phase 6 episode stream, then runs the handshake footnote.
    Writes results/barman_unpooled.json and prints the verdict."""
    device = cfg.device
    archived = json.loads((cfg.results_dir / "barman.json").read_text())
    _, tongue, tokenizer = resurrect_namer(cfg)
    eye = resurrect_sculptor(cfg)
    handshake = resurrect_handshake(cfg)

    @torch.no_grad()
    def order_goals(orders: list[Order]) -> Tensor:
        speech = tongue(tokenizer.encode_batch(
            [o.caption for o in orders]).to(device))
        return nn.functional.normalize(handshake(speech), dim=-1)  # (N, D)

    rng = np.random.default_rng(cfg.seed + 3)     # the Phase 6 stream, replayed
    canvases, owners, dists = random_rollouts(cfg, rng)
    episodes = [take_order(rng) for _ in range(cfg.barman_episodes)]

    states = tokens_of(eye, canvases, device)                 # (N, 64, D)
    flat = states.flatten(1)
    mean, std = flat.mean(dim=0), flat.std(dim=0) + 1e-6
    goals_roll = order_goals(owners).cpu()
    wide, dim = flat.shape[1], states.shape[-1]
    heads: dict[str, nn.Module] = {
        "token_glance": TokenGlance(wide, dim, mean, std),
        "attend": AttentiveGlance(states.shape[1], dim, mean, std),
        "token_scrutiny": TokenScrutiny(wide, dim, cfg.scrutiny_hidden,
                                        mean, std)}
    for head in heads.values():
        train_head(head, states, goals_roll, dists, cfg)

    tab = {"namer": archived["namer"],
           "sculptor": {"glance_pooled": archived["sculptor"]["glance"],
                        "scrutiny_pooled": archived["sculptor"]["scrutiny"]}}
    with torch.no_grad():
        goals_ep = order_goals(episodes)
        for name, head in heads.items():
            wins = sum(serve(o, goals_ep[i], lambda b: eye(b),
                             lambda f, g, h=head: h(f, g), cfg)
                       for i, o in enumerate(tqdm(episodes, desc=name,
                                                  disable=None)))
            tab["sculptor"][name] = round(wins / len(episodes), 4)

    s = tab["sculptor"]
    best = max(s["token_glance"], s["attend"], s["token_scrutiny"])
    pooled_best = max(s["glance_pooled"], s["scrutiny_pooled"])
    verdict = ("pooled blind" if best >= tab["namer"]["glance"] - 0.10 else
               "geometry blind" if best <= pooled_best + 0.05 else "mixed")
    tab.update(episodes=cfg.barman_episodes, verdict=verdict)
    with open(cfg.results_dir / "barman_unpooled.json", "w") as f:
        json.dump(tab, f, indent=1)

    grip = second_grip(cfg)                       # the handshake footnote
    return {
        "SCULPTOR PLANNING": (
            f"pooled glance {s['glance_pooled']:.1%}, pooled scrutiny "
            f"{s['scrutiny_pooled']:.1%} -> token glance {s['token_glance']:.1%},"
            f" attend {s['attend']:.1%} (token scrutiny {s['token_scrutiny']:.1%})"),
        "VERDICT": verdict,
        "HANDSHAKE": (f"pooled {grip['pooled@5']:.1%} @5 -> attentive "
                      f"{grip['attentive@5']:.1%} @5. Bottleneck: "
                      f"{grip['bottleneck']}.")}


def draw_second_pour(cfg: AtelierConfig, path: Path) -> None:
    """fig7: Sculptor planning success across all five cost heads, with
    the Namer's two Phase 6 cells alongside as reference. JSON only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = json.loads((cfg.results_dir / "barman_unpooled.json").read_text())
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    shades = ("#a9cce3", "#5499c7", "#2471a3", "#1a5276", "#0e3253",
              "#e6b0aa", "#c0392b")
    spots = [0, 1, 2, 3, 4, 5.5, 6.5]
    heights = [e["sculptor"][c] for c in CELLS] + [e["namer"]["glance"],
                                                   e["namer"]["scrutiny"]]
    for x, y, hue in zip(spots, heights, shades):
        ax.bar(x, y, 0.8, color=hue)
        ax.text(x, y + 0.015, f"{y:.1%}", ha="center", fontsize=9)
    ax.set_xticks(spots, [c.replace("_", "\n") for c in CELLS]
                  + ["namer\nglance", "namer\nscrutiny"])
    ax.set_ylabel(f"success within {e['episodes']} episodes, 12-step cap")
    ax.set_title("the second pour: what the barman reads when the pool is gone")
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report_section(e: dict, hs: dict) -> list[str]:
    """Markdown lines for report.md: the second pour's outcome, and the
    explicit retirement or reaffirmation of the Phase 6 sentence."""
    s = e["sculptor"]
    ruled = {
        "pooled blind": (
            "The Phase 6 sentence, that no readout rescues absent signal, is "
            "hereby retired for position: the signal was never absent, only "
            "pooled away. Planning tracks readout-reachable information."),
        "geometry blind": ("The Phase 6 sentence stands, and sharpens into a "
                           "gap: probe accessibility did not suffice to plan."),
        "mixed": ("The Phase 6 sentence is weakened but not retired: token "
                  "readouts help, yet planning still trails the probes."),
    }[e["verdict"]]
    return ["", "## Phase 9: the barman looks again", "", (
        f"Sculptor planning on the identical 500 episodes: pooled glance "
        f"{s['glance_pooled']:.1%}, pooled scrutiny {s['scrutiny_pooled']:.1%}, "
        f"token glance {s['token_glance']:.1%}, attend {s['attend']:.1%}, "
        f"token scrutiny {s['token_scrutiny']:.1%} (budget limited, per the "
        f"Phase 8 oddity). Namer reference: glance {e['namer']['glance']:.1%}, "
        f"scrutiny {e['namer']['scrutiny']:.1%}. Verdict: {e['verdict']}. "
        + ruled + " Cost heads are bilinear in (state, goal), not "
        "concatenated: an additive linear form cannot express state-goal "
        "interaction, and distance-to-goal is exactly that interaction."),
        "", (
        f"The handshake footnote: retraining the bridge into attentively "
        f"pooled Sculptor space (same MLP, InfoNCE and budget) moved "
        f"retrieval among {hs['candidates']} from {hs['pooled@5']:.1%} to "
        f"{hs['attentive@5']:.1%} at @5, and {hs['pooled@1']:.1%} to "
        f"{hs['attentive@1']:.1%} at @1. Bottleneck: {hs['bottleneck']}.")]
