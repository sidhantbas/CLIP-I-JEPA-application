"""Re-derivations for the vitrine: the ephemeral readouts, regrown.

The atelier checkpoints keep only the two encoders and the pooled
bridge; the probes, the barman cost heads, the attentive grip and the
imagination were trained, measured and discarded. This module regrows
each one against the frozen encoders by replaying the original
training recipes (same objective, same budget, seed 7), importing the
frozen classes and loops rather than copying them. Nothing here
touches an encoder weight.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from atelier.barman import Order, random_rollouts
from atelier.barman_heads import AttentiveGlance, TokenGlance, train_head
from atelier.config import AtelierConfig
from atelier.handshake import AttentiveGrip, Handshake, speak_all
from atelier.interrogation_unpooled import Attend, carve_token_grid
from atelier.namer import cosine_lr
from atelier.probing_common import build_probe, testify
from atelier.sculptor import Imagination, veil
from atelier.world import harvest


def regrow_imagination(cfg: AtelierConfig, eye: nn.Module) -> Imagination:
    """The predictor was never checkpointed, so it is retrained here
    against the frozen eye standing in for the memory (its own EMA
    limit): original objective, veil, budget and schedule. Encoder
    weights receive no gradient and do not move."""
    device = cfg.device
    canvases, _ = harvest(cfg, cfg.seed, cfg.sculptor_scenes)
    imagination = Imagination(cfg.latent_dim, cfg.sculptor_predictor_depth,
                              cfg.sculptor_heads, n_positions=64).to(device)
    optimizer = torch.optim.AdamW(imagination.parameters(), lr=cfg.sculptor_lr,
                                  weight_decay=cfg.sculptor_weight_decay)
    picker = torch.Generator().manual_seed(cfg.seed)
    for step in range(cfg.sculptor_steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, cfg.sculptor_steps, cfg.sculptor_lr)
        pick = torch.randint(len(canvases), (cfg.sculptor_batch,), generator=picker)
        images = canvases[pick].to(device).float() / 255.0
        visible, hidden = veil(8, cfg.sculptor_mask_ratio, picker)
        with torch.no_grad():
            context = eye(images, keep=visible.to(device))        # (B, Nv, D)
            carved = eye(images)                                  # (B, 64, D)
            target = nn.functional.layer_norm(
                carved[:, hidden.to(device)], carved.shape[-1:])
        guess = imagination(context, hidden.to(device))
        loss = nn.functional.smooth_l1_loss(guess, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  imagination regrown, final regret {loss.item():.4f}")
    return imagination.cpu().eval()


def train_probe(features: tuple[Tensor, Tensor], labels: tuple[Tensor, Tensor],
                forge, cfg: AtelierConfig) -> tuple[nn.Module, Tensor, Tensor, float]:
    """probing_common.question, replayed so the probe itself survives:
    identical standardization, optimizer, schedule and batch order.
    Returns the probe, the train mean and std, and test accuracy."""
    device = cfg.device
    x_tr, x_te = features
    y_tr, y_te = labels
    mean, std = x_tr.mean(dim=0), x_tr.std(dim=0) + 1e-6
    x_tr, x_te = ((x_tr - mean) / std).to(device), ((x_te - mean) / std).to(device)
    y_tr_d, y_te_d = y_tr.to(device), y_te.to(device)
    torch.manual_seed(cfg.seed)
    probe = forge().to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg.probe_lr)
    total = cfg.probe_epochs * (len(x_tr) // cfg.probe_batch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    picker = torch.Generator().manual_seed(cfg.seed)
    for _ in range(cfg.probe_epochs):
        order = torch.randperm(len(x_tr), generator=picker).to(device)
        for i in range(0, len(order) - cfg.probe_batch + 1, cfg.probe_batch):
            pick = order[i:i + cfg.probe_batch]
            loss = nn.functional.cross_entropy(probe(x_tr[pick]), y_tr_d[pick])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    with torch.no_grad():
        acc = (probe(x_te).argmax(dim=1) == y_te_d).float().mean().item()
    return probe.cpu().eval(), mean, std, acc


def regrow_probes(cfg: AtelierConfig, pooled: dict[str, tuple[Tensor, Tensor]],
                  tokens: tuple[Tensor, Tensor],
                  scenes: tuple[list, list]) -> dict:
    """The seed-7 member of each probe family the vitrine can verify:
    pooled glances for both encoders on all four factors, and the
    Sculptor's token glance and attentive probes for the two geometric
    factors. Scrutiny MLPs are omitted for size; their accuracies live
    in the shipped interrogation JSONs."""
    factors = {"form": 4, "color": 5, "position": 9, "orientation": 8}
    grown: dict = {}
    for witness, features in pooled.items():
        for factor, n_classes in factors.items():
            labels = (testify(scenes[0], factor), testify(scenes[1], factor))
            probe, mean, std, acc = train_probe(
                features, labels,
                lambda k=n_classes, d=features[0].shape[1]:
                    build_probe("glance", d, k, cfg.scrutiny_hidden), cfg)
            grown[f"{witness}.{factor}.glance_pooled"] = (probe, mean, std, acc)
    flat = (tokens[0].flatten(1), tokens[1].flatten(1))
    for factor, n_classes in (("position", 9), ("orientation", 8)):
        labels = (testify(scenes[0], factor), testify(scenes[1], factor))
        probe, mean, std, acc = train_probe(
            flat, labels, lambda k=n_classes: build_probe(
                "glance", flat[0].shape[1], k, cfg.scrutiny_hidden), cfg)
        grown[f"sculptor.{factor}.glance_tokens"] = (probe, mean, std, acc)
        probe, mean, std, acc = train_probe(
            tokens, labels, lambda k=n_classes: Attend(cfg.latent_dim, k), cfg)
        grown[f"sculptor.{factor}.attend"] = (probe, mean, std, acc)
    return grown


def regrow_heads(cfg: AtelierConfig, eye: nn.Module, goal_embed) -> dict:
    """The Phase 9 cost heads, replayed: the same seed-7 rollout stream,
    the same features, goals and training budget, via the frozen
    barman_heads recipe. Returns the trained heads plus the rollout
    stream statistics they standardize with."""
    rng = np.random.default_rng(cfg.seed + 3)
    canvases, owners, dists = random_rollouts(cfg, rng)
    states = []
    with torch.no_grad():
        for i in range(0, len(canvases), 256):
            states.append(eye(canvases[i:i + 256].to(cfg.device)).cpu())
    states = torch.cat(states)                                    # (N, 64, D)
    flat = states.flatten(1)
    mean, std = flat.mean(dim=0), flat.std(dim=0) + 1e-6
    goals = goal_embed(owners).cpu()
    token_glance = TokenGlance(flat.shape[1], states.shape[-1], mean, std)
    attend = AttentiveGlance(states.shape[1], states.shape[-1], mean, std)
    for head in (token_glance, attend):
        train_head(head, states, goals, dists, cfg)
    return {"token_glance": token_glance.cpu().eval(),
            "attend": attend.cpu().eval()}


def regrow_grip(cfg: AtelierConfig, tongue: nn.Module, tokenizer,
                eye: nn.Module) -> tuple[AttentiveGrip, Handshake]:
    """The attentive grip and its bridge, replayed from handshake.second_grip
    without rewriting any results file: same pairs, same symmetric
    InfoNCE, same steps, batch and learning rate as Phase 5."""
    device = cfg.device
    canvases, scenes = harvest(cfg, cfg.seed, cfg.namer_scenes)
    speech = speak_all(tongue, tokenizer, scenes, device)
    tokens = carve_token_grid(eye, canvases, device)
    bridge = Handshake(cfg.latent_dim, cfg.handshake_hidden).to(device)
    grip = AttentiveGrip(cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(bridge.parameters()) + list(grip.parameters()), lr=cfg.handshake_lr)
    picker = torch.Generator().manual_seed(cfg.seed)
    for step in range(cfg.handshake_steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, cfg.handshake_steps, cfg.handshake_lr)
        pick = torch.randint(len(speech), (cfg.handshake_batch,), generator=picker)
        loss = bridge.clasp(speech[pick].to(device), grip(tokens[pick].to(device)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return grip.cpu().eval(), bridge.cpu().eval()
