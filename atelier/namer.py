"""Act I: the Namer. To know the world is to name it.

A miniature CLIP: a convolutional gaze embeds canvases, a small
transformer tongue embeds captions, and the accord, a symmetric
InfoNCE loss with learnable temperature, pulls matching pairs
together across a batch. The Namer only ever learns what language
bothers to say.
"""

from __future__ import annotations

import json
import math

import torch
from torch import Tensor, nn

from .config import AtelierConfig
from .world import Tokenizer, World, harvest


class Gaze(nn.Module):
    """Image encoder: four stride-2 conv blocks (GroupNorm, GELU) taking
    (B, 3, 64, 64) down to a 4x4 grid, mean-pooled and projected to D."""

    def __init__(self, dim: int = 128):
        super().__init__()
        blocks: list[nn.Module] = []
        channels = (3, 64, 128, 192, 256)
        for c_in, c_out in zip(channels, channels[1:]):
            blocks += [
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                nn.GroupNorm(8, c_out), nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.GroupNorm(8, c_out), nn.GELU(),
            ]
        self.trunk = nn.Sequential(*blocks)
        self.project = nn.Linear(channels[-1], dim)

    def forward(self, images: Tensor) -> Tensor:
        grid = self.trunk(images)                # (B, 256, 4, 4)
        pooled = grid.mean(dim=(2, 3))           # (B, 256)
        return self.project(pooled)              # (B, D)


class Tongue(nn.Module):
    """Text encoder: word embeddings plus learned positions through a
    2-layer transformer; the last non-pad token's state is projected to D."""

    def __init__(self, vocab_size: int, max_len: int, dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=Tokenizer.PAD)
        self.positions = nn.Parameter(torch.zeros(max_len, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=512, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=2,
                                           enable_nested_tensor=False)
        self.project = nn.Linear(dim, dim)

    def forward(self, tokens: Tensor) -> Tensor:
        padding = tokens == Tokenizer.PAD                     # (B, L)
        states = self.trunk(self.embed(tokens) + self.positions,
                            src_key_padding_mask=padding)     # (B, L, D)
        last = (~padding).sum(dim=1) - 1                      # (B,)
        final = states[torch.arange(len(tokens)), last]       # (B, D)
        return self.project(final)


class Namer(nn.Module):
    """The pair of encoders and their accord."""

    def __init__(self, vocab_size: int, max_len: int, dim: int,
                 temperature_init: float):
        super().__init__()
        self.gaze = Gaze(dim)
        self.tongue = Tongue(vocab_size, max_len, dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature_init)))

    def accord(self, images: Tensor, tokens: Tensor) -> tuple[Tensor, float]:
        """Symmetric InfoNCE across the batch. Returns the loss and the
        in-batch image-to-caption retrieval@1 accuracy."""
        sight = nn.functional.normalize(self.gaze(images), dim=-1)    # (B, D)
        speech = nn.functional.normalize(self.tongue(tokens), dim=-1)  # (B, D)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits = scale * sight @ speech.T                             # (B, B)
        truth = torch.arange(len(images), device=images.device)
        loss = 0.5 * (nn.functional.cross_entropy(logits, truth)
                      + nn.functional.cross_entropy(logits.T, truth))
        hits = (logits.argmax(dim=1) == truth).float().mean().item()
        return loss, hits


def cosine_lr(step: int, total: int, base: float, warmup: int = 100) -> float:
    """Linear warmup then cosine decay to zero."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def embed_sights(gaze: Gaze, canvases: Tensor, device: str,
                 batch: int = 512) -> Tensor:
    """Runs the frozen gaze over uint8 canvases in batches, returns (N, D) on cpu."""
    gaze.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(canvases), batch):
            chunk = canvases[i:i + batch].to(device).float() / 255.0
            outs.append(gaze(chunk).cpu())
    return torch.cat(outs)


def perform(cfg: AtelierConfig) -> dict:
    """Trains the Namer with the accord, logs the curve, saves the weights.

    Batches are drawn with replacement from a fixed corpus. Every
    namer_log_every steps the loss and in-batch retrieval@1 are recorded
    to results/namer_curve.json. Weights go to results/namer.pt.
    """
    device = cfg.device
    canvases, scenes = harvest(cfg, cfg.seed, cfg.namer_scenes)
    captions = [s.caption for s in scenes]
    world = World(cfg.seed)
    tokenizer = Tokenizer(world.vocabulary(), cfg.max_caption_len)
    tokens = tokenizer.encode_batch(captions)                 # (N, L) on cpu

    namer = Namer(tokenizer.vocab_size, cfg.max_caption_len, cfg.latent_dim,
                  cfg.namer_temperature_init).to(device)
    n_params = sum(p.numel() for p in namer.parameters())
    optimizer = torch.optim.AdamW(namer.parameters(), lr=cfg.namer_lr,
                                  weight_decay=cfg.namer_weight_decay)
    picker = torch.Generator().manual_seed(cfg.seed)

    curve: dict[str, list] = {"step": [], "loss": [], "acc": []}
    final_acc = 0.0
    for step in range(cfg.namer_steps):
        lr = cosine_lr(step, cfg.namer_steps, cfg.namer_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        pick = torch.randint(len(canvases), (cfg.namer_batch,), generator=picker)
        images = canvases[pick].to(device).float() / 255.0    # (B, 3, 64, 64)
        loss, acc = namer.accord(images, tokens[pick].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % cfg.namer_log_every == 0 or step == cfg.namer_steps - 1:
            curve["step"].append(step)
            curve["loss"].append(round(loss.item(), 4))
            curve["acc"].append(round(acc, 4))
            final_acc = acc

    cfg.results_dir.mkdir(exist_ok=True)
    with open(cfg.results_dir / "namer_curve.json", "w") as f:
        json.dump(curve, f, indent=1)
    torch.save(
        {"gaze": namer.gaze.state_dict(), "tongue": namer.tongue.state_dict(),
         "logit_scale": namer.logit_scale.detach().cpu(),
         "vocab": world.vocabulary(), "dim": cfg.latent_dim},
        cfg.results_dir / "namer.pt",
    )
    return {
        "parameters": n_params,
        "final loss": curve["loss"][-1],
        "retrieval@1 in final batch": final_acc,
        "verdict": "accord reached" if final_acc > 0.6 else "ACCORD FAILED, debug me",
    }


def resurrect(cfg: AtelierConfig) -> tuple[Gaze, Tongue, Tokenizer]:
    """Loads the trained Namer from results/namer.pt, frozen, on cfg.device."""
    relic = torch.load(cfg.results_dir / "namer.pt", map_location="cpu",
                       weights_only=False)
    tokenizer = Tokenizer(relic["vocab"], cfg.max_caption_len)
    gaze, tongue = Gaze(relic["dim"]), Tongue(tokenizer.vocab_size,
                                              cfg.max_caption_len, relic["dim"])
    gaze.load_state_dict(relic["gaze"])
    tongue.load_state_dict(relic["tongue"])
    for p in list(gaze.parameters()) + list(tongue.parameters()):
        p.requires_grad_(False)
    return gaze.to(cfg.device).eval(), tongue.to(cfg.device).eval(), tokenizer
