"""Act I: the developer. The only bath that changes anything.

A ClipCap in miniature. The negative (one 512-d CLIP vector) is
developed into a short strip of prefix pseudo-tokens, ten 768-d vectors
that GPT-2 accepts as if they were words it had always known. The
frozen tongue then reads the true sentence after the strip, and the
cross-entropy of its guesses flows back through the frozen giant into
the developer alone. Roughly 31M parameters train; 275M watch.
"""

from __future__ import annotations

import json
import math

import torch
from torch import Tensor, nn
from tqdm import tqdm

from .config import DarkroomConfig
from .contact_sheet import load_negatives, read_captions, read_split


class Developer(nn.Module):
    """The mapping network: one negative in, prefix_len pseudo-tokens out.

    A two-layer MLP with a GELU waist, reshaped to (B, prefix_len, 768).
    This is the entire trainable surface of the darkroom."""

    def __init__(self, clip_dim: int, gpt2_dim: int, prefix_len: int, hidden: int):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt2_dim = gpt2_dim
        self.bath = nn.Sequential(
            nn.Linear(clip_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, prefix_len * gpt2_dim),
        )

    def forward(self, negatives: Tensor) -> Tensor:
        strip = self.bath(negatives)                              # (B, P*D)
        return strip.view(-1, self.prefix_len, self.gpt2_dim)     # (B, P, D)


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    """Linear warmup then cosine decay to zero."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress))


def tokenize_split(cfg: DarkroomConfig, split: str, tokenizer,
                   index: dict[str, int]) -> tuple[Tensor, Tensor, Tensor]:
    """Turns a split into aligned tensors: which negative, which tokens,
    how many of them are real. Each sentence ends with one true eos the
    developer must learn to reach; the rest of the row is padding."""
    captions = read_captions(cfg)
    rows, tokens, lengths = [], [], []
    eos = tokenizer.eos_token_id
    for name in read_split(cfg, split):
        for sentence in captions[name]:
            ids = tokenizer(sentence)["input_ids"][: cfg.max_caption_tokens - 1]
            ids.append(eos)
            lengths.append(len(ids))
            ids = ids + [eos] * (cfg.max_caption_tokens - len(ids))
            tokens.append(ids)
            rows.append(index[name])
    return torch.tensor(rows), torch.tensor(tokens), torch.tensor(lengths)


def caption_loss(tongue, wte, developer: Developer, negatives: Tensor,
                 tokens: Tensor, lengths: Tensor) -> Tensor:
    """One pass through the bath: prefix + sentence in, cross-entropy out.

    Prefix positions and padding get label -100, so the loss is averaged
    over real words (and the single closing eos) only."""
    batch, width = tokens.shape
    prefix = developer(negatives)                                  # (B, P, D)
    words = wte(tokens)                                            # (B, L, D)
    embeds = torch.cat([prefix, words], dim=1)

    positions = torch.arange(width, device=tokens.device)
    real = positions[None, :] < lengths[:, None]                   # (B, L)
    ones = torch.ones(batch, developer.prefix_len,
                      device=tokens.device, dtype=real.dtype)
    attention = torch.cat([ones, real], dim=1)

    silence = torch.full((batch, developer.prefix_len), -100,
                         device=tokens.device, dtype=tokens.dtype)
    labels = torch.cat([silence, tokens.masked_fill(~real, -100)], dim=1)

    out = tongue(inputs_embeds=embeds, attention_mask=attention, labels=labels)
    return out.loss


def perform(cfg: DarkroomConfig) -> dict:
    """Trains the developer, logs both curves, saves the weights.

    Each epoch shuffles the 30k (photo, sentence) pairs, and the full
    dev split is read after every epoch. Weights go to
    results/developer.pt, curves to results/developer_curve.json."""
    from .giants import summon_tongue

    device = cfg.device
    names, negatives = load_negatives(cfg)
    negatives = negatives.to(device)
    index = {n: i for i, n in enumerate(names)}
    tongue, tokenizer = summon_tongue(cfg)
    wte = tongue.get_input_embeddings()

    train = tokenize_split(cfg, "train", tokenizer, index)
    dev = tokenize_split(cfg, "dev", tokenizer, index)

    developer = Developer(cfg.clip_dim, cfg.gpt2_dim,
                          cfg.prefix_len, cfg.mapper_hidden).to(device)
    n_params = sum(p.numel() for p in developer.parameters())
    optimizer = torch.optim.AdamW(developer.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    picker = torch.Generator().manual_seed(cfg.seed)

    n_pairs = len(train[0])
    steps_per_epoch = n_pairs // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    curve: dict[str, list] = {"step": [], "loss": [], "epoch": [], "dev_loss": []}

    def read_dev() -> float:
        rows, tokens, lengths = dev
        losses = []
        developer.eval()
        with torch.no_grad():
            for i in range(0, len(rows), cfg.batch_size):
                sl = slice(i, i + cfg.batch_size)
                losses.append(caption_loss(
                    tongue, wte, developer, negatives[rows[sl]],
                    tokens[sl].to(device), lengths[sl].to(device)).item())
        developer.train()
        return sum(losses) / len(losses)

    step = 0
    for epoch in range(cfg.epochs):
        order = torch.randperm(n_pairs, generator=picker)
        bar = tqdm(range(steps_per_epoch), desc=f"developing (epoch {epoch + 1})")
        for b in bar:
            lr = cosine_lr(step, total_steps, cfg.lr, cfg.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            pick = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            rows, tokens, lengths = train
            loss = caption_loss(tongue, wte, developer, negatives[rows[pick]],
                                tokens[pick].to(device), lengths[pick].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % cfg.log_every == 0 or step == total_steps - 1:
                curve["step"].append(step)
                curve["loss"].append(round(loss.item(), 4))
                bar.set_postfix(loss=f"{loss.item():.3f}")
            step += 1
        curve["epoch"].append(epoch + 1)
        curve["dev_loss"].append(round(read_dev(), 4))

    cfg.results_dir.mkdir(exist_ok=True)
    with open(cfg.results_dir / "developer_curve.json", "w") as f:
        json.dump(curve, f, indent=1)
    torch.save(
        {"developer": developer.state_dict(),
         "clip_dim": cfg.clip_dim, "gpt2_dim": cfg.gpt2_dim,
         "prefix_len": cfg.prefix_len, "hidden": cfg.mapper_hidden},
        cfg.results_dir / "developer.pt",
    )
    best_dev = min(curve["dev_loss"])
    return {
        "parameters": n_params,
        "final train loss": curve["loss"][-1],
        "dev loss by epoch": curve["dev_loss"],
        "verdict": ("developed" if best_dev < 3.0
                    else "UNDERDEVELOPED, the bath needs debugging"),
    }


def resurrect(cfg: DarkroomConfig) -> Developer:
    """Loads the trained developer from results/developer.pt, frozen,
    on cfg.device."""
    relic = torch.load(cfg.results_dir / "developer.pt", map_location="cpu")
    developer = Developer(relic["clip_dim"], relic["gpt2_dim"],
                          relic["prefix_len"], relic["hidden"])
    developer.load_state_dict(relic["developer"])
    for p in developer.parameters():
        p.requires_grad_(False)
    return developer.to(cfg.device).eval()
