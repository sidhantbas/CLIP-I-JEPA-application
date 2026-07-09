"""Act II: the Sculptor. To know the world is to complete it.

A miniature I-JEPA: the canvas becomes 64 patch tokens, a veil hides a
contiguous block of them, the eye encodes only the visible remainder,
and the imagination predicts what the memory, an EMA twin that always
sees the whole canvas, encodes at the hidden positions. No pixels are
reconstructed and no words are involved. The regret is measured purely
in latent space.
"""

from __future__ import annotations

import copy
import json

import torch
from torch import Tensor, nn

from .config import AtelierConfig
from .namer import cosine_lr
from .world import harvest


class Eye(nn.Module):
    """Patch encoder: 8x8 patches embedded to D, given learned 2D positions
    (row plus column embeddings), passed through a transformer encoder with
    a final LayerNorm. Can encode the full canvas or only visible patches."""

    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4,
                 patch: int = 8, canvas: int = 64):
        super().__init__()
        self.grid = canvas // patch                       # 8 patches per side
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.row_pos = nn.Parameter(0.02 * torch.randn(self.grid, dim))
        self.col_pos = nn.Parameter(0.02 * torch.randn(self.grid, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=1024, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=depth,
                                           enable_nested_tensor=False)
        self.settle = nn.LayerNorm(dim)

    def positions(self) -> Tensor:
        """The 2D positional table, flattened row-major to (N, D)."""
        table = self.row_pos[:, None, :] + self.col_pos[None, :, :]
        return table.reshape(self.grid * self.grid, -1)   # (N, D)

    def forward(self, images: Tensor, keep: Tensor | None = None) -> Tensor:
        """Encodes images (B, 3, 64, 64) to token latents. If keep is given,
        only those patch indices are encoded; hidden patches are simply
        absent, never attended to. Returns (B, N_kept, D)."""
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = tokens + self.positions()
        if keep is not None:
            tokens = tokens[:, keep]                      # (B, N_vis, D)
        return self.settle(self.trunk(tokens))


class Imagination(nn.Module):
    """Predictor: context latents plus mask tokens carrying the hidden
    positions run through a small transformer; outputs at the mask token
    positions are projected back to D as guesses of the memory's latents."""

    def __init__(self, dim: int = 128, depth: int = 2, heads: int = 4,
                 n_positions: int = 64):
        super().__init__()
        self.mask_token = nn.Parameter(0.02 * torch.randn(dim))
        self.pos = nn.Parameter(0.02 * torch.randn(n_positions, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=512, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, num_layers=depth,
                                           enable_nested_tensor=False)
        self.project = nn.Linear(dim, dim)

    def forward(self, context: Tensor, hidden: Tensor) -> Tensor:
        """context (B, N_vis, D), hidden patch indices (N_hid,) ->
        guesses (B, N_hid, D). Mask tokens carry their own learned
        positional embeddings for the hidden patches."""
        b, n_hid = context.shape[0], hidden.shape[0]
        queries = self.mask_token + self.pos[hidden]      # (N_hid, D)
        queries = queries.expand(b, n_hid, -1)
        joined = torch.cat([context, queries], dim=1)     # (B, N_vis + N_hid, D)
        return self.project(self.trunk(joined)[:, -n_hid:])


def veil(grid: int, ratio: tuple[float, float],
         rng: torch.Generator) -> tuple[Tensor, Tensor]:
    """Samples one contiguous rectangular block of patches covering between
    ratio[0] and ratio[1] of the grid. Returns (visible_idx, hidden_idx)."""
    total = grid * grid
    while True:
        h = int(torch.randint(3, grid + 1, (1,), generator=rng))
        w = int(torch.randint(3, grid + 1, (1,), generator=rng))
        if ratio[0] * total <= h * w <= ratio[1] * total:
            break
    r = int(torch.randint(0, grid - h + 1, (1,), generator=rng))
    c = int(torch.randint(0, grid - w + 1, (1,), generator=rng))
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    mask[r:r + h, c:c + w] = True
    flat = mask.flatten()
    return (~flat).nonzero().squeeze(1), flat.nonzero().squeeze(1)


class Sculptor(nn.Module):
    """The eye, its EMA memory, and the imagination that bridges them."""

    def __init__(self, cfg: AtelierConfig):
        super().__init__()
        self.eye = Eye(cfg.latent_dim, cfg.sculptor_depth, cfg.sculptor_heads,
                       cfg.patch_size, cfg.canvas_size)
        self.memory = copy.deepcopy(self.eye)
        for p in self.memory.parameters():
            p.requires_grad_(False)
        grid = cfg.canvas_size // cfg.patch_size
        self.imagination = Imagination(cfg.latent_dim,
                                       cfg.sculptor_predictor_depth,
                                       cfg.sculptor_heads,
                                       n_positions=grid * grid)
        self.momentum = cfg.sculptor_ema_momentum

    def dream_step(self, images: Tensor, visible: Tensor,
                   hidden: Tensor) -> tuple[Tensor, float, float]:
        """Masks a region of patches, encodes visible context, predicts
        target-encoder latents of hidden patches, returns smooth L1 loss.

        Targets are standardized per token with LayerNorm so the loss
        measures structure, not scale. Besides the raw loss, returns the
        relative regret: raw loss divided by the loss of the best
        context-free guess (the per-position batch mean target). 1.0
        means the imagination knows no more than ignorance; the number
        is comparable across training even as targets grow richer. Also
        returns the collapse metric: mean per-dimension std of the raw
        memory latents across the batch."""
        context = self.eye(images, keep=visible)              # (B, N_vis, D)
        with torch.no_grad():
            carved = self.memory(images)                      # (B, N, D)
            spread = carved.std(dim=0).mean().item()
            target = nn.functional.layer_norm(                # (B, N_hid, D)
                carved[:, hidden], carved.shape[-1:])
            null = nn.functional.smooth_l1_loss(
                target.mean(dim=0, keepdim=True).expand_as(target).contiguous(),
                target)
        guess = self.imagination(context, hidden)
        loss = nn.functional.smooth_l1_loss(guess, target)
        return loss, loss.item() / max(null.item(), 1e-8), spread

    @torch.no_grad()
    def remember(self) -> None:
        """EMA update: the memory drifts toward the eye, never the reverse."""
        for slow, fast in zip(self.memory.parameters(), self.eye.parameters()):
            slow.mul_(self.momentum).add_(fast, alpha=1.0 - self.momentum)


def carve_features(eye: Eye, canvases: Tensor, device: str,
                   batch: int = 512) -> Tensor:
    """Frozen feature extraction for probing: the eye sees the full unmasked
    canvas, tokens are mean-pooled. uint8 (N, 3, 64, 64) -> (N, D) on cpu."""
    eye.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(canvases), batch):
            chunk = canvases[i:i + batch].to(device).float() / 255.0
            outs.append(eye(chunk).mean(dim=1).cpu())
    return torch.cat(outs)


def perform(cfg: AtelierConfig) -> dict:
    """Trains the Sculptor by dreaming, watches for collapse, saves weights.

    Each step veils one block (shared across the batch for simple
    gathering), the eye sees the rest, the imagination fills the gap in
    latent space. Curve and collapse metric go to
    results/sculptor_curve.json, weights to results/sculptor.pt.
    """
    device = cfg.device
    canvases, _ = harvest(cfg, cfg.seed, cfg.sculptor_scenes)
    sculptor = Sculptor(cfg).to(device)
    trainable = [p for p in sculptor.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in sculptor.eye.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cfg.sculptor_lr,
                                  weight_decay=cfg.sculptor_weight_decay)
    picker = torch.Generator().manual_seed(cfg.seed)

    curve: dict[str, list] = {"step": [], "regret": [], "spread": []}
    loss_at_100, spread_floor = None, float("inf")
    window: list[float] = []
    for step in range(cfg.sculptor_steps):
        lr = cosine_lr(step, cfg.sculptor_steps, cfg.sculptor_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        pick = torch.randint(len(canvases), (cfg.sculptor_batch,), generator=picker)
        images = canvases[pick].to(device).float() / 255.0
        visible, hidden = veil(sculptor.eye.grid, cfg.sculptor_mask_ratio, picker)
        loss, relative, spread = sculptor.dream_step(images, visible.to(device),
                                                     hidden.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sculptor.remember()
        window.append(relative)
        if step == 100:
            loss_at_100 = sum(window[-50:]) / len(window[-50:])
        if step % cfg.sculptor_log_every == 0 or step == cfg.sculptor_steps - 1:
            spread_floor = min(spread_floor, spread)
            curve["step"].append(step)
            curve["regret"].append(round(sum(window) / len(window), 5))
            curve["spread"].append(round(spread, 5))
            window = []

    with open(cfg.results_dir / "sculptor_curve.json", "w") as f:
        json.dump(curve, f, indent=1)
    torch.save({"eye": sculptor.eye.state_dict(), "dim": cfg.latent_dim},
               cfg.results_dir / "sculptor.pt")

    final = curve["regret"][-1]      # relative regret: 1.0 equals ignorance
    shrank = loss_at_100 / final if loss_at_100 else float("nan")
    collapsed = spread_floor < cfg.sculptor_collapse_floor
    return {
        "eye parameters": n_params,
        "regret at step 100": loss_at_100,
        "final regret": final,
        "regret shrank by": f"{shrank:.1f}x (need >= 3x)",
        "collapse metric floor": round(spread_floor, 5),
        "verdict": ("WARNING: LATENTS COLLAPSED, the memory speaks in one voice"
                    if collapsed else "no collapse, the memory kept its variety"),
    }


def resurrect(cfg: AtelierConfig) -> Eye:
    """Loads the trained eye from results/sculptor.pt, frozen, on cfg.device."""
    relic = torch.load(cfg.results_dir / "sculptor.pt", map_location="cpu",
                       weights_only=False)
    eye = Eye(relic["dim"], cfg.sculptor_depth, cfg.sculptor_heads,
              cfg.patch_size, cfg.canvas_size)
    eye.load_state_dict(relic["eye"])
    for p in eye.parameters():
        p.requires_grad_(False)
    return eye.to(cfg.device).eval()
