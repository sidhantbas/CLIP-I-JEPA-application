"""Act IV: the handshake. A bridge from words to the wordless space.

The Namer's tongue speaks a caption into its own space; a small MLP
reaches across and offers that sentence to the Sculptor's latent space,
where no word has ever been. Trained with InfoNCE across the two
spaces, judged by retrieval: given a caption, find the one canvas it
describes among 512 candidates, using only Sculptor features.
"""

from __future__ import annotations

import json
import math

import torch
from torch import Tensor, nn

from .config import AtelierConfig
from .namer import Tongue, cosine_lr
from .namer import resurrect as resurrect_namer
from .sculptor import carve_features
from .sculptor import resurrect as resurrect_sculptor
from .world import Scene, Tokenizer, harvest


class Handshake(nn.Module):
    """2-layer MLP mapping tongue embeddings (D) into Sculptor space (D)."""

    def __init__(self, dim: int = 128, hidden: int = 256,
                 temperature_init: float = 0.07):
        super().__init__()
        self.reach = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / temperature_init)))

    def forward(self, speech: Tensor) -> Tensor:
        return self.reach(speech)                      # (B, D)

    def clasp(self, speech: Tensor, carvings: Tensor) -> Tensor:
        """Symmetric InfoNCE between reached captions and Sculptor features:
        cosine similarities scaled by a learnable temperature."""
        offered = nn.functional.normalize(self(speech), dim=-1)
        carved = nn.functional.normalize(carvings, dim=-1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits = scale * offered @ carved.T            # (B, B)
        truth = torch.arange(len(speech), device=speech.device)
        return 0.5 * (nn.functional.cross_entropy(logits, truth)
                      + nn.functional.cross_entropy(logits.T, truth))


def speak_all(tongue: Tongue, tokenizer: Tokenizer, scenes: list[Scene],
              device: str, batch: int = 512) -> Tensor:
    """Frozen tongue embeddings for every caption, (N, D) on cpu."""
    tokens = tokenizer.encode_batch([s.caption for s in scenes])
    outs = []
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            outs.append(tongue(tokens[i:i + batch].to(device)).cpu())
    return torch.cat(outs)


def judge_retrieval(handshake: Handshake, speech: Tensor, carvings: Tensor,
                    block: int, device: str, top: int = 20) -> list[float]:
    """Retrieval@k, k = 1..top: captions query canvases within blocks of
    `block` candidates; scores are averaged over all complete blocks."""
    hits = torch.zeros(top)
    blocks = len(speech) // block
    with torch.no_grad():
        for b in range(blocks):
            sl = slice(b * block, (b + 1) * block)
            offered = nn.functional.normalize(
                handshake(speech[sl].to(device)), dim=-1)
            carved = nn.functional.normalize(carvings[sl].to(device), dim=-1)
            ranks = (offered @ carved.T).argsort(dim=1, descending=True)
            truth = torch.arange(block, device=device)[:, None]
            place = (ranks == truth).float().argmax(dim=1).cpu()  # (block,)
            for k in range(top):
                hits[k] += (place <= k).float().mean().item()
    return [round(h / blocks, 4) for h in hits.tolist()]


def perform(cfg: AtelierConfig) -> dict:
    """Trains the bridge on frozen pairs, evaluates retrieval, saves both.

    Pairs come from the main corpus: tongue(caption) as input,
    eye features of the matching canvas as target space. Held-out
    scenes provide the retrieval judgment. Weights go to
    results/handshake.pt, numbers to results/handshake.json.
    """
    device = cfg.device
    _, tongue, tokenizer = resurrect_namer(cfg)
    eye = resurrect_sculptor(cfg)

    canvases, scenes = harvest(cfg, cfg.seed, cfg.namer_scenes)
    speech = speak_all(tongue, tokenizer, scenes, device)      # (N, D)
    carvings = carve_features(eye, canvases, device)           # (N, D)

    handshake = Handshake(cfg.latent_dim, cfg.handshake_hidden).to(device)
    optimizer = torch.optim.AdamW(handshake.parameters(), lr=cfg.handshake_lr)
    picker = torch.Generator().manual_seed(cfg.seed)
    for step in range(cfg.handshake_steps):
        lr = cosine_lr(step, cfg.handshake_steps, cfg.handshake_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        pick = torch.randint(len(speech), (cfg.handshake_batch,), generator=picker)
        loss = handshake.clasp(speech[pick].to(device), carvings[pick].to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    held_canvases, held_scenes = harvest(cfg, cfg.seed + 4,
                                         4 * cfg.handshake_candidates)
    held_speech = speak_all(tongue, tokenizer, held_scenes, device)
    held_carvings = carve_features(eye, held_canvases, device)
    curve = judge_retrieval(handshake, held_speech, held_carvings,
                            cfg.handshake_candidates, device)

    verdict = ("the bridge holds" if curve[4] > 0.5 else
               "THE BRIDGE IS NARROW: retrieval@5 under 50%, a finding in itself")
    with open(cfg.results_dir / "handshake.json", "w") as f:
        json.dump({"candidates": cfg.handshake_candidates,
                   "retrieval_at_k": curve, "retrieval@1": curve[0],
                   "retrieval@5": curve[4], "verdict": verdict}, f, indent=1)
    torch.save({"handshake": handshake.state_dict(), "dim": cfg.latent_dim,
                "hidden": cfg.handshake_hidden},
               cfg.results_dir / "handshake.pt")
    return {"retrieval@1": curve[0], "retrieval@5": curve[4],
            "candidates": cfg.handshake_candidates, "verdict": verdict}


def resurrect(cfg: AtelierConfig) -> Handshake:
    """Loads the trained bridge from results/handshake.pt, frozen."""
    relic = torch.load(cfg.results_dir / "handshake.pt", map_location="cpu",
                       weights_only=False)
    handshake = Handshake(relic["dim"], relic["hidden"])
    handshake.load_state_dict(relic["handshake"])
    for p in handshake.parameters():
        p.requires_grad_(False)
    return handshake.to(cfg.device).eval()


class AttentiveGrip(nn.Module):
    """A learnable hand for the bridge to shake: one query, single-head
    scaled dot-product attention over eye tokens, (B, N, D) -> (B, D).
    Trained jointly with the bridge on the same InfoNCE, so the pooling
    itself learns what the bridge needs to hold."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.query = nn.Parameter(0.02 * torch.randn(dim))
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, tokens: Tensor) -> Tensor:
        scores = self.key(tokens) @ self.query / tokens.shape[-1] ** 0.5
        return (scores.softmax(dim=1).unsqueeze(-1)
                * self.value(tokens)).sum(dim=1)              # (B, D)


def second_grip(cfg: AtelierConfig) -> dict:
    """The footnote experiment: the bridge retrained into attentively
    pooled Sculptor space. Same MLP, same symmetric InfoNCE, same steps,
    batch and learning rate as Phase 5; the only change is that the
    target features come from AttentiveGrip, trained jointly, instead of
    the mean pool. Writes results/handshake_unpooled.json."""
    from .interrogation_unpooled import carve_token_grid
    device = cfg.device
    pooled = json.loads((cfg.results_dir / "handshake.json").read_text())
    _, tongue, tokenizer = resurrect_namer(cfg)
    eye = resurrect_sculptor(cfg)
    canvases, scenes = harvest(cfg, cfg.seed, cfg.namer_scenes)
    speech = speak_all(tongue, tokenizer, scenes, device)      # (N, D)
    tokens = carve_token_grid(eye, canvases, device)           # (N, 64, D)

    bridge = Handshake(cfg.latent_dim, cfg.handshake_hidden).to(device)
    grip = AttentiveGrip(cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(list(bridge.parameters())
                                  + list(grip.parameters()),
                                  lr=cfg.handshake_lr)
    picker = torch.Generator().manual_seed(cfg.seed)
    for step in range(cfg.handshake_steps):
        lr = cosine_lr(step, cfg.handshake_steps, cfg.handshake_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        pick = torch.randint(len(speech), (cfg.handshake_batch,), generator=picker)
        loss = bridge.clasp(speech[pick].to(device),
                            grip(tokens[pick].to(device)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    held_canvases, held_scenes = harvest(cfg, cfg.seed + 4,
                                         4 * cfg.handshake_candidates)
    held_speech = speak_all(tongue, tokenizer, held_scenes, device)
    with torch.no_grad():
        held_tokens = carve_token_grid(eye, held_canvases, device)
        carvings = torch.cat([grip(held_tokens[i:i + 512].to(device)).cpu()
                              for i in range(0, len(held_tokens), 512)])
    curve = judge_retrieval(bridge.eval(), held_speech, carvings,
                            cfg.handshake_candidates, device)

    lift = curve[4] - pooled["retrieval@5"]
    bottleneck = ("pooling" if lift >= 0.10 else
                  "feature space" if lift <= 0.02 else "mixed")
    out = {"candidates": cfg.handshake_candidates,
           "pooled@1": pooled["retrieval@1"], "pooled@5": pooled["retrieval@5"],
           "attentive@1": curve[0], "attentive@5": curve[4],
           "attentive_retrieval_at_k": curve, "bottleneck": bottleneck}
    with open(cfg.results_dir / "handshake_unpooled.json", "w") as f:
        json.dump(out, f, indent=1)
    return out
