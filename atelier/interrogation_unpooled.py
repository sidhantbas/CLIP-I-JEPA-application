"""Act III, second sitting: the unpooled interrogation.

Phase 4 pooled 64 tokens into one vector before asking anything, and
found the Sculptor nearly silent about position. Two suspects remain:
the objective destroyed the geometry inside the encoder (H-encoder), or
it lives per token and mean-pooling averaged it away (H-pooling). This
sitting walks the room token by token: same scenes, same seeds, same
training loop as Phase 4 (shared through probing_common), with readouts
that preserve spatial identity. Only position and orientation are
questioned; form and color were linearly saturated when pooled."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import AtelierConfig
from .namer import Gaze, resurrect as resurrect_namer
from .probing_common import CHANCE, build_probe, cross_examine, testify
from .sculptor import Eye, resurrect as resurrect_sculptor
from .world import harvest

FACTORS_UNPOOLED: dict[str, int] = {"position": 9, "orientation": 8}
CONDITIONS = ("glance_pooled", "scrutiny_pooled",
              "glance_tokens", "attend", "scrutiny_tokens")


class Attend(nn.Module):
    """Attentive pooling probe: one learnable query, single-head scaled
    dot-product attention over the tokens, linear head on the summary.
    About 35k parameters at D=128, a readout a real planner could afford."""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.query = nn.Parameter(0.02 * torch.randn(dim))
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, tokens: Tensor) -> Tensor:
        scores = self.key(tokens) @ self.query / math.sqrt(tokens.shape[-1])
        weights = scores.softmax(dim=1).unsqueeze(-1)         # (B, N, 1)
        summary = (weights * self.value(tokens)).sum(dim=1)   # (B, D)
        return self.head(summary)                             # (B, C)


def carve_token_grid(eye: Eye, canvases: Tensor, device: str,
                     batch: int = 512) -> Tensor:
    """The Sculptor's room, unpooled: eye tokens over the full canvas,
    uint8 (N, 3, 64, 64) -> (N, 64, D) on cpu."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(canvases), batch):
            chunk = canvases[i:i + batch].to(device).float() / 255.0
            outs.append(eye(chunk).cpu())
    return torch.cat(outs)


def gaze_token_grid(gaze: Gaze, canvases: Tensor, device: str,
                    batch: int = 512) -> Tensor:
    """The Namer's room, unpooled: the 4x4 conv grid before mean-pool,
    projected per cell by the same head, (N, 16, D) on cpu. Because the
    projection is linear, these tokens mean-pool exactly to the Phase 4
    pooled features, so the two sittings differ only in pooling."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(canvases), batch):
            chunk = canvases[i:i + batch].to(device).float() / 255.0
            grid = gaze.trunk(chunk).flatten(2).transpose(1, 2)  # (B, 16, 256)
            outs.append(gaze.project(grid).cpu())                # (B, 16, D)
    return torch.cat(outs)


def patch_token_grid(train: Tensor, test: Tensor, patch: int,
                     dim: int) -> tuple[Tensor, Tensor]:
    """The unpooled floor: each 8x8 patch flattened (192 values) and
    projected onto principal components fit on train patches alone.
    uint8 (N, 3, 64, 64) -> (N, 64, dim)."""
    def to_patches(x: Tensor) -> Tensor:
        n, c, s, _ = x.shape
        g = s // patch
        x = x.float().div(255.0).reshape(n, c, g, patch, g, patch)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, c * patch * patch)
    p_tr, p_te = to_patches(train), to_patches(test)      # (N, 64, 192)
    flat = p_tr.reshape(-1, p_tr.shape[-1])
    mean = flat.mean(dim=0)
    _, _, v = torch.pca_lowrank(flat - mean, q=dim, niter=4)  # (192, dim)
    return (p_tr - mean) @ v, (p_te - mean) @ v


def judge(transcript: dict, factor: str, labels: tuple[str, str, str]) -> str:
    """The decision rule: how much of the distance from the pooled score to
    the unpooled pixels floor did the unpooled readouts recover? Over
    half earns labels[0], under a fifth labels[1], else labels[2]."""
    pooled = transcript["sculptor"][factor]["scrutiny_pooled"]["mean"]
    floor = transcript["pixels"][factor]["glance_tokens"]["mean"]
    best = max(transcript["sculptor"][factor][c]["mean"]
               for c in ("glance_tokens", "attend", "scrutiny_tokens"))
    fraction = (best - pooled) / max(floor - pooled, 0.02)
    if fraction > 0.5:
        return labels[0]
    return labels[1] if fraction < 0.2 else labels[2]


def perform(cfg: AtelierConfig) -> dict:
    """Runs the unpooled grid, writes results/interrogation_unpooled.json,
    prints the three-line verdict. Pooled numbers are copied from the
    archived Phase 4 JSON, never retrained."""
    pooled_path = cfg.results_dir / "interrogation.json"
    if not pooled_path.exists():
        raise FileNotFoundError("run --stage interrogation first")
    archived = json.loads(pooled_path.read_text())

    train_canvases, train_scenes = harvest(cfg, cfg.seed + 1, cfg.probe_train_scenes)
    test_canvases, test_scenes = harvest(cfg, cfg.seed + 2, cfg.probe_test_scenes)
    gaze, _, _ = resurrect_namer(cfg)
    eye = resurrect_sculptor(cfg)
    rooms: dict[str, tuple[Tensor, Tensor]] = {
        "namer": (gaze_token_grid(gaze, train_canvases, cfg.device),
                  gaze_token_grid(gaze, test_canvases, cfg.device)),
        "sculptor": (carve_token_grid(eye, train_canvases, cfg.device),
                     carve_token_grid(eye, test_canvases, cfg.device)),
        "pixels": patch_token_grid(train_canvases, test_canvases,
                                   cfg.patch_size, cfg.pixel_pca_dim),
    }

    transcript: dict[str, dict] = {"chance": {f: CHANCE[f] for f in FACTORS_UNPOOLED}}
    for name, (tokens_tr, tokens_te) in rooms.items():
        transcript[name] = {}
        flat = (tokens_tr.flatten(1), tokens_te.flatten(1))   # (N, tokens * D)
        wide, dim = flat[0].shape[1], tokens_tr.shape[-1]
        for factor, n_classes in FACTORS_UNPOOLED.items():
            labels = (testify(train_scenes, factor), testify(test_scenes, factor))
            entry = {"glance_pooled": archived[name][factor]["glance"],
                     "scrutiny_pooled": archived[name][factor]["scrutiny"],
                     "glance_tokens": cross_examine(
                         flat, labels, lambda: build_probe(
                             "glance", wide, n_classes, cfg.scrutiny_hidden), cfg)}
            if name != "pixels":                  # pixels serve as floor only
                entry["attend"] = cross_examine(
                    (tokens_tr, tokens_te), labels,
                    lambda: Attend(dim, n_classes), cfg)
                entry["scrutiny_tokens"] = cross_examine(
                    flat, labels, lambda: build_probe(
                        "scrutiny", wide, n_classes, cfg.scrutiny_hidden), cfg)
            transcript[name][factor] = entry
            print(f"  {name:9s} {factor:12s} " + " ".join(
                f"{c} {entry[c]['mean']:.3f}" for c in CONDITIONS if c in entry))

    with open(cfg.results_dir / "interrogation_unpooled.json", "w") as f:
        json.dump(transcript, f, indent=1)

    s = transcript["sculptor"]
    p_verdict = judge(transcript, "position",
                      ("pooling confound", "encoder destruction", "mixed"))
    o_verdict = judge(transcript, "orientation",
                      ("recovered", "collapse confirmed", "collapse partial"))
    implication = (
        "average-pooled probing understates what a JEPA encoder knows about "
        "position, so full-scale probes should pool attentively; " if
        p_verdict == "pooling confound" else
        "the objective itself, not the probing protocol, limits positional "
        "knowledge at this scale; ") + (
        "the orientation flattening survives every readout, so it is the "
        "objective's own signature." if o_verdict == "collapse confirmed" else
        "orientation is at least partly a readout problem, softening the "
        "collapse claim.")
    def recap(f: str, label: str) -> str:
        return (f"pooled scrutiny {s[f]['scrutiny_pooled']['mean']:.2f} -> "
                f"token glance {s[f]['glance_tokens']['mean']:.2f}, "
                f"attend {s[f]['attend']['mean']:.2f}. Verdict: {label}.")

    return {"POSITION": recap("position", p_verdict),
            "ORIENTATION": recap("orientation", o_verdict),
            "IMPLICATION FOR FULL SCALE": implication}


def draw_verdict(cfg: AtelierConfig, path: Path) -> None:
    """fig6: for position and orientation, all five readout conditions per
    encoder, with the pooled and unpooled pixels floors dashed. Reads
    only the JSON files, as every figure must."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = json.loads((cfg.results_dir / "interrogation_unpooled.json").read_text())
    shades = ("#f1948a", "#c0392b", "#85c1e9", "#2471a3", "#1a5276")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, factor in zip(axes, FACTORS_UNPOOLED):
        for g, encoder in enumerate(("namer", "sculptor")):
            for i, c in enumerate(CONDITIONS):
                cell = u[encoder][factor][c]
                ax.bar(g * 6 + i, cell["mean"], 0.9, yerr=cell["std"],
                       color=shades[i], label=c if g == 0 else None)
        ax.axhline(u["pixels"][factor]["scrutiny_pooled"]["mean"], ls="--",
                   c="#566573", lw=1.2, label="pixels pooled scrutiny")
        ax.axhline(u["pixels"][factor]["glance_tokens"]["mean"], ls="--",
                   c="black", lw=1.2, label="pixels token glance")
        ax.axhline(u["chance"][factor], ls=":", c="black", lw=1, label="chance")
        ax.set_xticks([2, 8], ["namer", "sculptor"])
        ax.set_title(factor)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("probe accuracy")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("where the geometry died: pooled vs unpooled readouts")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report_section(u: dict) -> list[str]:
    """Markdown lines for report.md: which hypothesis won, per factor."""
    lines = ["", "## Phase 8: where the geometry died", "",
             "| encoder | factor | " + " | ".join(CONDITIONS) + " |",
             "|---|---|" + "---|" * len(CONDITIONS)]
    for encoder in ("namer", "sculptor", "pixels"):
        for factor in FACTORS_UNPOOLED:
            cells = [f"{u[encoder][factor][c]['mean']:.3f}"
                     if c in u[encoder][factor] else "" for c in CONDITIONS]
            lines.append(f"| {encoder} | {factor} | " + " | ".join(cells) + " |")
    p = judge(u, "position", ("pooling confound", "encoder destruction", "mixed"))
    o = judge(u, "orientation",
              ("recovered", "collapse confirmed", "collapse partial"))
    s, floor = u["sculptor"], u["pixels"]
    lines += ["", (
        f"Position: the Sculptor's pooled scrutiny read "
        f"{s['position']['scrutiny_pooled']['mean']:.3f}, its unpooled token "
        f"glance reads {s['position']['glance_tokens']['mean']:.3f} and its "
        f"attentive probe {s['position']['attend']['mean']:.3f}, against an "
        f"unpooled pixels floor of "
        f"{floor['position']['glance_tokens']['mean']:.3f}. Verdict: {p}."), "", (
        f"Orientation: pooled scrutiny "
        f"{s['orientation']['scrutiny_pooled']['mean']:.3f}, token glance "
        f"{s['orientation']['glance_tokens']['mean']:.3f}, attend "
        f"{s['orientation']['attend']['mean']:.3f}, pixels floor "
        f"{floor['orientation']['glance_tokens']['mean']:.3f}. Verdict: {o}."),
        "", (
        f"The unpooled pixels floor for position "
        f"({floor['position']['glance_tokens']['mean']:.3f}) sits below the "
        f"pooled global-PCA floor: a patch knows its content but not its "
        f"index, so the per-patch probe must assemble position by comparing "
        f"64 local views, while global PCA hands position to the probe in "
        f"its leading components. One oddity reported as found: the token "
        f"MLP underperforms the token glance on the Sculptor's position, a "
        f"wide-input MLP fitting worse than its own linear special case "
        f"under this training budget."),
        "", "Form and color were skipped: both were linearly saturated pooled."]
    return lines
