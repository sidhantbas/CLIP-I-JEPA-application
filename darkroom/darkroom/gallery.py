"""The exhibition. Figures come from the JSON files, never from live
memory, so the whole show can be re-hung at any time. The one exception
is fig1, the contact sheet itself, which needs the photographs back for
their portraits; the sentences under them still come from prints.json.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .config import DarkroomConfig

SAFELIGHT = "#c0392b"   # the model's own colour, after atelier's namer red
DAYLIGHT = "#2471a3"    # human reference blue
NEUTRAL = "#7f8c8d"     # baselines and floors


def _load(cfg: DarkroomConfig, name: str) -> dict | list | None:
    path = cfg.results_dir / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def draw_contact_sheet(cfg: DarkroomConfig, sheet: list, path: Path) -> None:
    """fig1: sixteen test photographs, each with its print (red) and the
    first human sentence (blue) beneath."""
    fig, axes = plt.subplots(4, 4, figsize=(13, 15))
    for ax, row in zip(axes.flat, sheet[:16]):
        ax.imshow(Image.open(cfg.image_dir / row["photo"]).convert("RGB"))
        ours = textwrap.fill(row["print"], 40)
        theirs = textwrap.fill(row["references"][0], 40)
        ax.set_title(ours, fontsize=7, color=SAFELIGHT)
        ax.set_xlabel(theirs, fontsize=7, color=DAYLIGHT)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("the contact sheet: prints in red, one human sentence in blue",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_developer_curve(curve: dict, path: Path) -> None:
    """fig2: the training loss by step, the dev loss by epoch."""
    steps_per_epoch = (max(curve["step"]) + 1) / max(curve["epoch"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["step"], curve["loss"], color=SAFELIGHT, lw=1.2,
            alpha=0.85, label="train loss (per step)")
    dev_steps = [e * steps_per_epoch for e in curve["epoch"]]
    ax.plot(dev_steps, curve["dev_loss"], color=DAYLIGHT, lw=1.6,
            marker="o", ms=4, label="dev loss (per epoch)")
    ax.annotate(f'{curve["dev_loss"][-1]:.3f}',
                (dev_steps[-1], curve["dev_loss"][-1]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="#333333")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy over real words")
    ax.set_title("the developer's bath", fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_loupe(verdict: dict, path: Path) -> None:
    """fig3: BLEU-1..4 on the left; the CLIP accord, with its human
    ceiling and shuffled floor, on the right."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))

    orders = [f"bleu{n}" for n in (1, 2, 3, 4)]
    values = [verdict["bleu"][o] for o in orders]
    left.bar(range(4), values, 0.62, color=SAFELIGHT)
    for i, v in enumerate(values):
        left.annotate(f"{v:.3f}", (i, v), ha="center",
                      textcoords="offset points", xytext=(0, 3),
                      fontsize=8, color="#333333")
    left.set_xticks(range(4), [o.upper() for o in orders])
    left.set_ylim(0, 1.0)
    left.set_title("BLEU against five human sentences", fontsize=10)
    left.grid(axis="y", alpha=0.25, lw=0.5)

    accord = verdict["accord"]
    names = ("prints", "humans", "shuffled")
    hues = (SAFELIGHT, DAYLIGHT, NEUTRAL)
    for i, (name, hue) in enumerate(zip(names, hues)):
        right.bar(i, accord[name], 0.62, color=hue)
        right.annotate(f"{accord[name]:.3f}", (i, accord[name]), ha="center",
                       textcoords="offset points", xytext=(0, 3),
                       fontsize=8, color="#333333")
    right.set_xticks(range(3), names)
    right.set_ylim(0, max(accord.values()) * 1.3)
    right.set_title("CLIP accord: does the sentence still point\nat its photograph?",
                    fontsize=10)
    right.grid(axis="y", alpha=0.25, lw=0.5)

    fig.suptitle("under the loupe", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def perform(cfg: DarkroomConfig) -> dict:
    """Hangs whatever results exist; missing acts are simply skipped."""
    hung = []
    sheet = _load(cfg, "prints")
    if sheet:
        draw_contact_sheet(cfg, sheet, cfg.results_dir / "fig1_contact_sheet.png")
        hung.append("fig1_contact_sheet")
    curve = _load(cfg, "developer_curve")
    if curve:
        draw_developer_curve(curve, cfg.results_dir / "fig2_developer_curve.png")
        hung.append("fig2_developer_curve")
    verdict = _load(cfg, "loupe")
    if verdict:
        draw_loupe(verdict, cfg.results_dir / "fig3_loupe.png")
        hung.append("fig3_loupe")
    return {"hung": ", ".join(hung) or "nothing; run the earlier acts first"}
