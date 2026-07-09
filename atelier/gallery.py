"""The argument, drawn. Figures come from the JSON files, never from
live memory, so the whole case can be re-drawn at any time. The one
exception is fig1, a portrait of the world itself, which is conjured
again from the master seed and is therefore identical every time."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import AtelierConfig
from .world import summon_world

FACTOR_NAMES = ("form", "color", "position", "orientation")
WITNESSES = ("namer", "sculptor", "pixels")
HUES = {"namer": "#c0392b", "sculptor": "#2471a3", "pixels": "#7f8c8d"}


def _load(cfg: AtelierConfig, name: str) -> dict | None:
    path = cfg.results_dir / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def draw_the_world(cfg: AtelierConfig, path: Path) -> None:
    """fig1: a 4x4 grid of scenes with their captions."""
    scenes = summon_world(cfg, cfg.seed).conjure(16)
    fig, axes = plt.subplots(4, 4, figsize=(12, 13))
    for ax, scene in zip(axes.flat, scenes):
        ax.imshow(scene.canvas.permute(1, 2, 0).numpy())
        ax.set_title(textwrap.fill(scene.caption, 34), fontsize=7)
        ax.axis("off")
    fig.suptitle("the world and its speech", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_presence_vs_access(t: dict, path: Path) -> None:
    """fig2, the thesis figure: per factor, glance vs scrutiny for every
    witness, with the chance floor dashed."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, factor in zip(axes, FACTOR_NAMES):
        for i, w in enumerate(WITNESSES):
            g, s = t[w][factor]["glance"], t[w][factor]["scrutiny"]
            ax.bar(i - 0.18, g["mean"], 0.34, yerr=g["std"], color=HUES[w],
                   alpha=0.45, label="glance (linear)" if i == 0 else None)
            ax.bar(i + 0.18, s["mean"], 0.34, yerr=s["std"], color=HUES[w],
                   label="scrutiny (MLP)" if i == 0 else None)
        ax.axhline(t["chance"][factor], ls="--", c="black", lw=1, label="chance")
        ax.set_xticks(range(3), WITNESSES)
        ax.set_title(factor)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("probe accuracy")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("presence (scrutiny) vs linear accessibility (glance);"
                 " pale bars are glances", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_gap_profile(t: dict, path: Path) -> None:
    """fig3: the accessibility gap, scrutiny minus glance, per factor
    and witness, side by side."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.26
    for i, w in enumerate(WITNESSES):
        gaps = [t[w][f]["scrutiny"]["mean"] - t[w][f]["glance"]["mean"]
                for f in FACTOR_NAMES]
        ax.bar([x + (i - 1) * width for x in range(4)], gaps, width,
               color=HUES[w], label=w)
    ax.axhline(0, c="black", lw=1)
    ax.set_xticks(range(4), FACTOR_NAMES)
    ax.set_ylabel("accessibility gap (scrutiny minus glance)")
    ax.set_title("what is present but not cheap to read")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_handshake(h: dict, path: Path) -> None:
    """fig4: retrieval@k for the bridge among the candidate canvases."""
    curve = h["retrieval_at_k"]
    ks = list(range(1, len(curve) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, curve, marker="o", color="#8e44ad", label="handshake")
    ax.plot(ks, [k / h["candidates"] for k in ks], ls="--", c="black",
            label="chance")
    ax.set_xlabel("k")
    ax.set_ylabel(f"retrieval@k among {h['candidates']}")
    ax.set_title("the bridge from words to the wordless space")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_barman(b: dict, path: Path) -> None:
    """fig5: the 2x2 planning success matrix, annotated."""
    grid = [[b["namer"]["glance"], b["namer"]["scrutiny"]],
            [b["sculptor"]["glance"], b["sculptor"]["scrutiny"]]]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(grid, cmap="YlGn", vmin=0, vmax=1)
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{grid[r][c]:.1%}", ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1], ["glance cost", "scrutiny cost"])
    ax.set_yticks([0, 1], ["Namer features", "Sculptor features"])
    ax.set_title(f"planning success over {b['episodes']} episodes")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(cfg: AtelierConfig, t: dict, h: dict | None, b: dict | None,
                 path: Path, u: dict | None = None, e: dict | None = None,
                 hs: dict | None = None) -> None:
    """Generates report.md from the JSON results: the numbers, the gaps,
    and whether the hypothesis held. Plain prose, honest about surprises."""
    lines = ["# atelier report", "",
             f"Master seed {cfg.seed}. All numbers below are read from the "
             "JSON files in results/.", "",
             "## The interrogation", "",
             "| witness | factor | glance | scrutiny | gap | chance |",
             "|---|---|---|---|---|---|"]
    for w in WITNESSES:
        for f in FACTOR_NAMES:
            g, s = t[w][f]["glance"]["mean"], t[w][f]["scrutiny"]["mean"]
            lines.append(f"| {w} | {f} | {g:.3f} | {s:.3f} | {s - g:+.3f} "
                         f"| {t['chance'][f]:.3f} |")
    n, s_, p = t["namer"], t["sculptor"], t["pixels"]
    named = ["form", "color", "position"]
    namer_named_gap = max(n[f]["scrutiny"]["mean"] - n[f]["glance"]["mean"]
                          for f in named)
    sculptor_geo_gap = max(s_[f]["scrutiny"]["mean"] - s_[f]["glance"]["mean"]
                           for f in ("position", "orientation"))
    ori_present = s_["orientation"]["scrutiny"]["mean"]
    lines += ["", "## What the numbers say", ""]
    lines.append(
        f"The Namer exposes caption-mentioned factors almost linearly: its "
        f"largest gap over form, color and position is {namer_named_gap:+.3f}. "
        + ("That matches the naming hypothesis."
           if namer_named_gap < 0.1 else
           "That is larger than the naming hypothesis expected, a surprise "
           "worth noting."))
    lines.append("")
    lines.append(
        f"The Sculptor's largest geometric gap (position, orientation) is "
        f"{sculptor_geo_gap:+.3f}. "
        + ("A clear linear-to-MLP gap: the information is present but not "
           "cheap, as the completion hypothesis predicts."
           if sculptor_geo_gap > 0.1 else
           "The gap is small, so at this scale the Sculptor's geometry is "
           "either already linear or simply absent; see the presence column."))
    lines.append("")
    chance_o = t["chance"]["orientation"]
    lines.append(
        f"Orientation under scrutiny: Namer {n['orientation']['scrutiny']['mean']:.3f}, "
        f"Sculptor {ori_present:.3f}, pixels {p['orientation']['scrutiny']['mean']:.3f}, "
        f"chance {chance_o:.3f}. "
        + ("The Sculptor's orientation sits near chance: the collapse seen in "
           "full I-JEPA reappears in miniature, localizing the cause to the "
           "objective rather than the scale."
           if ori_present < chance_o + 0.1 else
           "The mini Sculptor did not collapse orientation the way full "
           "I-JEPA does, an honest negative for that part of the hypothesis."))
    if h:
        lines += ["", "## The handshake", "",
                  f"Retrieval among {h['candidates']} candidates: "
                  f"@1 {h['retrieval@1']:.3f}, @5 {h['retrieval@5']:.3f}. "
                  f"{h['verdict']}."]
    if b:
        lines += ["", "## The barman", "",
                  "| features | glance cost | scrutiny cost |", "|---|---|---|",
                  f"| Namer | {b['namer']['glance']:.1%} "
                  f"| {b['namer']['scrutiny']:.1%} |",
                  f"| Sculptor | {b['sculptor']['glance']:.1%} "
                  f"| {b['sculptor']['scrutiny']:.1%} |", ""]
        n_gain = b["namer"]["scrutiny"] - b["namer"]["glance"]
        s_gain = b["sculptor"]["scrutiny"] - b["sculptor"]["glance"]
        lines.append(
            f"Training the cost head moved the Namer by {n_gain:+.1%} and the "
            f"Sculptor by {s_gain:+.1%}. "
            + ("Planning tracked what the probes said each space affords: a "
               "trained readout paid off exactly where the goal factor was "
               "present, and no readout rescued features that lack it."
               if (n_gain > 0.05) != (s_gain > 0.05) else
               ("Both spaces gained from a trained readout: success tracked "
                "accessibility, not presence." if n_gain > 0.05 else
                "Neither space gained from a trained readout, so the readout "
                "was not the limiting factor in this world.")))
        if e:
            lines.append("")
            lines.append(
                "Caveat, added by Phase 9: the Sculptor cells above scored "
                "mean-pooled features, which Phase 8 showed destroy position; "
                "the confound is part of the record, and the second pour "
                "below rescores those cells with token-aware readouts.")
    if u:
        from .interrogation_unpooled import report_section
        lines += report_section(u)
    if e and hs:
        from .barman_heads import report_section as encore_section
        lines += encore_section(e, hs)
    path.write_text("\n".join(lines) + "\n")


def perform(cfg: AtelierConfig) -> dict:
    """Draws every figure whose JSON exists and writes report.md."""
    done = {}
    draw_the_world(cfg, cfg.results_dir / "fig1_the_world.png")
    done["fig1"] = "drawn"
    t, u, h, b = (_load(cfg, n) for n in
                  ("interrogation", "interrogation_unpooled", "handshake",
                   "barman"))
    if t:
        draw_presence_vs_access(t, cfg.results_dir / "fig2_presence_vs_access.png")
        draw_gap_profile(t, cfg.results_dir / "fig3_gap_profile.png")
        done["fig2, fig3"] = "drawn"
    if h:
        draw_handshake(h, cfg.results_dir / "fig4_handshake.png")
        done["fig4"] = "drawn"
    if b:
        draw_barman(b, cfg.results_dir / "fig5_barman.png")
        done["fig5"] = "drawn"
    if u:
        from .interrogation_unpooled import draw_verdict
        draw_verdict(cfg, cfg.results_dir / "fig6_pooling_verdict.png")
        done["fig6"] = "drawn"
    e, hs = _load(cfg, "barman_unpooled"), _load(cfg, "handshake_unpooled")
    if e:
        from .barman_heads import draw_second_pour
        draw_second_pour(cfg, cfg.results_dir / "fig7_the_second_pour.png")
        done["fig7"] = "drawn"
    if t:
        write_report(cfg, t, h, b, cfg.results_dir / "report.md", u,
                     e if e and hs else None, hs)
        done["report"] = "written"
    return done
