"""Orchestration: the whole play, act by act.

Running `python -m atelier` performs everything in order: the world is
checked, the Namer and the Sculptor are trained, the Interrogation
probes both, the Handshake bridges them, the Barman plans, the Gallery
draws the argument. `--stage <name>` performs a single act.

Every act module exposes one entry point, `perform(cfg)`, which returns
a small dict of headline numbers that is echoed to the console.
"""

from __future__ import annotations

import argparse
import time

from .config import AtelierConfig, summon_determinism

STAGES: tuple[str, ...] = (
    "world",
    "namer",
    "sculptor",
    "interrogation",
    "unpooled",
    "handshake",
    "barman",
    "encore",
    "gallery",
)


def _perform(stage: str, cfg: AtelierConfig) -> dict:
    """Imports the act lazily and runs it, so early acts never need late ones."""
    if stage == "world":
        from . import world as act
    elif stage == "namer":
        from . import namer as act
    elif stage == "sculptor":
        from . import sculptor as act
    elif stage == "interrogation":
        from . import interrogation as act
    elif stage == "unpooled":
        from . import interrogation_unpooled as act
    elif stage == "handshake":
        from . import handshake as act
    elif stage == "barman":
        from . import barman as act
    elif stage == "encore":
        from . import barman_heads as act
    elif stage == "gallery":
        from . import gallery as act
    else:
        raise ValueError(f"unknown stage: {stage}")
    return act.perform(cfg)


def main() -> None:
    """Parses arguments, prints the config, performs the requested acts."""
    parser = argparse.ArgumentParser(
        prog="atelier",
        description="A miniature world where two ways of knowing meet.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "none") + STAGES,
        help="which act to perform; 'all' runs the whole play in order",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the master seed")
    args = parser.parse_args()

    cfg = AtelierConfig()
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    print(cfg.banner())
    if args.stage == "none":
        return

    program = STAGES if args.stage == "all" else (args.stage,)
    curtain = time.time()
    for stage in program:
        summon_determinism(cfg.seed)
        opening = time.time()
        print(f"\n=== act: {stage} ===")
        headline = _perform(stage, cfg)
        elapsed = time.time() - opening
        for key, value in headline.items():
            print(f"  {key}: {value}")
        print(f"  ({stage} took {elapsed:.1f}s)")
    print(f"\nthe play took {time.time() - curtain:.1f}s in total.")


if __name__ == "__main__":
    main()
