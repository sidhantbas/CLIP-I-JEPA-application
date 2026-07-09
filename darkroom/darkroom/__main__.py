"""Orchestration: the session, act by act.

Running `python -m darkroom` performs everything in order: the contact
sheet exposes every photograph through the frozen CLIP eye, the
Developer trains, the Prints are made from the test negatives, the
Loupe judges them, the Gallery hangs the results. `--stage <name>`
performs a single act.

Every act module exposes one entry point, `perform(cfg)`, which returns
a small dict of headline numbers that is echoed to the console.
"""

from __future__ import annotations

import argparse
import time

from .config import DarkroomConfig, summon_determinism

STAGES: tuple[str, ...] = (
    "contact",
    "developer",
    "prints",
    "loupe",
    "gallery",
)


def _perform(stage: str, cfg: DarkroomConfig) -> dict:
    """Imports the act lazily and runs it, so early acts never need late ones."""
    if stage == "contact":
        from . import contact_sheet as act
    elif stage == "developer":
        from . import developer as act
    elif stage == "prints":
        from . import prints as act
    elif stage == "loupe":
        from . import loupe as act
    elif stage == "gallery":
        from . import gallery as act
    else:
        raise ValueError(f"unknown stage: {stage}")
    return act.perform(cfg)


def main() -> None:
    """Parses arguments, prints the config, performs the requested acts."""
    parser = argparse.ArgumentParser(
        prog="darkroom",
        description="Where photographs are developed into sentences.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", "none") + STAGES,
        help="which act to perform; 'all' runs the whole session in order",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the master seed")
    args = parser.parse_args()

    cfg = DarkroomConfig()
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
    print(f"\nthe session took {time.time() - curtain:.1f}s in total.")


if __name__ == "__main__":
    main()
