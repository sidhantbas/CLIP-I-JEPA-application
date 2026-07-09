"""Act III: the interrogation. What is present, and what is legible.

Both encoders are frozen and questioned about the largest shape in each
scene. For every factor two probes are trained: a glance, one linear
layer, measuring what the representation exposes cheaply, and a
scrutiny, a small MLP, measuring what the representation contains at
all. Raw pixels reduced by PCA sit below both as the floor. The
headline quantity is the accessibility gap: scrutiny minus glance.

The probe bodies and the training loop live in probing_common, shared
with the unpooled second sitting, so both sittings ask their questions
in exactly the same voice.
"""

from __future__ import annotations

import json

from torch import Tensor

from .config import AtelierConfig
from .namer import embed_sights
from .namer import resurrect as resurrect_namer
from .probing_common import (CHANCE, FACTORS, build_probe, cross_examine,
                             flatten_to_pca, testify)
from .sculptor import carve_features
from .sculptor import resurrect as resurrect_sculptor
from .world import harvest


def perform(cfg: AtelierConfig) -> dict:
    """Runs the full probe grid and writes results/interrogation.json.

    Witnesses: Namer gaze features, Sculptor eye features (mean-pooled,
    full canvas), and PCA pixels. For each witness and factor, glance
    and scrutiny accuracies over probe_seeds runs, on held-out scenes.
    """
    train_canvases, train_scenes = harvest(cfg, cfg.seed + 1, cfg.probe_train_scenes)
    test_canvases, test_scenes = harvest(cfg, cfg.seed + 2, cfg.probe_test_scenes)

    gaze, _, _ = resurrect_namer(cfg)
    eye = resurrect_sculptor(cfg)
    witnesses: dict[str, tuple[Tensor, Tensor]] = {
        "namer": (embed_sights(gaze, train_canvases, cfg.device),
                  embed_sights(gaze, test_canvases, cfg.device)),
        "sculptor": (carve_features(eye, train_canvases, cfg.device),
                     carve_features(eye, test_canvases, cfg.device)),
        "pixels": flatten_to_pca(train_canvases, test_canvases, cfg.pixel_pca_dim),
    }

    transcript: dict[str, dict] = {"chance": CHANCE}
    for name, features in witnesses.items():
        transcript[name] = {}
        width = features[0].shape[1]
        for factor, n_classes in FACTORS.items():
            labels = (testify(train_scenes, factor), testify(test_scenes, factor))
            transcript[name][factor] = {
                kind: cross_examine(
                    features, labels,
                    lambda k=kind: build_probe(k, width, n_classes,
                                               cfg.scrutiny_hidden), cfg)
                for kind in ("glance", "scrutiny")
            }
            g = transcript[name][factor]
            print(f"  {name:9s} {factor:12s} glance {g['glance']['mean']:.3f} "
                  f"scrutiny {g['scrutiny']['mean']:.3f} "
                  f"gap {g['scrutiny']['mean'] - g['glance']['mean']:+.3f}")

    with open(cfg.results_dir / "interrogation.json", "w") as f:
        json.dump(transcript, f, indent=1)

    headline = {}
    for name in witnesses:
        gaps = {f: round(transcript[name][f]["scrutiny"]["mean"]
                         - transcript[name][f]["glance"]["mean"], 3)
                for f in FACTORS}
        headline[f"{name} accessibility gaps"] = gaps
    return headline
