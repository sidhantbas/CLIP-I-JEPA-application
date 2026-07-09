"""The one export: everything the vitrine needs, made from the frozen work.

Run as `python -m export.for_the_vitrine` from the repo root. Loads the
trained checkpoints, regrows the ephemeral readouts (readouts.py),
traces the encoders to ONNX (onnx_bodies.py), serializes the exhibits
(exhibits.py), and writes a parity manifest: the PyTorch outputs of
every exported component on 32 fixed seed-7 scenes, so the browser can
prove, before any lab opens, that it runs the same models. Training
happens on the fast device; every parity number is computed on cpu in
fp32, which is what ONNX Runtime's WASM backend speaks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from torch import nn

from atelier.config import AtelierConfig, summon_determinism
from atelier.handshake import resurrect as resurrect_handshake
from atelier.interrogation_unpooled import carve_token_grid
from atelier.namer import embed_sights, resurrect as resurrect_namer
from atelier.sculptor import carve_features, resurrect as resurrect_sculptor, veil
from atelier.world import harvest, summon_world

from .exhibits import (as_list, canvas_b64, dump_bank, dump_constellation,
                       dump_heads, dump_probes, dump_tokenizer, world_reference)
from .onnx_bodies import EyeBody, ImaginationBody, TongueBody, ship
from .readouts import regrow_grip, regrow_heads, regrow_imagination, regrow_probes

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "export" / "out"


def main(cfg: AtelierConfig | None = None) -> None:
    cfg = cfg or AtelierConfig()
    summon_determinism(cfg.seed)
    for leaf in ("models", "data", "results"):
        (OUT / leaf).mkdir(parents=True, exist_ok=True)
    gaze, tongue, tokenizer = resurrect_namer(cfg)
    eye = resurrect_sculptor(cfg)
    bridge_pooled = resurrect_handshake(cfg)

    print("regrowing the imagination")
    imagination = regrow_imagination(cfg, eye)
    print("regrowing the grip and its bridge")
    grip, bridge_attentive = regrow_grip(cfg, tongue, tokenizer, eye)

    print("regrowing the probes")
    tr_canv, tr_scenes = harvest(cfg, cfg.seed + 1, cfg.probe_train_scenes)
    te_canv, te_scenes = harvest(cfg, cfg.seed + 2, cfg.probe_test_scenes)
    pooled = {"namer": (embed_sights(gaze, tr_canv, cfg.device),
                        embed_sights(gaze, te_canv, cfg.device)),
              "sculptor": (carve_features(eye, tr_canv, cfg.device),
                           carve_features(eye, te_canv, cfg.device))}
    tokens_pair = (carve_token_grid(eye, tr_canv, cfg.device),
                   carve_token_grid(eye, te_canv, cfg.device))
    probes = regrow_probes(cfg, pooled, tokens_pair, (tr_scenes, te_scenes))

    print("regrowing the cost heads")

    @torch.no_grad()
    def goal_embed(orders):
        speech = tongue(tokenizer.encode_batch(
            [o.caption for o in orders]).to(cfg.device))
        return nn.functional.normalize(bridge_pooled(speech), dim=-1)

    heads = regrow_heads(cfg, eye, goal_embed)
    torch.save({"imagination": imagination, "grip": grip,
                "bridge_attentive": bridge_attentive, "heads": heads,
                "probes": probes}, OUT / "regrown.pt")

    print("dumping the exhibits")
    data = OUT / "data"
    dump_tokenizer(summon_world(cfg, cfg.seed).vocabulary(),
                   cfg.max_caption_len, data / "tokenizer.json")
    dump_probes(probes, data / "probes.json")
    dump_heads(heads, data)
    dump_bank(cfg, eye, data)
    dump_constellation(cfg, eye, grip, data)
    reference = world_reference(cfg, gaze, tongue, tokenizer)

    print("tracing to onnx")
    gaze, tongue, eye = gaze.cpu(), tongue.cpu(), eye.cpu()
    bridge_pooled = bridge_pooled.cpu()
    models = OUT / "models"
    img = torch.rand(2, 3, 64, 64)
    keep = torch.arange(40, dtype=torch.int64)
    ship(gaze, (img,), models / "gaze.onnx", ["canvas"], "embedding",
         {"canvas": {0: "batch"}, "embedding": {0: "batch"}})
    ship(TongueBody(tongue), (torch.ones(2, 24, dtype=torch.int64),),
         models / "tongue.onnx", ["tokens"], "embedding",
         {"tokens": {0: "batch"}, "embedding": {0: "batch"}})
    ship(EyeBody(eye), (img, keep), models / "eye.onnx", ["canvas", "keep"],
         "tokens", {"canvas": {0: "batch"}, "keep": {0: "kept"},
                    "tokens": {0: "batch", 1: "kept"}})
    ship(ImaginationBody(imagination),
         (torch.rand(2, 40, 128), torch.arange(40, 64, dtype=torch.int64)),
         models / "imagination.onnx", ["context", "hidden"], "sequence",
         {"context": {0: "batch", 1: "visible"}, "hidden": {0: "hiddenn"},
          "sequence": {0: "batch", 1: "length"}})
    ship(grip, (torch.rand(2, 64, 128),), models / "grip.onnx", ["tokens"],
         "summary", {"tokens": {0: "batch"}, "summary": {0: "batch"}})
    for name, bridge in (("bridge_pooled", bridge_pooled),
                         ("bridge_attentive", bridge_attentive)):
        ship(bridge, (torch.rand(2, 128),), models / f"{name}.onnx", ["speech"],
             "reached", {"speech": {0: "batch"}, "reached": {0: "batch"}})

    print("writing the parity manifest")
    manifest = attest(cfg, gaze, tongue, tokenizer, eye, imagination, grip,
                      bridge_pooled, bridge_attentive, probes, heads)
    manifest["world_reference"] = reference
    (ROOT / "export" / "parity.json").write_text(json.dumps(manifest))
    for source in list((cfg.results_dir).glob("*.json")):
        shutil.copy(source, OUT / "results" / source.name)
    shutil.copy(ROOT / "export" / "parity.json", OUT / "results" / "parity.json")
    print(f"done; the vitrine's crates are in {OUT}")


@torch.no_grad()
def attest(cfg, gaze, tongue, tokenizer, eye, imagination, grip,
           bridge_pooled, bridge_attentive, probes, heads) -> dict:
    """PyTorch outputs of every exported component on 32 fixed scenes,
    all on cpu fp32: the truth the browser must reproduce within 1e-4."""
    scenes = summon_world(cfg, cfg.seed).conjure(32)
    images = torch.stack([s.canvas for s in scenes])              # (32, 3, 64, 64)
    images = (images * 255).round() / 255      # the browser sees uint8 pixels
    tokens = tokenizer.encode_batch([s.caption for s in scenes])
    visible, hidden = veil(8, cfg.sculptor_mask_ratio,
                           torch.Generator().manual_seed(cfg.seed))
    gaze_out, tongue_out, eye_out = gaze(images), tongue(tokens), eye(images)
    context = eye(images, keep=visible)
    goal = nn.functional.normalize(bridge_pooled(tongue_out), dim=-1)
    probe_logits = {}
    for name, (probe, mean, std, _) in probes.items():
        feats = {"namer": gaze_out, "sculptor": eye_out.mean(dim=1)}[
            name.split(".")[0]]
        if name.endswith("glance_tokens"):
            feats = eye_out.flatten(1)
        elif name.endswith("attend"):
            feats = eye_out
        probe_logits[name] = as_list(probe((feats - mean) / std))
    return {
        "tolerance": 1e-4,
        "veil": {"visible": visible.tolist(), "hidden": hidden.tolist()},
        "scenes": [{"caption": s.caption, "tokens": tokens[i].tolist(),
                    "canvas_b64": canvas_b64((s.canvas * 255).round().byte()),
                    "factors": [vars(f) for f in s.factors]}
                   for i, s in enumerate(scenes)],
        "outputs": {
            "gaze": as_list(gaze_out), "tongue": as_list(tongue_out),
            "eye": as_list(eye_out),
            "context": as_list(context),
            "imagination": as_list(imagination(context, hidden)),
            "grip": as_list(grip(eye_out)),
            "bridge_pooled": as_list(bridge_pooled(tongue_out)),
            "bridge_attentive": as_list(bridge_attentive(tongue_out)),
            "probes": probe_logits,
            "heads": {name: as_list(head(eye_out, goal))
                      for name, head in heads.items()},
        },
    }


if __name__ == "__main__":
    main()
