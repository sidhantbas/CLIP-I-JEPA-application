"""Exhibits for the vitrine: the shipped data, serialized honestly.

Small tensors go to JSON rounded at six decimals (the parity tolerance
is 1e-4, two orders coarser). Large tensors go to raw little-endian
float32 or uint8 files beside a JSON manifest, because a million
numbers spelled out in text would quadruple the page weight for
nothing. Every file is written from the trained weights or the frozen
world, never by hand.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from atelier.config import AtelierConfig
from atelier.interrogation_unpooled import carve_token_grid
from atelier.world import Tokenizer, harvest, summon_world


def as_list(t: Tensor):
    """Nested python lists with six-decimal floats, for JSON."""
    return np.round(t.detach().cpu().numpy().astype(np.float64), 6).tolist()


def write_f32(t: Tensor, path: Path) -> str:
    """Raw little-endian float32, returns the filename for manifests."""
    t.detach().cpu().numpy().astype("<f4").tofile(path)
    return path.name


def canvas_b64(canvas_u8: Tensor) -> str:
    """(3, 64, 64) uint8 -> base64 of HWC row-major RGB bytes."""
    return base64.b64encode(
        canvas_u8.permute(1, 2, 0).contiguous().numpy().tobytes()).decode()


def linear_json(linear: nn.Linear) -> dict:
    return {"weight": as_list(linear.weight), "bias": as_list(linear.bias)}


def dump_tokenizer(vocabulary: list[str], max_len: int, path: Path) -> None:
    """The closed language and its encoding rules: word ids start at 2
    in sorted-vocabulary order, 0 pads, 1 marks the unknown, commas are
    stripped and words split on spaces."""
    path.write_text(json.dumps({
        "vocabulary": vocabulary, "max_len": max_len,
        "pad": Tokenizer.PAD, "unk": Tokenizer.UNK,
        "rules": "strip commas, split on spaces, ids are 2 + sorted index",
    }, indent=1))


def dump_probes(grown: dict, path: Path) -> None:
    """Every regrown probe: weights, the standardization it was trained
    behind, and its recomputed accuracy for cross-checking the JSONs."""
    bundle = {}
    for name, (probe, mean, std, acc) in grown.items():
        entry = {"mean": as_list(mean), "std": as_list(std),
                 "accuracy": round(acc, 4)}
        if isinstance(probe, nn.Linear):
            entry["kind"] = "glance"
            entry.update(linear_json(probe))
        else:                                     # the attentive probe
            entry["kind"] = "attend"
            entry.update(query=as_list(probe.query), key=linear_json(probe.key),
                         value=linear_json(probe.value),
                         head=linear_json(probe.head))
        bundle[name] = entry
    path.write_text(json.dumps(bundle))


def dump_heads(heads: dict, out: Path) -> None:
    """The barman's cost heads. The token glance's bilinear table is a
    million floats, shipped as raw f32; everything else is JSON."""
    tg, at = heads["token_glance"], heads["attend"]
    manifest = {
        "press_mean": write_f32(tg.mean, out / "press_mean.f32"),
        "press_std": write_f32(tg.std, out / "press_std.f32"),
        "token_glance": {
            "pair": write_f32(tg.pair.weight.squeeze(0), out / "token_glance_pair.f32"),
            "pair_bias": as_list(tg.pair.bias),
            "state": write_f32(tg.state.weight.squeeze(0), out / "token_glance_state.f32"),
            "state_bias": as_list(tg.state.bias), "goal": linear_json(tg.goal),
        },
        "attend": {"query": as_list(at.query), "key": linear_json(at.key),
                   "value": linear_json(at.value),
                   "pair": as_list(at.pair.weight.squeeze(0)),
                   "pair_bias": as_list(at.pair.bias),
                   "state": linear_json(at.state), "goal": linear_json(at.goal)},
    }
    (out / "heads.json").write_text(json.dumps(manifest))


def dump_bank(cfg: AtelierConfig, eye: nn.Module, out: Path) -> None:
    """The imagination's picture book: 128 fresh scenes cut into 8192
    patches, each with its settled, per-token LayerNormed eye latent,
    the same normalization the imagination was trained to predict."""
    canvases, _ = harvest(cfg, cfg.seed + 5, 128)
    with torch.no_grad():
        tokens = carve_token_grid(eye, canvases, cfg.device)      # (N, 64, D)
        latents = nn.functional.layer_norm(tokens, tokens.shape[-1:])
    write_f32(latents.reshape(-1, latents.shape[-1]), out / "bank_latents.f32")
    n = len(canvases)
    patches = (canvases.reshape(n, 3, 8, 8, 8, 8)
               .permute(0, 2, 4, 3, 5, 1)                 # (N, gy, gx, py, px, c)
               .reshape(n * 64, 192).numpy())
    patches.tofile(out / "bank_patches.u8")
    (out / "bank.json").write_text(json.dumps(
        {"count": n * 64, "dim": int(latents.shape[-1]), "patch": 8,
         "latents": "bank_latents.f32", "patches": "bank_patches.u8"}))


def dump_constellation(cfg: AtelierConfig, eye: nn.Module, grip: nn.Module,
                       out: Path) -> None:
    """512 held-out scenes placed on a fixed 2D map per space: features
    are unit-normalized, then projected by that space's own two leading
    principal axes. The axes ship too, so a live goal embedding lands
    on the same map by the same arithmetic. It is a projection and the
    interface says so."""
    canvases, scenes = harvest(cfg, cfg.seed + 4, 512)
    with torch.no_grad():
        tokens = carve_token_grid(eye, canvases, cfg.device)
        feats = {"pooled": tokens.mean(dim=1), "attentive": grip(tokens)}
    spaces, coords = {}, {}
    for name, f in feats.items():
        f = nn.functional.normalize(f, dim=-1)
        center = f.mean(dim=0)
        _, _, v = torch.pca_lowrank(f - center, q=2, niter=6)     # (D, 2)
        spaces[name] = {"center": as_list(center), "axes": as_list(v)}
        coords[name] = (f - center) @ v                           # (512, 2)
    points = [{"pooled": as_list(coords["pooled"][i]),
               "attentive": as_list(coords["attentive"][i]),
               "caption": s.caption, "form": s.factors[0].form,
               "color": s.factors[0].color, "grid": s.factors[0].grid}
              for i, s in enumerate(scenes)]
    (out / "constellation.json").write_text(
        json.dumps({"spaces": spaces, "points": points}))


def world_reference(cfg: AtelierConfig, gaze: nn.Module, tongue: nn.Module,
                    tokenizer: Tokenizer) -> dict:
    """The Gate W yardstick: the Namer's in-set retrieval@1 over 1024
    Python-rendered scenes; the TS world must come within 3 points.
    1024 keeps the sampling noise near one point so the gate judges the
    worlds, not the dice."""
    n = 1024
    world = summon_world(cfg, cfg.seed + 6)
    scenes = world.conjure(n)
    with torch.no_grad():
        images = torch.stack([s.canvas for s in scenes]).to(cfg.device)
        tokens = tokenizer.encode_batch([s.caption for s in scenes]).to(cfg.device)
        sight = nn.functional.normalize(gaze(images), dim=-1)
        speech = nn.functional.normalize(tongue(tokens), dim=-1)
        hits = ((sight @ speech.T).argmax(dim=1).cpu()
                == torch.arange(n)).float().mean().item()
    return {"python_retrieval_at_1": round(hits, 4), "n": n,
            "gate": "TS-rendered scenes must retrieve within 0.03 of this"}
