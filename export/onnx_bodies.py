"""Bodies for the vitrine: the frozen encoders, re-expressed for export.

torch.onnx tracing bakes python ints into constants, so every adapter
here rewrites the frozen modules' forwards with explicit tensor ops
(unflatten instead of view, gather instead of arange indexing, additive
masks instead of masked_fill on bools). The weights are the trained
ones, untouched; only the choreography of the forward changes, and the
parity gate proves the two dances are the same step for step.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HEADS = 4


def flow_layer(x: Tensor, layer: nn.TransformerEncoderLayer,
               pad_bias: Tensor | None = None) -> Tensor:
    """One norm-first encoder layer with explicit attention math:
    x + attn(ln1(x)), then x + ff(ln2(x)). pad_bias is (B, 1, 1, L),
    zero where visible and -1e9 where padded."""
    width = (layer.norm1.weight.shape[0],)         # static, so ONNX-constant
    h = F.layer_norm(x, width, layer.norm1.weight, layer.norm1.bias)
    qkv = h @ layer.self_attn.in_proj_weight.T + layer.self_attn.in_proj_bias
    q, k, v = qkv.chunk(3, dim=-1)                     # each (B, L, D)
    q = q.unflatten(-1, (HEADS, -1)).transpose(1, 2)   # (B, H, L, dh)
    k = k.unflatten(-1, (HEADS, -1)).transpose(1, 2)
    v = v.unflatten(-1, (HEADS, -1)).transpose(1, 2)
    scores = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)  # (B, H, L, L)
    if pad_bias is not None:
        scores = scores + pad_bias
    mixed = (scores.softmax(dim=-1) @ v).transpose(1, 2).flatten(-2)  # (B, L, D)
    x = x + mixed @ layer.self_attn.out_proj.weight.T + layer.self_attn.out_proj.bias
    h = F.layer_norm(x, width, layer.norm2.weight, layer.norm2.bias)
    return x + layer.linear2(F.gelu(layer.linear1(h)))


class TongueBody(nn.Module):
    """The tongue, export-safe: token ids (B, 24) int64 -> (B, D).
    Last non-pad token gathered with tensor indices, never arange."""

    def __init__(self, tongue: nn.Module):
        super().__init__()
        self.tongue = tongue

    def forward(self, tokens: Tensor) -> Tensor:
        real = (tokens != 0).to(torch.float32)                    # (B, L)
        x = self.tongue.embed(tokens) + self.tongue.positions
        pad_bias = (1.0 - real)[:, None, None, :] * -1e9          # (B,1,1,L)
        for layer in self.tongue.trunk.layers:
            x = flow_layer(x, layer, pad_bias)
        last = real.sum(dim=-1, keepdim=True).to(torch.int64) - 1  # (B, 1)
        idx = last.unsqueeze(-1).expand(-1, -1, x.shape[-1])       # (B, 1, D)
        return self.tongue.project(x.gather(1, idx).squeeze(1))    # (B, D)


class EyeBody(nn.Module):
    """The eye, export-safe: canvas (B, 3, 64, 64) plus kept patch
    indices (Nk,) int64 -> tokens (B, Nk, D). Passing all 64 indices is
    the full unveiled encoding; passing fewer is the veiled context,
    hidden patches absent before attention, exactly as in training."""

    def __init__(self, eye: nn.Module):
        super().__init__()
        self.eye = eye

    def forward(self, images: Tensor, keep: Tensor) -> Tensor:
        x = self.eye.patch_embed(images).flatten(2).transpose(1, 2)
        x = (x + self.eye.positions()).index_select(1, keep)      # (B, Nk, D)
        for layer in self.eye.trunk.layers:
            x = flow_layer(x, layer)
        return F.layer_norm(x, (self.eye.settle.weight.shape[0],),
                            self.eye.settle.weight, self.eye.settle.bias)


class ImaginationBody(nn.Module):
    """The imagination, export-safe: visible context (B, Nv, D) plus
    hidden patch indices (Nh,) -> the full predicted sequence
    (B, Nv + Nh, D). The caller slices off the last Nh guesses; slicing
    stays outside the graph so both axes remain dynamic."""

    def __init__(self, imagination: nn.Module):
        super().__init__()
        self.imagination = imagination

    def forward(self, context: Tensor, hidden: Tensor) -> Tensor:
        queries = self.imagination.mask_token \
            + self.imagination.pos.index_select(0, hidden)        # (Nh, D)
        queries = queries.unsqueeze(0) + torch.zeros_like(context[:, :1, :1])
        x = torch.cat([context, queries], dim=1)                  # (B, Nv+Nh, D)
        for layer in self.imagination.trunk.layers:
            x = flow_layer(x, layer)
        return self.imagination.project(x)


def ship(module: nn.Module, args: tuple, path: Path, input_names: list[str],
         output_name: str, dynamic: dict) -> None:
    """Traces the module to ONNX, fp32, opset 17, dynamic batch axes."""
    module.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module, args, str(path), input_names=input_names,
        output_names=[output_name], dynamic_axes=dynamic,
        opset_version=17, dynamo=False,
    )
