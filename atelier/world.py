"""Act 0: the world and its speech.

A closed procedural world of 64x64 canvases holding 1 to 3 shapes, each
with four factors: form, color, position, orientation. Every canvas is
born with a templated caption, and orientation is mentioned in only a
configurable fraction of them, so language sometimes omits what the eye
can see. Orientation is made visible by a nose: a small background
colored notch punched into each shape toward its facing direction.
Without it the circle would have no readable orientation at all and the
square only a quarter of one, and the orientation probes are ill-posed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

FORMS: tuple[str, ...] = ("triangle", "square", "circle", "star")
COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.90, 0.12, 0.12), "blue": (0.15, 0.35, 0.92),
    "green": (0.10, 0.75, 0.30), "yellow": (0.95, 0.85, 0.12),
    "violet": (0.62, 0.22, 0.85),
}
COMPASS_8: tuple[str, ...] = ("east", "north-east", "north", "north-west",
                              "west", "south-west", "south", "south-east")
GRID_NAMES: tuple[tuple[str, str, str], ...] = (
    ("north-west", "north", "north-east"),
    ("west", "center", "east"),
    ("south-west", "south", "south-east"),
)
BACKGROUND: tuple[float, float, float] = (0.08, 0.08, 0.10)


@dataclass
class ShapeFactors:
    """Ground truth for one shape: the four factors and their discretizations."""

    form: str
    color: str
    x: float
    y: float
    grid: str
    angle: float          # degrees in [0, 360), 0 is east, counterclockwise
    facing: str           # angle discretized into 8 compass bins
    size: float           # circumradius in pixels

    @property
    def size_word(self) -> str:
        return "small" if self.size < 11.5 else "large"


@dataclass
class Scene:
    canvas: Tensor        # (3, 64, 64), float in [0, 1]
    caption: str
    factors: list[ShapeFactors]   # sorted largest first; factors[0] is primary


def _polygon_sdf(px: np.ndarray, py: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Signed distance from every pixel to a polygon, negative inside.
    Winding-sign algorithm, vectorized over pixels, handles the star."""
    d = np.full(px.shape, np.inf)
    sign = np.ones(px.shape)
    n = len(verts)
    for i in range(n):
        j = (i - 1) % n
        ex, ey = verts[j, 0] - verts[i, 0], verts[j, 1] - verts[i, 1]
        wx, wy = px - verts[i, 0], py - verts[i, 1]
        t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
        d = np.minimum(d, (wx - ex * t) ** 2 + (wy - ey * t) ** 2)
        c1, c2, c3 = py >= verts[i, 1], py < verts[j, 1], ex * wy > ey * wx
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        sign = np.where(flip, -sign, sign)
    return sign * np.sqrt(d)


def shape_alpha(px: np.ndarray, py: np.ndarray, f: ShapeFactors) -> np.ndarray:
    """Anti-aliased coverage mask of a shape, with the nose punched out."""
    a = np.deg2rad(f.angle)
    heading = np.array([np.cos(a), -np.sin(a)])   # image y axis points down
    if f.form == "circle":
        sdf = np.sqrt((px - f.x) ** 2 + (py - f.y) ** 2) - f.size
    else:
        if f.form == "triangle":
            angles = a + np.deg2rad([0.0, 120.0, 240.0])
        elif f.form == "square":
            angles = a + np.deg2rad([45.0, 135.0, 225.0, 315.0])
        else:                                     # star, 5 points
            angles = a + np.deg2rad(np.arange(10) * 36.0)
        radii = f.size * np.ones(len(angles))
        if f.form == "star":
            radii[1::2] *= 0.45
        verts = np.stack(
            [f.x + radii * np.cos(angles), f.y - radii * np.sin(angles)], axis=1
        )
        sdf = _polygon_sdf(px, py, verts)
    alpha = np.clip(0.5 - sdf, 0.0, 1.0)          # 1 px anti-aliased edge
    nose = np.array([f.x, f.y]) + 0.55 * f.size * heading
    nose_sdf = np.sqrt((px - nose[0]) ** 2 + (py - nose[1]) ** 2) - 0.22 * f.size
    return alpha * (1.0 - np.clip(0.5 - nose_sdf, 0.0, 1.0))


class World:
    """The generator of scenes and the closed language that describes them."""

    def __init__(self, seed: int, canvas_size: int = 64, min_shapes: int = 1,
                 max_shapes: int = 3, caption_mentions_orientation_p: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.canvas_size = canvas_size
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes
        self.orientation_p = caption_mentions_orientation_p
        ys, xs = np.mgrid[0:canvas_size, 0:canvas_size]
        self._px, self._py = xs + 0.5, ys + 0.5    # pixel centers

    def _birth_shape(self) -> ShapeFactors:
        size = float(self.rng.uniform(8.0, 15.0))
        lo, hi = size + 1.0, self.canvas_size - size - 1.0
        x, y = float(self.rng.uniform(lo, hi)), float(self.rng.uniform(lo, hi))
        cell = self.canvas_size / 3.0
        grid = GRID_NAMES[min(int(y / cell), 2)][min(int(x / cell), 2)]
        angle = float(self.rng.uniform(0.0, 360.0))
        facing = COMPASS_8[int(((angle + 22.5) % 360.0) // 45.0)]
        return ShapeFactors(
            form=str(self.rng.choice(FORMS)),
            color=str(self.rng.choice(list(COLORS))),
            x=x, y=y, grid=grid, angle=angle, facing=facing, size=size,
        )

    def _speak(self, factors: list[ShapeFactors]) -> str:
        """Caption: primary shape with place, sometimes facing, then
        each companion as a beside clause."""
        primary = factors[0]
        caption = (
            f"a {primary.size_word} {primary.color} {primary.form}"
            f" rests in the {primary.grid}"
        )
        if self.rng.random() < self.orientation_p:
            caption += f", facing {primary.facing}"
        for other in factors[1:]:
            caption += f", beside a {other.size_word} {other.color} {other.form}"
        return caption

    def conjure(self, n: int) -> list[Scene]:
        """Generates n scenes: renders shapes largest-last so the primary
        shape is never occluded, and captions each canvas."""
        scenes = []
        for _ in range(n):
            count = int(self.rng.integers(self.min_shapes, self.max_shapes + 1))
            factors = sorted(
                (self._birth_shape() for _ in range(count)),
                key=lambda f: -f.size,
            )
            canvas = np.empty((self.canvas_size, self.canvas_size, 3))
            canvas[:] = BACKGROUND
            for f in reversed(factors):            # smallest first, primary on top
                alpha = shape_alpha(self._px, self._py, f)[..., None]  # (64, 64, 1)
                canvas = canvas * (1.0 - alpha) + np.array(COLORS[f.color]) * alpha
            tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float()  # (3, 64, 64)
            scenes.append(Scene(canvas=tensor, caption=self._speak(factors),
                                factors=factors))
        return scenes

    def vocabulary(self) -> list[str]:
        """Every word the world can speak, sorted, punctuation stripped."""
        words = {"a", "small", "large", "rests", "in", "the", "facing", "beside",
                 "center"}
        words |= set(FORMS) | set(COLORS) | set(COMPASS_8)
        return sorted(words)


def recite_back(caption: str) -> dict:
    """Parses a caption back into its stated facts: primary form, color,
    size word, grid, facing (or None), plus companion shape words."""
    clauses = caption.split(", ")
    head = clauses[0].split()                     # a SIZE COLOR FORM rests in the GRID
    fact = {"size_word": head[1], "color": head[2], "form": head[3],
            "grid": head[7], "facing": None, "beside": []}
    for clause in clauses[1:]:
        part = clause.split()
        if part[0] == "facing":
            fact["facing"] = part[1]
        else:                                     # beside a SIZE COLOR FORM
            fact["beside"].append((part[2], part[3], part[4]))
    return fact


class Tokenizer:
    """Word-level tokenizer over the closed vocabulary, fixed length,
    with pad and unk. Encodes to (max_len,) long tensors."""

    PAD, UNK = 0, 1

    def __init__(self, vocabulary: list[str], max_len: int = 24):
        self.max_len = max_len
        self.word_to_id = {w: i + 2 for i, w in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary) + 2

    def encode(self, caption: str) -> Tensor:
        ids = [self.word_to_id.get(w, self.UNK)
               for w in caption.replace(",", "").split()][: self.max_len]
        ids += [self.PAD] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)   # (max_len,)

    def encode_batch(self, captions: list[str]) -> Tensor:
        return torch.stack([self.encode(c) for c in captions])  # (B, max_len)


def summon_world(cfg, seed: int) -> World:
    """Builds a World from the config with an explicit (derived) seed."""
    return World(seed, cfg.canvas_size, cfg.min_shapes, cfg.max_shapes,
                 cfg.caption_mentions_orientation_p)


def harvest(cfg, seed: int, n: int) -> tuple[Tensor, list[Scene]]:
    """Conjures n scenes and stacks canvases as uint8 (n, 3, 64, 64) to keep
    memory small. Returns the stack and the scenes (captions, factors)."""
    scenes = summon_world(cfg, seed).conjure(n)
    canvases = torch.stack([(s.canvas * 255).round().byte() for s in scenes])
    return canvases, scenes


def perform(cfg) -> dict:
    """The data check: conjures scenes, verifies every caption against its
    factors via recite_back, and reports the world's basic statistics."""
    world = summon_world(cfg, cfg.seed)
    scenes = world.conjure(256)
    mentioned = 0
    for scene in scenes:
        fact = recite_back(scene.caption)
        primary = scene.factors[0]
        assert (fact["form"], fact["color"], fact["grid"], fact["size_word"]) == (
            primary.form, primary.color, primary.grid, primary.size_word), scene.caption
        if fact["facing"] is not None:
            assert fact["facing"] == primary.facing, scene.caption
            mentioned += 1
        assert len(fact["beside"]) == len(scene.factors) - 1
    longest = max(len(Tokenizer(world.vocabulary()).encode(s.caption).nonzero())
                  for s in scenes)
    return {"scenes checked": len(scenes),
            "captions faithful": "yes, all parsed back to their factors",
            "orientation mentioned": f"{mentioned}/{len(scenes)}",
            "vocabulary size": len(world.vocabulary()),
            "longest caption (tokens)": longest}
