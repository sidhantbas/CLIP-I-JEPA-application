# atelier

A miniature world where two ways of knowing meet.

Two small encoders are trained from scratch on the same procedurally
generated world of colored shapes. One learns by naming, one learns by
completing. Then both are interrogated: not only "what do you know?"
but "how cheaply will you tell?" The distinction between what a
representation contains (presence) and what it exposes to a linear
readout (accessibility) turns out to be the whole story, all the way
down to whether a toy planner can act on a sentence.

Everything trains end to end in about 16 minutes on an Apple M4 Pro
(MPS), and runs, slowly, on CPU. Pure PyTorch, no downloads, no
external weights, one seed for the entire pipeline.

## The fable

**The Namer** believes that to know the world is to name it. A
convolutional gaze looks at each canvas, a small transformer tongue
reads its caption, and a contrastive accord pulls matching pairs
together. It works: in-batch retrieval@1 reaches 97%. But the Namer
only ever learns what language bothers to say, and in this world the
captions mention orientation just half the time.

**The Sculptor** believes that to know the world is to complete it. The
canvas becomes 64 patch tokens, a veil hides a contiguous block, the
eye encodes only what remains, and the imagination predicts what the
memory, an EMA twin that always sees everything, encodes at the hidden
positions. No pixels are reconstructed, no words are involved. Its
relative regret falls fourfold, ending three times better than the best
context-free guess, and its latents never collapse.

**The Handshake** is a two-layer MLP that carries a sentence from the
Namer's text space into the Sculptor's wordless latent space. It is a
narrow bridge: among 512 candidate canvases, the right one is found
first 9.6% of the time and within the top five 38.8% of the time. Far
above the 1% of chance, but narrow. Phase 9's footnote found the
narrowness was the pool's fault, not the space's: the same bridge,
retrained into attentively pooled Sculptor space on the same budget,
reaches 31.4% at first guess and 75.5% within five.

**The Second Sitting** came later, when the interrogators returned to
the Sculptor's studio and, instead of asking for a summary, walked the
room token by token. Position, which the pooled probes had declared
almost absent, was there all along: a linear glance across the 64
unpooled tokens reads it at 0.69, better than the same glance over raw
pixel patches. Mean-pooling had averaged it away before the first
probes ever saw it. Orientation, though, stayed at chance under every
readout, pooled or not, linear or attentive or deep. The position
verdict was a pooling confound; the orientation collapse is the
objective's own signature.

**The Barman** is the smallest planner that makes the argument
operational. Given an order, a caption naming the cell where a shape
should rest, he greedily nudges the shape, scoring candidate next
states against the goal embedding. With Namer features, a plain cosine
glance succeeds 42.4% of the time and a trained scrutiny head 75.8%.
With mean-pooled Sculptor features, the glance manages 24.6% and the
scrutiny head drops to 14.4%. Phase 6 read this as signal absence; the
Second Sitting revealed those cells had scored a representation with a
known wound.

**The Second Pour** rescored exactly those Sculptor cells, same 500
episodes, with cost heads that reach the tokens. An attentive head of
35k parameters lifts planning success to 35.2%, within seven points of
the Namer's glance, and retraining the bridge into attentively pooled
Sculptor space lifts caption retrieval@5 among 512 from 38.8% to
75.5%. The barman was never geometry-blind; he was pooled blind. The
honest wrinkle: the widest heads (a bilinear over all 8192 token
dimensions, and a token MLP) plan worse than the little attentive one,
the same wide-input underfitting Phase 8 flagged.

## The findings

The full numbers live in `results/report.md` and the JSON files it is
generated from. In brief, with master seed 7:

| witness | factor | glance (linear) | scrutiny (MLP) | gap |
|---|---|---|---|---|
| Namer | form | 0.814 | 0.801 | -0.012 |
| Namer | color | 0.875 | 0.852 | -0.023 |
| Namer | position | 0.793 | 0.761 | -0.032 |
| Namer | orientation | 0.319 | 0.286 | -0.033 |
| Sculptor | form | 0.375 | 0.418 | +0.043 |
| Sculptor | color | 0.835 | 0.860 | +0.024 |
| Sculptor | position | 0.348 | 0.379 | +0.030 |
| Sculptor | orientation | 0.130 | 0.124 | -0.006 |
| pixels (PCA) | position | 0.701 | 0.721 | +0.020 |
| pixels (PCA) | orientation | 0.130 | 0.192 | +0.062 |

1. **The naming hypothesis held.** Every factor the captions name is
   exposed by the Namer almost perfectly linearly: all its gaps are
   zero or negative. Orientation, named only half the time, is learned
   only halfway (0.32 against 0.125 chance).

2. **The orientation collapse reproduced in miniature, in its strongest
   form.** The Sculptor's orientation probes sit at chance pooled
   (0.130 glance, 0.124 scrutiny) and stay at chance unpooled (token
   glance 0.127, attentive 0.124, token MLP 0.122), even though a
   scrutiny of raw pixels reaches 0.192. No readout can rescue it, so
   the flattening reported for full I-JEPA is an encoder-level
   destruction by the objective itself, reproduced at 1.3M parameters
   and 3,000 steps.

3. **The gap hypothesis failed for pooled features, and Phase 8 found
   out why.** Pooled, the Sculptor's position presence looked low
   (scrutiny 0.38, below the 0.72 of PCA pixels). Unpooled, a plain
   linear glance across its 64 tokens reads position at 0.689, above
   the unpooled per-patch pixels floor of 0.558. The geometry did not
   die in the encoder; it died in the mean-pool. Position was a pooling
   confound, and full-scale probing pipelines should pool attentively
   rather than average. (The attentive probe here recovers part of it,
   0.436, with only 35k parameters.)

4. **Planning tracked readout-reachable information, and Phase 9 retired
   the stronger claim.** Phase 6's 2x2 said a better readout pays off
   only where the pooled probes found signal (Namer +33.4%, Sculptor
   -10.2%). Phase 9 reran the Sculptor cells with token-aware cost
   heads on the identical episodes: an attentive head reaches 35.2%
   (from a pooled best of 24.6%), within the decision rule's margin of
   the Namer's glance cell, and the bridge retrained into attentively
   pooled space nearly doubles retrieval. The Phase 6 sentence that no
   readout rescues absent signal is retired for position: the signal
   was never absent, only pooled away. The record of the confound
   stays in the report; it is part of the story.

One design note: every shape is drawn with a nose, a small notch toward
its facing direction. Without it, a circle has no visible orientation
at all and the orientation probes would be ill-posed for a quarter of
all scenes.

## The figures

| | |
|---|---|
| `fig1_the_world.png` | sixteen scenes and their captions |
| `fig2_presence_vs_access.png` | the thesis figure: glance vs scrutiny per factor per witness |
| `fig3_gap_profile.png` | the accessibility gap profiles |
| `fig4_handshake.png` | retrieval@k for the bridge |
| `fig5_barman.png` | the 2x2 planning success matrix |
| `fig6_pooling_verdict.png` | where the geometry died: pooled vs unpooled readouts |
| `fig7_the_second_pour.png` | Sculptor planning across all cost heads, Namer as reference |

Figures are rendered from the JSON results only, never from live
memory, so the argument can always be re-drawn:
`python -m atelier --stage gallery`.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install torch numpy matplotlib tqdm
.venv/bin/python -m atelier                       # the whole play, about 16 min on MPS
.venv/bin/python -m atelier --stage namer         # a single act
.venv/bin/python -m atelier --stage none          # just print the config
.venv/bin/python -m pytest tests                  # the tests
```

Stages: `world`, `namer`, `sculptor`, `interrogation`, `unpooled`,
`handshake`, `barman`, `encore`, `gallery`. One seed (`--seed`,
default 7) controls world generation, both trainings, the probes, the
bridge and the planner.

## The layout

```
atelier/
  __main__.py        # orchestration: the whole play, act by act
  config.py          # every knob in one dataclass, one seed to rule the run
  world.py           # Act 0: the world and its speech
  namer.py           # Act I: mini-CLIP, to know is to name
  sculptor.py        # Act II: mini-JEPA, to know is to complete
  interrogation.py   # Act III: probes, what is present vs what is legible
  interrogation_unpooled.py  # Act III, second sitting: token-level readouts
  probing_common.py  # the interrogation room's shared tools
  handshake.py       # Act IV: the bridge from words to latent space
  barman.py          # Act V: the planner that reads only what a glance affords
  barman_heads.py    # Act V, second pour: cost heads that reach the tokens
  gallery.py         # the argument, drawn
```

The code is written in a style we call lyrical: names carry the theory
(veil, regret, glance, scrutiny), docstrings carry the mechanism, and
the module structure mirrors the conceptual structure, so that reading
the repo teaches the ideas. Poetry in the name, physics in the
docstring.
