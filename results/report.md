# atelier report

Master seed 7. All numbers below are read from the JSON files in results/.

## The interrogation

| witness | factor | glance | scrutiny | gap | chance |
|---|---|---|---|---|---|
| namer | form | 0.814 | 0.801 | -0.012 | 0.250 |
| namer | color | 0.875 | 0.852 | -0.023 | 0.200 |
| namer | position | 0.793 | 0.761 | -0.032 | 0.111 |
| namer | orientation | 0.319 | 0.286 | -0.033 | 0.125 |
| sculptor | form | 0.375 | 0.418 | +0.043 | 0.250 |
| sculptor | color | 0.835 | 0.860 | +0.024 | 0.200 |
| sculptor | position | 0.348 | 0.379 | +0.030 | 0.111 |
| sculptor | orientation | 0.130 | 0.124 | -0.006 | 0.125 |
| pixels | form | 0.365 | 0.442 | +0.077 | 0.250 |
| pixels | color | 0.798 | 0.833 | +0.034 | 0.200 |
| pixels | position | 0.701 | 0.721 | +0.020 | 0.111 |
| pixels | orientation | 0.130 | 0.192 | +0.062 | 0.125 |

## What the numbers say

The Namer exposes caption-mentioned factors almost linearly: its largest gap over form, color and position is -0.012. That matches the naming hypothesis.

The Sculptor's largest geometric gap (position, orientation) is +0.030. The gap is small, so at this scale the Sculptor's geometry is either already linear or simply absent; see the presence column.

Orientation under scrutiny: Namer 0.286, Sculptor 0.124, pixels 0.192, chance 0.125. The Sculptor's orientation sits near chance: the collapse seen in full I-JEPA reappears in miniature, localizing the cause to the objective rather than the scale.

## The handshake

Retrieval among 512 candidates: @1 0.096, @5 0.388. THE BRIDGE IS NARROW: retrieval@5 under 50%, a finding in itself.

## The barman

| features | glance cost | scrutiny cost |
|---|---|---|
| Namer | 42.4% | 75.8% |
| Sculptor | 24.6% | 14.4% |

Training the cost head moved the Namer by +33.4% and the Sculptor by -10.2%. Planning tracked what the probes said each space affords: a trained readout paid off exactly where the goal factor was present, and no readout rescued features that lack it.

Caveat, added by Phase 9: the Sculptor cells above scored mean-pooled features, which Phase 8 showed destroy position; the confound is part of the record, and the second pour below rescores those cells with token-aware readouts.

## Phase 8: where the geometry died

| encoder | factor | glance_pooled | scrutiny_pooled | glance_tokens | attend | scrutiny_tokens |
|---|---|---|---|---|---|---|
| namer | position | 0.793 | 0.761 | 0.773 | 0.786 | 0.807 |
| namer | orientation | 0.319 | 0.286 | 0.307 | 0.325 | 0.350 |
| sculptor | position | 0.348 | 0.379 | 0.689 | 0.436 | 0.338 |
| sculptor | orientation | 0.130 | 0.124 | 0.127 | 0.124 | 0.122 |
| pixels | position | 0.701 | 0.721 | 0.558 |  |  |
| pixels | orientation | 0.130 | 0.192 | 0.117 |  |  |

Position: the Sculptor's pooled scrutiny read 0.379, its unpooled token glance reads 0.689 and its attentive probe 0.436, against an unpooled pixels floor of 0.558. Verdict: pooling confound.

Orientation: pooled scrutiny 0.124, token glance 0.127, attend 0.124, pixels floor 0.117. Verdict: collapse confirmed.

The unpooled pixels floor for position (0.558) sits below the pooled global-PCA floor: a patch knows its content but not its index, so the per-patch probe must assemble position by comparing 64 local views, while global PCA hands position to the probe in its leading components. One oddity reported as found: the token MLP underperforms the token glance on the Sculptor's position, a wide-input MLP fitting worse than its own linear special case under this training budget.

Form and color were skipped: both were linearly saturated pooled.

## Phase 9: the barman looks again

Sculptor planning on the identical 500 episodes: pooled glance 24.6%, pooled scrutiny 14.4%, token glance 21.0%, attend 35.2%, token scrutiny 20.0% (budget limited, per the Phase 8 oddity). Namer reference: glance 42.4%, scrutiny 75.8%. Verdict: pooled blind. The Phase 6 sentence, that no readout rescues absent signal, is hereby retired for position: the signal was never absent, only pooled away. Planning tracks readout-reachable information. Cost heads are bilinear in (state, goal), not concatenated: an additive linear form cannot express state-goal interaction, and distance-to-goal is exactly that interaction.

The handshake footnote: retraining the bridge into attentively pooled Sculptor space (same MLP, InfoNCE and budget) moved retrieval among 512 from 38.8% to 75.5% at @5, and 9.6% to 31.4% at @1. Bottleneck: pooling.
