# darkroom

Where photographs are developed into sentences.

After atelier, where everything was trained from scratch on a world
that fit in a seed, the darkroom works the opposite trade: two frozen
giants and one small bath between them. A miniature ClipCap. The
frozen CLIP eye (ViT-B/32) looks at a photograph once and leaves a
512-d negative behind; the frozen GPT-2 tongue (124M) can continue any
strip of token embeddings put in front of it. The only thing that
trains is **the developer**: a two-layer MLP (31.5M parameters) that
turns one negative into ten prefix pseudo-tokens GPT-2 accepts as if
they were words it had always known. 275M parameters watch; 31.5M
learn.

Flickr8k is the contact sheet: 8,000 photographs, five human sentences
each, the official 6,000 / 1,000 / 1,000 split. Everything runs on an
Apple M4 Pro (MPS); the whole session is about 45 minutes, almost all
of it the developer's bath.

## The acts

**The contact sheet** exposes every photograph through the frozen eye
exactly once (40 seconds for all 8,000) and caches the unit-norm
negatives. No photograph is looked at twice until the gallery wants
portraits.

**The developer** reads 30,000 (photo, sentence) pairs per epoch for
ten epochs. The prefix strip goes in front of the true sentence, and
the tongue's cross-entropy over the real words flows back through the
frozen giant into the developer alone. Train loss falls from 9.6 to
2.0; dev loss bottoms out at 2.530 around epoch 5 and drifts to 2.613
by epoch 10, the honest signature of a small dataset under a 31M-
parameter bath.

**The prints** develop each of the 1,000 test negatives and let the
tongue speak under a five-beam search, blind — no reference caption is
in the room. A thousand prints take 30 seconds, 9.4 words each.

**The loupe** judges twice. BLEU compares each print's n-grams to its
five human sentences. The accord asks the frozen eye itself: does the
print's CLIP text embedding still point at the photograph it came
from? Human captions set the ceiling, shuffled pairings the floor.

## The findings

With master seed 7:

| judge | score |
|---|---|
| BLEU-1 | 0.665 |
| BLEU-2 | 0.482 |
| BLEU-3 | 0.338 |
| BLEU-4 | **0.228** |
| CLIP accord, prints | 0.299 |
| CLIP accord, human captions | 0.319 |
| CLIP accord, shuffled pairs | 0.174 |

1. **Ten pseudo-tokens are enough.** A single frozen CLIP vector,
   developed into ten prefix embeddings, carries enough of the
   photograph for frozen GPT-2 to caption it at BLEU-4 0.228 against
   five references — in the territory of full show-and-tell models
   trained end to end on this dataset.

2. **The eye recognises its own descriptions.** The prints' accord
   (0.299) sits within 0.02 of the human ceiling (0.319) and far above
   the shuffled floor (0.174). By CLIP's own measure, the developed
   sentences point at their photographs almost as firmly as human ones.

3. **The prints are fluent but cautious.** At 9.4 words they run
   shorter than human sentences and favour safe nouns; specificity is
   what BLEU-4 still charges for. The first print of the test set:
   *"Two brown dogs play in the snow."* The human line: *"The dogs are
   in the snow in front of a fence."*

## The figures

| | |
|---|---|
| `fig1_contact_sheet.png` | sixteen test photographs, print in red, one human sentence in blue |
| `fig2_developer_curve.png` | the bath: train loss by step, dev loss by epoch |
| `fig3_loupe.png` | BLEU-1..4 and the accord with its ceiling and floor |

Figures are rendered from the JSON results only (the contact sheet
alone reopens the photographs for their portraits):
`python -m darkroom --stage gallery`.

## Running it

Flickr8k must sit in `data/` (the Illinois mirror's zips, unpacked:
`Flicker8k_Dataset/` and the caption/split text files).

```bash
cd darkroom
../.venv/bin/python -m darkroom                  # the whole session
../.venv/bin/python -m darkroom --stage prints   # a single act
../.venv/bin/python -m darkroom --stage none     # just print the config
../.venv/bin/python -m pytest tests              # the tests
```

Stages: `contact`, `developer`, `prints`, `loupe`, `gallery`. One seed
(`--seed`, default 7) controls the shuffle, the bath and everything
downstream; the giants are frozen, so the negatives never depend on it.

## The layout

```
darkroom/
  __main__.py       # orchestration: the session, act by act
  config.py         # every knob in one dataclass, one seed for the run
  giants.py         # the frozen eye (CLIP) and tongue (GPT-2)
  contact_sheet.py  # Act 0: Flickr8k parsed, every photo exposed once
  developer.py      # Act I: the only bath that changes anything
  prints.py         # Act II: a thousand negatives become sentences
  loupe.py          # Act III: BLEU and the eye's own accord
  gallery.py        # the exhibition, drawn from the JSON alone
```

Same house style as atelier: poetry in the name, physics in the
docstring.
