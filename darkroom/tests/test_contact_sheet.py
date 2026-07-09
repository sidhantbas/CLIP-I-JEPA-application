"""The contact sheet's paperwork: parsing, splits, alignment."""

from darkroom.config import DarkroomConfig
from darkroom.contact_sheet import read_captions, read_split


def test_every_photo_has_five_sentences():
    cfg = DarkroomConfig()
    sheet = read_captions(cfg)
    assert len(sheet) >= 8000
    assert all(len(sentences) == 5 for sentences in sheet.values())


def test_sentences_are_tidy():
    cfg = DarkroomConfig()
    sheet = read_captions(cfg)
    some = next(iter(sheet.values()))
    assert all("\t" not in s and "  " not in s for s in some)
    assert not any(s.endswith(" .") for s in some)


def test_splits_are_disjoint_and_full():
    cfg = DarkroomConfig()
    train, dev, test = (read_split(cfg, s) for s in ("train", "dev", "test"))
    assert len(train) == 6000
    assert 990 <= len(dev) <= 1000 and 990 <= len(test) <= 1000
    assert not (set(train) & set(dev)) and not (set(dev) & set(test))
    captions = read_captions(cfg)
    assert all(n in captions for n in train + dev + test)
