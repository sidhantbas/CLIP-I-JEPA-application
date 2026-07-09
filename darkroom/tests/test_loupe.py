"""The loupe must judge fairly before it judges the prints."""

from darkroom.loupe import corpus_bleu, words_of


def test_words_of_drops_punctuation_and_case():
    assert words_of("A dog runs .") == ["a", "dog", "runs"]
    assert words_of("The man's hat, red!") == ["the", "man's", "hat", "red"]


def test_perfect_print_scores_one():
    refs = [[words_of("a black dog leaps into the water")]]
    hyp = [words_of("a black dog leaps into the water")]
    scores = corpus_bleu(hyp, refs)
    assert all(abs(scores[f"bleu{n}"] - 1.0) < 1e-9 for n in (1, 2, 3, 4))


def test_disjoint_print_scores_zero():
    refs = [[words_of("a black dog leaps into the water")]]
    hyp = [words_of("purple elephants discuss philosophy quietly")]
    assert corpus_bleu(hyp, refs)["bleu4"] == 0.0


def test_brevity_is_penalised():
    ref = words_of("a black dog leaps into the cold water")
    full = corpus_bleu([ref], [[ref]])
    short = corpus_bleu([ref[:4]], [[ref]])
    assert short["bleu1"] < full["bleu1"]


def test_clipping_blocks_stuttering():
    refs = [[words_of("the dog runs")]]
    hyp = [words_of("the the the the")]
    assert corpus_bleu(hyp, refs)["bleu1"] < 0.5
