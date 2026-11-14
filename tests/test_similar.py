import numpy as np

from teamlens.similar import MEANING_THRESHOLD, Finder, tokens

ISSUES = [
    ("PAY-1", "Payout request returns 500 when the beneficiary IBAN has spaces"),
    ("PAY-2", "Dashboard chart overflows its card on narrow laptop screens"),
    ("PAY-3", "Flaky test in the refund suite fails once in ten runs"),
    ("PAY-4", "Rotate the database credentials and move them to the secret store"),
]


def test_tokens_drop_noise_and_plurals():
    assert tokens("The payouts are failing for IBANs with spaces") == ["payout", "failing", "iban", "space"]


def test_words_only_finds_a_reworded_copy_and_ignores_a_new_ticket():
    finder = Finder()
    assert finder.mode == "words only"
    found = finder.search("Payout returns a 500 error if the IBAN contains spaces", ISSUES)
    assert [m.key for m in found] == ["PAY-1"] and found[0].possible_duplicate
    assert finder.search("Translate the help centre into Spanish", ISSUES) == []
    assert finder.search("", ISSUES) == [] and finder.search("anything", []) == []


def test_meaning_ranks_by_embedding_and_applies_its_own_threshold():
    """A stand-in embedder: each text maps to a known direction, so the expected order is unambiguous."""
    directions = {"query": [1, 0, 0], ISSUES[0][1]: [0.9, 0.1, 0], ISSUES[1][1]: [0.3, 0.9, 0],
                  ISSUES[2][1]: [0, 0, 1], ISSUES[3][1]: [0.1, 1, 0]}
    calls = []

    def embed(texts):
        calls.append(list(texts))
        vectors = np.array([directions[t] for t in texts], dtype=float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    finder = Finder(embed)
    ranked = finder.rank("query", ISSUES)
    assert finder.mode == "words and meaning" and [m.key for m in ranked] == ["PAY-1", "PAY-2", "PAY-4", "PAY-3"]
    assert ranked[0].similarity > MEANING_THRESHOLD > ranked[1].similarity
    assert [m.key for m in finder.search("query", ISSUES)] == ["PAY-1"]
    finder.rank("query", ISSUES)
    assert len(calls) == 2  # titles and the query were embedded once, then served from the cache
