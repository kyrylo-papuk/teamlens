"""Is this ticket already in Jira under other words?

Two rankings of the existing issues are available and can be merged (reciprocal rank fusion):
  words    BM25 over the title's tokens: exact terms, issue-specific names, error codes
  meaning  cosine similarity of sentence embeddings: the same problem described differently
How much the word ranking counts is a measured choice, not a default. On duplicates reported in
other words, which is what the evaluation pairs are, ranking by meaning alone found the right
issue most often and mixing in the word ranking made it worse, so the weight of words is zero.
That is a statement about reworded duplicates. Tickets that share an error code or an endpoint
name are where words earn their weight, and real history is what should decide it.

An item is called a *possible duplicate* only above a similarity threshold, so a new ticket with
nothing like it gets no warning. Weight and thresholds are chosen on the validation pairs and
reported on the test pairs by ``python -m teamlens.similar``.

The index lives in memory and is rebuilt from the team's open issues, a few hundred titles at
most. At company scale the same two rankings would come from Postgres (pgvector plus full-text
search); the interface would not change.

Without the optional ``ml`` extra there are no embeddings and the search runs on words alone.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The three values below are chosen on the validation pairs by `python -m teamlens.similar`, which prints them.
MEANING_THRESHOLD = 0.44  # cosine similarity above which an issue is called a possible duplicate
WORDS_THRESHOLD = 0.20  # the same for word overlap, used when there are no embeddings
WORDS_WEIGHT = 0.0  # share of the word ranking in the merged ranking; 0 means rank by meaning alone
RRF_K = 60  # the usual constant of reciprocal rank fusion
OUT = Path("evals/results/duplicates.json")

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("a an and are as at be by for from has have if in into is it its not of on or so than that the their "
           "there this to up use when while with we you once about after before every each all any can cannot".split())
Embedder = Callable[[list[str]], np.ndarray]


class Match(BaseModel):
    key: str
    summary: str
    similarity: float  # cosine similarity when embeddings are available, word overlap otherwise
    possible_duplicate: bool


def tokens(text: str) -> list[str]:
    words = [w for w in TOKEN.findall(text.lower()) if w not in STOP and len(w) > 1]
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]  # crude plural stripping


def load_embedder() -> Embedder | None:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception:  # optional dependency missing, or the model cannot be fetched
        return None
    return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class Finder:
    def __init__(self, embed: Embedder | None = None) -> None:
        self._embed = embed
        self._vectors: dict[str, np.ndarray] = {}

    @property
    def mode(self) -> str:
        return "words and meaning" if self._embed else "words only"

    def _vector(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if new:
            self._vectors.update(zip(new, self._embed(new)))
        return np.array([self._vectors[t] for t in texts])

    @staticmethod
    def _bm25(query: list[str], documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
        n = len(documents)
        average = sum(len(d) for d in documents) / max(1, n)
        seen_in = Counter(word for d in documents for word in set(d))
        scores = []
        for d in documents:
            counts = Counter(d)
            score = 0.0
            for word in set(query):
                if counts[word]:
                    idf = math.log(1 + (n - seen_in[word] + 0.5) / (seen_in[word] + 0.5))
                    score += idf * counts[word] * (k1 + 1) / (counts[word] + k1 * (1 - b + b * len(d) / average))
            scores.append(score)
        return scores

    def rank(self, query: str, items: list[tuple[str, str]], method: str = "both",
             words_weight: float = WORDS_WEIGHT) -> list[Match]:
        """All ``items`` (key, title), most similar first. ``method``: words | meaning | both."""
        if not items or not query.strip():
            return []
        q = tokens(query)
        documents = [tokens(text) for _, text in items]
        words = self._bm25(q, documents)
        overlap = [len(set(q) & set(d)) / max(1, len(set(q))) for d in documents]
        meaning = None
        if self._embed and method != "words":
            meaning = (self._vector([text for _, text in items]) @ self._vector([query])[0]).tolist()

        def order(scores: list[float]) -> dict[int, int]:
            ranked = sorted(range(len(items)), key=lambda i: -scores[i])
            return {i: position for position, i in enumerate(ranked, start=1)}

        if meaning is None:
            fused = words
        elif method == "meaning" or (method == "both" and words_weight == 0):
            fused = meaning
        else:
            by_words, by_meaning = order(words), order(meaning)
            fused = [words_weight / (RRF_K + by_words[i]) + (1 - words_weight) / (RRF_K + by_meaning[i])
                     for i in range(len(items))]

        out = []
        for i in sorted(range(len(items)), key=lambda i: -fused[i]):
            similarity = meaning[i] if meaning is not None else overlap[i]
            limit = MEANING_THRESHOLD if meaning is not None else WORDS_THRESHOLD
            out.append(Match(key=items[i][0], summary=items[i][1], similarity=round(float(similarity), 2),
                             possible_duplicate=similarity >= limit))
        return out

    def search(self, query: str, items: list[tuple[str, str]], k: int = 3) -> list[Match]:
        """The few most similar issues that pass the threshold. Empty when nothing looks like a duplicate."""
        return [m for m in self.rank(query, items)[:k] if m.possible_duplicate]


# --- evaluation ----------------------------------------------------------------------------------

WEIGHTS = (0.0, 0.25, 0.5)


def _evaluate() -> dict:
    from teamlens.similar_data import NOVEL, PAIRS, VALIDATION_NOVEL, VALIDATION_PAIRS
    from teamlens.triage.dataset import TICKETS

    titles = [t for area in TICKETS.values() for t in area]
    items = [(f"T-{n}", t) for n, t in enumerate(titles, start=1)]
    key_of = {t: k for k, t in items}
    finder, words_only = Finder(load_embedder()), Finder(None)
    if finder.mode == "words only":
        sys.exit("The comparison needs embeddings: install the `ml` extra.")
    validation, test = PAIRS[:VALIDATION_PAIRS], PAIRS[VALIDATION_PAIRS:]
    new_validation, new_test = NOVEL[:VALIDATION_NOVEL], NOVEL[VALIDATION_NOVEL:]

    def ranked(query: str, method: str, weight: float = 0.0) -> list[Match]:
        return words_only.rank(query, items) if method == "words" else finder.rank(query, items, method, weight)

    def recall(pairs, method: str, weight: float = 0.0) -> dict:
        tops = [[m.key for m in ranked(p, method, weight)[:3]] for _, p in pairs]
        return {"right_one_first": round(sum(top[0] == key_of[o] for top, (o, _) in zip(tops, pairs)) / len(pairs), 3),
                "in_the_top_3": round(sum(key_of[o] in top for top, (o, _) in zip(tops, pairs)) / len(pairs), 3)}

    # 1. How much should words count? Decided on validation.
    by_weight = {w: recall(validation, "both", w) for w in WEIGHTS}
    weight = max(WEIGHTS, key=lambda w: (by_weight[w]["right_one_first"], by_weight[w]["in_the_top_3"], -w))

    # 2. Where is the line between "a duplicate" and "just the nearest thing"? Decided on validation.
    def similarity_of_original(pair, method: str) -> float:
        """Similarity of the true original if it is among the top three, else 0: a hint nobody sees catches nothing."""
        original, paraphrase = pair
        return next((m.similarity for m in ranked(paraphrase, method, weight)[:3] if m.key == key_of[original]), 0.0)

    def nearest(query: str, method: str) -> float:
        return ranked(query, method, weight)[0].similarity

    def line(method: str) -> dict:
        dup = [similarity_of_original(p, method) for p in validation]
        new = [nearest(n, method) for n in new_validation]
        steps = [x / 100 for x in range(5, 96)]

        def separation(t: float) -> float:  # share of duplicates caught minus share of new tickets flagged
            return sum(d >= t for d in dup) / len(dup) - sum(n >= t for n in new) / len(new)
        best = max(separation(t) for t in steps)
        tied = [t for t in steps if separation(t) == best]
        chosen = tied[len(tied) // 2]  # the middle of the widest gap between the two groups
        caught = sum(similarity_of_original(p, method) >= chosen for p in test)
        flagged = sum(nearest(n, method) >= chosen for n in new_test)
        return {"threshold": chosen, "duplicates_caught": f"{caught}/{len(test)}",
                "new_tickets_wrongly_flagged": f"{flagged}/{len(new_test)}"}

    return {"indexed_titles": len(items), "validation_pairs": len(validation), "test_pairs": len(test),
            "test_new_tickets": len(new_test), "embedding_model": EMBEDDING_MODEL,
            "words_weight": {"chosen_on_validation": weight, "validation": {str(w): r for w, r in by_weight.items()}},
            "ranking_on_test": {"words": recall(test, "words"), "meaning": recall(test, "meaning"),
                                "equal mix": recall(test, "both", 0.5)},
            "flagging_on_test": {"meaning": line("meaning"), "words": line("words")}}


if __name__ == "__main__":
    result = _evaluate()
    print(f"{result['indexed_titles']} titles indexed. Chosen on {result['validation_pairs']} validation pairs, reported on "
          f"{result['test_pairs']} reworded duplicates and {result['test_new_tickets']} genuinely new tickets.\n")
    print(f"{'ranking':12}{'right one first':>17}{'in the top 3':>14}")
    for method, r in result["ranking_on_test"].items():
        print(f"{method:12}{r['right_one_first']:>17.0%}{r['in_the_top_3']:>14.0%}")
    print(f"\nweight of words in the merged ranking, chosen on validation: {result['words_weight']['chosen_on_validation']}")
    for method, r in result["flagging_on_test"].items():
        print(f"flag by {method:8} threshold {r['threshold']:.2f}:  duplicates caught {r['duplicates_caught']},  "
              f"new tickets wrongly flagged {r['new_tickets_wrongly_flagged']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nsaved to {OUT}")
