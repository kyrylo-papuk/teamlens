"""Train the SetFit model and compare the classifiers.

  python -m teamlens.triage.compare train      try the variants below, keep the best on validation
  python -m teamlens.triage.compare            score keywords, SetFit and Claude on the test titles
  python -m teamlens.triage.compare --no-llm   the same without the paid calls

The model variant is chosen on the validation titles and reported on the test titles, which play no
part in the choice. The test set is small (8 titles per area), so differences of a few points
between classifiers are within noise; the table is read for the order of magnitude.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from teamlens.config import Settings
from teamlens.triage import AREAS, CONFIDENT, MODEL_DIR, ClaudeTriage, KeywordTriage, Prediction, SetFitTriage
from teamlens.triage.dataset import split

# (base model, training titles per area). All run on a laptop CPU.
VARIANTS = [
    ("sentence-transformers/paraphrase-MiniLM-L3-v2", 8),
    ("sentence-transformers/paraphrase-MiniLM-L3-v2", 16),
    ("sentence-transformers/all-MiniLM-L6-v2", 8),
    ("sentence-transformers/all-MiniLM-L6-v2", 16),
    ("BAAI/bge-small-en-v1.5", 16),
]
OUT = Path("evals/results/triage.json")
SELECTION = MODEL_DIR / "selection.json"
PARALLEL = 8


def accuracy(predictions: list[Prediction], truth: list[str]) -> float:
    return sum(p.area == t for p, t in zip(predictions, truth)) / len(truth)


def train() -> None:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    _, validation, _ = split()
    titles, truth = [t for t, _ in validation], [a for _, a in validation]
    tried, best = [], None
    for base, shots in VARIANTS:
        rows, _, _ = split(shots)
        data = Dataset.from_dict({"text": [t for t, _ in rows], "label": [a for _, a in rows]})
        model = SetFitModel.from_pretrained(base, labels=AREAS)
        started = time.perf_counter()
        args = TrainingArguments(batch_size=16, num_epochs=1, seed=7,
                                 output_dir=str(MODEL_DIR.with_name("triage-checkpoints")))  # under models/, not in git
        Trainer(model=model, args=args, train_dataset=data).train()
        seconds = time.perf_counter() - started
        folder = MODEL_DIR.with_name("triage-candidate")
        model.save_pretrained(str(folder))
        score = accuracy(asyncio.run(SetFitTriage(folder).classify(titles)), truth)
        tried.append({"base_model": base, "titles_per_area": shots, "validation_accuracy": round(score, 3),
                      "train_seconds": round(seconds)})
        print(f"  {base:48} {shots:>2} per area   validation {score:.1%}   trained in {seconds:.0f}s")
        if best is None or score > best[0]:  # earlier variants are smaller, so a tie keeps the smaller one
            best = (score, base, shots)
            model.save_pretrained(str(MODEL_DIR))
    SELECTION.write_text(json.dumps({"chosen": {"base_model": best[1], "titles_per_area": best[2]}, "tried": tried},
                                    indent=1), encoding="utf-8")
    print(f"kept {best[1]} with {best[2]} titles per area ({best[0]:.1%} on validation), saved to {MODEL_DIR}")


def score(name: str, titles: list[str], predictions: list[Prediction], truth: list[str], ms: float, usd: float,
          note: str = "") -> dict:
    predicted = [p.area for p in predictions]
    f1 = []
    for area in AREAS:
        hit = sum(p == area == t for p, t in zip(predicted, truth))
        precision, recall = hit / max(1, predicted.count(area)), hit / max(1, truth.count(area))
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"classifier": name, "accuracy": round(accuracy(predictions, truth), 3), "macro_f1": round(sum(f1) / len(f1), 3),
            "ms_per_ticket": round(ms, 1), "usd_per_1000_tickets": round(1000 * usd / len(truth), 4), "note": note,
            "wrong": [{"title": title, "said": p, "is": t} for title, p, t in zip(titles, predicted, truth) if p != t]}


def compare(with_llm: bool) -> None:
    _, _, test = split()
    titles, truth = [t for t, _ in test], [a for _, a in test]
    results = []

    local: dict[str, list[Prediction]] = {}
    for classifier in (KeywordTriage(), SetFitTriage()):
        asyncio.run(classifier.classify(titles[:2]))  # warm up, so one-off loading is not counted
        started = time.perf_counter()
        local[classifier.name] = asyncio.run(classifier.classify(titles))
        ms = 1000 * (time.perf_counter() - started) / len(titles)
        results.append(score(classifier.name, titles, local[classifier.name], truth, ms, 0.0))
    setfit_ms = results[-1]["ms_per_ticket"]

    if with_llm:
        llm = ClaudeTriage(Settings().anthropic_api_key, PARALLEL)
        started = time.perf_counter()
        answers = asyncio.run(llm.classify(titles))
        llm_ms = 1000 * (time.perf_counter() - started) * PARALLEL / len(titles)  # time of one call, not wall clock / n
        results.append(score(llm.name, titles, answers, truth, llm_ms, llm.usd))
        # A cascade: keep SetFit's answer when it is confident, ask Claude only about the rest.
        unsure = [p.confidence < CONFIDENT for p in local["setfit"]]
        mixed = [a if u else s for s, a, u in zip(local["setfit"], answers, unsure)]
        share = sum(unsure) / len(unsure)
        results.append(score("setfit, then claude", titles, mixed, truth, setfit_ms + share * llm_ms, llm.usd * share,
                             note=f"{share:.0%} of tickets went to Claude (SetFit confidence below {CONFIDENT})"))

    chosen = json.loads(SELECTION.read_text(encoding="utf-8")) if SELECTION.exists() else {}
    print(f"{len(titles)} test titles ({len(titles) // len(AREAS)} per area)\n")
    print(f"{'classifier':22}{'accuracy':>9}{'macro-F1':>10}{'ms/ticket':>11}{'$ per 1000':>12}")
    for r in results:
        print(f"{r['classifier']:22}{r['accuracy']:>9.1%}{r['macro_f1']:>10.3f}{r['ms_per_ticket']:>11}"
              f"{r['usd_per_1000_tickets']:>12}  {r['note']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"test_titles": len(titles), "setfit_selection": chosen, "results": results},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
    else:
        compare(with_llm="--no-llm" not in sys.argv)
