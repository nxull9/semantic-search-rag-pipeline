"""Tune the settings of each collection and the router.

    python scripts/tune_collections.py

For every document it tries several chunk sizes and overlaps, measures how often the
top chunk contains the answer, and calibrates the minimum similarity score. It then
calibrates the router threshold that decides whether any document covers a question.
"""

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from multi_document import load_config  # noqa: E402
from semantic_search import SemanticSearch, chunk_text, cosine_similarity, load_text  # noqa: E402

SIZES = [300, 500, 800]
OVERLAPS = [10, 15, 20]


def separating_threshold(positives, negatives):
    """Threshold between scores that should be accepted and scores that should be rejected.

    If the two groups don't overlap, use the midpoint of the gap so there is a safety margin on
    both sides. Otherwise use the value that classifies the most questions correctly.
    """
    if positives.min() > negatives.max():
        threshold = (positives.min() + negatives.max()) / 2
    else:
        candidates = np.unique(np.concatenate([positives, negatives]))
        threshold = max(candidates, key=lambda t: (positives >= t).sum() + (negatives < t).sum())
    accuracy = ((positives >= threshold).sum() + (negatives < threshold).sum()) / (len(positives) + len(negatives))
    return float(threshold), float(accuracy)


def main() -> None:
    model_name, _, configs = load_config(ROOT / "config" / "collections.json")
    paths = {c.name: c for c in configs}
    model = SentenceTransformer(model_name)
    questions = json.loads((ROOT / "data" / "eval_questions.json").read_text())
    unrelated = questions["unrelated"]
    tuned = {}

    for name, qs in questions["collections"].items():
        text, _ = load_text(ROOT / paths[name].document)
        answerable = qs["answerable"]
        negatives_q = qs["near_topic"] + unrelated
        print(f"\n## {name}")
        print("  size overlap chunks  hit@1  coverage  hit@3  avg_best_score")
        results = []
        for size, overlap in product(SIZES, OVERLAPS):
            engine = SemanticSearch(chunk_text(text, size, overlap), model=model)
            hit1 = hit3 = 0
            best_scores = []
            for question, keyword in answerable:
                scores = engine.scores(question)
                top = np.argsort(scores)[::-1][:3]
                hit1 += keyword.lower() in engine.chunks[top[0]].lower()
                hit3 += any(keyword.lower() in engine.chunks[i].lower() for i in top)
                best_scores.append(scores[top[0]])
            covered = 0
            for question, keywords in qs.get("broad", []):
                top3 = " ".join(engine.chunks[i] for i in np.argsort(engine.scores(question))[::-1][:3]).lower()
                covered += all(k.lower() in top3 for k in keywords)
            n, coverage = len(answerable), covered / max(len(qs.get("broad", [])), 1)
            results.append((hit1 / n, coverage, hit3 / n, float(np.mean(best_scores)), size, overlap, engine))
            print(f"  {size:4} {overlap:5}%  {len(engine.chunks):5}  {hit1 / n:5.0%}  {coverage:8.0%}  {hit3 / n:5.0%}  {np.mean(best_scores):.3f}")

        # best = highest hit@1, then full coverage of broad questions, then hit@3, then the strongest average match
        hit1, coverage, hit3, avg, size, overlap, engine = max(results, key=lambda r: r[:4])
        positives = np.array([engine.scores(q).max() for q, _ in answerable])
        negatives = np.array([engine.scores(q).max() for q in negatives_q])
        threshold, accuracy = separating_threshold(positives, negatives)
        tuned[name] = {"chunk_size": size, "overlap_percent": overlap, "min_score": round(threshold, 2)}
        print(f"  -> chosen: size {size}, overlap {overlap}% (hit@1 {hit1:.0%}, coverage {coverage:.0%}, hit@3 {hit3:.0%})")
        print(f"     answerable best scores: min {positives.min():.3f}, mean {positives.mean():.3f}")
        print(f"     rejected   best scores: max {negatives.max():.3f}, mean {negatives.mean():.3f}")
        print(f"     min_score {threshold:.3f} -> {accuracy:.0%} of questions accepted/rejected correctly")

    # Router: question vs document summaries
    names = list(questions["collections"])
    summaries = [(ROOT / paths[n].summary).read_text().strip() for n in names]
    summary_vectors = np.asarray(model.encode(summaries))
    routed, route_scores = 0, []
    print("\n## Router")
    for expected, qs in questions["collections"].items():
        for question, _ in qs["answerable"]:
            scores = cosine_similarity(np.asarray(model.encode(question)), summary_vectors)
            chosen = names[int(np.argmax(scores))]
            routed += chosen == expected
            route_scores.append(scores.max())
            if chosen != expected:
                print(f"  misrouted: '{question}' -> {chosen} (expected {expected})")
    total = sum(len(qs["answerable"]) for qs in questions["collections"].values())
    unrelated_scores = np.array([cosine_similarity(np.asarray(model.encode(q)), summary_vectors).max() for q in unrelated])
    # The router must never block a real question, so its threshold sits just below the lowest
    # answerable score. Rejecting unrelated questions is left to each collection's min_score.
    route_threshold = np.floor(min(route_scores) * 100) / 100
    blocked = (unrelated_scores < route_threshold).sum()
    print(f"  routing accuracy: {routed}/{total} = {routed / total:.0%}")
    print(f"  answerable route scores: min {min(route_scores):.3f}, mean {np.mean(route_scores):.3f}")
    print(f"  unrelated  route scores: max {unrelated_scores.max():.3f}, mean {unrelated_scores.mean():.3f}")
    print(f"  route_min_score {route_threshold:.2f}: lets every answerable question through, "
          f"blocks {blocked}/{len(unrelated)} unrelated questions at the routing step")

    tuned["_router"] = {"route_min_score": round(float(route_threshold), 2)}
    print("\nTuned settings (copy into config/collections.json):\n" + json.dumps(tuned, indent=2))


if __name__ == "__main__":
    main()
