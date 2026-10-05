"""Reproduce the experiments in docs/EXPERIMENTS.md.

    python scripts/run_experiments.py

Prints the chunk-size, overlap and threshold-calibration results and saves
docs/images/threshold_calibration.png.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402

from semantic_search import DEFAULT_MODEL, SemanticSearch, chunk_text, load_text  # noqa: E402

DATA = ROOT / "data" / "sample_football_clubs.txt"
SAUDI_CLUBS = ["Al Hilal", "Al Nassr", "Al Ittihad"]
BROAD_QUESTION = "how many saudi clubs do we have"
IN_TEXT = [
    "Which club signed Karim Benzema?", "Where does Liverpool play?", "What is Juventus called?",
    "Who managed Manchester United?", "When was Al Hilal founded?", "What is Barcelona's motto?",
    "Which fans sing You'll Never Walk Alone?", "Which club shares the San Siro with Inter?",
]
NOT_IN_TEXT = [
    "What is the capital of France?", "How do I bake bread?", "Who won the 2022 World Cup?",
    "What is photosynthesis?", "How much does an iPhone cost?", "Who is the best basketball player?",
    "How do I learn Python?", "What is the stock price of Apple?",
]
SIZES = [200, 300, 500, 800, 1000]


def main() -> None:
    model = SentenceTransformer(DEFAULT_MODEL)
    text, paragraphs = load_text(DATA)
    print(f"{DATA.name}: {len(paragraphs)} paragraphs, {len(text):,} characters\n")

    print(f"## Chunk size vs a broad question: '{BROAD_QUESTION}' (overlap 15%, top_k 3)")
    for size in SIZES:
        engine = SemanticSearch(chunk_text(text, size, 15), model=model)
        for min_score in (0.6, 0.7):
            results = engine.search(BROAD_QUESTION, top_k=3, min_score=min_score)
            found = [c for c in SAUDI_CLUBS if any(c in r.text for r in results)]
            print(f"  size {size:4} | chunks {len(engine.chunks):2} | min_score {min_score} | "
                  f"answers {len(results)} | clubs found {len(found)}/3 {found}")

    print("\n## Overlap vs number of chunks (size 500)")
    for overlap in (10, 15, 20):
        print(f"  overlap {overlap}% -> {len(chunk_text(text, 500, overlap))} chunks")

    print("\n## Threshold calibration (best chunk score per question)")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, size in enumerate(SIZES):
        engine = SemanticSearch(chunk_text(text, size, 15), model=model)
        inside = np.array([engine.scores(q).max() for q in IN_TEXT])
        outside = np.array([engine.scores(q).max() for q in NOT_IN_TEXT])
        correct = ((inside >= 0.6).sum() + (outside < 0.6).sum()) / (len(inside) + len(outside))
        print(f"  size {size:4} | in-text min {inside.min():.3f} mean {inside.mean():.3f} | "
              f"not-in-text max {outside.max():.3f} mean {outside.mean():.3f} | correct at 0.6: {correct:.0%}")
        ax.scatter(inside, [k + 0.12] * len(inside), color="tab:green", s=45,
                   label="answer is in the text" if k == 0 else None)
        ax.scatter(outside, [k - 0.12] * len(outside), color="tab:red", marker="x", s=45,
                   label="answer is NOT in the text" if k == 0 else None)
    ax.axvline(0.6, color="black", ls="--", label="MIN_SCORE = 0.6")
    ax.set_yticks(range(len(SIZES)), [f"{s} chars" for s in SIZES])
    ax.set(xlabel="best cosine similarity among all chunks", ylabel="chunk size",
           title=f"Threshold calibration: in-text vs out-of-text questions ({DEFAULT_MODEL.split('/')[-1]})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    plt.tight_layout()
    out = ROOT / "docs" / "images" / "threshold_calibration.png"
    fig.savefig(out, dpi=110)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
