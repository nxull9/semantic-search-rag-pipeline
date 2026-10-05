"""Generate notebooks/multi_document_routing_colab.ipynb.

    python scripts/build_multi_document_notebook.py

The notebook is self-contained so it runs in Google Colab without the rest of the repository.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "multi_document_routing_colab.ipynb"

CELLS = [
    ("markdown", r"""# Multi-Document Semantic Search with Summary Routing

Extends the single-document pipeline to **three documents on different subjects**. Each document gets its own vector database with settings tuned for its content, plus a short summary. A question is answered in two stages:

1. **Routing**: compare the question with the three summaries and pick the document whose subject matches.
2. **Retrieval**: compare the question with the chunks of that document only, and return the top 3.

```
question ──► compare with 3 summaries ──► best document ──► cosine search in its vector DB ──► top 3 chunks
```

**Files needed** (upload them to the Colab Files panel, or set the folders in Settings):
`sample_football_clubs.txt`, `solar_system.txt`, `coffee.txt` and their summaries `*_summary.txt`."""),

    ("markdown", r"""## Settings
Each collection has its own chunk size, overlap and minimum score. These values were chosen by testing each document with questions it can and cannot answer (see `scripts/tune_collections.py` in the repository)."""),
    ("code", r"""DOCS_DIR = "."                 # folder with the documents      (use "data" when running from the repo)
SUMMARIES_DIR = "."            # folder with the summary files  (use "data/summaries" when running from the repo)
VECTOR_DB_DIR = "vector_db"    # where the three vector databases are saved

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # same model for documents, summaries and questions
TOP_K = 3                                    # number of chunks to return

COLLECTIONS = [
    {"name": "football_clubs", "document": "sample_football_clubs.txt", "summary": "sample_football_clubs_summary.txt",
     "chunk_size": 300, "overlap_percent": 10, "min_score": 0.63,
     "why": "short facts (founding year, stadium, nickname): small chunks give the sharpest matches"},
    {"name": "solar_system", "document": "solar_system.txt", "summary": "solar_system_summary.txt",
     "chunk_size": 300, "overlap_percent": 10, "min_score": 0.66,
     "why": "one numeric fact per sentence: small chunks keep each fact isolated"},
    {"name": "coffee", "document": "coffee.txt", "summary": "coffee_summary.txt",
     "chunk_size": 800, "overlap_percent": 10, "min_score": 0.67,
     "why": "explanations span several sentences: larger chunks keep each explanation together"},
]

ROUTE_MIN_SCORE = 0.40   # below this, no summary is related to the question at all
MAX_ROUTES = 2           # if the best document has no confident chunk, also try the second best"""),

    ("markdown", "## Install"),
    ("code", r"""!pip install -q sentence-transformers"""),

    ("markdown", r"""## Step 1: Load the three documents and their summaries"""),
    ("code", r"""import os, re, json
import numpy as np
import pandas as pd

def path_of(folder, name):
    return os.path.join(folder, name)

needed = [path_of(DOCS_DIR, c["document"]) for c in COLLECTIONS] + [path_of(SUMMARIES_DIR, c["summary"]) for c in COLLECTIONS]
missing = [p for p in needed if not os.path.exists(p)]
if missing:
    from google.colab import files
    print("Missing files, please upload them:", [os.path.basename(p) for p in missing])
    files.upload()

def load_text(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) < 2:
        paragraphs = [" ".join(line.split()) for line in raw.splitlines() if line.strip()]
    return " ".join(paragraphs), paragraphs

for c in COLLECTIONS:
    c["text"], c["paragraphs"] = load_text(path_of(DOCS_DIR, c["document"]))
    with open(path_of(SUMMARIES_DIR, c["summary"]), encoding="utf-8") as f:
        c["summary_text"] = f.read().strip()

pd.DataFrame([{
    "collection": c["name"],
    "paragraphs": len(c["paragraphs"]),
    "characters": len(c["text"]),
    "10+ paragraphs": "yes" if len(c["paragraphs"]) >= 10 else "NO",
    "summary": c["summary_text"][:90] + "...",
} for c in COLLECTIONS])"""),

    ("markdown", r"""## Step 2: Chunk each document with its own settings
Same fixed-size sliding window as before, but every collection uses its own `chunk_size` and `overlap_percent`."""),
    ("code", r"""def sliding_windows(length, size, step):
    windows, start = [], 0
    while True:
        windows.append((start, min(start + size, length)))
        if start + size >= length:
            return windows
        start += step

def chunk_text(text, size, overlap_percent):
    assert 10 <= overlap_percent <= 20, "overlap_percent must be between 10 and 20"
    step = size - round(size * overlap_percent / 100)
    return [text[s:e] for s, e in sliding_windows(len(text), size, step)]

for c in COLLECTIONS:
    c["chunks"] = chunk_text(c["text"], c["chunk_size"], c["overlap_percent"])

pd.DataFrame([{
    "collection": c["name"],
    "chunk_size": c["chunk_size"],
    "overlap": f'{c["overlap_percent"]}% = {round(c["chunk_size"] * c["overlap_percent"] / 100)} chars',
    "chunks": len(c["chunks"]),
    "why these settings": c["why"],
} for c in COLLECTIONS])"""),

    ("markdown", r"""## Step 3: Build one vector database per document
Each database stores the chunks, their vectors and the collection's settings, and is saved to disk under `vector_db/<collection>/` so it can be reloaded without re-embedding."""),
    ("code", r"""from sentence_transformers import SentenceTransformer

model = SentenceTransformer(EMBEDDING_MODEL)
vector_dbs = {}

for c in COLLECTIONS:
    vectors = model.encode(c["chunks"])
    vector_dbs[c["name"]] = {"chunks": c["chunks"], "vectors": vectors, "min_score": c["min_score"],
                             "summary": c["summary_text"]}

    folder = os.path.join(VECTOR_DB_DIR, c["name"])
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "vectors.npy"), vectors)
    with open(os.path.join(folder, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(c["chunks"], f, ensure_ascii=False, indent=1)
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump({k: c[k] for k in ("name", "document", "summary", "chunk_size", "overlap_percent", "min_score")}
                  | {"summary_text": c["summary_text"]}, f, ensure_ascii=False, indent=2)

pd.DataFrame([{"vector database": f"{VECTOR_DB_DIR}/{name}", "chunks": db["vectors"].shape[0],
               "vector size": db["vectors"].shape[1], "min_score": db["min_score"]}
              for name, db in vector_dbs.items()])"""),

    ("markdown", r"""## Cosine similarity
cosine(A, B) = (A · B) / (‖A‖ × ‖B‖)"""),
    ("code", r"""def cosine_similarity(query_vector, matrix):
    dot_products = matrix @ query_vector
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    return dot_products / norms"""),

    ("markdown", r"""## New step: the summary router
The three summaries are embedded with the same model. To route a question, it is compared with each summary, and the most similar one decides which vector database to search.

The table below compares the summaries with each other. Low values off the diagonal mean the subjects are clearly different, which makes routing reliable."""),
    ("code", r"""names = [c["name"] for c in COLLECTIONS]
summary_vectors = model.encode([vector_dbs[n]["summary"] for n in names])

def route(query_vector):
    # Collections ranked by how similar their summary is to the question.
    scores = cosine_similarity(query_vector, summary_vectors)
    return sorted(zip(names, scores.astype(float)), key=lambda item: item[1], reverse=True)

summary_matrix = np.array([cosine_similarity(v, summary_vectors) for v in summary_vectors])
pd.DataFrame(summary_matrix.round(3), index=names, columns=names)"""),

    ("markdown", r"""## Steps 4–7: Ask a question → route by summary → search that database → top 3
If the selected document has no chunk above its `min_score`, the second-best document is tried before the question is reported as not covered."""),
    ("code", r"""def ask(question, top_k=TOP_K, verbose=True):
    query_vector = model.encode(question)                       # same embedding model
    ranked = route(query_vector)                                # routing by summary

    if verbose:
        print("=" * 80)
        print(f"Question: {question}")
        print("=" * 80)
        print("Routing (similarity to each document summary):")
        for name, score in ranked:
            print(f"  {name:<16} {score:.4f}")

    for name, route_score in ranked[:MAX_ROUTES]:
        if route_score < ROUTE_MIN_SCORE:
            break
        db = vector_dbs[name]
        scores = cosine_similarity(query_vector, db["vectors"])   # cosine similarity inside this database
        ranking = np.argsort(scores)[::-1][:top_k]                 # rank, highest first
        confident = [i for i in ranking if scores[i] >= db["min_score"]]
        if confident:
            if verbose:
                print(f"\nSelected document: {name}  (min_score {db['min_score']})")
                print(f"Answers (top {len(confident)} chunk(s)):\n")
                for rank, i in enumerate(confident, start=1):
                    print(f"{rank}. Chunk {i} | Similarity score: {scores[i]:.4f}")
                    print(f"   {db['chunks'][i]}\n")
            return name, [(int(i), float(scores[i])) for i in confident]
        if verbose:
            print(f"\n{name}: no chunk above its min_score ({scores[ranking[0]]:.4f} < {db['min_score']}), trying the next document")

    if verbose:
        print("\nNot found: this is not in the documents provided.")
    return None, []

question = input("Type your question: ")
selected, results = ask(question)"""),

    ("markdown", r"""## Graph: routing scores and chunk scores
Left: how similar the question is to each summary (the routing decision). Right: how similar it is to every chunk of the selected document, with the returned chunks in orange."""),
    ("code", r"""import matplotlib.pyplot as plt

def plot_routing(question):
    query_vector = model.encode(question)
    ranked = dict(route(query_vector))
    selected, results = ask(question, verbose=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2.2]})

    colors = ["tab:orange" if n == selected else "tab:blue" for n in names]
    ax1.bar(names, [ranked[n] for n in names], color=colors)
    ax1.axhline(ROUTE_MIN_SCORE, color="grey", ls="--", label=f"route min {ROUTE_MIN_SCORE}")
    ax1.set(title="Step 1: similarity to each summary", ylabel="cosine similarity")
    ax1.legend(loc="lower right")

    if selected:
        db = vector_dbs[selected]
        scores = cosine_similarity(query_vector, db["vectors"])
        returned = {i for i, _ in results}
        ax2.bar(range(len(scores)), scores, color=["tab:orange" if i in returned else "tab:blue" for i in range(len(scores))])
        ax2.axhline(db["min_score"], color="grey", ls="--", label=f"min_score {db['min_score']}")
        ax2.set_ylim(scores.min() - 0.05, scores.max() + 0.05)
        ax2.set_xticks(range(0, len(scores), max(1, len(scores) // 20)))
        ax2.set(title=f"Step 2: similarity to every chunk of '{selected}'", xlabel="chunk number")
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.5, 0.5, "Not found in any document", ha="center", va="center", fontsize=14)
        ax2.axis("off")

    fig.suptitle(f'"{question[:80]}"')
    plt.tight_layout()
    plt.show()

plot_routing(question)"""),

    ("markdown", r"""## Routing check
A few test questions for each subject, plus questions that none of the documents answer."""),
    ("code", r"""TEST_QUESTIONS = [
    ("football_clubs", "Which club signed Karim Benzema?"),
    ("football_clubs", "Where does Liverpool play its home matches?"),
    ("solar_system", "Which planet is the hottest?"),
    ("solar_system", "What is the largest moon in the Solar System?"),
    ("coffee", "Which country produces the most coffee?"),
    ("coffee", "What is Arabic coffee served with?"),
    (None, "What is the capital of France?"),
    (None, "How do I learn Python?"),
]

rows = []
for expected, q in TEST_QUESTIONS:
    selected, results = ask(q, verbose=False)
    rows.append({"question": q, "expected": expected or "not found", "answered from": selected or "not found",
                 "best score": round(results[0][1], 4) if results else None,
                 "correct": (selected or None) == expected})

check = pd.DataFrame(rows)
print(f"Correct: {check.correct.sum()}/{len(check)}")
check"""),

    ("markdown", r"""## Ask more questions
Press Enter on an empty line to stop."""),
    ("code", r"""while True:
    question = input("\nYour question (Enter to stop): ").strip()
    if not question:
        break
    ask(question)
    plot_routing(question)"""),
]


def main() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [{"cell_type": kind, "metadata": {}, "source": source.splitlines(True),
                   **({"execution_count": None, "outputs": []} if kind == "code" else {})}
                  for kind, source in CELLS],
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
