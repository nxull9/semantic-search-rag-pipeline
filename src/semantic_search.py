"""Semantic search pipeline for Retrieval-Augmented Generation (RAG).

Pipeline:
    1. Load a plain-text file (at least 10 paragraphs).
    2. Split it into fixed-size chunks (characters or tokens) with a 10-20% sliding-window overlap.
    3. Embed every chunk with a Sentence-Transformers model.
    4. Embed a user query with the same model.
    5. Score every chunk against the query with cosine similarity.
    6. Return the top-k chunks, or report that the answer is not in the document
       when no chunk reaches the minimum similarity score.

Run from the command line:
    python src/semantic_search.py --file data/sample_football_clubs.txt --query "Which club signed Karim Benzema?"
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP_PERCENT = 15
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.6
MIN_PARAGRAPHS = 10


# --------------------------------------------------------------------------- loading

def load_text(path: str | Path) -> tuple[str, list[str]]:
    """Read a .txt file and return (clean_text, paragraphs).

    Paragraphs are separated by blank lines. If the file has none, each line is a paragraph.
    The clean text joins all paragraphs with single spaces.
    """
    raw = Path(path).read_text(encoding="utf-8")
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) < 2:
        paragraphs = [" ".join(line.split()) for line in raw.splitlines() if line.strip()]
    return " ".join(paragraphs), paragraphs


# --------------------------------------------------------------------------- chunking

def overlap_size(chunk_size: int, overlap_percent: float) -> int:
    """Number of characters/tokens shared by neighbouring chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 10 <= overlap_percent <= 20:
        raise ValueError("overlap_percent must be between 10 and 20")
    return round(chunk_size * overlap_percent / 100)


def sliding_windows(length: int, size: int, step: int) -> list[tuple[int, int]]:
    """(start, end) positions of windows of `size` that move forward by `step`."""
    if step <= 0:
        raise ValueError("step must be positive")
    windows, start = [], 0
    while True:
        windows.append((start, min(start + size, length)))
        if start + size >= length:
            return windows
        start += step


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap_percent: float = DEFAULT_OVERLAP_PERCENT,
               unit: str = "characters", tokenizer=None) -> list[str]:
    """Split text into fixed-size chunks with a sliding-window overlap.

    unit="characters" cuts every `chunk_size` characters.
    unit="tokens" cuts every `chunk_size` tokens of `tokenizer` and returns the matching text.
    """
    step = chunk_size - overlap_size(chunk_size, overlap_percent)
    if unit == "characters":
        return [text[s:e] for s, e in sliding_windows(len(text), chunk_size, step)]
    if unit == "tokens":
        if tokenizer is None:
            raise ValueError("unit='tokens' needs a tokenizer")
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        return [text[offsets[s][0]:offsets[e - 1][1]] for s, e in sliding_windows(len(offsets), chunk_size, step)]
    raise ValueError("unit must be 'characters' or 'tokens'")


# --------------------------------------------------------------------------- similarity

def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector and every row of a matrix: (A . B) / (|A| x |B|)."""
    dot_products = matrix @ query_vector
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    return dot_products / norms


# --------------------------------------------------------------------------- search

@dataclass
class SearchResult:
    rank: int
    chunk_id: int
    score: float
    text: str


class SemanticSearch:
    """Embeds chunks once, then answers queries with cosine-similarity ranking."""

    def __init__(self, chunks: list[str], model_name: str = DEFAULT_MODEL, model=None):
        if model is None:
            from sentence_transformers import SentenceTransformer  # imported lazily: heavy dependency
            model = SentenceTransformer(model_name)
        self.model = model
        self.chunks = chunks
        self.chunk_vectors = np.asarray(model.encode(chunks))

    def scores(self, question: str) -> np.ndarray:
        """Cosine similarity of the question with every chunk (same embedding model)."""
        return cosine_similarity(np.asarray(self.model.encode(question)), self.chunk_vectors)

    def search(self, question: str, top_k: int = DEFAULT_TOP_K,
               min_score: float = DEFAULT_MIN_SCORE) -> list[SearchResult]:
        """Top-k chunks scoring at least `min_score`; empty list if nothing is relevant."""
        scores = self.scores(question)
        ranking = np.argsort(scores)[::-1][:top_k]
        confident = [i for i in ranking if scores[i] >= min_score]
        return [SearchResult(rank, int(i), float(scores[i]), self.chunks[i])
                for rank, i in enumerate(confident, start=1)]


def format_results(question: str, results: list[SearchResult], min_score: float) -> str:
    """Readable Question / Answers block."""
    lines = ["=" * 80, f"Question: {question}", "=" * 80]
    if not results:
        lines.append(f"❌ This is not in the document you provided (no chunk scored ≥ {min_score}).")
        return "\n".join(lines)
    lines.append(f"Answers ({len(results)} chunk(s) with similarity ≥ {min_score}):\n")
    for r in results:
        lines += [f"{r.rank}. Chunk {r.chunk_id} | Similarity score: {r.score:.4f}", f"   {r.text}\n"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search over a text file with chunking + embeddings.")
    parser.add_argument("--file", required=True, help="path to a .txt file (at least 10 paragraphs)")
    parser.add_argument("--query", help="question to ask; omit for interactive mode")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_PERCENT, help="overlap percent (10-20)")
    parser.add_argument("--unit", choices=["characters", "tokens"], default="characters")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    text, paragraphs = load_text(args.file)
    if len(paragraphs) < MIN_PARAGRAPHS:
        print(f"⚠️  {args.file} has {len(paragraphs)} paragraphs; at least {MIN_PARAGRAPHS} are expected.")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    tokenizer = model.tokenizer if args.unit == "tokens" else None
    chunks = chunk_text(text, args.chunk_size, args.overlap, args.unit, tokenizer)
    engine = SemanticSearch(chunks, model=model)

    overlap = overlap_size(args.chunk_size, args.overlap)
    print(f"Loaded {len(paragraphs)} paragraphs -> {len(chunks)} chunks "
          f"({args.chunk_size} {args.unit}, overlap {overlap} = {args.overlap:g}%), model {args.model}\n")

    questions = [args.query] if args.query else iter(lambda: input("\nYour question (Enter to quit): ").strip(), "")
    for question in questions:
        print(format_results(question, engine.search(question, args.top_k, args.min_score), args.min_score))


if __name__ == "__main__":
    main()
