"""Multi-document semantic search with summary-based routing.

Each document gets its own vector store, built with settings tuned for that document.
A question is answered in two stages:

    1. Routing: the question is compared with a short summary of every document, and the
       best-matching document is selected.
    2. Retrieval: the question is compared with the chunks of that document only, and the
       top-k chunks above the document's minimum score are returned. If none qualify, the
       next-best document is tried before the question is reported as not covered.

Run from the command line:
    python src/multi_document.py --query "Which planet is the hottest?"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from semantic_search import SearchResult, chunk_text, cosine_similarity, load_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "collections.json"


# --------------------------------------------------------------------------- configuration

@dataclass
class CollectionConfig:
    name: str
    document: str
    summary: str
    chunk_size: int
    overlap_percent: float
    min_score: float
    notes: str = ""


@dataclass
class RouterConfig:
    route_min_score: float = 0.40
    max_routes: int = 2


def load_config(path: str | Path = DEFAULT_CONFIG) -> tuple[str, RouterConfig, list[CollectionConfig]]:
    """Read the embedding model name, router settings and collection settings from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    collections = [CollectionConfig(**c) for c in data["collections"]]
    return data["embedding_model"], RouterConfig(**data.get("router", {})), collections


# --------------------------------------------------------------------------- vector store

@dataclass
class VectorStore:
    """A small vector database for one document: its chunks, their vectors and its settings."""

    config: CollectionConfig
    summary: str
    chunks: list[str]
    vectors: np.ndarray

    @classmethod
    def build(cls, config: CollectionConfig, model, base_dir: str | Path = ROOT) -> "VectorStore":
        """Load the document, chunk it with this collection's settings and embed every chunk."""
        base_dir = Path(base_dir)
        text, _ = load_text(base_dir / config.document)
        summary = (base_dir / config.summary).read_text(encoding="utf-8").strip()
        chunks = chunk_text(text, config.chunk_size, config.overlap_percent)
        return cls(config, summary, chunks, np.asarray(model.encode(chunks)))

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> tuple[list[SearchResult], float]:
        """Top-k chunks scoring at least this collection's min_score, plus the best score found."""
        scores = cosine_similarity(query_vector, self.vectors)
        ranking = np.argsort(scores)[::-1][:top_k]
        results = [SearchResult(rank, int(i), float(scores[i]), self.chunks[i])
                   for rank, i in enumerate((i for i in ranking if scores[i] >= self.config.min_score), start=1)]
        return results, float(scores[ranking[0]])

    def save(self, directory: str | Path) -> Path:
        """Persist the store as vectors.npy + chunks.json + config.json."""
        out = Path(directory) / self.config.name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self.vectors)
        (out / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "config.json").write_text(json.dumps({**asdict(self.config), "summary_text": self.summary},
                                                    ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a store previously written by save()."""
        directory = Path(directory)
        meta = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        summary = meta.pop("summary_text")
        chunks = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls(CollectionConfig(**meta), summary, chunks, np.load(directory / "vectors.npy"))


# --------------------------------------------------------------------------- router

@dataclass
class RoutedAnswer:
    question: str
    route_scores: dict[str, float]
    collection: str | None = None
    results: list[SearchResult] = field(default_factory=list)
    tried: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.results)


class DocumentRouter:
    """Routes a question to the right vector store using the document summaries."""

    def __init__(self, stores: list[VectorStore], model, router: RouterConfig | None = None):
        self.stores = {s.config.name: s for s in stores}
        self.model = model
        self.router = router or RouterConfig()
        self.summary_vectors = np.asarray(model.encode([s.summary for s in stores]))

    def route(self, question: str) -> list[tuple[str, float]]:
        """Collections ranked by how similar their summary is to the question."""
        return self._route(np.asarray(self.model.encode(question)))

    def _route(self, query_vector: np.ndarray) -> list[tuple[str, float]]:
        scores = cosine_similarity(query_vector, self.summary_vectors)
        return sorted(zip(self.stores, map(float, scores)), key=lambda item: item[1], reverse=True)

    def ask(self, question: str, top_k: int = 3) -> RoutedAnswer:
        """Route the question by summary, then search the selected collection(s)."""
        query_vector = np.asarray(self.model.encode(question))
        route = self._route(query_vector)
        answer = RoutedAnswer(question, dict(route))
        for name, score in route[: self.router.max_routes]:
            if score < self.router.route_min_score:
                break
            answer.tried.append(name)
            results, _ = self.stores[name].search(query_vector, top_k)
            if results:
                answer.collection, answer.results = name, results
                break
        return answer


def format_answer(answer: RoutedAnswer) -> str:
    lines = ["=" * 80, f"Question: {answer.question}", "=" * 80, "Routing (similarity to each document summary):"]
    for name, score in answer.route_scores.items():
        marker = "  <- selected" if name == answer.collection else ""
        lines.append(f"  {name:<18} {score:.4f}{marker}")
    lines.append("")
    if not answer.found:
        reason = ("no document summary is related to the question" if not answer.tried
                  else f"no chunk in {', '.join(answer.tried)} scored above its minimum")
        lines.append(f"Not found: this is not in the documents provided ({reason}).")
        return "\n".join(lines)
    lines.append(f"Answers from '{answer.collection}' ({len(answer.results)} chunk(s)):\n")
    for r in answer.results:
        lines += [f"{r.rank}. Chunk {r.chunk_id} | Similarity score: {r.score:.4f}", f"   {r.text}\n"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search several documents, routing each question by summary.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="collections JSON file")
    parser.add_argument("--query", help="question to ask; omit for interactive mode")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--save-dir", help="optional folder to save the vector stores to")
    args = parser.parse_args(argv)

    model_name, router_config, configs = load_config(args.config)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    base_dir = Path(args.config).resolve().parents[1]
    stores = [VectorStore.build(c, model, base_dir) for c in configs]
    for s in stores:
        print(f"{s.config.name:<18} {len(s.chunks):3} chunks  (size {s.config.chunk_size}, "
              f"overlap {s.config.overlap_percent:g}%, min_score {s.config.min_score})")
        if args.save_dir:
            s.save(args.save_dir)
    router = DocumentRouter(stores, model, router_config)

    questions = [args.query] if args.query else iter(lambda: input("\nYour question (Enter to quit): ").strip(), "")
    for question in questions:
        print(format_answer(router.ask(question, args.top_k)))


if __name__ == "__main__":
    main()
