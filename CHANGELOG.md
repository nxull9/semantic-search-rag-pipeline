# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-10-05

Multi-document routing, submitted for Assignment 2 of the Generative AI Solutions Development training program.

### Added
- Two new documents on different subjects: the Solar System and coffee (12 paragraphs each).
- A routing summary for each document in `data/summaries/`.
- `src/multi_document.py`: one vector store per document with its own chunk size, overlap and minimum score; stores can be saved to and loaded from disk.
- Summary-based router: questions are compared with the document summaries to pick the document before searching its chunks, with a fallback to the second-best document.
- `config/collections.json` with the tuned settings for each collection and the router.
- `scripts/tune_collections.py` and `data/eval_questions.json` to tune and evaluate each collection and the router.
- Colab notebook `notebooks/multi_document_routing_colab.ipynb` with routing and chunk-score charts and a routing check.
- Tests for vector stores, routing, fallback and the project configuration.
- `docs/MULTI_DOCUMENT_ROUTING.md` describing the design, tuning method and results.

### Changed
- Rewrote the README for professional use: plain-text formatting, a guide to using your own documents, design decisions and a roadmap.
- Command-line messages use plain text instead of emoji.
- Shared the fake embedding model used by the tests in `tests/fakes.py`.

## [1.0.0] - 2026-10-05

First release, submitted for Assignment 1 of the Generative AI Solutions Development training program.

### Added
- Text loading with paragraph detection and a 10-paragraph check.
- Fixed-size window chunking by characters or tokens with a 10–20% sliding overlap.
- Chunk embeddings with Sentence-Transformers (`BAAI/bge-small-en-v1.5`).
- Explicit cosine similarity, top-k ranking and a Question / Answers output.
- Confidence threshold (`MIN_SCORE`) that reports when the answer is not in the document.
- Command-line interface with single-question and interactive modes.
- Google Colab notebook with a similarity bar chart and a similarity matrix.
- Unit tests and a GitHub Actions workflow.
- Technical documentation and experiments on chunk size, overlap and threshold calibration.
