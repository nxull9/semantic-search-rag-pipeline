# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

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
