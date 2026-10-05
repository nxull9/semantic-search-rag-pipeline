# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-10-05

Final submission for Assignment 1 of the Generative AI Solutions Development training program. The repository now contains only the submitted notebook and documentation of its results.

### Changed
- Replaced the notebook with the final Assignment 1 submission: 800-character chunks, 15% overlap, top 3 results and `MIN_SCORE = 0.20`.
- Rewrote the README and technical documentation to describe only the submitted notebook and the results it produces on the sample document.

### Removed
- The Python package and command-line interface (`src/`).
- Unit tests and the GitHub Actions workflow.
- The experiment script, the experiment report and its charts.

## [1.0.0] - 2026-10-05

Initial version of the pipeline.

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
