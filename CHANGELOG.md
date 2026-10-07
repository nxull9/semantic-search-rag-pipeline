# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [4.0.0] - 2026-10-07

Final project of the Generative AI Solutions Development program at SDAIA Academy. Every project now includes its notebook as both `.ipynb` and `.py`.

### Added
- `final-project-rag-agent/`: the final project notebook. It builds on the Assignment 2 retrieval pipeline and adds Phase 3, where an LLM (`google/gemini-2.5-flash` through OpenRouter, temperature 0) answers from the top 3 chunks and lists the files and chunks it used. A ReAct research agent with two tools, `search_documents` and a safe `calculator`, answers questions that need several facts. Guardrails limit the agent to 6 steps and to the allowed tools, return tool errors to the model, and check that every cited chunk was retrieved.
- A test set of 4 RAG questions and 2 agent questions, with results collected in a table.
- README and technical documentation for the final project.
- A `.py` file next to each notebook (`semantic_search_colab.py`, `semantic_chunking_faiss_colab.py`, `rag_research_agent_colab.py`) with the same code as the notebook.

### Changed
- The main README lists three completed projects and compares all three.
- The Assignment 1 and 2 READMEs list the new `.py` files.

## [3.0.0] - 2026-10-06

Assignment 2 of the Generative AI Solutions Development program at SDAIA Academy. The repository now holds one folder per project.

### Added
- `assignment-2-semantic-faiss/`: the submitted Assignment 2 notebook. Three documents from different fields (football clubs, the Solar System, coffee) are split by semantic chunking, embedded with `BAAI/bge-small-en-v1.5`, and stored in a FAISS vector database saved to Google Drive. A summary of each document is stored in a second index for summary-first search. Results show the top 3 chunks with their scores and source file.
- Step-by-step verification cells (27 checks), token usage per file and per tokenizer, and a chart of chunk scores per file.
- README and technical documentation for Assignment 2, with the results produced by the notebook.

### Changed
- Moved Assignment 1 into `assignment-1-semantic-search/` without changing its notebook, data or documentation.
- The main README is now an overview of the program and its projects, with a link to SDAIA Academy.

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
