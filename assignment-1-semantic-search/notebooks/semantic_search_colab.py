# -*- coding: utf-8 -*-
"""Assignment 1: Semantic Search Pipeline

Python export of notebooks/semantic_search_colab.ipynb, with the same code cell by cell.
Run it in Google Colab: lines starting with "!" are Colab shell commands.
"""

FILE_PATH = "my_text.txt"                  # your text file in the Colab Files panel

CHUNK_UNIT = "characters"                  # "characters" or "tokens"
CHUNK_SIZE = 800                      # size of each chunk (in characters or tokens)
OVERLAP_PERCENT = 15                     # sliding window overlap, between 10 and 20 (%)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" # embedding model used for BOTH chunks and queries
TOP_K = 3                                  # number of results to show

!pip install -q sentence-transformers

import os, re
import numpy as np
import pandas as pd
from google.colab import files

if not os.path.exists(FILE_PATH):
    print(f"{FILE_PATH} not found, please upload it:")
    FILE_PATH = list(files.upload().keys())[0]

with open(FILE_PATH, encoding="utf-8") as f:
    raw_text = f.read()

paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
if len(paragraphs) < 2:                                   # no blank lines: one paragraph per line
    paragraphs = [" ".join(p.split()) for p in raw_text.splitlines() if p.strip()]

text = " ".join(paragraphs)                               # clean text: one space between words

print(f"File: {FILE_PATH}")
print(f"Paragraphs: {len(paragraphs)} | Characters: {len(text):,} | Words: {len(text.split()):,}")
print("✅ At least 10 paragraphs" if len(paragraphs) >= 10 else "⚠️ The assignment needs at least 10 paragraphs")
print("\nStart of the text:\n", text[:300], "...")

assert 10 <= OVERLAP_PERCENT <= 20, "OVERLAP_PERCENT must be between 10 and 20"
overlap = round(CHUNK_SIZE * OVERLAP_PERCENT / 100)
step = CHUNK_SIZE - overlap

def sliding_windows(length, size, step):
    """Start and end positions of each window, sliding by `step`."""
    windows, start = [], 0
    while True:
        windows.append((start, min(start + size, length)))
        if start + size >= length:
            break
        start += step
    return windows

def chunk_text(text, unit, size, step):
    if unit == "characters":
        return [text[s:e] for s, e in sliding_windows(len(text), size, step)]
    if unit == "tokens":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        # cut by token positions, then take the matching piece of the original text
        return [text[offsets[s][0]:offsets[e - 1][1]] for s, e in sliding_windows(len(offsets), size, step)]
    raise ValueError("CHUNK_UNIT must be 'characters' or 'tokens'")

chunks = chunk_text(text, CHUNK_UNIT, CHUNK_SIZE, step)

print(f"Chunk size: {CHUNK_SIZE} {CHUNK_UNIT} | Overlap: {overlap} {CHUNK_UNIT} ({OVERLAP_PERCENT}%) | Step: {step}")
print(f"Number of chunks: {len(chunks)}\n")

chunks_df = pd.DataFrame({"chunk_id": range(len(chunks)),
                          "characters": [len(c) for c in chunks],
                          "preview": [c[:80] + "..." for c in chunks]})
chunks_df

# Check the overlap: the end of chunk 0 is repeated at the start of chunk 1
if len(chunks) > 1:
    shared = next(k for k in range(min(len(chunks[0]), len(chunks[1])), 0, -1) if chunks[0].endswith(chunks[1][:k]))
    print(f"End of chunk 0:   ...{chunks[0][-shared:]}")
    print(f"Start of chunk 1: {chunks[1][:shared]}...")
    print(f"\nShared text: {shared} characters")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(EMBEDDING_MODEL)
chunk_vectors = model.encode(chunks)                      # one vector per chunk

print(f"Model: {EMBEDDING_MODEL}")
print(f"Vectors: {chunk_vectors.shape[0]} chunks x {chunk_vectors.shape[1]} numbers each")
print("First 8 numbers of chunk 0:", np.round(chunk_vectors[0][:8], 4))

def cosine_similarity(query_vector, matrix):
    """Cosine similarity between one query vector and every row of a matrix."""
    dot_products = matrix @ query_vector                              # A · B for every chunk
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)  # ‖A‖ × ‖B‖
    return dot_products / norms

MIN_SCORE = 0.20  # only chunks scoring at least this are returned

def semantic_search(question, top_k=TOP_K):
    query_vector = model.encode(question)                       # Step 5: same embedding model
    scores = cosine_similarity(query_vector, chunk_vectors)     # Step 6: cosine similarity
    ranking = np.argsort(scores)[::-1][:top_k]                  # Step 7: rank, highest first
    confident = [i for i in ranking if scores[i] >= MIN_SCORE]  # keep only confident matches

    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)

    if not confident:
        print("❌ This is not in the document you provided.")
        print(f"   (the closest chunk scored {scores[ranking[0]]:.4f}, below the minimum of {MIN_SCORE})")
        return []

    print(f"Answers ({len(confident)} chunk(s) with similarity ≥ {MIN_SCORE}):\n")
    for rank, i in enumerate(confident, start=1):
        print(f"{rank}. Chunk {i} | Similarity score: {scores[i]:.4f}")
        print(f"   {chunks[i]}\n")
    return [(int(i), float(scores[i])) for i in confident]

question = input("Type your question about the text: ")      # Step 4: user query in plain text
results = semantic_search(question)
