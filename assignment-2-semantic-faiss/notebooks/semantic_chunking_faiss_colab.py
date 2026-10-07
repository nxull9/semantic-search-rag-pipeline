# -*- coding: utf-8 -*-
"""Assignment 2: Semantic Chunking and a FAISS Vector Database

Python export of notebooks/semantic_chunking_faiss_colab.ipynb, with the same code cell by cell.
Run it in Google Colab: lines starting with "!" are Colab shell commands.
"""

FILES = ["football_clubs.txt", "solar_system.txt", "coffee.txt"]   # three text files from different fields

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # same model for chunks and questions

# Semantic chunking
SEMANTIC_PERCENTILE = 10     # inside a paragraph, cut where sentence-to-sentence similarity is in the lowest 10% for that file
MAX_CHUNK_CHARS = 1000       # dynamic safety limit: a chunk never grows beyond this many characters

TOP_K = 3                    # number of results to show

# Vector database storage
USE_GOOGLE_DRIVE = True      # keep the database in Google Drive so it survives Colab restarts
VECTOR_DB_DIR = "/content/drive/MyDrive/assignment2_vector_db" if USE_GOOGLE_DRIVE else "vector_db"
REBUILD = False              # True = re-run the ingestion phase even if a saved database exists

!pip install -q sentence-transformers faiss-cpu

import os, re, json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

model = SentenceTransformer(EMBEDDING_MODEL)

INDEX_PATH = os.path.join(VECTOR_DB_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(VECTOR_DB_DIR, "chunks.json")
CONFIG_PATH = os.path.join(VECTOR_DB_DIR, "config.json")
SETTINGS = {"files": FILES, "embedding_model": EMBEDDING_MODEL,
            "semantic_percentile": SEMANTIC_PERCENTILE, "max_chunk_chars": MAX_CHUNK_CHARS}

def saved_db_matches():
    # True if a vector database was saved earlier with exactly these settings.
    if not all(os.path.exists(p) for p in (INDEX_PATH, CHUNKS_PATH, CONFIG_PATH)):
        return False
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)["settings"] == SETTINGS

NEED_INGESTION = REBUILD or not saved_db_matches()
print(f"Model: {EMBEDDING_MODEL} | Vector DB folder: {VECTOR_DB_DIR}")
print("Ingestion phase will run: building the vector database" if NEED_INGESTION
      else "Saved vector database found with the same settings: the ingestion phase will be skipped")

def load_paragraphs(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) < 2:                                    # no blank lines: one paragraph per line
        paragraphs = [" ".join(p.split()) for p in raw.splitlines() if p.strip()]
    return paragraphs

if NEED_INGESTION:
    missing = [f for f in FILES if not os.path.exists(f)]
    if missing:
        from google.colab import files
        print("Please upload:", missing)
        files.upload()

    documents = {name: load_paragraphs(name) for name in FILES}
    display(pd.DataFrame([{"file": name, "paragraphs": len(p), "characters": sum(len(x) for x in p),
                           "words": sum(len(x.split()) for x in p)} for name, p in documents.items()]))
else:
    print("Skipped: using the saved vector database")

def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def semantic_chunks(paragraphs):
    sentences = [(p_id, s) for p_id, p in enumerate(paragraphs) for s in split_sentences(p)]
    texts = [s for _, s in sentences]
    vectors = model.encode(texts, normalize_embeddings=True)
    neighbour_sim = (vectors[:-1] * vectors[1:]).sum(axis=1)          # similarity of each sentence with the next
    threshold = float(np.percentile(neighbour_sim, SEMANTIC_PERCENTILE))

    chunks, current, current_para = [], [texts[0]], sentences[0][0]
    for k in range(1, len(texts)):
        new_paragraph = sentences[k][0] != sentences[k - 1][0]
        meaning_changes = neighbour_sim[k - 1] < threshold
        too_long = len(" ".join(current + [texts[k]])) > MAX_CHUNK_CHARS
        if new_paragraph or meaning_changes or too_long:
            chunks.append((current_para, " ".join(current)))
            current, current_para = [], sentences[k][0]
        current.append(texts[k])
    chunks.append((current_para, " ".join(current)))
    return chunks, threshold

if NEED_INGESTION:
    chunk_records, chunk_stats = [], []
    for name, paragraphs in documents.items():
        file_chunks, threshold = semantic_chunks(paragraphs)
        for p_id, text in file_chunks:
            chunk_records.append({"chunk_id": len(chunk_records), "file": name, "paragraph": p_id, "text": text})
        sizes = [len(t) for _, t in file_chunks]
        chunk_stats.append({"file": name, "paragraphs": len(paragraphs), "chunks": len(file_chunks),
                            "threshold": round(threshold, 3), "avg chars": round(np.mean(sizes)),
                            "min chars": min(sizes), "max chars": max(sizes)})
    display(pd.DataFrame(chunk_stats))
    pd.set_option("display.max_colwidth", 110)
    display(pd.DataFrame(chunk_records)[["chunk_id", "file", "paragraph", "text"]].head(10))
else:
    print("Skipped: using the saved vector database")

SUMMARIES = {"football_clubs.txt": "football_summary.txt",
             "solar_system.txt":   "solar_system_summary.txt",
             "coffee.txt":         "coffee_summary.txt"}

SUMMARY_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "summaries.faiss")
SUMMARY_LIST_PATH = os.path.join(VECTOR_DB_DIR, "summaries.json")
SUMMARY_SETTINGS = {"summaries": SUMMARIES, "embedding_model": EMBEDDING_MODEL}

def saved_summaries_match():
    if not (os.path.exists(SUMMARY_INDEX_PATH) and os.path.exists(SUMMARY_LIST_PATH)):
        return False
    with open(SUMMARY_LIST_PATH, encoding="utf-8") as f:
        return json.load(f)["settings"] == SUMMARY_SETTINGS

if REBUILD or not saved_summaries_match():
    missing = [s for s in SUMMARIES.values() if not os.path.exists(s)]
    if missing:
        from google.colab import files
        print("Please upload:", missing)
        files.upload()

    summary_records = []
    for document, summary_file in SUMMARIES.items():
        with open(summary_file, encoding="utf-8") as f:
            summary_records.append({"file": document, "summary_file": summary_file, "summary": f.read().strip()})

    summary_vectors = model.encode([r["summary"] for r in summary_records]).astype("float32")
    faiss.normalize_L2(summary_vectors)                         # unit length: inner product = cosine
    summary_index = faiss.IndexFlatIP(summary_vectors.shape[1])
    summary_index.add(summary_vectors)

    faiss.write_index(summary_index, SUMMARY_INDEX_PATH)
    with open(SUMMARY_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump({"settings": SUMMARY_SETTINGS, "summaries": summary_records}, f, ensure_ascii=False, indent=2)
    print(f"Stored {summary_index.ntotal} summaries in {SUMMARY_INDEX_PATH}")
else:
    print("Saved summaries found: loading them from the vector database")

summary_index = faiss.read_index(SUMMARY_INDEX_PATH)
with open(SUMMARY_LIST_PATH, encoding="utf-8") as f:
    summary_records = json.load(f)["summaries"]

pd.DataFrame([{"document": r["file"], "summary": r["summary"][:100] + "..."} for r in summary_records])

if NEED_INGESTION:
    chunk_vectors = model.encode([c["text"] for c in chunk_records]).astype("float32")
    faiss.normalize_L2(chunk_vectors)                               # unit length: inner product = cosine

    index = faiss.IndexFlatIP(chunk_vectors.shape[1])               # exact search over every stored vector
    index.add(chunk_vectors)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=1)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"settings": SETTINGS, "chunks": len(chunk_records), "dimension": int(chunk_vectors.shape[1])}, f, indent=2)

    print(f"Stored {index.ntotal} vectors of {chunk_vectors.shape[1]} numbers in {VECTOR_DB_DIR}")
    print("Saved files:", sorted(os.listdir(VECTOR_DB_DIR)))
else:
    print("Skipped: using the saved vector database")

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunk_records = json.load(f)
stored_vectors = index.reconstruct_n(0, index.ntotal)             # the stored chunk vectors

print(f"Loaded vector database: {index.ntotal} chunks, {index.d} numbers per vector")
pd.DataFrame(chunk_records).groupby("file").size().rename("chunks").to_frame()

def cosine_similarity(query_vector, matrix):
    dot_products = matrix @ query_vector
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    return dot_products / norms

def summary_first_search(question, top_k=TOP_K):
    query_vector = model.encode([question]).astype("float32")      # same embedding model
    faiss.normalize_L2(query_vector)

    # Step A: compare the question with every document summary to choose the file
    route_scores, route_ids = summary_index.search(query_vector, summary_index.ntotal)
    ranked_files = [(summary_records[i]["file"], float(s)) for i, s in zip(route_ids[0], route_scores[0])]
    selected_file = ranked_files[0][0]

    # Step B: cosine similarity with the chunks, keeping only the chunks of the selected file
    scores, ids = index.search(query_vector, index.ntotal)
    in_file = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if chunk_records[i]["file"] == selected_file]
    return ranked_files, selected_file, in_file[:top_k]

def print_summary_first(question):
    ranked_files, selected_file, results = summary_first_search(question)
    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    print("Step A: similarity to each document summary")
    for name, score in ranked_files:
        print(f"  {name:<20} {score:.4f}" + ("   <- selected" if name == selected_file else ""))
    print(f"\nStep B: answers from {selected_file} (top {len(results)} chunks):\n")
    for rank, (i, score) in enumerate(results, start=1):
        c = chunk_records[i]
        print(f"{rank}. [{c['file']}] Chunk {c['chunk_id']} | Similarity score: {score:.4f}")
        print(f"   {c['text']}\n")

summary_question = input("Type your question (summaries are read first): ")
print_summary_first(summary_question)

def search(question, top_k=TOP_K):
    query_vector = model.encode([question]).astype("float32")      # Step 5: same embedding model
    faiss.normalize_L2(query_vector)
    scores, ids = index.search(query_vector, top_k)                 # Step 6: cosine similarity with every stored chunk
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0])]  # Step 7: ranked, highest first

def print_answers(question, results):
    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    print(f"Answers (top {len(results)} chunks):\n")
    for rank, (i, score) in enumerate(results, start=1):
        c = chunk_records[i]
        print(f"{rank}. [{c['file']}] Chunk {c['chunk_id']} | Similarity score: {score:.4f}")
        print(f"   {c['text']}\n")

question = input("Type your question: ")                            # Step 4: user query in plain text
results = search(question)
print_answers(question, results)

query_vector = model.encode(question)
direct_scores = cosine_similarity(query_vector, stored_vectors)

pd.DataFrame([{"rank": r, "file": chunk_records[i]["file"], "chunk": i,
               "FAISS score": round(s, 4), "cosine formula": round(float(direct_scores[i]), 4)}
              for r, (i, s) in enumerate(results, start=1)])

import matplotlib.pyplot as plt

def plot_file_scores(question):
    scores = cosine_similarity(model.encode(question), stored_vectors)
    files_order = list(dict.fromkeys(c["file"] for c in chunk_records))
    fig, axes = plt.subplots(1, len(files_order), figsize=(15, 4), sharey=True)
    top = {i for i, _ in search(question)}
    for ax, name in zip(axes, files_order):
        ids = [c["chunk_id"] for c in chunk_records if c["file"] == name]
        ax.bar(range(len(ids)), scores[ids], color=["tab:orange" if i in top else "tab:blue" for i in ids])
        ax.set(title=f"{name}\nbest {scores[ids].max():.3f}", xlabel="chunk in file")
    axes[0].set_ylabel("cosine similarity")
    axes[0].set_ylim(scores.min() - 0.05, scores.max() + 0.05)
    fig.suptitle(f'"{question[:80]}"  (orange = top {TOP_K})')
    plt.tight_layout()
    plt.show()

plot_file_scores(question)

while True:
    question = input("\nYour question (Enter to stop): ").strip()
    if not question:
        break
    print_answers(question, search(question))

TEST_QUESTION = "Which club signed Karim Benzema?"   # question used by the retrieval checks

checks = []
current_step = ""

def check(name, passed, detail=""):
    """Record one test and print a check mark (or a cross) with its details."""
    checks.append({"step": current_step, "check": name, "result": "✓ PASS" if passed else "✗ FAIL", "detail": detail})
    print(f"{'✓' if passed else '✗'} {name}" + (f"   [{detail}]" if detail else ""))

def first_sentence(text):
    return split_sentences(text)[0]

def last_sentence(text):
    return split_sentences(text)[-1]

print("Checklist ready")

current_step = "Step 1: load files"
print(f"===== {current_step} =====\n")

check("three files configured", len(FILES) == 3, ", ".join(FILES))

files_here = [f for f in FILES if os.path.exists(f)]
if len(files_here) == len(FILES):
    docs_check = {f: load_paragraphs(f) for f in FILES}
    for f, paras in docs_check.items():
        words = sum(len(p.split()) for p in paras)
        check(f"{f} has at least 10 paragraphs", len(paras) >= 10, f"{len(paras)} paragraphs, {words} words")
    check("the three files are different documents", len({" ".join(p) for p in docs_check.values()}) == 3)
else:
    docs_check = None
    print("(text files not uploaded in this session: file checks skipped, the saved database is used)")

check("every file is in the vector database", {c["file"] for c in chunk_records} == set(FILES),
      ", ".join(sorted({c["file"] for c in chunk_records})))

current_step = "Step 2: semantic chunking"
print(f"===== {current_step} =====\n")

# Chunks per file
chunk_table = pd.DataFrame([{"file": c["file"], "chars": len(c["text"])} for c in chunk_records])
display(chunk_table.groupby("file")["chars"].agg(chunks="count", avg_chars="mean", min_chars="min", max_chars="max")
        .round(0).astype(int))

check("no empty chunks", all(c["text"].strip() for c in chunk_records))
longest = max(len(c["text"]) for c in chunk_records)
check(f"no chunk longer than MAX_CHUNK_CHARS ({MAX_CHUNK_CHARS})", longest <= MAX_CHUNK_CHARS, f"longest {longest} chars")
check("every chunk ends at the end of a sentence", all(c["text"].rstrip()[-1] in ".!?\"')" for c in chunk_records))

if docs_check:
    rebuilt_ok, total_sentences, sentence_copies = True, 0, 0
    for f, paras in docs_check.items():
        for p_id, para in enumerate(paras):
            parts = [c["text"] for c in chunk_records if c["file"] == f and c["paragraph"] == p_id]
            rebuilt_ok &= " ".join(parts) == para
        doc_sentences = [s for para in paras for s in split_sentences(para)]
        chunk_sentences = [s for c in chunk_records if c["file"] == f for s in split_sentences(c["text"])]
        total_sentences += len(doc_sentences)
        sentence_copies += len(chunk_sentences)
    check("chunks rebuild every paragraph exactly (no text lost)", rebuilt_ok)
    check("no overlap: every sentence is in exactly one chunk", total_sentences == sentence_copies,
          f"{total_sentences} sentences in the files, {sentence_copies} in the chunks")

boundary_rows = []
for f in FILES:
    file_chunks = [c for c in chunk_records if c["file"] == f]
    sentences = [s for c in file_chunks for s in split_sentences(c["text"])]
    vectors = model.encode(sentences, normalize_embeddings=True)
    neighbour_sim = (vectors[:-1] * vectors[1:]).sum(axis=1)
    threshold = float(np.percentile(neighbour_sim, SEMANTIC_PERCENTILE))     # same rule as Step 2
    position = 0
    for a, b in zip(file_chunks, file_chunks[1:]):
        position += len(split_sentences(a["text"]))
        sim = float(neighbour_sim[position - 1])                              # last sentence of a vs first of b
        if b["paragraph"] != a["paragraph"]:
            reason = "new paragraph"
        elif len(a["text"]) + 1 + len(first_sentence(b["text"])) > MAX_CHUNK_CHARS:
            reason = "size limit"
        else:
            reason = "meaning changes"
        boundary_rows.append({"file": f, "between chunks": f"{a['chunk_id']} | {b['chunk_id']}", "reason": reason,
                              "similarity": round(sim, 4), "file threshold": round(threshold, 4),
                              "below threshold": sim < threshold,                # compared before rounding
                              "end of first chunk": "..." + last_sentence(a["text"])[-60:],
                              "start of next chunk": first_sentence(b["text"])[:60] + "..."})

boundaries = pd.DataFrame(boundary_rows)
meaning_cuts = boundaries[boundaries.reason == "meaning changes"]
check("every cut inside a paragraph is where the meaning changes (similarity below the file's threshold)",
      bool(meaning_cuts["below threshold"].all()),
      f"{len(meaning_cuts)} meaning cuts, {(boundaries.reason == 'new paragraph').sum()} paragraph cuts, "
      f"{(boundaries.reason == 'size limit').sum()} size cuts")

print("\nCuts made because the meaning changed inside a paragraph:")
pd.set_option("display.max_colwidth", 70)
display(meaning_cuts.reset_index(drop=True))

print("\nExample: the first three boundaries in each file")
display(boundaries.groupby("file").head(3).reset_index(drop=True))

current_step = "Step 3: embeddings + FAISS"
print(f"===== {current_step} =====\n")

check("one vector per chunk", index.ntotal == len(chunk_records), f"{index.ntotal} vectors, {len(chunk_records)} chunks")
check("vector size matches the embedding model", index.d == model.get_sentence_embedding_dimension(),
      f"{index.d} numbers per vector")
norms = np.linalg.norm(stored_vectors, axis=1)
check("all vectors have length 1, so inner product = cosine", bool(np.allclose(norms, 1, atol=1e-4)),
      f"lengths between {norms.min():.4f} and {norms.max():.4f}")
check("FAISS uses inner-product search (exact, over every vector)", index.metric_type == faiss.METRIC_INNER_PRODUCT,
      type(index).__name__)
fresh = model.encode([chunk_records[0]["text"]], normalize_embeddings=True)[0]
check("stored vector matches a fresh embedding of the same chunk", float(fresh @ stored_vectors[0]) > 0.9999,
      f"similarity {float(fresh @ stored_vectors[0]):.6f}")
check("summary index holds one vector per document", summary_index.ntotal == len(FILES), f"{summary_index.ntotal} summaries")

print("\nFiles saved in the vector database folder:")
display(pd.DataFrame([{"file": f, "size (KB)": round(os.path.getsize(os.path.join(VECTOR_DB_DIR, f)) / 1024, 1)}
                      for f in sorted(os.listdir(VECTOR_DB_DIR))]))
check("vector database saved to disk", all(os.path.exists(p) for p in (INDEX_PATH, CHUNKS_PATH, CONFIG_PATH)), VECTOR_DB_DIR)

current_step = "Tokens"
print(f"===== {current_step} =====\n")

tokenizer = model.tokenizer
limit = model.max_seq_length

def count_tokens(text, tok=tokenizer):
    return len(tok(text)["input_ids"])                    # includes the [CLS] and [SEP] markers

for c in chunk_records:
    c["tokens"] = count_tokens(c["text"])

print(f"Model: {EMBEDDING_MODEL} | limit: {limit} tokens per text\n")
token_table = pd.DataFrame([{"file": c["file"], "tokens": c["tokens"], "words": len(c["text"].split())} for c in chunk_records])
per_file = token_table.groupby("file").agg(chunks=("tokens", "count"), total_tokens=("tokens", "sum"),
                                           avg_tokens=("tokens", "mean"), max_tokens=("tokens", "max"), words=("words", "sum"))
per_file["tokens_per_word"] = (per_file.total_tokens / per_file.words).round(2)
display(per_file.round(1))

check(f"every chunk fits the model limit ({limit} tokens)", token_table.tokens.max() <= limit,
      f"largest chunk {token_table.tokens.max()} tokens")

sentence_tokens = sum(count_tokens(s) for c in chunk_records for s in split_sentences(c["text"]))
summary_tokens = sum(count_tokens(r["summary"]) for r in summary_records)
question_tokens = count_tokens(TEST_QUESTION)
print(f"\nTokens embedded during ingestion:")
print(f"  sentences (to decide where to cut)   {sentence_tokens:>6}")
print(f"  chunks (stored in FAISS)             {token_table.tokens.sum():>6}")
print(f"  summaries                            {summary_tokens:>6}")
print(f"Tokens per question at search time:   {question_tokens:>6}   ('{TEST_QUESTION}')")

example = chunk_records[0]["text"]
example_tokens = tokenizer.convert_ids_to_tokens(tokenizer(example)["input_ids"])
print(f"\nExample: chunk 0 as the model reads it ({len(example_tokens)} tokens, first 25 shown):")
print(example_tokens[:25])

# The same chunks counted by other models' tokenizers
from transformers import AutoTokenizer
COMPARE_MODELS = [EMBEDDING_MODEL, "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
all_text = [c["text"] for c in chunk_records]
model_rows = []
for name in COMPARE_MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    counts = [len(tok(t)["input_ids"]) for t in all_text]
    model_rows.append({"model": name, "vocabulary size": tok.vocab_size, "total tokens": sum(counts),
                       "avg per chunk": round(np.mean(counts), 1), "largest chunk": max(counts),
                       "tokens per word": round(sum(counts) / sum(len(t.split()) for t in all_text), 2)})
print("\nToken count of all chunks with different models' tokenizers:")
display(pd.DataFrame(model_rows))

current_step = "Steps 4-7: search"
print(f"===== {current_step} =====\n")
print(f"Test question: {TEST_QUESTION}\n")

q = model.encode([TEST_QUESTION]).astype("float32")
faiss.normalize_L2(q)
test_results = search(TEST_QUESTION)
test_scores = [s for _, s in test_results]
direct = cosine_similarity(model.encode(TEST_QUESTION), stored_vectors)
brute_force_top = list(np.argsort(direct)[::-1][:TOP_K])

check("question embedded with the same model (same vector size)", q.shape[1] == index.d, f"{q.shape[1]} numbers")
check(f"returns the top {TOP_K} chunks", len(test_results) == TOP_K)
check("results are ranked from highest to lowest score", test_scores == sorted(test_scores, reverse=True),
      ", ".join(f"{s:.4f}" for s in test_scores))
check("FAISS scores equal the cosine formula",
      all(abs(s - direct[i]) < 1e-4 for i, s in test_results), "max difference "
      f"{max(abs(s - direct[i]) for i, s in test_results):.2e}")
check("FAISS compared the question with every chunk (same top 3 as checking all chunks one by one)",
      [i for i, _ in test_results] == [int(i) for i in brute_force_top])
check("every answer shows its file name", all(chunk_records[i]["file"] in FILES for i, _ in test_results),
      ", ".join(chunk_records[i]["file"] for i, _ in test_results))

ranked_files, selected_file, _ = summary_first_search(TEST_QUESTION)
check("summary routing picks a file that exists", selected_file in FILES,
      f"selected {selected_file} ({ranked_files[0][1]:.4f})")
same = selected_file == chunk_records[test_results[0][0]]["file"]
print(f"\nInfo: the summary picked {selected_file}; the best chunk overall is in "
      f"{chunk_records[test_results[0][0]]['file']} -> {'they agree' if same else 'they differ'}")

print()
print_answers(TEST_QUESTION, test_results)
