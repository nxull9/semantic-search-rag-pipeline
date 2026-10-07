# -*- coding: utf-8 -*-
"""Final Project: RAG Pipeline with a Research Agent

Python export of notebooks/rag_research_agent_colab.ipynb, with the same code cell by cell.
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

"""Agent starts"""

import requests
from google.colab import userdata

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {"Authorization": f"Bearer {userdata.get('OPENROUTER_API_KEY')}", "Content-Type": "application/json"}
LLM_MODEL = "google/gemini-2.5-flash"     # any OpenRouter model that supports tools, e.g. "openai/gpt-4o"

def call_llm(messages, tools=None, temperature=0):
    """Send messages (and optional tools) to OpenRouter and return the model's reply message."""
    body = {"model": LLM_MODEL, "messages": messages, "temperature": temperature}
    if tools:
        body["tools"] = tools
    response = requests.post(OPENROUTER_URL, headers=OPENROUTER_HEADERS, json=body, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text[:300]}")
    return response.json()["choices"][0]["message"]

def parse_json_reply(text):
    """Read the JSON object the model was asked to return (also handles ```json fences)."""
    text = (text or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # fallback: plain text answer, sources taken from any [1], [2] markers
    return {"answer": text, "sources_used": sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})}

reply = call_llm([{"role": "user", "content": "Reply with the single word: ready"}])
print(f"LLM connected: {LLM_MODEL} -> {reply.get('content')}")

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the numbered sources provided.
Rules:
- Use only information found in the sources. Do not use outside knowledge.
- If the sources do not contain the answer, set "answer" to "The provided documents do not contain this information." and "sources_used" to [].
- Keep the answer short and clear.
Return only a JSON object: {"answer": "<your answer>", "sources_used": [<numbers of the sources you used>]}"""

def build_prompt(question, retrieved):
    """Phase 3.1, context assembly: the question plus the top 3 chunks, numbered with their file names."""
    blocks = []
    for n, (i, score) in enumerate(retrieved, start=1):
        c = chunk_records[i]
        blocks.append(f"[{n}] (file: {c['file']}, chunk {c['chunk_id']}, similarity {score:.4f})\n{c['text']}")
    return f"Question: {question}\n\nSources:\n" + "\n\n".join(blocks)

def rag_answer(question, top_k=TOP_K, show_prompt=False):
    retrieved = search(question, top_k)                              # Phase 2: top 3 chunks from FAISS
    prompt = build_prompt(question, retrieved)                       # Phase 3.1: prompt template
    if show_prompt:
        print("----- Prompt sent to the LLM -----\n" + prompt + "\n----------------------------------\n")
    reply = call_llm([{"role": "system", "content": RAG_SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}])          # Phase 3.2: LLM inference
    result = parse_json_reply(reply.get("content"))
    used = [int(n) for n in result.get("sources_used", []) if str(n).isdigit() and 1 <= int(n) <= len(retrieved)]

    # Phase 3.3: final output
    print("=" * 80)
    print(f"Question: {question}")
    print("=" * 80)
    print(f"Answer:\n  {result.get('answer')}\n")
    print("References used by the model:")
    for n in used:
        i, score = retrieved[n - 1]
        print(f"  [{n}] {chunk_records[i]['file']} | chunk {i} | similarity {score:.4f}")
    if not used:
        print("  (none)")
    unused = [n for n in range(1, len(retrieved) + 1) if n not in used]
    if unused:
        print("Retrieved but not used: " + ", ".join(
            f"[{n}] {chunk_records[retrieved[n - 1][0]]['file']} chunk {retrieved[n - 1][0]}" for n in unused))
    return {"question": question, "answer": result.get("answer"),
            "sources": [chunk_records[retrieved[n - 1][0]]["file"] for n in used]}

rag_question = input("Ask a question (the LLM answers from the top 3 chunks): ")
rag_result = rag_answer(rag_question, show_prompt=True)

import ast, operator

def search_documents(query: str) -> list:
    """Tool 1: semantic search in the FAISS vector database (top 3 chunks with file names)."""
    return [{"chunk_id": i, "file": chunk_records[i]["file"], "score": round(s, 4), "text": chunk_records[i]["text"]}
            for i, s in search(query, TOP_K)]

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

def calculator(expression: str):
    """Tool 2: safe arithmetic such as '1957 - 1927'. Only numbers and + - * / ** ( ) are accepted."""
    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError("only numbers and + - * / ** ( ) are allowed")
    return evaluate(ast.parse(expression, mode="eval").body)

AGENT_TOOLS = {"search_documents": search_documents, "calculator": calculator}

AGENT_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_documents",
        "description": "Search three documents (football_clubs.txt: football clubs; solar_system.txt: astronomy; "
                       "coffee.txt: coffee) and return the 3 most relevant chunks with chunk_id, file, score and text. "
                       "Search once for each separate fact you need.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "What to look for, in plain English"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression such as '1957 - 1927' or '465 - 430'. Use it for every calculation.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string", "description": "Numbers and + - * / ** ( ) only"}},
                       "required": ["expression"]}}},
]

print("Tools ready:", ", ".join(AGENT_TOOLS))
print("Test:", calculator("1957 - 1927"), "|", search_documents("Which club signed Karim Benzema?")[0]["file"])

AGENT_SYSTEM_PROMPT = """You are a research agent. You answer questions about three documents:
football_clubs.txt (football clubs), solar_system.txt (astronomy) and coffee.txt (coffee).
Rules:
- Use search_documents to find facts. Search separately for each fact you need.
- Use calculator for every calculation; never calculate in your head.
- Answer ONLY from the search results. Do not use outside knowledge.
- If the documents do not contain the information, answer "The provided documents do not contain this information." with no sources.
When you are finished, reply with only a JSON object:
{"answer": "<your answer>", "sources": [<chunk_id of every chunk you used>]}"""

MAX_AGENT_STEPS = 6       # guardrail: maximum number of reason -> act rounds

def run_agent(question, max_steps=MAX_AGENT_STEPS, verbose=True):
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": question}]
    retrieved_ids, unknown_tools, steps_used = set(), [], 0
    result, finished = None, False

    if verbose:
        print("=" * 80)
        print(f"Question: {question}")
        print("=" * 80)

    for step in range(1, max_steps + 1):
        steps_used = step
        msg = call_llm(messages, tools=AGENT_TOOL_SCHEMAS)                  # REASON: the model decides
        tool_calls = msg.get("tool_calls")
        if not tool_calls:                                                  # no tool needed: final answer
            result, finished = parse_json_reply(msg.get("content")), True
            break

        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})
        for call in tool_calls:                                             # ACT: the notebook runs the tool
            name, args = call["function"]["name"], {}
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                if name not in AGENT_TOOLS:                                 # guardrail: allowed tools only
                    unknown_tools.append(name)
                    raise ValueError(f"tool '{name}' is not allowed")
                output = AGENT_TOOLS[name](**args)
            except Exception as error:                                      # errors go back to the model
                output = {"error": str(error)}

            if name == "search_documents" and isinstance(output, list):
                retrieved_ids.update(r["chunk_id"] for r in output)
                shown = ", ".join(f"chunk {r['chunk_id']} ({r['file']}, {r['score']:.2f})" for r in output)
            else:
                shown = output
            if verbose:
                print(f"Step {step} | {name}({', '.join(f'{k}={v!r}' for k, v in args.items())}) -> {shown}")
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(output, ensure_ascii=False)})   # OBSERVE: result goes back

    if result is None:                                                      # guardrail: step limit reached
        result = {"answer": f"Stopped after {max_steps} steps without a final answer.", "sources": []}

    sources = [int(s) for s in result.get("sources", result.get("sources_used", [])) if str(s).isdigit()]
    grounded = all(s in retrieved_ids for s in sources)

    if verbose:
        print(f"\nFinal answer:\n  {result.get('answer')}\n")
        print("Sources used:")
        for s in sources:
            print(f"  chunk {s} | {chunk_records[s]['file'] if 0 <= s < len(chunk_records) else 'unknown'}")
        if not sources:
            print("  (none)")
        print("\nGuardrails:")
        print(f"  {'✓' if finished else '✗'} used {steps_used} of {max_steps} allowed steps"
              + ("" if finished else " (stopped by the step limit)"))
        print(f"  {'✓' if not unknown_tools else '✗'} only allowed tools were called" + (f" (blocked: {unknown_tools})" if unknown_tools else ""))
        print(f"  {'✓' if grounded else '✗'} every cited chunk was retrieved by the agent in this run"
              + ("" if grounded else f" (not retrieved: {[s for s in sources if s not in retrieved_ids]})"))
    return {"question": question, "answer": result.get("answer"), "sources": sources,
            "steps": steps_used, "finished": finished, "grounded": grounded}

print("Try a question that needs more than one fact, for example:")
print("  How many years after Al Ittihad was Al Hilal founded?")
print("  How much hotter is Venus than the daytime temperature on Mercury?")
agent_question = input("\nAsk the agent: ")
agent_result = run_agent(agent_question)

TEST_QUESTIONS = [
    "Which club signed Karim Benzema?",
    "Where is Khawlani coffee grown?",
    "Which planet is the hottest?",
    "What is the capital of France?",                                  # not in the documents
]
AGENT_TEST_QUESTIONS = [
    "How many years after Al Ittihad was Al Hilal founded?",           # 2 searches + calculator
    "How much hotter is Venus than the daytime temperature on Mercury?",
]

rows = []
for q in TEST_QUESTIONS:
    r = rag_answer(q)
    rows.append({"type": "RAG", "question": q, "answer": r["answer"], "sources": ", ".join(r["sources"]) or "-"})
    print()
for q in AGENT_TEST_QUESTIONS:
    r = run_agent(q)
    rows.append({"type": "Agent", "question": q, "answer": r["answer"],
                 "sources": ", ".join(chunk_records[s]["file"] for s in r["sources"]) or "-"})
    print()

pd.set_option("display.max_colwidth", 120)
pd.DataFrame(rows)
