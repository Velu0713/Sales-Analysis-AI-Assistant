# rag.py
import os
import faiss
import pickle
import numpy as np
import requests
import json

# CONFIG: LM Studio embedding endpoint & model name
LMSTUDIO_EMBED_URL = os.getenv("LMSTUDIO_BASE", "http://127.0.0.1:1234/v1") + "/embeddings"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")  # adjust if needed

# VECTORSTORE
INDEX_DIR = "vectorstore"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
META_FILE = os.path.join(INDEX_DIR, "meta.pkl")

# safe default embedding dim - adjust to your chosen model
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

os.makedirs(INDEX_DIR, exist_ok=True)


def embed_text_local(text_list):
    """Call LM Studio local embeddings API. Returns list of vectors."""
    if not text_list:
        return []
    payload = {"model": EMBED_MODEL_NAME, "input": text_list}
    r = requests.post(LMSTUDIO_EMBED_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=300)
    r.raise_for_status()
    data = r.json()
    embeddings = [item["embedding"] for item in data.get("data", [])]
    return embeddings


def save_index(index, meta):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)


def load_index():
    if not (os.path.exists(INDEX_FILE) and os.path.exists(META_FILE)):
        return None
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    ids = [m["id"] for m in meta]
    texts = [m["text"] for m in meta]
    embeddings = None
    try:
        embeddings = np.array([m.get("embedding") for m in meta if m.get("embedding") is not None], dtype="float32")
    except Exception:
        embeddings = None
    return index, ids, texts, embeddings


def build_index_from_text_rows(text_rows):
    """
    text_rows: list of (row_id (str/int), text (str))
    """
    if not text_rows:
        raise ValueError("text_rows empty")
    ids = [rid for rid, _ in text_rows]
    texts = [txt for _, txt in text_rows]

    print("📌 Generating embeddings...")
    embeddings = embed_text_local(texts)

    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding count mismatch")

    emb_matrix = np.array(embeddings).astype("float32")

    print("📌 Creating FAISS index...")
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)

    # Save meta: keep id, text, (optionally embedding to speed future load)
    meta = [{"id": ids[i], "text": texts[i], "embedding": embeddings[i]} for i in range(len(ids))]
    save_index(index, meta)
    print("✅ FAISS index saved.")
    return index, ids, texts, emb_matrix


def retrieve(query, _unused, index, ids, texts, k=5):
    """Return list of dicts: {'id', 'text', 'score'} (distance)."""
    if index is None:
        return []
    q_vec = embed_text_local([query])[0]
    q_vector = np.array(q_vec).astype("float32").reshape(1, -1)
    distances, indices = index.search(q_vector, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < len(ids):
            results.append({"id": ids[idx], "text": texts[idx], "score": float(distances[0][rank])})
    return results


def rerank_with_cosine(query, _unused, retrieved, top_k=5):
    """Compute cosine between query and candidate texts (asks embeddings for each)."""
    if not retrieved:
        return []
    q_vec = np.array(embed_text_local([query])[0])
    scored = []
    for item in retrieved:
        t = item.get("text", "")
        t_vec = np.array(embed_text_local([t])[0])
        denom = (np.linalg.norm(q_vec) * np.linalg.norm(t_vec))
        score = float(np.dot(q_vec, t_vec) / denom) if denom != 0 else 0.0
        item["rerank_score"] = score
        scored.append(item)
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]