import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re

index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/id_map.pkl", "rb") as f:
    metadata = pickle.load(f)  

# embedding model
model = SentenceTransformer("BAAI/bge-small-en")
model.max_seq_length = 512

def tokenize(text):
    return {w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', text)}

def normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())

def search_faiss(query: str, top_k: int = 5, score_threshold: float = 0.88, overlap_thresh: float = 0.25):
    norm_query = normalize(query)

    # For the Exact match case
    for meta in metadata:
        if normalize(meta["question"]) == norm_query:
            return [meta]

    # Semantic + overlap filtering
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_vec).astype("float32"), top_k)

    results = []
    q_terms = tokenize(query)

    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(metadata):
            meta = metadata[idx]
            m_terms = tokenize(meta["question"])
            jaccard = len(q_terms & m_terms) / max(len(q_terms | m_terms), 1)

            if score >= score_threshold and jaccard >= overlap_thresh:
                results.append(meta)

    return results[:1]

#testing
if __name__ == "__main__":
    query = "What is the effect of B12 deficiency?"
    results = search_faiss(query)
    for res in results:
        print(f"\nQ: {res['question']}")
        for i, opt in enumerate(res['options']):
            print(f"  {chr(65+i)}: {opt}")
        correct = res.get("correct", -1)
        if 0 <= correct < len(res["options"]):
            print(f"Correct Answer: {chr(65+correct)}: {res['options'][correct]}")