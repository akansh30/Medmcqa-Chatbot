import json
import uuid
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en")

model.max_seq_length = 512

data_path = "data/medmcqa_raw.json"

with open(data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

print(f"Loaded {len(data)} records from {data_path}")

documents = []
metadata = []  

for item in tqdm(data, desc="Preparing documents"):
    q_id = item.get("id", str(uuid.uuid4()))
    options = [item["opa"], item["opb"], item["opc"], item["opd"]]
    option_str = "\n".join([f"{label}: {text}" for label, text in zip("ABCD", options)])
    content = f"{item['question']}\n{option_str}"
    documents.append(content)
    metadata.append({
        "id": q_id,
        "question": item["question"],
        "options": options,
        "correct": item["cop"],
        "subject": item.get("subject_name", ""),
        "topic": item.get("topic_name", ""),
        "explanation": item.get("exp", "")
    })


print("Generating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True, batch_size=64, normalize_embeddings=True)


embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  
index.add(embeddings)

faiss.write_index(index, "faiss_index/index.faiss")
with open("faiss_index/id_map.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("FAISS index and metadata saved.")
