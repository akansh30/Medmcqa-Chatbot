import faiss
import pickle

index = faiss.read_index("faiss_index/index.faiss")
print(f" Total vectors in FAISS index: {index.ntotal}")

with open("faiss_index/id_map.pkl", "rb") as f:
    id_map = pickle.load(f)

print(f" Total metadata records: {len(id_map)}")

if index.ntotal == len(id_map):
    print(" All records successfully ingested.")
else:
    print(" Mismatch: Some records may be missing.")