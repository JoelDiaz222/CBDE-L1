import chromadb
from sentence_transformers import SentenceTransformer
import time

client = chromadb.PersistentClient()
source_collection = client.get_collection(name="bookcorpus_sentences")

ids_and_docs = source_collection.get()
ids = ids_and_docs['ids']
documents = ids_and_docs['documents']

model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"Generating embeddings for {len(ids)} sentences...")
embeddings = model.encode(documents).tolist()

try:
    client.delete_collection(name="bookcorpus_sentences_cosine")
except:
    pass

embeddings_collection = client.create_collection(
    name="bookcorpus_sentences_cosine",
    metadata = {"hnsw:space": "cosine"}
)

print("Storing embeddings in new collection...")

batch_size = 5000

start = time.time()
for i in range(0, len(ids), batch_size):
    end_idx = min(i + batch_size, len(ids))
    embeddings_collection.add(
        ids=ids[i:end_idx],
        documents=documents[i:end_idx],
        embeddings=embeddings[i:end_idx]
    )
end = time.time()

total_time = (end - start) * 1000
print(f"Embeddings stored in new collection in {total_time:.8f} ms")
print(f"Average time per embedding: {total_time / len(ids):.8f} ms")
