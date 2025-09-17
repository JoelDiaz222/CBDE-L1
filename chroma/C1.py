import chromadb
from sentence_transformers import SentenceTransformer
import time

client = chromadb.PersistentClient()
collection = client.get_collection(name="bookcorpus_sentences_cosine")

ids_and_docs = collection.get()
ids = ids_and_docs['ids']
documents = ids_and_docs['documents']

model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"Generating embeddings for {len(ids)} sentences...")

embeddings = model.encode(documents).tolist()

print(f"Storing embeddings...")

batch_size = 5000

start = time.time()
for i in range(0, len(ids), batch_size):
    end_idx = min(i + batch_size, len(ids))
    collection.update(
        ids=ids[i:end_idx],
        embeddings=embeddings[i:end_idx]
    )
end = time.time()

total_time = (end - start) * 1000
print(f"Embeddings stored for all {len(ids)} sentences in {total_time:.8f} ms")
print(f"Average time per embedding: {total_time / len(ids):.8f} ms")
