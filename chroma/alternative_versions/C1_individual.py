import chromadb
from sentence_transformers import SentenceTransformer
import time
import numpy as np

client = chromadb.PersistentClient()
collection = client.get_collection(name="bookcorpus_sentences")

ids_and_docs = collection.get()
ids = ids_and_docs['ids']
documents = ids_and_docs['documents']

model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"Generating and storing embeddings for {len(ids)} sentences...")

embedding_insert_times = []

for id, doc  in zip(ids, documents):
    embedding_as_list = model.encode(doc).tolist()

    start = time.time()
    collection.update(
        ids=[id],
        embeddings=[embedding_as_list]
    )
    end = time.time()

    embedding_insert_times.append((end - start) * 1000)

print(f"Embeddings generated and stored for all sentences.\n")

times = np.array(embedding_insert_times)

print("Embedding insertion timing stats (milliseconds):")
print(f"Total time: {times.sum()}")
print(f"Minimum: {times.min():.8f}")
print(f"Maximum: {times.max():.8f}")
print(f"Standard deviation: {times.std():.8f}")
print(f"Average time: {times.mean():.8f}")
