import chromadb
import time
import numpy as np
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient()
collection_cosine = client.get_collection(name="bookcorpus_sentences_cosine")

try:
    client.delete_collection(name="bookcorpus_sentences_euclidean")
except:
    pass

collection_euclidean = client.create_collection(
    name="bookcorpus_sentences_euclidean",
    embedding_function=None,
    metadata={"hnsw:space": "l2"}
)

# Copy all the data from original collection to Euclidean collection
initial_collection = collection_cosine.get(include=['documents', 'embeddings'])
ids = initial_collection['ids']
documents = initial_collection['documents']
embeddings = initial_collection['embeddings']

batch_size = 5000

for i in range(0, len(ids), batch_size):
    end_idx = min(i + batch_size, len(ids))
    collection_euclidean.add(
        ids=ids[i:end_idx],
        documents=documents[i:end_idx],
        embeddings=embeddings[i:end_idx]
    )

file_path = "../data_used/our_10_sentences.txt"
with open(file_path, "r", encoding="utf-8") as f:
    chosen_sentences = [line.strip() for line in f if line.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')

# Store timing results
euclidean_times = []
cosine_times = []

for sentence in chosen_sentences:
    query_embedding = model.encode(sentence).tolist()
    print(f"\nQuery: {sentence}\n")

    # Euclidean distance
    start_time = time.time()
    results_euclidean = collection_euclidean.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=['documents', 'distances']
    )
    end_time = time.time()
    euclidean_times.append(end_time - start_time)

    print("Top Euclidean:")
    for doc, dist in zip(results_euclidean['documents'][0], results_euclidean['distances'][0]):
        euclidean_dist = dist ** 0.5    # The square root has to be calculated to obtain the Euclidean distance
        print(f" {doc} (distance={euclidean_dist:.8f})")

    # Cosine distance
    start_time = time.time()
    results_cosine = collection_cosine.query(
        query_embeddings=[query_embedding],
        n_results=2,
        include=['documents', 'distances']
    )
    end_time = time.time()
    cosine_times.append(end_time - start_time)

    print("Top Cosine:")
    for doc, dist in zip(results_cosine['documents'][0], results_cosine['distances'][0]):
        print(f" {doc} (distance={dist:.8f})")

euclidean_times = np.array(euclidean_times)
cosine_times = np.array(cosine_times)

print("\n" + "=" * 50)
print("EUCLIDEAN DISTANCE TIMING STATISTICS:")
print(f"Minimum: {euclidean_times.min():.8f}")
print(f"Maximum: {euclidean_times.max():.8f}")
print(f"Standard deviation: {euclidean_times.std():.8f}")
print(f"Average time: {euclidean_times.mean():.8f}")

print("\n" + "=" * 50)
print("COSINE DISTANCE TIMING STATISTICS:")
print(f"Minimum: {cosine_times.min():.8f}")
print(f"Maximum: {cosine_times.max():.8f}")
print(f"Standard deviation: {cosine_times.std():.8f}")
print(f"Average time: {cosine_times.mean():.8f}")
