import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from scipy.spatial.distance import cosine, euclidean

DB_NAME = "cbde"
DB_HOST = "localhost"
DB_PORT = "5432"

conn = psycopg2.connect(
    dbname=DB_NAME,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

file_path = "../data_used/our_10_sentences.txt"
with open(file_path, "r", encoding="utf-8") as f:
    chosen_sentences = [line.strip() for line in f.readlines() if line.strip()]

cur.execute("SELECT id, sentence, embedding FROM bookcorpus_sentences")
rows = cur.fetchall()
all_ids = [r[0] for r in rows]
all_sentences = [r[1] for r in rows]
all_embeddings = [np.array(r[2]) for r in rows]

cur.close()
conn.close()

model = SentenceTransformer('all-MiniLM-L6-v2')

similarity_times = []

print("Top-2 similar sentences for chosen sentences:\n")

for sentence in chosen_sentences:
    start = time.time()
    query_embedding = model.encode(sentence)

    # Compute distances to all stored sentences
    cosine_dists = [cosine(query_embedding, e) for e in all_embeddings]
    euclidean_dists = [euclidean(query_embedding, e) for e in all_embeddings]

    # Get top-2 closest sentences
    top2_cosine_idx = np.argsort(cosine_dists)[:2]
    top2_euc_idx = np.argsort(euclidean_dists)[:2]

    end = time.time()
    similarity_times.append((end - start) * 1000)

    print(f"Query: {sentence}")
    print("Top 2 by Cosine:", [all_sentences[i] for i in top2_cosine_idx])
    print("Top 2 by Euclidean:", [all_sentences[i] for i in top2_euc_idx])
    print("-" * 60)

times = np.array(similarity_times)
print("\nDistances computation timing stats (milliseconds):")
print(f"Minimum: {times.min():.8f}")
print(f"Maximum: {times.max():.8f}")
print(f"Standard deviation: {times.std():.8f}")
print(f"Average time: {times.mean():.8f}")
