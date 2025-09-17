import psycopg2
import time
import numpy as np
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(dbname="cbde", host="localhost", port="5432")
cur = conn.cursor()

file_path = "../data_used/our_10_sentences.txt"
with open(file_path, "r", encoding="utf-8") as f:
    chosen_sentences = [line.strip() for line in f if line.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')

euclidean_times = []
cosine_times = []

for sentence in chosen_sentences:
    query_embedding = model.encode(sentence).tolist()

    print(f"\nQuery: {sentence}\n")

    # Euclidean distance
    euclidean_query = """
    WITH query AS (SELECT %s::float8[] AS embedding),
    distances AS (
        SELECT s.id, s.sentence,
               sqrt(sum(power(e_val - q_val, 2))) AS euclidean_distance
        FROM sentences s
        JOIN embeddings e ON s.id = e.id
        CROSS JOIN query q
        JOIN LATERAL (
            SELECT unnest(e.embedding) AS e_val, unnest(q.embedding) AS q_val
        ) AS vals ON true
        GROUP BY s.id, s.sentence
    )
    SELECT id, sentence, euclidean_distance
    FROM distances
    ORDER BY euclidean_distance ASC
    LIMIT 2;
    """

    start_time = time.time()
    cur.execute(euclidean_query, (query_embedding,))
    euclidean_results = cur.fetchall()
    end_time = time.time()
    euclidean_times.append(end_time - start_time)

    print("Top Euclidean:")
    for r in euclidean_results:
        print(f" {r[1]} (distance={r[2]:.8f})")

    # Cosine distance
    cosine_query = """
    WITH query AS (SELECT %s::float8[] AS embedding),
    distances AS (
        SELECT s.id, s.sentence,
               1 - (sum(e_val * q_val) / 
                   (sqrt(sum(e_val * e_val)) * sqrt(sum(q_val * q_val)))) AS cosine_distance
        FROM sentences s
        JOIN embeddings e ON s.id = e.id
        CROSS JOIN query q
        JOIN LATERAL (
            SELECT unnest(e.embedding) AS e_val, unnest(q.embedding) AS q_val
        ) AS vals ON true
        GROUP BY s.id, s.sentence
    )
    SELECT id, sentence, cosine_distance
    FROM distances
    ORDER BY cosine_distance ASC
    LIMIT 2;
    """

    start_time = time.time()
    cur.execute(cosine_query, (query_embedding,))
    cosine_results = cur.fetchall()
    end_time = time.time()
    cosine_times.append(end_time - start_time)

    print("Top Cosine:")
    for r in cosine_results:
        print(f" {r[1]} (distance={r[2]:.8f})")

cur.close()
conn.close()

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
