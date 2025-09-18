import psycopg2
from sentence_transformers import SentenceTransformer
import time
import numpy as np

DB_NAME = "cbde"
DB_HOST = "localhost"
DB_PORT = "5432"

conn = psycopg2.connect(
    dbname=DB_NAME,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

cur.execute("SELECT id, sentence FROM sentences")
rows = cur.fetchall()

model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_insert_times = []

print(f"Generating and storing embeddings for {len(rows)} sentences...")

for row in rows:
    id, sentence = row
    embedding_as_list = model.encode(sentence).tolist()

    start = time.time()
    cur.execute(
        "INSERT INTO embeddings (id, embedding) VALUES (%s, %s)",
        (id, embedding_as_list)
    )
    conn.commit()
    end = time.time()

    embedding_insert_times.append((end - start) * 1000)

cur.close()
conn.close()

print(f"Embeddings generated and stored for all sentences.\n")

times = np.array(embedding_insert_times)

print("Embedding insertion timing stats (milliseconds):")
print(f"Total time: {times.sum():.8f}")
print(f"Minimum: {times.min():.8f}")
print(f"Maximum: {times.max():.8f}")
print(f"Standard deviation: {times.std():.8f}")
print(f"Average time: {times.mean():.8f}")
