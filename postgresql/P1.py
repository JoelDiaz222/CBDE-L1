import time
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

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

print(f"Generating embeddings for {len(rows)} sentences...")

embedding_data = []
for row in rows:
    id, sentence = row
    embedding_as_list = model.encode(sentence).tolist()
    embedding_data.append((id, embedding_as_list))

print(f"Storing embeddings...")

start = time.time()

execute_values(
    cur,
    "INSERT INTO embeddings (id, embedding) VALUES %s",
    embedding_data,
    template=None,
    page_size=10000
)
conn.commit()

end = time.time()

cur.close()
conn.close()

total_time = (end - start) * 1000
print(f"Embeddings stored for all {len(rows)} sentences in {total_time:.8f} ms")
print(f"Average time per embedding: {total_time / len(rows):.8f} ms")
