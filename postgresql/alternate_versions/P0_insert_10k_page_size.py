import psycopg2
import time
from psycopg2.extras import execute_values

DB_NAME = "cbde"
DB_HOST = "localhost"
DB_PORT = "5432"

conn = psycopg2.connect(
    dbname=DB_NAME,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

file_path = "../../data_used/bookcorpus_sentences.txt"
with open(file_path, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loading {len(sentences)} sentences into PostgreSQL...")

start = time.time()

execute_values(
    cur,
    "INSERT INTO sentences (sentence) VALUES %s",
    [(sentence,) for sentence in sentences],
    template=None,
    page_size=10000
)
conn.commit()

end = time.time()

cur.close()
conn.close()

total_time = (end - start) * 1000
print(f"Loaded all sentences in {total_time:.8f} ms")
