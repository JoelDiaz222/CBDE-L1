import psycopg2
import time

DB_NAME = "cbde"
DB_HOST = "localhost"
DB_PORT = "5432"

conn = psycopg2.connect(
    dbname=DB_NAME,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

file_path = "../data_used/bookcorpus_sentences.txt"

print(f"Loading sentences from {file_path} into PostgreSQL...")

with open(file_path, 'r', encoding='utf-8') as f:
    start = time.time()
    cur.copy_expert("COPY sentences (sentence) FROM STDIN WITH (FORMAT TEXT)", f)
    conn.commit()
    end = time.time()

cur.close()
conn.close()

total_time = (end - start) * 1000
print(f"Loaded all sentences in {total_time:.8f} ms")
