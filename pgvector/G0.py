import psycopg2
import time

# This is the same script as P0, just changing the sencences table name to sentences_pgvector

conn = psycopg2.connect(dbname="cbde", host="localhost", port="5432")
cur = conn.cursor()

file_path = "../data_used/bookcorpus_sentences.txt"

print("Loading sentences into PostgreSQL...")

with open(file_path, 'r', encoding='utf-8') as f:
    start = time.time()
    cur.copy_expert("COPY sentences_pgvector (sentence) FROM STDIN WITH (FORMAT TEXT)", f)
    conn.commit()
    end = time.time()

cur.close()
conn.close()

total_time = (end - start) * 1000
print(f"Loaded all sentences in {total_time:.8f} ms")
