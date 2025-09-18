import psycopg2
import time
import numpy as np

conn = psycopg2.connect(dbname="cbde", host="localhost", port="5432")
cur = conn.cursor()

file_path = "../../data_used/bookcorpus_sentences.txt"
with open(file_path, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

text_insert_times = []

print(f"Loading {len(sentences)} sentences into PostgreSQL...")

for sentence in sentences:
    start = time.time()
    cur.execute("INSERT INTO sentences (sentence) VALUES (%s)", (sentence,))
    conn.commit()
    end = time.time()
    text_insert_times.append((end - start) * 1000)

cur.close()
conn.close()

print(f"Loaded all sentences.\n")

times = np.array(text_insert_times)

print("Text insertion timing stats (milliseconds):")
print(f"Total time: {times.sum():.8f}")
print(f"Minimum: {times.min():.8f}")
print(f"Maximum: {times.max():.8f}")
print(f"Standard deviation: {times.std():.8f}")
print(f"Average time: {times.mean():.8f}")
