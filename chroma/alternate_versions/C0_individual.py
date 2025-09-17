import chromadb
import time
import numpy as np

class DummyEmbedding:
    def __call__(self, input):
        return [[0.0] * 384 for _ in input]

dummy_ef = DummyEmbedding()

client = chromadb.PersistentClient()

try:
   client.delete_collection(name="bookcorpus_sentences_cosine")
except:
   pass

collection = client.create_collection(name="bookcorpus_sentences_cosine", embedding_function=dummy_ef)

file_path = "../../data_used/bookcorpus_sentences.txt"
with open(file_path, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

text_insert_times = []

print(f"Loading {len(sentences)} sentences into Chroma...")

for i, sentence in enumerate(sentences):
    start = time.time()
    collection.add(
        ids=[f"{i}"],
        documents=[sentence]
    )
    end = time.time()

    text_insert_times.append((end - start) * 1000)


print(f"Loaded all sentences.\n")

times = np.array(text_insert_times)

print("Text insertion timing stats (milliseconds):")
print(f"Total time: {times.sum()}")
print(f"Minimum: {times.min():.8f}")
print(f"Maximum: {times.max():.8f}")
print(f"Standard deviation: {times.std():.8f}")
print(f"Average time: {times.mean():.8f}")
