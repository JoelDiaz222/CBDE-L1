import chromadb
import time

class DummyEmbedding:
    def __call__(self, input):
        return [[0.0] * 384 for _ in input]

dummy_ef = DummyEmbedding()

client = chromadb.PersistentClient()

try:
   client.delete_collection(name="bookcorpus_sentences_cosine")
except:
   pass

collection = client.create_collection(
    name="bookcorpus_sentences_cosine",
    embedding_function=dummy_ef,
    metadata={"hnsw:space": "cosine"}
)

file_path = "../data_used/bookcorpus_sentences.txt"
with open(file_path, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loading {len(sentences)} sentences into Chroma...")

ids = [str(i) for i in range(len(sentences))]
batch_size = 5000

start = time.time()
for i in range(0, len(sentences), batch_size):
    end_idx = min(i + batch_size, len(sentences))
    collection.add(
        ids=ids[i:end_idx],
        documents=sentences[i:end_idx]
    )
end = time.time()

total_time = (end - start) * 1000
print(f"Loaded all {len(sentences)} sentences in {total_time:.8f} ms")
print(f"Average time per sentence: {total_time / len(sentences):.8f} ms")
