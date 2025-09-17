from datasets import load_dataset

ds = load_dataset("rojagtap/bookcorpus", split="train")

sentences = ds[:10000]["text"]

with open("bookcorpus_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence.strip() + "\n")

print("Saved 10,000 sentences successfully.")
