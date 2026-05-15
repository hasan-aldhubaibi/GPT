from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading TinyStories dataset...")

dataset = load_dataset(
    "roneneldan/TinyStories",
    trust_remote_code=True
)

print("Dataset downloaded!")

train_texts = dataset["train"]["text"]
val_texts = dataset["validation"]["text"]

print("Saving training file...")

with open("data/raw/train.txt", "w", encoding="utf-8") as f:
    for i, story in enumerate(train_texts):
        f.write(story + "\n")

        if i % 10000 == 0:
            print(f"Saved {i} training stories...")

print("Saving validation file...")

with open("data/raw/val.txt", "w", encoding="utf-8") as f:
    for i, story in enumerate(val_texts):
        f.write(story + "\n")

        if i % 1000 == 0:
            print(f"Saved {i} validation stories...")

print("Done!")