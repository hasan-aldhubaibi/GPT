import numpy as np
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer


PROCESSED_DIR = Path("data/processed")
TOKENIZER_DIR = Path("tokenizer")

TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 16000


print("\n==============================")
print("TRAINING TOKENIZER")
print("==============================\n")


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.normalizer = Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

tokenizer.decoder = ByteLevelDecoder()


trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ],
    show_progress=True
)


# =========================================================
# TRAIN
# =========================================================

tokenizer.train(
    files=[str(PROCESSED_DIR / "train.txt")],
    trainer=trainer
)


# =========================================================
# SAVE TOKENIZER
# =========================================================

tokenizer.save(str(TOKENIZER_DIR / "tokenizer.json"))


# human-readable vocab
vocab = tokenizer.get_vocab()
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])

with open(TOKENIZER_DIR / "tokenizer.vocab", "w", encoding="utf-8") as f:
    for token, idx in vocab_sorted:
        f.write(f"{idx}\t{token}\n")


print(f"\nTokenizer vocab size: {len(vocab):,}")


# =========================================================
# SANITY CHECK
# =========================================================

sample = (
    PROCESSED_DIR / "train.txt"
).read_text(encoding="utf-8")[:1000]

encoded = tokenizer.encode(sample)

decoded = tokenizer.decode(encoded.ids)

print("\n==============================")
print("TOKENIZER SANITY CHECK")
print("==============================")

print("\nORIGINAL:\n")
print(sample[:500])

print("\nDECODED:\n")
print(decoded[:500])


# =========================================================
# ENCODE DATASETS (MEMORY SAFE)
# =========================================================

CHUNK_SIZE = 2_000_000  # characters


for split in ["train", "val"]:

    print(f"\n==============================")
    print(f"ENCODING {split.upper()}")
    print(f"==============================")

    path = PROCESSED_DIR / f"{split}.txt"

    temp_bin_path = PROCESSED_DIR / f"{split}_temp.bin"

    total_tokens = 0

    with open(path, "r", encoding="utf-8") as f, \
         open(temp_bin_path, "wb") as out_f:

        chunk_idx = 0

        while True:

            text_chunk = f.read(CHUNK_SIZE)

            if not text_chunk:
                break

            encoding = tokenizer.encode(text_chunk)

            ids = np.array(
                encoding.ids,
                dtype=np.uint16
            )

            ids.tofile(out_f)

            total_tokens += len(ids)

            chunk_idx += 1

            print(
                f"Chunk {chunk_idx} | "
                f"Chunk Tokens: {len(ids):,} | "
                f"Total Tokens: {total_tokens:,}"
            )

    # ============================================
    # CONVERT BIN -> NUMPY
    # ============================================

    print("\nCreating final numpy array...")

    mmap = np.memmap(
        temp_bin_path,
        dtype=np.uint16,
        mode="r"
    )

    out_path = PROCESSED_DIR / f"{split}_ids.npy"

    np.save(out_path, mmap)

    del mmap

    # remove temp file
    import os
    os.remove(temp_bin_path)

    print(f"\nSaved: {out_path}")
    print(f"Final token count: {total_tokens:,}")