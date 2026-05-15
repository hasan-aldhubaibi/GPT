import re
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    # Remove wiki section headers (= = Title = =)
    text = re.sub(r'=+\s.*?\s=+', '', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove lines that are empty or just whitespace
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) > 5]
    return '\n'.join(lines)

for split in ['train', 'test']:
    # Adjust filename pattern if yours differs (e.g. wiki.train.tokens)
    candidates = list(RAW_DIR.glob(f'*{split}*'))
    if not candidates:
        print(f"WARNING: no file found for split '{split}' in {RAW_DIR}")
        continue
    src = candidates[0]
    print(f"Processing {src.name} ...")
    text = src.read_text(encoding='utf-8', errors='ignore')
    cleaned = clean_text(text)
    out_path = PROCESSED_DIR / f"{split}.txt"
    out_path.write_text(cleaned, encoding='utf-8')
    raw_lines = len(text.splitlines())
    clean_lines = len(cleaned.splitlines())
    print(f"  {raw_lines:,} lines → {clean_lines:,} lines saved to {out_path}")

print("Done.")