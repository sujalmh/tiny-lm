import bz2
import xml.etree.ElementTree as ET
import re
import struct
import json
import os
from collections import deque, Counter

# ======================
# CONFIG
# ======================

WIKI_DUMP = "wiki-dump\\enwiki-20260101-pages-articles-multistream.xml.bz2"
TOKENS_FILE = "tokens.bin"
CHECKPOINT_FILE = "wiki-dump\\checkpoint.json"

CONTEXT_LEN = 256
EOS_TOKEN = 256          # byte-level eos
TOKENIZER_VERSION = "byte-v1"
MIN_ARTICLE_CHARS = 200

# ======================
# STEP 1: STREAM XML
# ======================

def stream_pages(path):
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if elem.tag.endswith("page"):
                yield elem
                elem.clear()

def parse_page(page):
    def get(tag):
        el = page.find(tag)
        return el.text if el is not None else None

    return {
        "title": get("./title") or "",
        "ns": int(get("./ns") or -1),
        "redirect": page.find("./redirect") is not None,
        "text": page.find(".//text").text if page.find(".//text") is not None else ""
    }

# ======================
# STEP 2: FILTER
# ======================

def is_valid_article(p):
    if p["ns"] != 0:
        return False
    if p["redirect"]:
        return False
    if p["title"].lower().startswith("list of"):
        return False
    if not p["text"] or len(p["text"]) < MIN_ARTICLE_CHARS:
        return False
    return True

# ======================
# STEP 3: CLEAN TEXT
# ======================

def clean_wikitext(text):
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\<ref.*?\>.*?\<\/ref\>", "", text, flags=re.DOTALL)
    text = re.sub(r"\<.*?\>", "", text)
    text = re.sub(r"\[\[|\]\]", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

# ======================
# STEP 4: TOKENIZATION
# ======================

def byte_tokenize(text):
    return list(text.encode("utf-8"))

def document_to_tokens(text):
    tokens = byte_tokenize(text)
    tokens.append(EOS_TOKEN)
    return tokens

# ======================
# STEP 5: SAVE / LOAD TOKENS
# ======================

def save_tokens(f, tokens):
    f.write(struct.pack("I", len(tokens)))
    f.write(struct.pack(f"{len(tokens)}I", *tokens))

def load_tokens(f):
    raw = f.read(4)
    if not raw:
        return None
    length = struct.unpack("I", raw)[0]
    data = f.read(4 * length)
    return list(struct.unpack(f"{length}I", data))

# ======================
# STEP 6: BUILD TOKEN CACHE (ONE TIME)
# ======================

def build_token_cache():
    if os.path.exists(TOKENS_FILE):
        print("[!] tokens.bin already exists — skipping tokenization")
        return

    print("[+] Building token cache...")
    count = 0

    with open(TOKENS_FILE, "ab") as fout:
        for page in stream_pages(WIKI_DUMP):
            p = parse_page(page)
            if not is_valid_article(p):
                continue

            clean = clean_wikitext(p["text"])
            tokens = document_to_tokens(clean)
            save_tokens(fout, tokens)

            count += 1
            if count % 10000 == 0:
                print(f"  tokenized {count} articles")

    print(f"[✓] Tokenization complete: {count} articles")

# ======================
# STEP 7: CHECKPOINT
# ======================

def save_checkpoint(page_idx, buffer):
    state = {
        "page_index": page_idx,
        "buffer": list(buffer),
        "tokenizer": TOKENIZER_VERSION
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f)

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    with open(CHECKPOINT_FILE) as f:
        return json.load(f)

# ======================
# STEP 8: TRAINING STREAM (RESUMABLE)
# ======================

def training_stream():
    checkpoint = load_checkpoint()
    buffer = deque()
    start_page = 0

    if checkpoint:
        assert checkpoint["tokenizer"] == TOKENIZER_VERSION
        buffer.extend(checkpoint["buffer"])
        start_page = checkpoint["page_index"]
        print(f"[+] Resuming from article {start_page}")

    with open(TOKENS_FILE, "rb") as f:
        page_idx = 0

        while True:
            tokens = load_tokens(f)
            if tokens is None:
                break

            page_idx += 1
            if page_idx <= start_page:
                continue

            buffer.extend(tokens)

            while len(buffer) >= CONTEXT_LEN + 1:
                seq = [buffer.popleft() for _ in range(CONTEXT_LEN + 1)]
                yield seq[:-1], seq[1:], page_idx, buffer

# ======================
# STEP 9: EXAMPLE RUN
# ======================

if __name__ == "__main__":
    build_token_cache()

    print("[+] Starting training stream (CTRL+C to stop)")
    try:
        for i, (x, y, page_idx, buffer) in enumerate(training_stream()):
            if i % 1000 == 0:
                print(f"batch {i} | article {page_idx}")

            # simulate training step here

    except KeyboardInterrupt:
        print("\n[!] Interrupted — saving checkpoint")
        save_checkpoint(page_idx, buffer)
        print("[✓] Checkpoint saved")
