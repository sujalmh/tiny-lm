import bz2
import xml.etree.ElementTree as ET
import re
import json
import os
import io
import array
from dotenv import load_dotenv
load_dotenv()
# ======================
# CONFIG
# ======================

XML_PATH = os.getenv("XML_PATH", "wiki-dump/enwiki-20260101-pages-articles-multistream.xml.bz2")
INDEX_PATH = os.getenv("INDEX_PATH", "wiki-dump/enwiki-20260101-pages-articles-multistream-index.txt.bz2")
TOKENS_FILE = os.getenv("TOKENS_FILE", "wiki-dump/tokens.bin")
TOKENIZE_CHECKPOINT = os.getenv("TOKENIZE_CHECKPOINT", "wiki-dump/tokenize_checkpoint.json")

EOS_TOKEN = 256
MIN_ARTICLE_CHARS = 200

# Pre-compile Regex for speed
RE_TEMPLATE = re.compile(r"\{\{.*?\}\}", flags=re.DOTALL)
RE_REF = re.compile(r"<ref.*?>.*?</ref>", flags=re.DOTALL)
RE_TAGS = re.compile(r"<.*?>")
RE_LINKS = re.compile(r"\[\[|\]\]")
RE_NEWLINES = re.compile(r"\n{2,}")

# ======================
# MULTISTREAM INDEX
# ======================

def load_multistream_index(index_path: str) -> list[int]:
    offsets = set() # Use set to dedup, as index contains multiple lines per block
    print("[-] Loading index...")
    with bz2.open(index_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line:
                # Format: byte_offset:page_id:title
                offsets.add(int(line.split(":", 1)[0]))
    return sorted(list(offsets))

# ======================
# XML STREAM
# ======================

def stream_pages_multistream(xml_path: str, offsets: list[int], start_block: int):
    with open(xml_path, "rb") as f:
        # Checkpoint logic: Only process from start_block + 1 to avoid duplicates on resume
        # or rely on the user to handle the partially written block.
        # Here we assume we restart at the exact block start.
        
        for block_idx in range(start_block, len(offsets)):
            f.seek(offsets[block_idx])
            
            # Decompress entire block
            decompressor = bz2.BZ2Decompressor()
            data = bytearray()
            
            # Read until end of BZ2 stream (one block)
            while not decompressor.eof:
                chunk = f.read(65536)
                if not chunk: break
                try:
                    data.extend(decompressor.decompress(chunk))
                except OSError:
                    break # Handle potential stream corruption gracefully

            # Wrap simply for ElementTree
            wrapped = b"<mediawiki>" + data + b"</mediawiki>"
            
            try:
                # iterparse is safer for memory, even on blocks
                context = ET.iterparse(io.BytesIO(wrapped), events=("end",))
                for _, elem in context:
                    if elem.tag.endswith("page"):
                        yield elem
                        elem.clear() 
            except ET.ParseError:
                pass # Skip malformed blocks
            
            # Yield End of Block signal
            yield None 

# ======================
# PARSING & CLEANING
# ======================

def parse_page(page):
    def get(tag):
        el = page.find(tag)
        return el.text if el is not None else ""

    return {
        "title": get("./title"),
        "ns": int(get("./ns") or -1),
        "redirect": page.find("./redirect") is not None,
        "text": page.find(".//text").text or ""
    }

def is_valid_article(p) -> bool:
    return (
        p["ns"] == 0 and
        not p["redirect"] and
        not p["title"].lower().startswith("list of") and
        len(p["text"]) >= MIN_ARTICLE_CHARS
    )

def clean_wikitext(text: str) -> str:
    # Use pre-compiled regex
    text = RE_TEMPLATE.sub("", text)
    text = RE_REF.sub("", text)
    text = RE_TAGS.sub("", text)
    text = RE_LINKS.sub("", text)
    text = RE_NEWLINES.sub("\n\n", text)
    return text.strip()

# ======================
# TOKENIZATION & IO
# ======================

def document_to_tokens(text: str) -> array.array:
    # 1. Encode text to UTF-8 bytes
    byte_vals = text.encode("utf-8")
    
    # 2. Convert bytes -> list of ints -> array of unsigned shorts ('H')
    # This ensures 'H' treats them as numeric values (0-255), not raw memory.
    tokens = array.array('H', list(byte_vals))
    
    # 3. Append EOS (256), which fits in 'H' (0-65535) but not 'B' (0-255)
    tokens.append(EOS_TOKEN)
    
    return tokens

def save_checkpoint(block_idx: int):
    with open(TOKENIZE_CHECKPOINT, "w") as f:
        json.dump({"block_idx": block_idx}, f)

def load_checkpoint() -> int:
    if not os.path.exists(TOKENIZE_CHECKPOINT):
        return 0
    with open(TOKENIZE_CHECKPOINT, "r") as f:
        return json.load(f)["block_idx"]

# ======================
# MAIN
# ======================

def build_token_cache():
    offsets = load_multistream_index(INDEX_PATH)
    start_block = load_checkpoint()
    
    # If resuming, we might have partial data for start_block in tokens.bin.
    # Ideally, you'd truncate tokens.bin or increment start_block by 1.
    # Here, we will just start processing.
    
    print(f"[+] Starting from block {start_block}/{len(offsets)}")

    # Open with 'ab' (append binary)
    with open(TOKENS_FILE, "ab") as fout:
        
        # We manually track index to handle the 'None' signal from generator
        current_block_idx = start_block
        
        generator = stream_pages_multistream(XML_PATH, offsets, start_block)
        
        for item in generator:
            # Check for End of Block signal
            if item is None:
                current_block_idx += 1
                if current_block_idx % 10 == 0:
                    save_checkpoint(current_block_idx)
                    print(f"[{current_block_idx}] Checkpoint saved.")
                continue

            # Process Page
            page = item
            p = parse_page(page)
            if is_valid_article(p):
                text = clean_wikitext(p["text"])
                tokens = document_to_tokens(text)
                # Fast binary write using array
                tokens.tofile(fout)

    print("[✓] Tokenization complete")

if __name__ == "__main__":
    build_token_cache()