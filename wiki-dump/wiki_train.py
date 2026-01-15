import struct
import json
import os
import array
from collections import deque
from dotenv import load_dotenv
load_dotenv()
# ======================
# CONFIG
# ======================


TOKENS_FILE = os.getenv("TOKENS_FILE", "wiki-dump/tokens.bin")
TRAIN_CHECKPOINT = os.getenv("TRAIN_CHECKPOINT", "wiki-dump/train_checkpoint.json")

CONTEXT_LEN = 256
BATCH_SIZE = 32 # Added for realistic loop usage
CHUNK_SIZE = 65536 * 2 # How many bytes to read from disk at once

# ======================
# CHECKPOINT
# ======================

def load_checkpoint():
    if not os.path.exists(TRAIN_CHECKPOINT):
        # file_offset (bytes), buffer (list of ints)
        return 0, []
    
    try:
        ckpt = json.load(open(TRAIN_CHECKPOINT))
        return ckpt.get("file_offset", 0), ckpt.get("buffer", [])
    except (json.JSONDecodeError, KeyError):
        return 0, []

def save_checkpoint(file_offset: int, buffer: deque):
    # We only save a small part of the buffer to keep JSON small
    # Ideally, you shouldn't save the buffer in JSON for production, 
    # but for this scale it is fine.
    json.dump(
        {
            "file_offset": file_offset, 
            "buffer": list(buffer)
        },
        open(TRAIN_CHECKPOINT, "w")
    )

# ======================
# TRAINING STREAM
# ======================

def training_stream():
    # 1. Load state
    file_offset, saved_buffer = load_checkpoint()
    
    # Reconstruct deque
    buffer = deque(saved_buffer)
    
    # 2. Open binary file
    with open(TOKENS_FILE, "rb") as f:
        # Fast Seek to where we left off
        f.seek(file_offset)
        print(f"[+] Resumed from byte offset: {file_offset}")

        while True:
            # Maintain buffer size (CONTEXT_LEN + 1 is min required for x,y pair)
            while len(buffer) < CONTEXT_LEN + 1:
                # Read raw bytes
                chunk_bytes = f.read(CHUNK_SIZE)
                if not chunk_bytes:
                    return # End of dataset
                
                # Convert bytes -> Unsigned Shorts (H) -> Ints
                # Matches array.array('H') from tokenization
                new_tokens = array.array('H')
                new_tokens.frombytes(chunk_bytes)
                buffer.extend(new_tokens)
                
                # Update offset for next checkpoint
                file_offset += len(chunk_bytes)

            # Yield one sample
            # (In production, you would batch this logic)
            seq = [buffer[i] for i in range(CONTEXT_LEN + 1)]
            
            # Remove only the first token to slide the window
            buffer.popleft()
            
            x = seq[:-1]
            y = seq[1:]
            
            yield x, y, file_offset, buffer

# ======================
# EXAMPLE RUN
# ======================

if __name__ == "__main__":
    print("[+] Starting training stream...")
    
    try:
        stream = training_stream()
        
        # Simulate training loop
        for step, (x, y, offset, buffer) in enumerate(stream):
            
            # Print first batch only to verify data looks correct
            if step == 0:
                print(f"Sample X (first 10): {x[:10]}")
                print(f"Sample Y (first 10): {y[:10]}")

            if step % 1000 == 0:
                print(f"Step {step} | Offset {offset} bytes")
                save_checkpoint(offset, buffer)

            # 🔥 Optimizer step here

    except KeyboardInterrupt:
        print("\n[!] Interrupted — Checkpoint saved.")
        # Save exact state on exit
        save_checkpoint(offset, buffer)