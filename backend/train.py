"""
train.py — Semantic Embedding Index Builder
===========================================
2026 approach: replaces NLTK + Bag-of-Words + Keras NN entirely.

Pipeline:
  intents.json patterns
       ↓
  sentence-transformers (all-MiniLM-L6-v2)
       ↓
  L2-normalised 384-dim dense embeddings
       ↓
  semantic_index.pkl   ← single artefact used at inference time

Classification at inference = cosine similarity (dot product on
normalised vectors). No epochs, no overfitting, no .h5 deprecation.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────
# train.py lives in  <project_root>/src/  (or whichever sub-folder you use)
# Adjust the number of .parent calls to match your layout:
#   same folder as dataset/ and model/  → Path(__file__).resolve().parent
#   one level in (src/ etc.)            → Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "intents.json"
MODEL_DIR    = PROJECT_ROOT / "model"
INDEX_PATH   = MODEL_DIR / "semantic_index.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 22 MB, CPU-friendly, strong quality
BATCH_SIZE      = 64                    # increase if you have GPU / more RAM


def build_index() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load dataset ────────────────────────────────────────────────────
    print("\n[1/4] Loading intents dataset …")
    if not DATASET_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        intents = json.load(fh)

    intent_list = intents.get("intents", [])
    if not intent_list:
        sys.exit("ERROR: 'intents' key is empty in the dataset.")

    print(f"      Loaded {len(intent_list)} intent categories.")

    # ── 2. Collect all (pattern, tag) pairs ────────────────────────────────
    print("\n[2/4] Collecting patterns …")
    pattern_texts: list[str] = []
    pattern_tags:  list[str] = []

    for intent in intent_list:
        tag      = intent.get("tag", "").strip()
        patterns = intent.get("patterns", [])
        if not tag:
            print(f"      WARNING: intent with no tag skipped → {intent}")
            continue
        for p in patterns:
            cleaned = p.strip()
            if cleaned:
                pattern_texts.append(cleaned)
                pattern_tags.append(tag)

    if not pattern_texts:
        sys.exit("ERROR: no patterns found in the dataset.")

    all_classes = sorted(set(pattern_tags))
    print(f"      {len(pattern_texts)} patterns across {len(all_classes)} classes.")

    # ── 3. Encode ──────────────────────────────────────────────────────────
    print(f"\n[3/4] Loading encoder: {EMBEDDING_MODEL!r} …")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("      Encoding all patterns (this may take a moment on first run) …")
    embeddings: np.ndarray = encoder.encode(
        pattern_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise → cosine sim = dot product
        convert_to_numpy=True,
    )
    print(f"      Embedding matrix shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    # ── 4. Save index ──────────────────────────────────────────────────────
    print("\n[4/4] Saving semantic index …")
    index = {
        "embeddings":     embeddings,          # np.ndarray (N, 384)
        "pattern_tags":   pattern_tags,        # list[str]  length N
        "pattern_texts":  pattern_texts,       # list[str]  length N  (for debug)
        "classes":        all_classes,         # sorted unique tags
        "embedding_model": EMBEDDING_MODEL,
    }
    with open(INDEX_PATH, "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"      Saved → {INDEX_PATH}")
    print(f"\n{'=' * 60}")
    print("  INDEX BUILT. Run main.py to start the server.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  AI COACH — SEMANTIC INDEX BUILDER  (2026 stack)")
    print("=" * 60)
    build_index()