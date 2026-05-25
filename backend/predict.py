"""
predict.py — Semantic Inference Engine
=======================================
2026 approach: sentence-transformer cosine similarity, no Keras, no TensorFlow.

Query → encode → dot-product vs stored embeddings → highest score → response.
All heavy objects are loaded once (singletons) and reused across requests.
"""

import json
import logging
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Paths (same convention as train.py) ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "intents.json"
INDEX_PATH   = PROJECT_ROOT / "model" / "semantic_index.pkl"

# ── Tuning ─────────────────────────────────────────────────────────────────
# Cosine similarity range: 1.0 = identical, 0.0 = orthogonal.
# Raise this if you get too many unrelated matches;
# lower it if too many legitimate queries fall through to fallback.
CONFIDENCE_THRESHOLD: float = 0.42

FALLBACK_RESPONSES: list[str] = [
    "I'm not certain about that one — could you rephrase or be more specific?",
    "That topic is outside my current scope. Try asking about Precision vs Recall, "
    "the Bias-Variance Tradeoff, or say 'mock interview' to get started!",
    "I didn't quite catch that. Ask me about Deep Learning, NLP, Statistics, "
    "MLOps, or career advice.",
]

# ── Singletons (populated by load_artifacts) ──────────────────────────────
_index:        Optional[dict]                 = None
_encoder:      Optional[SentenceTransformer]  = None
_intents_data: Optional[dict]                 = None


def load_artifacts() -> None:
    """
    Load the semantic index and dataset into memory.
    Safe to call multiple times — only loads once.
    """
    global _index, _encoder, _intents_data

    if _index is not None:
        return  # already loaded

    logger.info("Loading semantic index from %s …", INDEX_PATH)
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Semantic index not found at {INDEX_PATH}. "
            "Run train.py first to build the index."
        )

    with open(INDEX_PATH, "rb") as fh:
        _index = pickle.load(fh)

    _validate_index(_index)

    model_name: str = _index["embedding_model"]
    logger.info("Loading encoder: %r …", model_name)
    _encoder = SentenceTransformer(model_name)

    logger.info("Loading intents dataset from %s …", DATASET_PATH)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")

    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        _intents_data = json.load(fh)

    logger.info(
        "Artifacts ready — %d patterns, %d classes.",
        len(_index["pattern_tags"]),
        len(_index["classes"]),
    )


def _validate_index(index: dict) -> None:
    """Raise ValueError if the index is missing expected keys or is misshapen."""
    required = {"embeddings", "pattern_tags", "pattern_texts", "classes", "embedding_model"}
    missing = required - index.keys()
    if missing:
        raise ValueError(f"Semantic index is missing keys: {missing}. Rebuild with train.py.")

    emb: np.ndarray = index["embeddings"]
    tags: list = index["pattern_tags"]
    if emb.ndim != 2:
        raise ValueError(f"Expected 2-D embedding matrix, got shape {emb.shape}.")
    if emb.shape[0] != len(tags):
        raise ValueError(
            f"Embedding count ({emb.shape[0]}) != tag count ({len(tags)}). Rebuild with train.py."
        )


def predict_intent(sentence: str) -> dict:
    """
    Encode *sentence* and return the best-matching intent.

    Returns
    -------
    dict with keys:
        intent          (str)   — matched tag
        confidence      (float) — cosine similarity in [0, 1]
        matched_pattern (str)   — the stored pattern closest to the query
    """
    if _index is None or _encoder is None:
        load_artifacts()

    # Encode and L2-normalise the query (same normalisation used at index build)
    query_vec: np.ndarray = _encoder.encode(
        sentence,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )                                      # shape: (384,)

    # Cosine similarity = dot product for L2-normalised vectors → shape: (N,)
    scores: np.ndarray = _index["embeddings"] @ query_vec

    best_idx:    int   = int(np.argmax(scores))
    confidence:  float = float(scores[best_idx])
    matched_tag: str   = _index["pattern_tags"][best_idx]
    matched_pat: str   = _index["pattern_texts"][best_idx]

    return {
        "intent":          matched_tag,
        "confidence":      confidence,
        "matched_pattern": matched_pat,
    }


def get_response(user_message: str) -> str:
    """
    Public API consumed by main.py.
    Returns a human-readable response string.
    """
    user_message = user_message.strip()
    if not user_message:
        return "Please type a question so I can help you prepare!"

    try:
        result = predict_intent(user_message)
        logger.debug(
            "Intent: %r  confidence: %.3f  pattern: %r",
            result["intent"],
            result["confidence"],
            result["matched_pattern"],
        )

        if result["confidence"] < CONFIDENCE_THRESHOLD:
            return random.choice(FALLBACK_RESPONSES)

        # Find the matching intent and pick a random response
        tag = result["intent"]
        for intent in _intents_data["intents"]:  # type: ignore[index]
            if intent.get("tag") == tag:
                responses = intent.get("responses", [])
                if responses:
                    return random.choice(responses)
                break

        # Tag found in index but not in intents.json (shouldn't happen after rebuild)
        logger.warning("Tag %r found in index but not in intents.json.", tag)
        return random.choice(FALLBACK_RESPONSES)

    except FileNotFoundError as exc:
        logger.error("Artifact missing: %s", exc)
        return "The model index is not available. Please ask the administrator to run train.py."
    except Exception as exc:
        logger.exception("Unexpected error in get_response: %s", exc)
        return "An unexpected error occurred. Please try again."


# ── Quick smoke-test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("  SEMANTIC INFERENCE — SMOKE TEST")
    print("=" * 60)

    load_artifacts()

    test_queries = [
        "Explain the bias variance tradeoff",
        "Give me a mock interview question",
        "What is data leakage?",
        "How does gradient descent work?",
        "Tell me about transformers in NLP",
        "What should my GitHub portfolio look like?",
        "asdfjkl qwerty nonsense",          # expected: fallback
    ]

    all_passed = True
    for q in test_queries:
        r = predict_intent(q)
        response = get_response(q)
        status = "✅" if r["confidence"] >= CONFIDENCE_THRESHOLD else "🔁 fallback"
        print(f"\n  Query    : {q}")
        print(f"  Intent   : {r['intent']}  ({r['confidence']:.3f})  {status}")
        print(f"  Pattern  : {r['matched_pattern']}")
        print(f"  Response : {response[:120]}…")

    sys.exit(0 if all_passed else 1)