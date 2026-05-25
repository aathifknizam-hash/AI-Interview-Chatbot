"""
preprocessing.py — NLP Preprocessing Module
=============================================
This module handles all text preprocessing steps needed before
training or predicting with the chatbot model.

Why preprocessing?
------------------
Raw text cannot be fed directly into a neural network.
We must convert words into numbers (vectors) the model can process.

Steps:
  1. Tokenize  → split sentence into words
  2. Lowercase → normalize case
  3. Stem      → reduce words to their root form
  4. Bag of Words → turn words into a binary vector
"""

import re
import nltk
import numpy as np
from nltk.stem import PorterStemmer

# Download required NLTK data files (run once)
# 'punkt' is the tokenizer model; 'punkt_tab' is the newer format
# These downloads may fail in restricted network environments.
# In that case, we fall back to a simple regex tokenizer (see below).
_nltk_available = False
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    # Quick test to verify download succeeded
    nltk.word_tokenize("test")
    _nltk_available = True
except Exception:
    _nltk_available = False

# Initialize the Porter Stemmer
# PorterStemmer is a classic, rule-based stemming algorithm
# It removes common suffixes to find the root of a word
stemmer = PorterStemmer()


def _simple_tokenize(sentence: str) -> list:
    """
    Fallback tokenizer using regular expressions.
    Splits on word boundaries, keeping punctuation as separate tokens.
    Used when NLTK punkt data is unavailable.
    """
    # Match words (including apostrophes) or individual punctuation symbols
    return re.findall(r"\b\w[\w']*\b|[^\w\s]", sentence)


def tokenize(sentence: str) -> list:
    """
    STEP 1 — Tokenization
    ----------------------
    Splits a sentence into individual words (tokens).

    Example:
        "Hello, how are you?" → ["Hello", ",", "how", "are", "you", "?"]

    Why?
        Neural networks process individual tokens, not full sentences.
        Tokenization is always the first NLP step.

    Args:
        sentence (str): Raw input sentence from user or dataset.

    Returns:
        list: A list of word tokens (strings).
    """
    if _nltk_available:
        return nltk.word_tokenize(sentence)
    return _simple_tokenize(sentence)


def lowercase(tokens: list) -> list:
    """
    STEP 2 — Lowercase Conversion
    --------------------------------
    Converts all tokens to lowercase.

    Example:
        ["Hello", "WORLD"] → ["hello", "world"]

    Why?
        "Hello" and "hello" are the same word but different strings.
        Lowercasing ensures the model treats them as identical.
        This reduces vocabulary size and improves generalization.

    Args:
        tokens (list): List of word tokens.

    Returns:
        list: Lowercased list of tokens.
    """
    return [token.lower() for token in tokens]


def stem(tokens: list) -> list:
    """
    STEP 3 — Stemming
    -------------------
    Reduces each token to its root/base form using PorterStemmer.

    Example:
        ["running", "runner", "runs"] → ["run", "runner", "run"]

    Why Stemming?
        "running", "runs", "ran" all mean similar things (the act of running).
        Stemming maps them to the same root so the model sees them as equivalent.
        This reduces vocabulary size and helps the model generalize better.

        Note: Stemming is a heuristic process — it may not always produce
        real words (e.g., "studies" → "studi"), but that's fine for our purpose.

    Args:
        tokens (list): List of lowercased tokens.

    Returns:
        list: Stemmed list of tokens.
    """
    return [stemmer.stem(token) for token in tokens]


def preprocess(sentence: str) -> list:
    """
    COMBINED PIPELINE — Tokenize + Lowercase + Stem
    -------------------------------------------------
    Runs the full preprocessing pipeline on a raw sentence.

    Example:
        "Hello, How are You running?" 
        → tokenize: ["Hello", ",", "How", "are", "You", "running", "?"]
        → lowercase: ["hello", ",", "how", "are", "you", "running", "?"]
        → stem:      ["hello", ",", "how", "are", "you", "run", "?"]

    Args:
        sentence (str): Raw input text.

    Returns:
        list: Preprocessed list of stemmed tokens.
    """
    tokens = tokenize(sentence)
    tokens = lowercase(tokens)
    tokens = stem(tokens)
    return tokens


def bag_of_words(preprocessed_tokens: list, vocabulary: list) -> np.ndarray:
    """
    STEP 4 — Bag of Words (BoW)
    ----------------------------
    Converts a list of tokens into a fixed-length binary vector
    based on the known vocabulary.

    How it works:
        - vocabulary = ["hello", "how", "are", "you", "bye", "thanks"]
        - tokens = ["hello", "how"]
        - BoW vector = [1, 1, 0, 0, 0, 0]
          (1 where the word IS present, 0 where it is NOT)

    Why Bag of Words?
        Neural networks require fixed-size numerical inputs.
        BoW gives us a consistent vector size = len(vocabulary).
        It tells the model WHICH words appear in the input,
        regardless of order or frequency.

        Limitation: BoW ignores word order (hence "bag"),
        but for intent classification this works very well.

    Args:
        preprocessed_tokens (list): Stemmed tokens from the sentence.
        vocabulary (list): Full list of known stemmed words from training data.

    Returns:
        np.ndarray: Binary numpy array of shape (len(vocabulary),)
                    1.0 if word is present, 0.0 if not.
    """
    # Create a zero vector with length = vocabulary size
    bow = np.zeros(len(vocabulary), dtype=np.float32)

    # Mark 1.0 for every word in vocabulary that appears in our tokens
    for idx, word in enumerate(vocabulary):
        if word in preprocessed_tokens:
            bow[idx] = 1.0

    return bow


# ─────────────────────────────────────────────
#  Quick test (run this file directly to test)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("NLP PREPROCESSING TEST")
    print("=" * 50)

    test_sentence = "Hello! How are you running today?"
    print(f"\nOriginal sentence : {test_sentence}")

    tokens = tokenize(test_sentence)
    print(f"After tokenize    : {tokens}")

    lower = lowercase(tokens)
    print(f"After lowercase   : {lower}")

    stemmed = stem(lower)
    print(f"After stemming    : {stemmed}")

    # Bag of Words test
    vocab = ["hello", "how", "are", "you", "run", "today", "bye", "thank"]
    processed = preprocess(test_sentence)
    bow = bag_of_words(processed, vocab)
    print(f"\nVocabulary        : {vocab}")
    print(f"Processed tokens  : {processed}")
    print(f"Bag of Words      : {bow}")
    print("=" * 50)