"""Text preprocessing and feature engineering."""

from __future__ import annotations

import re
import string

import pandas as pd


KEYWORDS = [
    "ignore",
    "system",
    "instruction",
    "developer",
    "previous",
    "secret",
    "password",
    "role",
    "admin",
    "jailbreak",
]


def normalize_for_tfidf(text: str) -> str:
    """Light normalization for TF-IDF without destroying attack-specific structure."""

    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def uppercase_ratio(text: str) -> float:
    """Return uppercase-letter share among alphabetic characters."""

    letters = [char for char in str(text) if char.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters)


def special_char_count(text: str) -> int:
    """Count punctuation and non-alphanumeric symbols."""

    punctuation = set(string.punctuation)
    return sum(1 for char in str(text) if char in punctuation or (not char.isalnum() and not char.isspace()))


def add_text_features(df: pd.DataFrame, text_column: str = "text_raw") -> pd.DataFrame:
    """Add interpretable text features used in cluster analysis."""

    result = df.copy()
    texts = result[text_column].astype(str)
    lower_texts = texts.str.lower()

    result["char_len"] = texts.str.len()
    result["word_len"] = texts.str.split().str.len()
    result["line_count"] = texts.str.count(r"\n") + 1
    result["uppercase_ratio"] = texts.map(uppercase_ratio)
    result["special_char_count"] = texts.map(special_char_count)

    for keyword in KEYWORDS:
        result[f"has_{keyword}"] = lower_texts.str.contains(rf"\b{re.escape(keyword)}\b", regex=True)

    return result


def preprocess_prompts(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Clean prompt rows conservatively and add raw/TF-IDF text columns."""

    result = df.copy()
    result["text_raw"] = result[text_column].astype(str)
    result["text_raw"] = result["text_raw"].str.strip()
    result = result[result["text_raw"].str.len() > 0].copy()
    result = result.drop_duplicates(subset=["text_raw"]).reset_index(drop=True)
    result["text_tfidf"] = result["text_raw"].map(normalize_for_tfidf)
    result = add_text_features(result, text_column="text_raw")
    return result
