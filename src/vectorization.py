"""Vectorization helpers: TF-IDF/SVD and local Qwen embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


DEFAULT_QWEN_BASE_URL = "http://localhost:6620/v1"
DEFAULT_QWEN_MODEL = "qwen3-embedding"


@dataclass
class TfidfArtifacts:
    matrix: object
    vectors: np.ndarray
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD


@dataclass
class EmbeddingResult:
    vectors: np.ndarray | None
    source: str
    available: bool
    message: str


def build_tfidf_svd(
    texts: Sequence[str],
    max_features: int = 5_000,
    min_df: int = 5,
    max_df: float = 0.8,
    ngram_range: tuple[int, int] = (1, 2),
    n_components: int = 100,
    random_state: int = 42,
) -> TfidfArtifacts:
    """Build TF-IDF and reduce it with TruncatedSVD."""

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    max_components = max(2, min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    vectors = svd.fit_transform(tfidf_matrix)
    vectors = normalize(vectors)
    return TfidfArtifacts(matrix=tfidf_matrix, vectors=vectors, vectorizer=vectorizer, svd=svd)


def qwen_settings_from_env() -> tuple[str, str]:
    """Read Qwen endpoint settings from environment variables."""

    base_url = os.getenv("QWEN_EMBEDDING_BASE_URL", DEFAULT_QWEN_BASE_URL).rstrip("/")
    model = os.getenv("QWEN_EMBEDDING_MODEL", DEFAULT_QWEN_MODEL)
    return base_url, model


def check_qwen_available(base_url: str | None = None, model: str | None = None, timeout: int = 5) -> tuple[bool, str]:
    """Check whether the local Qwen embedding endpoint is reachable."""

    default_base_url, default_model = qwen_settings_from_env()
    base_url = (base_url or default_base_url).rstrip("/")
    model = model or default_model

    try:
        response = requests.get(f"{base_url}/models", timeout=timeout)
        response.raise_for_status()
        model_ids = [item.get("id") for item in response.json().get("data", [])]
    except Exception as exc:
        return False, f"Qwen endpoint is not available: {exc}"

    if model not in model_ids:
        return False, f"Qwen endpoint is reachable, but model {model!r} was not found. Available: {model_ids}"
    return True, f"Qwen endpoint is available with model {model!r}."


def encode_with_qwen(
    texts: Sequence[str],
    base_url: str | None = None,
    model: str | None = None,
    batch_size: int = 64,
    timeout: int = 120,
    normalize_embeddings: bool = True,
    show_progress: bool = True,
) -> EmbeddingResult:
    """Encode texts through an OpenAI-compatible local Qwen embeddings endpoint."""

    default_base_url, default_model = qwen_settings_from_env()
    base_url = (base_url or default_base_url).rstrip("/")
    model = model or default_model

    available, message = check_qwen_available(base_url=base_url, model=model)
    if not available:
        return EmbeddingResult(vectors=None, source="qwen", available=False, message=message)

    embeddings: list[list[float]] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Qwen embeddings", unit="batch")

    try:
        for start in iterator:
            batch = [str(text) for text in texts[start : start + batch_size]]
            response = requests.post(
                f"{base_url}/embeddings",
                json={"model": model, "input": batch},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            batch_vectors = sorted(payload["data"], key=lambda item: item["index"])
            embeddings.extend(item["embedding"] for item in batch_vectors)
    except Exception as exc:
        return EmbeddingResult(vectors=None, source="qwen", available=False, message=f"Qwen embedding failed: {exc}")

    vectors = np.asarray(embeddings, dtype=np.float32)
    if normalize_embeddings:
        vectors = normalize(vectors)
    return EmbeddingResult(vectors=vectors, source="qwen", available=True, message=message)
