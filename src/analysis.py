"""Cluster interpretation helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def attach_kmeans_distances(df: pd.DataFrame, X: np.ndarray, labels: Iterable[int], kmeans_model) -> pd.DataFrame:
    """Attach cluster labels and distance to assigned K-Means centroid."""

    result = df.copy()
    labels_array = np.asarray(labels)
    distances = pairwise_distances(X, kmeans_model.cluster_centers_, metric="cosine")
    assigned_distances = distances[np.arange(len(labels_array)), labels_array]
    result["cluster"] = labels_array
    result["distance_to_centroid"] = assigned_distances
    return result


def get_cluster_examples(
    df: pd.DataFrame,
    cluster_id: int,
    n_typical: int = 10,
    n_outliers: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return central and farthest examples inside a cluster."""

    part = df[df["cluster"] == cluster_id].copy()
    typical = part.sort_values("distance_to_centroid").head(n_typical)
    outliers = part.sort_values("distance_to_centroid", ascending=False).head(n_outliers)
    return typical, outliers


def top_terms_by_cluster(tfidf_matrix, labels: Iterable[int], vectorizer, cluster_id: int, n: int = 20) -> list[tuple[str, float]]:
    """Return top mean TF-IDF terms for one cluster."""

    labels_array = np.asarray(labels)
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_mean = tfidf_matrix[labels_array == cluster_id].mean(axis=0)
    scores = np.asarray(cluster_mean).ravel()
    top_idx = scores.argsort()[::-1][:n]
    return list(zip(terms[top_idx], scores[top_idx]))


def short_text(text: str, width: int = 220) -> str:
    """Compact a prompt for tables and Markdown previews."""

    compact = " ".join(str(text).split())
    return textwrap.shorten(compact, width=width, placeholder="...")


def heuristic_cluster_name(top_terms: list[str], feature_means: pd.Series) -> str:
    """Create a draft human-readable cluster name from terms and feature statistics."""

    term_set = set(top_terms)
    if {"pretend", "role"} & term_set or {"character", "act"} & term_set:
        return "Roleplay / pretend-mode jailbreaks"
    if {"system", "prompt"} & term_set or {"secret", "password"} & term_set:
        return "Запросы на раскрытие скрытых правил и system prompt"
    if {"ignore", "previous", "instruction"} & term_set:
        return "Прямое игнорирование инструкций"
    if feature_means.get("char_len", 0) > 600:
        return "Длинные социально-инженерные атаки"
    if feature_means.get("special_char_count", 0) > 80:
        return "Обфусцированные или структурно странные промпты"
    if {"admin", "developer"} & term_set:
        return "Промпты с техническими маркерами system/admin/developer"
    return "Смешанные jailbreak-команды"


def homogeneity_label(distances: pd.Series) -> str:
    """Estimate cluster homogeneity from centroid-distance dispersion."""

    if distances.empty:
        return "неизвестная"
    spread = distances.quantile(0.75) - distances.quantile(0.25)
    median = distances.median()
    if spread < 0.08 and median < 0.35:
        return "высокая"
    if spread < 0.15:
        return "средняя"
    return "низкая / размытая"


def build_cluster_summary(
    df_with_clusters: pd.DataFrame,
    tfidf_matrix,
    vectorizer,
    labels: Iterable[int],
    n_terms: int = 12,
) -> pd.DataFrame:
    """Build the final interpretation table for all clusters."""

    rows = []
    labels_array = np.asarray(labels)
    cluster_ids = sorted(cluster_id for cluster_id in np.unique(labels_array) if cluster_id != -1)

    feature_columns = [
        "char_len",
        "word_len",
        "line_count",
        "uppercase_ratio",
        "special_char_count",
    ]

    for cluster_id in cluster_ids:
        part = df_with_clusters[df_with_clusters["cluster"] == cluster_id]
        terms = top_terms_by_cluster(tfidf_matrix, labels_array, vectorizer, cluster_id, n=n_terms)
        top_term_strings = [term for term, _score in terms]
        typical, outliers = get_cluster_examples(df_with_clusters, cluster_id)
        feature_means = part[feature_columns].mean(numeric_only=True)
        rows.append(
            {
                "cluster_id": cluster_id,
                "name": heuristic_cluster_name(top_term_strings, feature_means),
                "size": int(len(part)),
                "top_terms": ", ".join(top_term_strings),
                "typical_examples_summary": " | ".join(typical["text_raw"].map(short_text).head(3)),
                "outliers_summary": " | ".join(outliers["text_raw"].map(short_text).head(2)),
                "homogeneity": homogeneity_label(part["distance_to_centroid"]),
                "mean_char_len": round(float(feature_means.get("char_len", 0)), 2),
                "mean_word_len": round(float(feature_means.get("word_len", 0)), 2),
            }
        )

    return pd.DataFrame(rows)


def save_cluster_examples(
    df_with_clusters: pd.DataFrame,
    output_dir: str | Path,
    n_typical: int = 10,
    n_outliers: int = 3,
) -> None:
    """Save central and outlier examples for each cluster as Markdown files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for cluster_id in sorted(df_with_clusters["cluster"].unique()):
        if cluster_id == -1:
            continue
        typical, outliers = get_cluster_examples(
            df_with_clusters,
            cluster_id=cluster_id,
            n_typical=n_typical,
            n_outliers=n_outliers,
        )

        lines = [f"# Cluster {cluster_id}", "", "## Typical examples", ""]
        for idx, row in typical.iterrows():
            lines.append(f"### Example {idx}")
            lines.append("")
            lines.append(str(row["text_raw"]))
            lines.append("")

        lines.extend(["## Outlier examples", ""])
        for idx, row in outliers.iterrows():
            lines.append(f"### Example {idx}")
            lines.append("")
            lines.append(str(row["text_raw"]))
            lines.append("")

        (output_path / f"cluster_{cluster_id:02d}.md").write_text("\n".join(lines), encoding="utf-8")
