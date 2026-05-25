"""Clustering model selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass
class ClusteringRun:
    """A fitted clustering run and its summary metrics."""

    name: str
    representation: str
    params: dict
    labels: np.ndarray
    model: object | None
    metrics: dict


def cluster_size_summary(labels: Iterable[int]) -> dict:
    """Return cluster-size summary excluding DBSCAN noise when present."""

    counts = pd.Series(labels).value_counts().sort_index()
    non_noise = counts[counts.index != -1]
    if non_noise.empty:
        return {
            "n_clusters": 0,
            "min_cluster_size": 0,
            "max_cluster_size": 0,
            "noise_ratio": float((pd.Series(labels) == -1).mean()),
        }
    return {
        "n_clusters": int(len(non_noise)),
        "min_cluster_size": int(non_noise.min()),
        "max_cluster_size": int(non_noise.max()),
        "noise_ratio": float((pd.Series(labels) == -1).mean()),
    }


def clustering_metrics(X: np.ndarray, labels: Iterable[int], ignore_noise: bool = False) -> dict:
    """Calculate standard metrics when they are well-defined."""

    labels_array = np.asarray(labels)
    if ignore_noise:
        mask = labels_array != -1
        X = X[mask]
        labels_array = labels_array[mask]

    unique_labels = np.unique(labels_array)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels_array):
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
            **cluster_size_summary(labels),
        }

    try:
        silhouette = silhouette_score(X, labels_array, metric="cosine")
    except Exception:
        silhouette = np.nan

    try:
        calinski = calinski_harabasz_score(X, labels_array)
    except Exception:
        calinski = np.nan

    try:
        davies = davies_bouldin_score(X, labels_array)
    except Exception:
        davies = np.nan

    return {
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        **cluster_size_summary(labels),
    }


def run_kmeans_grid(
    X: np.ndarray,
    representation: str,
    k_values: Iterable[int] = range(4, 16),
    random_state: int = 42,
) -> tuple[list[ClusteringRun], pd.DataFrame]:
    """Fit K-Means over several k values."""

    runs: list[ClusteringRun] = []
    rows: list[dict] = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        metrics = clustering_metrics(X, labels)
        params = {"k": k}
        run = ClusteringRun("K-Means", representation, params, labels, model, metrics)
        runs.append(run)
        rows.append({"method": run.name, "representation": representation, **params, **metrics})

    return runs, pd.DataFrame(rows)


def _fit_agglomerative(X: np.ndarray, n_clusters: int, linkage: str, metric: str):
    """Fit agglomerative clustering across sklearn metric/affinity API versions."""

    kwargs = {"n_clusters": n_clusters, "linkage": linkage}
    if linkage == "ward":
        kwargs["metric"] = "euclidean"
    else:
        kwargs["metric"] = metric

    try:
        model = AgglomerativeClustering(**kwargs)
    except TypeError:
        affinity = kwargs.pop("metric")
        kwargs["affinity"] = affinity
        model = AgglomerativeClustering(**kwargs)

    labels = model.fit_predict(X)
    return model, labels


def run_agglomerative_grid(
    X: np.ndarray,
    representation: str,
    n_clusters_values: Iterable[int] = range(4, 16),
) -> tuple[list[ClusteringRun], pd.DataFrame]:
    """Fit agglomerative clustering for several cluster counts and linkage strategies."""

    configs = []
    for n_clusters in n_clusters_values:
        configs.append((n_clusters, "ward", "euclidean"))
        configs.append((n_clusters, "average", "cosine"))
        configs.append((n_clusters, "complete", "cosine"))

    runs: list[ClusteringRun] = []
    rows: list[dict] = []
    for n_clusters, linkage, metric in configs:
        model, labels = _fit_agglomerative(X, n_clusters=n_clusters, linkage=linkage, metric=metric)
        metrics = clustering_metrics(X, labels)
        params = {"n_clusters": n_clusters, "linkage": linkage, "metric": metric}
        run = ClusteringRun("Agglomerative", representation, params, labels, model, metrics)
        runs.append(run)
        rows.append({"method": run.name, "representation": representation, **params, **metrics})

    return runs, pd.DataFrame(rows)


def run_dbscan_grid(
    X: np.ndarray,
    representation: str,
    eps_values: Iterable[float] = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
    min_samples_values: Iterable[int] = (5, 10, 20),
    metric: str = "cosine",
) -> tuple[list[ClusteringRun], pd.DataFrame]:
    """Fit DBSCAN over eps/min_samples values."""

    runs: list[ClusteringRun] = []
    rows: list[dict] = []

    for eps, min_samples in product(eps_values, min_samples_values):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = model.fit_predict(X)
        metrics = clustering_metrics(X, labels, ignore_noise=True)
        params = {"eps": eps, "min_samples": min_samples, "metric": metric}
        run = ClusteringRun("DBSCAN", representation, params, labels, model, metrics)
        runs.append(run)
        rows.append({"method": run.name, "representation": representation, **params, **metrics})

    return runs, pd.DataFrame(rows)


def choose_kmeans_run(runs: list[ClusteringRun], preferred_k: int | None = None) -> ClusteringRun:
    """Choose a K-Means run, prioritizing interpretability-friendly balanced solutions."""

    if preferred_k is not None:
        for run in runs:
            if run.params.get("k") == preferred_k:
                return run

    scored = []
    for run in runs:
        metrics = run.metrics
        silhouette = metrics.get("silhouette")
        if pd.isna(silhouette):
            continue
        size_ratio = metrics["max_cluster_size"] / max(metrics["min_cluster_size"], 1)
        balance_penalty = np.log1p(size_ratio)
        scored.append((silhouette - 0.02 * balance_penalty, run))

    if not scored:
        return runs[0]
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]
