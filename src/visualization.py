"""Visualization helpers for the clustering notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ensure_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def compute_pca_2d(X, random_state: int = 42):
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X)


def compute_tsne_2d(X, random_state: int = 42, perplexity: int = 30):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    return tsne.fit_transform(X)


def compute_umap_2d(X, random_state: int = 42, metric: str = "cosine"):
    import umap

    reducer = umap.UMAP(n_components=2, random_state=random_state, metric=metric)
    return reducer.fit_transform(X)


def plot_clusters_2d(
    coords,
    labels,
    cluster_summary: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
):
    """Plot cluster scatter with numeric labels near cluster centers."""

    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "cluster": labels})
    plot_df["cluster"] = plot_df["cluster"].astype(str)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="cluster",
        palette="tab20",
        s=16,
        linewidth=0,
        alpha=0.75,
        ax=ax,
    )

    centers = plot_df.groupby("cluster")[["x", "y"]].mean()
    name_map = dict(zip(cluster_summary["cluster_id"].astype(str), cluster_summary["name"]))
    for cluster_id, row in centers.iterrows():
        label = f"{cluster_id}: {name_map.get(cluster_id, 'noise')}"
        ax.text(row["x"], row["y"], label, fontsize=9, weight="bold")

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_cluster_sizes(cluster_summary: pd.DataFrame, output_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=cluster_summary, x="cluster_id", y="size", color="#4C78A8", ax=ax)
    ax.set_title("Размеры итоговых кластеров")
    ax.set_xlabel("cluster_id")
    ax.set_ylabel("size")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_length_distribution(df: pd.DataFrame, output_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="cluster", y="char_len", color="#72B7B2", showfliers=False, ax=ax)
    ax.set_title("Распределение длины промптов по кластерам")
    ax.set_xlabel("cluster")
    ax.set_ylabel("characters")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, ax


def plot_feature_heatmap(df: pd.DataFrame, output_path: str | Path | None = None):
    feature_columns = ["char_len", "word_len", "line_count", "uppercase_ratio", "special_char_count"]
    means = df.groupby("cluster")[feature_columns].mean()
    normalized = (means - means.mean()) / means.std(ddof=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(normalized, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Средние текстовые признаки по кластерам, z-score")
    ax.set_xlabel("feature")
    ax.set_ylabel("cluster")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, ax
