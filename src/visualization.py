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
    label_column: str | None = None,
):
    """Plot cluster scatter with compact labels near cluster centers."""

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

    centers = []
    for cluster_id, part in plot_df.groupby("cluster"):
        median = part[["x", "y"]].median()
        distances = (part["x"] - median["x"]) ** 2 + (part["y"] - median["y"]) ** 2
        representative = part.loc[distances.idxmin()]
        centers.append({"cluster": cluster_id, "x": representative["x"], "y": representative["y"]})
    centers = pd.DataFrame(centers).set_index("cluster")
    if label_column is None:
        label_column = "plot_label" if "plot_label" in cluster_summary.columns else "name"
    name_map = dict(zip(cluster_summary["cluster_id"].astype(str), cluster_summary[label_column]))

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_margin = (x_max - x_min) * 0.03
    y_margin = (y_max - y_min) * 0.03
    x_mid = (x_min + x_max) / 2

    for cluster_id, row in centers.iterrows():
        label_value = str(name_map.get(cluster_id, "noise"))
        label = label_value if label_value.startswith(f"{cluster_id} ") else f"{cluster_id}: {label_value}"
        x = min(max(row["x"], x_min + x_margin), x_max - x_margin)
        y = min(max(row["y"], y_min + y_margin), y_max - y_margin)
        ha = "right" if x > x_mid else "left"
        ax.text(
            x,
            y,
            label,
            fontsize=8,
            weight="bold",
            ha=ha,
            va="center",
            clip_on=True,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, ax


def _scatter_with_center_labels(
    ax,
    plot_df: pd.DataFrame,
    label_column: str,
    title: str,
    legend_title: str,
    label_map: dict | None = None,
):
    """Draw one labeled 2D scatter plot on an existing axis."""

    plot_df = plot_df.copy()
    plot_df[label_column] = plot_df[label_column].astype(str)
    label_map = {str(key): value for key, value in (label_map or {}).items()}

    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue=label_column,
        palette="tab20",
        s=14,
        linewidth=0,
        alpha=0.72,
        ax=ax,
    )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_margin = (x_max - x_min) * 0.03
    y_margin = (y_max - y_min) * 0.03
    x_mid = (x_min + x_max) / 2

    for label_value, part in plot_df.groupby(label_column):
        median = part[["x", "y"]].median()
        distances = (part["x"] - median["x"]) ** 2 + (part["y"] - median["y"]) ** 2
        representative = part.loc[distances.idxmin()]
        text = str(label_map.get(label_value, label_value))
        x = min(max(representative["x"], x_min + x_margin), x_max - x_margin)
        y = min(max(representative["y"], y_min + y_margin), y_max - y_margin)
        ha = "right" if x > x_mid else "left"
        ax.text(
            x,
            y,
            text,
            fontsize=7,
            weight="bold",
            ha=ha,
            va="center",
            clip_on=True,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title_fontsize=8)


def plot_cluster_level_comparison_2d(
    coords,
    cluster_labels,
    level_labels,
    cluster_summary: pd.DataFrame,
    title: str,
    output_path: str | Path | None = None,
    level_label_map: dict | None = None,
):
    """Plot the same 2D map colored by discovered clusters and by HackAPrompt levels."""

    plot_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": cluster_labels,
            "level": level_labels,
        }
    )
    cluster_label_map = dict(zip(cluster_summary["cluster_id"].astype(str), cluster_summary["plot_label"]))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    _scatter_with_center_labels(
        axes[0],
        plot_df,
        label_column="cluster",
        title="Discovered clusters",
        legend_title="cluster",
        label_map=cluster_label_map,
    )
    _scatter_with_center_labels(
        axes[1],
        plot_df,
        label_column="level",
        title="Original HackAPrompt levels",
        legend_title="level",
        label_map=level_label_map,
    )

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig, axes


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
