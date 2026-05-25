"""Cluster interpretation helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


LEVEL_INTERPRETATIONS = {
    0: {
        "name": "Прямое игнорирование инструкций в working-from-space боте",
        "plot_label": "Direct override",
        "typical_pattern": "Короткие команды ignore previous instructions внутри простого позитивного бота.",
        "outliers_note": "Выбросы используют более косвенные формулировки: вычисления, удаления букв или сокращенные маркеры.",
        "interpretation": "Базовый кластер прямого jailbreak: пользователь явно просит игнорировать предыдущие инструкции и вывести запрещенную фразу.",
    },
    1: {
        "name": "Jailbreaks в English-to-Spanish переводчике",
        "plot_label": "ES translator",
        "typical_pattern": "Payload вводится в переводческую задачу, часто через кавычки, HTML или условные инструкции.",
        "outliers_note": "Выбросы содержат длинные неанглийские тексты или метаописания ChatGPT вместо коротких команд.",
        "interpretation": "Кластер соответствует переводчику English-to-Spanish: атаки пытаются заставить модель не переводить, а вывести целевую фразу.",
    },
    2: {
        "name": "Secret-key extraction через переводческую задачу",
        "plot_label": "Secret key",
        "typical_pattern": "Шаблон содержит secret key и запрет раскрытия, а пользователь пытается вытащить ключ через перевод.",
        "outliers_note": "Выбросы используют более агрессивные IGNORE-инструкции или нестандартную формулировку запроса.",
        "interpretation": "Кластер объединяет атаки на скрытую переменную: пользователь пытается заставить модель раскрыть secret key несмотря на запрет.",
    },
    3: {
        "name": "Мультиязычные переводческие jailbreak-команды",
        "plot_label": "Translate mix",
        "typical_pattern": "Короткие payloads встроены в набор примеров перевода с английского на испанский.",
        "outliers_note": "Дальние объекты используют base64 или добавляют чужой QA-контекст поверх переводческого шаблона.",
        "interpretation": "Кластер отражает уровень с мультиязычным переводческим шаблоном: атака пытается встроить целевую фразу рядом с легитимными примерами перевода.",
    },
    4: {
        "name": "Инъекции в search-engine / QA задачу",
        "plot_label": "Search QA",
        "typical_pattern": "Атака вставляется в поле вопроса перед инструкцией search engine.",
        "outliers_note": "Выбросы чаще используют base64, несколько вопросов или явное переопределение правил.",
        "interpretation": "Кластер соответствует QA/search-engine сценарию: пользователь пытается заставить систему ответить payload вместо ответа по документу.",
    },
    5: {
        "name": "Инъекции в writing-feedback ассистента",
        "plot_label": "Writing feedback",
        "typical_pattern": "Атака находится внутри текста, который ассистент должен оценить как writing/grammar feedback.",
        "outliers_note": "Дальние объекты отличаются длиной или дополнительными инструкциями поверх feedback task.",
        "interpretation": "Хороший пример indirect prompt injection: вредная инструкция подается как пользовательский текст для анализа, а не как явная команда ассистенту.",
    },
    6: {
        "name": "Инъекции в задачу генерации short story",
        "plot_label": "Short story",
        "typical_pattern": "Payload помещается в bracketed sentence, из которого нужно написать short story во втором лице.",
        "outliers_note": "Дальние объекты маскируют payload под обычный story prompt или заменяют целевую фразу похожими словами.",
        "interpretation": "Кластер отражает творческий wrapper: атака спрятана внутри входной sentence, которую модель должна развить в рассказ.",
    },
    7: {
        "name": "Инъекции в задачу генерации названия фильма",
        "plot_label": "Movie title",
        "typical_pattern": "Payload помещается в поле sentence, после чего модель должна сгенерировать movie title.",
        "outliers_note": "Выбросы длиннее и маскируют payload через narrative, encoding или псевдофункции.",
        "interpretation": "Кластер объединяет атаки на шаблон генерации названия фильма, где forbidden words и формат задания обходятся через пустые строки, склейки и дополнительные правила.",
    },
    8: {
        "name": "Длинные roleplay / narrative jailbreaks",
        "plot_label": "Roleplay",
        "typical_pattern": "Длинные roleplay истории с fictional world, Sigma/UCAR и переопределением поведения модели.",
        "outliers_note": "Дальние объекты отличаются длиной, количеством вложенных правил или переходят к более короткому структурному формату.",
        "interpretation": "Кластер объединяет narrative jailbreaks, где атака маскируется под художественный или ролевой сценарий.",
    },
    9: {
        "name": "Обфусцированные slash-separated payloads",
        "plot_label": "Slash obfuscation",
        "typical_pattern": "Целевая фраза разбивается слэшами и unicode-символами.",
        "outliers_note": "Дальние объекты содержат длинные тексты на других языках или полноценные сценарии вместо короткого payload.",
        "interpretation": "Самый яркий кластер обфускации: модель видит повторяющийся шаблон с символами, слэшами и необычной типографикой.",
    },
    10: {
        "name": "Короткие emoji / nature-mapping prompts",
        "plot_label": "Emoji mapping",
        "typical_pattern": "Короткие prompts, где модель должна сопоставить emoji с nature emojis.",
        "outliers_note": "Выбросы могут быть ближе к roleplay-кластеру, если содержат длинные дополнительные инструкции.",
        "interpretation": "Кластер отделяет специфический уровень с короткими emoji-задачами; в Qwen-прогоне он мог смешиваться с длинными roleplay prompts.",
    },
}


# Backward-compatible alias for older notebooks/imports.
CLUSTER_INTERPRETATIONS = LEVEL_INTERPRETATIONS


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
        level_mode = int(part["level"].mode().iat[0]) if "level" in part.columns and not part["level"].mode().empty else np.nan
        manual = LEVEL_INTERPRETATIONS.get(level_mode, {})
        plot_label = f"{cluster_id} {manual.get('plot_label', heuristic_cluster_name(top_term_strings, feature_means))}"
        rows.append(
            {
                "cluster_id": cluster_id,
                "level_mode": level_mode,
                "name": manual.get("name", heuristic_cluster_name(top_term_strings, feature_means)),
                "plot_label": plot_label,
                "size": int(len(part)),
                "top_terms": ", ".join(top_term_strings),
                "typical_examples_summary": " | ".join(typical["text_raw"].map(short_text).head(3)),
                "outliers_summary": " | ".join(outliers["text_raw"].map(short_text).head(2)),
                "typical_pattern": manual.get("typical_pattern", ""),
                "outliers_note": manual.get("outliers_note", ""),
                "interpretation": manual.get("interpretation", ""),
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


def explain_cluster_outliers(
    df_with_clusters: pd.DataFrame,
    cluster_summary: pd.DataFrame | None = None,
    n_outliers: int = 3,
) -> pd.DataFrame:
    """Create short human-readable explanations for farthest examples in every cluster."""

    rows = []
    summary_by_cluster = {}
    if cluster_summary is not None and not cluster_summary.empty:
        summary_by_cluster = {
            int(row["cluster_id"]): row
            for _, row in cluster_summary.iterrows()
            if not pd.isna(row.get("cluster_id"))
        }

    for cluster_id in sorted(df_with_clusters["cluster"].unique()):
        if cluster_id == -1:
            continue

        part = df_with_clusters[df_with_clusters["cluster"] == cluster_id].copy()
        _typical, outliers = get_cluster_examples(df_with_clusters, cluster_id, n_typical=1, n_outliers=n_outliers)
        medians = part[["char_len", "word_len", "line_count", "special_char_count"]].median(numeric_only=True)
        q1 = part[["char_len", "word_len", "line_count", "special_char_count"]].quantile(0.25, numeric_only=True)
        q3 = part[["char_len", "word_len", "line_count", "special_char_count"]].quantile(0.75, numeric_only=True)
        iqr = (q3 - q1).replace(0, 1)
        level_mode = int(part["level"].mode().iat[0]) if "level" in part.columns and not part["level"].mode().empty else None
        summary_row = summary_by_cluster.get(int(cluster_id))
        cluster_note = ""
        if summary_row is not None:
            cluster_note = str(summary_row.get("outliers_note", "") or "")

        for rank, (_idx, row) in enumerate(outliers.iterrows(), start=1):
            reasons = []
            if row.get("char_len", 0) > medians.get("char_len", 0) + 1.5 * iqr.get("char_len", 1):
                reasons.append("заметно длиннее ядра кластера")
            elif row.get("char_len", 0) < max(1, medians.get("char_len", 0) - 1.5 * iqr.get("char_len", 1)):
                reasons.append("заметно короче типичных объектов")

            if row.get("special_char_count", 0) > medians.get("special_char_count", 0) + 1.5 * iqr.get("special_char_count", 1):
                reasons.append("содержит больше спецсимволов или форматной обфускации")

            if row.get("line_count", 0) > medians.get("line_count", 0) + 1.5 * iqr.get("line_count", 1):
                reasons.append("имеет больше строк и вложенной структуры")

            if level_mode is not None and "level" in row and not pd.isna(row["level"]) and int(row["level"]) != level_mode:
                reasons.append(f"пришел из level {int(row['level'])}, тогда как ядро кластера связано с level {level_mode}")

            text_lower = str(row.get("text_raw", "")).lower()
            if "base64" in text_lower or "ignore" in text_lower or "system" in text_lower:
                reasons.append("содержит явные мета-инструкции или маркеры обхода")

            if not reasons:
                reasons.append("отличается формулировкой в embedding-пространстве, но сохраняет общий шаблон кластера")

            explanation = "; ".join(dict.fromkeys(reasons))
            if cluster_note:
                explanation = f"{explanation}. Общая заметка по кластеру: {cluster_note}"

            rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "outlier_rank": rank,
                    "level": int(row["level"]) if "level" in row and not pd.isna(row["level"]) else np.nan,
                    "distance_to_centroid": round(float(row.get("distance_to_centroid", np.nan)), 4),
                    "char_len": int(row.get("char_len", 0)),
                    "word_len": int(row.get("word_len", 0)),
                    "text_preview": short_text(str(row.get("text_raw", "")), width=260),
                    "outlier_explanation": explanation,
                }
            )

    return pd.DataFrame(rows)
