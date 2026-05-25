# Code Explanation: HackAPrompt Clustering

Этот файл объясняет, как устроен код проекта: какие модули за что отвечают, как notebook собирает pipeline и какие артефакты появляются после запуска.

## Общая Архитектура

Проект разделен на две части:

1. `notebooks/hackaprompt_clustering.ipynb` — основной отчет и исполняемый pipeline.
2. `src/*.py` — повторяемая логика, вынесенная из notebook.

Такой подход удобен для финального проекта: notebook остается читаемым для сдачи, а вспомогательные функции не размазываются по ячейкам.

## Структура Папок

```text
.
├── notebooks/
│   └── hackaprompt_clustering.ipynb
├── src/
│   ├── analysis.py
│   ├── clustering.py
│   ├── data_loading.py
│   ├── env.py
│   ├── preprocessing.py
│   ├── vectorization.py
│   └── visualization.py
├── outputs/
│   ├── cluster_examples/
│   ├── figures/
│   └── tables/
├── embedding_cache/
├── README.md
├── RESULTS.md
├── CODE_EXPLANATION.md
└── PROJECT_EXPLANATION.md
```

## Notebook Flow

### 1. Настройка

Notebook задает:

```python
RANDOM_STATE = 42
SAMPLE_SIZE = 10000
METHOD_SAMPLE_SIZE = 5000
```

`SAMPLE_SIZE` управляет размером основной выборки. `METHOD_SAMPLE_SIZE` используется для тяжелых методов вроде Agglomerative и DBSCAN, чтобы не считать их на всех 9571 объектах.

Также notebook создает папки:

```python
outputs/figures
outputs/tables
outputs/cluster_examples
embedding_cache
```

### 2. Загрузка Данных

Функция:

```python
load_hackaprompt_dataframe(...)
```

находится в `src/data_loading.py`.

Она умеет:

- читать HackAPrompt с Hugging Face через `datasets`;
- использовать токен из `.env`;
- читать локальный файл через `HACKAPROMPT_LOCAL_FILE`;
- автоматически выбирать split;
- автоматически определять текстовую колонку;
- делать воспроизводимый sample.

Результат — `LoadedData`, где лежат:

- `dataframe`;
- имя датасета;
- имя split;
- имя текстовой колонки.

### 3. Предобработка

Функция:

```python
preprocess_prompts(...)
```

находится в `src/preprocessing.py`.

Что она делает:

- удаляет пустые тексты;
- удаляет дубликаты;
- сохраняет оригинальный текст в `text_raw`;
- создает мягко нормализованный текст для TF-IDF в `text_tfidf`;
- считает признаки:
  - `char_len`;
  - `word_len`;
  - `line_count`;
  - `uppercase_ratio`;
  - `special_char_count`;
  - keyword flags.

Почему текст не чистится агрессивно: в prompt injection важны регистр, спецсимволы, переносы строк, кавычки и структура. Если удалить это слишком рано, можно потерять саму технику атаки.

### 4. Векторизация

Файл: `src/vectorization.py`.

Есть два представления текста.

Первое:

```python
build_tfidf_svd(...)
```

Строит:

- `TfidfVectorizer`;
- sparse TF-IDF matrix;
- `TruncatedSVD`;
- нормализованные dense vectors.

TF-IDF+SVD нужен как:

- baseline;
- fallback;
- источник top terms для интерпретации кластеров.

Второе:

```python
check_qwen_available(...)
encode_with_qwen(...)
```

`check_qwen_available` проверяет:

```text
GET http://localhost:6620/v1/models
```

и ищет модель `qwen3-embedding`.

`encode_with_qwen` отправляет тексты батчами:

```text
POST http://localhost:6620/v1/embeddings
```

Если Qwen доступен, notebook использует Qwen embeddings как `X_final`. Если нет — переключается на `X_tfidf_svd`.

### 5. Кэш Embeddings

Notebook сохраняет Qwen embeddings в:

```text
embedding_cache/qwen_qwen3-embedding_9571_rs42.npy
```

Имя зависит от:

- модели;
- числа строк после предобработки;
- `random_state`.

Если файл существует, embeddings не пересчитываются, а загружаются из `.npy`. Это ускоряет повторный запуск.

### 6. Кластеризация

Файл: `src/clustering.py`.

Основная структура:

```python
@dataclass
class ClusteringRun:
    name: str
    representation: str
    params: dict
    labels: np.ndarray
    model: object | None
    metrics: dict
```

Она хранит один запуск кластеризации: метод, представление, параметры, labels, модель и метрики.

Метрики считаются функцией:

```python
clustering_metrics(...)
```

Она возвращает:

- `silhouette`;
- `calinski_harabasz`;
- `davies_bouldin`;
- `n_clusters`;
- `min_cluster_size`;
- `max_cluster_size`;
- `noise_ratio`.

Используемые методы:

```python
run_kmeans_grid(...)
run_agglomerative_grid(...)
run_dbscan_grid(...)
```

K-Means перебирает `k=4..15`.

Agglomerative перебирает:

- `n_clusters=4..15`;
- `ward/euclidean`;
- `average/cosine`;
- `complete/cosine`.

DBSCAN перебирает:

- `eps`;
- `min_samples`;
- `metric=cosine`.

Финальный K-Means выбирается через:

```python
choose_kmeans_run(...)
```

Он учитывает Silhouette и штраф за слишком несбалансированные размеры кластеров.

### 7. Анализ Кластеров

Файл: `src/analysis.py`.

Основные функции:

```python
attach_kmeans_distances(...)
```

Добавляет каждому объекту:

- `cluster`;
- `distance_to_centroid`.

Это нужно, чтобы найти типичные и аномальные примеры.

```python
get_cluster_examples(...)
```

Возвращает:

- ближайшие к центроиду объекты;
- самые дальние от центроида объекты.

```python
top_terms_by_cluster(...)
```

Считает top TF-IDF terms для каждого кластера. Даже если финальные кластеры построены на Qwen embeddings, top terms берутся из TF-IDF, потому что они объяснимы.

```python
build_cluster_summary(...)
```

Создает итоговую таблицу:

- `cluster_id`;
- `level_mode`;
- `name`;
- `plot_label`;
- `size`;
- `top_terms`;
- `typical_examples_summary`;
- `outliers_summary`;
- `typical_pattern`;
- `outliers_note`;
- `interpretation`;
- `homogeneity`;
- `mean_char_len`;
- `mean_word_len`.

Ручные интерпретации лежат в словаре:

```python
LEVEL_INTERPRETATIONS
```

Они привязаны к доминирующему уровню HackAPrompt.

```python
explain_cluster_outliers(...)
```

Создает таблицу `outlier_explanations.csv`. Она объясняет, почему дальний объект считается outlier:

- длиннее ядра кластера;
- короче типичных объектов;
- содержит больше спецсимволов;
- имеет больше строк;
- пришел из другого `level`;
- содержит явные мета-инструкции.

### 8. Визуализация

Файл: `src/visualization.py`.

Основные функции:

```python
compute_pca_2d(...)
compute_umap_2d(...)
compute_tsne_2d(...)
```

Они строят 2D-проекции.

```python
plot_clusters_2d(...)
```

Строит scatter plot с цветом по cluster и короткими подписями кластеров.

```python
plot_cluster_level_comparison_2d(...)
```

Строит side-by-side график:

- слева цвет по найденным кластерам;
- справа цвет по исходным уровням HackAPrompt.

Это главный график для доказательства вывода, что кластеры восстановили task templates.

Дополнительные графики:

```python
plot_cluster_sizes(...)
plot_length_distribution(...)
plot_feature_heatmap(...)
```

Они показывают:

- размеры кластеров;
- распределение длины prompts;
- средние инженерные признаки по кластерам.

## Output Files

После запуска notebook создает:

```text
outputs/tables/kmeans_grid_results.csv
outputs/tables/method_comparison_results.csv
outputs/tables/embedding_comparison_results.csv
outputs/tables/cluster_summary.csv
outputs/tables/cluster_level_crosstab.csv
outputs/tables/outlier_explanations.csv
outputs/tables/prompts_with_clusters.csv
```

`prompts_with_clusters.csv` не коммитится, потому что он большой и содержит сырые prompts.

Графики:

```text
outputs/figures/clusters_pca.png
outputs/figures/clusters_umap.png
outputs/figures/clusters_vs_levels_umap.png
outputs/figures/cluster_sizes.png
outputs/figures/length_by_cluster.png
outputs/figures/feature_heatmap.png
```

Примеры:

```text
outputs/cluster_examples/cluster_00.md
...
outputs/cluster_examples/cluster_09.md
```

## Как Объяснять Код На Защите

Короткая версия:

1. Данные загружаются из Hugging Face или локального файла.
2. Текст осторожно предобрабатывается, потому что спецсимволы и структура важны для атак.
3. Строятся два embedding-пространства: TF-IDF+SVD и Qwen.
4. Если Qwen доступен, финальная кластеризация идет по Qwen embeddings.
5. K-Means, Agglomerative и DBSCAN сравниваются по метрикам.
6. Финальный анализ делается через K-Means, потому что у него есть центроиды.
7. Для каждого кластера находятся top terms, типичные examples, outliers и человекочитаемое описание.
8. UMAP показывает, что кластеры почти совпали с уровнями HackAPrompt.

## Почему Такой Код Воспроизводим

- фиксирован `random_state=42`;
- sample size задан явно;
- Qwen endpoint и модель задаются через env vars;
- есть fallback на TF-IDF+SVD;
- outputs сохраняются в CSV/PNG/Markdown;
- тяжелый prompt-level файл не нужен для проверки, потому что все summary-артефакты сохраняются отдельно.
