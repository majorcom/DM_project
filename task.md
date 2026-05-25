# ТЗ для Codex: проект по кластеризации HackAPrompt Dataset

## 0. Контекст проекта

Нужно реализовать финальный проект по курсу кластеризации / Data Mining.
Тема проекта: **кластеризация атакующих промптов из HackAPrompt Dataset**.

Идея проекта: взять реальные adversarial prompts / prompt injection / jailbreak попытки из соревнования HackAPrompt и попробовать найти естественные группы атак:

- прямое игнорирование инструкций;
- roleplay / impersonation;
- просьбы раскрыть системный промпт;
- обфускация;
- многошаговые обходы;
- шаблонные короткие jailbreak-команды;
- длинные социально-инженерные промпты;
- промпты, завязанные на конкретные уровни соревнования.

Главная цель — не просто получить высокий Silhouette Score, а построить интерпретируемые кластеры и дать им человекочитаемые названия.

Официальный датасет:

- Hugging Face: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
- Paper / project page: https://paper.hackaprompt.com/
- arXiv paper: https://arxiv.org/abs/2311.16119

## 1. Что нужно получить на выходе

Сделай воспроизводимый проект в формате Jupyter Notebook.

Минимальный набор файлов:

```text
hackaprompt-clustering/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_hackaprompt_clustering.ipynb
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── vectorization.py
│   ├── clustering.py
│   ├── analysis.py
│   └── visualization.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── cluster_examples/
└── .gitignore
```

Если проект проще удобнее сделать почти полностью в ноутбуке — можно, но повторяющиеся функции лучше вынести в `src/`.

## 2. Главные требования преподавателя

В ноутбуке должны быть явно раскрыты этапы:

1. выбор датасета и обоснование;
2. предобработка;
3. минимум два метода кластеризации с подбором параметров;
4. содержательный анализ кластеров;
5. визуализация в 2D;
6. выводы.

Особый акцент: **анализ кластеров важнее метрик**.

Для каждого итогового кластера нужно показать:

- 5–10 типичных объектов — ближайших к центроиду / наиболее центральных;
- 2–3 аномальных объекта — самых дальних от центроида внутри кластера;
- словесное описание: что объединяет объекты;
- насколько кластер однородный или размытый;
- человекочитаемое название кластера.

## 3. Датасет

Используй `datasets` от Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("hackaprompt/hackaprompt-dataset")
```

Сначала выведи:

```python
print(dataset)
print(dataset.keys())
print(dataset[list(dataset.keys())[0]].column_names)
```

Затем автоматически найди текстовую колонку. Возможные варианты названий: `prompt`, `user_input`, `submission`, `text`, `attack`, `content`.

Если точное имя колонки неизвестно, сделай функцию:

```python
def detect_text_column(df):
    candidates = ["prompt", "user_input", "submission", "text", "attack", "content"]
    for col in candidates:
        if col in df.columns:
            return col
    object_cols = df.select_dtypes(include="object").columns.tolist()
    object_cols = sorted(object_cols, key=lambda c: df[c].astype(str).str.len().mean(), reverse=True)
    return object_cols[0]
```

Важно: не надо грузить весь датасет, если это тормозит. Для учебного проекта достаточно семпла.

Рекомендуемый размер:

- минимум: 5 000 объектов;
- оптимально: 10 000–30 000 объектов;
- если машина слабая: 3 000–5 000 объектов допустимо, но в ноутбуке написать, что это репрезентативная выборка из большого датасета.

Сделай `random_state=42`.

## 4. Предобработка текстов

Нужно подготовить текстовые данные для кластеризации.

Обязательные шаги:

1. удалить пустые тексты;
2. удалить дубликаты;
3. привести к строковому типу;
4. посчитать базовые признаки текста:
   - длина в символах;
   - длина в словах;
   - количество строк;
   - доля uppercase;
   - количество спецсимволов;
   - наличие слов `ignore`, `system`, `instruction`, `developer`, `previous`, `secret`, `password`, `role`, `admin`, `jailbreak`;
5. сохранить очищенный датафрейм.

Не надо агрессивно чистить текст, потому что в prompt injection важны регистр, спецсимволы, повторения и странная структура.

Для TF-IDF можно сделать отдельную нормализованную версию текста:

- lowercase;
- убрать лишние пробелы;
- не удалять полностью пунктуацию, если это ухудшает интерпретацию.

## 5. Векторизация

Реализуй два варианта признакового пространства.

### Вариант A: TF-IDF

Используй:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Рекомендуемые параметры:

```python
TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)
```

После TF-IDF сделай `TruncatedSVD` до 100–300 измерений:

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

svd = TruncatedSVD(n_components=100, random_state=42)
X_tfidf_svd = svd.fit_transform(X_tfidf)
X_tfidf_svd = normalize(X_tfidf_svd)
```

### Вариант B: sentence-transformers embeddings

Используй модель:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Код:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
```

Если нет интернета или модель не скачивается, сделай fallback на TF-IDF + SVD и явно напиши это в ноутбуке.

## 6. Методы кластеризации

Нужно реализовать минимум два метода из курса. Сделай три, чтобы проект выглядел сильнее:

1. K-Means;
2. Agglomerative Clustering;
3. DBSCAN или HDBSCAN, если получится.

### 6.1 K-Means

Подбери `k` не по умолчанию, а через перебор.

Проверить диапазон:

```python
k_values = range(4, 16)
```

Для каждого `k` посчитать:

- Silhouette Score;
- Calinski-Harabasz Score;
- Davies-Bouldin Score;
- размер кластеров.

Важно: метрики — вспомогательные. Итоговый `k` можно выбрать не только по максимуму метрик, а по интерпретируемости.

Пример:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

results = []
for k in range(4, 16):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    results.append({
        "k": k,
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "min_cluster_size": pd.Series(labels).value_counts().min(),
        "max_cluster_size": pd.Series(labels).value_counts().max(),
    })
```

### 6.2 Agglomerative Clustering

Проверить несколько вариантов:

- `n_clusters`: 4–15;
- `linkage`: `ward`, `average`, `complete`;
- для cosine distance использовать `metric="cosine"` и `linkage="average"`.

Пример:

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(
    n_clusters=8,
    metric="cosine",
    linkage="average"
)
labels_agg = agg.fit_predict(X)
```

Если версия sklearn ругается на `metric`, использовать `affinity`.

### 6.3 DBSCAN / HDBSCAN

DBSCAN может плохо работать на текстовых эмбеддингах. Это нормально. Его стоит попробовать и честно описать результат.

Параметры:

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.25, min_samples=10, metric="cosine")
labels_db = db.fit_predict(X)
```

Подобрать `eps`:

```python
eps_values = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
min_samples_values = [5, 10, 20]
```

Для каждого варианта посчитать:

- число кластеров без `-1`;
- долю шума;
- silhouette только по не-шумовым точкам, если кластеров больше 1.

Если почти всё стало шумом или одним кластером — написать, что DBSCAN плохо подходит для данного признакового пространства.

## 7. Выбор итоговой модели

После сравнения выбрать одну итоговую модель для глубокого анализа.

Скорее всего, лучшим вариантом будет:

```text
sentence-transformers embeddings + K-Means
```

или:

```text
TF-IDF + SVD + K-Means
```

Выбор должен быть обоснован:

- кластеры достаточно сбалансированы;
- есть нормальная визуальная структура;
- по примерам можно дать понятные названия;
- метрики не провальные.

## 8. Анализ кластеров

Это самая важная часть. Нужно сделать функции для интерпретации.

### 8.1 Центральные и аномальные объекты

Для K-Means:

```python
import numpy as np
from sklearn.metrics import pairwise_distances

centroids = kmeans.cluster_centers_
distances = pairwise_distances(X, centroids, metric="cosine")
assigned_distances = distances[np.arange(len(labels)), labels]

df["cluster"] = labels
df["distance_to_centroid"] = assigned_distances
```

Для каждого кластера:

```python
def get_cluster_examples(df, cluster_id, n_typical=10, n_outliers=3):
    part = df[df["cluster"] == cluster_id].copy()
    typical = part.sort_values("distance_to_centroid").head(n_typical)
    outliers = part.sort_values("distance_to_centroid", ascending=False).head(n_outliers)
    return typical, outliers
```

### 8.2 Ключевые слова кластера

Для TF-IDF желательно вывести топ-термы по кластерам:

```python
def top_terms_by_cluster(tfidf_matrix, labels, vectorizer, cluster_id, n=20):
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_mean = tfidf_matrix[labels == cluster_id].mean(axis=0)
    scores = np.asarray(cluster_mean).ravel()
    top_idx = scores.argsort()[::-1][:n]
    return list(zip(terms[top_idx], scores[top_idx]))
```

Если итоговая модель на embeddings, всё равно можно использовать TF-IDF только для объяснения кластеров.

### 8.3 Автоматическая черновая разметка кластера

Сделай таблицу:

```text
cluster_id | size | top_terms | typical_examples | preliminary_name | interpretation | homogeneity
```

Названия можно сначала дать автоматически по ключевым словам, но финально в ноутбуке должны быть нормальные человеческие названия.

Примеры возможных названий:

- “Прямое игнорирование системных инструкций”;
- “Roleplay / pretend-mode jailbreaks”;
- “Запросы на раскрытие скрытых правил и system prompt”;
- “Короткие шаблонные команды обхода”;
- “Длинные социально-инженерные атаки”;
- “Обфусцированные или структурно странные промпты”;
- “Промпты с техническими маркерами: system/admin/developer”;
- “Многошаговые инструкции с переопределением роли”.

## 9. Визуализация

Обязательно сделать 2D-визуализацию итоговых кластеров.

### 9.1 PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_2d_pca = pca.fit_transform(X)
```

### 9.2 t-SNE или UMAP

Если хватает ресурсов:

```python
from sklearn.manifold import TSNE

X_2d_tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
).fit_transform(X)
```

UMAP можно использовать, если библиотека установлена:

```python
import umap

reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
X_2d_umap = reducer.fit_transform(X)
```

На графике должны быть:

- точки объектов;
- цвет = кластер;
- легенда;
- подписи с человекочитаемыми названиями кластеров;
- желательно центры кластеров.

Сохранять графики в `outputs/figures/`.

## 10. Дополнительные графики

Сделай несколько полезных графиков:

1. размер кластеров;
2. распределение длины промптов по кластерам;
3. heatmap средних текстовых признаков по кластерам;
4. barplot топ-слов для каждого кластера;
5. таблица примеров по кластерам.

## 11. Notebook structure

Ноутбук должен быть понятным и похожим на готовую сдачу.

Структура:

```markdown
# Кластеризация атакующих промптов HackAPrompt

## 1. Введение
- что такое HackAPrompt;
- почему датасет интересен для кластеризации;
- какие паттерны ожидаем найти.

## 2. Загрузка данных
- ссылка на датасет;
- размер;
- колонки;
- примеры объектов.

## 3. Предобработка
- удаление пустых значений;
- удаление дублей;
- базовые текстовые признаки;
- объяснение, почему не чистим текст слишком агрессивно.

## 4. Векторизация
- TF-IDF + SVD;
- sentence-transformers embeddings или fallback.

## 5. Кластеризация
- K-Means с подбором k;
- Agglomerative Clustering;
- DBSCAN/HDBSCAN;
- сравнение метрик.

## 6. Выбор итоговой модели
- почему выбран конкретный метод.

## 7. Анализ кластеров
- размер кластеров;
- топ-слова;
- типичные примеры;
- аномальные примеры;
- названия и интерпретация.

## 8. Визуализация
- PCA;
- t-SNE или UMAP;
- подписи кластеров.

## 9. Выводы
- что нашли;
- какие типы атак выделились;
- какой метод сработал лучше;
- ограничения проекта;
- что можно улучшить.
```

## 12. Requirements

Создай `requirements.txt`:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
jupyter
notebook
datasets
sentence-transformers
tqdm
wordcloud
umap-learn
```

Если `umap-learn` или `sentence-transformers` не ставятся, проект должен работать без них через fallback.

## 13. README.md

README должен содержать:

```markdown
# HackAPrompt Clustering Project

## Описание
Проект по кластеризации adversarial prompts из HackAPrompt Dataset.

## Датасет
Ссылка: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset

## Цель
Найти интерпретируемые группы prompt injection / jailbreak атак.

## Методы
- TF-IDF + SVD
- Sentence embeddings
- K-Means
- Agglomerative Clustering
- DBSCAN

## Как запустить
pip install -r requirements.txt
jupyter notebook notebooks/01_hackaprompt_clustering.ipynb

## Результат
Итоговые кластеры получают человекочитаемые названия и анализируются через типичные и аномальные примеры.
```

## 14. Важные замечания по качеству

Не делай проект как сухой ML-скрипт. Это учебный проект по интерпретации кластеров.

Плохо:

```text
Cluster 0: size 1342, silhouette 0.12
```

Хорошо:

```text
Кластер 0 — короткие прямые jailbreak-команды.
В него попали промпты, где пользователь напрямую просит модель игнорировать предыдущие инструкции, забыть правила или перейти в другой режим. Кластер довольно однородный: большинство примеров короткие и используют похожие слова ignore, previous, instructions, now.
```

## 15. Критерии готовности

Проект готов, если:

- датасет загружается автоматически или есть понятная инструкция;
- в данных больше 500 объектов;
- есть предобработка;
- есть минимум два метода кластеризации;
- параметры методов подбираются, а не оставлены по умолчанию;
- есть таблица сравнения методов;
- есть итоговая выбранная модель;
- каждый кластер имеет название и словесное описание;
- для каждого кластера показаны типичные и аномальные объекты;
- есть 2D-визуализация с подписями кластеров;
- есть выводы.

## 16. Что делать, если кластеры плохие

Если кластеры плохо интерпретируются, не скрывай это. Напиши:

- какие признаки использовались;
- почему они могли не выделить естественную структуру;
- какой метод дал наиболее осмысленный результат;
- что можно улучшить: другой embedding model, больший sample, ручная таксономия, HDBSCAN, тематическое моделирование BERTopic.

Но всё равно нужно показать анализ и попытаться дать кластерам осторожные описания.

## 17. Желательный финальный результат

В конце ноутбука должна быть итоговая таблица примерно такого вида:

| cluster_id | name | size | top_terms | typical_examples_summary | outliers_summary | homogeneity |
|---|---|---:|---|---|---|---|
| 0 | Прямое игнорирование инструкций | 812 | ignore, previous, instruction | короткие команды обхода | длинные промпты с теми же словами | высокая |
| 1 | Roleplay jailbreaks | 640 | pretend, role, character | просьбы сыграть роль | технические инструкции | средняя |
| 2 | Раскрытие скрытых правил | 530 | system, prompt, secret | запросы системного промпта | общие jailbreak-команды | средняя |

Названия выше — только примеры. Реальные названия нужно дать после просмотра объектов.
