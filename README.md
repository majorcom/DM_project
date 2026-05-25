# HackAPrompt Clustering Project

Проект по кластеризации вредоносных prompts из HackAPrompt Dataset для финального проекта курса по кластеризации / Data Mining.

Главная идея: не просто получить высокий Silhouette Score, а найти группы prompts, которые можно объяснить человеческим языком: переводческие jailbreaks, secret-key extraction, slash/unicode обфускация, roleplay-атаки, prompt injection внутри QA/search и writing-feedback задач.

## Что Есть В Проекте

- воспроизводимый Jupyter Notebook: `notebooks/hackaprompt_clustering.ipynb`;
- локальный Qwen embedding endpoint как основной embedding-вариант;
- fallback на `TF-IDF + SVD`, если Qwen недоступен;
- сравнение K-Means, Agglomerative Clustering и DBSCAN;
- подбор гиперпараметров, а не запуск методов с дефолтами;
- интерпретация кластеров через названия, top terms, типичные примеры и outliers;
- графики PCA, UMAP, размеры кластеров, длины prompts, heatmap признаков;
- side-by-side UMAP: найденные кластеры против исходных уровней HackAPrompt;
- отдельные Markdown-отчеты для защиты.

## Датасет

- Hugging Face: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
- Project page: https://paper.hackaprompt.com/
- Paper: https://arxiv.org/abs/2311.16119

HackAPrompt Dataset содержит adversarial prompts из соревнования по prompt injection и jailbreak. Датасет подходит для кластеризации, потому что каждый объект является текстовым prompt, у которого есть структура, сценарий атаки и содержательная интерпретация.

Датасет на Hugging Face является gated. Перед запуском нужно получить доступ на странице датасета и передать токен через `.env`.

```powershell
copy .env.example .env
```

Затем открой `.env` и заполни:

```text
HF_TOKEN=hf_...
```

Если есть локальная выгрузка, можно не использовать Hugging Face и указать:

```text
HACKAPROMPT_LOCAL_FILE=C:\path\to\hackaprompt.csv
```

Поддерживаются `CSV`, `JSON`, `JSONL` и `Parquet`; текстовая колонка определяется автоматически.

По умолчанию notebook берет выборку `10000` объектов с `random_state=42`. После удаления пустых строк и дублей в текущем прогоне осталось `9571` prompts.

## Локальный Qwen-Эмбеддер

Основной embedding-вариант использует локальный OpenAI-compatible endpoint:

```text
QWEN_EMBEDDING_BASE_URL=http://localhost:6620/v1
QWEN_EMBEDDING_MODEL=qwen3-embedding
```

Ожидаемые методы API:

```text
GET  /v1/models
POST /v1/embeddings
```

Если Qwen endpoint недоступен, notebook автоматически переключается на `TF-IDF + SVD` и явно пишет, что был использован fallback.

Qwen embeddings кэшируются в `embedding_cache/`, чтобы повторный запуск notebook не пересчитывал 9571 embedding заново.

## Методы

Представления текста:

- `TF-IDF + TruncatedSVD` как объяснимый baseline и fallback;
- `Qwen embeddings` как основное семантическое embedding-пространство.

Методы кластеризации:

- `K-Means` с перебором `k=4..15`;
- `Agglomerative Clustering` с разными `n_clusters`, `linkage`, `metric`;
- `DBSCAN` с grid search по `eps` и `min_samples`.

Финальная модель текущего прогона:

```text
Qwen embeddings + K-Means
k = 10
silhouette = 0.65265
Davies-Bouldin = 0.98906
```

Дополнительно есть мини-эксперимент `Qwen embeddings` vs `TF-IDF + SVD`. В текущем прогоне `TF-IDF + SVD` дает более высокий Silhouette (`0.8562`), потому что лучше цепляется за повторяющийся boilerplate уровней. Qwen оставлен основным вариантом, так как это требуемое семантическое embedding-представление проекта.

## Установка

Рекомендуется отдельное виртуальное окружение.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Если какая-то ML-библиотека конфликтует с Python 3.13, проще запустить проект на Python 3.11 или 3.12.

## Запуск

```bash
jupyter notebook notebooks/hackaprompt_clustering.ipynb
```

Для чистого воспроизводимого запуска в Jupyter:

```text
Kernel -> Restart & Run All
```

## Результаты И Артефакты

После запуска notebook сохраняет:

- графики в `outputs/figures/`;
- таблицы в `outputs/tables/`;
- Markdown-примеры кластеров в `outputs/cluster_examples/`.

Легкие итоговые артефакты можно коммитить, чтобы проект было удобно смотреть без повторного запуска. Полный `outputs/tables/prompts_with_clusters.csv` не коммитится: он тяжелый и содержит сырые prompts.

Ключевые файлы:

- `outputs/figures/clusters_umap.png` — главный UMAP с найденными кластерами;
- `outputs/figures/clusters_vs_levels_umap.png` — сравнение найденных кластеров с исходными уровнями HackAPrompt;
- `outputs/figures/length_by_cluster.png` — распределение длины prompts по кластерам;
- `outputs/figures/feature_heatmap.png` — средние текстовые признаки по кластерам;
- `outputs/tables/cluster_summary.csv` — названия, размеры, top terms, интерпретации кластеров;
- `outputs/tables/cluster_level_crosstab.csv` — связь кластеров с уровнями HackAPrompt;
- `outputs/tables/embedding_comparison_results.csv` — сравнение Qwen и TF-IDF+SVD;
- `outputs/tables/outlier_explanations.csv` — пояснения аномальных объектов;
- `outputs/tables/method_comparison_results.csv` — сравнение K-Means, Agglomerative, DBSCAN;
- `outputs/cluster_examples/*.md` — типичные и аномальные prompts по каждому кластеру.

## Документы Для Защиты

- `RESULTS.md` — актуальный отчет по текущему прогону и соответствие критериям оценки.
- `CODE_EXPLANATION.md` — подробное объяснение кода, модулей и notebook pipeline.
- `PROJECT_EXPLANATION.md` — подробное объяснение, что делали, зачем, почему выбраны такие методы и как читать результат.

## Что Показывать На Защите

1. `notebooks/hackaprompt_clustering.ipynb` — основной сдаваемый файл со всеми этапами.
2. `RESULTS.md` — краткий отчет по текущему прогону.
3. `outputs/figures/clusters_vs_levels_umap.png` — лучший график для главного вывода.
4. `outputs/tables/cluster_summary.csv` — человекочитаемая интерпретация кластеров.
5. `outputs/tables/embedding_comparison_results.csv` — честное сравнение Qwen и TF-IDF.
6. `outputs/tables/outlier_explanations.csv` — объяснение аномальных объектов.
7. `outputs/cluster_examples/*.md` — реальные типичные и дальние примеры.

## Главный Вывод

Кластеризация в основном восстановила уровни / task templates HackAPrompt: переводчик, secret key, movie title, QA/search, writing feedback, short story, slash obfuscation и roleplay. Это содержательный результат: embeddings видят структуру атакующих prompts. Одновременно это ограничение: текущая кластеризация больше отражает шаблоны заданий, чем чистую универсальную таксономию jailbreak-техник.
