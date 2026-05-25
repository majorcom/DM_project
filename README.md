# HackAPrompt Clustering Project

Проект по кластеризации атакующих prompts из HackAPrompt Dataset для курса Data Mining.

## Описание

Цель проекта — найти интерпретируемые группы adversarial prompts, prompt injection и jailbreak-попыток. Главный результат — не только метрики кластеризации, а человекочитаемые названия кластеров, типичные примеры, аномальные примеры и выводы о паттернах атак.

## Датасет

- Hugging Face: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
- Project page: https://paper.hackaprompt.com/
- Paper: https://arxiv.org/abs/2311.16119

Датасет на Hugging Face является gated. Перед запуском нужно получить доступ на странице датасета и передать токен через `.env`.

```powershell
copy .env.example .env
```

Затем открой `.env` и заполни:

```text
HF_TOKEN=hf_...
```

Если есть локальная выгрузка, можно не использовать Hugging Face и указать в `.env`:

```text
HACKAPROMPT_LOCAL_FILE=C:\path\to\hackaprompt.csv
```

Поддерживаются `CSV`, `JSON`, `JSONL` и `Parquet`; текстовая колонка определяется автоматически.

По умолчанию ноутбук берет репрезентативную выборку `10000` объектов с `random_state=42`.

## Методы

- TF-IDF + TruncatedSVD
- Локальные Qwen embeddings через OpenAI-compatible endpoint
- K-Means с подбором `k`
- Agglomerative Clustering
- DBSCAN grid search
- PCA и UMAP/t-SNE для 2D-визуализации

## Локальный Qwen-эмбеддер

Основной embedding-вариант использует локальный endpoint:

```text
QWEN_EMBEDDING_BASE_URL=http://localhost:6620/v1
QWEN_EMBEDDING_MODEL=qwen3-embedding
```

Ожидается OpenAI-compatible API:

```text
GET  /v1/models
POST /v1/embeddings
```

Если Qwen endpoint недоступен, ноутбук автоматически переключается на `TF-IDF + SVD` и явно фиксирует fallback в выводах.

## Установка

Рекомендуется использовать отдельное виртуальное окружение. Если какая-то ML-библиотека конфликтует с Python 3.13, проще всего запустить проект на Python 3.11 или 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск

```bash
jupyter notebook notebooks/hackaprompt_clustering.ipynb
```

## Результаты

После запуска ноутбук сохраняет локальные артефакты:

- графики в `outputs/figures/`;
- таблицы в `outputs/tables/`;
- Markdown-примеры кластеров в `outputs/cluster_examples/`.

Эти файлы не коммитятся: они воспроизводятся запуском ноутбука.
