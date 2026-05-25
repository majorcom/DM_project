"""Data loading helpers for the HackAPrompt dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import DatasetDict, load_dataset

from .env import load_project_env


TEXT_COLUMN_CANDIDATES = ["prompt", "user_input", "submission", "text", "attack", "content"]


@dataclass(frozen=True)
class LoadedData:
    """Container with the sampled dataframe and metadata used in the notebook."""

    dataframe: pd.DataFrame
    dataset: DatasetDict | None
    split_name: str
    text_column: str
    source: str


class HackAPromptDataAccessError(RuntimeError):
    """Raised when the gated HackAPrompt dataset cannot be loaded."""


def detect_text_column(df: pd.DataFrame, candidates: Iterable[str] = TEXT_COLUMN_CANDIDATES) -> str:
    """Detect the most likely text column in a dataframe."""

    for column in candidates:
        if column in df.columns:
            return column

    object_columns = df.select_dtypes(include="object").columns.tolist()
    if not object_columns:
        raise ValueError("Could not detect a text column: no object columns were found.")

    object_columns = sorted(
        object_columns,
        key=lambda column: df[column].astype(str).str.len().mean(),
        reverse=True,
    )
    return object_columns[0]


def _choose_split_name(dataset: DatasetDict) -> str:
    """Prefer train when present; otherwise use the first available split."""

    if "train" in dataset:
        return "train"
    return list(dataset.keys())[0]


def _sample_dataframe(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Return a reproducible sample when the dataframe is larger than requested."""

    if sample_size and len(df) > sample_size:
        return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df.reset_index(drop=True)


def _load_env_from_current_project() -> None:
    """Load .env from cwd or its parent, matching project-root and notebooks launches."""

    cwd = Path.cwd()
    for root in (cwd, cwd.parent):
        if (root / ".env").exists():
            load_project_env(root)
            return


def load_local_dataframe(
    file_path: str | Path,
    sample_size: int = 10_000,
    random_state: int = 42,
) -> LoadedData:
    """Load prompts from a local CSV/JSON/JSONL/Parquet file."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported local dataset format: {suffix}")

    text_column = detect_text_column(df)
    df = _sample_dataframe(df, sample_size=sample_size, random_state=random_state)
    return LoadedData(
        dataframe=df,
        dataset=None,
        split_name="local",
        text_column=text_column,
        source=str(path),
    )


def load_hackaprompt_dataframe(
    sample_size: int = 10_000,
    random_state: int = 42,
    split_name: str | None = None,
    token: str | None = None,
    local_file: str | Path | None = None,
) -> LoadedData:
    """Load HackAPrompt from Hugging Face and return a reproducible sample."""

    _load_env_from_current_project()

    local_file = local_file or os.getenv("HACKAPROMPT_LOCAL_FILE")
    if local_file:
        return load_local_dataframe(local_file, sample_size=sample_size, random_state=random_state)

    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    try:
        dataset = load_dataset("hackaprompt/hackaprompt-dataset", token=token)
    except Exception as exc:
        message = (
            "Could not load hackaprompt/hackaprompt-dataset from Hugging Face. "
            "The dataset is gated, so you need access approval and authentication. "
            "Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, or set HACKAPROMPT_LOCAL_FILE "
            "to a local CSV/JSON/JSONL/Parquet export with a prompt/text column. "
            f"Original error: {exc}"
        )
        raise HackAPromptDataAccessError(message) from exc

    selected_split = split_name or _choose_split_name(dataset)
    df = dataset[selected_split].to_pandas()
    text_column = detect_text_column(df)
    df = _sample_dataframe(df, sample_size=sample_size, random_state=random_state)

    return LoadedData(
        dataframe=df,
        dataset=dataset,
        split_name=selected_split,
        text_column=text_column,
        source="huggingface:hackaprompt/hackaprompt-dataset",
    )
