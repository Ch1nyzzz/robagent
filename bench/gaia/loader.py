from __future__ import annotations

import os
from typing import Iterator

from datasets import load_dataset
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def iter_tasks(split: str = "validation", levels: list[int] | None = None) -> Iterator[dict]:
    """Iterate GAIA tasks. Requires HF auth (gated dataset).

    Yields dicts with keys: task_id, question, level, final_answer, file_name.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    ds = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        split=split,
        token=token,
    )
    for row in ds:
        if levels is not None and row.get("Level") not in levels:
            continue
        yield {
            "task_id": row["task_id"],
            "question": row["Question"],
            "level": row.get("Level"),
            "final_answer": row.get("Final answer", ""),
            "file_name": row.get("file_name", "") or "",
        }


def build_prompt(task: dict) -> str:
    if task.get("file_name"):
        return (
            f"{task['question']}\n\n"
            f"[A file '{task['file_name']}' is attached. "
            f"Use the file_read tool with file_name='{task['file_name']}' to read it.]"
        )
    return task["question"]
