# Vendored verbatim from the official GAIA leaderboard scorer.
# Source: https://huggingface.co/spaces/gaia-benchmark/leaderboard
#         (scorer.py / question_scorer)
# DO NOT MODIFY. Treated as ground-truth scoring for this benchmark.

from __future__ import annotations

import re
import string
import warnings
from typing import Any


def normalize_number_str(number_str: str) -> float:
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        warnings.warn(f"String {number_str} cannot be normalized to number str.")
        return float("inf")


def split_string(s: str, char_list: list[str] | None = None) -> list[str]:
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    if model_answer is None:
        model_answer = "None"

    if is_float(ground_truth):
        normalized_answer = normalize_number_str(str(model_answer))
        return normalized_answer == float(ground_truth)

    elif any(char in ground_truth for char in [",", ";"]):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            warnings.warn("Answer lists have different lengths, returning False.", UserWarning)
            return False
        comparisons = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                comparisons.append(normalized_ma_elem == float(gt_elem))
            else:
                ma_elem = normalize_str(ma_elem, remove_punct=False)
                gt_elem = normalize_str(gt_elem, remove_punct=False)
                comparisons.append(ma_elem == gt_elem)
        return all(comparisons)

    else:
        ma = normalize_str(model_answer)
        gt = normalize_str(ground_truth)
        return ma == gt
