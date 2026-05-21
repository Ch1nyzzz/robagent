from agent.v4.reducers import (
    extract_final_answer,
    infer_answer_shape,
    is_unknown_response,
    normalize_answer,
    reshape_answer,
    route_by_extras,
)
from .extract_strict import extract_final_answer_strict

__all__ = [
    "extract_final_answer",
    "extract_final_answer_strict",
    "infer_answer_shape",
    "is_unknown_response",
    "normalize_answer",
    "reshape_answer",
    "route_by_extras",
]
