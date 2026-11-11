"""RULER answer normalisation shared by the decode and dense RULER evaluations."""
from __future__ import annotations

import re


_TRIM = " \t\r\n\"'`.,;:"


def normalize_prediction(text: str) -> str:
    value = str(text).strip()
    value = re.sub(r"(?is)<think>.*?</think>", "", value).strip()
    value = value.splitlines()[0].strip() if value else value
    value = re.sub(r"^(answer|final answer)\s*:\s*", "", value, flags=re.IGNORECASE).strip()
    return value.strip(_TRIM).lower()
