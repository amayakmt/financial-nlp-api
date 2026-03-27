"""Load and cache the Loughran-McDonald Master Dictionary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Project root: parent of the `app` package
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DICTIONARY_PATH = PROJECT_ROOT / "data" / "lm_dictionary.csv"
ENV_DICTIONARY_PATH = "LM_DICTIONARY_PATH"

_CATEGORY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("positive", "Positive"),
    ("negative", "Negative"),
    ("uncertainty", "Uncertainty"),
    ("constraining", "Constraining"),
    ("litigious", "Litigious"),
    ("strong_modal", "Strong_Modal"),
    ("weak_modal", "Weak_Modal"),
)


@dataclass(frozen=True, slots=True)
class LMDictionary:
    """Category word sets (lowercased) and corpus metadata."""

    positive_words: frozenset[str]
    negative_words: frozenset[str]
    uncertainty_words: frozenset[str]
    constraining_words: frozenset[str]
    litigious_words: frozenset[str]
    strong_modal_words: frozenset[str]
    weak_modal_words: frozenset[str]
    total_terms: int

    def all_category_sets(self) -> dict[str, frozenset[str]]:
        """Map internal category keys to word sets."""
        return {
            "positive": self.positive_words,
            "negative": self.negative_words,
            "uncertainty": self.uncertainty_words,
            "constraining": self.constraining_words,
            "litigious": self.litigious_words,
            "strong_modal": self.strong_modal_words,
            "weak_modal": self.weak_modal_words,
        }


def _resolve_dictionary_path() -> Path:
    """Return CSV path from ``LM_DICTIONARY_PATH`` or the default under project root."""
    env = os.environ.get(ENV_DICTIONARY_PATH)
    if env:
        p = Path(env).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    return DEFAULT_DICTIONARY_PATH.resolve()


def load_lm_dictionary(csv_path: Path | None = None) -> LMDictionary:
    """
    Load the Loughran-McDonald Master Dictionary from CSV.

    A word belongs to a category when that category column is numeric and > 0.
    All stored tokens are lowercased for matching.
    """
    path = csv_path if csv_path is not None else _resolve_dictionary_path()
    if not path.is_file():
        raise FileNotFoundError(f"Loughran-McDonald dictionary not found: {path}")

    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode dictionary CSV: {path}")

    if "Word" not in df.columns:
        raise ValueError("Dictionary CSV must contain a 'Word' column.")

    sets: dict[str, set[str]] = {}
    for key, col in _CATEGORY_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Dictionary CSV missing column: {col}")
        numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
        mask = numeric > 0
        words = df.loc[mask, "Word"].astype(str).str.lower().str.strip()
        sets[key] = set(words[words.str.len() > 0])

    return LMDictionary(
        positive_words=frozenset(sets["positive"]),
        negative_words=frozenset(sets["negative"]),
        uncertainty_words=frozenset(sets["uncertainty"]),
        constraining_words=frozenset(sets["constraining"]),
        litigious_words=frozenset(sets["litigious"]),
        strong_modal_words=frozenset(sets["strong_modal"]),
        weak_modal_words=frozenset(sets["weak_modal"]),
        total_terms=int(len(df)),
    )
