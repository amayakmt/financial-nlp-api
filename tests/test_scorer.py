"""Tests for dictionary loading and sentiment scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.dictionary import load_lm_dictionary  # noqa: E402
from app.main import app  # noqa: E402
from app.scorer import SentimentScorer  # noqa: E402

DICT_PATH = _ROOT / "data" / "lm_dictionary.csv"


@pytest.fixture(scope="module")
def lm() -> object:
    if not DICT_PATH.is_file():
        pytest.skip(f"Dictionary not found at {DICT_PATH}")
    return load_lm_dictionary(DICT_PATH)


@pytest.fixture(scope="module")
def scorer(lm: object) -> SentimentScorer:
    return SentimentScorer(lm)


def test_dictionary_loads_with_nonempty_categories(lm: object) -> None:
    assert lm.total_terms > 50_000
    assert len(lm.positive_words) > 0
    assert len(lm.negative_words) > 0
    assert len(lm.uncertainty_words) > 0
    assert len(lm.constraining_words) > 0
    assert len(lm.litigious_words) > 0
    assert len(lm.strong_modal_words) > 0
    assert len(lm.weak_modal_words) > 0


def test_tokenization_known_string(scorer: SentimentScorer) -> None:
    tokens, n = scorer.tokenize("Item 7: MD&A — FY2024 (revenue up!)")
    assert tokens == ["item", "md", "a", "fy", "revenue", "up"]
    assert n == 6


def test_negative_text_net_sentiment(scorer: SentimentScorer) -> None:
    text = (
        "The company reported losses and adverse developments including impairment, "
        "litigation exposure, default risk, and declining margins across operations."
    )
    result, _ = scorer.compute_scores(text)
    assert result.scores["net_sentiment"] < 0


def test_positive_text_net_sentiment(scorer: SentimentScorer) -> None:
    text = (
        "Strong profit growth and favorable gains with robust performance exceeded "
        "expectations; improved efficiency and successful expansion supported results."
    )
    result, _ = scorer.compute_scores(text)
    assert result.scores["net_sentiment"] > 0


def test_composite_signal_product(scorer: SentimentScorer, lm: object) -> None:
    u = next(iter(lm.uncertainty_words))
    constr = lm.constraining_words - {u}
    if not constr:
        pytest.skip("No constraining word distinct from uncertainty sample.")
    c = next(iter(constr))
    text = f"{u} {u} {c} {c} padding words here for length and more words"
    result, _ = scorer.compute_scores(text)
    tw = result.total_words
    uc = 2 / tw
    cc = 2 / tw
    expected = uc * cc
    assert result.scores["composite_signal"] == pytest.approx(expected, rel=1e-9)


def test_batch_returns_correct_count() -> None:
    body = {
        "texts": [
            {"id": "a", "text": "Strong profit and favorable gains exceeded plans."},
            {"id": "b", "text": "Losses and adverse litigation created impairment risk."},
        ]
    }
    with TestClient(app) as client:
        r = client.post("/analyze/batch", json=body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    assert data["batch_metadata"]["total_processed"] == 2


def test_short_input_no_tokens_returns_400() -> None:
    with TestClient(app) as client:
        r = client.post("/analyze", json={"text": "1234567890"})
    assert r.status_code == 400


def test_no_dictionary_matches(scorer: SentimentScorer, lm: object) -> None:
    union: set[str] = set()
    for s in lm.all_category_sets().values():
        union |= s
    tokens: list[str] = []
    for i in range(12):
        w = f"qxzz{i}jw"
        while w in union:
            w += "x"
        tokens.append(w)
    text = " ".join(tokens)
    result, _ = scorer.compute_scores(text)
    assert result.dictionary_matches == 0
    assert result.scores["composite_signal"] == 0.0
