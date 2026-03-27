"""Dictionary-based sentiment scoring using Loughran-McDonald categories."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from app.dictionary import LMDictionary

_TOKEN_PATTERN = re.compile(r"[a-z]+")


@dataclass(slots=True)
class ScoreResult:
    """Raw scoring output before mapping to API models."""

    scores: dict[str, float]
    signal: dict[str, str]
    total_words: int
    dictionary_matches: int
    word_details: dict[str, list[str]] | None


class SentimentScorer:
    """
    Score financial text using Loughran-McDonald category word lists.

    Tokenization: lowercase text, extract contiguous alphabetic sequences (a-z only).
    """

    def __init__(self, dictionary: LMDictionary) -> None:
        self._dict = dictionary
        self._sets = dictionary.all_category_sets()

    @staticmethod
    def tokenize(text: str) -> tuple[list[str], int]:
        """Lowercase *text* and return (tokens, total word count)."""
        tokens = _TOKEN_PATTERN.findall(text.lower())
        return tokens, len(tokens)

    def _category_counts(
        self, tokens: list[str], track_details: bool
    ) -> tuple[dict[str, int], dict[str, list[str]] | None]:
        counts = {k: 0 for k in self._sets}
        details: dict[str, list[str]] | None = {k: [] for k in self._sets} if track_details else None
        seen_per_cat: dict[str, set[str]] = {k: set() for k in self._sets}

        for tok in tokens:
            for cat, word_set in self._sets.items():
                if tok in word_set:
                    counts[cat] += 1
                    if details is not None and tok not in seen_per_cat[cat]:
                        seen_per_cat[cat].add(tok)
                        details[cat].append(tok)

        return counts, details

    def _dictionary_match_tokens(self, tokens: list[str]) -> int:
        """Count tokens that appear in at least one LM category."""
        union: set[str] = set()
        for s in self._sets.values():
            union |= s
        return sum(1 for t in tokens if t in union)

    def _build_signal(
        self,
        net_sentiment: float,
        composite_signal: float,
        positive_prop: float,
        negative_prop: float,
        uncertainty_prop: float,
        constraining_prop: float,
    ) -> tuple[str, str, str]:
        direction: str
        confidence: str

        if composite_signal > 0.0005 and net_sentiment < -0.01:
            direction, confidence = "bearish", "high"
        elif composite_signal > 0.0003 and net_sentiment < 0:
            direction, confidence = "bearish", "moderate"
        elif net_sentiment > 0.02 and composite_signal < 0.0002:
            direction, confidence = "bullish", "moderate"
        elif net_sentiment > 0.01 and composite_signal < 0.0003:
            direction, confidence = "bullish", "low"
        else:
            direction, confidence = "neutral", "low"

        explanation = (
            f"Net sentiment (proportion) is {net_sentiment:.6f} "
            f"(positive {positive_prop:.6f}, negative {negative_prop:.6f}); "
            f"composite signal (uncertainty × constraining proportions) is {composite_signal:.6f} "
            f"(uncertainty {uncertainty_prop:.6f}, constraining {constraining_prop:.6f}). "
            f"Threshold rules yield a {confidence} confidence {direction} reading."
        )
        return direction, confidence, explanation

    def compute_scores(
        self, text: str, *, include_word_details: bool = False
    ) -> tuple[ScoreResult, float]:
        """
        Analyze *text* and return (result, elapsed_seconds).

        Raises:
            ValueError: if there are no alphabetic tokens.
        """
        t0 = time.perf_counter()
        tokens, total_words = self.tokenize(text)
        if total_words == 0:
            raise ValueError("No alphabetic tokens found after tokenization.")

        counts, details = self._category_counts(tokens, include_word_details)
        tw = float(total_words)

        positive_prop = counts["positive"] / tw
        negative_prop = counts["negative"] / tw
        uncertainty_prop = counts["uncertainty"] / tw
        constraining_prop = counts["constraining"] / tw

        net_sentiment = (counts["positive"] - counts["negative"]) / tw
        pos_n = counts["positive"] + counts["negative"]
        if pos_n == 0:
            tone = 0.5
        else:
            tone = counts["positive"] / pos_n

        composite_signal = uncertainty_prop * constraining_prop

        scores: dict[str, float] = {
            "positive": positive_prop,
            "negative": negative_prop,
            "uncertainty": uncertainty_prop,
            "constraining": constraining_prop,
            "litigious": counts["litigious"] / tw,
            "strong_modal": counts["strong_modal"] / tw,
            "weak_modal": counts["weak_modal"] / tw,
            "net_sentiment": net_sentiment,
            "tone": tone,
            "composite_signal": composite_signal,
        }

        direction, confidence, explanation = self._build_signal(
            net_sentiment,
            composite_signal,
            positive_prop,
            negative_prop,
            uncertainty_prop,
            constraining_prop,
        )

        signal = {
            "direction": direction,
            "confidence": confidence,
            "explanation": explanation,
        }

        dict_matches = self._dictionary_match_tokens(tokens)

        word_details_out: dict[str, list[str]] | None = None
        if details is not None:
            word_details_out = {
                "positive_words": details["positive"],
                "negative_words": details["negative"],
                "uncertainty_words": details["uncertainty"],
                "constraining_words": details["constraining"],
                "litigious_words": details["litigious"],
                "strong_modal_words": details["strong_modal"],
                "weak_modal_words": details["weak_modal"],
            }

        elapsed = time.perf_counter() - t0
        result = ScoreResult(
            scores=scores,
            signal=signal,
            total_words=total_words,
            dictionary_matches=dict_matches,
            word_details=word_details_out,
        )
        return result, elapsed

    def score_to_response_payload(
        self, text: str, *, include_word_details: bool = False
    ) -> dict[str, Any]:
        """Return a dict suitable for building ``AnalyzeResponse`` models."""
        result, elapsed = self.compute_scores(text, include_word_details=include_word_details)
        return {
            "scores": result.scores,
            "signal": result.signal,
            "total_words": result.total_words,
            "dictionary_matches": result.dictionary_matches,
            "processing_time_ms": elapsed * 1000.0,
            "word_details": result.word_details,
        }
