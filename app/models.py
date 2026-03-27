"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Direction = Literal["bullish", "bearish", "neutral"]
Confidence = Literal["high", "moderate", "low"]


class AnalyzeRequest(BaseModel):
    """Single-document analysis request."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": (
                        "Our results may fluctuate due to uncertain market conditions. "
                        "Management cannot guarantee future performance; losses could "
                        "materially affect operations and constrain capital allocation."
                    ),
                    "include_word_details": False,
                }
            ]
        }
    )

    text: str = Field(..., min_length=10, description="Raw financial text to analyze.")
    include_word_details: bool = Field(
        False, description="If true, return matched words per LM category."
    )


class TextItem(BaseModel):
    """One item in a batch request."""

    id: str = Field(..., min_length=1, description="Client-supplied identifier.")
    text: str = Field(..., min_length=10, description="Text to analyze.")


class BatchRequest(BaseModel):
    """Batch analysis (1–50 documents)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "texts": [
                        {
                            "id": "filing-2024-risk",
                            "text": (
                                "Risk factors include litigation exposure and substantial "
                                "uncertainty regarding regulatory outcomes that may constrain "
                                "our ability to expand."
                            ),
                        }
                    ]
                }
            ]
        }
    )

    texts: list[TextItem] = Field(..., min_length=1, max_length=50)


class SentimentScores(BaseModel):
    """Per-category proportions and derived metrics (all in [0,1] except net_sentiment and composite)."""

    positive: float
    negative: float
    uncertainty: float
    constraining: float
    litigious: float
    strong_modal: float
    weak_modal: float
    net_sentiment: float
    tone: float
    composite_signal: float


class SignalOutput(BaseModel):
    """Directional signal derived from scores."""

    direction: Direction
    confidence: Confidence
    explanation: str


class AnalysisMetadata(BaseModel):
    """Processing statistics."""

    total_words: int
    dictionary_matches: int
    processing_time_ms: float


class WordDetails(BaseModel):
    """Matched dictionary tokens by category (first-seen order, unique per category)."""

    positive_words: list[str] = Field(default_factory=list)
    negative_words: list[str] = Field(default_factory=list)
    uncertainty_words: list[str] = Field(default_factory=list)
    constraining_words: list[str] = Field(default_factory=list)
    litigious_words: list[str] = Field(default_factory=list)
    strong_modal_words: list[str] = Field(default_factory=list)
    weak_modal_words: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """Full analysis result for one document."""

    scores: SentimentScores
    signal: SignalOutput
    metadata: AnalysisMetadata
    word_details: WordDetails | None = None


class BatchMetadata(BaseModel):
    """Aggregate timing for a batch run."""

    total_processed: int
    total_time_ms: float


class BatchResponse(BaseModel):
    """Batch analysis response."""

    results: list[AnalyzeResponse]
    batch_metadata: BatchMetadata


class HealthResponse(BaseModel):
    """Service and dictionary status."""

    status: str
    dictionary_loaded: bool
    dictionary_terms: int
    version: str
