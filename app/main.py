"""FastAPI application — Loughran-McDonald financial text sentiment API."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.dictionary import load_lm_dictionary
from app.models import (
    AnalysisMetadata,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchMetadata,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    SentimentScores,
    SignalOutput,
    WordDetails,
)
from app.scorer import SentimentScorer

API_VERSION = "1.0.0"
APP_TITLE = "Financial NLP API"
APP_DESCRIPTION = (
    "Corporate filing sentiment analysis using the Loughran-McDonald financial dictionary"
)

_scorer: SentimentScorer | None = None
_dictionary_terms: int = 0


def _payload_to_response(payload: dict[str, Any]) -> AnalyzeResponse:
    s = payload["scores"]
    scores = SentimentScores(
        positive=s["positive"],
        negative=s["negative"],
        uncertainty=s["uncertainty"],
        constraining=s["constraining"],
        litigious=s["litigious"],
        strong_modal=s["strong_modal"],
        weak_modal=s["weak_modal"],
        net_sentiment=s["net_sentiment"],
        tone=s["tone"],
        composite_signal=s["composite_signal"],
    )
    sig = payload["signal"]
    signal = SignalOutput(
        direction=sig["direction"],
        confidence=sig["confidence"],
        explanation=sig["explanation"],
    )
    meta = AnalysisMetadata(
        total_words=payload["total_words"],
        dictionary_matches=payload["dictionary_matches"],
        processing_time_ms=payload["processing_time_ms"],
    )
    wd = payload.get("word_details")
    word_details: WordDetails | None = None
    if wd is not None:
        word_details = WordDetails(
            positive_words=wd["positive_words"],
            negative_words=wd["negative_words"],
            uncertainty_words=wd["uncertainty_words"],
            constraining_words=wd["constraining_words"],
            litigious_words=wd["litigious_words"],
            strong_modal_words=wd["strong_modal_words"],
            weak_modal_words=wd["weak_modal_words"],
        )
    return AnalyzeResponse(
        scores=scores, signal=signal, metadata=meta, word_details=word_details
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load LM dictionary and scorer once at startup."""
    global _scorer, _dictionary_terms
    try:
        lm = load_lm_dictionary()
    except FileNotFoundError as e:
        raise RuntimeError(str(e)) from e
    _dictionary_terms = lm.total_terms
    _scorer = SentimentScorer(lm)
    yield
    _scorer = None


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        dictionary_loaded=_scorer is not None,
        dictionary_terms=_dictionary_terms,
        version=API_VERSION,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized.")
    try:
        payload = _scorer.score_to_response_payload(
            body.text, include_word_details=body.include_word_details
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _payload_to_response(payload)


@app.post("/analyze/batch", response_model=BatchResponse)
async def analyze_batch(body: BatchRequest) -> BatchResponse:
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized.")

    t0 = time.perf_counter()
    results: list[AnalyzeResponse] = []
    for item in body.texts:
        try:
            payload = _scorer.score_to_response_payload(
                item.text, include_word_details=False
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Item id={item.id!r}: {e}",
            ) from e
        results.append(_payload_to_response(payload))

    total_ms = (time.perf_counter() - t0) * 1000.0
    return BatchResponse(
        results=results,
        batch_metadata=BatchMetadata(
            total_processed=len(results),
            total_time_ms=total_ms,
        ),
    )
