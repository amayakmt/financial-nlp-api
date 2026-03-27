# Financial NLP API

REST API for financial text sentiment analysis using the Loughran-McDonald (2011) financial dictionary.

## Methodology

Corporate filings and earnings language differ systematically from general English. Off-the-shelf sentiment lists (for example the Harvard General Inquirer) misclassify a large share of negative financial terms. The **Loughran-McDonald Master Dictionary** provides domain-specific lexicons: **positive**, **negative**, **uncertainty**, **constraining**, **litigious**, **strong modal**, and **weak modal** words. This service tokenizes input, counts category hits, and reports proportions per total words.

The **composite signal** is the product of the uncertainty proportion and the constraining proportion—(uncertainty count / total words) × (constraining count / total words)—a formulation related to signals explored in long-horizon work on S&P 500 filing data. The API returns this magnitude as `composite_signal`; the `signal.direction` field applies fixed demo thresholds for bullish / bearish / neutral interpretation (not investment advice).

## Tech stack

Python · FastAPI · Pydantic · pandas · Docker

## Live demo

Deployed instance: [DEPLOYED_URL]

## Quick start (local)

```bash
cd financial-nlp-api
pip install -r requirements.txt
# optional: pip install pytest httpx
uvicorn app.main:app --reload
```

Open interactive docs at `http://127.0.0.1:8000/docs`.

### Dictionary path

By default the app loads `data/lm_dictionary.csv` relative to the project root. Override with:

```bash
set LM_DICTIONARY_PATH=C:\path\to\your\lm_dictionary.csv
```

## Quick start (Docker)

```bash
docker build -t financial-nlp-api .
docker run -p 8000:8000 financial-nlp-api
```

## Example request

```bash
curl -s -X POST "http://127.0.0.1:8000/analyze" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Results may vary materially; management cannot predict outcomes and may be constrained by regulation and litigation.\", \"include_word_details\": true}"
```

Example JSON (abridged):

```json
{
  "scores": {
    "positive": 0.0,
    "negative": 0.0,
    "uncertainty": 0.08,
    "constraining": 0.04,
    "litigious": 0.04,
    "strong_modal": 0.0,
    "weak_modal": 0.04,
    "net_sentiment": 0.0,
    "tone": 0.5,
    "composite_signal": 0.0032
  },
  "signal": {
    "direction": "bearish",
    "confidence": "moderate",
    "explanation": "Net sentiment (proportion) is ..."
  },
  "metadata": {
    "total_words": 25,
    "dictionary_matches": 5,
    "processing_time_ms": 2.5
  },
  "word_details": null
}
```

## Use cases

- Screen 10-K **risk factors** and MD&A for tone and uncertainty/constraining load  
- Monitor **earnings call** transcripts for shifts in cautious language  
- Prototype **sentiment-based** research signals (backtests in academic or proprietary pipelines)

Composite-style features have been studied in multi-decade panels of U.S. equity filings; this API exposes transparent, dictionary-only scoring for custom text.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Redirects to `/docs` |
| GET | `/health` | Health and dictionary load status |
| POST | `/analyze` | Score a single text |
| POST | `/analyze/batch` | Score 1–50 texts with client `id` per item |

## Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## License

MIT
