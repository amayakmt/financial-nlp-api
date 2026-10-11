# Financial NLP API

**A FastAPI service that scores financial text with the Loughran-McDonald dictionary and returns category proportions, a composite uncertainty-constraining signal, and a bullish, bearish, or neutral reading.**
 
```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Results may vary materially; management cannot predict outcomes and may be constrained by regulation and litigation."}'
```
 
```json
{
  "scores": {
    "negative": 0.0625,
    "uncertainty": 0.25,
    "constraining": 0.0625,
    "litigious": 0.125,
    "net_sentiment": -0.0625,
    "composite_signal": 0.015625
  },
  "signal": { "direction": "bearish", "confidence": "high" },
  "metadata": { "total_words": 16, "dictionary_matches": 7 }
}
```
 
*Real response, abridged. The full response also includes the remaining categories, `tone`, an `explanation` string, and processing time.*
 
## Motivation
 
General-purpose sentiment lists were built for everyday English, and finance breaks them. Words like "liability", "tax", and "cost" read as negative in a general lexicon and are routine in a 10-K. Loughran and McDonald (2011) built word lists from financial filings to fix this, but they ship as a spreadsheet, not as something a pipeline can call.
 
FilingTone wraps that dictionary in a typed HTTP API. Send text, get transparent, dictionary-only scores back in milliseconds, with the matched words available on request so every number can be traced to its source. There is no model to retrain and no black box to defend.
 
## Quick Start
 
Requires **Python 3.11+**. The Loughran-McDonald dictionary ships in `data/`, so there is nothing else to download.
 
```bash
git clone https://github.com/amayakmt/financial-nlp-api.git
cd financial-nlp-api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
 
Open the interactive docs at `http://127.0.0.1:8000/docs`, or run the `curl` call above.
 
Prefer Docker?
 
```bash
docker build -t financial-nlp-api .
docker run -p 8000:8000 financial-nlp-api
```
 
## Usage
 
### Endpoints
 
| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Redirects to `/docs` |
| GET | `/health` | Service status and number of dictionary terms loaded |
| POST | `/analyze` | Score one text |
| POST | `/analyze/batch` | Score 1 to 50 texts, each tagged with your own `id` |
 
### Analyze one text
 
`POST /analyze` takes:
 
- `text` (required): at least 10 characters.
- `include_word_details` (optional, default `false`): when `true`, returns the matched words for each category, unique and in first-seen order.
```bash
curl -s -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Strong growth and record profits improved our outlook considerably.", "include_word_details": true}'
```
 
### Analyze many texts
 
```bash
curl -s -X POST http://127.0.0.1:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": [
        {"id": "filing-a", "text": "Strong growth and record profits improved our outlook considerably."},
        {"id": "filing-b", "text": "Litigation may constrain our ability to operate under uncertain regulation."}
      ]}'
```
 
Results come back in request order with a `batch_metadata` block (`total_processed`, `total_time_ms`). Batch mode never returns word details. If any item fails validation, the whole request returns `400` and the error names the offending `id`.
 
### How scoring works
 
1. Lowercase the text and extract alphabetic tokens (`a-z` runs only).
2. Count tokens in each Loughran-McDonald category: positive, negative, uncertainty, constraining, litigious, strong modal, weak modal.
3. Divide each count by total words to get proportions.
| Field | Definition |
| --- | --- |
| `net_sentiment` | (positive count − negative count) / total words |
| `tone` | positive / (positive + negative), or `0.5` when neither appears |
| `composite_signal` | uncertainty proportion × constraining proportion |
 
`signal.direction` and `signal.confidence` come from fixed, hand-set thresholds, checked in this order:
 
| Condition | Direction | Confidence |
| --- | --- | --- |
| composite > 0.0005 and net sentiment < −0.01 | bearish | high |
| composite > 0.0003 and net sentiment < 0 | bearish | moderate |
| net sentiment > 0.02 and composite < 0.0002 | bullish | moderate |
| net sentiment > 0.01 and composite < 0.0003 | bullish | low |
| anything else | neutral | low |
 
These thresholds are heuristics for screening, not investment advice. The composite is a transparent feature, in the spirit of signals studied on long-horizon panels of 10-K filings, that you can feed into your own research.
 
### Limitations
 
- Dictionary-only: no negation handling ("not profitable" still counts "profitable") and no context.
- English text only. Numbers and punctuation are dropped, so `10-K` tokenizes to `k`.
- Very short or number-only text is rejected: input needs at least 10 characters and one alphabetic token.
### Configuration
 
By default the app loads `data/lm_dictionary.csv`. Point it at another copy with `LM_DICTIONARY_PATH` (absolute, or relative to the project root):
 
```bash
export LM_DICTIONARY_PATH=/path/to/lm_dictionary.csv
```
 
On Windows PowerShell, use `$env:LM_DICTIONARY_PATH = "C:\path\to\lm_dictionary.csv"`. CORS is open to all origins, so tighten `allow_origins` in `app/main.py` before exposing the service publicly.
 
## Contributing
 
### Clone the repo
 
```bash
git clone https://github.com/amayakmt/financial-nlp-api.git
cd financial-nlp-api
```
 
### Set up the environment
 
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest httpx
```
 
### Run the test suite
 
```bash
pytest tests/ -v
```
 
### Run the server with auto-reload
 
```bash
uvicorn app.main:app --reload
```
 
### Project layout
 
```text
app/main.py          # FastAPI routes and app lifecycle
app/scorer.py        # Tokenization, category counts, composite signal, thresholds
app/dictionary.py    # Loads and caches the Loughran-McDonald CSV
app/models.py        # Pydantic request and response schemas
data/                # Loughran-McDonald Master Dictionary (CSV)
tests/               # Scorer tests
```
 
### Submit a pull request
 
Fork the repository and open a pull request to `main` with a focused change and a test for any behavior change.
 
## Data and License
 
The code is released under the [MIT License](LICENSE). The dictionary is the work of Tim Loughran and Bill McDonald and is distributed through the Notre Dame Software Repository for Accounting and Finance under its own terms. Check those terms before any commercial use.
 
> Loughran, T. and McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65.
