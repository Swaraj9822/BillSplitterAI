# Local AI backend for Bill Splitter

This FastAPI service provides:
- /parse_text: spaCy NER (PERSON, MONEY, DATE) + heuristics to convert sentences into payer/amount/participants/date/description.
- /transcribe: faster-whisper speech-to-text for short audio snippets recorded from the browser.

## Setup

# Create and activate a virtual environment (example with venv)
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

## Run
uvicorn app:app --host 127.0.0.1 --port 8000

Keep this running while using the frontend so the site can call http://127.0.0.1:8000.

## Notes

- spaCy en_core_web_sm includes NER with PERSON, MONEY, and DATE entities used by the parser.
- faster-whisper "small" with int8 compute runs fully local on CPU; switch model size for accuracy/speed trade-offs.
- Browser MediaRecorder requires HTTPS or localhost for mic capture; a GitHub Pages site (HTTPS) may call localhost with proper CORS, but mixed-content and private network policies vary by browser; if blocked, open index.html locally or run via a local HTTP server while testing.
