# Sign Recognition Service

This document describes how to run the backend API and how to exercise the
American Sign Language (ASL) recognition service directly from the command
line.

## 1. Prerequisites

- Python 3.10 or newer
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

Install the backend dependencies:

```bash
cd backend
uv sync
```

> If you prefer `pip`, create a virtual environment and run `pip install .` in
> the `backend/` folder instead of `uv sync`.

## 2. Running the API server

1. Move into the backend directory: `cd backend`.
2. Make sure the environment variables for ASL recognition are configured. The
   most important ones are:
   - `ASL_CLASSIFIER_PATH`: optional path to a trained scikit-learn classifier
     saved with `joblib`. When omitted, the recognition endpoint loads but
     returns `503` until a model is configured.
   - `ASL_MIN_DETECTION_CONFIDENCE`: detection confidence threshold
     (default: `0.5`).
   - `ASL_MIN_TRACKING_CONFIDENCE`: tracking confidence threshold
     (default: `0.5`).
   - `ASL_MAX_NUM_HANDS`: maximum number of hands to track (default: `1`).
3. Start the server with Uvicorn:

   ```bash
   uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API is now available at <http://localhost:8000> and the automatic docs at
   <http://localhost:8000/docs>.

## 3. Running the recognition service directly

The recognition pipeline can be executed without the HTTP API. The snippet
below loads the `ASLRecognitionService`, reads an image from disk, and prints
the predicted letter:

```bash
uv run python - <<'PY'
from pathlib import Path

from app.services import ASLRecognitionService

# Update this path to point to your image file
image_path = Path("backend/tests/api/routes/asstes/letter-c-sign.png")
image_bytes = image_path.read_bytes()

service = ASLRecognitionService.from_settings()

try:
    result = service.predict(image_bytes)
    print(f"Predicted letter: {result.letter} (confidence={result.confidence})")
finally:
    service.close()
PY
```

If no classifier is configured, `ASLRecognitionService` raises
`ModelNotReadyError`. Provide a valid `ASL_CLASSIFIER_PATH` before running the
snippet to obtain predictions from a trained model.

## 4. Running the automated tests

To execute the backend test-suite (including the ASL recognition tests):

```bash
cd backend
uv run pytest
```

This command exercises the API tests and the unit test that validates the sign
recognition service with the bundled sample image.
