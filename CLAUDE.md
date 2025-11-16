# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ASL (American Sign Language) alphabet recognition portal built on the Full Stack FastAPI Template. The backend uses MediaPipe for hand detection and PyTorch for classification to recognize ASL alphabet signs (A-Z) from images.

## Technology Stack

- **Backend**: FastAPI (Python 3.10+) with PyTorch and MediaPipe
- **Frontend**: React with TypeScript, Vite, and Chakra UI
- **Database**: PostgreSQL (SQLModel ORM)
- **Package Management**: uv for Python dependencies
- **Testing**: Pytest with coverage reporting
- **Linting**: Ruff, mypy for type checking
- **Deployment**: Docker Compose with Traefik reverse proxy
- **CI/CD**: GitHub Actions

## Development Commands

### Backend Development

All backend commands should be run from the `backend/` directory.

**Setup and Dependencies:**
```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

**Running the Server:**
```bash
# Development mode with auto-reload
uvicorn app.main:app --reload

# Or with specific host/port
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Linting and Type Checking:**
```bash
# Run all linting checks (mypy, ruff check, ruff format)
uv run bash scripts/lint.sh

# Individual commands
uv run mypy app
uv run ruff check app
uv run ruff format app --check

# Auto-format code
uv run ruff format app
```

**Testing:**
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/api/routes/test_asl_recognition.py

# Run tests with coverage
uv run bash scripts/test.sh
# Coverage report: htmlcov/index.html

# Run with markers
uv run pytest -m "not integration"  # Skip integration tests
uv run pytest -m integration        # Only integration tests

# Stop on first failure
uv run pytest -x
```

**Docker Development:**
```bash
# Start services with Docker Compose
docker compose up -d

# Watch for changes and reload
docker compose watch

# Run tests in container
docker compose exec backend bash scripts/tests-start.sh

# Open shell in container
docker compose exec backend bash
```

### ASL Model Training and Preparation

**Dataset Preparation:**
```bash
# Download ASL Alphabet Dataset from Kaggle first
# Then prepare features
python scripts/prepare_dataset.py --input-dir data/asl_alphabet/ --output-file data/asl_features.npz
```

**Training:**
```bash
# Train the PyTorch model
python scripts/train_model.py --data-file data/asl_features.npz --output-dir models/

# Evaluate model
python scripts/evaluate_model.py

# Test inference
python scripts/test_inference.py
```

**Configuration:**
Set `ASL_CLASSIFIER_PATH=models/best_model.pt` in `.env` to use the trained model.

## Architecture

### Backend Structure

```
backend/
├── app/
│   ├── api/           # API routes and endpoints
│   │   ├── routes/    # Individual route modules (health, asl)
│   │   ├── deps.py    # Dependency injection
│   │   └── main.py    # API router setup
│   ├── core/          # Core application settings
│   │   └── config.py  # Pydantic settings (ASL config, CORS, etc.)
│   ├── ml/            # Machine learning components
│   │   ├── model.py   # PyTorch model architectures (ASLClassifier, ASLClassifierV2)
│   │   └── dataset.py # Dataset utilities and loaders
│   ├── services/      # Business logic layer
│   │   └── asl_recognition.py  # Main ASL recognition service
│   ├── models.py      # Pydantic models for API contracts
│   └── main.py        # FastAPI app initialization
├── scripts/           # Development and ML training scripts
├── tests/             # Comprehensive test suite
│   ├── api/routes/    # API endpoint tests
│   ├── ml/            # Model and dataset tests
│   └── services/      # Service layer tests
└── docs/              # Additional documentation
```

### Key Components

**ASL Recognition Pipeline:**
1. **Image Input** → `ASLRecognitionService.predict(image_bytes)`
2. **Hand Detection** → MediaPipe Hands detector finds hand landmarks
3. **Feature Extraction** → Converts landmarks to feature vector (84 dimensions: 63 landmarks + 21 distance features)
4. **Classification** → PyTorch model (`ASLClassifier` or `ASLClassifierV2`) predicts A-Z
5. **Result** → Returns predicted letter with confidence score

**Model Types:**
- `ASLClassifier` (V1): 3-layer fully connected network with batch norm and dropout
- `ASLClassifierV2`: Enhanced model with residual connections
- Both models accept 84-dimensional input, output 26 classes (A-Z)

**Configuration (app/core/config.py):**
- `ASL_CLASSIFIER_PATH`: Path to trained .pt/.pth model file
- `ASL_MIN_DETECTION_CONFIDENCE`: Hand detection threshold (default: 0.5)
- `ASL_MIN_TRACKING_CONFIDENCE`: Hand tracking threshold (default: 0.5)
- `ASL_MAX_NUM_HANDS`: Max hands to detect (default: 1)

### API Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/asl/predict` - Upload image, returns predicted ASL letter
  - Returns: `{"letter": "A", "confidence": 0.95, "hand_pose": {...}}`
  - Error codes: 400 (invalid image), 422 (no hand detected), 503 (model not ready)

### Testing Strategy

Tests are organized by layer:
- **API tests** (`tests/api/routes/`): HTTP endpoint behavior, status codes
- **Service tests** (`tests/services/`): Business logic, error handling, PyTorch integration
- **ML tests** (`tests/ml/`): Model architecture, dataset utilities, training loop

Coverage targets:
- Models: >95%
- Dataset utilities: >90%
- Service layer: >85%
- Overall: >85%

## Important Development Notes

### PyTorch Model Loading
- Service supports both `.pt` and `.pth` checkpoint formats
- Models are automatically moved to GPU if available (CUDA detection)
- Label mapping is extracted from checkpoint `label_mapping` key
- Mock mode available for testing without trained model

### MediaPipe Integration
- Hand detection runs on CPU (MediaPipe limitation)
- Feature extraction normalizes landmarks to [0, 1] range
- Includes additional distance features between key hand points
- Lazy initialization - detector only loaded when needed

### Thread Safety
- `ASLRecognitionService` uses lock for thread-safe predictions
- Safe for use in async FastAPI context

### Error Handling
Service raises specific exceptions:
- `InvalidImageFormatError` - Cannot parse image bytes
- `HandNotDetectedError` - No hand found in image
- `ModelNotReadyError` - Classifier not loaded/configured
- `MediapipeNotAvailableError` - MediaPipe import failed

### Environment Configuration
- Main config in `.env` at repository root
- Backend loads from `../.env` relative path
- Required for production: `SECRET_KEY`, `POSTGRES_PASSWORD` (if using DB)
- ASL-specific: Only `ASL_CLASSIFIER_PATH` is required for predictions

## CI/CD

GitHub Actions workflows:
- **Lint Backend** (`lint-backend.yml`): Runs on push/PR, executes mypy + ruff
- **Test Backend** (`test-backend.yml`): Runs pytest suite
- Both use `uv` for dependency management
- Python 3.10 target

## Additional Resources

- Backend README: `backend/README.md` - Quick start guide, detailed workflow
- Test README: `backend/tests/README.md` - Comprehensive testing guide
- Sign Recognition Doc: `backend/SIGN_RECOGNITION.md` - API usage examples
- Model Training Guide: `backend/docs/PYTORCH_MODEL_TRAINING.md`
- Migration Guide: `backend/docs/PYTORCH_MIGRATION.md`
- in order to use node you need to enter node toolbox first using "toolbox enter node"