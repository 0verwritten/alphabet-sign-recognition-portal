# FastAPI Project - Backend

## ASL Recognition System

This backend includes a complete ASL (American Sign Language) alphabet recognition system using MediaPipe for hand detection and PyTorch for classification.

### Quick Start with ASL Recognition

1. **Install dependencies**: `uv sync`
2. **Get dataset**: Download [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
3. **Prepare data**: `python scripts/prepare_dataset.py --input-dir data/asl_alphabet/ --output-file data/asl_features.npz`
4. **Train model**: `python scripts/train_model.py --data-file data/asl_features.npz --output-dir models/`
5. **Configure**: Set `ASL_CLASSIFIER_PATH=models/best_model.pt` in `.env`
6. **Start API**: `uvicorn app.main:app --reload`

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed instructions.

### Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started in 15 minutes
- [Full Training Guide](docs/PYTORCH_MODEL_TRAINING.md) - Comprehensive training documentation
- [Migration Guide](docs/PYTORCH_MIGRATION.md) - PyTorch implementation details
- [API Documentation](SIGN_RECOGNITION.md) - ASL recognition endpoint

## Requirements

* [Docker](https://www.docker.com/).
* [uv](https://docs.astral.sh/uv/) for Python package and environment management.

## Docker Compose

Start the local development environment with Docker Compose following the guide in [../development.md](../development.md).

## General Workflow

Dependencies are managed with [uv](https://docs.astral.sh/uv/). From `./backend/` you can install them with:

```console
$ uv sync
```

Then activate the virtual environment with:

```console
$ source .venv/bin/activate
```

Make sure your editor is using the interpreter at `backend/.venv/bin/python`.

The backend now exposes a lightweight mock API with a single health-check endpoint. You can adjust or extend the mock responses in `./backend/app/api/routes/` and related models in `./backend/app/models.py`.

## VS Code

Configurations are in place to run the backend through the VS Code debugger. You can also run the tests through the VS Code Python tests tab.

## Docker Compose Override

During development you can change Docker Compose settings that only affect the local environment in `docker-compose.override.yml`.

For example, the backend service runs with code mounted as a volume and `fastapi run --reload`, allowing live reload on changes. If you introduce a syntax error the container exits; fix the error and restart with:

```console
$ docker compose watch
```

You can open a shell in the running container with:

```console
$ docker compose exec backend bash
```

and run the live-reload server manually if needed:

```console
$ fastapi run --reload app/main.py
```

## Backend tests

To run the backend tests execute:

```console
$ bash ./scripts/test.sh
```

The tests use Pytest and validate the mock API behaviour. Adjust or add new tests in `./backend/tests/`.

If the stack is already running you can execute:

```bash
docker compose exec backend bash scripts/tests-start.sh
```

Additional arguments are forwarded to `pytest`, e.g. stop on first failure:

```bash
docker compose exec backend bash scripts/tests-start.sh -x
```

### Test Coverage

When the tests are run, a `htmlcov/index.html` report is generated.

## Database and migrations

This simplified backend does not use a database. The previous SQLModel models, CRUD utilities, and Alembic migrations have been removed. Use the mock routes as a starting point when integrating future services.
