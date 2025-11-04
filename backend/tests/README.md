# Test Suite for ASL Recognition System

This directory contains comprehensive tests for the ASL (American Sign Language) recognition system, including tests for PyTorch models, dataset utilities, and API endpoints.

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and test configuration
├── __init__.py
├── test_health.py                   # Health check endpoint tests
├── ml/                              # Machine learning component tests
│   ├── __init__.py
│   ├── test_model.py               # PyTorch model architecture tests
│   └── test_dataset.py             # Dataset utilities tests
├── services/                        # Service layer tests
│   ├── test_asl_recognition_service.py      # Service tests (existing)
│   └── test_asl_pytorch_integration.py      # PyTorch integration tests
└── api/                             # API endpoint tests
    └── routes/
        └── test_asl_recognition.py  # ASL API endpoint tests
```

## Train model

Install dependencies: cd backend && uv sync
Download dataset: ASL Alphabet Dataset
Prepare data: python scripts/prepare_dataset.py --input-dir data/asl_alphabet/ --output-file data/asl_features.npz
Train model: python scripts/train_model.py --data-file data/asl_features.npz --output-dir models/
Configure: Set ASL_CLASSIFIER_PATH=models/best_model.pt in .env
Start API: uvicorn app.main:app --reload


## Running Tests

### Run All Tests

```bash
cd backend
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test PyTorch models
pytest tests/ml/test_model.py -v

# Test dataset utilities
pytest tests/ml/test_dataset.py -v

# Test ASL service
pytest tests/services/test_asl_recognition_service.py -v

# Test PyTorch integration
pytest tests/services/test_asl_pytorch_integration.py -v

# Test API endpoints
pytest tests/api/routes/test_asl_recognition.py -v
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest tests/ -v -m "not integration"

# Run only integration tests
pytest tests/ -v -m integration
```

### Run with Coverage

```bash
pytest tests/ --cov=app --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Test Categories

### 1. Model Tests (`tests/ml/test_model.py`)

Tests for PyTorch model architectures:

**ASLClassifier (V1) Tests:**
- Model creation with default and custom parameters
- Forward pass output shapes
- Prediction methods (class indices and probabilities)
- Gradient flow
- Batch normalization and dropout layers

**ASLClassifierV2 Tests:**
- Model creation with residual connections
- Forward pass functionality
- Residual block presence

**ResidualBlock Tests:**
- Same and different dimensions
- Skip connection types (identity vs projection)

**Factory Function Tests:**
- Model creation with `create_model()`
- Invalid model type handling

**Integration Tests:**
- Model reproducibility
- Training vs eval mode behavior
- Model save/load functionality
- Different batch sizes
- Parameter counts

### 2. Dataset Tests (`tests/ml/test_dataset.py`)

Tests for dataset utilities:

**ASLDataset Tests:**
- Loading from .npz files
- Loading from directory structure
- Feature caching
- Label mappings
- Class weight calculation
- Transform application
- Error handling (empty/nonexistent paths)

**Train/Val Split Tests:**
- Correct split ratios
- Reproducibility with seeds
- No overlap between train/val sets

**Integration Tests:**
- DataLoader compatibility
- Full training loop simulation
- Dataset iteration and random access

### 3. Service Tests

#### Existing Service Tests (`tests/services/test_asl_recognition_service.py`)

Tests for core service functionality:
- Image loading and validation
- Hand landmark detection (mocked)
- Feature extraction
- Classification (mocked)
- End-to-end prediction

#### PyTorch Integration Tests (`tests/services/test_asl_pytorch_integration.py`)

Tests for PyTorch-specific functionality:
- Model loading from checkpoints
- Device selection (CPU/GPU)
- Classification with PyTorch models
- Model type detection (.pt/.pth)
- Label mapping
- Feature extraction shape
- Service lifecycle (init/cleanup)
- End-to-end inference
- Error handling
- Backward compatibility

### 4. API Tests (`tests/api/routes/test_asl_recognition.py`)

Tests for REST API endpoints:
- Successful recognition
- Invalid image handling
- Hand not detected errors
- Model not ready errors
- Different status codes (200, 400, 422, 503)

## Test Fixtures

### Global Fixtures (`conftest.py`)

**Client Fixtures:**
- `client`: FastAPI test client

**Model Fixtures:**
- `mock_model`: ASLClassifier instance
- `mock_model_checkpoint`: Saved PyTorch checkpoint file

**Data Fixtures:**
- `sample_features`: Single feature vector (84,)
- `sample_batch_features`: Batch of features (16, 84)
- `sample_labels`: Sample labels (16,)
- `sample_dataset_npz`: Complete dataset file
- `sample_image_bytes`: Test image bytes

### Local Fixtures

Test files may define additional fixtures specific to their test cases.

## Writing New Tests

### Test Naming Convention

```python
def test_<component>_<behavior>_<expected_result>() -> None:
    """Test description."""
```

Examples:
- `test_model_creation_default_params()`
- `test_dataset_load_from_npz()`
- `test_service_predict_returns_result()`

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_example() -> None:
    """Test description."""
    # Arrange - Set up test data
    input_data = create_test_data()

    # Act - Execute the code under test
    result = function_under_test(input_data)

    # Assert - Verify the results
    assert result.is_valid()
    assert result.value == expected_value
```

### Using Fixtures

```python
def test_with_fixture(mock_model: ASLClassifier) -> None:
    """Test using fixture."""
    output = mock_model(torch.randn(1, 84))
    assert output.shape == (1, 26)
```

### Mocking External Dependencies

```python
def test_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test with mocked dependency."""
    def mock_function(*args, **kwargs):
        return "mocked_result"

    monkeypatch.setattr(module, "function", mock_function)
    # Run test
```

### Temporary Files

```python
def test_with_temp_file(tmp_path: Path) -> None:
    """Test with temporary file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    # Run test
```

## Testing Best Practices

### 1. Test Independence

Each test should be independent and not rely on other tests:

```python
# Good
def test_feature_a() -> None:
    data = setup_data()  # Each test sets up its own data
    assert feature_a(data) == expected

def test_feature_b() -> None:
    data = setup_data()
    assert feature_b(data) == expected

# Bad - test_b depends on test_a
def test_a():
    global shared_data
    shared_data = process()

def test_b():
    assert shared_data is not None
```

### 2. Use Descriptive Names

```python
# Good
def test_model_returns_26_classes_for_alphabet():
    ...

# Bad
def test_output():
    ...
```

### 3. Test One Thing at a Time

```python
# Good
def test_model_output_shape():
    assert output.shape == (1, 26)

def test_model_output_type():
    assert isinstance(output, torch.Tensor)

# Bad
def test_model_everything():
    assert output.shape == (1, 26)
    assert isinstance(output, torch.Tensor)
    assert output.sum() > 0
    # Too many assertions
```

### 4. Use Fixtures for Common Setup

```python
@pytest.fixture
def trained_model():
    model = ASLClassifier()
    # ... train model ...
    return model

def test_with_trained_model(trained_model):
    # Use the fixture
    assert trained_model.accuracy > 0.8
```

### 5. Test Edge Cases

```python
def test_model_with_empty_input():
    with pytest.raises(ValueError):
        model(torch.empty(0, 84))

def test_model_with_wrong_dimensions():
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 50))  # Wrong input size
```

## Test Coverage Goals

Target coverage by component:

- **Models (`app/ml/model.py`)**: > 95%
- **Dataset (`app/ml/dataset.py`)**: > 90%
- **Service (`app/services/asl_recognition.py`)**: > 85%
- **API Routes**: > 90%
- **Overall**: > 85%

## Continuous Integration

Tests are automatically run on:
- Every commit (pre-commit hook)
- Pull requests (GitHub Actions)
- Before deployment

## Common Issues and Solutions

### Issue: Tests are slow

**Solution**: Use fixtures and mocks to avoid expensive operations:
```python
# Instead of training a real model
@pytest.fixture
def mock_trained_model():
    return MagicMock()
```

### Issue: Tests fail on CI but pass locally

**Solution**: Ensure tests don't depend on:
- Absolute file paths
- Specific hardware (GPU)
- External services
- Local environment variables

### Issue: Flaky tests (pass sometimes, fail sometimes)

**Solution**:
- Set random seeds: `torch.manual_seed(42)`
- Avoid time-dependent tests
- Use deterministic operations

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [PyTorch testing guide](https://pytorch.org/docs/stable/testing.html)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Python testing best practices](https://docs.python-guide.org/writing/tests/)

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Add tests to cover edge cases
4. Update this README if adding new test categories
5. Run coverage report and maintain > 85% coverage

## Questions?

For questions about:
- **Test failures**: Check the error message and relevant test file
- **Writing new tests**: Follow examples in existing test files
- **Coverage**: Run `pytest --cov` and check `htmlcov/index.html`
- **CI issues**: Check GitHub Actions logs
