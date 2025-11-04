from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.ml.model import ASLClassifier


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature vector for testing."""
    return np.random.randn(84).astype(np.float32)


@pytest.fixture
def sample_batch_features() -> np.ndarray:
    """Generate batch of sample features for testing."""
    return np.random.randn(16, 84).astype(np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Generate sample labels for testing."""
    return np.random.randint(0, 26, size=16, dtype=np.int64)


@pytest.fixture
def mock_model() -> ASLClassifier:
    """Create a mock ASL classifier model."""
    model = ASLClassifier(input_size=84, num_classes=26)
    model.eval()
    return model


@pytest.fixture
def mock_model_checkpoint(tmp_path: Path) -> Path:
    """Create a mock model checkpoint file."""
    model = ASLClassifier(input_size=84, num_classes=26)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 50,
        "val_accuracy": 0.92,
        "model_config": {
            "model_type": "v1",
            "input_size": 84,
            "num_classes": 26,
        },
        "label_mapping": {i: chr(65 + i) for i in range(26)},
    }

    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


@pytest.fixture
def sample_dataset_npz(tmp_path: Path) -> Path:
    """Create a sample dataset .npz file."""
    num_samples = 100
    features = np.random.randn(num_samples, 84).astype(np.float32)
    labels = np.random.randint(0, 26, size=num_samples, dtype=np.int64)
    label_names = np.array([chr(65 + i) for i in range(26)])

    npz_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        npz_path,
        features=features,
        labels=labels,
        label_names=label_names,
    )

    return npz_path


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing."""
    from PIL import Image
    import io

    # Create a simple test image
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")

    return img_bytes.getvalue()
