"""Integration tests for ASL recognition service with PyTorch models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.model import ASLClassifier
from app.services.asl_recognition import ASLRecognitionService, ModelNotReadyError


@pytest.fixture
def mock_pytorch_model(tmp_path: Path) -> Path:
    """Create a mock PyTorch model file for testing."""
    model = ASLClassifier(input_size=84, num_classes=26)

    # Create checkpoint with metadata
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 10,
        "val_accuracy": 0.95,
        "model_config": {
            "model_type": "v1",
            "input_size": 84,
            "num_classes": 26,
        },
        "label_mapping": {i: chr(65 + i) for i in range(26)},
    }

    model_path = tmp_path / "test_model.pt"
    torch.save(checkpoint, model_path)

    return model_path


@pytest.fixture
def mock_pytorch_model_legacy(tmp_path: Path) -> Path:
    """Create a legacy PyTorch model file (just state dict)."""
    model = ASLClassifier(input_size=84, num_classes=26)
    model_path = tmp_path / "legacy_model.pt"
    torch.save(model, model_path)

    return model_path


class TestPyTorchModelLoading:
    """Tests for loading PyTorch models."""

    def test_load_pytorch_model_with_checkpoint(self, mock_pytorch_model: Path) -> None:
        """Test loading PyTorch model from checkpoint."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        assert service._classifier is not None
        assert service._is_pytorch is True
        assert service._label_mapping is not None

    def test_load_pytorch_model_device_selection(self, mock_pytorch_model: Path) -> None:
        """Test device selection for PyTorch model."""
        service = ASLRecognitionService(model_path=mock_pytorch_model, device="cpu")

        assert service._device == "cpu"

    def test_model_not_found_returns_none(self, tmp_path: Path) -> None:
        """Test loading nonexistent model returns None."""
        nonexistent = tmp_path / "nonexistent.pt"
        service = ASLRecognitionService(model_path=nonexistent)

        assert service._classifier is None

    def test_no_model_path_returns_none(self) -> None:
        """Test service without model path."""
        service = ASLRecognitionService(model_path=None)

        assert service._classifier is None


class TestPyTorchInference:
    """Tests for PyTorch model inference."""

    def test_classify_pytorch_features(self, mock_pytorch_model: Path) -> None:
        """Test classification with PyTorch model."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        # Create mock features
        features = np.random.randn(84).astype(np.float32)

        result = service._classify(features)

        assert result.label in [chr(65 + i) for i in range(26)]  # A-Z
        assert result.confidence is not None
        assert 0 <= result.confidence <= 1

    def test_classify_pytorch_returns_consistent_results(
        self, mock_pytorch_model: Path
    ) -> None:
        """Test PyTorch classification is consistent for same input."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        features = np.random.randn(84).astype(np.float32)

        result1 = service._classify(features)
        result2 = service._classify(features)

        # Should return same result for same input
        assert result1.label == result2.label
        assert result1.confidence == result2.confidence

    def test_classify_without_model_raises_error(self) -> None:
        """Test classification without model raises error."""
        service = ASLRecognitionService(model_path=None)

        features = np.random.randn(84).astype(np.float32)

        with pytest.raises(ModelNotReadyError, match="not configured"):
            service._classify(features)

    def test_pytorch_model_on_cpu(self, mock_pytorch_model: Path) -> None:
        """Test PyTorch model works on CPU."""
        service = ASLRecognitionService(model_path=mock_pytorch_model, device="cpu")

        features = np.random.randn(84).astype(np.float32)
        result = service._classify(features)

        assert result.label is not None
        assert result.confidence is not None


class TestModelTypeDetection:
    """Tests for automatic model type detection."""

    def test_detect_pytorch_model_by_extension(self, mock_pytorch_model: Path) -> None:
        """Test PyTorch model detected by .pt extension."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        assert service._is_pytorch is True

    def test_detect_pytorch_pth_extension(self, tmp_path: Path) -> None:
        """Test PyTorch model detected by .pth extension."""
        model = ASLClassifier()
        model_path = tmp_path / "model.pth"

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "model_type": "v1",
                "input_size": 84,
                "num_classes": 26,
            },
        }
        torch.save(checkpoint, model_path)

        service = ASLRecognitionService(model_path=model_path)

        assert service._is_pytorch is True


class TestLabelMapping:
    """Tests for label mapping."""

    def test_label_mapping_from_checkpoint(self, mock_pytorch_model: Path) -> None:
        """Test label mapping loaded from checkpoint."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        assert service._label_mapping is not None
        assert len(service._label_mapping) == 26
        assert service._label_mapping[0] == "A"
        assert service._label_mapping[25] == "Z"

    def test_classification_uses_label_mapping(self, mock_pytorch_model: Path) -> None:
        """Test classification uses label mapping."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        features = np.random.randn(84).astype(np.float32)
        result = service._classify(features)

        # Result should be a letter A-Z
        assert len(result.label) == 1
        assert result.label.isupper()
        assert "A" <= result.label <= "Z"


class TestFeatureExtraction:
    """Tests for feature extraction (existing MediaPipe functionality)."""

    def test_extract_feature_vector_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test feature extraction produces correct shape."""
        import types

        service = ASLRecognitionService(model_path=None)

        # Create fake landmarks
        class _Landmark:
            __slots__ = ("x", "y", "z")

            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        landmarks = [
            _Landmark(float(i) / 40.0, float(i + 1) / 40.0, float(i + 2) / 40.0)
            for i in range(21)
        ]
        fake_landmarks = types.SimpleNamespace(landmark=landmarks)

        # Mock hand connections
        service._hand_connections = [(i, i + 1) for i in range(20)]

        features = service._extract_feature_vector(fake_landmarks)

        # 21 landmarks × 3 coords + 20 distances = 83 features
        # (Will be 84 if all connections are used)
        assert features.shape[0] >= 63  # At least landmark features
        assert features.dtype == np.float32


class TestServiceLifecycle:
    """Tests for service initialization and cleanup."""

    def test_service_initialization(self, mock_pytorch_model: Path) -> None:
        """Test service initializes correctly."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        assert service._classifier is not None
        assert service._detector is None  # Lazy loaded
        assert service._lock is not None

    def test_service_cleanup(self, mock_pytorch_model: Path) -> None:
        """Test service cleanup."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        # Initialize detector
        service._get_detector()
        assert service._detector is not None

        # Cleanup
        service.close()
        assert service._detector is None

    def test_service_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating service from settings."""
        from app.core.config import settings

        # Mock settings
        monkeypatch.setattr(settings, "ASL_CLASSIFIER_PATH", None)

        service = ASLRecognitionService.from_settings()

        assert service is not None


class TestEndToEndInference:
    """End-to-end integration tests."""

    def test_inference_pipeline_with_pytorch(
        self, mock_pytorch_model: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test complete inference pipeline with PyTorch model."""
        import types
        from PIL import Image

        service = ASLRecognitionService(model_path=mock_pytorch_model)

        # Create test image
        test_image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        import io

        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format="PNG")
        image_bytes = img_bytes.getvalue()

        # Mock hand detection
        class _Landmark:
            __slots__ = ("x", "y", "z")

            def __init__(self, x: float, y: float, z: float) -> None:
                self.x = x
                self.y = y
                self.z = z

        landmarks = [
            _Landmark(float(i) / 40.0, float(i + 1) / 40.0, float(i + 2) / 40.0)
            for i in range(21)
        ]
        fake_landmarks = types.SimpleNamespace(landmark=landmarks)

        def mock_detect(
            _service: ASLRecognitionService, image: np.ndarray
        ) -> tuple[types.SimpleNamespace, str]:
            return fake_landmarks, "right"

        monkeypatch.setattr(
            service,
            "_detect_hand_landmarks",
            mock_detect.__get__(service, ASLRecognitionService),
        )

        # Run prediction
        result = service.predict(image_bytes)

        assert result.letter in [chr(65 + i) for i in range(26)]
        assert result.confidence is not None
        assert 0 <= result.confidence <= 1
        assert result.handedness == "right"


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_features_shape(self, mock_pytorch_model: Path) -> None:
        """Test error handling for invalid feature shape."""
        service = ASLRecognitionService(model_path=mock_pytorch_model)

        # Wrong shape
        wrong_features = np.random.randn(50).astype(np.float32)

        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):  # Will raise during model forward pass
            service._classify(wrong_features)

    def test_corrupted_model_file(self, tmp_path: Path) -> None:
        """Test handling of corrupted model file."""
        corrupt_path = tmp_path / "corrupt.pt"
        corrupt_path.write_text("This is not a valid PyTorch model")

        with pytest.raises(ModelNotReadyError):
            ASLRecognitionService(model_path=corrupt_path)


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_service_works_without_pytorch_model(self, tmp_path: Path) -> None:
        """Test service still works without PyTorch (no model)."""
        service = ASLRecognitionService(model_path=None)

        # Should initialize without errors
        assert service._classifier is None
        assert service._is_pytorch is False
