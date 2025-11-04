"""ASL recognition service built on top of MediaPipe hand landmarks."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.models import ASLRecognitionResult

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
    from mediapipe.python.solutions.hands import Hands
    from sklearn.base import ClassifierMixin
    from app.ml.model import ASLClassifier


LOGGER = logging.getLogger(__name__)


class ASLRecognitionError(RuntimeError):
    """Base class for ASL recognition errors."""


class InvalidImageFormatError(ASLRecognitionError):
    """Raised when the input bytes cannot be parsed as an RGB image."""


class HandNotDetectedError(ASLRecognitionError):
    """Raised when no hands are detected in the supplied image."""


class ModelNotReadyError(ASLRecognitionError):
    """Raised when the classifier is missing or not ready for inference."""


class MediapipeNotAvailableError(ModelNotReadyError):
    """Raised when MediaPipe is not installed or cannot be imported."""


@dataclass(slots=True)
class _ClassificationOutput:
    label: str
    confidence: float | None


class ASLRecognitionService:
    """Encapsulates the ASL recognition pipeline."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        min_detection_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
        max_num_hands: int | None = None,
        device: str | None = None,
        use_mock: bool = False,
    ) -> None:
        self._model_path = self._resolve_model_path(model_path)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._use_mock = use_mock
        self._classifier, self._is_pytorch = self._load_classifier(self._model_path)
        self._label_mapping: dict[int, str] | None = None
        self._detector: Hands | None = None  # type: ignore[name-defined]
        self._lock = Lock()
        self._min_detection_confidence = (
            min_detection_confidence
            if min_detection_confidence is not None
            else settings.ASL_MIN_DETECTION_CONFIDENCE
        )
        self._min_tracking_confidence = (
            min_tracking_confidence
            if min_tracking_confidence is not None
            else settings.ASL_MIN_TRACKING_CONFIDENCE
        )
        self._max_num_hands = (
            max_num_hands if max_num_hands is not None else settings.ASL_MAX_NUM_HANDS
        )
        self._hand_connections: list[tuple[int, int]] | None = None

        # Set up mock mode if enabled
        if self._use_mock:
            LOGGER.info("Mock mode enabled - predictions will be randomized")
            if self._label_mapping is None:
                self._label_mapping = {i: chr(65 + i) for i in range(26)}

    @classmethod
    def from_settings(cls) -> ASLRecognitionService:
        """Create a service instance using values defined in the settings."""

        return cls(model_path=settings.ASL_CLASSIFIER_PATH)

    def predict(self, image_bytes: bytes) -> ASLRecognitionResult:
        """Predict the ASL letter represented by the provided image."""

        if not image_bytes:
            raise InvalidImageFormatError("No image data received.")

        image = self._load_image(image_bytes)
        landmarks, handedness = self._detect_hand_landmarks(image)
        if landmarks is None:
            raise HandNotDetectedError("No hands were detected in the provided image.")

        features = self._extract_feature_vector(landmarks)
        classification = self._classify(features)

        return ASLRecognitionResult(
            letter=classification.label,
            confidence=classification.confidence,
            handedness=handedness,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""

        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __del__(self) -> None:  # pragma: no cover - invoked by GC
        self.close()

    def _resolve_model_path(self, model_path: str | Path | None) -> Path | None:
        if model_path is None:
            return None

        path = Path(model_path)
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            if cwd_candidate.exists():
                return cwd_candidate
            package_candidate = Path(__file__).resolve().parent.parent / path
            if package_candidate.exists():
                return package_candidate
        return path

    def _load_classifier(
        self, model_path: Path | None
    ) -> tuple[ClassifierMixin | torch.nn.Module | None, bool]:  # type: ignore[name-defined]
        """
        Load classifier from path. Supports both PyTorch (.pt, .pth) and scikit-learn (.joblib, .pkl).

        Returns:
            Tuple of (classifier, is_pytorch_model)
        """
        if model_path is None:
            LOGGER.info("No ASL classifier configured; predictions will be disabled.")
            return None, False

        if not model_path.exists():
            LOGGER.warning("ASL classifier not found at %s", model_path)
            return None, False

        try:
            # Detect model type by extension
            suffix = model_path.suffix.lower()

            if suffix in {".pt", ".pth"}:
                # Load PyTorch model
                checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    # Import here to avoid circular imports
                    from app.ml.model import create_model

                    # Extract metadata from checkpoint
                    model_config = checkpoint.get("model_config", {})
                    model_type = model_config.get("model_type", "v1")
                    input_size = model_config.get("input_size", 84)
                    num_classes = model_config.get("num_classes", 26)

                    # Create model instance
                    model = create_model(
                        model_type=model_type,
                        input_size=input_size,
                        num_classes=num_classes,
                    )
                    model.load_state_dict(checkpoint["model_state_dict"])
                    model.to(self._device)
                    model.eval()

                    # Load label mapping if available
                    if "label_mapping" in checkpoint:
                        self._label_mapping = checkpoint["label_mapping"]
                    else:
                        # Default A-Z mapping
                        self._label_mapping = {i: chr(65 + i) for i in range(26)}

                    LOGGER.info("Loaded PyTorch model from %s on device %s", model_path, self._device)
                    return model, True
                else:
                    # Direct model state dict (legacy format)
                    model = checkpoint
                    model.to(self._device)
                    model.eval()
                    self._label_mapping = {i: chr(65 + i) for i in range(26)}
                    LOGGER.info("Loaded PyTorch model from %s on device %s", model_path, self._device)
                    return model, True

            else:
                # Load scikit-learn model
                classifier: ClassifierMixin = joblib.load(model_path)
                LOGGER.info("Loaded scikit-learn model from %s", model_path)
                return classifier, False

        except Exception as exc:  # pragma: no cover - defensive programming
            LOGGER.exception("Failed to load ASL classifier from %s", model_path)
            raise ModelNotReadyError(
                f"Could not load ASL classifier from '{model_path}'."
            ) from exc

    def _import_mediapipe(self) -> Any:
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise MediapipeNotAvailableError(
                "mediapipe is required for ASL recognition. Install it to enable this endpoint."
            ) from exc
        return mp

    def _get_detector(self) -> Hands:  # type: ignore[name-defined]
        if self._detector is None:
            mp = self._import_mediapipe()
            self._hand_connections = list(mp.solutions.hands.HAND_CONNECTIONS)
            self._detector = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=self._max_num_hands,
                model_complexity=1,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
        return self._detector

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb_image = ImageOps.exif_transpose(image).convert("RGB")
                np_image = np.asarray(rgb_image, dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageFormatError("Uploaded file is not a supported image.") from exc

        if np_image.ndim != 3 or np_image.shape[2] != 3:
            raise InvalidImageFormatError(
                "Uploaded file is not a valid 3-channel RGB image."
            )

        return np_image

    def _detect_hand_landmarks(
        self, image: np.ndarray
    ) -> tuple[NormalizedLandmarkList | None, str | None]:  # type: ignore[name-defined]
        detector = self._get_detector()
        # image.flags.writeable = False
        with self._lock:
            results = detector.process(image)
        # image.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, None

        handedness: str | None = None
        if results.multi_handedness:
            classification = results.multi_handedness[0].classification[0]
            handedness = classification.label.lower()

        return results.multi_hand_landmarks[0], handedness

    def _extract_feature_vector(self, landmarks: NormalizedLandmarkList) -> np.ndarray:  # type: ignore[name-defined]
        coords = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float32
        )
        wrist = coords[0]
        centered = coords - wrist
        max_norm = np.linalg.norm(centered, axis=1).max()
        if max_norm > 0:
            centered /= max_norm

        flat_landmarks = centered.flatten()

        connection_distances: list[float] = []
        if self._hand_connections:
            for start_idx, end_idx in self._hand_connections:
                diff = centered[start_idx] - centered[end_idx]
                connection_distances.append(float(np.linalg.norm(diff)))

        feature_vector = (
            np.concatenate([flat_landmarks, np.asarray(connection_distances, dtype=np.float32)])
            if connection_distances
            else flat_landmarks
        )
        return feature_vector.astype(np.float32)

    def _classify(self, features: np.ndarray) -> _ClassificationOutput:
        # Use mock prediction if enabled
        if self._use_mock:
            return self._classify_mock(features)

        if self._classifier is None:
            target = self._model_path if self._model_path else "<not configured>"
            raise ModelNotReadyError(
                "ASL classifier is not configured. "
                f"Set ASL_CLASSIFIER_PATH to a trained model (expected at {target})."
            )

        try:
            if self._is_pytorch:
                return self._classify_pytorch(features)
            else:
                return self._classify_sklearn(features)
        except Exception as exc:  # pragma: no cover - defensive programming
            LOGGER.exception("Classifier failed to produce a prediction")
            raise ModelNotReadyError("ASL classifier failed during inference.") from exc

    def _classify_pytorch(self, features: np.ndarray) -> _ClassificationOutput:
        """Classify using PyTorch model."""
        # Convert to tensor and move to device
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self._device)

        # Get predictions
        with torch.no_grad():
            logits = self._classifier(features_tensor)  # type: ignore[misc]
            probabilities = torch.softmax(logits, dim=1)
            confidence_tensor, predicted_class = torch.max(probabilities, dim=1)

        predicted_idx = predicted_class.item()
        confidence = confidence_tensor.item()

        # Map index to label
        if self._label_mapping:
            label = self._label_mapping[predicted_idx]
        else:
            # Fallback to A-Z mapping
            label = chr(65 + predicted_idx) if predicted_idx < 26 else str(predicted_idx)

        return _ClassificationOutput(label=label, confidence=confidence)

    def _classify_sklearn(self, features: np.ndarray) -> _ClassificationOutput:
        """Classify using scikit-learn model."""
        features_2d = features.reshape(1, -1)
        predicted = self._classifier.predict(features_2d)  # type: ignore[union-attr]
        label = str(predicted[0])

        confidence: float | None = None
        if hasattr(self._classifier, "predict_proba"):
            try:
                probabilities = self._classifier.predict_proba(features_2d)  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - classifier-specific behaviour
                probabilities = None
            if probabilities is not None:
                best_idx = int(np.argmax(probabilities[0]))
                confidence = float(probabilities[0][best_idx])
                if hasattr(self._classifier, "classes_"):
                    label = str(self._classifier.classes_[best_idx])
        elif hasattr(self._classifier, "decision_function"):
            decision = self._classifier.decision_function(features_2d)  # type: ignore[union-attr]
            if np.ndim(decision) == 1:
                confidence = float(1 / (1 + np.exp(-float(decision[0]))))
            else:
                logits = np.asarray(decision[0], dtype=np.float64)
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / exp_logits.sum()
                best_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[best_idx])
                if hasattr(self._classifier, "classes_"):
                    label = str(self._classifier.classes_[best_idx])

        return _ClassificationOutput(label=label, confidence=confidence)

    def _classify_mock(self, features: np.ndarray) -> _ClassificationOutput:
        """
        Mock classifier for testing without a trained model.

        Returns a deterministic prediction based on feature hash for consistency,
        with randomized confidence scores.

        Args:
            features: Feature vector (not used in mock mode, but kept for compatibility)

        Returns:
            Mock classification output
        """
        # Use feature hash to get deterministic but varied predictions
        feature_hash = hash(features.tobytes()) % 26

        # Generate a confidence score based on the hash (between 0.65 and 0.98)
        confidence_base = (hash(features.tobytes()) % 33) / 100  # 0.00 to 0.32
        confidence = 0.65 + confidence_base

        # Map to label
        label = self._label_mapping[feature_hash] if self._label_mapping else chr(65 + feature_hash)

        LOGGER.debug(f"Mock prediction: {label} (confidence: {confidence:.2f})")

        return _ClassificationOutput(label=label, confidence=confidence)


__all__ = [
    "ASLRecognitionService",
    "ASLRecognitionError",
    "HandNotDetectedError",
    "InvalidImageFormatError",
    "ModelNotReadyError",
    "MediapipeNotAvailableError",
]
