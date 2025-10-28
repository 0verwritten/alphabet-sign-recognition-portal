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
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.models import ASLRecognitionResult

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
    from mediapipe.python.solutions.hands import Hands
    from sklearn.base import ClassifierMixin


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
    ) -> None:
        self._model_path = self._resolve_model_path(model_path)
        self._classifier = self._load_classifier(self._model_path)
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

    @classmethod
    def from_settings(cls) -> "ASLRecognitionService":
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

    def _load_classifier(self, model_path: Path | None) -> ClassifierMixin | None:  # type: ignore[name-defined]
        if model_path is None:
            LOGGER.info("No ASL classifier configured; predictions will be disabled.")
            return None

        if not model_path.exists():
            LOGGER.warning("ASL classifier not found at %s", model_path)
            return None

        try:
            classifier: ClassifierMixin = joblib.load(model_path)
        except Exception as exc:  # pragma: no cover - defensive programming
            LOGGER.exception("Failed to load ASL classifier from %s", model_path)
            raise ModelNotReadyError(
                f"Could not load ASL classifier from '{model_path}'."
            ) from exc
        return classifier

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
        image.flags.writeable = False
        with self._lock:
            results = detector.process(image)
        image.flags.writeable = True

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
        if self._classifier is None:
            target = self._model_path if self._model_path else "<not configured>"
            raise ModelNotReadyError(
                "ASL classifier is not configured. "
                f"Set ASL_CLASSIFIER_PATH to a trained model (expected at {target})."
            )

        features_2d = features.reshape(1, -1)
        try:
            predicted = self._classifier.predict(features_2d)
            label = str(predicted[0])
        except Exception as exc:  # pragma: no cover - defensive programming
            LOGGER.exception("Classifier failed to produce a prediction")
            raise ModelNotReadyError("ASL classifier failed during inference.") from exc

        confidence: float | None = None
        if hasattr(self._classifier, "predict_proba"):
            try:
                probabilities = getattr(self._classifier, "predict_proba")(features_2d)
            except Exception:  # pragma: no cover - classifier-specific behaviour
                probabilities = None
            if probabilities is not None:
                best_idx = int(np.argmax(probabilities[0]))
                confidence = float(probabilities[0][best_idx])
                if hasattr(self._classifier, "classes_"):
                    label = str(self._classifier.classes_[best_idx])
        elif hasattr(self._classifier, "decision_function"):
            decision = getattr(self._classifier, "decision_function")(features_2d)
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


__all__ = [
    "ASLRecognitionService",
    "ASLRecognitionError",
    "HandNotDetectedError",
    "InvalidImageFormatError",
    "ModelNotReadyError",
    "MediapipeNotAvailableError",
]
