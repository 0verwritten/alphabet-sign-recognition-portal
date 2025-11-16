from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from app.services.asl_recognition import (
    ASLRecognitionService,
    HandNotDetectedError,
    InvalidImageFormatError,
    _ClassificationOutput,
)


def _make_fake_landmarks() -> types.SimpleNamespace:
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
    return types.SimpleNamespace(landmark=landmarks)


def test_service_predicts_letter_c(monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = (
        Path(__file__).resolve().parent.parent / "api" / "routes" / "asstes" / "letter-c-sign.png"
    )
    image_bytes = image_path.read_bytes()

    service = ASLRecognitionService(model_path=None)

    def _fake_detect(_service: ASLRecognitionService, image: np.ndarray):
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3 and image.shape[2] == 3
        return _make_fake_landmarks(), "right"

    def _fake_classify(_service: ASLRecognitionService, features: np.ndarray) -> _ClassificationOutput:
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        return _ClassificationOutput(label="C", confidence=0.91)

    monkeypatch.setattr(service, "_detect_hand_landmarks", _fake_detect.__get__(service, ASLRecognitionService))
    monkeypatch.setattr(service, "_classify", _fake_classify.__get__(service, ASLRecognitionService))

    result = service.predict(image_bytes)

    assert result.letter == "C"
    assert result.confidence == pytest.approx(0.91, rel=1e-3)
    assert result.handedness == "right"


def test_extract_hand_pose_success(monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = (
        Path(__file__).resolve().parent.parent / "api" / "routes" / "asstes" / "letter-c-sign.png"
    )
    image_bytes = image_path.read_bytes()

    service = ASLRecognitionService(model_path=None)

    # Mock MediaPipe results
    class _FakeHandedness:
        def __init__(self):
            self.classification = [types.SimpleNamespace(label="Right", score=0.97)]

    class _FakeResults:
        def __init__(self):
            self.multi_hand_landmarks = [_make_fake_landmarks()]
            self.multi_handedness = [_FakeHandedness()]

    def _fake_process(_image):
        return _FakeResults()

    # Mock detector
    fake_detector = types.SimpleNamespace(process=_fake_process)
    monkeypatch.setattr(service, "_get_detector", lambda: fake_detector)

    result = service.extract_hand_pose(image_bytes)

    assert len(result.landmarks) == 21
    assert result.handedness == "right"
    assert result.handedness_confidence == pytest.approx(0.97, rel=1e-3)
    assert result.bounding_box is not None
    assert result.bounding_box.width > 0
    assert result.bounding_box.height > 0


def test_extract_hand_pose_no_hand(monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = (
        Path(__file__).resolve().parent.parent / "api" / "routes" / "asstes" / "letter-c-sign.png"
    )
    image_bytes = image_path.read_bytes()

    service = ASLRecognitionService(model_path=None)

    # Mock MediaPipe results with no hands
    class _FakeResults:
        def __init__(self):
            self.multi_hand_landmarks = []
            self.multi_handedness = []

    def _fake_process(_image):
        return _FakeResults()

    fake_detector = types.SimpleNamespace(process=_fake_process)
    monkeypatch.setattr(service, "_get_detector", lambda: fake_detector)

    with pytest.raises(HandNotDetectedError, match="No hands were detected"):
        service.extract_hand_pose(image_bytes)


def test_extract_hand_pose_invalid_image() -> None:
    service = ASLRecognitionService(model_path=None)

    with pytest.raises(InvalidImageFormatError, match="not a supported image"):
        service.extract_hand_pose(b"not an image")


def test_extract_hand_pose_empty_bytes() -> None:
    service = ASLRecognitionService(model_path=None)

    with pytest.raises(InvalidImageFormatError, match="No image data"):
        service.extract_hand_pose(b"")
