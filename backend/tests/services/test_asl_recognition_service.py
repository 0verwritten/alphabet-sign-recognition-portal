from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from app.services.asl_recognition import (
    ASLRecognitionService,
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

    def _fake_detect(self: ASLRecognitionService, image: np.ndarray):
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3 and image.shape[2] == 3
        return _make_fake_landmarks(), "right"

    def _fake_classify(self: ASLRecognitionService, features: np.ndarray) -> _ClassificationOutput:
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1
        return _ClassificationOutput(label="C", confidence=0.91)

    monkeypatch.setattr(service, "_detect_hand_landmarks", _fake_detect.__get__(service, ASLRecognitionService))
    monkeypatch.setattr(service, "_classify", _fake_classify.__get__(service, ASLRecognitionService))

    result = service.predict(image_bytes)

    assert result.letter == "C"
    assert result.confidence == pytest.approx(0.91, rel=1e-3)
    assert result.handedness == "right"
