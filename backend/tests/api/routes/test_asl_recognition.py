from __future__ import annotations

from typing import Callable

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_asl_recognition_service
from app.core.config import settings
from app.main import app
from app.models import ASLRecognitionResult
from app.services import (
    HandNotDetectedError,
    InvalidImageFormatError,
    ModelNotReadyError,
)


class _StubASLService:
    def __init__(self, *, result: ASLRecognitionResult | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def predict(self, image_bytes: bytes) -> ASLRecognitionResult:
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("No result configured for stub")
        return self._result


def _override_service(service: _StubASLService) -> Callable[[], _StubASLService]:
    return lambda: service


@pytest.fixture(autouse=True)
def clear_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_recognize_asl_letter_success(client: TestClient) -> None:
    result = ASLRecognitionResult(letter="A", confidence=0.92, handedness="right")
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _StubASLService(result=result)
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/recognitions",
        files={"file": ("test.png", b"fake", "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "letter": "A",
        "confidence": pytest.approx(0.92, rel=1e-3),
        "handedness": "right",
    }


def test_recognize_asl_letter_invalid_image(client: TestClient) -> None:
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _StubASLService(error=InvalidImageFormatError("bad image"))
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/recognitions",
        files={"file": ("test.png", b"fake", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad image"


def test_recognize_asl_letter_no_hand(client: TestClient) -> None:
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _StubASLService(error=HandNotDetectedError("no hand"))
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/recognitions",
        files={"file": ("test.png", b"fake", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "no hand"


def test_recognize_asl_letter_model_not_ready(client: TestClient) -> None:
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _StubASLService(error=ModelNotReadyError("classifier missing"))
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/recognitions",
        files={"file": ("test.png", b"fake", "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "classifier missing"
