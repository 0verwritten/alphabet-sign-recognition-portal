from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_asl_recognition_service
from app.core.config import settings
from app.main import app
from app.models import ASLRecognitionResult, BoundingBox, HandPoseResult, Landmark
from app.services import (
    HandNotDetectedError,
    InvalidImageFormatError,
    ModelNotReadyError,
)


class _StubASLService:
    def __init__(
        self,
        *,
        result: ASLRecognitionResult | None = None,
        hand_pose_result: HandPoseResult | None = None,
        error: Exception | None = None
    ) -> None:
        self._result = result
        self._hand_pose_result = hand_pose_result
        self._error = error

    def predict(self, image_bytes: bytes) -> ASLRecognitionResult:
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("No result configured for stub")
        return self._result

    def extract_hand_pose(self, image_bytes: bytes) -> HandPoseResult:
        if self._error is not None:
            raise self._error
        if self._hand_pose_result is None:
            raise RuntimeError("No hand pose result configured for stub")
        return self._hand_pose_result


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


def test_recognize_asl_letter_from_fixture_image(client: TestClient) -> None:
    image_path = Path(__file__).resolve().parent / "asstes" / "letter-c-sign.png"
    image_bytes = image_path.read_bytes()

    expected = ASLRecognitionResult(letter="C", confidence=0.87, handedness="right")

    class _RecordingService(_StubASLService):
        def predict(self, image_bytes: bytes) -> ASLRecognitionResult:
            assert image_bytes == image_bytes_expected
            return super().predict(image_bytes)

    image_bytes_expected = image_bytes
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _RecordingService(result=expected)
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/recognitions",
        files={"file": (image_path.name, image_bytes, "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "letter": "C",
        "confidence": pytest.approx(0.87, rel=1e-3),
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


# Hand Pose Detection Tests


def test_detect_hand_pose_success(client: TestClient) -> None:
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    bbox = BoundingBox(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9, width=0.6, height=0.6)
    hand_pose = HandPoseResult(
        landmarks=landmarks,
        handedness="right",
        handedness_confidence=0.95,
        bounding_box=bbox,
    )
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _StubASLService(hand_pose_result=hand_pose)
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/hand-pose",
        files={"file": ("test.png", b"fake", "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["landmarks"]) == 21
    assert data["handedness"] == "right"
    assert data["handedness_confidence"] == pytest.approx(0.95, rel=1e-3)
    assert data["bounding_box"]["x_min"] == pytest.approx(0.2, rel=1e-3)


def test_detect_hand_pose_invalid_image(client: TestClient) -> None:
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _StubASLService(error=InvalidImageFormatError("bad image"))
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/hand-pose",
        files={"file": ("test.png", b"fake", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad image"


def test_detect_hand_pose_no_hand(client: TestClient) -> None:
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _StubASLService(error=HandNotDetectedError("no hand detected"))
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/hand-pose",
        files={"file": ("test.png", b"fake", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "no hand detected"


def test_detect_hand_pose_from_fixture_image(client: TestClient) -> None:
    image_path = Path(__file__).resolve().parent / "asstes" / "letter-c-sign.png"
    image_bytes = image_path.read_bytes()

    # Create sample landmarks for a C sign
    landmarks = [
        Landmark(x=0.5 + i * 0.01, y=0.5 + i * 0.01, z=0.0 - i * 0.001)
        for i in range(21)
    ]
    bbox = BoundingBox(x_min=0.3, y_min=0.2, x_max=0.7, y_max=0.8, width=0.4, height=0.6)
    expected = HandPoseResult(
        landmarks=landmarks,
        handedness="right",
        handedness_confidence=0.98,
        bounding_box=bbox,
    )

    class _RecordingService(_StubASLService):
        def extract_hand_pose(self, image_bytes: bytes) -> HandPoseResult:
            assert image_bytes == image_bytes_expected
            return super().extract_hand_pose(image_bytes)

    image_bytes_expected = image_bytes
    app.dependency_overrides[get_asl_recognition_service] = _override_service(
        _RecordingService(hand_pose_result=expected)
    )

    response = client.post(
        f"{settings.API_V1_STR}/asl/hand-pose",
        files={"file": (image_path.name, image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["landmarks"]) == 21
    assert data["handedness"] == "right"
    assert data["handedness_confidence"] == pytest.approx(0.98, rel=1e-3)
