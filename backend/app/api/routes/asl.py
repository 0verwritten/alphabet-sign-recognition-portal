"""Routes for ASL recognition."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import ASLServiceDep
from app.models import ASLRecognitionResult, HandPoseResult
from app.services import (
    HandNotDetectedError,
    InvalidImageFormatError,
    MediapipeNotAvailableError,
    ModelNotReadyError,
)

router = APIRouter(prefix="/asl", tags=["asl"])


@router.post(
    "/recognitions",
    response_model=ASLRecognitionResult,
    status_code=status.HTTP_200_OK,
    summary="Recognize an ASL letter from an image",
)
async def recognize_asl_letter(
    *,
    file: UploadFile = File(..., description="RGB image containing an ASL letter gesture."),
    service: ASLServiceDep,
) -> ASLRecognitionResult:
    """Recognize the ASL letter present in the uploaded image."""

    image_bytes = await file.read()
    try:
        prediction = await run_in_threadpool(service.predict, image_bytes)
    except InvalidImageFormatError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HandNotDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except MediapipeNotAvailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return prediction


@router.post(
    "/hand-pose",
    response_model=HandPoseResult,
    status_code=status.HTTP_200_OK,
    summary="Extract hand pose landmarks from an image",
)
async def detect_hand_pose(
    *,
    file: UploadFile = File(..., description="RGB image containing a hand."),
    service: ASLServiceDep,
) -> HandPoseResult:
    """
    Extract 21 hand landmarks with 3D coordinates from the uploaded image.

    Returns detailed information about hand pose including:
    - 21 normalized landmarks (x, y, z coordinates)
    - Hand orientation (left/right) with confidence
    - Bounding box around the detected hand

    The landmarks follow MediaPipe Hands convention:
    - 0: WRIST
    - 1-4: THUMB (CMC, MCP, IP, TIP)
    - 5-8: INDEX_FINGER (MCP, PIP, DIP, TIP)
    - 9-12: MIDDLE_FINGER (MCP, PIP, DIP, TIP)
    - 13-16: RING_FINGER (MCP, PIP, DIP, TIP)
    - 17-20: PINKY (MCP, PIP, DIP, TIP)
    """
    image_bytes = await file.read()
    try:
        hand_pose = await run_in_threadpool(service.extract_hand_pose, image_bytes)
    except InvalidImageFormatError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HandNotDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except MediapipeNotAvailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return hand_pose
