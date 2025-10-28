"""Routes for ASL recognition."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.deps import ASLServiceDep
from app.models import ASLRecognitionResult
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
        prediction = service.predict(image_bytes)
    except InvalidImageFormatError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except HandNotDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except MediapipeNotAvailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return prediction
