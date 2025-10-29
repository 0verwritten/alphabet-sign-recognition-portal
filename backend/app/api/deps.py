from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.services import ASLRecognitionService


def get_asl_recognition_service() -> ASLRecognitionService:
    return _get_cached_asl_service()


ASLServiceDep = Annotated[ASLRecognitionService, Depends(get_asl_recognition_service)]

@lru_cache(maxsize=1)
def _get_cached_asl_service() -> ASLRecognitionService:
    return ASLRecognitionService.from_settings()
