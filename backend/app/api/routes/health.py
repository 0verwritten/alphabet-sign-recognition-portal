from fastapi import APIRouter

from app.models import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="API health check")
def get_health() -> HealthResponse:
    """Return a static response confirming the service is online."""

    return HealthResponse(status="ok", message="Alphabet Sign Recognition API is running")
