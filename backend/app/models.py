from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Simple response model confirming that the API is reachable."""

    status: str
    message: str

class ASLRecognitionResult(BaseModel):
    letter: str
    confidence: float | None
    handedness: str | None
