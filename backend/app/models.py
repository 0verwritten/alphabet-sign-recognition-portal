from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Simple response model confirming that the API is reachable."""

    status: str
    message: str
