from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Simple response model confirming that the API is reachable."""

    status: str
    message: str

class ASLRecognitionResult(BaseModel):
    letter: str
    confidence: float | None
    handedness: str | None


class Landmark(BaseModel):
    """Represents a single hand landmark with 3D coordinates."""

    x: float = Field(..., description="Normalized x coordinate (0-1)")
    y: float = Field(..., description="Normalized y coordinate (0-1)")
    z: float = Field(..., description="Normalized z coordinate (depth)")


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected hand."""

    x_min: float = Field(..., description="Minimum x coordinate (0-1)")
    y_min: float = Field(..., description="Minimum y coordinate (0-1)")
    x_max: float = Field(..., description="Maximum x coordinate (0-1)")
    y_max: float = Field(..., description="Maximum y coordinate (0-1)")
    width: float = Field(..., description="Width of bounding box (0-1)")
    height: float = Field(..., description="Height of bounding box (0-1)")


class HandPoseResult(BaseModel):
    """Result of hand pose detection containing landmarks and metadata."""

    landmarks: list[Landmark] = Field(
        ...,
        description="List of 21 hand landmarks in MediaPipe order: "
        "WRIST(0), THUMB_CMC(1), THUMB_MCP(2), THUMB_IP(3), THUMB_TIP(4), "
        "INDEX_FINGER_MCP(5), INDEX_FINGER_PIP(6), INDEX_FINGER_DIP(7), INDEX_FINGER_TIP(8), "
        "MIDDLE_FINGER_MCP(9), MIDDLE_FINGER_PIP(10), MIDDLE_FINGER_DIP(11), MIDDLE_FINGER_TIP(12), "
        "RING_FINGER_MCP(13), RING_FINGER_PIP(14), RING_FINGER_DIP(15), RING_FINGER_TIP(16), "
        "PINKY_MCP(17), PINKY_PIP(18), PINKY_DIP(19), PINKY_TIP(20)",
    )
    handedness: str | None = Field(
        None, description="Hand orientation: 'left' or 'right'"
    )
    handedness_confidence: float | None = Field(
        None, description="Confidence score for handedness classification"
    )
    bounding_box: BoundingBox | None = Field(
        None, description="Bounding box around the detected hand"
    )
