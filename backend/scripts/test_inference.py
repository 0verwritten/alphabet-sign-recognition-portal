"""Simple script to test model inference on a single image."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.asl_recognition import ASLRecognitionService


def main() -> None:
    """Test inference on a single image."""
    parser = argparse.ArgumentParser(description="Test ASL recognition on an image")

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to image file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (.pt or .joblib)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image.exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Load image
    print(f"Loading image: {args.image}")
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    # Create service
    print(f"Loading model: {args.model}")
    service = ASLRecognitionService(
        model_path=args.model,
        device=args.device,
    )

    # Predict
    print("Running prediction...")
    try:
        result = service.predict(image_bytes)

        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Letter: {result.letter}")
        print(f"Confidence: {result.confidence:.2%}" if result.confidence else "Confidence: N/A")
        print(f"Handedness: {result.handedness or 'N/A'}")
        print("=" * 50)

    except Exception as e:
        print(f"\nError during prediction: {e}")
        sys.exit(1)

    finally:
        service.close()


if __name__ == "__main__":
    main()
