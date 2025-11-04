"""Demo script to test ASL recognition with mock predictions."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.asl_recognition import ASLRecognitionService


def main() -> None:
    """Demonstrate mock prediction functionality."""
    parser = argparse.ArgumentParser(
        description="Test ASL recognition with mock predictions (no trained model needed)"
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to image file",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image.exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # Load image
    print(f"Loading image: {args.image}")
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    # Create service with mock mode enabled
    print("Creating ASL Recognition Service in MOCK mode...")
    print("(No trained model required - predictions are deterministic based on hand features)\n")

    service = ASLRecognitionService(
        use_mock=True,  # Enable mock mode
        device="cpu",
    )

    # Predict
    print("Running prediction...")
    try:
        result = service.predict(image_bytes)

        print("\n" + "=" * 50)
        print("MOCK PREDICTION RESULT")
        print("=" * 50)
        print(f"Letter: {result.letter}")
        print(f"Confidence: {result.confidence:.2%}" if result.confidence else "Confidence: N/A")
        print(f"Handedness: {result.handedness or 'N/A'}")
        print("=" * 50)
        print("\nNote: This is a MOCK prediction for testing purposes.")
        print("To get real predictions, train a model using:")
        print("  python backend/scripts/train_model.py --data-file <data.npz>")

    except Exception as e:
        print(f"\nError during prediction: {e}")
        sys.exit(1)

    finally:
        service.close()


if __name__ == "__main__":
    main()
