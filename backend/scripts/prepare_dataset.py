"""Script to prepare ASL dataset from images by extracting MediaPipe features."""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.asl_recognition import ASLRecognitionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def extract_features_from_directory(
    input_dir: Path,
    output_file: Path,
    service: ASLRecognitionService,
) -> None:
    """
    Extract MediaPipe features from directory of ASL images.

    Expected directory structure:
        input_dir/
            A/
                image1.jpg
                image2.jpg
            B/
                image1.jpg
            ...

    Args:
        input_dir: Directory containing class subdirectories
        output_file: Output .npz file path
        service: ASL recognition service for feature extraction
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Get class directories
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class directories found in {input_dir}")

    LOGGER.info("Found %d classes: %s", len(class_dirs), [d.name for d in class_dirs])

    # Prepare data structures
    all_features = []
    all_labels = []
    label_names = [d.name for d in class_dirs]
    label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    skipped = 0
    processed = 0

    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        label_idx = label_to_idx[class_name]

        LOGGER.info("Processing class: %s", class_name)

        # Get all image files
        image_files = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        # Process each image with progress bar
        progress_handler = tqdm(image_files, desc=f"Processing {class_name}", unit="image")
        for image_file in progress_handler:
            try:
                # Load image
                with open(image_file, "rb") as f:
                    image_bytes = f.read()

                # Load image as array for feature extraction
                image = Image.open(image_file).convert("RGB")
                image_array = np.array(image)

                # Detect hand landmarks
                landmarks, _ = service._detect_hand_landmarks(image_array)

                if landmarks is None:
                    progress_handler.set_description(f"Skipping {image_file.name} (no hand detected)")
                    # LOGGER.warning("No hand detected in %s, skipping", image_file.name)
                    skipped += 1
                    continue
                else:
                    progress_handler.set_description(f"Processing {image_file.name}")

                # Extract features
                features = service._extract_feature_vector(landmarks)

                all_features.append(features)
                all_labels.append(label_idx)
                processed += 1

            except Exception as e:
                LOGGER.error("Error processing %s: %s", image_file, e)
                skipped += 1
                continue

    if not all_features:
        raise ValueError("No features extracted! Check your images and MediaPipe installation.")

    # Convert to numpy arrays
    features_array = np.array(all_features, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int64)

    LOGGER.info(
        "Extracted features from %d images (%d skipped)",
        processed,
        skipped,
    )
    LOGGER.info("Feature shape: %s", features_array.shape)
    LOGGER.info("Labels shape: %s", labels_array.shape)

    # Save to npz file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        features=features_array,
        labels=labels_array,
        label_names=np.array(label_names),
    )

    LOGGER.info("Saved preprocessed dataset to %s", output_file)

    # Print statistics
    LOGGER.info("\nDataset Statistics:")
    for label_name in label_names:
        label_idx = label_to_idx[label_name]
        count = np.sum(labels_array == label_idx)
        LOGGER.info("  %s: %d samples", label_name, count)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe features from ASL images"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing class subdirectories with images",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output .npz file path",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for hand detection (default: 0.5)",
    )

    args = parser.parse_args()

    # Create ASL recognition service (without classifier, just for feature extraction)
    LOGGER.info("Initializing MediaPipe...")
    service = ASLRecognitionService(
        model_path=None,  # No classifier needed
        min_detection_confidence=args.min_detection_confidence,
    )

    # Extract features
    try:
        extract_features_from_directory(
            input_dir=args.input_dir,
            output_file=args.output_file,
            service=service,
        )
    finally:
        service.close()

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
