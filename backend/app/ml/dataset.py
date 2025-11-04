"""Dataset utilities for ASL recognition training."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class ASLDataset(Dataset):
    """
    PyTorch Dataset for ASL alphabet recognition.

    Expects a directory structure like:
        data/
            A/
                image1.jpg
                image2.jpg
            B/
                image1.jpg
            ...
            Z/
                image1.jpg

    Or a preprocessed numpy file with features and labels.
    """

    def __init__(
        self,
        data_path: str | Path,
        feature_extractor: Any | None = None,
        transform: Any | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to directory with class subdirectories or .npz file
            feature_extractor: Function to extract features from images
            transform: Optional transforms to apply to images
            use_cache: Whether to cache extracted features in memory
        """
        self.data_path = Path(data_path)
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.use_cache = use_cache

        self.features: list[np.ndarray] = []
        self.labels: list[int] = []
        self.label_names: list[str] = []
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}

        self._feature_cache: dict[int, torch.Tensor] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load data from directory or preprocessed file."""
        if self.data_path.suffix == ".npz":
            self._load_from_npz()
        else:
            self._load_from_directory()

        LOGGER.info(
            "Loaded %d samples with %d classes",
            len(self.features),
            len(self.label_names),
        )

    def _load_from_npz(self) -> None:
        """Load preprocessed features and labels from npz file."""
        data = np.load(self.data_path, allow_pickle=True)

        self.features = list(data["features"])
        self.labels = list(data["labels"])

        if "label_names" in data:
            self.label_names = list(data["label_names"])
        else:
            # Generate A-Z labels
            unique_labels = sorted(set(self.labels))
            self.label_names = [chr(65 + i) for i in unique_labels]

        self._build_label_mappings()

    def _load_from_directory(self) -> None:
        """Load images from directory structure."""
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")

        if not self.data_path.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_path}")

        # Get class directories
        class_dirs = sorted([d for d in self.data_path.iterdir() if d.is_dir()])

        if not class_dirs:
            raise ValueError(f"No class directories found in {self.data_path}")

        self.label_names = [d.name for d in class_dirs]
        self._build_label_mappings()

        # Load image paths and labels
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        for class_dir in class_dirs:
            class_label = class_dir.name
            label_idx = self.label_to_idx[class_label]

            image_files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]

            for image_file in image_files:
                # Store path as string (will be loaded on demand)
                self.features.append(np.array([str(image_file)], dtype=object))
                self.labels.append(label_idx)

        if not self.features:
            raise ValueError(f"No images found in {self.data_path}")

    def _build_label_mappings(self) -> None:
        """Build bidirectional label mappings."""
        self.label_to_idx = {name: idx for idx, name in enumerate(self.label_names)}
        self.idx_to_label = {idx: name for name, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (features, label)
        """
        # Check cache first
        if self.use_cache and idx in self._feature_cache:
            return self._feature_cache[idx], self.labels[idx]

        feature_data = self.features[idx]

        # If feature_data is a path, load and extract features
        if feature_data.dtype == object:
            image_path = str(feature_data[0])
            feature_vector = self._extract_features_from_image(image_path)
        else:
            # Already extracted features
            feature_vector = torch.from_numpy(feature_data).float()

        label = self.labels[idx]

        # Apply transforms if provided
        if self.transform is not None:
            feature_vector = self.transform(feature_vector)

        # Cache if enabled
        if self.use_cache:
            self._feature_cache[idx] = feature_vector

        return feature_vector, label

    def _extract_features_from_image(self, image_path: str) -> torch.Tensor:
        """
        Extract features from image using the feature extractor.

        Args:
            image_path: Path to image file

        Returns:
            Feature tensor
        """
        if self.feature_extractor is None:
            raise ValueError(
                "Feature extractor is required when loading from image files. "
                "Provide a feature_extractor function in the dataset constructor."
            )

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_array = np.array(image)

        # Extract features
        features = self.feature_extractor(image_array)

        return torch.from_numpy(features).float()

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.

        Returns:
            Tensor of shape (num_classes,) with weights for each class
        """
        label_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        num_classes = len(self.label_names)

        # Inverse frequency weighting
        weights = total_samples / (num_classes * label_counts)
        return torch.from_numpy(weights).float()


def create_train_val_split(
    dataset: ASLDataset,
    val_split: float = 0.2,
    random_seed: int = 42,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Split dataset into training and validation sets.

    Args:
        dataset: ASLDataset instance
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    from torch.utils.data import random_split

    # Set random seed
    generator = torch.Generator().manual_seed(random_seed)

    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    LOGGER.info("Split dataset: %d train, %d validation", train_size, val_size)

    return train_dataset, val_dataset
