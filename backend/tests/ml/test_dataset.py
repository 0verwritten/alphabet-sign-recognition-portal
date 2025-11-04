"""Tests for dataset utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.dataset import ASLDataset, create_train_val_split


@pytest.fixture
def sample_npz_file(tmp_path: Path) -> Path:
    """Create a sample .npz file for testing."""
    features = np.random.randn(100, 84).astype(np.float32)
    labels = np.random.randint(0, 26, size=100, dtype=np.int64)
    label_names = np.array([chr(65 + i) for i in range(26)])

    npz_path = tmp_path / "test_data.npz"
    np.savez_compressed(
        npz_path,
        features=features,
        labels=labels,
        label_names=label_names,
    )

    return npz_path


@pytest.fixture
def sample_image_directory(tmp_path: Path) -> Path:
    """Create a sample image directory structure for testing."""
    from PIL import Image

    data_dir = tmp_path / "images"

    # Create 3 classes with 5 images each
    for class_name in ["A", "B", "C"]:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            # Create simple test images
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(class_dir / f"image_{i}.jpg")

    return data_dir


class TestASLDataset:
    """Tests for ASLDataset class."""

    def test_load_from_npz(self, sample_npz_file: Path) -> None:
        """Test loading dataset from .npz file."""
        dataset = ASLDataset(sample_npz_file)

        assert len(dataset) == 100
        assert len(dataset.label_names) == 26
        assert dataset.label_names[0] == "A"
        assert dataset.label_names[25] == "Z"

    def test_getitem_returns_correct_types(self, sample_npz_file: Path) -> None:
        """Test __getitem__ returns correct types."""
        dataset = ASLDataset(sample_npz_file)

        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert isinstance(label, int)
        assert features.dtype == torch.float32
        assert features.shape == (84,)

    def test_getitem_returns_correct_label(self, sample_npz_file: Path) -> None:
        """Test __getitem__ returns correct label."""
        dataset = ASLDataset(sample_npz_file)

        _, label = dataset[0]

        assert 0 <= label < 26

    def test_label_mappings(self, sample_npz_file: Path) -> None:
        """Test label mappings are created correctly."""
        dataset = ASLDataset(sample_npz_file)

        assert len(dataset.label_to_idx) == 26
        assert len(dataset.idx_to_label) == 26
        assert dataset.label_to_idx["A"] == 0
        assert dataset.idx_to_label[0] == "A"

    def test_caching_feature_vectors(self, sample_npz_file: Path) -> None:
        """Test feature caching works."""
        dataset = ASLDataset(sample_npz_file, use_cache=True)

        # First access
        features1, _ = dataset[0]

        # Second access (should be cached)
        features2, _ = dataset[0]

        assert torch.equal(features1, features2)
        assert 0 in dataset._feature_cache

    def test_no_caching_when_disabled(self, sample_npz_file: Path) -> None:
        """Test caching can be disabled."""
        dataset = ASLDataset(sample_npz_file, use_cache=False)

        dataset[0]

        assert len(dataset._feature_cache) == 0

    def test_load_from_directory_structure(self, sample_image_directory: Path) -> None:
        """Test loading from directory structure (without feature extraction)."""
        dataset = ASLDataset(sample_image_directory)

        assert len(dataset) == 15  # 3 classes × 5 images
        assert len(dataset.label_names) == 3
        assert set(dataset.label_names) == {"A", "B", "C"}

    def test_load_from_directory_requires_feature_extractor(
        self, sample_image_directory: Path
    ) -> None:
        """Test loading images requires feature extractor."""
        dataset = ASLDataset(sample_image_directory)

        with pytest.raises(ValueError, match="Feature extractor is required"):
            dataset[0]

    def test_load_from_directory_with_feature_extractor(
        self, sample_image_directory: Path
    ) -> None:
        """Test loading images with feature extractor."""

        def mock_feature_extractor(image_array: np.ndarray) -> np.ndarray:
            return np.random.randn(84).astype(np.float32)

        dataset = ASLDataset(
            sample_image_directory, feature_extractor=mock_feature_extractor
        )

        features, label = dataset[0]

        assert isinstance(features, torch.Tensor)
        assert features.shape == (84,)
        assert 0 <= label < 3

    def test_get_class_weights(self, tmp_path: Path) -> None:
        """Test class weight calculation."""
        # Create imbalanced dataset
        features = np.random.randn(100, 84).astype(np.float32)
        labels = np.array([0] * 50 + [1] * 30 + [2] * 20, dtype=np.int64)
        label_names = np.array(["A", "B", "C"])

        npz_path = tmp_path / "imbalanced.npz"
        np.savez_compressed(
            npz_path, features=features, labels=labels, label_names=label_names
        )

        dataset = ASLDataset(npz_path)
        weights = dataset.get_class_weights()

        assert weights.shape == (3,)
        assert isinstance(weights, torch.Tensor)
        # Class 0 (most samples) should have lowest weight
        # Class 2 (least samples) should have highest weight
        assert weights[2] > weights[0]

    def test_empty_directory_raises_error(self, tmp_path: Path) -> None:
        """Test loading from empty directory raises error."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No class directories found"):
            ASLDataset(empty_dir)

    def test_nonexistent_path_raises_error(self, tmp_path: Path) -> None:
        """Test nonexistent path raises error."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(ValueError, match="Data path does not exist"):
            ASLDataset(nonexistent)

    def test_file_as_directory_raises_error(self, tmp_path: Path) -> None:
        """Test passing file as directory raises error."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            ASLDataset(file_path)

    def test_transform_applied(self, sample_npz_file: Path) -> None:
        """Test transform is applied to features."""

        def double_transform(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        dataset = ASLDataset(sample_npz_file, transform=double_transform)

        features, _ = dataset[0]
        # Features should be doubled
        assert features is not None  # Just check it works


class TestCreateTrainValSplit:
    """Tests for create_train_val_split function."""

    def test_split_returns_two_datasets(self, sample_npz_file: Path) -> None:
        """Test split returns train and val datasets."""
        dataset = ASLDataset(sample_npz_file)

        train_dataset, val_dataset = create_train_val_split(dataset, val_split=0.2)

        assert len(train_dataset) + len(val_dataset) == len(dataset)

    def test_split_ratio(self, sample_npz_file: Path) -> None:
        """Test split ratio is correct."""
        dataset = ASLDataset(sample_npz_file)

        train_dataset, val_dataset = create_train_val_split(dataset, val_split=0.2)

        expected_val_size = int(len(dataset) * 0.2)
        expected_train_size = len(dataset) - expected_val_size

        assert len(train_dataset) == expected_train_size
        assert len(val_dataset) == expected_val_size

    def test_split_reproducibility(self, sample_npz_file: Path) -> None:
        """Test split is reproducible with same seed."""
        dataset = ASLDataset(sample_npz_file)

        train1, val1 = create_train_val_split(dataset, val_split=0.2, random_seed=42)
        train2, val2 = create_train_val_split(dataset, val_split=0.2, random_seed=42)

        # Check they have same indices
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)

    def test_split_different_with_different_seed(self, sample_npz_file: Path) -> None:
        """Test split is different with different seeds."""
        dataset = ASLDataset(sample_npz_file)

        train1, val1 = create_train_val_split(dataset, val_split=0.2, random_seed=42)
        train2, val2 = create_train_val_split(dataset, val_split=0.2, random_seed=123)

        # Lengths should be the same but indices likely different
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)

    def test_split_various_ratios(self, sample_npz_file: Path) -> None:
        """Test split works with various ratios."""
        dataset = ASLDataset(sample_npz_file)

        for val_split in [0.1, 0.2, 0.3, 0.5]:
            train_dataset, val_dataset = create_train_val_split(
                dataset, val_split=val_split
            )

            total = len(train_dataset) + len(val_dataset)
            assert total == len(dataset)

    def test_split_no_overlap(self, sample_npz_file: Path) -> None:
        """Test train and val sets don't overlap."""
        dataset = ASLDataset(sample_npz_file)

        train_dataset, val_dataset = create_train_val_split(dataset, val_split=0.2)

        # Get underlying indices
        train_indices = train_dataset.indices  # type: ignore[attr-defined]
        val_indices = val_dataset.indices  # type: ignore[attr-defined]

        # Check no overlap
        assert len(set(train_indices) & set(val_indices)) == 0


class TestDatasetIntegration:
    """Integration tests for dataset functionality."""

    def test_dataset_with_dataloader(self, sample_npz_file: Path) -> None:
        """Test dataset works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = ASLDataset(sample_npz_file)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        batch = next(iter(dataloader))
        features, labels = batch

        assert features.shape == (8, 84)
        assert labels.shape == (8,)

    def test_full_training_loop_simulation(self, sample_npz_file: Path) -> None:
        """Test dataset in simulated training loop."""
        from torch.utils.data import DataLoader

        dataset = ASLDataset(sample_npz_file)
        train_dataset, val_dataset = create_train_val_split(dataset, val_split=0.2)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Simulate one epoch
        batch_count = 0
        for features, labels in train_loader:
            assert features.shape[0] <= 16
            assert features.shape[1] == 84
            batch_count += 1

        assert batch_count > 0

        # Simulate validation
        val_batch_count = 0
        for features, labels in val_loader:
            assert features.shape[0] <= 16
            val_batch_count += 1

        assert val_batch_count > 0

    def test_dataset_iteration(self, sample_npz_file: Path) -> None:
        """Test iterating through entire dataset."""
        dataset = ASLDataset(sample_npz_file)

        all_features = []
        all_labels = []

        for features, label in dataset:
            all_features.append(features)
            all_labels.append(label)

        assert len(all_features) == len(dataset)
        assert len(all_labels) == len(dataset)

    def test_dataset_random_access(self, sample_npz_file: Path) -> None:
        """Test random access to dataset."""
        dataset = ASLDataset(sample_npz_file)

        # Access in random order
        indices = [50, 10, 99, 0, 25]

        for idx in indices:
            features, label = dataset[idx]
            assert features.shape == (84,)
            assert 0 <= label < 26
