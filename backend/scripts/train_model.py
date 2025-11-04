"""Training script for ASL recognition PyTorch model."""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.model import create_model
from app.ml.dataset import ASLDataset, create_train_val_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


class Trainer:
    """Training manager for ASL recognition model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
        device: str = "cpu",
        model_dir: Path = Path("models"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_acc = 0.0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_accuracies: list[float] = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int) -> None:
        """Train the model for multiple epochs."""
        LOGGER.info("Starting training for %d epochs on device: %s", num_epochs, self.device)
        LOGGER.info("Model: %s", self.model.__class__.__name__)
        LOGGER.info("Train samples: %d", len(self.train_loader.dataset))
        LOGGER.info("Val samples: %d", len(self.val_loader.dataset))

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Log progress
            LOGGER.info(
                "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, Val Acc: %.2f%%",
                epoch + 1,
                num_epochs,
                train_loss,
                val_loss,
                val_acc,
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pt", epoch, val_acc)
                LOGGER.info("Saved new best model with accuracy: %.2f%%", val_acc)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt", epoch, val_acc)

        # Save final model
        self.save_checkpoint("final_model.pt", num_epochs - 1, self.val_accuracies[-1])
        LOGGER.info("Training complete! Best validation accuracy: %.2f%%", self.best_val_acc)

    def save_checkpoint(self, filename: str, epoch: int, val_acc: float) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.model_dir / filename

        # Get model configuration
        model_config = {
            "model_type": "v1" if "ASLClassifier" in self.model.__class__.__name__ else "v2",
            "input_size": self.model.input_size,
            "num_classes": self.model.num_classes,
        }

        # Get label mapping from dataset
        dataset = self.train_loader.dataset
        if hasattr(dataset, "dataset"):  # Handle Subset wrapper
            dataset = dataset.dataset
        label_mapping = dataset.idx_to_label if hasattr(dataset, "idx_to_label") else None

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_acc,
            "model_config": model_config,
            "label_mapping": label_mapping,
        }

        torch.save(checkpoint, checkpoint_path)
        LOGGER.info("Saved checkpoint to %s", checkpoint_path)


def main() -> None:
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ASL recognition model")

    # Data arguments
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to preprocessed .npz data file",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="Model architecture version (default: v1)",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Hidden layer sizes (default: 256 128 64)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate (default: 0.3)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights for imbalanced data",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for models (default: models/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load dataset
    LOGGER.info("Loading dataset from %s", args.data_file)
    dataset = ASLDataset(args.data_file)

    # Split into train/val
    train_dataset, val_dataset = create_train_val_split(
        dataset,
        val_split=args.val_split,
        random_seed=args.seed,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if args.device == "cuda" else False,
    )

    # Get input size from first sample
    sample_features, _ = dataset[0]
    input_size = sample_features.shape[0]
    num_classes = len(dataset.label_names)

    LOGGER.info("Input size: %d", input_size)
    LOGGER.info("Number of classes: %d", num_classes)
    LOGGER.info("Classes: %s", dataset.label_names)

    # Create model
    LOGGER.info("Creating model: %s", args.model_type)
    model = create_model(
        model_type=args.model_type,
        input_size=input_size,
        num_classes=num_classes,
        hidden_sizes=tuple(args.hidden_sizes),
        dropout_rate=args.dropout,
    )

    # Create criterion with optional class weights
    if args.use_class_weights:
        class_weights = dataset.get_class_weights().to(args.device)
        LOGGER.info("Using class weights: %s", class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.1,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        model_dir=args.output_dir,
    )

    # Train
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()
