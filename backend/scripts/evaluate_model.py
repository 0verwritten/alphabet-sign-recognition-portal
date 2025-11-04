"""Evaluation script for ASL recognition model."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ml.dataset import ASLDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    class_names: list[str],
) -> dict[str, float]:
    """
    Evaluate model on dataset.

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to use
        class_names: List of class names

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)

    # Calculate accuracy
    accuracy = 100.0 * np.sum(all_predictions == all_labels) / len(all_labels)

    # Generate classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        output_dict=True,
    )

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    results = {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "probabilities": all_probabilities,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
    }

    return results


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix.

    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        output_path: Path to save plot
        normalize: Whether to normalize values
    """
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(conf_matrix.shape[1]),
        yticks=np.arange(conf_matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Confusion Matrix (Normalized)" if normalize else "Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(
                j,
                i,
                format(conf_matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
                fontsize=6,
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    LOGGER.info("Saved confusion matrix to %s", output_path)
    plt.close()


def plot_per_class_accuracy(
    report: dict,
    output_path: Path,
) -> None:
    """
    Plot per-class accuracy.

    Args:
        report: Classification report dictionary
        output_path: Path to save plot
    """
    # Extract per-class metrics
    classes = []
    f1_scores = []

    for key, value in report.items():
        if isinstance(value, dict) and 'f1-score' in value:
            classes.append(key)
            f1_scores.append(value['f1-score'])

    # Sort by F1 score
    sorted_indices = np.argsort(f1_scores)
    classes = [classes[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(classes))
    ax.barh(y_pos, f1_scores, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlabel('F1 Score')
    ax.set_title('Per-Class F1 Scores')
    ax.set_xlim([0, 1.0])

    # Add value labels
    for i, v in enumerate(f1_scores):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    LOGGER.info("Saved per-class accuracy plot to %s", output_path)
    plt.close()


def main() -> None:
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate ASL recognition model")

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to evaluation data (.npz file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results (default: evaluation_results/)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    LOGGER.info("Loading model from %s", args.model_path)
    checkpoint = torch.load(args.model_path, map_location=args.device)

    # Import model creation function
    from app.ml.model import create_model

    # Extract model config
    model_config = checkpoint.get("model_config", {})
    model_type = model_config.get("model_type", "v1")
    input_size = model_config.get("input_size", 84)
    num_classes = model_config.get("num_classes", 26)

    # Create model
    model = create_model(
        model_type=model_type,
        input_size=input_size,
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    LOGGER.info("Model loaded successfully")

    # Load dataset
    LOGGER.info("Loading dataset from %s", args.data_file)
    dataset = ASLDataset(args.data_file)
    class_names = dataset.label_names

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Evaluate
    LOGGER.info("Evaluating model...")
    results = evaluate_model(model, data_loader, args.device, class_names)

    # Print results
    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("EVALUATION RESULTS")
    LOGGER.info("=" * 60)
    LOGGER.info("Overall Accuracy: %.2f%%", results["accuracy"])
    LOGGER.info("\nClassification Report:")
    LOGGER.info("-" * 60)

    report = results["classification_report"]
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            LOGGER.info(
                "%s: Precision=%.3f, Recall=%.3f, F1=%.3f, Support=%d",
                class_name.ljust(10),
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                int(metrics["support"]),
            )

    # Print overall metrics
    if "macro avg" in report:
        macro = report["macro avg"]
        LOGGER.info("-" * 60)
        LOGGER.info(
            "Macro Avg: Precision=%.3f, Recall=%.3f, F1=%.3f",
            macro["precision"],
            macro["recall"],
            macro["f1-score"],
        )

    if "weighted avg" in report:
        weighted = report["weighted avg"]
        LOGGER.info(
            "Weighted Avg: Precision=%.3f, Recall=%.3f, F1=%.3f",
            weighted["precision"],
            weighted["recall"],
            weighted["f1-score"],
        )

    # Generate plots
    LOGGER.info("\nGenerating visualizations...")

    plot_confusion_matrix(
        results["confusion_matrix"],
        class_names,
        args.output_dir / "confusion_matrix.png",
        normalize=True,
    )

    plot_per_class_accuracy(
        report,
        args.output_dir / "per_class_f1_scores.png",
    )

    # Save results
    results_file = args.output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ASL Recognition Model Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.data_file}\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n\n")

        f.write("Per-Class Metrics:\n")
        f.write("-" * 60 + "\n")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                f.write(
                    f"{class_name.ljust(10)}: "
                    f"Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, "
                    f"F1={metrics['f1-score']:.3f}, "
                    f"Support={int(metrics['support'])}\n"
                )

        f.write("-" * 60 + "\n")
        if "macro avg" in report:
            macro = report["macro avg"]
            f.write(
                f"Macro Avg: Precision={macro['precision']:.3f}, "
                f"Recall={macro['recall']:.3f}, F1={macro['f1-score']:.3f}\n"
            )

        if "weighted avg" in report:
            weighted = report["weighted avg"]
            f.write(
                f"Weighted Avg: Precision={weighted['precision']:.3f}, "
                f"Recall={weighted['recall']:.3f}, F1={weighted['f1-score']:.3f}\n"
            )

    LOGGER.info("Saved results to %s", results_file)
    LOGGER.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
