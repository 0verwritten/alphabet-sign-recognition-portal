"""PyTorch model architecture for ASL alphabet recognition."""

import torch
import torch.nn as nn
from typing import Literal


class ASLClassifier(nn.Module):
    """
    Neural network for ASL alphabet classification.

    This model takes hand landmark features extracted from MediaPipe
    and predicts the corresponding ASL alphabet letter (A-Z).

    Architecture:
    - Input: Feature vector from hand landmarks (default: 63 + distance features)
    - Hidden layers: 3 fully connected layers with dropout and batch normalization
    - Output: 26 classes (A-Z) or 27 classes if including "nothing" class

    Args:
        input_size: Size of input feature vector (default: 63 for MediaPipe landmarks)
        hidden_sizes: Tuple of hidden layer sizes
        num_classes: Number of output classes (default: 26 for A-Z)
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        input_size: int = 84,  # 63 landmarks + 21 distance features
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        num_classes: int = 26,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Build the network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)

    def predict(self, x: torch.Tensor, return_probabilities: bool = False) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            return_probabilities: If True, return probabilities; otherwise return class indices

        Returns:
            Predictions (class indices or probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)

            if return_probabilities:
                return probabilities
            else:
                return torch.argmax(probabilities, dim=1)


class ASLClassifierV2(nn.Module):
    """
    Enhanced version of ASL classifier with residual connections.

    This model includes skip connections for better gradient flow
    and improved training stability.
    """

    def __init__(
        self,
        input_size: int = 84,
        hidden_sizes: tuple[int, ...] = (512, 256, 128),
        num_classes: int = 26,
        dropout_rate: float = 0.4,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.blocks.append(
                ResidualBlock(
                    hidden_sizes[i],
                    hidden_sizes[i + 1],
                    dropout_rate=dropout_rate,
                )
            )

        # Output layer
        self.output = nn.Linear(hidden_sizes[-1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output(x)

    def predict(self, x: torch.Tensor, return_probabilities: bool = False) -> torch.Tensor:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)

            if return_probabilities:
                return probabilities
            else:
                return torch.argmax(probabilities, dim=1)


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.3):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Skip connection projection if dimensions don't match
        self.skip = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.block(x) + self.skip(x)


def create_model(
    model_type: Literal["v1", "v2"] = "v1",
    input_size: int = 84,
    num_classes: int = 26,
    **kwargs: int | float,
) -> ASLClassifier | ASLClassifierV2:
    """
    Factory function to create ASL classifier models.

    Args:
        model_type: Type of model to create ("v1" or "v2")
        input_size: Size of input feature vector
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters

    Returns:
        Initialized model

    Example:
        >>> model = create_model("v1", input_size=84, num_classes=26)
        >>> model = create_model("v2", input_size=84, num_classes=26, dropout_rate=0.5)
    """
    if model_type == "v1":
        return ASLClassifier(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type == "v2":
        return ASLClassifierV2(input_size=input_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'v1' or 'v2'.")
