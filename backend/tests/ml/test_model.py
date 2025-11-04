"""Tests for PyTorch model architectures."""

from __future__ import annotations

import pytest
import torch

from app.ml.model import ASLClassifier, ASLClassifierV2, ResidualBlock, create_model


class TestASLClassifier:
    """Tests for ASLClassifier (V1) model."""

    def test_model_creation_default_params(self) -> None:
        """Test model creation with default parameters."""
        model = ASLClassifier()

        assert model.input_size == 84
        assert model.num_classes == 26
        assert isinstance(model, torch.nn.Module)

    def test_model_creation_custom_params(self) -> None:
        """Test model creation with custom parameters."""
        model = ASLClassifier(
            input_size=100,
            hidden_sizes=(128, 64),
            num_classes=30,
            dropout_rate=0.5,
        )

        assert model.input_size == 100
        assert model.num_classes == 30

    def test_forward_pass_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        model = ASLClassifier(input_size=84, num_classes=26)
        batch_size = 4
        input_tensor = torch.randn(batch_size, 84)

        output = model(input_tensor)

        assert output.shape == (batch_size, 26)

    def test_forward_pass_single_sample(self) -> None:
        """Test forward pass with single sample."""
        model = ASLClassifier()
        input_tensor = torch.randn(1, 84)

        output = model(input_tensor)

        assert output.shape == (1, 26)

    def test_predict_returns_class_indices(self) -> None:
        """Test predict method returns class indices."""
        model = ASLClassifier()
        model.eval()
        input_tensor = torch.randn(3, 84)

        predictions = model.predict(input_tensor, return_probabilities=False)

        assert predictions.shape == (3,)
        assert predictions.dtype == torch.long
        assert torch.all((predictions >= 0) & (predictions < 26))

    def test_predict_returns_probabilities(self) -> None:
        """Test predict method returns probabilities."""
        model = ASLClassifier()
        model.eval()
        input_tensor = torch.randn(3, 84)

        probabilities = model.predict(input_tensor, return_probabilities=True)

        assert probabilities.shape == (3, 26)
        assert torch.all((probabilities >= 0) & (probabilities <= 1))
        # Check probabilities sum to 1 (approximately)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_model_eval_mode_in_predict(self) -> None:
        """Test predict method sets model to eval mode."""
        model = ASLClassifier()
        model.train()  # Set to training mode
        input_tensor = torch.randn(2, 84)

        model.predict(input_tensor)

        # Model should be in eval mode after predict
        assert not model.training

    def test_gradient_flow(self) -> None:
        """Test gradients flow through the model."""
        model = ASLClassifier()
        input_tensor = torch.randn(2, 84, requires_grad=True)
        target = torch.randint(0, 26, (2,))

        output = model(input_tensor)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        assert input_tensor.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    def test_batch_normalization_layers(self) -> None:
        """Test model contains batch normalization layers."""
        model = ASLClassifier()

        has_batchnorm = any(
            isinstance(module, torch.nn.BatchNorm1d) for module in model.modules()
        )
        assert has_batchnorm

    def test_dropout_layers(self) -> None:
        """Test model contains dropout layers."""
        model = ASLClassifier()

        has_dropout = any(
            isinstance(module, torch.nn.Dropout) for module in model.modules()
        )
        assert has_dropout


class TestASLClassifierV2:
    """Tests for ASLClassifierV2 (with residual connections)."""

    def test_model_creation(self) -> None:
        """Test V2 model creation."""
        model = ASLClassifierV2()

        assert model.input_size == 84
        assert model.num_classes == 26
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass_shape(self) -> None:
        """Test forward pass produces correct output shape."""
        model = ASLClassifierV2()
        batch_size = 4
        input_tensor = torch.randn(batch_size, 84)

        output = model(input_tensor)

        assert output.shape == (batch_size, 26)

    def test_predict_method(self) -> None:
        """Test predict method works."""
        model = ASLClassifierV2()
        model.eval()
        input_tensor = torch.randn(3, 84)

        predictions = model.predict(input_tensor)

        assert predictions.shape == (3,)
        assert torch.all((predictions >= 0) & (predictions < 26))

    def test_residual_blocks_present(self) -> None:
        """Test model contains residual blocks."""
        model = ASLClassifierV2()

        has_residual = any(
            isinstance(module, ResidualBlock) for module in model.modules()
        )
        assert has_residual

    def test_gradient_flow(self) -> None:
        """Test gradients flow through residual connections."""
        model = ASLClassifierV2()
        input_tensor = torch.randn(2, 84, requires_grad=True)
        target = torch.randint(0, 26, (2,))

        output = model(input_tensor)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        assert input_tensor.grad is not None
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


class TestResidualBlock:
    """Tests for ResidualBlock."""

    def test_residual_block_same_dimensions(self) -> None:
        """Test residual block with same input/output dimensions."""
        block = ResidualBlock(in_features=128, out_features=128)
        input_tensor = torch.randn(4, 128)

        output = block(input_tensor)

        assert output.shape == (4, 128)

    def test_residual_block_different_dimensions(self) -> None:
        """Test residual block with different input/output dimensions."""
        block = ResidualBlock(in_features=256, out_features=128)
        input_tensor = torch.randn(4, 256)

        output = block(input_tensor)

        assert output.shape == (4, 128)

    def test_skip_connection_identity(self) -> None:
        """Test skip connection is identity when dimensions match."""
        block = ResidualBlock(in_features=128, out_features=128)

        assert isinstance(block.skip, torch.nn.Identity)

    def test_skip_connection_projection(self) -> None:
        """Test skip connection is linear projection when dimensions differ."""
        block = ResidualBlock(in_features=256, out_features=128)

        assert isinstance(block.skip, torch.nn.Linear)


class TestCreateModel:
    """Tests for create_model factory function."""

    def test_create_v1_model(self) -> None:
        """Test creating V1 model."""
        model = create_model(model_type="v1", input_size=84, num_classes=26)

        assert isinstance(model, ASLClassifier)
        assert model.input_size == 84
        assert model.num_classes == 26

    def test_create_v2_model(self) -> None:
        """Test creating V2 model."""
        model = create_model(model_type="v2", input_size=84, num_classes=26)

        assert isinstance(model, ASLClassifierV2)
        assert model.input_size == 84
        assert model.num_classes == 26

    def test_create_model_with_kwargs(self) -> None:
        """Test creating model with additional kwargs."""
        model = create_model(
            model_type="v1",
            input_size=100,
            num_classes=30,
            dropout_rate=0.5,
        )

        assert isinstance(model, ASLClassifier)
        assert model.input_size == 100
        assert model.num_classes == 30

    def test_create_model_invalid_type(self) -> None:
        """Test creating model with invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(model_type="v3", input_size=84, num_classes=26)  # type: ignore[arg-type]


class TestModelIntegration:
    """Integration tests for model behavior."""

    def test_model_reproducibility_with_seed(self) -> None:
        """Test model produces same results with same seed."""
        torch.manual_seed(42)
        model1 = ASLClassifier()
        input_tensor = torch.randn(2, 84)
        output1 = model1(input_tensor)

        torch.manual_seed(42)
        model2 = ASLClassifier()
        output2 = model2(input_tensor)

        assert torch.allclose(output1, output2)

    def test_model_training_mode_affects_dropout(self) -> None:
        """Test dropout behaves differently in train vs eval mode."""
        model = ASLClassifier()
        input_tensor = torch.randn(100, 84)

        # Training mode - dropout active
        model.train()
        with torch.no_grad():
            output_train1 = model(input_tensor)
            output_train2 = model(input_tensor)

        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2)

        # Eval mode - no dropout
        model.eval()
        with torch.no_grad():
            output_eval1 = model(input_tensor)
            output_eval2 = model(input_tensor)

        # Outputs should be identical
        assert torch.allclose(output_eval1, output_eval2)

    def test_model_save_and_load(self, tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[name-defined]
        """Test model can be saved and loaded."""
        model = ASLClassifier()
        save_path = tmp_path / "test_model.pt"  # type: ignore[operator]

        # Save model
        torch.save(model.state_dict(), save_path)

        # Load into new model
        new_model = ASLClassifier()
        new_model.load_state_dict(torch.load(save_path))

        # Test both models produce same output
        input_tensor = torch.randn(2, 84)
        model.eval()
        new_model.eval()

        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = new_model(input_tensor)

        assert torch.allclose(output1, output2)

    def test_model_handles_different_batch_sizes(self) -> None:
        """Test model works with various batch sizes."""
        model = ASLClassifier()
        model.eval()

        batch_sizes = [1, 2, 8, 32, 64]

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 84)
            with torch.no_grad():
                output = model(input_tensor)

            assert output.shape == (batch_size, 26)

    def test_model_parameter_count(self) -> None:
        """Test model has reasonable number of parameters."""
        model = ASLClassifier()

        total_params = sum(p.numel() for p in model.parameters())

        # Should have parameters (not zero) but not too many
        assert 10_000 < total_params < 1_000_000

    def test_v2_more_parameters_than_v1(self) -> None:
        """Test V2 model has more parameters than V1."""
        model_v1 = ASLClassifier(hidden_sizes=(256, 128, 64))
        model_v2 = ASLClassifierV2(hidden_sizes=(512, 256, 128))

        params_v1 = sum(p.numel() for p in model_v1.parameters())
        params_v2 = sum(p.numel() for p in model_v2.parameters())

        assert params_v2 > params_v1
