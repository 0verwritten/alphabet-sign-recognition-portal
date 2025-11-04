# PyTorch Model Training Guide for ASL Recognition

This guide explains how to train a PyTorch-based model for ASL (American Sign Language) alphabet recognition using the provided tools and scripts.

## Overview

The ASL recognition system uses a hybrid approach:
1. **MediaPipe Hands** for hand landmark detection (21 3D landmarks)
2. **PyTorch Neural Network** for classifying hand poses into letters A-Z

## Architecture

### Model Architectures

Two model architectures are available:

#### V1: Standard Feed-Forward Network
- Input: 84 features (63 landmark coordinates + 21 connection distances)
- Hidden layers: Configurable (default: 256 → 128 → 64)
- Batch normalization and dropout for regularization
- Output: 26 classes (A-Z)

#### V2: Enhanced with Residual Connections
- Similar to V1 but includes skip connections
- Better gradient flow and training stability
- Recommended for larger datasets

### Feature Extraction

The system extracts the following features from each image:
- **63 landmark features**: 21 hand landmarks × (x, y, z) coordinates
- **21 distance features**: Euclidean distances between connected joints
- **Normalization**: Coordinates centered at wrist and scaled by max distance
- **Total**: 84 features per image

## Prerequisites

### Install Dependencies

```bash
cd backend
pip install -e .
pip install -e ".[dev]"
```

Required packages:
- `torch >= 2.0.0` - PyTorch framework
- `mediapipe >= 0.10.0` - Hand landmark detection
- `numpy >= 2.2.6` - Numerical operations
- `pillow >= 12.0.0` - Image processing
- `opencv-python >= 4.8.0` - Computer vision utilities
- `scikit-learn >= 1.3.0` - Metrics and evaluation
- `matplotlib >= 3.7.0` - Visualization
- `tqdm >= 4.65.0` - Progress bars

### Dataset Preparation

You need a dataset of ASL hand sign images organized by letter:

```
data/
  A/
    image1.jpg
    image2.jpg
    ...
  B/
    image1.jpg
    image2.jpg
    ...
  ...
  Z/
    image1.jpg
    image2.jpg
```

**Recommended Dataset**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Step 1: Data Preparation

Extract MediaPipe features from your images:

```bash
python scripts/prepare_dataset.py \
    --input-dir data/asl_alphabet/ \
    --output-file data/asl_features.npz \
    --min-detection-confidence 0.5
```

**Parameters:**
- `--input-dir`: Directory with class subdirectories containing images
- `--output-file`: Output file to save preprocessed features (.npz format)
- `--min-detection-confidence`: Minimum confidence for hand detection (default: 0.5)

**Output:**
- `asl_features.npz` containing:
  - `features`: Array of shape (N, 84) with extracted features
  - `labels`: Array of shape (N,) with class indices
  - `label_names`: Array of class names (e.g., ['A', 'B', ..., 'Z'])

**Notes:**
- Images without detected hands will be skipped
- The script shows progress and statistics for each class
- Processing time depends on dataset size (~1-2 images/second)

## Step 2: Model Training

Train the PyTorch model on extracted features:

### Basic Training

```bash
python scripts/train_model.py \
    --data-file data/asl_features.npz \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output-dir models/
```

### Advanced Training with Custom Architecture

```bash
python scripts/train_model.py \
    --data-file data/asl_features.npz \
    --model-type v2 \
    --hidden-sizes 512 256 128 \
    --dropout 0.4 \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --weight-decay 1e-5 \
    --use-class-weights \
    --device cuda \
    --output-dir models/asl_v2/
```

**Parameters:**

**Data:**
- `--data-file`: Path to preprocessed .npz file (required)
- `--val-split`: Validation split ratio (default: 0.2)

**Model:**
- `--model-type`: Architecture version - "v1" or "v2" (default: v1)
- `--hidden-sizes`: Hidden layer sizes (default: 256 128 64)
- `--dropout`: Dropout rate (default: 0.3)

**Training:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--weight-decay`: Weight decay for regularization (default: 1e-5)
- `--use-class-weights`: Use class weights for imbalanced datasets

**Other:**
- `--device`: Device to use - "cuda" or "cpu" (default: cuda if available)
- `--output-dir`: Output directory for models (default: models/)
- `--seed`: Random seed (default: 42)

**Output Files:**
- `best_model.pt`: Best model based on validation accuracy
- `final_model.pt`: Model after final epoch
- `checkpoint_epoch_N.pt`: Checkpoints every 10 epochs

**Checkpoint Format:**
```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": OrderedDict,
    "val_accuracy": float,
    "model_config": {
        "model_type": "v1" or "v2",
        "input_size": 84,
        "num_classes": 26
    },
    "label_mapping": {0: "A", 1: "B", ..., 25: "Z"}
}
```

### Training Tips

1. **Start with default parameters** for baseline performance
2. **Monitor validation accuracy** - training stops when no improvement
3. **Use class weights** if your dataset is imbalanced
4. **Increase dropout** (0.4-0.5) if overfitting occurs
5. **Try model v2** for larger datasets (>10K samples per class)
6. **Use GPU** if available for faster training

## Step 3: Model Evaluation

Evaluate your trained model:

```bash
python scripts/evaluate_model.py \
    --model-path models/best_model.pt \
    --data-file data/asl_features.npz \
    --batch-size 32 \
    --output-dir evaluation_results/
```

**Parameters:**
- `--model-path`: Path to trained model checkpoint (.pt file)
- `--data-file`: Path to evaluation data (.npz file)
- `--batch-size`: Batch size (default: 32)
- `--device`: Device to use (default: cuda if available)
- `--output-dir`: Output directory for results (default: evaluation_results/)

**Output:**

1. **Console Output**: Per-class and overall metrics
2. **results.txt**: Detailed text report
3. **confusion_matrix.png**: Normalized confusion matrix visualization
4. **per_class_f1_scores.png**: F1 scores for each class

**Metrics:**
- Overall accuracy
- Per-class precision, recall, F1-score
- Macro and weighted averages
- Confusion matrix

## Step 4: Deploy Model

### Update Configuration

Edit `.env` file in the project root:

```bash
# Path to your trained model
ASL_CLASSIFIER_PATH=models/best_model.pt

# MediaPipe configuration
ASL_MIN_DETECTION_CONFIDENCE=0.5
ASL_MIN_TRACKING_CONFIDENCE=0.5
ASL_MAX_NUM_HANDS=1
```

### Start the API

```bash
cd backend
uvicorn app.main:app --reload
```

The model will be automatically loaded at startup.

### Test the API

```bash
curl -X POST "http://localhost:8000/api/v1/asl/recognitions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/sign_image.jpg"
```

**Response:**
```json
{
  "letter": "A",
  "confidence": 0.95,
  "handedness": "right"
}
```

## Model Performance Tips

### Improving Accuracy

1. **More Data**: Collect diverse images with different:
   - Lighting conditions
   - Hand sizes and skin tones
   - Backgrounds
   - Camera angles

2. **Data Augmentation**: Add variations during data collection:
   - Horizontal flips (consider letter symmetry!)
   - Slight rotations
   - Brightness/contrast adjustments

3. **Hyperparameter Tuning**:
   - Try different learning rates: [0.0001, 0.001, 0.01]
   - Adjust dropout: [0.2, 0.3, 0.4, 0.5]
   - Experiment with architecture sizes

4. **Ensemble Methods**: Train multiple models and combine predictions

### Common Issues

**Low Accuracy (<80%)**
- Check if MediaPipe detects hands correctly in your images
- Ensure dataset has balanced classes
- Increase model capacity (more/larger hidden layers)
- Train for more epochs

**Overfitting (High train, low validation accuracy)**
- Increase dropout rate
- Add more data
- Reduce model capacity
- Use class weights

**Underfitting (Low train and validation accuracy)**
- Increase model capacity
- Reduce dropout
- Train for more epochs
- Check if features are extracted correctly

## Advanced Usage

### Custom Model Architecture

Modify `app/ml/model.py` to create custom architectures:

```python
from app.ml.model import ASLClassifier

class CustomASLModel(ASLClassifier):
    def __init__(self, input_size=84, num_classes=26):
        super().__init__(
            input_size=input_size,
            hidden_sizes=(512, 256, 128, 64),
            num_classes=num_classes,
            dropout_rate=0.4,
        )
```

### Transfer Learning

Start from a pre-trained model:

```python
# In train_model.py, before training:
if args.pretrained_path:
    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint["model_state_dict"])
```

### Continuous Training

Resume training from checkpoint:

```python
# In train_model.py:
checkpoint = torch.load("models/checkpoint_epoch_50.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

## Troubleshooting

### MediaPipe Not Detecting Hands
- Ensure good lighting and clear hand visibility
- Adjust `--min-detection-confidence` (try 0.3-0.7)
- Check image quality and resolution

### CUDA Out of Memory
- Reduce `--batch-size` (try 16 or 8)
- Use smaller model architecture
- Use `--device cpu` if GPU memory is limited

### Model Not Loading in API
- Check `ASL_CLASSIFIER_PATH` in `.env`
- Verify file path is absolute or relative to backend/
- Check file permissions
- Review backend logs for error messages

## References

- [MediaPipe Hands Documentation](https://google.github.io/mediapipe/solutions/hands.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Support

For issues or questions:
1. Check the logs in the backend console
2. Review evaluation metrics to understand model performance
3. Ensure all dependencies are correctly installed
4. Verify dataset format and preprocessing
