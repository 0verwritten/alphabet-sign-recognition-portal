# Quick Start Guide: ASL Recognition Model

Get your ASL recognition system running in 15 minutes!

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager
- (Optional) CUDA-capable GPU for faster training

## Step 1: Install Dependencies (2 minutes)

```bash
cd backend
pip install -e .
pip install -e ".[dev]"
```

## Step 2: Download Dataset (5 minutes)

Download the ASL Alphabet dataset:

1. Go to [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
2. Download and extract to `data/asl_alphabet/`

Your directory structure should look like:
```
backend/
  data/
    asl_alphabet/
      A/
        *.jpg
      B/
        *.jpg
      ...
```

## Step 3: Prepare Dataset (5-10 minutes)

Extract MediaPipe features from images:

```bash
python scripts/prepare_dataset.py \
    --input-dir data/asl_alphabet/ \
    --output-file data/asl_features.npz
```

This will:
- Process all images in the dataset
- Extract hand landmarks using MediaPipe
- Save preprocessed features to `asl_features.npz`

**Note:** Images without detected hands will be skipped.

## Step 4: Train Model (10-30 minutes)

Train a PyTorch model:

```bash
python scripts/train_model.py \
    --data-file data/asl_features.npz \
    --epochs 100 \
    --batch-size 32 \
    --output-dir models/
```

Training time:
- **CPU**: ~30 minutes
- **GPU**: ~10 minutes

The best model will be saved as `models/best_model.pt`.

## Step 5: Evaluate Model (1 minute)

Check model performance:

```bash
python scripts/evaluate_model.py \
    --model-path models/best_model.pt \
    --data-file data/asl_features.npz \
    --output-dir evaluation_results/
```

Review the results:
- `evaluation_results/results.txt` - Detailed metrics
- `evaluation_results/confusion_matrix.png` - Visual confusion matrix
- `evaluation_results/per_class_f1_scores.png` - Per-class performance

## Step 6: Deploy Model (1 minute)

Update the `.env` file in the project root:

```bash
ASL_CLASSIFIER_PATH=backend/models/best_model.pt
```

Start the API:

```bash
cd backend
uvicorn app.main:app --reload
```

## Step 7: Test It! (1 minute)

Test with an image:

```bash
curl -X POST "http://localhost:8000/api/v1/asl/recognitions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

Response:
```json
{
  "letter": "A",
  "confidence": 0.95,
  "handedness": "right"
}
```

## Expected Performance

With the Kaggle ASL Alphabet dataset, you should achieve:
- **Training Accuracy**: 95-99%
- **Validation Accuracy**: 85-95%
- **Inference Time**: ~50-100ms per image

## Next Steps

1. **Improve Performance**: See [PYTORCH_MODEL_TRAINING.md](PYTORCH_MODEL_TRAINING.md) for advanced training techniques
2. **Integrate with Frontend**: Connect the API to your web interface
3. **Collect More Data**: Add your own images for better real-world performance

## Troubleshooting

### "No hands detected in image"
- Ensure images have clear, visible hands
- Check lighting and image quality
- Lower `--min-detection-confidence` to 0.3

### Training is slow
- Use a smaller batch size: `--batch-size 16`
- Reduce epochs: `--epochs 50`
- Use GPU if available

### Low accuracy (<80%)
- Train for more epochs: `--epochs 150`
- Try model v2: `--model-type v2`
- Use class weights: `--use-class-weights`

## Resources

- Full training guide: [PYTORCH_MODEL_TRAINING.md](PYTORCH_MODEL_TRAINING.md)
- API documentation: [SIGN_RECOGNITION.md](../SIGN_RECOGNITION.md)
- MediaPipe Hands: https://google.github.io/mediapipe/solutions/hands.html
