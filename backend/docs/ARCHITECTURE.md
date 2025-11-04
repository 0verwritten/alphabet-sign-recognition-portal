# ASL Recognition System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ASL Recognition System                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼────────┐       ┌───────▼────────┐
            │  Training       │       │  Inference      │
            │  Pipeline       │       │  Pipeline       │
            └────────────────┘       └─────────────────┘
```

## Training Pipeline

```
┌──────────────┐
│ Raw Images   │
│ (A-Z folders)│
└──────┬───────┘
       │
       │ scripts/prepare_dataset.py
       │
┌──────▼────────────────────────────┐
│ MediaPipe Hand Detection          │
│ - Detect 21 hand landmarks        │
│ - Extract x, y, z coordinates     │
│ - Calculate connection distances  │
└──────┬────────────────────────────┘
       │
       │ Feature Extraction
       │
┌──────▼────────────────────────────┐
│ Preprocessed Features             │
│ - 63 landmark features            │
│ - 21 distance features            │
│ - Total: 84 features/sample       │
│ - Saved as .npz file              │
└──────┬────────────────────────────┘
       │
       │ scripts/train_model.py
       │
┌──────▼────────────────────────────┐
│ PyTorch Training                  │
│ - Train/Val split                 │
│ - Batch training                  │
│ - Loss calculation                │
│ - Backpropagation                 │
│ - Validation                      │
└──────┬────────────────────────────┘
       │
       │ Checkpoint Saving
       │
┌──────▼────────────────────────────┐
│ Trained Model (.pt)               │
│ - Model weights                   │
│ - Configuration                   │
│ - Label mapping                   │
└───────────────────────────────────┘
```

## Inference Pipeline

```
┌──────────────┐
│ Image Upload │
│ (API Request)│
└──────┬───────┘
       │
       │ POST /api/v1/asl/recognitions
       │
┌──────▼────────────────────────────┐
│ ASLRecognitionService             │
└──────┬────────────────────────────┘
       │
┌──────▼────────────────────────────┐
│ Image Loading & Validation        │
│ - Parse image bytes               │
│ - Convert to RGB                  │
│ - Apply EXIF orientation          │
└──────┬────────────────────────────┘
       │
┌──────▼────────────────────────────┐
│ MediaPipe Hand Detection          │
│ - Initialize detector             │
│ - Process image                   │
│ - Extract landmarks               │
│ - Detect handedness               │
└──────┬────────────────────────────┘
       │
       │ If no hand detected
       ├──────────> HandNotDetectedError (422)
       │
┌──────▼────────────────────────────┐
│ Feature Extraction                │
│ - Normalize coordinates           │
│ - Calculate distances             │
│ - Create feature vector (84)     │
└──────┬────────────────────────────┘
       │
┌──────▼────────────────────────────┐
│ Model Classification              │
│ - Load PyTorch/sklearn model     │
│ - Convert to tensor (PyTorch)    │
│ - Forward pass                    │
│ - Apply softmax                   │
└──────┬────────────────────────────┘
       │
┌──────▼────────────────────────────┐
│ Result                            │
│ - Predicted letter                │
│ - Confidence score                │
│ - Handedness (left/right)         │
└──────┬────────────────────────────┘
       │
       │ JSON Response
       │
┌──────▼────────────────────────────┐
│ {                                 │
│   "letter": "A",                  │
│   "confidence": 0.95,             │
│   "handedness": "right"           │
│ }                                 │
└───────────────────────────────────┘
```

## Model Architecture

### V1: Standard Feed-Forward Network

```
Input Layer (84 features)
    │
    ├─> Linear(84 → 256)
    ├─> BatchNorm1d(256)
    ├─> ReLU
    ├─> Dropout(0.3)
    │
    ├─> Linear(256 → 128)
    ├─> BatchNorm1d(128)
    ├─> ReLU
    ├─> Dropout(0.3)
    │
    ├─> Linear(128 → 64)
    ├─> BatchNorm1d(64)
    ├─> ReLU
    ├─> Dropout(0.3)
    │
    └─> Linear(64 → 26)
         │
         └─> Softmax → Output (26 classes)
```

### V2: Enhanced with Residual Connections

```
Input Layer (84 features)
    │
    ├─> Linear(84 → 512)
    ├─> BatchNorm1d(512)
    ├─> ReLU
    ├─> Dropout(0.4)
    │
    ├─> ResidualBlock(512 → 256)
    │   ├─> Linear(512 → 256)
    │   ├─> BatchNorm1d(256)
    │   ├─> ReLU
    │   ├─> Dropout(0.4)
    │   └─> + Skip Connection(512 → 256)
    │
    ├─> ResidualBlock(256 → 128)
    │   ├─> Linear(256 → 128)
    │   ├─> BatchNorm1d(128)
    │   ├─> ReLU
    │   ├─> Dropout(0.4)
    │   └─> + Skip Connection(256 → 128)
    │
    └─> Linear(128 → 26)
         │
         └─> Softmax → Output (26 classes)
```

## Feature Extraction Details

### Hand Landmark Structure

```
MediaPipe Hands provides 21 landmarks:

       8   12  16  20        (fingertips)
       │   │   │   │
       7   11  15  19        (finger joints)
       │   │   │   │
       6   10  14  18
       │   │   │   │
    4  5   9   13  17
    │  └───┴────┴───┘
    3
    │
    2
    │
    1
    │
    0 (wrist)

Each landmark has (x, y, z) coordinates
```

### Feature Vector Construction

```
1. Extract 21 landmarks × 3 coordinates = 63 features
   - x: horizontal position (0-1, normalized)
   - y: vertical position (0-1, normalized)
   - z: depth (relative to wrist)

2. Normalize coordinates:
   - Center at wrist (landmark 0)
   - Scale by maximum distance from wrist

3. Calculate connection distances:
   - Distance between each connected joint
   - 21 distances for hand connections

4. Final feature vector:
   [normalized_landmarks (63), distances (21)] = 84 features
```

## Component Diagram

```
┌────────────────────────────────────────────────────────────┐
│                      Backend Application                    │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐         ┌──────────────┐                │
│  │   FastAPI   │────────>│  API Routes  │                │
│  │   Server    │         │  /asl/...    │                │
│  └─────────────┘         └──────┬───────┘                │
│                                  │                         │
│                          ┌───────▼────────┐               │
│                          │  Dependencies  │               │
│                          │  get_service() │               │
│                          └───────┬────────┘               │
│                                  │                         │
│  ┌───────────────────────────────▼───────────────┐       │
│  │      ASLRecognitionService                    │       │
│  ├───────────────────────────────────────────────┤       │
│  │ - predict(image_bytes)                        │       │
│  │ - _load_image()                               │       │
│  │ - _detect_hand_landmarks()                    │       │
│  │ - _extract_feature_vector()                   │       │
│  │ - _classify()                                 │       │
│  │   ├─> _classify_pytorch() [if .pt/.pth]      │       │
│  │   └─> _classify_sklearn() [if .joblib/.pkl]  │       │
│  └───────┬──────────────────────┬────────────────┘       │
│          │                      │                         │
│  ┌───────▼────────┐    ┌───────▼──────────┐             │
│  │   MediaPipe    │    │  Model (PyTorch  │             │
│  │     Hands      │    │  or scikit-learn)│             │
│  └────────────────┘    └──────────────────┘             │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Data Flow

```
Raw Images
    │
    ├─> Image Loading
    │
    ├─> MediaPipe Processing
    │   ├─> Hand Detection
    │   ├─> Landmark Extraction
    │   └─> Feature Calculation
    │
    ├─> Feature Vector (84 dims)
    │
    ├─> Save to .npz
    │
Training Script
    │
    ├─> Load .npz
    │
    ├─> Train/Val Split
    │
    ├─> Create DataLoader
    │
    ├─> Training Loop
    │   ├─> Forward Pass
    │   ├─> Loss Calculation
    │   ├─> Backpropagation
    │   └─> Weight Update
    │
    ├─> Validation
    │
    └─> Save Checkpoint (.pt)
```

### Inference Data Flow

```
Client Image
    │
    └─> HTTP POST /api/v1/asl/recognitions
         │
         └─> ASL Route Handler
              │
              └─> ASLRecognitionService.predict()
                   │
                   ├─> Load & Validate Image
                   │
                   ├─> MediaPipe Hand Detection
                   │   (Lazy initialization on first use)
                   │
                   ├─> Extract Features (84)
                   │
                   ├─> Model Inference
                   │   ├─> PyTorch: tensor → GPU → forward → softmax
                   │   └─> sklearn: array → predict → proba
                   │
                   ├─> Map to Label
                   │
                   └─> Return Result
                        │
                        └─> {letter, confidence, handedness}
                             │
                             └─> JSON Response to Client
```

## File Organization

```
backend/
├── app/
│   ├── ml/                      # Machine Learning Module
│   │   ├── model.py            # PyTorch models
│   │   ├── dataset.py          # Dataset utilities
│   │   └── __init__.py
│   │
│   ├── services/               # Business Logic
│   │   └── asl_recognition.py # Main service
│   │
│   ├── api/                    # API Layer
│   │   ├── routes/
│   │   │   └── asl.py         # ASL endpoints
│   │   └── deps.py            # Dependencies
│   │
│   ├── core/                   # Core Configuration
│   │   └── config.py          # Settings
│   │
│   └── models.py              # Pydantic models
│
├── scripts/                    # Training Tools
│   ├── prepare_dataset.py     # Data preprocessing
│   ├── train_model.py         # Model training
│   ├── evaluate_model.py      # Model evaluation
│   └── test_inference.py      # Testing
│
├── docs/                       # Documentation
│   ├── QUICKSTART.md
│   ├── PYTORCH_MODEL_TRAINING.md
│   ├── PYTORCH_MIGRATION.md
│   └── ARCHITECTURE.md        # This file
│
├── tests/                      # Test Suite
│   ├── services/
│   └── api/
│
├── data/                       # Data Storage
│   ├── asl_alphabet/          # Raw images
│   └── asl_features.npz       # Preprocessed
│
└── models/                     # Trained Models
    ├── best_model.pt
    └── checkpoints/
```

## Technology Stack

### Core Technologies
- **FastAPI**: Web framework
- **PyTorch**: Deep learning
- **MediaPipe**: Hand detection
- **NumPy**: Numerical computing
- **Pydantic**: Data validation

### ML/CV Stack
- **MediaPipe Hands**: Hand landmark detection
- **PyTorch**: Neural network training and inference
- **scikit-learn**: Metrics and evaluation
- **OpenCV**: Image processing utilities

### Development Tools
- **pytest**: Testing framework
- **uv**: Package management
- **mypy**: Type checking
- **ruff**: Linting

## Design Principles

### 1. Separation of Concerns
- API layer handles HTTP
- Service layer handles business logic
- ML layer handles model architecture
- Clear boundaries between components

### 2. Flexibility
- Support multiple model types
- Configurable via environment variables
- Extensible architecture

### 3. Backward Compatibility
- Existing scikit-learn models work
- No breaking API changes
- Gradual migration path

### 4. Production Ready
- Error handling at every level
- Resource cleanup (MediaPipe)
- Lazy initialization
- Thread-safe operations

### 5. Developer Experience
- Clear documentation
- Type hints everywhere
- Comprehensive examples
- Easy to test and debug

## Performance Considerations

### Training
- **GPU Acceleration**: 5-10x faster training
- **Batch Processing**: Efficient memory usage
- **Checkpointing**: Resume interrupted training
- **Early Stopping**: Prevent overtraining

### Inference
- **Lazy Loading**: Models loaded on demand
- **Caching**: Single model instance
- **Device Management**: Auto GPU/CPU selection
- **Efficient Features**: Pre-normalized vectors

### Scalability
- **Stateless Service**: Easy horizontal scaling
- **Thread-Safe**: Multiple concurrent requests
- **Memory Efficient**: Minimal overhead
- **Fast Response**: ~50-100ms per prediction

## Security Considerations

### Input Validation
- Image format validation
- Size limits (handled by FastAPI)
- Content type checking
- EXIF data handling

### Model Security
- Model path validation
- File existence checks
- Safe model loading
- Error message sanitization

### API Security
- CORS configuration
- Rate limiting (recommended)
- Input sanitization
- Error handling

## Monitoring & Observability

### Logging
- Service initialization
- Model loading
- Prediction errors
- Performance metrics

### Metrics to Track
- Request latency
- Prediction confidence
- Error rates
- Model accuracy

### Health Checks
- MediaPipe availability
- Model loaded status
- GPU availability
- Service readiness
