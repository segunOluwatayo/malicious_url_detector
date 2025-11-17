# Malicious URL Detector

A hybrid machine learning system for detecting malicious URLs using character-level CNN and hand-crafted features.

## Overview

This project implements a sophisticated URL classification system that combines deep learning with traditional feature engineering to identify potentially malicious URLs. The system uses a multi-layered approach: allow-lists, threat intelligence feeds (via Bloom filters), and a hybrid neural network model.

## Features

- **Hybrid Architecture**: Combines character-level CNN with numeric feature extraction
- **Multi-Layer Detection**: Allow-list → Bloom filter → ML model pipeline
- **TensorFlow Lite Export**: Optimized models for mobile/embedded deployment
- **Feed Integration**: Incorporates threat intelligence from malicious URL feeds
- **Calibrated Thresholds**: Optimized decision boundaries for production use

## Architecture

### Model Components

1. **Character-level CNN**
   - Processes domain strings as character sequences
   - Embedding layer (vocab size: 97, dimension: 32)
   - Two Conv1D layers (64 and 128 filters) with BatchNormalization
   - Global max pooling for feature extraction

2. **Numeric Features**
   - Domain length
   - Special character count and ratios
   - Digit count and density
   - Shannon entropy
   - Subdomain depth
   - Hyphen count

3. **Combined Classification**
   - Concatenated CNN and numeric features
   - Dense layers (64 units) with ReLU activation
   - Sigmoid output for binary classification

### Detection Pipeline

```
URL Input
    ↓
Allow-list Check (known benign domains)
    ↓
Bloom Filter Check (known malicious from feeds)
    ↓
ML Model Prediction
    ↓
Threshold-based Verdict
```

## Project Structure

```
malicious_url_detector/
├── data/
│   ├── raw.csv              # Training dataset
│   ├── dataset.npz          # Processed train/val/test splits
│   ├── scaler.json          # Feature normalization parameters
│   ├── url_model.h5         # Trained Keras model
│   └── url_classifier.tflite # TFLite model for deployment
├── scripts/
│   ├── build_dataset.py     # Data preprocessing and feature extraction
│   ├── train_model.py       # Model training script
│   ├── test.py              # Testing and demo script
│   ├── export_tflite.py     # TFLite conversion
│   ├── build_bloom.py       # Bloom filter construction
│   ├── allowlist.py         # Benign domain allow-list
│   ├── features.py          # Feature extraction utilities
│   ├── feeds.py             # Threat feed integration
│   ├── calibrate.py         # Threshold calibration
│   └── find_threshold.py    # Optimal threshold search
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/segunOluwatayo/malicious_url_detector.git
cd malicious_url_detector

# Install dependencies
pip install numpy pandas scikit-learn tensorflow tldextract pybloom-live
```

## Usage

### Training the Model

```bash
# 1. Build the dataset from raw CSV
python scripts/build_dataset.py

# 2. Train the hybrid model
python scripts/train_model.py

# 3. Find optimal classification threshold
python scripts/find_threshold.py

# 4. Export to TensorFlow Lite (optional)
python scripts/export_tflite.py
```

### Testing URLs

```python
from scripts.test import verdict

# Test individual URLs
url = "https://suspicious-domain.xyz"
result = verdict(url)
print(result)  # e.g., "Malicious (p=0.923)"
```

Run the test script with demo URLs:
```bash
python scripts/test.py
```

### Building the Bloom Filter

```python
# Create Bloom filter from malicious domain feeds
python scripts/build_bloom.py
```

## Model Performance

The model is trained with:
- **Optimizer**: Adam (learning rate: 5e-4)
- **Loss**: Binary cross-entropy
- **Metrics**: AUC, Recall
- **Early Stopping**: Monitors validation AUC with patience=5
- **Batch Size**: 256
- **Max Epochs**: 25

## Feature Engineering

The system extracts 8 numeric features from each URL:
1. Domain length
2. Non-alphanumeric character count
3. Hyphen count
4. Total digit count
5. Digit density ratio
6. Registered domain length
7. Subdomain depth
8. Shannon entropy

Features are standardized using `StandardScaler` fitted on the training set.

## TensorFlow Lite Deployment

Two TFLite models are available:
- `url_classifier.tflite`: Float32 model (~4.8KB)
- `url_classifier_int8.tflite`: Quantized int8 model (~3.7KB)

These models are optimized for mobile and edge device deployment.

## Dataset Format

The raw dataset (`data/raw.csv`) should contain:
```csv
url,label
https://example.com,0
https://malicious-site.xyz,1
```

Where `label` is:
- `0`: Benign
- `1`: Malicious

## Dependencies

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- tldextract
- pybloom-live

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is available under the MIT License.

## Acknowledgments

- Built using TensorFlow and Keras
- URL parsing powered by tldextract
- Bloom filter implementation from pybloom-live

## Author

Segun Oluwatayo
