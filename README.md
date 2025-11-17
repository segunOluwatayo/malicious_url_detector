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

