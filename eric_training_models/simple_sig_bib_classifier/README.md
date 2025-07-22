# Simple SIG/BIB Classifier

This folder contains neural network models for classifying between "sig" and "bib" signals using x and z profiles, including both regular and quantized variants.

## Model Variants

### 1. Regular Model (`simple_sig_bib_model.py`)
- Standard neural network with full precision (32-bit floating point)
- Uses x_profile and y_profile inputs

### 2. Quantized Models (`simple_sig_bib_quantized_model.py`)
- **Standard Quantization**: 4-bit quantization throughout the entire model
- **Advanced Quantization**: Mixed bit-widths (8-bit for input/output, 6-bit for intermediate layers)
- **Input Quantization**: 4-bit quantization including input layers

## Model Architecture

The models take two inputs:
- **x_profile**: 21-dimensional vector representing the x-profile
- **z_global**: 1-dimensional vector representing the z-global feature

The architecture consists of:
1. Two separate dense layers (64 units) for processing x and z profiles
2. Concatenation of the processed features
3. Second dense layer (32 units) with dropout (0.3)
4. Output layer with sigmoid activation for binary classification

## Files

- `simple_sig_bib_model.py`: Regular model implementation and training script
- `simple_sig_bib_quantized_model.py`: Quantized model variants with QKeras
- `README.md`: This documentation file

## Usage

### Prerequisites

Make sure you have the required environment:
```bash
conda activate mlproj_qkeras
```

### Training the Regular Model

To train the regular (non-quantized) model:

```bash
cd smart_pixels_ml/eric_training_models/simple_sig_bib_classifier
python simple_sig_bib_model.py
```

### Training Quantized Models

To train the quantized models:

```bash
cd smart_pixels_ml/eric_training_models/simple_sig_bib_classifier
python simple_sig_bib_quantized_model.py
```

#### Quantization Options

The quantized model supports different quantization schemes:

1. **Standard Quantization (4-bit throughout)**:
   ```python
   results = train_and_evaluate_quantized(
       model_name="quantized_sig_bib_classifier",
       base_dir=base_dir,
       results_dir=results_dir,
       epochs=75,
       use_advanced_quantization=False,
       use_input_quantization=False
   )
   ```

2. **Advanced Quantization (mixed bit-widths)**:
   ```python
   results = train_and_evaluate_quantized(
       model_name="quantized_sig_bib_classifier_advanced",
       base_dir=base_dir,
       results_dir=results_dir,
       epochs=75,
       use_advanced_quantization=True
   )
   ```

3. **Input Quantization (4-bit including inputs)**:
   ```python
   results = train_and_evaluate_quantized(
       model_name="quantized_sig_bib_classifier_input_quant",
       base_dir=base_dir,
       results_dir=results_dir,
       epochs=75,
       use_input_quantization=True
   )
   ```

### Model Comparison

To compare regular vs quantized models:

```python
from simple_sig_bib_quantized_model import compare_models

regular_results, quantized_results = compare_models(
    base_dir=base_dir,
    results_dir=results_dir,
    epochs=50
)
```

## Quantization Results

Based on our experiments with 4-bit quantization:

| Model Variant | Validation Accuracy | Validation Precision | Validation Recall | ROC AUC |
|---------------|-------------------|---------------------|------------------|---------|
| Regular (32-bit) | ~85.96% | ~81.04% | ~97.53% | ~0.9019 |
| Standard Quantized (4-bit) | 85.70% | 82.00% | 95.14% | 0.8928 |
| Input Quantized (4-bit) | 84.50% | 83.25% | 90.25% | - |

### Key Findings

1. **4-bit quantization achieves competitive performance**: Only ~0.26% drop in accuracy compared to full precision
2. **Memory efficiency**: 4-bit models use 8x less memory than 32-bit models
3. **Input quantization works**: Quantizing input layers is possible and maintains good performance
4. **Mixed precision**: Advanced quantization with different bit-widths provides flexibility

## Model Features

### Regular Model
- **Input**: x_profile (21-dim) and y_profile (13-dim)
- **Architecture**: Two dense layers with dropout
- **Output**: Binary classification (sigmoid activation)
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Optimizer**: Adam with learning rate 0.001

### Quantized Models
- **Input**: x_profile (21-dim) and z_global (1-dim)
- **Architecture**: Quantized dense layers with QKeras
- **Quantization**: 4-bit or mixed bit-widths
- **Output**: Binary classification (sigmoid activation)
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Optimizer**: Adam with learning rate 0.001

## Expected Results

The models will generate:
- ROC curve plots
- Training history plots
- Saved model files (`.h5` format)
- Console output with final metrics
- Quantization information for quantized models

## Data Requirements

The models expect TFRecords containing:
- `x_profile`: 21-dimensional feature vector
- `z_global`: 1-dimensional feature vector (quantized models)
- `y_profile`: 13-dimensional feature vector (regular model)
- Binary labels (0 for bib, 1 for sig)

## Configuration

The models can be configured by modifying parameters:

### Regular Model (`simple_sig_bib_model.py`)
- `epochs`: Number of training epochs (default: 50)
- `x_profile_length`: Length of x profile (default: 21)
- `y_profile_length`: Length of y profile (default: 13)

### Quantized Models (`simple_sig_bib_quantized_model.py`)
- `epochs`: Number of training epochs (default: 75)
- `x_profile_length`: Length of x profile (default: 21)
- `z_global_length`: Length of z_global (default: 1)
- `use_advanced_quantization`: Enable mixed bit-width quantization
- `use_input_quantization`: Enable input layer quantization

## Environment Setup

Required packages:
```bash
pip install tensorflow qkeras scikit-learn numpy
```

Or use the provided conda environment:
```bash
conda activate mlproj_qkeras
```

## Performance Notes

- Quantized models achieve similar performance to regular models with significantly reduced memory usage
- 4-bit quantization provides a good balance between performance and efficiency
- Input quantization is feasible and maintains competitive accuracy
- Advanced quantization with mixed bit-widths offers flexibility 