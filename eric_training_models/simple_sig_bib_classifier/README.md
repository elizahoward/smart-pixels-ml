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

## Quantization Results

Based on our experiments with 4-bit quantization:

| Model Variant | Validation Accuracy | Validation Precision | Validation Recall | ROC AUC |
|---------------|-------------------|---------------------|------------------|---------|
| Regular (32-bit) | ~85.96% | ~81.04% | ~97.53% | ~0.9019 |
| Standard Quantized (4-bit) | 85.70% | 82.00% | 95.14% | 0.8928 |
| Input Quantized (4-bit) | 84.50% | 83.25% | 90.25% | - |