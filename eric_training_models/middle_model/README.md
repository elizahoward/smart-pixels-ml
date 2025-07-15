# Middle Model - Improved Quantized CNN

This folder contains the improved version of the quantized CNN model with better training stability.

## Files

- `quantized_cnn_model_simple.py` - Simplified quantized model with explicit bias handling
- `quantized_cnn_model_improved.py` - Improved quantized model with better gradient flow
- `train_improved_quantized.py` - Training script for the improved model

## Model Characteristics

- **Quantization**: Mixed precision - 8-bit Conv2D, 16-bit Dense layers
- **Improvements**: Better gradient flow, standard ReLU activations
- **Performance**: More stable training than first version
- **Complexity**: Improved approach with less aggressive quantization

## Usage

```bash
# Train the improved quantized model
python train_improved_quantized.py

# The script offers multiple options:
# 1. Train improved quantized model
# 2. Compare improved vs hybrid models
# 3. Show improvements explanation
```

## Key Improvements

- **Better Quantization Strategy**:
  - 8-bit Conv2D (instead of 4-bit)
  - 16-bit Dense layers (instead of 8-bit)
  - Less aggressive quantization = better gradients

- **Improved Activation Functions**:
  - Standard ReLU instead of quantized_relu
  - Better gradient flow
  - More stable training

- **Lighter Regularization**:
  - L2(0.001) instead of L1L2(0.01)
  - Less aggressive regularization
  - Better for quantized models

- **Hybrid Option Available**:
  - Train with full precision
  - Quantize only for inference
  - Best of both worlds

## Expected Improvements

1. Better training performance (higher accuracy)
2. Lower loss values
3. More stable training
4. Still maintains quantization benefits 