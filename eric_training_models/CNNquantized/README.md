# CNN Quantized Model System

This folder contains a consolidated CNN quantized model system that combines the best features from all three model versions (first_model, middle_model, final_model) with the ability to save and load trained models to avoid retraining every time.

## Files Overview

### Core Model Files
- **`consolidated_quantized_model.py`**: Main model file that combines all three quantization strategies
- **`train_and_save_model.py`**: Script to train and save a model for later use
- **`train_and_test_quantized_with_saved_model.py`**: Modified training script that can load saved models

### Model Features

The consolidated model supports three quantization strategies:

1. **`qkeras_training`**: QKeras quantization during training (from first_model)
   - Uses 4-bit quantization for Conv2D layers
   - Uses 8-bit quantization for Dense layers
   - Quantized activations

2. **`improved_qkeras`**: Improved quantization with better gradient flow (from middle_model)
   - Uses 8-bit quantization for Conv2D layers
   - Uses 16-bit quantization for Dense layers
   - Standard ReLU activations for better gradient flow

3. **`post_training`**: Post-training quantization for better bias handling (from final_model)
   - Standard layers during training
   - Quantization applied after training for inference
   - Best bias gradient handling

## Usage Instructions

### Step 1: Train and Save a Model

First, train a model and save it for later use:

```bash
cd smart_pixels_ml/eric_training_models/CNNquantized
python train_and_save_model.py
```

This will:
- Train a model using the post-training quantization strategy
- Save the trained model to `saved_trained_model/`
- Show training results and weight analysis

### Step 2: Test Quantization with Saved Model

Once you have a saved model, you can test different quantization levels without retraining:

```bash
python train_and_test_quantized_with_saved_model.py
```

This will:
- Load the saved model (or train a new one if not available)
- Test quantization at different bit levels (32, 16, 8, 4, 2 bits)
- Show accuracy comparison tables
- Analyze weight and bias distributions
- Calculate model size reductions

## Key Benefits

1. **No More Retraining**: Once you train and save a model, you can test different quantization levels without retraining
2. **Comprehensive Analysis**: Tests both weights AND biases quantization
3. **Multiple Strategies**: Combines the best features from all three model versions
4. **Easy Comparison**: Side-by-side comparison of different quantization levels
5. **Detailed Analysis**: Weight distribution analysis and MSE calculations

## Model Architecture

The consolidated model uses:
- **Inputs**: cluster (13x21), z_global (1), y_local (1)
- **Conv2D branch**: 32 filters, 3x3 kernel, max pooling
- **Dense branches**: 32 units each for z_global and y_local
- **Head layers**: 96 → 48 units with dropout
- **Output**: Binary classification (sigmoid)

## Quantization Testing

The system tests quantization at:
- **32-bit**: Original full precision
- **16-bit**: 50% size reduction
- **8-bit**: 75% size reduction (recommended)
- **4-bit**: 87.5% size reduction
- **2-bit**: 93.75% size reduction

## File Structure

```
CNNquantized/
├── consolidated_quantized_model.py          # Main model with all strategies
├── train_and_save_model.py                 # Train and save model
├── train_and_test_quantized_with_saved_model.py  # Test quantization with saved model
├── saved_trained_model/                    # Saved model directory (created after training)
└── README.md                               # This file
```

## Troubleshooting

1. **Import Errors**: Make sure TensorFlow and QKeras are installed
2. **Data Path Issues**: Verify the data directory paths in the scripts
3. **Model Not Found**: Run `train_and_save_model.py` first to create a saved model
4. **Memory Issues**: Reduce batch size or model complexity if needed

## Integration with Original Scripts

The original `train_and_test_quantized_with_bias.py` can be modified to use the saved model by:

1. Import the load function: `from consolidated_quantized_model import load_trained_model`
2. Replace the training section with model loading
3. Use the same quantization testing functions

This provides a seamless way to avoid retraining while maintaining all the original functionality. 