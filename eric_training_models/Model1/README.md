# Model1 Training Scripts

This directory contains scripts for training and evaluating Model1, a quantized multilayer perceptron (MLP) for binary classification.

## Model Architecture

Model1 is a simple MLP with the following architecture:
- **Input**: 4 features (z_global, x_size, y_size, y_local)
- **Hidden layers**: 17 → 20 → 9 → 16 → 8 neurons
- **Output**: 1 neuron with smooth_sigmoid activation
- **Quantization**: Configurable bit-width for weights and activations

## Files

- `quantized_mlp_model.py`: Model architecture definitions
- `train_quantized_model1.py`: Main training script with multiple trials and bit configurations
- `plot_model1_results.py`: Results visualization script
- `README.md`: This file

## Usage

### 1. Training Models

Run the training script to train both non-quantized and quantized versions with multiple bit configurations:

```bash
cd PixelML/smart_pixels_ml/eric_training_models/Model1
python train_quantized_model1.py
```

Optional arguments:
- `--epochs N`: Number of epochs per trial (default: 150)
- `--trials N`: Number of trials per configuration (default: 6)

Example with custom settings:
```bash
python train_quantized_model1.py --epochs 200 --trials 10
```

### 2. Generating Plots

After training is complete, generate comprehensive plots and analysis:

```bash
python plot_model1_results.py quantized_model1_results_YYYYMMDD_HHMMSS/
```

Replace `YYYYMMDD_HHMMSS` with the actual timestamp from your training run.

## Bit Configurations Tested

The training script automatically tests the following bit configurations:
- Non-quantized (baseline)
- 2-bit quantization (0 integer bits, 2 fractional bits)
- 3-bit quantization (0 integer bits, 3 fractional bits)
- 4-bit quantization (0 integer bits, 4 fractional bits)
- 6-bit quantization (0 integer bits, 6 fractional bits)
- 8-bit quantization (0 integer bits, 8 fractional bits)
- 16-bit quantization (0 integer bits, 16 fractional bits)
- 32-bit quantization (0 integer bits, 32 fractional bits)

## Output Structure

Training creates a timestamped results directory with the following structure:

```
quantized_model1_results_YYYYMMDD_HHMMSS/
├── non_quantized_model/
│   ├── trial_1/
│   ├── trial_2/
│   ├── ...
│   ├── overall_results.json
│   └── non_quantized_model_history.npz
├── quantized_2w0i_2a0i/
│   ├── trial_1/
│   ├── trial_2/
│   ├── ...
│   ├── overall_results.json
│   └── quantized_2w0i_2a0i_history.npz
├── quantized_3w0i_3a0i/
│   └── ...
├── all_roc_data.json
├── validation_comparison.png
├── best_metrics_comparison.png
├── roc_curves_overlay.png
├── model1_results_summary.csv
└── model1_results_summary.txt
```

## Generated Plots

The plotting script generates:

1. **Validation Comparison**: Training curves for all configurations
2. **Best Metrics Comparison**: Bar charts comparing best validation accuracy, loss, and ROC AUC
3. **ROC Curves Overlay**: ROC curves for all models on one plot
4. **Summary Tables**: CSV and text summaries of all results

## Data Requirements

The training script expects data in the following format:
- TensorFlow records in the specified directories
- Features: `z_global`, `x_size`, `y_size`, `y_local`
- Binary classification labels

Modify the data paths in `train_quantized_model1.py` if needed:
```python
self.base_dir = Path("/your/data/path")
```

## Dependencies

- TensorFlow
- QKeras
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## Notes

- Each configuration is trained for multiple trials to ensure statistical significance
- Early stopping is used to prevent overfitting
- Results are averaged across trials for robust comparisons
- All models use polynomial learning rate decay
- ROC curves are averaged across trials using interpolation