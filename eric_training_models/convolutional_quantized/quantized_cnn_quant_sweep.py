import os
import numpy as np
import pandas as pd
from pathlib import Path
import quantized_cnn_classifier as qcnn_mod

# Set up sweep parameters
bit_options = [2, 3, 6, 8]
results = []

# Output directory for sweep results
sweep_dir = Path(__file__).resolve().parent / 'quant_sweep_results'
sweep_dir.mkdir(exist_ok=True)
results_csv = sweep_dir / 'quantization_sweep_results.csv'

# Data/config (update if needed)
base_dir = str(Path(__file__).resolve().parent.parent.parent / "filtering_models" / "filtering_records2048test")
results_dir = str(sweep_dir)

for w_bits in bit_options:
    for a_bits in bit_options:
        print(f"\n=== Training with weight_quantizer={w_bits} bits, activation_quantizer={a_bits} bits ===")
        # Build quantizers
        from qkeras.quantizers import quantized_bits, quantized_relu
        weight_quantizer = quantized_bits(w_bits, 0, 1)
        activation_quantizer = quantized_relu(a_bits, 0)

        # Train and evaluate with explicit quantizer arguments
        result = qcnn_mod.train_and_evaluate_quantized_cnn(
            model_name=f"cnn_w{w_bits}_a{a_bits}",
            base_dir=base_dir,
            results_dir=results_dir,
            epochs=40,
            weight_quantizer=weight_quantizer,
            activation_quantizer=activation_quantizer
        )
        val_acc = result['val_acc']
        print(f"Validation accuracy for w={w_bits}, a={a_bits}: {val_acc:.4f}")
        results.append({
            'weight_bits': w_bits,
            'activation_bits': a_bits,
            'val_accuracy': val_acc
        })

# Save results as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(results_csv, index=False)
print(f"\nSweep complete. Results saved to {results_csv}")
print(results_df.pivot(index='weight_bits', columns='activation_bits', values='val_accuracy')) 