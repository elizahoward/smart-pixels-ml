import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from qkeras.quantizers import quantized_bits, quantized_relu

# Import quantized model
from quantized_cnn_model import build_quantized_cnn_model

# Import data generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import OptimizedDataGenerator4 as ODG

# Data directories
base_dir = Path("/local/d1/smartpixML/filtering_models/shuffling_data/filtering_records1024_data_shuffled_single")
train_dir = base_dir / "tfrecords_train"
val_dir = base_dir / "tfrecords_validation"

# Results directory
results_dir = Path(__file__).resolve().parent / "quantized_results"
os.makedirs(results_dir, exist_ok=True)

# Data generators
train_gen = ODG.OptimizedDataGenerator(
    load_records=True,
    tf_records_dir=str(train_dir),
    x_feature_description=['x_profile', 'z_global', 'y_profile', 'y_local']
)
val_gen = ODG.OptimizedDataGenerator(
    load_records=True,
    tf_records_dir=str(val_dir),
    x_feature_description=['x_profile', 'z_global', 'y_profile', 'y_local']
)

# Learning rate schedule config
config = {
    "kind": "polynomial_decay",
    "initial_lr": 1e-3,
    "end_lr": 1e-5,
    "power": 2,
}
n_epochs = 140
steps_per_epoch = len(train_gen)
decay_steps = steps_per_epoch * n_epochs

# Fractional bit configurations (weight_bits, integer_bits=0)
# weight_bits = fractional_bits + integer_bits, so for integer_bits=0, weight_bits = fractional_bits
fractional_bits = [2, 3, 4, 6, 8, 16, 32]
bit_settings = [(bits, 0) for bits in fractional_bits]  # integer_bits always 0

print(f"Testing {len(bit_settings)} fractional bit configurations")
print(f"Configurations: {fractional_bits} fractional bits (integer bits = 0)")

# Storage for results
acc_histories = {}
val_histories = {}
loss_histories = {}
val_loss_histories = {}
final_results = []

# Number of trials per configuration (for averaging)
n_trials = 8

for weight_bits, weight_integer_bits in bit_settings:
    activation_bits = 8
    activation_integer_bits = 0
    print(f"\n=== Testing {weight_bits}-bit quantized model (fractional bits: {weight_bits}, integer bits: {weight_integer_bits}) ===")
    print(f"  Weight quantizer: quantized_bits({weight_bits}, {weight_integer_bits}, 1)")
    print(f"  Activation quantizer: quantized_relu({activation_bits}, {activation_integer_bits})")
    
    # Storage for this configuration's trials
    trial_accuracies = []
    trial_val_accuracies = []
    trial_losses = []
    trial_val_losses = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}")
        
        if config["kind"] == "polynomial_decay":
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=config["initial_lr"],
                decay_steps=decay_steps,
                end_learning_rate=config["end_lr"],
                power=config["power"]
            )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=config["initial_lr"])
        
        model = build_quantized_cnn_model(
            weight_bits=weight_bits,
            weight_integer_bits=weight_integer_bits,
            activation_bits=activation_bits,
            activation_integer_bits=activation_integer_bits
        )
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks
        )
        
        # Store results for this trial
        trial_accuracies.append(history.history['accuracy'])
        trial_val_accuracies.append(history.history['val_accuracy'])
        trial_losses.append(history.history['loss'])
        trial_val_losses.append(history.history['val_loss'])
        
        # Save model for this trial
        model.save(results_dir / f'quantized_cnn_{weight_bits}bit_int{weight_integer_bits}_trial{trial+1}.h5')
    
    # Average the results from all trials
    # Handle variable-length histories due to early stopping
    min_length = min(len(acc) for acc in trial_accuracies)
    
    # Truncate all histories to the minimum length
    trial_accuracies_truncated = [acc[:min_length] for acc in trial_accuracies]
    trial_val_accuracies_truncated = [acc[:min_length] for acc in trial_val_accuracies]
    trial_losses_truncated = [loss[:min_length] for loss in trial_losses]
    trial_val_losses_truncated = [loss[:min_length] for loss in trial_val_losses]
    
    avg_acc = np.mean(trial_accuracies_truncated, axis=0)
    avg_val_acc = np.mean(trial_val_accuracies_truncated, axis=0)
    avg_loss = np.mean(trial_losses_truncated, axis=0)
    avg_val_loss = np.mean(trial_val_losses_truncated, axis=0)
    
    # Store averaged results
    config_key = f"{weight_bits}bit_int{weight_integer_bits}"
    acc_histories[config_key] = avg_acc
    val_histories[config_key] = avg_val_acc
    loss_histories[config_key] = avg_loss
    val_loss_histories[config_key] = avg_val_loss
    
    # Store final results for CSV
    final_results.append({
        'fractional_bits': weight_bits,
        'integer_bits': weight_integer_bits,
        'final_train_accuracy': avg_acc[-1],
        'final_val_accuracy': avg_val_acc[-1],
        'final_train_loss': avg_loss[-1],
        'final_val_loss': avg_val_loss[-1],
        'epochs_trained': len(avg_acc),
        'n_trials': n_trials
    })
    
    # Save averaged history
    np.savez(results_dir / f'quantized_cnn_{config_key}_history.npz', 
             accuracy=avg_acc, val_accuracy=avg_val_acc,
             loss=avg_loss, val_loss=avg_val_loss)

# Save results to CSV
results_df = pd.DataFrame(final_results)
results_df.to_csv(results_dir / 'quantized_model_results.csv', index=False)
print(f"\nSaved results to {results_dir / 'quantized_model_results.csv'}")

# Create validation plots for all configurations
print("Creating validation plots...")

# Create a single figure with subplots for each configuration
n_configs = len(bit_settings)
n_cols = 2
n_rows = (n_configs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
if n_configs == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes.reshape(1, -1)
else:
    axes = axes.flatten()

for i, (weight_bits, weight_integer_bits) in enumerate(bit_settings):
    config_key = f"{weight_bits}bit_int{weight_integer_bits}"
    
    # Get the subplot
    if i < len(axes):
        ax = axes[i]
    else:
        break
    
    # Plot validation accuracy and loss
    epochs = range(len(val_histories[config_key]))
    ax.plot(epochs, val_histories[config_key], 'b-', linewidth=2, label=f'Val Accuracy (avg over {n_trials} trials)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f'{weight_bits} fractional bits (int bits: {weight_integer_bits})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits for better comparison
    min_acc = min(val_histories[config_key])
    max_acc = max(val_histories[config_key])
    ax.set_ylim(max(0, min_acc - 0.05), min(1, max_acc + 0.05))

# Remove empty subplots
for i in range(len(bit_settings), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(results_dir / 'validation_accuracy_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Create validation loss plots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
if n_configs == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes.reshape(1, -1)
else:
    axes = axes.flatten()

for i, (weight_bits, weight_integer_bits) in enumerate(bit_settings):
    config_key = f"{weight_bits}bit_int{weight_integer_bits}"
    
    # Get the subplot
    if i < len(axes):
        ax = axes[i]
    else:
        break
    
    # Plot validation loss
    epochs = range(len(val_loss_histories[config_key]))
    ax.plot(epochs, val_loss_histories[config_key], 'r-', linewidth=2, label=f'Val Loss (avg over {n_trials} trials)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title(f'{weight_bits} fractional bits (int bits: {weight_integer_bits})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits for better comparison
    min_loss = min(val_loss_histories[config_key])
    max_loss = max(val_loss_histories[config_key])
    ax.set_ylim(max(0, min_loss - 0.05), max_loss + 0.05)

# Remove empty subplots
for i in range(len(bit_settings), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(results_dir / 'validation_loss_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Create combined comparison plot
plt.figure(figsize=(16, 10))

# Plot validation accuracy comparison
plt.subplot(2, 1, 1)
for weight_bits, weight_integer_bits in bit_settings:
    config_key = f"{weight_bits}bit_int{weight_integer_bits}"
    epochs = range(len(val_histories[config_key]))
    plt.plot(epochs, val_histories[config_key], linewidth=2, label=f'{weight_bits} fractional bits')

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison (200 epochs)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot validation loss comparison
plt.subplot(2, 1, 2)
for weight_bits, weight_integer_bits in bit_settings:
    config_key = f"{weight_bits}bit_int{weight_integer_bits}"
    epochs = range(len(val_loss_histories[config_key]))
    plt.plot(epochs, val_loss_histories[config_key], linewidth=2, label=f'{weight_bits} fractional bits')

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison (200 epochs)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'validation_comparison_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Print final results summary
print("\n=== FINAL RESULTS SUMMARY ===")
print(f"{'Fractional Bits':<15} {'Train Acc':<10} {'Val Acc':<10} {'Train Loss':<12} {'Val Loss':<10} {'Epochs':<8}")
print("-" * 70)

for result in final_results:
    print(f"{result['fractional_bits']:<15} "
          f"{result['final_train_accuracy']:<10.4f} "
          f"{result['final_val_accuracy']:<10.4f} "
          f"{result['final_train_loss']:<12.4f} "
          f"{result['final_val_loss']:<10.4f} "
          f"{result['epochs_trained']:<8}")

# Find best configuration
best_result = max(final_results, key=lambda x: x['final_val_accuracy'])
print(f"\nBEST CONFIGURATION:")
print(f"Fractional bits: {best_result['fractional_bits']}")
print(f"Validation accuracy: {best_result['final_val_accuracy']:.4f}")
print(f"Training accuracy: {best_result['final_train_accuracy']:.4f}")
print(f"Validation loss: {best_result['final_val_loss']:.4f}")
print(f"Training loss: {best_result['final_train_loss']:.4f}")

print(f"\nResults saved to: {results_dir}")
print("Files created:")
print("- quantized_model_results.csv (final metrics)")
print("- validation_accuracy_plots.png (individual accuracy plots)")
print("- validation_loss_plots.png (individual loss plots)")
print("- validation_comparison_plots.png (combined comparison)")
print("- Individual model files and history files") 