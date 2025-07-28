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
base_dir = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
train_dir = base_dir / "tfrecords_train"
val_dir = base_dir / "tfrecords_validation"

# Results directory
results_dir = Path(__file__).resolve().parent / "results"
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
    "end_lr":      1e-5,
    "power":       2,
}
n_epochs = 120
steps_per_epoch = len(train_gen)
decay_steps = steps_per_epoch * n_epochs


# Predefined bit configurations (weight_bits, integer_bits)
bit_settings = [
    (2, 0), (2, 1),
    (3, 0), (3, 1), (3, 2),
    (4, 0), (4, 1), (4, 2),
    (5, 0), (5, 1), (5, 2),
    (6, 0), (6, 1), (6, 2),
    (8, 0), (8, 1), (8, 2), (8, 3), (8, 4),
    (10, 0), (10, 2), (10, 4), (10, 6),
    (16, 0), (16, 4), (16, 8), (16, 12),
    (20, 0), (20, 5),
    (24, 0), (24, 6),
    (32, 0), (32, 8), (32, 16),
    (48, 0), (48, 12),
    (64, 0), (64, 16),
    (96, 0), (96, 24),
    (128, 0), (128, 32)
]

print(f"Testing {len(bit_settings)} bit configurations")

# Storage for results
acc_histories = {}
val_histories = {}
loss_histories = {}
val_loss_histories = {}
roc_curves = {}
auc_scores = {}
final_results = []

# Number of trials per configuration (for averaging)
n_trials = 2

for weight_bits, weight_integer_bits in bit_settings:
    activation_bits = 16
    activation_integer_bits = 0
    print(f"\n=== Testing {weight_bits}-bit quantized model (int bits: {weight_integer_bits}) ===")
    print(f"  Weight quantizer: quantized_bits({weight_bits}, {weight_integer_bits}, 1)")
    print(f"  Activation quantizer: quantized_relu({activation_bits}, {activation_integer_bits})")
    
    # Storage for this configuration's trials
    trial_accuracies = []
    trial_val_accuracies = []
    trial_losses = []
    trial_val_losses = []
    trial_aucs = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}")
        
        if config["kind"] == "polynomial_decay":
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate = config["initial_lr"],
                decay_steps           = decay_steps,
                end_learning_rate     = config["end_lr"],
                power                 = config["power"]
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
                patience=40,
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
        
        # ROC curve and AUC
        val_preds = model.predict(val_gen, verbose=0).ravel()
        val_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
        fpr, tpr, _ = roc_curve(val_true, val_preds)
        trial_aucs.append(auc(fpr, tpr))
        
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
    avg_auc = np.mean(trial_aucs)
    
    # Store averaged results
    config_key = f"{weight_bits}bit_int{weight_integer_bits}"
    acc_histories[config_key] = avg_acc
    val_histories[config_key] = avg_val_acc
    loss_histories[config_key] = avg_loss
    val_loss_histories[config_key] = avg_val_loss
    auc_scores[config_key] = avg_auc
    
    # Store final results for CSV
    final_results.append({
        'weight_bits': weight_bits,
        'integer_bits': weight_integer_bits,
        'final_train_accuracy': avg_acc[-1],
        'final_val_accuracy': avg_val_acc[-1],
        'final_train_loss': avg_loss[-1],
        'final_val_loss': avg_val_loss[-1],
        'auc_score': avg_auc,
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

# Create individual plots folder
individual_plots_dir = results_dir / 'individual_plots'
os.makedirs(individual_plots_dir, exist_ok=True)

# Create individual plots for each configuration
print("Creating individual plots...")
for config_key in acc_histories.keys():
    # Extract bit configuration info
    if 'bit_int' in config_key:
        weight_bits = config_key.split('bit_int')[0]
        integer_bits = config_key.split('int')[1]
        title_suffix = f"{weight_bits}-bit, {integer_bits} int bits"
    else:
        title_suffix = config_key
    
    # Create individual plot with training and validation on same graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training and Validation Accuracy on same plot
    epochs = range(len(acc_histories[config_key]))
    ax1.plot(epochs, acc_histories[config_key], 'b-', linewidth=2, label=f'Train (avg over {n_trials} trials)')
    ax1.plot(epochs, val_histories[config_key], 'r-', linewidth=2, label=f'Val (avg over {n_trials} trials)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Accuracy - {title_suffix}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set y-axis limits for better comparison
    min_acc = min(min(acc_histories[config_key]), min(val_histories[config_key]))
    max_acc = max(max(acc_histories[config_key]), max(val_histories[config_key]))
    ax1.set_ylim(max(0, min_acc - 0.05), min(1, max_acc + 0.05))
    
    # Training and Validation Loss on same plot
    ax2.plot(epochs, loss_histories[config_key], 'g-', linewidth=2, label=f'Train (avg over {n_trials} trials)')
    ax2.plot(epochs, val_loss_histories[config_key], 'm-', linewidth=2, label=f'Val (avg over {n_trials} trials)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Loss - {title_suffix}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(individual_plots_dir / f'{config_key}_individual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"Created {len(acc_histories)} individual plots in {individual_plots_dir} (averaged over {n_trials} trials each)")

# Create conglomerated plots with selected configurations
print("Creating conglomerated plots...")

# Select representative configurations for overlay plots
selected_configs = []
config_keys = list(acc_histories.keys())

# Select configurations: lowest, medium, and highest bit widths
if len(config_keys) >= 3:
    # Get configurations with different bit widths
    low_bit_configs = [k for k in config_keys if int(k.split('bit')[0]) <= 8][:3]
    mid_bit_configs = [k for k in config_keys if 8 < int(k.split('bit')[0]) <= 32][:3]
    high_bit_configs = [k for k in config_keys if int(k.split('bit')[0]) > 32][:3]
    
    selected_configs = low_bit_configs + mid_bit_configs + high_bit_configs
else:
    selected_configs = config_keys[:min(9, len(config_keys))]

# Create single conglomerated plot with training and validation on same graph
plt.figure(figsize=(20, 15))

# Create subplots for each selected configuration
n_configs = len(selected_configs)
n_cols = 3
n_rows = (n_configs + n_cols - 1) // n_cols

for i, config_key in enumerate(selected_configs):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # Extract configuration info for title
    if 'bit_int' in config_key:
        weight_bits = config_key.split('bit_int')[0]
        integer_bits = config_key.split('int')[1]
        title = f"{weight_bits}-bit, {integer_bits} int bits"
    else:
        title = config_key
    
    # Plot training and validation accuracy on same graph
    epochs = range(len(acc_histories[config_key]))
    plt.plot(epochs, acc_histories[config_key], 'b-', linewidth=2, label=f'Train (avg over {n_trials} trials)')
    plt.plot(epochs, val_histories[config_key], 'r-', linewidth=2, label=f'Val (avg over {n_trials} trials)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits for better comparison
    min_acc = min(min(acc_histories[config_key]), min(val_histories[config_key]))
    max_acc = max(max(acc_histories[config_key]), max(val_histories[config_key]))
    plt.ylim(max(0, min_acc - 0.05), min(1, max_acc + 0.05))

plt.tight_layout()
plt.savefig(results_dir / 'quantized_conglomerated_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# Final comparison bar chart
plt.figure(figsize=(16, 10))

# Extract final metrics for all configurations
configs = list(val_histories.keys())
final_accs = [val_histories[config][-1] for config in configs]
final_losses = [val_loss_histories[config][-1] for config in configs]

# Create labels
if 'bit_int' in configs[0]:
    labels = [f"{config.split('bit_int')[0]}b,{config.split('int')[1]}" for config in configs]
else:
    labels = configs

# Final accuracy comparison
plt.subplot(2, 1, 1)
bars1 = plt.bar(range(len(configs)), final_accs, color='skyblue', alpha=0.7)
plt.xlabel('Bit Configuration')
plt.ylabel('Final Validation Accuracy')
plt.title('Final Validation Accuracy by Bit Configuration')
plt.xticks(range(len(configs)), labels, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Final loss comparison
plt.subplot(2, 1, 2)
bars2 = plt.bar(range(len(configs)), final_losses, color='lightcoral', alpha=0.7)
plt.xlabel('Bit Configuration')
plt.ylabel('Final Validation Loss')
plt.title('Final Validation Loss by Bit Configuration')
plt.xticks(range(len(configs)), labels, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, bar in enumerate(bars2):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(results_dir / 'quantized_final_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total configurations tested: {len(final_results)}")
print(f"Trials per configuration: {n_trials}")
print(f"Best validation accuracy: {max(final_results, key=lambda x: x['final_val_accuracy'])['final_val_accuracy']:.4f}")
print(f"Best configuration: {max(final_results, key=lambda x: x['final_val_accuracy'])['weight_bits']} bits, {max(final_results, key=lambda x: x['final_val_accuracy'])['integer_bits']} integer bits")
print(f"Best AUC score: {max(final_results, key=lambda x: x['auc_score'])['auc_score']:.4f}")

print(f"\nResults saved to: {results_dir}")
print("Files created:")
print("- quantized_model_results.csv (final metrics)")
print("- quantized_comprehensive_plots.png (training curves)")
print("- quantized_final_comparison.png (final comparison)")
print("- Individual model files and history files") 