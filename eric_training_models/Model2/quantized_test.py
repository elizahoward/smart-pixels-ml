import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
base_dir = Path(__file__).resolve().parent.parent.parent / "filtering_models" / "filtering_records2000"
train_dir = base_dir / "tfrecords_train"
val_dir = base_dir / "tfrecords_validation"

# Results directory - save in same folder as script
results_dir = Path(__file__).resolve().parent

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

# Model configuration
weight_bits = 64
weight_integer_bits = 8
activation_bits = 25
activation_integer_bits = 0
n_epochs = 500

# Learning rate schedule config
config = {
    "kind": "polynomial_decay",
    "initial_lr": 1e-3,
    "end_lr": 1e-5,
    "power": 2,
}
steps_per_epoch = len(train_gen)
decay_steps = steps_per_epoch * n_epochs

print(f"=== Testing Quantized Model Configuration ===")
print(f"Weight bits: {weight_bits}")
print(f"Weight integer bits: {weight_integer_bits}")
print(f"Activation bits: {activation_bits}")
print(f"Activation integer bits: {activation_integer_bits}")
print(f"Epochs: {n_epochs}")
print(f"Learning rate schedule: {config['kind']}")
print(f"Initial learning rate: {config['initial_lr']}")
print(f"End learning rate: {config['end_lr']}")
print(f"Power: {config['power']}")

# Build and compile model
model = build_quantized_cnn_model(
    weight_bits=weight_bits,
    weight_integer_bits=weight_integer_bits,
    activation_bits=activation_bits,
    activation_integer_bits=activation_integer_bits
)

# Set up learning rate schedule
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
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=70,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=18,
        min_lr=1e-7
    )
]

# Train model
print("\n=== Training Model ===")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=n_epochs,
    verbose=1,
    callbacks=callbacks
)

# Evaluate model
print("\n=== Evaluating Model ===")
val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
print(f"Final validation accuracy: {val_accuracy:.4f}")
print(f"Final validation loss: {val_loss:.4f}")

# ROC curve and AUC
val_preds = model.predict(val_gen, verbose=0).ravel()
val_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
fpr, tpr, _ = roc_curve(val_true, val_preds)
auc_score = auc(fpr, tpr)
print(f"AUC score: {auc_score:.4f}")

# Save model
model.save(results_dir / f'quantized_test_{weight_bits}bit_int{weight_integer_bits}.h5')

# Save history
np.savez(results_dir / f'quantized_test_{weight_bits}bit_int{weight_integer_bits}_history.npz',
         accuracy=history.history['accuracy'],
         val_accuracy=history.history['val_accuracy'],
         loss=history.history['loss'],
         val_loss=history.history['val_loss'])

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy plot
epochs = range(len(history.history['accuracy']))
ax1.plot(epochs, history.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
ax1.plot(epochs, history.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.set_title(f'Accuracy - {weight_bits}-bit, {weight_integer_bits} int bits')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Set y-axis limits for accuracy
min_acc = min(min(history.history['accuracy']), min(history.history['val_accuracy']))
max_acc = max(max(history.history['accuracy']), max(history.history['val_accuracy']))
ax1.set_ylim(max(0, min_acc - 0.05), min(1, max_acc + 0.05))

# Loss plot
ax2.plot(epochs, history.history['loss'], 'g-', linewidth=2, label='Training Loss')
ax2.plot(epochs, history.history['val_loss'], 'm-', linewidth=2, label='Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title(f'Loss - {weight_bits}-bit, {weight_integer_bits} int bits')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / f'quantized_test_{weight_bits}bit_int{weight_integer_bits}_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - {weight_bits}-bit, {weight_integer_bits} int bits')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(results_dir / f'quantized_test_{weight_bits}bit_int{weight_integer_bits}_roc.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n=== Results Summary ===")
print(f"Configuration: {weight_bits}-bit weights, {weight_integer_bits} integer bits")
print(f"Activation: {activation_bits}-bit")
print(f"Final validation accuracy: {val_accuracy:.4f}")
print(f"Final validation loss: {val_loss:.4f}")
print(f"AUC score: {auc_score:.4f}")
print(f"Epochs trained: {len(history.history['accuracy'])}")
print(f"\nFiles saved to: {results_dir}")
print(f"- quantized_test_{weight_bits}bit_int{weight_integer_bits}.h5 (model)")
print(f"- quantized_test_{weight_bits}bit_int{weight_integer_bits}_history.npz (training history)")
print(f"- quantized_test_{weight_bits}bit_int{weight_integer_bits}_plot.png (accuracy/loss plot)")
print(f"- quantized_test_{weight_bits}bit_int{weight_integer_bits}_roc.png (ROC curve)") 