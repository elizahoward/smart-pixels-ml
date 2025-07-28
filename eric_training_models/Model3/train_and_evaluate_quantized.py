import os
os.environ.pop("TF_USE_LEGACY_KERAS", None)
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

# Import quantized model
from cnn_model_quantized import build_quantized_cnn_model

# Import data generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import OptimizedDataGenerator4 as ODG

# Data directories
base_dir = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
train_dir = base_dir / "tfrecords_train"
val_dir = base_dir / "tfrecords_validation"

# Results directory
results_dir = Path(__file__).resolve().parent / "results_quantized"
os.makedirs(results_dir, exist_ok=True)

# Data generators - only need cluster, z_global, y_local
train_gen = ODG.OptimizedDataGenerator(
    load_records=True,
    tf_records_dir=str(train_dir),
    x_feature_description=['cluster', 'z_global', 'y_local']
)
val_gen = ODG.OptimizedDataGenerator(
    load_records=True,
    tf_records_dir=str(val_dir),
    x_feature_description=['cluster', 'z_global', 'y_local']
)

# Learning rate schedule config
config = {
    "kind": "polynomial_decay",
    "initial_lr": 1e-3,
    "end_lr": 1e-5,
    "power": 0.5,
}
n_epochs = 120
steps_per_epoch = len(train_gen)
decay_steps = steps_per_epoch * n_epochs

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

# Build and compile quantized model
print("=== Building Quantized CNN Model ===")
model = build_quantized_cnn_model()
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=40,
        restore_best_weights=True
    )
]

# Train
print("=== Training Quantized CNN Model ===")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=n_epochs,
    verbose=2,
    callbacks=callbacks
)

# Evaluate
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (Quantized)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (Quantized)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'training_plots_quantized.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC curve
val_preds = model.predict(val_gen, verbose=0).ravel()
val_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])

fpr, tpr, thresholds = roc_curve(val_true, val_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Quantized)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(results_dir / 'roc_curve_quantized.png', dpi=300, bbox_inches='tight')
plt.close()

# Save training history
lr_hist = [float(lr_schedule(step * steps_per_epoch).numpy()) for step in range(len(train_acc))]
np.savez(results_dir / 'training_history_quantized.npz', 
         accuracy=train_acc, 
         val_accuracy=val_acc, 
         loss=train_loss,
         val_loss=val_loss,
         learning_rate=lr_hist)

# Save quantized model
model.save(results_dir / 'quantized_cnn_model.h5')

# Convert to TFLite with quantization for actual size reduction
print("=== Converting to Quantized TFLite ===")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert the model
tflite_model = converter.convert()

# Save TFLite model
tflite_path = results_dir / 'quantized_cnn_model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

# Also try 8-bit quantization for maximum compression
print("=== Converting to 8-bit Quantized TFLite ===")
converter_8bit = tf.lite.TFLiteConverter.from_keras_model(model)
converter_8bit.optimizations = [tf.lite.Optimize.DEFAULT]
converter_8bit.representative_dataset = lambda: iter([(tf.random.normal((1, 13, 21, 20)), 
                                                      tf.random.normal((1, 1)), 
                                                      tf.random.normal((1, 1))) for _ in range(100)])

try:
    tflite_model_8bit = converter_8bit.convert()
    tflite_8bit_path = results_dir / 'quantized_cnn_model_8bit.tflite'
    with open(tflite_8bit_path, 'wb') as f:
        f.write(tflite_model_8bit)
    print("8-bit quantization successful")
except Exception as e:
    print(f"8-bit quantization failed: {e}")
    tflite_8bit_path = None

print(f"\n=== Quantized Model Results Summary ===")
print(f"Final Training Accuracy: {train_acc[-1]:.4f}")
print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Epochs trained: {len(train_acc)}")
print(f"\nFiles saved to: {results_dir}")
print(f"- quantized_cnn_model.h5 (quantized model)")
print(f"- training_history_quantized.npz (training history)")
print(f"- training_plots_quantized.png (accuracy/loss plots)")
print(f"- roc_curve_quantized.png (ROC curve)")

# Model size comparison
import os
original_model_path = Path(__file__).resolve().parent / "results" / "cnn_model.h5"
quantized_model_path = results_dir / "quantized_cnn_model.h5"
tflite_model_path = results_dir / "quantized_cnn_model.tflite"

if original_model_path.exists():
    original_size = os.path.getsize(original_model_path) / (1024 * 1024)  # MB
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)  # MB
    
    print(f"\n=== Model Size Comparison ===")
    print(f"Original model (H5): {original_size:.2f} MB")
    print(f"QKeras model (H5): {quantized_size:.2f} MB")
    print(f"TFLite model (FP16): {tflite_size:.2f} MB")
    print(f"Size reduction (TFLite vs Original): {((original_size - tflite_size) / original_size * 100):.1f}%")
    
    if tflite_8bit_path and tflite_8bit_path.exists():
        tflite_8bit_size = os.path.getsize(tflite_8bit_path) / (1024 * 1024)  # MB
        print(f"TFLite model (8-bit): {tflite_8bit_size:.2f} MB")
        print(f"Size reduction (8-bit vs Original): {((original_size - tflite_8bit_size) / original_size * 100):.1f}%")
    
    print(f"\nNote: QKeras H5 files don't reduce size, but TFLite conversion does.")
    print(f"TFLite models are ready for deployment on edge devices.") 