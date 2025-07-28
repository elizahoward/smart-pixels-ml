import os
os.environ.pop("TF_USE_LEGACY_KERAS", None)
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

# Import model
from cnn_model import build_cnn_model

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
n_epochs = 160
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

# Build and compile model
model = build_cnn_model()
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
print("=== Training CNN Model ===")
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
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
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
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(results_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Save training history
lr_hist = [float(lr_schedule(step * steps_per_epoch).numpy()) for step in range(len(train_acc))]
np.savez(results_dir / 'training_history.npz', 
         accuracy=train_acc, 
         val_accuracy=val_acc, 
         loss=train_loss,
         val_loss=val_loss,
         learning_rate=lr_hist)

# Save model
model.save(results_dir / 'cnn_model.h5')

print(f"\n=== Results Summary ===")
print(f"Final Training Accuracy: {train_acc[-1]:.4f}")
print(f"Final Validation Accuracy: {val_acc[-1]:.4f}")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Epochs trained: {len(train_acc)}")
print(f"\nFiles saved to: {results_dir}")
print(f"- cnn_model.h5 (model)")
print(f"- training_history.npz (training history)")
print(f"- training_plots.png (accuracy/loss plots)")
print(f"- roc_curve.png (ROC curve)") 