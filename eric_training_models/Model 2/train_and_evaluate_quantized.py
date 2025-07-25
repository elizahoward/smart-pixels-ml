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
base_dir = Path(__file__).resolve().parent.parent.parent / "filtering_models" / "filtering_records2048test"
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
    "power":       0.5,
}
n_epochs = 100
steps_per_epoch = len(train_gen)
decay_steps = steps_per_epoch * n_epochs

bit_settings = [2, 3, 4]
acc_histories = {}
val_histories = {}
roc_curves = {}
auc_scores = {}

for bits in bit_settings:
    print(f"\n=== Training {bits}-bit quantized model ===")
    weight_quantizer = quantized_bits(bits, bits, 1)
    activation_quantizer = quantized_relu(bits, bits)

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
        weight_quantizer=weight_quantizer,
        activation_quantizer=activation_quantizer
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
        verbose=2,
        callbacks=callbacks
    )
    acc_histories[bits] = history.history['accuracy']
    val_histories[bits] = history.history['val_accuracy']
    # ROC curve
    val_preds = model.predict(val_gen, verbose=0).ravel()
    val_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
    fpr, tpr, _ = roc_curve(val_true, val_preds)
    roc_curves[bits] = (fpr, tpr)
    auc_scores[bits] = auc(fpr, tpr)
    # Save model
    model.save(results_dir / f'quantized_cnn_{bits}bit.h5')
    # Save history
    np.savez(results_dir / f'quantized_cnn_{bits}bit_history.npz', accuracy=acc_histories[bits], val_accuracy=val_histories[bits])

# Overlayed accuracy plot
plt.figure()
for bits in bit_settings:
    plt.plot(acc_histories[bits], label=f'Train {bits}-bit')
    plt.plot(val_histories[bits], '--', label=f'Val {bits}-bit')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (Quantized Models)')
plt.legend()
plt.savefig(results_dir / 'quantized_accuracy_overlay.png')
plt.close()

# Overlayed ROC curve
plt.figure()
for bits in bit_settings:
    fpr, tpr = roc_curves[bits]
    plt.plot(fpr, tpr, label=f'{bits}-bit (AUC={auc_scores[bits]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Quantized Models)')
plt.legend(loc='lower right')
plt.savefig(results_dir / 'quantized_roc_overlay.png')
plt.close()

# Print final validation accuracy for each quantized model
print("\nFinal Validation Accuracies:")
for bits in bit_settings:
    print(f"  {bits}-bit: {val_histories[bits][-1]:.4f}")

print("Saved quantized model overlays and histories in results directory.") 