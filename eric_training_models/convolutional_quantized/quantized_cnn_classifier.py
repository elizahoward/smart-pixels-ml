import os
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers.schedules import PolynomialDecay

# QKeras imports
try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OptimizedDataGenerator4 as ODG
import matplotlib.pyplot as plt

def build_quantized_cnn_model(cluster_shape=(13, 21, 20), z_global_length=1, y_local_length=1, weight_quantizer=None, activation_quantizer=None):
    """
    Build a quantized CNN for sig/bib classification using the last timestamp of cluster (2D), z_global, and y_local statistics.
    Args:
        cluster_shape (tuple): Shape of cluster input (default (13, 21, 20))
        z_global_length (int): Length of z_global (default 1)
        y_local_length (int): Length of y_local (default 1)
        weight_quantizer: QKeras quantizer for weights
        activation_quantizer: QKeras quantizer for activations
    Returns:
        tf.keras.Model: Compiled quantized CNN model
    """
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")
    if weight_quantizer is None:
        weight_quantizer = quantized_bits(8, 4, 1)
    if activation_quantizer is None:
        activation_quantizer = quantized_relu(8, 4)

    # Inputs
    cluster_input = tf.keras.layers.Input(shape=cluster_shape, name="cluster")
    z_global_input = tf.keras.layers.Input(shape=(z_global_length,), name="z_global")
    y_local_input = tf.keras.layers.Input(shape=(y_local_length,), name="y_local")

    # Take only the last timestamp (last axis)
    x = tf.keras.layers.Lambda(lambda x: x[..., -1], name="last_timestamp_slice")(cluster_input)  # shape (height, width)
    x = tf.keras.layers.Reshape((cluster_shape[0], cluster_shape[1], 1), name="add_channel")(x)
    # Quantized Conv2D branch
    x = QConv2D(32, (3, 3), padding="same", kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="conv2d")(x)
    x = QActivation(activation_quantizer, name="conv2d_act")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)
    x = QConv2D(64, (3, 3), padding="same", kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="conv2d_1")(x)
    x = QActivation(activation_quantizer, name="conv2d_1_act")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)

    # z_global branch
    z_dense = QDense(32, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="dense_z")(z_global_input)
    z_act = QActivation(activation_quantizer, name="z_act")(z_dense)
    # y_local branch
    y_dense = QDense(32, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="dense_y")(y_local_input)
    y_act = QActivation(activation_quantizer, name="y_act")(y_dense)

    # Concatenate all
    merged = tf.keras.layers.Concatenate(name="concatenate")([x, z_act, y_act])
    h = QDense(96, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="dense_1")(merged)
    h = QActivation(activation_quantizer, name="dense_1_act")(h)
    h = tf.keras.layers.Dropout(0.3, name="dropout")(h)
    h = QDense(48, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="dense_2")(h)
    h = QActivation(activation_quantizer, name="dense_2_act")(h)
    output = QDense(1, activation="sigmoid", kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="dense_3")(h)

    model = Model([cluster_input, z_global_input, y_local_input], output, name="quantized_cnn_classifier")
    optimizer = Adam(learning_rate=0.001)  # Will be replaced by lr_schedule in train_and_evaluate
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate_quantized_cnn(model_name, base_dir, results_dir, epochs=150, weight_quantizer=None, activation_quantizer=None):
    """
    Train and evaluate the quantized CNN sig/bib classifier.
    Args:
        model_name (str): Name for the model
        base_dir (str): Directory containing tfrecords
        results_dir (str): Directory to save results
        epochs (int): Number of training epochs
        weight_quantizer: QKeras quantizer for weights (optional)
        activation_quantizer: QKeras quantizer for activations (optional)
    Returns:
        dict: Training results
    """
    print(f"\n=== Training Quantized CNN {model_name} ===")

    train_dir = Path(base_dir) / "tfrecords_train"
    val_dir = Path(base_dir) / "tfrecords_validation"

    # Data generators - load cluster, z_global, y_local
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

    # Print shape of first batch for verification
    X_batch, y_batch = train_gen[0]
    print(f"First batch cluster shape: {X_batch['cluster'].shape}")
    print(f"First batch z_global shape: {X_batch['z_global'].shape}")
    print(f"First batch y_local shape: {X_batch['y_local'].shape}")
    print(f"First batch labels shape: {y_batch.shape}")

    # Build model with custom quantizers if provided
    model = build_quantized_cnn_model(
        cluster_shape=X_batch['cluster'].shape[1:],
        z_global_length=X_batch['z_global'].shape[1],
        y_local_length=X_batch['y_local'].shape[1],
        weight_quantizer=weight_quantizer,
        activation_quantizer=activation_quantizer
    )

    # --- Polynomial Decay Learning Rate Schedule ---
    steps_per_epoch = len(train_gen)
    lr_schedule = PolynomialDecay(
        initial_learning_rate=0.001,
        decay_steps=steps_per_epoch * epochs,
        end_learning_rate=1e-4,
        power=1.0
    )
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\nQuantized CNN Model Summary:")
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]

    print(f"\nFinal Training Metrics:")
    print(f"  Accuracy:  {train_acc:.4f}")

    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:  {val_acc:.4f}")

    # ROC curve
    y_score = model.predict(val_gen, verbose=0).ravel()
    y_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    os.makedirs(results_dir, exist_ok=True)
    model.save(os.path.join(results_dir, f'{model_name}_quantized_cnn_model.h5'))

    # --- Plot ROC curve ---
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, f'{model_name}_roc_curve.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"ROC curve saved to {roc_path}")

    # --- Plot accuracy curves ---
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path = os.path.join(results_dir, f'{model_name}_accuracy_curve.png')
    plt.savefig(acc_path)
    plt.close()
    print(f"Accuracy curve saved to {acc_path}")

    quantization_info = {
        'model_type': 'quantized_cnn',
        'weight_quantizer': str(weight_quantizer) if weight_quantizer is not None else 'default',
        'activation_quantizer': str(activation_quantizer) if activation_quantizer is not None else 'default'
    }

    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'roc_auc': roc_auc,
        'quantization_info': quantization_info
    }

if __name__ == "__main__":
    if not QKERAS_AVAILABLE:
        print("QKeras not available. Please install with: pip install qkeras")
        exit(1)

    # Configuration (update these paths as needed)
    base_dir = str(Path(__file__).resolve().parent.parent.parent / "filtering_models" / "filtering_records2048test")
    results_dir = str(Path(__file__).resolve().parent / 'results')

    print(f"Base directory: {base_dir}")
    print(f"Results directory: {results_dir}")

    # Train and evaluate quantized CNN model
    results = train_and_evaluate_quantized_cnn(
        model_name="quantized_cnn_classifier",
        base_dir=base_dir,
        results_dir=results_dir,
        epochs=150
    )

    print(f"\n=== Final Quantized CNN Results ===")
    print(f"Training Accuracy: {results['train_acc']:.4f}")
    print(f"Validation Accuracy: {results['val_acc']:.4f}")
    print(f"ROC AUC:          {results['roc_auc']:.4f}")
    print(f"Quantization Info: {results['quantization_info']}")
    print(f"\nResults saved in: {results_dir}") 