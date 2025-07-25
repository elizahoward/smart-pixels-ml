import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OptimizedDataGenerator4 as ODG
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Only import QKeras if available
try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False

def build_float_cnn_model(cluster_shape=(13, 21, 20), z_global_length=1, y_local_length=1):
    # Inputs
    cluster_input = tf.keras.layers.Input(shape=cluster_shape, name="cluster")
    z_global_input = tf.keras.layers.Input(shape=(z_global_length,), name="z_global")
    y_local_input = tf.keras.layers.Input(shape=(y_local_length,), name="y_local")

    # Take only the last timestamp (last axis)
    x = tf.keras.layers.Lambda(lambda x: x[..., -1], name="last_timestamp_slice")(cluster_input)  # shape (height, width)
    x = tf.keras.layers.Reshape((cluster_shape[0], cluster_shape[1], 1), name="add_channel")(x)
    # Conv2D branch
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2d")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2d_1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)

    # z_global branch
    z_dense = tf.keras.layers.Dense(32, activation="relu", name="dense_z")(z_global_input)
    # y_local branch
    y_dense = tf.keras.layers.Dense(32, activation="relu", name="dense_y")(y_local_input)

    # Concatenate all
    merged = tf.keras.layers.Concatenate(name="concatenate")([x, z_dense, y_dense])
    h = tf.keras.layers.Dense(96, activation="relu", name="dense_1")(merged)
    h = tf.keras.layers.Dropout(0.3, name="dropout")(h)
    h = tf.keras.layers.Dense(48, activation="relu", name="dense_2")(h)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_3")(h)

    model = Model([cluster_input, z_global_input, y_local_input], output, name="float_cnn_classifier")
    return model

def build_quantized_cnn_model(cluster_shape, z_global_length, y_local_length, weight_bits, activation_bits):
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")
    weight_quantizer = quantized_bits(weight_bits, 0, 1)
    activation_quantizer = quantized_relu(activation_bits, 0)
    cluster_input = tf.keras.layers.Input(shape=cluster_shape, name="cluster")
    z_global_input = tf.keras.layers.Input(shape=(z_global_length,), name="z_global")
    y_local_input = tf.keras.layers.Input(shape=(y_local_length,), name="y_local")
    cluster_last = tf.keras.layers.Lambda(lambda x: x[..., -1], name="last_timestamp_slice")(cluster_input)
    cluster_last = tf.keras.layers.Reshape((cluster_shape[0], cluster_shape[1], 1), name="add_channel")(cluster_last)
    x = QConv2D(8, (3, 3), padding="same", kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="qconv2d")(cluster_last)
    x = QActivation(activation_quantizer, name="conv_act")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2d")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    z_dense = QDense(8, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="z_dense")(z_global_input)
    z_act = QActivation(activation_quantizer, name="z_act")(z_dense)
    y_dense = QDense(8, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="y_dense")(y_local_input)
    y_act = QActivation(activation_quantizer, name="y_act")(y_dense)
    merged = tf.keras.layers.Concatenate(name="concat")([x, z_act, y_act])
    dense1 = QDense(32, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="dense1")(merged)
    act1 = QActivation(activation_quantizer, name="act1")(dense1)
    dropout = tf.keras.layers.Dropout(0.3, name="dropout")(act1)
    output = QDense(1, activation="sigmoid", kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="output")(dropout)
    model = Model([cluster_input, z_global_input, y_local_input], output, name=f"quantized_cnn_classifier_{weight_bits}_{activation_bits}")
    return model

def train_and_evaluate(model, train_gen, val_gen, epochs=50, results_dir=None, prefix=None):
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

    # Save accuracy curve if requested
    if results_dir is not None and prefix is not None:
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        acc_path = Path(results_dir) / f'{prefix}_accuracy_curve.png'
        plt.savefig(acc_path)
        plt.close()

        # ROC curve
        y_score = model.predict(val_gen, verbose=0).ravel()
        y_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        roc_path = Path(results_dir) / f'{prefix}_roc_curve.png'
        plt.savefig(roc_path)
        plt.close()

    return train_acc, val_acc

if __name__ == "__main__":
    base_dir = str(Path(__file__).resolve().parent.parent.parent / "filtering_models" / "filtering_records2048test")
    results_dir = str(Path(__file__).resolve().parent / 'results')
    print(f"Base directory: {base_dir}")
    print(f"Results directory: {results_dir}")
    train_dir = Path(base_dir) / "tfrecords_train"
    val_dir = Path(base_dir) / "tfrecords_validation"
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
    X_batch, y_batch = train_gen[0]
    cluster_shape = X_batch['cluster'].shape[1:]
    z_global_length = X_batch['z_global'].shape[1]
    y_local_length = X_batch['y_local'].shape[1]
    # Float model
    print("\n=== Training Float CNN ===")
    float_model = build_float_cnn_model(cluster_shape, z_global_length, y_local_length)
    float_train_acc, float_val_acc = train_and_evaluate(float_model, train_gen, val_gen, epochs=50, results_dir=results_dir, prefix="float_cnn_classifier")
    print(f"Float CNN - Train Acc: {float_train_acc:.4f}, Val Acc: {float_val_acc:.4f}")
    # Quantized models
    quant_results = []
    for bits in [2, 3, 4]:
        print(f"\n=== Training Quantized CNN ({bits} bits) ===")
        if not QKERAS_AVAILABLE:
            print("QKeras not available. Skipping quantized model.")
            quant_results.append((bits, None, None))
            continue
        quant_model = build_quantized_cnn_model(cluster_shape, z_global_length, y_local_length, bits, bits)
        q_train_acc, q_val_acc = train_and_evaluate(quant_model, train_gen, val_gen, epochs=50)
        print(f"Quantized CNN ({bits} bits) - Train Acc: {q_train_acc:.4f}, Val Acc: {q_val_acc:.4f}")
        quant_results.append((bits, q_train_acc, q_val_acc))
    # Print comparison table
    print("\n=== Accuracy Comparison Table ===")
    print(f"{'Model':<20} {'Train Acc':<12} {'Val Acc':<12}")
    print(f"{'Float CNN':<20} {float_train_acc:<12.4f} {float_val_acc:<12.4f}")
    for bits, q_train_acc, q_val_acc in quant_results:
        if q_train_acc is None:
            print(f"Quantized CNN ({bits}b){'':<7} {'N/A':<12} {'N/A':<12}")
        else:
            print(f"Quantized CNN ({bits}b){'':<7} {q_train_acc:<12.4f} {q_val_acc:<12.4f}") 