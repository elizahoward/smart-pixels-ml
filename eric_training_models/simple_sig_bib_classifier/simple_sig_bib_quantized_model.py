import os
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import sys
from tensorflow.keras.metrics import Precision, Recall

# Import QKeras for quantization
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False

# Import the data generator from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OptimizedDataGenerator4 as ODG

def build_quantized_model(x_profile_length=21, z_global_length=1,
                         weight_quantizer=None,
                         activation_quantizer=None):
    """
    Build a quantized neural network for sig/bib classification using x_profile and z_global.
    Args:
        x_profile_length (int): Length of x profile (default 21)
        z_global_length (int): Length of z_global (default 1)
        weight_quantizer: QKeras quantizer for weights
        activation_quantizer: QKeras quantizer for activations
    Returns:
        tf.keras.Model: Compiled quantized model
    """
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")
    if weight_quantizer is None:
        weight_quantizer = quantized_bits(4, 0, 1)
    if activation_quantizer is None:
        activation_quantizer = quantized_relu(4, 0)

    # Input layers
    x_profile_input = tf.keras.layers.Input(shape=(x_profile_length,), name="x_profile")
    z_global_input = tf.keras.layers.Input(shape=(z_global_length,), name="z_global")

    # First quantized dense layer for each input
    x_dense1 = QDense(
        64,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="x_dense1"
    )(x_profile_input)
    x_act1 = QActivation(activation_quantizer, name="x_act1")(x_dense1)

    z_dense1 = QDense(
        64,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="z_dense1"
    )(z_global_input)
    z_act1 = QActivation(activation_quantizer, name="z_act1")(z_dense1)

    # Concatenate the features
    merged = tf.keras.layers.Concatenate(name="concat_inputs")([x_act1, z_act1])

    # Second quantized dense layer
    dense2 = QDense(
        32,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="dense2"
    )(merged)
    act2 = QActivation(activation_quantizer, name="act2")(dense2)

    # Dropout layer (no quantization needed)
    dropout = tf.keras.layers.Dropout(0.3, name="dropout")(act2)

    # Output layer for binary classification (sig vs bib)
    output = QDense(
        1,
        activation="sigmoid",
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="output"
    )(dropout)

    # Create model
    model = Model([x_profile_input, z_global_input], output, name="quantized_sig_bib_classifier")
    # Compile model with standard binary crossentropy (no weight regularization)
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )
    return model

def build_quantized_model_advanced(x_profile_length=21, z_global_length=1):
    """
    Build an advanced quantized model with different quantization schemes for different layers.
    Args:
        x_profile_length (int): Length of x profile (default 21)
        z_global_length (int): Length of z_global (default 1)
    Returns:
        tf.keras.Model: Compiled quantized model
    """
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")

    # Input layers
    x_profile_input = tf.keras.layers.Input(shape=(x_profile_length,), name="x_profile")
    z_global_input = tf.keras.layers.Input(shape=(z_global_length,), name="z_global")

    # First quantized dense layer for each input - using 8-bit quantization
    x_dense1 = QDense(
        64,
        kernel_quantizer=quantized_bits(8, 0, 1),
        bias_quantizer=quantized_bits(8, 0, 1),
        name="x_dense1"
    )(x_profile_input)
    x_act1 = QActivation(quantized_relu(8, 0), name="x_act1")(x_dense1)

    z_dense1 = QDense(
        64,
        kernel_quantizer=quantized_bits(8, 0, 1),
        bias_quantizer=quantized_bits(8, 0, 1),
        name="z_dense1"
    )(z_global_input)
    z_act1 = QActivation(quantized_relu(8, 0), name="z_act1")(z_dense1)

    # Concatenate the features
    merged = tf.keras.layers.Concatenate(name="concat_inputs")([x_act1, z_act1])

    # Second quantized dense layer - using 6t quantization for intermediate layer
    dense2 = QDense(
        32,
        kernel_quantizer=quantized_bits(6, 0, 1),
        bias_quantizer=quantized_bits(6, 0, 1),
        name="dense2"
    )(merged)
    act2 = QActivation(quantized_relu(6, 0), name="act2")(dense2)

    # Dropout layer (no quantization needed)
    dropout = tf.keras.layers.Dropout(0.3, name="dropout")(act2)

    # Output layer - using 8t quantization for final layer
    output = QDense(
        1,
        activation="sigmoid",
        kernel_quantizer=quantized_bits(8, 0, 1),
        bias_quantizer=quantized_bits(8, 0, 1),
        name="output"
    )(dropout)

    # Create model
    model = Model([x_profile_input, z_global_input], output, name="quantized_sig_bib_classifier_advanced")
    # Compile model with standard binary crossentropy (no weight regularization)
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )
    return model

def build_quantized_model_with_input_quantization(x_profile_length=21, z_global_length=1,
                                                weight_quantizer=None,
                                                activation_quantizer=None,
                                                input_quantizer=None):
    """
    Build a quantized neural network with quantized input layers.
    Args:
        x_profile_length (int): Length of x profile (default 21)
        z_global_length (int): Length of z_global (default 1)
        weight_quantizer: QKeras quantizer for weights
        activation_quantizer: QKeras quantizer for activations
        input_quantizer: QKeras quantizer for input layers
    Returns:
        tf.keras.Model: Compiled quantized model with quantized inputs
    """
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")
    if weight_quantizer is None:
        weight_quantizer = quantized_bits(4, 0, 1)
    if activation_quantizer is None:
        activation_quantizer = quantized_relu(4, 0)
    if input_quantizer is None:
        input_quantizer = quantized_relu(4, 0)

    # Input layers
    x_profile_input = tf.keras.layers.Input(shape=(x_profile_length,), name="x_profile")
    z_global_input = tf.keras.layers.Input(shape=(z_global_length,), name="z_global")

    # Quantize input layers
    x_profile_quantized = QActivation(input_quantizer, name="x_profile_quantized")(x_profile_input)
    z_global_quantized = QActivation(input_quantizer, name="z_global_quantized")(z_global_input)

    # First quantized dense layer for each input
    x_dense1 = QDense(
        64,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="x_dense1"
    )(x_profile_quantized)
    x_act1 = QActivation(activation_quantizer, name="x_act1")(x_dense1)

    z_dense1 = QDense(
        64,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="z_dense1"
    )(z_global_quantized)
    z_act1 = QActivation(activation_quantizer, name="z_act1")(z_dense1)

    # Concatenate the features
    merged = tf.keras.layers.Concatenate(name="concat_inputs")([x_act1, z_act1])

    # Second quantized dense layer
    dense2 = QDense(
        32,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="dense2"
    )(merged)
    act2 = QActivation(activation_quantizer, name="act2")(dense2)

    # Dropout layer (no quantization needed)
    dropout = tf.keras.layers.Dropout(0.3, name="dropout")(act2)

    # Output layer for binary classification (sig vs bib)
    output = QDense(
        1,
        activation="sigmoid",
        kernel_quantizer=weight_quantizer,
        bias_quantizer=weight_quantizer,
        name="output"
    )(dropout)

    # Create model
    model = Model([x_profile_input, z_global_input], output, name="quantized_sig_bib_classifier_with_input_quantization")
    # Compile model with standard binary crossentropy (no weight regularization)
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", Precision(), Recall()]
    )
    return model

def train_and_evaluate_quantized(model_name, base_dir, results_dir, epochs=50,
                                use_advanced_quantization=False, use_input_quantization=False):
    """
    Train and evaluate the quantized sig/bib classifier.
    Args:
        model_name (str): Name for the model
        base_dir (str): Directory containing tfrecords
        results_dir (str): Directory to save results
        epochs (int): Number of training epochs
        use_advanced_quantization (bool): Whether to use advanced quantization scheme
    Returns:
        dict: Training results
    """
    print(f"\n=== Training Quantized {model_name} ===")

    train_dir = Path(base_dir) / "tfrecords_train"
    val_dir = Path(base_dir) / "tfrecords_validation"

    # Data generators - load x_profile and z_global
    train_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(train_dir),
        x_feature_description=['x_profile', 'z_global']
    )

    val_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(val_dir),
        x_feature_description=['x_profile', 'z_global']
    )

    # Print shape of first batch for verification
    X_batch, y_batch = train_gen[0]
    print(f"First batch x_profile shape: {X_batch['x_profile'].shape}")
    print(f"First batch z_global shape: {X_batch['z_global'].shape}")
    print(f"First batch labels shape: {y_batch.shape}")

    # Build quantized model
    if use_input_quantization:
        model = build_quantized_model_with_input_quantization()
        print("Using quantization scheme with quantized input layers (4-bit throughout)")
    elif use_advanced_quantization:
        model = build_quantized_model_advanced()
        print("Using advanced quantization scheme (mixed bit-widths)")
    else:
        model = build_quantized_model()
        print("Using standard quantization scheme (4-bit throughout)")

    print(f"\nQuantized Model Summary:")
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluate final performance
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_precision = history.history['precision'][0]
    val_precision = history.history['val_precision'][-1]
    train_recall = history.history['recall'][0]
    val_recall = history.history['val_recall'][-1]

    print(f"\nFinal Training Metrics:")
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")

    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall:    {val_recall:.4f}")

    # Generate predictions for ROC curve
    y_score = model.predict(val_gen, verbose=0).ravel()
    y_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Save model
    os.makedirs(results_dir, exist_ok=True)
    model.save(os.path.join(results_dir, f'{model_name}_quantized_model.h5'))

    # Save quantization info
    quantization_info = {
        'model_type': 'quantized',
        'weight_quantizer': 'quantized_bits(4, 0, 1)' if not use_advanced_quantization else 'mixed',
        'activation_quantizer': 'quantized_relu(4, 0)' if not use_advanced_quantization else 'mixed',
        'input_quantization': use_input_quantization,
        'advanced_quantization': use_advanced_quantization
    }

    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_precision': train_precision,
        'val_precision': val_precision,
        'train_recall': train_recall,
        'val_recall': val_recall,
        'roc_auc': roc_auc,
        'quantization_info': quantization_info
    }

def compare_models(base_dir, results_dir, epochs=50):
    """
    Compare the performance of regular and quantized models.
    Args:
        base_dir (str): Directory containing tfrecords
        results_dir (str): Directory to save results
        epochs (int): Number of training epochs
    Returns:
        tuple: (regular_results, quantized_results)
    """
    print("=== Model Comparison: Regular vs Quantized ===")

    # Import the regular model function
    from simple_sig_bib_model import train_and_evaluate, build_simple_model

    # Train regular model
    print("\n1. Training Regular Model...")
    regular_results = train_and_evaluate(
        model_name="simple_sig_bib_classifier",
        base_dir=base_dir,
        results_dir=results_dir,
        epochs=epochs
    )

    # Train quantized model
    print("\n2. Training Quantized Model...")
    quantized_results = train_and_evaluate_quantized(
        model_name="quantized_sig_bib_classifier",
        base_dir=base_dir,
        results_dir=results_dir,
        epochs=epochs,
        use_advanced_quantization=False
    )

    # Compare results
    print("\n=== Comparison Results ===")
    print(f"{'Metric:':<20} {'Regular:':<12} {'Quantized':<12} {'Difference':<12}")
    print("-" * 56)
    print(f"{'Validation Accuracy':<20} {regular_results['val_acc']:<12.4f} {quantized_results['val_acc']:<12.4f} {quantized_results['val_acc'] - regular_results['val_acc']:<12.4f}")
    print(f"{'Validation Precision':<20} {regular_results['val_precision']:<12.4f} {quantized_results['val_precision']:<12.4f} {quantized_results['val_precision'] - regular_results['val_precision']:<12.4f}")
    print(f"{'Validation Recall':<20} {regular_results['val_recall']:<12.4f} {quantized_results['val_recall']:<12.4f} {quantized_results['val_recall'] - regular_results['val_recall']:<12.4f}")
    print(f"{'ROCAUC':<20} {regular_results['roc_auc']:<12.4f} {quantized_results['roc_auc']:<12.4f} {quantized_results['roc_auc'] - regular_results['roc_auc']:<12.4f}")
    return regular_results, quantized_results

if __name__ == "__main__":
    if not QKERAS_AVAILABLE:
        print("QKeras not available. Please install with: pip install qkeras")
        exit(1)

    # Configuration
    base_dir = str(Path(__file__).resolve().parent.parent.parent / "filtering_models" / "filtering_records2048test")
    results_dir = str(Path(__file__).resolve().parent / 'results')

    print(f"Base directory: {base_dir}")
    print(f"Results directory: {results_dir}")

    # Train and evaluate quantized model
    results = train_and_evaluate_quantized(
        model_name="quantized_sig_bib_classifier",
        base_dir=base_dir,
        results_dir=results_dir,
        epochs=75,
        use_advanced_quantization=False
    )

    print(f"\n=== Final Quantized Results ===")
    print(f"Training Accuracy: {results['train_acc']:.4f}")
    print(f"Validation Accuracy: {results['val_acc']:.4f}")
    print(f"Training Precision:  {results['train_precision']:.4f}")
    print(f"Validation Precision: {results['val_precision']:.4f}")
    print(f"Training Recall:  {results['train_recall']:.4f}")
    print(f"Validation Recall:   {results['val_recall']:.4f}")
    print(f"ROC AUC:          {results['roc_auc']:.4f}")
    print(f"Quantization Info: {results['quantization_info']}")
    print(f"\nResults saved in: {results_dir}") 