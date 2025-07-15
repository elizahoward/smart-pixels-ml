#!/usr/bin/env python3
"""
Consolidated Quantized CNN Model
This file combines the best features from all three model versions:
- First model: QKeras quantization during training
- Middle model: Improved quantization with better gradient flow
- Final model: Post-training quantization for better bias handling

Key features:
- Multiple quantization strategies
- Model saving/loading capabilities
- Comprehensive weight analysis
- Training utilities
"""

import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

# Enable eager execution for QKeras compatibility
tf.config.run_functions_eagerly(True)

# Add parent directory to path for imports
parentdir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.insert(0, parentdir)

import OptimizedDataGenerator4 as ODG
from qkeras import *

def create_consolidated_quantized_model(
    conv_filters=32,
    kernel_rows=3,
    kernel_cols=3,
    z_units=32,
    y_units=32,
    head_units=96,
    drop_rate=0.4,
    initial_lr=0.001,
    end_lr=1e-5,
    power=0.5,
    decay_steps=6000,
    quantization_strategy="post_training"
):
    """
    Create a consolidated quantized CNN model with multiple quantization strategies.
    
    Args:
        quantization_strategy: "qkeras_training", "improved_qkeras", or "post_training"
    """
    
    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=power
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Inputs
    vol_input = tf.keras.layers.Input(shape=(13, 21), name="cluster")
    z_input = tf.keras.layers.Input(shape=(1,), name="z_global")
    y_input = tf.keras.layers.Input(shape=(1,), name="y_local")
    
    # Conv2D branch
    x = tf.keras.layers.Reshape((13, 21, 1), name="add_channel")(vol_input)
    
    if quantization_strategy == "qkeras_training":
        # First model approach: QKeras quantization during training
        x = QConv2D(
            filters=conv_filters,
            kernel_size=(kernel_rows, kernel_cols),
            padding="same",
            kernel_quantizer=quantized_bits(4, 0, alpha=1),
            bias_quantizer=quantized_bits(4, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
            name=f"qconv2d_{kernel_rows}x{kernel_cols}"
        )(x)
        x = QActivation("quantized_relu(4, 0, 1)")(x)
        
    elif quantization_strategy == "improved_qkeras":
        # Middle model approach: Improved quantization
        x = QConv2D(
            filters=conv_filters,
            kernel_size=(kernel_rows, kernel_cols),
            padding="same",
            kernel_quantizer=quantized_bits(8, 0, alpha=1),
            bias_quantizer=quantized_bits(8, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            activity_regularizer=tf.keras.regularizers.L2(0.001),
            use_bias=True,
            name=f"qconv2d_{kernel_rows}x{kernel_cols}"
        )(x)
        x = tf.keras.layers.Activation("relu")(x)
        
    else:  # post_training
        # Final model approach: Standard layers for training, quantize later
        x = tf.keras.layers.Conv2D(
            filters=conv_filters,
            kernel_size=(kernel_rows, kernel_cols),
            padding="same",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name=f"conv2d_{kernel_rows}x{kernel_cols}"
        )(x)
    
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2d_1")(x)
    x = tf.keras.layers.Flatten(name="flatten_vol")(x)
    
    # Scalar branches
    if quantization_strategy == "qkeras_training":
        # First model approach
        z_dense = QDense(
            z_units,
            kernel_quantizer=quantized_bits(8, 0, alpha=1),
            bias_quantizer=quantized_bits(8, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
            name="qdense_z"
        )(z_input)
        z_dense = QActivation("quantized_relu(8, 0, 1)")(z_dense)
        
        y_dense = QDense(
            y_units,
            kernel_quantizer=quantized_bits(8, 0, alpha=1),
            bias_quantizer=quantized_bits(8, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
            name="qdense_y"
        )(y_input)
        y_dense = QActivation("quantized_relu(8, 0, 1)")(y_dense)
        
    elif quantization_strategy == "improved_qkeras":
        # Middle model approach
        z_dense = QDense(
            z_units,
            kernel_quantizer=quantized_bits(16, 0, alpha=1),
            bias_quantizer=quantized_bits(16, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            activity_regularizer=tf.keras.regularizers.L2(0.001),
            use_bias=True,
            name="qdense_z"
        )(z_input)
        z_dense = tf.keras.layers.Activation("relu")(z_dense)
        
        y_dense = QDense(
            y_units,
            kernel_quantizer=quantized_bits(16, 0, alpha=1),
            bias_quantizer=quantized_bits(16, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            activity_regularizer=tf.keras.regularizers.L2(0.001),
            use_bias=True,
            name="qdense_y"
        )(y_input)
        y_dense = tf.keras.layers.Activation("relu")(y_dense)
        
    else:  # post_training
        # Final model approach
        z_dense = tf.keras.layers.Dense(
            z_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="dense_z"
        )(z_input)
        
        y_dense = tf.keras.layers.Dense(
            y_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="dense_y"
        )(y_input)
    
    # Merge all features
    merged = tf.keras.layers.Concatenate(name="concat_all")([x, z_dense, y_dense])
    
    # Head layers
    if quantization_strategy == "qkeras_training":
        # First model approach
        h = QDense(
            head_units,
            kernel_quantizer=quantized_bits(8, 0, alpha=1),
            bias_quantizer=quantized_bits(8, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
            name="qhead_dense1"
        )(merged)
        h = QActivation("quantized_relu(8, 0, 1)")(h)
        
    elif quantization_strategy == "improved_qkeras":
        # Middle model approach
        h = QDense(
            head_units,
            kernel_quantizer=quantized_bits(16, 0, alpha=1),
            bias_quantizer=quantized_bits(16, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            activity_regularizer=tf.keras.regularizers.L2(0.001),
            use_bias=True,
            name="qhead_dense1"
        )(merged)
        h = tf.keras.layers.Activation("relu")(h)
        
    else:  # post_training
        # Final model approach
        h = tf.keras.layers.Dense(
            head_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="head_dense1"
        )(merged)
    
    h = tf.keras.layers.Dropout(drop_rate, name="head_dropout")(h)
    
    # Second head layer
    if quantization_strategy == "qkeras_training":
        h = QDense(
            head_units // 2,
            kernel_quantizer=quantized_bits(8, 0, alpha=1),
            bias_quantizer=quantized_bits(8, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            activity_regularizer=tf.keras.regularizers.L2(0.01),
            name="qhead_dense2"
        )(h)
        h = QActivation("quantized_relu(8, 0, 1)")(h)
        
    elif quantization_strategy == "improved_qkeras":
        h = QDense(
            head_units // 2,
            kernel_quantizer=quantized_bits(16, 0, alpha=1),
            bias_quantizer=quantized_bits(16, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            activity_regularizer=tf.keras.regularizers.L2(0.001),
            use_bias=True,
            name="qhead_dense2"
        )(h)
        h = tf.keras.layers.Activation("relu")(h)
        
    else:  # post_training
        h = tf.keras.layers.Dense(
            head_units // 2,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="head_dense2"
        )(h)
    
    # Output layer
    if quantization_strategy == "qkeras_training":
        output = QDense(
            1,
            activation="sigmoid",
            kernel_quantizer=quantized_bits(8, 0, alpha=1),
            bias_quantizer=quantized_bits(8, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            name="qoutput"
        )(h)
        
    elif quantization_strategy == "improved_qkeras":
        output = QDense(
            1,
            activation="sigmoid",
            kernel_quantizer=quantized_bits(16, 0, alpha=1),
            bias_quantizer=quantized_bits(16, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            use_bias=True,
            name="qoutput"
        )(h)
        
    else:  # post_training
        output = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.L2(0.001),
            name="output"
        )(h)
    
    model = tf.keras.models.Model(
        [vol_input, z_input, y_input],
        output,
        name=f"consolidated_quantized_cnn_{quantization_strategy}"
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def quantize_weights_and_biases(weights, bits):
    """
    Enhanced quantization function that handles both weights and biases.
    """
    min_val = np.min(weights)
    max_val = np.max(weights)
    
    # Handle edge case where all weights are the same
    if max_val == min_val:
        return weights
    
    # Scale to [0, 2^bits - 1]
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((weights - min_val) * scale)
    
    # Scale back
    dequantized = quantized / scale + min_val
    
    return dequantized

def quantize_model_weights_and_biases(model, bits=8):
    """
    Quantize both model weights AND biases after training for inference.
    """
    quantized_weights = {}
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            # Quantize kernel weights
            kernel_weights = layer.kernel.numpy()
            quantized_kernel = quantize_weights_and_biases(kernel_weights, bits)
            quantized_weights[layer.name + '/kernel'] = quantized_kernel
            
            # Quantize bias as well
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_weights = layer.bias.numpy()
                quantized_bias = quantize_weights_and_biases(bias_weights, bits)
                quantized_weights[layer.name + '/bias'] = quantized_bias

    return quantized_weights

def analyze_model_weights(model):
    """
    Analyze and display model weights information.
    """
    print("\nModel Weights Analysis")
    print("=" * 50)
    
    total_params = 0
    trainable_params = 0
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            kernel_shape = layer.kernel.shape
            kernel_params = np.prod(kernel_shape)
            total_params += kernel_params
            
            if layer.trainable:
                trainable_params += kernel_params
            
            print(f"{layer.name:20} Kernel: {kernel_shape} ({kernel_params:,} params)")
            
            # Show weight statistics
            weights = layer.kernel.numpy()
            print(f"{'':20} Min: {weights.min():.6f}, Max: {weights.max():.6f}, Mean: {weights.mean():.6f}")
            
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_shape = layer.bias.shape
                bias_params = np.prod(bias_shape)
                total_params += bias_params
                
                if layer.trainable:
                    trainable_params += bias_params
                
                print(f"{'':20} Bias: {bias_shape} ({bias_params:,} params)")
                
                bias_weights = layer.bias.numpy()
                print(f"{'':20} Bias Min: {bias_weights.min():.6f}, Max: {bias_weights.max():.6f}, Mean: {bias_weights.mean():.6f}")
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def create_simple_consolidated_model():
    """
    Create a simple consolidated model with post-training quantization strategy.
    """
    return create_consolidated_quantized_model(
        conv_filters=32,
        kernel_rows=3,
        kernel_cols=3,
        z_units=32,
        y_units=32,
        head_units=96,
        drop_rate=0.4,
        quantization_strategy="post_training"
    )

def train_consolidated_model_manual(model, train_gen, val_gen, epochs=50, patience=10):
    """
    Train the consolidated model manually.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            restore_best_weights=True
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def save_trained_model(model, save_path):
    """
    Save a trained model to disk.
    """
    # Ensure the save path has a proper extension
    if not save_path.endswith('.keras'):
        save_path = save_path + '.keras'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")

def load_trained_model(load_path):
    """
    Load a trained model from disk.
    """
    # Try different possible extensions
    possible_paths = [
        load_path,
        load_path + '.keras',
        load_path + '.h5'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            print(f"Model loaded from: {path}")
            return model
    
    print(f"Model file not found. Tried: {possible_paths}")
    return None

def setup_data_generators():
    """Setup data generators."""
    BASE_DIR = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
    TRAIN_DIR = BASE_DIR / "tfrecords_train"
    VALIDATION_DIR = BASE_DIR / "tfrecords_validation"

    train_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(TRAIN_DIR),
        x_feature_description=['cluster', 'y_local', 'z_global']
    )

    val_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(VALIDATION_DIR),
        x_feature_description=['cluster', 'y_local', 'z_global']
    )
    
    return train_gen, val_gen

def train_and_save_model():
    """
    Train a model and save it for later use.
    """
    print("=" * 60)
    print("TRAINING AND SAVING CONSOLIDATED MODEL")
    print("=" * 60)
    
    # Setup data generators
    train_gen, val_gen = setup_data_generators()
    
    # Create model
    model = create_simple_consolidated_model()
    
    print("Model summary:")
    model.summary()
    
    # Analyze initial weights
    print("\nInitial weights analysis:")
    analyze_model_weights(model)
    
    # Train the model
    print("\nTraining model...")
    history = train_consolidated_model_manual(
        model, 
        train_gen, 
        val_gen, 
        epochs=120, 
        patience=50
    )
    
    # Evaluate trained model
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    train_loss, train_acc = model.evaluate(train_gen, verbose=0)
    
    print(f"\nTRAINED MODEL RESULTS:")
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Training loss: {train_loss:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Validation loss: {val_loss:.4f}")
    
    # Analyze final weights
    print("\nFinal weights analysis:")
    analyze_model_weights(model)
    
    # Save the model
    save_path = "smart_pixels_ml/eric_training_models/CNNquantized/saved_trained_model"
    save_trained_model(model, save_path)
    
    print(f"\n✅ Model trained and saved successfully!")
    print(f"Model can be loaded using: load_trained_model('{save_path}')")
    
    return model, train_gen, val_gen, history

if __name__ == "__main__":
    # Train and save a model
    train_and_save_model() 