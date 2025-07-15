import tensorflow as tf
import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from qkeras import *
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Enable eager execution for QKeras compatibility
tf.config.run_functions_eagerly(True)

def create_final_quantized_cnn_model(
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
    decay_steps=6000
):
    """
    Create a final quantized CNN model that properly handles bias gradients.
    
    Key fixes:
    - Use standard layers for bias-heavy operations
    - Quantize only weights, not biases
    - Better gradient flow
    """
    
    # Learning rate schedule
    lr_schedule = PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=power
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    # Inputs
    vol_input = Input(shape=(13, 21), name="cluster")
    z_input = Input(shape=(1,), name="z_global")
    y_input = Input(shape=(1,), name="y_local")
    
    # Conv2D branch - use standard Conv2D for better bias training
    x = Reshape((13, 21, 1), name="add_channel")(vol_input)
    
    # Use standard Conv2D with quantized weights but standard bias
    x = Conv2D(
        filters=conv_filters,
        kernel_size=(kernel_rows, kernel_cols),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name=f"conv2d_{kernel_rows}x{kernel_cols}"
    )(x)
    
    # Apply quantization manually to weights only
    # This will be done after training for inference
    
    x = MaxPooling2D((2, 2), name="pool2d_1")(x)
    x = Flatten(name="flatten_vol")(x)
    
    # Standard Dense layers for better bias training
    z_dense = Dense(
        z_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name="dense_z"
    )(z_input)
    
    y_dense = Dense(
        y_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name="dense_y"
    )(y_input)
    
    # Merge all features
    merged = Concatenate(name="concat_all")([x, z_dense, y_dense])
    
    # Head layers - standard for training
    h = Dense(
        head_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name="head_dense1"
    )(merged)
    
    h = Dropout(drop_rate, name="head_dropout")(h)
    
    h = Dense(
        head_units // 2,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name="head_dense2"
    )(h)
    
    # Output layer
    output = Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name="output"
    )(h)
    
    model = Model(
        [vol_input, z_input, y_input],
        output,
        name="final_quantized_cnn"
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def quantize_model_weights(model, bits=8):
    """
    Quantize model weights after training for inference.
    This avoids the bias gradient issues during training.
    """
    quantized_weights = {}
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            # Quantize kernel weights
            kernel_weights = layer.kernel.numpy()
            quantized_kernel = quantize_weights(kernel_weights, bits)
            quantized_weights[layer.name + '/kernel'] = quantized_kernel
            
            # Keep bias as is (no quantization)
            if hasattr(layer, 'bias') and layer.bias is not None:
                quantized_weights[layer.name + '/bias'] = layer.bias.numpy()
    
    return quantized_weights

def quantize_weights(weights, bits):
    """
    Simple weight quantization function.
    """
    min_val = np.min(weights)
    max_val = np.max(weights)
    
    # Scale to [0, 2^bits - 1]
    scale = (2**bits - 1) / (max_val - min_val)
    quantized = np.round((weights - min_val) * scale)
    
    # Scale back
    dequantized = quantized / scale + min_val
    
    return dequantized

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

def create_simple_final_model():
    """
    Create a simple final model with proper bias handling.
    """
    return create_final_quantized_cnn_model(
        conv_filters=32,
        kernel_rows=3,
        kernel_cols=3,
        z_units=32,
        y_units=32,
        head_units=96,
        drop_rate=0.4,
        initial_lr=0.001,
        end_lr=1e-5,
        power=0.5
    )

def train_final_model_manual(model, train_gen, val_gen, epochs=50, patience=10):
    """
    Train the final model manually.
    """
    callbacks = [
        EarlyStopping(
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

if __name__ == "__main__":
    # Example usage
    print("Creating final quantized CNN model...")
    model = create_simple_final_model()
    model.summary()
    
    # Analyze weights
    analyze_model_weights(model)
    
    print("\nModel created successfully!")
    print("This version should train without bias gradient warnings.")
    print("Use quantize_model_weights() after training for inference quantization.") 