import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model

# QKeras imports
try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False

def build_quantized_cnn_model(cluster_shape=(13, 21, 20), z_global_length=1, y_local_length=1, 
                             weight_quantizer=None, activation_quantizer=None, dropout_rate=0.1):
    """
    Build a quantized CNN model that processes cluster data (last timestamp only) through Conv2D and MaxPooling,
    then concatenates with z_global and y_local for binary classification.
    
    Args:
        cluster_shape (tuple): Shape of cluster input (default (13, 21, 20))
        z_global_length (int): Length of z_global input (default 1)
        y_local_length (int): Length of y_local input (default 1)
        weight_quantizer: QKeras quantizer for weights
        activation_quantizer: QKeras quantizer for activations
        dropout_rate (float): Dropout rate for regularization
    Returns:
        tf.keras.Model: Uncompiled quantized model
    """
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")
    
    if weight_quantizer is None:
        weight_quantizer = quantized_bits(2, 0, 1)
    if activation_quantizer is None:
        activation_quantizer = quantized_relu(6, 0)

    # Inputs
    cluster_input = Input(shape=cluster_shape, name="cluster")
    z_global_input = Input(shape=(z_global_length,), name="z_global")
    y_local_input = Input(shape=(y_local_length,), name="y_local")

    # Extract last timestamp from cluster and reshape for Conv2D
    # cluster shape: (batch, 13, 21, 20) -> extract last timestamp -> (batch, 13, 21)
    cluster_last = Lambda(lambda x: x[..., -1], name="last_timestamp_slice")(cluster_input)
    # Add channel dimension for Conv2D: (batch, 13, 21) -> (batch, 13, 21, 1)
    cluster_reshaped = Reshape((cluster_shape[0], cluster_shape[1], 1), name="add_channel")(cluster_last)

    # Conv2D network for cluster processing with quantization
    x = QConv2D(48, (3, 3), padding='same', kernel_quantizer=weight_quantizer, 
                bias_quantizer=weight_quantizer, name="conv1")(cluster_reshaped)
    x = QActivation(activation_quantizer, name="conv1_act")(x)
    x = MaxPooling2D((2, 2), name="maxpool1")(x)
    x = Flatten(name="flatten")(x)

    # Process z_global and y_local with separate dense layers and quantization
    z_global_dense = QDense(16, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, 
                           name="z_global_dense")(z_global_input)
    z_global_act = QActivation(activation_quantizer, name="z_global_act")(z_global_dense)
    
    y_local_dense = QDense(16, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, 
                          name="y_local_dense")(y_local_input)
    y_local_act = QActivation(activation_quantizer, name="y_local_act")(y_local_dense)
    
    # Concatenate CNN features with processed z_global and y_local features directly
    merged = Concatenate(name="merged_features")([x, z_global_act, y_local_act])
    
    # Final dense layers with quantization
    merged_dense = QDense(64, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, 
                         name="merged_dense1")(merged)
    merged_dense = QActivation(activation_quantizer, name="merged_dense1_act")(merged_dense)
    merged_dense = Dropout(dropout_rate, name="dropout1")(merged_dense)
    
    merged_dense = QDense(64, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, 
                         name="merged_dense2")(merged_dense)
    merged_dense = QActivation(activation_quantizer, name="merged_dense2_act")(merged_dense)
    merged_dense = Dropout(dropout_rate, name="dropout2")(merged_dense)

    # Output layer for binary classification with quantization
    output = QDense(1, activation="sigmoid", kernel_quantizer=weight_quantizer, 
                   bias_quantizer=weight_quantizer, name="output")(merged_dense)

    model = Model(inputs=[cluster_input, z_global_input, y_local_input], outputs=output, name="quantized_cnn_model")
    return model

if __name__ == "__main__":
    if not QKERAS_AVAILABLE:
        print("QKeras not available. Please install with: pip install qkeras")
        exit(1)
    
    model = build_quantized_cnn_model()
    model.summary() 