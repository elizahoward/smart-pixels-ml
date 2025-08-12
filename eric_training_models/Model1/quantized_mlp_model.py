import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

# QKeras imports
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False

def build_quantized_mlp_model(weight_bits=8, weight_integer_bits=0, weight_alpha=1,
                              activation_bits=8, activation_integer_bits=0):
    """
    Build a quantized MLP model for binary classification following the Model1 architecture.
    
    Architecture:
    - Input: 4 features (z_global, x_size, y_size, y_local)
    - Hidden layers: 17 -> 20 -> 9 -> 16 -> 8
    - Output: 1 unit with smooth_sigmoid activation
    
    Args:
        weight_bits: total bits for quantized_bits
        weight_integer_bits: integer bits for quantized_bits  
        weight_alpha: alpha for quantized_bits
        activation_bits: total bits for quantized_relu
        activation_integer_bits: integer bits for quantized_relu
    """
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")
    
    weight_quantizer = quantized_bits(weight_bits, weight_integer_bits, weight_alpha)
    bias_quantizer = quantized_bits(weight_bits, weight_integer_bits, weight_alpha)
    activation_quantizer = quantized_relu(activation_bits, activation_integer_bits)
    
    # Define inputs
    input1 = Input(shape=(1,), name="z_global")
    input2 = Input(shape=(1,), name="x_size") 
    input3 = Input(shape=(1,), name="y_size")
    input4 = Input(shape=(1,), name="y_local")
    
    # Concatenate all inputs
    x = Concatenate()([input1, input2, input3, input4])
    
    # Layer 1
    x = QDense(
        17,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=bias_quantizer,
        name="dense1"
    )(x)
    x = QActivation(
        activation=activation_quantizer,
        name="q_relu1"
    )(x)
    
    # Layer 2
    x = QDense(
        20,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=bias_quantizer,
        name="dense2"
    )(x)
    x = QActivation(
        activation=activation_quantizer,
        name="q_relu2"
    )(x)
    
    # Layer 3
    x = QDense(
        9,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=bias_quantizer,
        name="dense3"
    )(x)
    x = QActivation(
        activation=activation_quantizer,
        name="q_relu3"
    )(x)
    
    # Layer 4
    x = QDense(
        16,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=bias_quantizer,
        name="dense4"
    )(x)
    x = QActivation(
        activation=activation_quantizer,
        name="q_relu4"
    )(x)
    
    # Layer 5
    x = QDense(
        8,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=bias_quantizer,
        name="dense5"
    )(x)
    x = QActivation(
        activation=activation_quantizer,
        name="q_relu5"
    )(x)
    
    # Output layer
    x = QDense(
        1,
        kernel_quantizer=weight_quantizer,
        bias_quantizer=bias_quantizer,
        name="output_dense"
    )(x)
    output = QActivation("smooth_sigmoid", name="output")(x)
    
    model = Model(inputs=[input1, input2, input3, input4], outputs=output, name="quantized_mlp_model")
    return model

def build_non_quantized_mlp_model():
    """
    Build a non-quantized version of the MLP model for comparison.
    """
    # Define inputs
    input1 = Input(shape=(1,), name="z_global")
    input2 = Input(shape=(1,), name="x_size")
    input3 = Input(shape=(1,), name="y_size") 
    input4 = Input(shape=(1,), name="y_local")
    
    # Concatenate all inputs
    x = Concatenate()([input1, input2, input3, input4])
    
    # Layer 1
    x = tf.keras.layers.Dense(17, activation="relu", name="dense1")(x)
    
    # Layer 2
    x = tf.keras.layers.Dense(20, activation="relu", name="dense2")(x)
    
    # Layer 3
    x = tf.keras.layers.Dense(9, activation="relu", name="dense3")(x)
    
    # Layer 4
    x = tf.keras.layers.Dense(16, activation="relu", name="dense4")(x)
    
    # Layer 5
    x = tf.keras.layers.Dense(8, activation="relu", name="dense5")(x)
    
    # Output layer
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    
    model = Model(inputs=[input1, input2, input3, input4], outputs=output, name="mlp_model")
    return model

if __name__ == "__main__":
    if not QKERAS_AVAILABLE:
        print("QKeras not available. Please install with: pip install qkeras")
        exit(1)
    
    print("Building quantized MLP model...")
    model = build_quantized_mlp_model()
    model.summary()
    
    print("\nBuilding non-quantized MLP model...")
    model_nq = build_non_quantized_mlp_model()
    model_nq.summary()