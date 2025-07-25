import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Dropout
from tensorflow.keras.models import Model

# QKeras imports
try:
    from qkeras import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False

def build_quantized_cnn_model(x_profile_length=21, z_global_length=1, y_profile_length=13, y_local_length=1, dropout_rate=0.2,
                              weight_quantizer=None, activation_quantizer=None):
    """
    Build a quantized CNN model for binary classification, mirroring the non-quantized architecture.
    """
    if not QKERAS_AVAILABLE:
        raise ImportError("QKeras is required for quantized models")
    if weight_quantizer is None:
        weight_quantizer = quantized_bits(8, 0, 1)
    if activation_quantizer is None:
        activation_quantizer = quantized_relu(8, 0)

    # Inputs
    x_profile_input = Input(shape=(x_profile_length,), name="x_profile")
    z_global_input = Input(shape=(z_global_length,), name="z_global")
    y_profile_input = Input(shape=(y_profile_length,), name="y_profile")
    y_local_input = Input(shape=(y_local_length,), name="y_local")

    # x_profile + z_global branch (single 128-unit QDense layer)
    xz_concat = Concatenate(name="xz_concat")([x_profile_input, z_global_input])
    xz_dense = QDense(128, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="xz_dense1")(xz_concat)
    xz_dense = QActivation(activation_quantizer, name="xz_act1")(xz_dense)

    # y_profile + y_local branch (single 128-unit QDense layer)
    yl_concat = Concatenate(name="yl_concat")([y_profile_input, y_local_input])
    yl_dense = QDense(128, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="yl_dense1")(yl_concat)
    yl_dense = QActivation(activation_quantizer, name="yl_act1")(yl_dense)

    # Concatenate both branches
    merged = Concatenate(name="merged_features")([xz_dense, yl_dense])
    merged_dense = QDense(64, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="merged_dense1")(merged)
    merged_dense = QActivation(activation_quantizer, name="merged_act1")(merged_dense)
    merged_dense = Dropout(dropout_rate, name="dropout1")(merged_dense)
    merged_dense = QDense(32, kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="merged_dense2")(merged_dense)
    merged_dense = QActivation(activation_quantizer, name="merged_act2")(merged_dense)

    # Output layer for binary classification
    output = QDense(1, activation="sigmoid", kernel_quantizer=weight_quantizer, bias_quantizer=weight_quantizer, name="output")(merged_dense)

    model = Model(inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], outputs=output, name="quantized_custom_cnn_model")
    return model

if __name__ == "__main__":
    if not QKERAS_AVAILABLE:
        print("QKeras not available. Please install with: pip install qkeras")
        exit(1)
    model = build_quantized_cnn_model()
    model.summary() 