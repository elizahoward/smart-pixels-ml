import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

def build_custom_cnn_model(x_profile_length=21, z_global_length=1, y_profile_length=13, y_local_length=1, dropout_rate=0.0):
    """
    Build a CNN model that processes x_profile+z_global and y_profile+y_local in parallel, then concatenates and passes through a dense layer.
    Args:
        x_profile_length (int): Length of x_profile input (default 21)
        z_global_length (int): Length of z_global input (default 1)
        y_profile_length (int): Length of y_profile input (default 13)
        y_local_length (int): Length of y_local input (default 1)
        dropout_rate (float): Dropout rate for regularization
    Returns:
        tf.keras.Model: Uncompiled model
    """
    # Inputs
    x_profile_input = Input(shape=(x_profile_length,), name="x_profile")
    z_global_input = Input(shape=(z_global_length,), name="z_global")
    y_profile_input = Input(shape=(y_profile_length,), name="y_profile")
    y_local_input = Input(shape=(y_local_length,), name="y_local")

    # x_profile + z_global branch (single 32-unit layer)
    xz_concat = Concatenate(name="xz_concat")([x_profile_input, z_global_input])
    xz_dense = Dense(32, activation="relu", name="xz_dense1")(xz_concat)

    # y_profile + y_local branch (single 32-unit layer)
    yl_concat = Concatenate(name="yl_concat")([y_profile_input, y_local_input])
    yl_dense = Dense(32, activation="relu", name="yl_dense1")(yl_concat)

    # Concatenate both branches
    merged = Concatenate(name="merged_features")([xz_dense, yl_dense])
    merged_dense = Dense(128, activation="relu", name="merged_dense1")(merged)
    merged_dense = Dropout(dropout_rate, name="dropout1")(merged_dense)
    merged_dense = Dense(64, activation="relu", name="merged_dense2")(merged_dense)
    merged_dense = Dense(32, activation="relu", name="merged_dense3")(merged_dense)

    # Output layer for binary classification
    output = Dense(1, activation="sigmoid", name="output")(merged_dense)

    model = Model(inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], outputs=output, name="custom_cnn_model")
    return model

if __name__ == "__main__":
    model = build_custom_cnn_model()
    model.summary() 