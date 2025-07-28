import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model

def build_cnn_model(cluster_shape=(13, 21, 20), z_global_length=1, y_local_length=1, dropout_rate=0.1):
    """
    Build a CNN model that processes cluster data (last timestamp only) through Conv2D and MaxPooling,
    then concatenates with z_global and y_local for binary classification.
    
    Args:
        cluster_shape (tuple): Shape of cluster input (default (13, 21, 20))
        z_global_length (int): Length of z_global input (default 1)
        y_local_length (int): Length of y_local input (default 1)
        dropout_rate (float): Dropout rate for regularization
    Returns:
        tf.keras.Model: Uncompiled model
    """
    # Inputs
    cluster_input = Input(shape=cluster_shape, name="cluster")
    z_global_input = Input(shape=(z_global_length,), name="z_global")
    y_local_input = Input(shape=(y_local_length,), name="y_local")

    # Extract last timestamp from cluster and reshape for Conv2D
    # cluster shape: (batch, 13, 21, 20) -> extract last timestamp -> (batch, 13, 21)
    cluster_last = Lambda(lambda x: x[..., -1], name="last_timestamp_slice")(cluster_input)
    # Add channel dimension for Conv2D: (batch, 13, 21) -> (batch, 13, 21, 1)
    cluster_reshaped = Reshape((cluster_shape[0], cluster_shape[1], 1), name="add_channel")(cluster_last)

    # Conv2D network for cluster processing
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv1")(cluster_reshaped)
    x = MaxPooling2D((2, 2), name="maxpool1")(x)
    x = Flatten(name="flatten")(x)

    # Process z_global and y_local with separate dense layers
    z_global_dense = Dense(32, activation="relu", name="z_global_dense")(z_global_input)
    y_local_dense = Dense(32, activation="relu", name="y_local_dense")(y_local_input)
    
    # Concatenate CNN features with processed z_global and y_local features directly
    merged = Concatenate(name="merged_features")([x, z_global_dense, y_local_dense])
    
    # Final dense layers
    merged_dense = Dense(128, activation="relu", name="merged_dense1")(merged)
    merged_dense = Dropout(dropout_rate, name="dropout1")(merged_dense)
    merged_dense = Dense(64, activation="relu", name="merged_dense2")(merged_dense)
    merged_dense = Dropout(dropout_rate, name="dropout2")(merged_dense)

    # Output layer for binary classification
    output = Dense(1, activation="sigmoid", name="output")(merged_dense)

    model = Model(inputs=[cluster_input, z_global_input, y_local_input], outputs=output, name="cnn_model")
    return model

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary() 