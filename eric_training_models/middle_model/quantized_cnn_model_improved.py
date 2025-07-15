import tensorflow as tf
import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from qkeras import *
from tensorflow.keras.callbacks import EarlyStopping

# Enable eager execution for QKeras compatibility
tf.config.run_functions_eagerly(True)

def create_improved_quantized_cnn_model(
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
    Create an improved quantized CNN model with better gradient flow.
    
    Key improvements:
    - Mixed precision: 8-bit for Conv2D, 16-bit for Dense layers
    - Better activation functions
    - Improved regularization
    - Better learning rate schedule
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
    
    # Improved Conv2D branch with better quantization
    x = Reshape((13, 21, 1), name="add_channel")(vol_input)
    
    # Use 8-bit quantization for Conv2D (less aggressive than 4-bit)
    x = QConv2D(
        filters=conv_filters,
        kernel_size=(kernel_rows, kernel_cols),
        padding="same",
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001),  # Lighter regularization
        activity_regularizer=tf.keras.regularizers.L2(0.001),
        use_bias=True,
        name=f"qconv2d_{kernel_rows}x{kernel_cols}"
    )(x)
    
    # Use standard ReLU for better gradient flow
    x = Activation("relu")(x)
    
    # MaxPooling2D
    x = MaxPooling2D((2, 2), name="pool2d_1")(x)
    x = Flatten(name="flatten_vol")(x)
    
    # Improved scalar branches with 16-bit quantization
    # Z global branch
    z_dense = QDense(
        z_units,
        kernel_quantizer=quantized_bits(16, 0, alpha=1),  # 16-bit for better precision
        bias_quantizer=quantized_bits(16, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        activity_regularizer=tf.keras.regularizers.L2(0.001),
        use_bias=True,
        name="qdense_z"
    )(z_input)
    z_dense = Activation("relu")(z_dense)  # Standard ReLU
    
    # Y local branch
    y_dense = QDense(
        y_units,
        kernel_quantizer=quantized_bits(16, 0, alpha=1),  # 16-bit for better precision
        bias_quantizer=quantized_bits(16, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        activity_regularizer=tf.keras.regularizers.L2(0.001),
        use_bias=True,
        name="qdense_y"
    )(y_input)
    y_dense = Activation("relu")(y_dense)  # Standard ReLU
    
    # Merge all features
    merged = Concatenate(name="concat_all")([x, z_dense, y_dense])
    
    # Improved head layers with 16-bit quantization
    h = QDense(
        head_units,
        kernel_quantizer=quantized_bits(16, 0, alpha=1),  # 16-bit for better precision
        bias_quantizer=quantized_bits(16, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        activity_regularizer=tf.keras.regularizers.L2(0.001),
        use_bias=True,
        name="qhead_dense1"
    )(merged)
    h = Activation("relu")(h)  # Standard ReLU
    
    # Dropout
    h = Dropout(drop_rate, name="head_dropout")(h)
    
    # Second head layer
    h = QDense(
        head_units // 2,
        kernel_quantizer=quantized_bits(16, 0, alpha=1),  # 16-bit for better precision
        bias_quantizer=quantized_bits(16, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        activity_regularizer=tf.keras.regularizers.L2(0.001),
        use_bias=True,
        name="qhead_dense2"
    )(h)
    h = Activation("relu")(h)  # Standard ReLU
    
    # Output layer with 16-bit quantization
    output = QDense(
        1,
        activation="sigmoid",
        kernel_quantizer=quantized_bits(16, 0, alpha=1),  # 16-bit for better precision
        bias_quantizer=quantized_bits(16, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        use_bias=True,
        name="qoutput"
    )(h)
    
    model = Model(
        [vol_input, z_input, y_input],
        output,
        name="improved_quantized_cnn"
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def create_hybrid_quantized_model():
    """
    Create a hybrid model that uses quantization only for inference,
    training with full precision for better performance.
    """
    # Learning rate schedule
    lr_schedule = PolynomialDecay(
        initial_learning_rate=0.001,
        decay_steps=6000,
        end_learning_rate=1e-5,
        power=0.5
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    # Inputs
    vol_input = Input(shape=(13, 21), name="cluster")
    z_input = Input(shape=(1,), name="z_global")
    y_input = Input(shape=(1,), name="y_local")
    
    # Standard Conv2D for training (better gradients)
    x = Reshape((13, 21, 1), name="add_channel")(vol_input)
    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.L2(0.001),
        name="conv2d_3x3"
    )(x)
    
    x = MaxPooling2D((2, 2), name="pool2d_1")(x)
    x = Flatten(name="flatten_vol")(x)
    
    # Standard Dense layers for training
    z_dense = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001), name="dense_z")(z_input)
    y_dense = Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001), name="dense_y")(y_input)
    
    # Merge features
    merged = Concatenate(name="concat_all")([x, z_dense, y_dense])
    
    # Head layers
    h = Dense(96, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001), name="head_dense1")(merged)
    h = Dropout(0.4, name="head_dropout")(h)
    h = Dense(48, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001), name="head_dense2")(h)
    
    # Output
    output = Dense(1, activation="sigmoid", name="output")(h)
    
    model = Model(
        [vol_input, z_input, y_input],
        output,
        name="hybrid_cnn"
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def create_simple_quantized_model():
    """
    Create a simple quantized model with conservative quantization.
    """
    return create_improved_quantized_cnn_model(
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

def train_quantized_model_manual(model, train_gen, val_gen, epochs=50, patience=10):
    """
    Train a quantized model manually without keras-tuner.
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
    print("Creating improved quantized CNN model...")
    model = create_simple_quantized_model()
    model.summary()
    
    print("\nModel created successfully!")
    print("Use create_improved_quantized_cnn_model() for custom parameters")
    print("Use create_hybrid_quantized_model() for better training performance") 