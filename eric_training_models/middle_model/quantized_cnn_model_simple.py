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

def create_quantized_cnn_model(
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
    Create a quantized CNN model similar to the one in testinghyperCNN.ipynb
    but with QKeras quantization layers.
    
    Args:
        conv_filters: Number of filters in Conv2D layer
        kernel_rows: Kernel height
        kernel_cols: Kernel width
        z_units: Units in z_global dense layer
        y_units: Units in y_local dense layer
        head_units: Units in head dense layers
        drop_rate: Dropout rate
        initial_lr: Initial learning rate
        end_lr: End learning rate
        power: Polynomial decay power
        decay_steps: Number of decay steps
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
    
    # Quantized Conv2D branch with rectangular kernel
    x = Reshape((13, 21, 1), name="add_channel")(vol_input)
    
    # Quantized Conv2D layer
    x = QConv2D(
        filters=conv_filters,
        kernel_size=(kernel_rows, kernel_cols),
        padding="same",
        kernel_quantizer=quantized_bits(4, 0, alpha=1),
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        use_bias=True,  # Explicitly enable bias
        name=f"qconv2d_{kernel_rows}x{kernel_cols}"
    )(x)
    
    # Quantized activation
    x = QActivation("quantized_relu(4, 0, 1)")(x)
    
    # MaxPooling2D (no quantization needed for pooling)
    x = MaxPooling2D((2, 2), name="pool2d_1")(x)
    x = Flatten(name="flatten_vol")(x)
    
    # Quantized scalar branches
    # Z global branch
    z_dense = QDense(
        z_units,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        use_bias=True,  # Explicitly enable bias
        name="qdense_z"
    )(z_input)
    z_dense = QActivation("quantized_relu(8, 0, 1)")(z_dense)
    
    # Y local branch
    y_dense = QDense(
        y_units,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        use_bias=True,  # Explicitly enable bias
        name="qdense_y"
    )(y_input)
    y_dense = QActivation("quantized_relu(8, 0, 1)")(y_dense)
    
    # Merge all features
    merged = Concatenate(name="concat_all")([x, z_dense, y_dense])
    
    # Quantized head layers
    h = QDense(
        head_units,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        use_bias=True,  # Explicitly enable bias
        name="qhead_dense1"
    )(merged)
    h = QActivation("quantized_relu(8, 0, 1)")(h)
    
    # Dropout (no quantization needed)
    h = Dropout(drop_rate, name="head_dropout")(h)
    
    # Second head layer
    h = QDense(
        head_units // 2,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        use_bias=True,  # Explicitly enable bias
        name="qhead_dense2"
    )(h)
    h = QActivation("quantized_relu(8, 0, 1)")(h)
    
    # Output layer (quantized)
    output = QDense(
        1,
        activation="sigmoid",
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        use_bias=True,  # Explicitly enable bias
        name="qoutput"
    )(h)
    
    model = Model(
        [vol_input, z_input, y_input],
        output,
        name="quantized_rect_kernel_conv2d"
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def create_simple_quantized_model():
    """
    Create a simple quantized model with fixed hyperparameters
    for quick testing and comparison.
    """
    return create_quantized_cnn_model(
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

def create_quantized_model_variants():
    """
    Create different variants of the quantized model for manual hyperparameter testing.
    Returns a dictionary of models with different configurations.
    """
    variants = {}
    
    # Variant 1: Small model
    variants['small'] = create_quantized_cnn_model(
        conv_filters=16,
        kernel_rows=3,
        kernel_cols=3,
        z_units=16,
        y_units=16,
        head_units=64,
        drop_rate=0.3,
        initial_lr=0.001,
        end_lr=1e-5,
        power=0.5
    )
    
    # Variant 2: Large model
    variants['large'] = create_quantized_cnn_model(
        conv_filters=64,
        kernel_rows=5,
        kernel_cols=5,
        z_units=48,
        y_units=48,
        head_units=128,
        drop_rate=0.5,
        initial_lr=0.0005,
        end_lr=1e-6,
        power=1.0
    )
    
    # Variant 3: Medium model (default)
    variants['medium'] = create_simple_quantized_model()
    
    return variants

def train_quantized_model_manual(model, train_gen, val_gen, epochs=50, patience=10):
    """
    Train a quantized model manually without keras-tuner.
    
    Args:
        model: The quantized model to train
        train_gen: Training data generator
        val_gen: Validation data generator
        epochs: Number of training epochs
        patience: Early stopping patience
    
    Returns:
        history: Training history
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
    print("Creating quantized CNN model...")
    model = create_simple_quantized_model()
    model.summary()
    
    print("\nModel created successfully!")
    print("Use create_quantized_cnn_model() for custom parameters")
    print("Use create_quantized_model_variants() for different configurations")
    print("Use train_quantized_model_manual() for training without keras-tuner") 