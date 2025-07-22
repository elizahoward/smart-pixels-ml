import os
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import sys

# Import the data generator from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OptimizedDataGenerator4 as ODG

def build_simple_model(x_profile_length=21, z_global_length=1):
    """
    Build a simple neural network for sig/bib classification using x_profile and z_global.
    
    Args:
        x_profile_length (int): Length of x profile (default 21)
        z_global_length (int): Length of z_global (default 1)
    
    Returns:
        tf.keras.Model: Compiled model
    """
    # Input layers
    x_profile_input = Input(shape=(x_profile_length,), name="x_profile")
    z_global_input = Input(shape=(z_global_length,), name="z_global")
    
    # First dense layer for each input
    x_dense1 = Dense(64, activation="relu", name="x_dense1")(x_profile_input)
    z_dense1 = Dense(64, activation="relu", name="z_dense1")(z_global_input)
    
    # Concatenate the features
    merged = Concatenate(name="concat_inputs")([x_dense1, z_dense1])
    
    # Second dense layer
    dense2 = Dense(32, activation="relu", name="dense2")(merged)
    dropout = Dropout(0.3, name="dropout")(dense2)
    
    # Output layer for binary classification (sig vs bib)
    output = Dense(1, activation="sigmoid", name="output")(dropout)
    
    # Create model
    model = Model([x_profile_input, z_global_input], output, name="simple_sig_bib_classifier")
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "precision", "recall"]
    )
    
    return model

def train_and_evaluate(model_name, base_dir, results_dir, epochs=50):
    """
    Train and evaluate the simple sig/bib classifier.
    
    Args:
        model_name (str): Name for the model
        base_dir (str): Directory containing tfrecords
        results_dir (str): Directory to save results
        epochs (int): Number of training epochs
    
    Returns:
        dict: Training results
    """
    print(f"\n=== Training {model_name} ===")
    
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
    print(f"First batch 'x_profile' shape: {X_batch['x_profile'].shape}")
    print(f"First batch 'z_global' shape: {X_batch['z_global'].shape}")
    print(f"First batch labels shape: {y_batch.shape}")
    
    # Build model
    model = build_simple_model()
    print(f"\nModel Summary:")
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
    train_precision = history.history['precision'][-1]
    val_precision = history.history['val_precision'][-1]
    train_recall = history.history['recall'][-1]
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
    model.save(os.path.join(results_dir, f'{model_name}_model.h5'))
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_precision': train_precision,
        'val_precision': val_precision,
        'train_recall': train_recall,
        'val_recall': val_recall,
        'roc_auc': roc_auc
    }

if __name__ == '__main__':  # Configuration
    base_dir = str(Path(__file__).resolve().parent.parent.parent / "filtering_models" / "filtering_records2048test")
    results_dir = str(Path(__file__).resolve().parent / 'results')
    
    print(f"Base directory: {base_dir}")
    print(f"Results directory: {results_dir}")
    
    # Train and evaluate
    results = train_and_evaluate(
        model_name="simple_sig_bib_classifier",
        base_dir=base_dir,
        results_dir=results_dir,
        epochs=75
    )
    
    print(f"\n=== Final Results ===")
    print(f"Training Accuracy: {results['train_acc']:.4f}")
    print(f"Validation Accuracy: {results['val_acc']:.4f}")
    print(f"Training Precision:  {results['train_precision']:.4f}")
    print(f"Validation Precision: {results['val_precision']:.4f}")
    print(f"Training Recall:  {results['train_recall']:.4f}")
    print(f"Validation Recall:   {results['val_recall']:.4f}")
    print(f"ROC AUC:           {results['roc_auc']:0.4f}")
    print(f"\nResults saved in: {results_dir}") 