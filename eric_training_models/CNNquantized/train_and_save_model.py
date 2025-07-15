#!/usr/bin/env python3
"""
Script to train and save a model for later use.
This avoids retraining every time you want to test quantization.
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
from consolidated_quantized_model import (
    create_simple_consolidated_model,
    train_consolidated_model_manual,
    save_trained_model,
    analyze_model_weights
)

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
    print("TRAINING AND SAVING MODEL FOR LATER USE")
    print("=" * 60)
    
    # Setup data generators
    print("Setting up data generators...")
    train_gen, val_gen = setup_data_generators()
    
    # Create model
    print("Creating model...")
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
    print("\nEvaluating trained model...")
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
    save_path = "saved_trained_model"
    save_trained_model(model, save_path)
    
    print(f"\n✅ Model trained and saved successfully!")
    print(f"Model can be loaded using: load_trained_model('{save_path}')")
    print(f"Model file location: {save_path}.keras")
    print(f"\nYou can now modify train_and_test_quantized_with_bias.py to load this model instead of training a new one.")
    
    return model, train_gen, val_gen, history

if __name__ == "__main__":
    # Train and save a model
    train_and_save_model() 