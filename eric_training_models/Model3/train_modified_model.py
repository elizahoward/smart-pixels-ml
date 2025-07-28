import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import json
from datetime import datetime
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path for data generator import
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)
import OptimizedDataGenerator4 as ODG

class ModifiedModelTrainer:
    """
    Class to train the model from the screenshot with modifications:
    - 3x5 convolutional kernel
    - Head dense layers: 200 -> 100
    """
    
    def __init__(self):
        # Model parameters based on the screenshot
        self.conv_filters = 32  # From screenshot
        self.kernel_rows = 3    # Modified from 5x7 to 3x5
        self.kernel_cols = 5    # Modified from 5x7 to 3x5
        self.z_units = 16       # From screenshot
        self.y_units = 32       # From screenshot
        self.head_units = 200   # Modified from 224 to 200
        self.head_units_2 = 100 # Modified from 112 to 100
        self.dropout_rate = 0.0 # From screenshot
        
        # Learning rate parameters (using typical values)
        self.initial_lr = 0.000871145
        self.end_lr =  5.3e-05
        self.power = 2
        
    def build_modified_model(self):
        """
        Build the model with the specified modifications.
        """
        print("Building modified model...")
        print(f"  - Conv2D: {self.kernel_rows}x{self.kernel_cols} kernel, {self.conv_filters} filters")
        print(f"  - Head dense layers: {self.head_units} -> {self.head_units_2}")
        print(f"  - Z units: {self.z_units}, Y units: {self.y_units}")
        print(f"  - Dropout: {self.dropout_rate}")
        
        # Learning rate schedule
        decay_steps = 30 * 200  # Based on typical training setup
        lr_schedule = PolynomialDecay(
            initial_learning_rate=self.initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=self.end_lr,
            power=self.power
        )
        optimizer = Adam(learning_rate=lr_schedule)
        
        # Build model
        vol_input = Input(shape=(13, 21), name="cluster")
        z_input = Input(shape=(1,), name="z_global")
        y_input = Input(shape=(1,), name="y_local")
        
        # Conv2D branch with 3x5 kernel
        x = Reshape((13, 21, 1), name="add_channel")(vol_input)
        x = Conv2D(
            filters=self.conv_filters,
            kernel_size=(self.kernel_rows, self.kernel_cols),
            padding="same",
            activation="relu",
            name=f"conv2d_{self.kernel_rows}x{self.kernel_cols}"
        )(x)
        x = MaxPooling2D((2, 2), name="pool2d_1")(x)
        x = Flatten(name="flatten_vol")(x)
        
        # Scalar branches
        z_dense = Dense(self.z_units, activation="relu", name="dense_z")(z_input)
        y_dense = Dense(self.y_units, activation="relu", name="dense_y")(y_input)
        
        # Merge & head with modified units
        merged = Concatenate(name="concat_all")([x, z_dense, y_dense])
        h = Dense(self.head_units, activation="relu", name="head_dense1")(merged)
        h = Dropout(self.dropout_rate, name="head_dropout")(h)
        h = Dense(self.head_units_2, activation="relu", name="head_dense2")(h)
        
        output = Dense(1, activation="sigmoid", name="output")(h)
        
        model = Model(
            [vol_input, z_input, y_input],
            output,
            name="modified_model_3x5_200_100"
        )
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        print("Model built successfully!")
        model.summary()
        
        return model
    
    def setup_data_generators(self):
        """
        Setup data generators using the same structure as the original script.
        """
        print("Setting up data generators...")
        
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
        
        print(f"Training generator length: {len(train_gen)}")
        print(f"Validation generator length: {len(val_gen)}")
        
        return train_gen, val_gen
    
    def train_model(self, epochs=200, patience=15):
        """
        Train the modified model.
        """
        print(f"\nTraining modified model for {epochs} epochs...")
        
        # Build model
        model = self.build_modified_model()
        
        # Setup data generators
        train_gen, val_gen = self.setup_data_generators()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"modified_model_3x5_200_100_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # Callbacks
        checkpoint_path = output_dir / "best_model.h5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train the model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        history_file = output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(history.history, f, indent=2)
        
        # Save model parameters
        model_params = {
            'conv_filters': self.conv_filters,
            'kernel_rows': self.kernel_rows,
            'kernel_cols': self.kernel_cols,
            'z_units': self.z_units,
            'y_units': self.y_units,
            'head_units': self.head_units,
            'head_units_2': self.head_units_2,
            'dropout_rate': self.dropout_rate,
            'initial_lr': self.initial_lr,
            'end_lr': self.end_lr,
            'power': self.power
        }
        
        params_file = output_dir / "model_parameters.json"
        with open(params_file, 'w') as f:
            json.dump(model_params, f, indent=2)
        
        # Plot training history
        self.plot_training_history(history, output_dir)
        
        # Evaluate final model
        print("\nEvaluating final model...")
        test_loss, test_accuracy = model.evaluate(val_gen, verbose=0)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Save evaluation results
        eval_results = {
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'model_name': 'modified_model_3x5_200_100'
        }
        
        eval_file = output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\nTraining completed! Results saved to: {output_dir}")
        print(f"Best model saved to: {checkpoint_path}")
        
        return model, history, output_dir
    
    def plot_training_history(self, history, output_dir):
        """
        Plot training history and save to output directory.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_file = output_dir / "training_history.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {plot_file}")

def main():
    """
    Main function to train the modified model.
    """
    print("=" * 60)
    print("TRAIN MODIFIED MODEL (3x5 conv, 200->100 head)")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ModifiedModelTrainer()
    
    # Train the model
    model, history, output_dir = trainer.train_model(epochs=200, patience=15)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model: 3x5 conv kernel, 200->100 head dense layers")
    print(f"Results saved to: {output_dir}")
    
    return model, history, output_dir

if __name__ == "__main__":
    model, history, output_dir = main() 