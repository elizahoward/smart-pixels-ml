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
from sklearn.metrics import roc_curve, auc

# QKeras imports
try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
except ImportError:
    print("QKeras not available. Please install with: pip install qkeras")
    QKERAS_AVAILABLE = False

# Add parent directory to path for data generator import
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)
import OptimizedDataGenerator4 as ODG

class QuantizedModifiedModelTrainer:
    """
    Class to train quantized versions of the modified model (3x5 conv, 200->100 head)
    with different bit-widths for weights and biases.
    """
    
    def __init__(self):
        # Model parameters based on the original modified model
        self.conv_filters = 32
        self.kernel_rows = 3
        self.kernel_cols = 5
        self.z_units = 16
        self.y_units = 32
        self.head_units = 200
        self.head_units_2 = 100
        self.dropout_rate = 0.0
        
        # Learning rate parameters (from the best hyperparameters)
        self.initial_lr = 0.000871145
        self.end_lr = 5.3e-05
        self.power = 2
        
        # Quantization settings
        self.weight_bits = [2, 3, 6]  # Different bit-widths to test
        self.activation_bits = 8  # Fixed 8-bit activation quantization
        
    def build_quantized_model(self, weight_bits):
        """
        Build the quantized model with specified bit-width for weights.
        """
        print(f"Building quantized model with {weight_bits}-bit weights and {self.activation_bits}-bit activations...")
        
        # Define quantizers
        weight_quantizer = quantized_bits(weight_bits, 0, 1)
        activation_quantizer = quantized_relu(self.activation_bits, 0)
        
        # Learning rate schedule
        decay_steps = 30 * 200
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
        
        # Conv2D branch with quantization
        x = Reshape((13, 21, 1), name="add_channel")(vol_input)
        x = QConv2D(
            filters=self.conv_filters,
            kernel_size=(self.kernel_rows, self.kernel_cols),
            padding="same",
            kernel_quantizer=weight_quantizer,
            bias_quantizer=weight_quantizer,
            name=f"conv2d_{self.kernel_rows}x{self.kernel_cols}"
        )(x)
        x = QActivation(activation_quantizer, name="conv2d_act")(x)
        x = MaxPooling2D((2, 2), name="pool2d_1")(x)
        x = Flatten(name="flatten_vol")(x)
        
        # Scalar branches with quantization
        z_dense = QDense(
            self.z_units, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=weight_quantizer, 
            name="dense_z"
        )(z_input)
        z_dense = QActivation(activation_quantizer, name="dense_z_act")(z_dense)
        
        y_dense = QDense(
            self.y_units, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=weight_quantizer, 
            name="dense_y"
        )(y_input)
        y_dense = QActivation(activation_quantizer, name="dense_y_act")(y_dense)
        
        # Merge & head with quantization
        merged = Concatenate(name="concat_all")([x, z_dense, y_dense])
        
        h = QDense(
            self.head_units, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=weight_quantizer, 
            name="head_dense1"
        )(merged)
        h = QActivation(activation_quantizer, name="head_dense1_act")(h)
        h = Dropout(self.dropout_rate, name="head_dropout")(h)
        
        h = QDense(
            self.head_units_2, 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=weight_quantizer, 
            name="head_dense2"
        )(h)
        h = QActivation(activation_quantizer, name="head_dense2_act")(h)
        
        output = QDense(
            1, 
            activation="sigmoid", 
            kernel_quantizer=weight_quantizer, 
            bias_quantizer=weight_quantizer, 
            name="output"
        )(h)
        
        model = Model(
            [vol_input, z_input, y_input],
            output,
            name=f"quantized_model_{weight_bits}bit"
        )
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        print(f"Quantized model built successfully!")
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
    
    def train_quantized_model(self, weight_bits, epochs=200, patience=15):
        """
        Train a quantized model with specified bit-width.
        """
        print(f"\nTraining {weight_bits}-bit quantized model for {epochs} epochs...")
        
        # Build model
        model = self.build_quantized_model(weight_bits)
        
        # Setup data generators
        train_gen, val_gen = self.setup_data_generators()
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"quantized_model_{weight_bits}bit_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # Callbacks
        checkpoint_path = output_dir / f"best_model_{weight_bits}bit.h5"
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
            'weight_bits': weight_bits,
            'activation_bits': self.activation_bits,
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
        self.plot_training_history(history, output_dir, weight_bits)
        
        # Evaluate final model
        print(f"\nEvaluating {weight_bits}-bit quantized model...")
        test_loss, test_accuracy = model.evaluate(val_gen, verbose=0)
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        
        # Calculate ROC AUC
        val_preds = model.predict(val_gen, verbose=0).ravel()
        val_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
        fpr, tpr, thresholds = roc_curve(val_true, val_preds)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Save evaluation results
        eval_results = {
            'weight_bits': weight_bits,
            'activation_bits': self.activation_bits,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'roc_auc': float(roc_auc),
            'model_name': f'quantized_model_{weight_bits}bit'
        }
        
        eval_file = output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save ROC curve
        self.plot_roc_curve(fpr, tpr, roc_auc, output_dir, weight_bits)
        
        print(f"\nTraining completed! Results saved to: {output_dir}")
        print(f"Best model saved to: {checkpoint_path}")
        
        return model, history, output_dir, eval_results
    
    def plot_training_history(self, history, output_dir, weight_bits):
        """
        Plot training history and save to output directory.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'Model Accuracy ({weight_bits}-bit Quantized)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'Model Loss ({weight_bits}-bit Quantized)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_file = output_dir / f"training_history_{weight_bits}bit.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {plot_file}")
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, output_dir, weight_bits):
        """
        Plot ROC curve and save to output directory.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({weight_bits}-bit Quantized)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plot_file = output_dir / f"roc_curve_{weight_bits}bit.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve plot saved to: {plot_file}")
    
    def compare_quantization_results(self, results_dict):
        """
        Compare results across different quantization levels.
        """
        print("\n" + "=" * 60)
        print("QUANTIZATION COMPARISON RESULTS")
        print("=" * 60)
        
        # Create comparison table
        print(f"{'Bit-width':<10} {'Accuracy':<12} {'Loss':<12} {'ROC AUC':<12} {'Epochs':<8}")
        print("-" * 60)
        
        for weight_bits, results in results_dict.items():
            accuracy = results['test_accuracy']
            loss = results['test_loss']
            roc_auc = results['roc_auc']
            epochs = results['epochs']
            print(f"{weight_bits}-bit:   {accuracy:.4f}      {loss:.4f}      {roc_auc:.4f}      {epochs}")
        
        # Find best performing model
        best_bit = max(results_dict.keys(), key=lambda x: results_dict[x]['test_accuracy'])
        best_acc = results_dict[best_bit]['test_accuracy']
        print(f"\nBest performing model: {best_bit}-bit quantization")
        print(f"Best accuracy: {best_acc:.4f}")
        
        # Plot comparison
        self.plot_quantization_comparison(results_dict)
    
    def plot_quantization_comparison(self, results_dict):
        """
        Plot comparison of different quantization levels.
        """
        bit_widths = list(results_dict.keys())
        accuracies = [results_dict[bits]['test_accuracy'] for bits in bit_widths]
        losses = [results_dict[bits]['test_loss'] for bits in bit_widths]
        roc_aucs = [results_dict[bits]['roc_auc'] for bits in bit_widths]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy comparison
        ax1.bar(bit_widths, accuracies, color='skyblue')
        ax1.set_title('Test Accuracy by Quantization Level')
        ax1.set_xlabel('Weight Bit-width')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(bit_widths[i], v + 0.01, f'{v:.4f}', ha='center')
        
        # Loss comparison
        ax2.bar(bit_widths, losses, color='lightcoral')
        ax2.set_title('Test Loss by Quantization Level')
        ax2.set_xlabel('Weight Bit-width')
        ax2.set_ylabel('Loss')
        for i, v in enumerate(losses):
            ax2.text(bit_widths[i], v + 0.01, f'{v:.4f}', ha='center')
        
        # ROC AUC comparison
        ax3.bar(bit_widths, roc_aucs, color='lightgreen')
        ax3.set_title('ROC AUC by Quantization Level')
        ax3.set_xlabel('Weight Bit-width')
        ax3.set_ylabel('ROC AUC')
        ax3.set_ylim(0, 1)
        for i, v in enumerate(roc_aucs):
            ax3.text(bit_widths[i], v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = Path(f"quantization_comparison_{timestamp}.png")
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Quantization comparison plot saved to: {comparison_file}")

def main():
    """
    Main function to train quantized models and compare results.
    """
    if not QKERAS_AVAILABLE:
        print("QKeras not available. Please install with: pip install qkeras")
        return
    
    print("=" * 60)
    print("TRAIN QUANTIZED MODIFIED MODELS (2, 3, 6-bit)")
    print("=" * 60)
    
    # Initialize trainer
    trainer = QuantizedModifiedModelTrainer()
    
    # Train models with different quantization levels
    results_dict = {}
    
    for weight_bits in trainer.weight_bits:
        print(f"\n{'='*20} Training {weight_bits}-bit model {'='*20}")
        
        model, history, output_dir, eval_results = trainer.train_quantized_model(
            weight_bits, epochs=200, patience=15
        )
        
        # Add epochs trained to results
        eval_results['epochs'] = len(history.history['accuracy'])
        results_dict[weight_bits] = eval_results
    
    # Compare results
    trainer.compare_quantization_results(results_dict)
    
    print("\n" + "=" * 60)
    print("QUANTIZATION ANALYSIS COMPLETE!")
    print("=" * 60)
    print("Models trained with:")
    print(f"  - Weight quantization: {trainer.weight_bits}-bit")
    print(f"  - Activation quantization: {trainer.activation_bits}-bit")
    print(f"  - Architecture: 3x5 conv, 200->100 head dense layers")
    
    return results_dict

if __name__ == "__main__":
    results = main() 