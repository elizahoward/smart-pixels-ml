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

class CombinedModelTrainer:
    """
    Class to train both non-quantized and quantized versions of the modified model.
    Quantized models: 2, 3, 4, 6, 8, 16-bit with zero integer bits.
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
        self.weight_bits = [2, 3, 4, 6, 8, 16, 32]  # Expanded bit-widths to test
        self.activation_bits = 8  # Fixed 8-bit activation quantization
        # Integer bit configurations for each weight bit width
        self.integer_bits_configs = {
            2: [0],
            3: [0],
            4: [0],
            6: [0],
            8: [0],
            16: [0],
            32: [0]
        }
        
        # Create main combined results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.combined_results_dir = Path(f"combined_results_{timestamp}")
        self.combined_results_dir.mkdir(exist_ok=True)
        print(f"Created combined results directory: {self.combined_results_dir}")
        
    def build_modified_model(self):
        """
        Build the non-quantized modified model.
        """
        print("Building non-quantized modified model...")
        print(f"  - Conv2D: {self.kernel_rows}x{self.kernel_cols} kernel, {self.conv_filters} filters")
        print(f"  - Head dense layers: {self.head_units} -> {self.head_units_2}")
        print(f"  - Z units: {self.z_units}, Y units: {self.y_units}")
        print(f"  - Dropout: {self.dropout_rate}")
        
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
        
        print("Non-quantized model built successfully!")
        model.summary()
        
        return model
    
    def build_quantized_model(self, weight_bits, integer_bits):
        """
        Build the quantized model with specified bit-width and integer bits.
        """
        print(f"Building quantized model with {weight_bits}-bit weights, {integer_bits} integer bits, and {self.activation_bits}-bit activations...")
        
        # Define quantizers with specified integer bits
        weight_quantizer = quantized_bits(weight_bits, integer_bits, 1)
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
            name=f"quantized_model_{weight_bits}bit_{integer_bits}int"
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
        
        # BASE_DIR = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
        BASE_DIR = Path("/local/d1/smartpixML/filtering_models/shuffling_data/filtering_records1024_data_shuffled_single")
        TRAIN_DIR = BASE_DIR / "tfrecords_train"
        VALIDATION_DIR = BASE_DIR / "tfrecords_validation"
        
        train_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(TRAIN_DIR),
            x_feature_description=['cluster', 'y_local', 'z_global'],
            time_stamps=[19]  # Use only the last timestamp
        )
        
        val_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(VALIDATION_DIR),
            x_feature_description=['cluster', 'y_local', 'z_global'],
            time_stamps=[19]  # Use only the last timestamp
        )
        
        print(f"Training generator length: {len(train_gen)}")
        print(f"Validation generator length: {len(val_gen)}")
        
        return train_gen, val_gen
    
    def train_non_quantized_model(self, epochs=200, patience=15):
        """
        Train the non-quantized modified model.
        """
        print(f"\nTraining non-quantized model for {epochs} epochs...")
        
        # Build model
        model = self.build_modified_model()
        
        # Setup data generators
        train_gen, val_gen = self.setup_data_generators()
        
        # Create output directory within combined_results
        output_dir = self.combined_results_dir / "non_quantized_model"
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
            'model_type': 'non_quantized',
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
        self.plot_training_history(history, output_dir, "non_quantized")
        
        # Evaluate final model
        print(f"\nEvaluating non-quantized model...")
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
            'model_type': 'non_quantized',
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'roc_auc': float(roc_auc),
            'model_name': 'modified_model_3x5_200_100',
            'epochs_trained': len(history.history['accuracy'])
        }
        
        eval_file = output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save model in both .h5 and .keras formats
        keras_model_path = output_dir / "best_model.keras"
        model.save(str(keras_model_path))
        print(f"Model saved in .keras format: {keras_model_path}")
        
        print(f"\nNon-quantized training completed! Results saved to: {output_dir}")
        print(f"Best model saved to: {checkpoint_path}")
        print(f"Model also saved in .keras format: {keras_model_path}")
        
        return model, history, output_dir, eval_results
    
    def train_quantized_model(self, weight_bits, integer_bits, epochs=200, patience=15):
        """
        Train a quantized model with specified bit-width and integer bits.
        """
        print(f"\nTraining {weight_bits}-bit quantized model with {integer_bits} integer bits for {epochs} epochs...")
        
        # Build model
        model = self.build_quantized_model(weight_bits, integer_bits)
        
        # Setup data generators
        train_gen, val_gen = self.setup_data_generators()
        
        # Create output directory within combined_results
        output_dir = self.combined_results_dir / f"quantized_model_{weight_bits}bit_{integer_bits}int"
        output_dir.mkdir(exist_ok=True)
        
        # Callbacks
        checkpoint_path = output_dir / f"best_model_{weight_bits}bit_{integer_bits}int.h5"
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
            'model_type': 'quantized',
            'weight_bits': weight_bits,
            'activation_bits': self.activation_bits,
            'integer_bits': integer_bits,
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
        self.plot_training_history(history, output_dir, f"{weight_bits}bit_{integer_bits}int")
        
        # Evaluate final model
        print(f"\nEvaluating {weight_bits}-bit quantized model with {integer_bits} integer bits...")
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
            'model_type': 'quantized',
            'weight_bits': weight_bits,
            'activation_bits': self.activation_bits,
            'integer_bits': integer_bits,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'roc_auc': float(roc_auc),
            'model_name': f'quantized_model_{weight_bits}bit_{integer_bits}int',
            'epochs_trained': len(history.history['accuracy'])
        }
        
        eval_file = output_dir / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Save ROC curve
        self.plot_roc_curve(fpr, tpr, roc_auc, output_dir, f"{weight_bits}bit_{integer_bits}int")
        
        # Save model in both .h5 and .keras formats
        keras_model_path = output_dir / f"best_model_{weight_bits}bit_{integer_bits}int.keras"
        model.save(str(keras_model_path))
        print(f"Model saved in .keras format: {keras_model_path}")
        
        print(f"\nTraining completed! Results saved to: {output_dir}")
        print(f"Best model saved to: {checkpoint_path}")
        print(f"Model also saved in .keras format: {keras_model_path}")
        
        return model, history, output_dir, eval_results
    
    def plot_training_history(self, history, output_dir, model_type):
        """
        Plot training history and save to output directory.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'Model Accuracy ({model_type})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'Model Loss ({model_type})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_file = output_dir / f"training_history_{model_type}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history plot saved to: {plot_file}")
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, output_dir, model_suffix):
        """
        Plot ROC curve and save to output directory.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve ({model_suffix})')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plot_file = output_dir / f"roc_curve_{model_suffix}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve plot saved to: {plot_file}")
    
    def plot_comparison_graphs(self, all_results):
        """
        Create comparison graphs for validation accuracy and loss across all models.
        """
        print("\nCreating comparison graphs...")
        
        # Prepare data for plotting
        model_names = []
        val_accuracies = []
        val_losses = []
        colors = []
        
        # Non-quantized model
        if 'non_quantized' in all_results:
            non_quantized = all_results['non_quantized']
            model_names.append('Non-quantized')
            val_accuracies.append(non_quantized['history'].history['val_accuracy'])
            val_losses.append(non_quantized['history'].history['val_loss'])
            colors.append('black')
        
        # Quantized models
        for weight_bits in self.weight_bits:
            for integer_bits in self.integer_bits_configs[weight_bits]:
                model_key = f"{weight_bits}bit_{integer_bits}int"
                if model_key in all_results:
                    quantized = all_results[model_key]
                    model_names.append(f'{weight_bits}-bit_{integer_bits}-int')
                    val_accuracies.append(quantized['history'].history['val_accuracy'])
                    val_losses.append(quantized['history'].history['val_loss'])
                    colors.append(plt.cm.viridis(weight_bits / max(self.weight_bits)))
        
        # Plot validation accuracy comparison
        plt.figure(figsize=(12, 8))
        for i, (name, acc, color) in enumerate(zip(model_names, val_accuracies, colors)):
            plt.plot(acc, label=name, color=color, linewidth=2)
        
        plt.title('Validation Accuracy Comparison (200 Epochs)', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        acc_plot_file = self.combined_results_dir / "validation_accuracy_comparison.png"
        plt.savefig(acc_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot validation loss comparison
        plt.figure(figsize=(12, 8))
        for i, (name, loss, color) in enumerate(zip(model_names, val_losses, colors)):
            plt.plot(loss, label=name, color=color, linewidth=2)
        
        plt.title('Validation Loss Comparison (200 Epochs)', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        loss_plot_file = self.combined_results_dir / "validation_loss_comparison.png"
        plt.savefig(loss_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Validation accuracy comparison saved to: {acc_plot_file}")
        print(f"Validation loss comparison saved to: {loss_plot_file}")
        
        # Create bar histograms for best validation accuracy and loss
        self.plot_best_metrics_bar_charts(all_results)
        
        return acc_plot_file, loss_plot_file
    
    def plot_best_metrics_bar_charts(self, all_results):
        """
        Create bar histograms for best validation accuracy and loss across all models.
        """
        print("\nCreating bar histograms for best metrics...")
        
        # Prepare data for bar charts
        model_names = []
        best_val_accuracies = []
        best_val_losses = []
        colors = []
        
        # Non-quantized model
        if 'non_quantized' in all_results:
            non_quantized = all_results['non_quantized']
            model_names.append('Non-quantized')
            best_val_accuracies.append(max(non_quantized['history'].history['val_accuracy']))
            best_val_losses.append(min(non_quantized['history'].history['val_loss']))
            colors.append('black')
        
        # Quantized models
        for weight_bits in self.weight_bits:
            for integer_bits in self.integer_bits_configs[weight_bits]:
                model_key = f"{weight_bits}bit_{integer_bits}int"
                if model_key in all_results:
                    quantized = all_results[model_key]
                    model_names.append(f'{weight_bits}-bit_{integer_bits}-int')
                    best_val_accuracies.append(max(quantized['history'].history['val_accuracy']))
                    best_val_losses.append(min(quantized['history'].history['val_loss']))
                    colors.append(plt.cm.viridis(weight_bits / max(self.weight_bits)))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart for best validation accuracy
        bars1 = ax1.bar(model_names, best_val_accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('Best Validation Accuracy by Model', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Model Type', fontsize=12)
        ax1.set_ylabel('Best Validation Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, best_val_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax1.tick_params(axis='x', rotation=45)
        
        # Bar chart for best validation loss
        bars2 = ax2.bar(model_names, best_val_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_title('Best Validation Loss by Model', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model Type', fontsize=12)
        ax2.set_ylabel('Best Validation Loss', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, loss in zip(bars2, best_val_losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{loss:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the bar charts
        bar_charts_file = self.combined_results_dir / "best_metrics_bar_charts.png"
        plt.savefig(bar_charts_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Best metrics bar charts saved to: {bar_charts_file}")
        
        # Print summary of best metrics
        print("\n" + "=" * 60)
        print("BEST METRICS SUMMARY")
        print("=" * 60)
        print(f"{'Model Type':<15} {'Best Val Acc':<15} {'Best Val Loss':<15}")
        print("-" * 60)
        for name, acc, loss in zip(model_names, best_val_accuracies, best_val_losses):
            print(f"{name:<15} {acc:.4f}          {loss:.4f}")
        
        # Find best performing models
        best_acc_model = model_names[np.argmax(best_val_accuracies)]
        best_loss_model = model_names[np.argmin(best_val_losses)]
        print(f"\nBest validation accuracy: {best_acc_model} ({max(best_val_accuracies):.4f})")
        print(f"Best validation loss: {best_loss_model} ({min(best_val_losses):.4f})")
        
        return bar_charts_file
    
    def compare_final_results(self, all_results):
        """
        Compare final results across all models.
        """
        print("\n" + "=" * 80)
        print("FINAL RESULTS COMPARISON")
        print("=" * 80)
        
        # Create comparison table
        print(f"{'Model Type':<20} {'Accuracy':<12} {'Loss':<12} {'ROC AUC':<12} {'Epochs':<8}")
        print("-" * 85)
        
        results_list = []
        
        # Non-quantized model
        if 'non_quantized' in all_results:
            non_quantized = all_results['non_quantized']
            accuracy = non_quantized['eval_results']['test_accuracy']
            loss = non_quantized['eval_results']['test_loss']
            roc_auc = non_quantized['eval_results']['roc_auc']
            epochs = non_quantized['eval_results']['epochs_trained']
            print(f"{'Non-quantized':<20} {accuracy:.4f}      {loss:.4f}      {roc_auc:.4f}      {epochs}")
            results_list.append(('Non-quantized', accuracy, loss, roc_auc, epochs))
        
        # Quantized models
        for weight_bits in self.weight_bits:
            for integer_bits in self.integer_bits_configs[weight_bits]:
                model_key = f"{weight_bits}bit_{integer_bits}int"
                if model_key in all_results:
                    quantized = all_results[model_key]
                    accuracy = quantized['eval_results']['test_accuracy']
                    loss = quantized['eval_results']['test_loss']
                    roc_auc = quantized['eval_results']['roc_auc']
                    epochs = quantized['eval_results']['epochs_trained']
                    model_name = f'{weight_bits}-bit_{integer_bits}-int'
                    print(f"{model_name:<20} {accuracy:.4f}      {loss:.4f}      {roc_auc:.4f}      {epochs}")
                    results_list.append((model_name, accuracy, loss, roc_auc, epochs))
        
        # Find best performing model
        if results_list:
            best_model = max(results_list, key=lambda x: x[1])  # Best accuracy
            best_acc = best_model[1]
            print(f"\nBest performing model: {best_model[0]}")
            print(f"Best accuracy: {best_acc:.4f}")
        
        return results_list
    
    def plot_roc_curves_overlay(self, all_results):
        """
        Plot an overlay of all ROC curves on one common graph.
        """
        print("\nCreating ROC curves overlay...")
        
        plt.figure(figsize=(10, 10))
        
        # Colors for different models
        colors = ['black'] + [plt.cm.viridis(i/len(self.weight_bits)) for i in range(len(self.weight_bits))]
        
        # Plot non-quantized model ROC curve
        if 'non_quantized' in all_results:
            non_quantized = all_results['non_quantized']
            roc_auc = non_quantized['eval_results']['roc_auc']
            
            # Get the saved ROC curve data or recalculate
            val_gen = self.setup_data_generators()[1]  # Get validation generator
            val_preds = non_quantized['model'].predict(val_gen, verbose=0).ravel()
            val_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
            fpr, tpr, _ = roc_curve(val_true, val_preds)
            
            plt.plot(fpr, tpr, color=colors[0], linewidth=2, 
                    label=f'Non-quantized (AUC = {roc_auc:.4f})')
        
        # Plot quantized models ROC curves
        for i, weight_bits in enumerate(self.weight_bits):
            for integer_bits in self.integer_bits_configs[weight_bits]:
                model_key = f"{weight_bits}bit_{integer_bits}int"
                if model_key in all_results:
                    quantized = all_results[model_key]
                    roc_auc = quantized['eval_results']['roc_auc']
                    
                    # Get the saved ROC curve data or recalculate
                    val_gen = self.setup_data_generators()[1]  # Get validation generator
                    val_preds = quantized['model'].predict(val_gen, verbose=0).ravel()
                    val_true = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
                    fpr, tpr, _ = roc_curve(val_true, val_preds)
                    
                    plt.plot(fpr, tpr, color=colors[i+1], linewidth=2, 
                            label=f'{weight_bits}-bit_{integer_bits}-int (AUC = {roc_auc:.4f})')
        
        # Add diagonal line for random classifier
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - All Models', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Save the overlay plot
        overlay_file = self.combined_results_dir / "roc_curves_overlay.png"
        plt.savefig(overlay_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves overlay saved to: {overlay_file}")
        
        return overlay_file

def main():
    """
    Main function to train both non-quantized and quantized models and compare results.
    """
    if not QKERAS_AVAILABLE:
        print("QKeras not available. Please install with: pip install qkeras")
        return
    
    print("=" * 80)
    print("TRAIN COMBINED MODELS (Non-quantized + Quantized 2,3,4,6,8,16-bit)")
    print("=" * 80)
    
    # Initialize trainer
    trainer = CombinedModelTrainer()
    
    # Dictionary to store all results
    all_results = {}
    
    # Train non-quantized model first
    print(f"\n{'='*20} Training Non-quantized Model {'='*20}")
    non_quantized_model, non_quantized_history, non_quantized_dir, non_quantized_eval = trainer.train_non_quantized_model(
        epochs=200, patience=35
    )
    all_results['non_quantized'] = {
        'model': non_quantized_model,
        'history': non_quantized_history,
        'output_dir': non_quantized_dir,
        'eval_results': non_quantized_eval
    }
    
    # Train quantized models
    for weight_bits in trainer.weight_bits:
        for integer_bits in trainer.integer_bits_configs[weight_bits]:
            print(f"\n{'='*20} Training {weight_bits}-bit Quantized Model with {integer_bits} Integer Bits {'='*20}")
            
            quantized_model, quantized_history, quantized_dir, quantized_eval = trainer.train_quantized_model(
                weight_bits, integer_bits, epochs=200, patience=35
            )
            
            model_key = f"{weight_bits}bit_{integer_bits}int"
            all_results[model_key] = {
                'model': quantized_model,
                'history': quantized_history,
                'output_dir': quantized_dir,
                'eval_results': quantized_eval
            }
    
    # Create comparison graphs
    trainer.plot_comparison_graphs(all_results)
    
    # Create ROC curves overlay
    trainer.plot_roc_curves_overlay(all_results)
    
    # Compare final results
    trainer.compare_final_results(all_results)
    
    print("\n" + "=" * 80)
    print("COMBINED TRAINING COMPLETE!")
    print("=" * 80)
    print("Models trained:")
    print(f"  - Non-quantized model")
    print(f"  - Quantized models with integer bit search:")
    for weight_bits in trainer.weight_bits:
        int_bits = trainer.integer_bits_configs[weight_bits]
        print(f"    * {weight_bits}-bit: {int_bits} integer bits")
    print(f"  - Activation bits: {trainer.activation_bits}")
    print(f"  - Architecture: 3x5 conv, 200->100 head dense layers")
    print(f"  - Training epochs: 200")
    print(f"  - All results saved to: {trainer.combined_results_dir}")
    
    return all_results

if __name__ == "__main__":
    results = main() 