#!/usr/bin/env python3
"""
Script to plot ROC curves for both non-quantized and 6-bit quantized models (Model1, Model2, Model3) on the same graph.

This script:
1. Loads the three non-quantized models and their 6-bit quantized counterparts
2. Evaluates them on validation data
3. Generates ROC curves for all six models on a single plot for comparison
4. Uses solid lines for non-quantized models and dashed lines for 6-bit quantized models

Usage:
    python plot_roc_curves_comparison.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import json

# QKeras imports for loading quantized models
try:
    from qkeras import QDense, QActivation, QConv2D
    from qkeras.quantizers import quantized_bits, quantized_relu
    QKERAS_AVAILABLE = True
    print("QKeras loaded successfully for quantized model support")
except ImportError:
    print("WARNING: QKeras not available. Quantized models will not load.")
    QKERAS_AVAILABLE = False

# Add parent directory to path for data generator import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OptimizedDataGenerator4 as ODG

# Import model building functions
from Model1.quantized_mlp_model import build_non_quantized_mlp_model
from Model2.cnn_model import build_custom_cnn_model
from Model3.cnn_model import build_cnn_model as build_model3_cnn

class ROCCurveComparer:
    """
    Class to generate and compare ROC curves for the three non-quantized models.
    """
    
    def __init__(self):
        # Data directories (using the same paths as the training scripts)
        self.base_dir = Path("/local/d1/smartpixML/filtering_models/shuffling_data/filtering_records1024_data_shuffled_single")
        self.test_dir = self.base_dir / "tfrecords_test"  # Assuming test data exists
        self.val_dir = self.base_dir / "tfrecords_validation"  # Use validation as test if no test dir
        
        # Check which data directory exists
        if self.test_dir.exists():
            self.eval_dir = self.test_dir
            self.eval_name = "test"
        elif self.val_dir.exists():
            self.eval_dir = self.val_dir
            self.eval_name = "validation"
        else:
            # Fallback to local path structure
            self.base_dir = Path("/home/youeric/PixelML/smart_pixels_ml/shuffling_data/filtering_records1024_data_shuffled_single")
            self.eval_dir = self.base_dir / "tfrecords_validation"
            self.eval_name = "validation"
        
        print(f"Using {self.eval_name} data from: {self.eval_dir}")
        
        # Model paths - using trained unquantized models
        self.model1_path = Path(__file__).parent / "Model1/quantized_model1_results_20250731_134522/non_quantized_model"
        self.model2_path = Path(__file__).parent / "Model2/quantized_complicated_results_20250730_114020/non_quantized_model/trial_1/non_quantized_trial1.h5"
        self.model3_path = Path(__file__).parent / "Model3/combined_results_20250730_021907/non_quantized_model/best_model.h5"
        
        # 6-bit quantized model paths
        self.model1_6bit_path = Path(__file__).parent / "Model1/quantized_model1_results_20250731_134522/quantized_6w0i_6a0i/trial_1/quantized_mlp_quantized_6w0i_6a0i_trial1.h5"
        self.model2_6bit_path = Path(__file__).parent / "Model2/quantized_complicated_results_20250730_114020/quantized_6bit_int0/trial_1/quantized_cnn_6bit_int0_trial1.h5"
        self.model3_6bit_path = Path(__file__).parent / "Model3/combined_results_20250730_021907/quantized_model_6bit_0int/best_model_6bit_0int.h5"
        
        # Results storage
        self.results = {}
        
        # Colors for the models (Set2 colormap like validation plots)
        self.set2_colors = plt.cm.Set2(np.linspace(0, 1, 8))  # Generate Set2 colors
        self.colors = [self.set2_colors[i] for i in range(3)]  # Use first 3 Set2 colors
        self.linestyles = ['-', '--']  # Solid for non-quantized, dashed for 6-bit
        
    def load_or_train_model1(self):
        """Load or train Model1 (MLP)"""
        print("\n=== Processing Model1 (MLP) ===")
        
        # Check for existing trained model
        if self.model1_path.exists():
            trial_dirs = list(self.model1_path.glob("trial_*"))
            if trial_dirs:
                # Use the first trial
                trial_dir = trial_dirs[0]
                keras_files = list(trial_dir.glob("*.keras"))
                h5_files = list(trial_dir.glob("*.h5"))
                
                if keras_files:
                    model_path = keras_files[0]
                    print(f"Attempting to load existing Model1 from: {model_path}")
                    try:
                        model = tf.keras.models.load_model(model_path)
                        print("Model1 loaded successfully")
                        return model
                    except Exception as e:
                        print(f"Error loading Model1 (.keras): {e}")
                        print("Falling back to building new model...")
                        
                if h5_files:
                    model_path = h5_files[0]
                    print(f"Attempting to load existing Model1 from: {model_path}")
                    try:
                        model = tf.keras.models.load_model(model_path)
                        print("Model1 loaded successfully")
                        return model
                    except Exception as e:
                        print(f"Error loading Model1 (.h5): {e}")
                        print("Falling back to building new model...")
                
                print("No loadable saved Model1 found, building new model...")
                model = build_non_quantized_mlp_model()
                print("WARNING: Model1 is untrained - results may not be meaningful")
            else:
                print("No trial directories found, building new model...")
                model = build_non_quantized_mlp_model()
                print("WARNING: Model1 is untrained - results may not be meaningful")
        else:
            print("Model1 path not found, building new model...")
            model = build_non_quantized_mlp_model()
            print("WARNING: Model1 is untrained - results may not be meaningful")
        
        return model
    
    def load_model2(self):
        """Load trained Model2 (CNN-like)"""
        print("\n=== Processing Model2 (CNN-like) ===")
        
        if self.model2_path.exists():
            print(f"Loading trained Model2 from: {self.model2_path}")
            try:
                model = tf.keras.models.load_model(self.model2_path)
                print("Model2 loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading Model2: {e}")
                print("Falling back to building new model...")
                model = build_custom_cnn_model()
                print("WARNING: Model2 is untrained - results may not be meaningful")
                return model
        else:
            print("Model2 not found - expected path:", self.model2_path)
            print("Building new model...")
            model = build_custom_cnn_model()
            print("WARNING: Model2 is untrained - results may not be meaningful")
            return model
    
    def load_model3(self):
        """Load trained Model3 (Conv2D)"""
        print("\n=== Processing Model3 (Conv2D) ===")
        
        if self.model3_path.exists():
            print(f"Loading trained Model3 from: {self.model3_path}")
            try:
                model = tf.keras.models.load_model(self.model3_path)
                print("Model3 loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading Model3: {e}")
                print("Model3 file exists but couldn't be loaded")
                return None
        else:
            print("Model3 not found - expected path:", self.model3_path)
            print("Available Model3 files:")
            model3_dir = self.model3_path.parent
            if model3_dir.exists():
                for file in model3_dir.iterdir():
                    print(f"  - {file.name}")
            return None
    
    def load_model1_6bit(self):
        """Load Model1 6-bit quantized model"""
        print("\n=== Processing Model1 6-bit (MLP) ===")
        
        if not QKERAS_AVAILABLE:
            print("QKeras not available - cannot load Model1 6-bit")
            return None
        
        if self.model1_6bit_path.exists():
            print(f"Loading Model1 6-bit from: {self.model1_6bit_path}")
            try:
                # Load with QKeras custom objects
                with tf.keras.utils.custom_object_scope({
                    'QDense': QDense,
                    'QActivation': QActivation,
                    'quantized_bits': quantized_bits,
                    'quantized_relu': quantized_relu
                }):
                    model = tf.keras.models.load_model(self.model1_6bit_path)
                print("Model1 6-bit loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading Model1 6-bit: {e}")
                print("Model1 6-bit file exists but couldn't be loaded")
                return None
        else:
            print("Model1 6-bit not found - expected path:", self.model1_6bit_path)
            return None
    
    def load_model2_6bit(self):
        """Load Model2 6-bit quantized model"""
        print("\n=== Processing Model2 6-bit (CNN-like) ===")
        
        if not QKERAS_AVAILABLE:
            print("QKeras not available - cannot load Model2 6-bit")
            return None
        
        if self.model2_6bit_path.exists():
            print(f"Loading Model2 6-bit from: {self.model2_6bit_path}")
            try:
                # Load with QKeras custom objects
                with tf.keras.utils.custom_object_scope({
                    'QDense': QDense,
                    'QActivation': QActivation,
                    'QConv2D': QConv2D,
                    'quantized_bits': quantized_bits,
                    'quantized_relu': quantized_relu
                }):
                    model = tf.keras.models.load_model(self.model2_6bit_path)
                print("Model2 6-bit loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading Model2 6-bit: {e}")
                print("Model2 6-bit file exists but couldn't be loaded")
                return None
        else:
            print("Model2 6-bit not found - expected path:", self.model2_6bit_path)
            return None
    
    def load_model3_6bit(self):
        """Load Model3 6-bit quantized model"""
        print("\n=== Processing Model3 6-bit (Conv2D) ===")
        
        if not QKERAS_AVAILABLE:
            print("QKeras not available - cannot load Model3 6-bit")
            return None
        
        if self.model3_6bit_path.exists():
            print(f"Loading Model3 6-bit from: {self.model3_6bit_path}")
            try:
                # Load with QKeras custom objects
                with tf.keras.utils.custom_object_scope({
                    'QDense': QDense,
                    'QActivation': QActivation,
                    'QConv2D': QConv2D,
                    'quantized_bits': quantized_bits,
                    'quantized_relu': quantized_relu
                }):
                    model = tf.keras.models.load_model(self.model3_6bit_path)
                print("Model3 6-bit loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading Model3 6-bit: {e}")
                print("Model3 6-bit file exists but couldn't be loaded")
                return None
        else:
            print("Model3 6-bit not found - expected path:", self.model3_6bit_path)
            return None
    
    def setup_data_generators(self):
        """Setup data generators for each model type"""
        print(f"\nSetting up data generators from: {self.eval_dir}")
        
        # Model1 uses: z_global, x_size, y_size, y_local
        self.model1_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(self.eval_dir),
            x_feature_description=['z_global', 'x_size', 'y_size', 'y_local']
        )
        
        # Model2 uses: x_profile, z_global, y_profile, y_local
        self.model2_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(self.eval_dir),
            x_feature_description=['x_profile', 'z_global', 'y_profile', 'y_local']
        )
        
        # Model3 uses: cluster, z_global, y_local (cluster has shape (13, 21))
        self.model3_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(self.eval_dir),
            x_feature_description=['cluster', 'z_global', 'y_local']
        )
        
        print(f"Model1 generator length: {len(self.model1_gen)}")
        print(f"Model2 generator length: {len(self.model2_gen)}")
        print(f"Model3 generator length: {len(self.model3_gen)}")
    
    def evaluate_model(self, model, data_gen, model_name):
        """Evaluate a model and return predictions and true labels"""
        print(f"\nEvaluating {model_name}...")
        
        try:
            # Get predictions
            predictions = model.predict(data_gen, verbose=1)
            # Convert TensorFlow tensors to numpy arrays
            if hasattr(predictions, 'numpy'):
                predictions = predictions.numpy()
            predictions = np.array(predictions).ravel()  # Flatten to 1D
            
            # Get true labels
            true_labels = []
            for i in range(len(data_gen)):
                _, y_batch = data_gen[i]
                # Convert TensorFlow tensors to numpy arrays
                if hasattr(y_batch, 'numpy'):
                    y_batch = y_batch.numpy()
                y_batch = np.array(y_batch)
                true_labels.append(y_batch.ravel())
            true_labels = np.concatenate(true_labels)
            
            print(f"{model_name} - Predictions shape: {predictions.shape}, True labels shape: {true_labels.shape}")
            print(f"{model_name} - Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            print(f"{model_name} - True labels unique: {np.unique(true_labels)}")
            
            return predictions, true_labels
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return None, None
    
    def generate_roc_curves(self):
        """Generate ROC curves for all three models"""
        print("\n=== Generating ROC Curves ===")
        
        # Setup data generators
        self.setup_data_generators()
        
        # Load trained models (both non-quantized and 6-bit)
        model1 = self.load_or_train_model1()
        model2 = self.load_model2()
        model3 = self.load_model3()
        
        model1_6bit = self.load_model1_6bit()
        model2_6bit = self.load_model2_6bit()
        model3_6bit = self.load_model3_6bit()
        
        models = [
            # Non-quantized models (solid lines)
            (model1, self.model1_gen, "Model1 (MLP)", 0, 0),
            (model2, self.model2_gen, "Model2 (CNN-like)", 1, 0),
            (model3, self.model3_gen, "Model3 (Conv2D)", 2, 0),
            # 6-bit quantized models (dashed lines)
            (model1_6bit, self.model1_gen, "Model1 6-bit (MLP)", 0, 1),
            (model2_6bit, self.model2_gen, "Model2 6-bit (CNN-like)", 1, 1),
            (model3_6bit, self.model3_gen, "Model3 6-bit (Conv2D)", 2, 1)
        ]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        
        for model, data_gen, model_name, color_idx, linestyle_idx in models:
            if model is None:
                print(f"Skipping {model_name} - model not available")
                continue
                
            predictions, true_labels = self.evaluate_model(model, data_gen, model_name)
            
            if predictions is not None and true_labels is not None:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(true_labels, predictions)
                roc_auc = auc(fpr, tpr)
                
                # Store results
                self.results[model_name] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc,
                    'predictions': predictions,
                    'true_labels': true_labels
                }
                
                # Plot ROC curve with appropriate color and linestyle
                plt.plot(fpr, tpr, color=self.colors[color_idx], 
                        linestyle=self.linestyles[linestyle_idx], lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
                
                print(f"{model_name} - AUC: {roc_auc:.4f}")
            else:
                print(f"Skipping {model_name} due to evaluation error")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
        
        # Customize plot
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
        plt.title('ROC and AUC Comparison', fontsize=18, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = Path(__file__).parent / "roc_curves_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nROC curves plot saved to: {output_path}")
        print(f"Full path: {output_path.absolute()}")
        
        # Save results to JSON
        results_json = {}
        for model_name, data in self.results.items():
            results_json[model_name] = {
                'auc': float(data['auc']),
                'fpr': data['fpr'].tolist(),
                'tpr': data['tpr'].tolist()
            }
        
        json_path = Path(__file__).parent / "roc_curves_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Results saved to: {json_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("ROC CURVES COMPARISON SUMMARY")
        print("="*60)
        
        if self.results:
            # Sort by AUC score
            sorted_results = sorted(self.results.items(), key=lambda x: x[1]['auc'], reverse=True)
            
            print(f"{'Model':<20} {'AUC Score':<10} {'Samples':<10}")
            print("-" * 40)
            
            for model_name, data in sorted_results:
                auc_score = data['auc']
                n_samples = len(data['true_labels'])
                print(f"{model_name:<20} {auc_score:<10.4f} {n_samples:<10}")
            
            print("\nBest performing model:", sorted_results[0][0])
        else:
            print("No results to display")

def main():
    """Main function"""
    print("ROC and AUC Comparison: Non-Quantized vs 6-bit Quantized Models")
    print("=" * 65)
    
    # Show where output will be saved
    output_path = Path(__file__).parent / "roc_curves_comparison.png"
    json_path = Path(__file__).parent / "roc_curves_results.json"
    print(f"Plot will be saved to: {output_path.absolute()}")
    print(f"Results JSON will be saved to: {json_path.absolute()}")
    print("-" * 50)
    
    try:
        comparer = ROCCurveComparer()
        comparer.generate_roc_curves()
        comparer.print_summary()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()