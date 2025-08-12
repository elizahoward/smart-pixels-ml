#!/usr/bin/env python3
"""
Model Summary Report Generator

This script generates a comprehensive summary of Model1, Model2, and Model3 performance,
focusing on 6-bit quantized models with key metrics including:
- Model size (before/after quantization)
- Total parameters
- Signal efficiency and background rejection
- Performance comparison across models

Usage:
    python model_summary_report.py
"""

import json
import os
import sys
from pathlib import Path

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.utils import get_file
import numpy as np

# Try to import sklearn with fallback
try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sklearn ({e}). Real confusion matrix calculation will be disabled.")
    SKLEARN_AVAILABLE = False
    
    # Fallback confusion matrix implementation
    def confusion_matrix(y_true, y_pred):
        """Simple fallback confusion matrix implementation"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return np.array([[tn, fp], [fn, tp]])

# Add paths for model imports
sys.path.append('Model1')
sys.path.append('Model2') 
sys.path.append('Model3')

# Import data generator
try:
    import OptimizedDataGenerator4 as ODG
except ImportError:
    print("Warning: Could not import OptimizedDataGenerator4. Real confusion matrix calculation will be disabled.")
    ODG = None

def calculate_model_size_kb(model):
    """Calculate approximate model size in KB"""
    total_params = model.count_params()
    # Assume 32-bit floats for non-quantized (4 bytes per param)
    size_kb = (total_params * 4) / 1024
    return size_kb

def calculate_quantized_size_kb(total_params, bits=6):
    """Calculate quantized model size in KB"""
    # Calculate size with specified bit width
    size_kb = (total_params * bits / 8) / 1024
    return size_kb

def calculate_signal_efficiency_background_rejection(accuracy, roc_auc):
    """
    Calculate signal efficiency and background rejection from accuracy and ROC AUC.
    
    Signal efficiency ≈ True Positive Rate at optimal threshold
    Background rejection ≈ True Negative Rate = 1 - False Positive Rate
    
    These are approximations based on overall accuracy and ROC AUC.
    """
    # For binary classification with balanced data, these are rough estimates
    signal_efficiency = accuracy  # Approximation: TPR ≈ accuracy for balanced data
    
    # Background rejection = TNR = 1 - FPR
    # For balanced data: ROC_AUC ≈ (TPR + TNR) / 2
    # Therefore: TNR ≈ 2 * ROC_AUC - TPR = 2 * ROC_AUC - accuracy
    background_rejection = 2 * roc_auc - accuracy
    
    # Ensure background rejection is within [0, 1] bounds
    background_rejection = max(0, min(1, background_rejection))
    
    return signal_efficiency, background_rejection

def create_data_generator(model_type, data_type='validation'):
    """
    Create data generator for the specified model type.
    
    Args:
        model_type: 'model1', 'model2', or 'model3'
        data_type: 'validation' or 'training'
    
    Returns:
        Data generator or None if creation fails
    """
    if ODG is None or not SKLEARN_AVAILABLE:
        return None
        
    try:
        # Data paths
        base_path = Path("/local/d1/smartpixML/filtering_models/shuffling_data/filtering_records1024_data_shuffled_single")
        
        if data_type == 'validation':
            data_dir = base_path / "tfrecords_validation"
        else:
            data_dir = base_path / "tfrecords_training"
            
        if not data_dir.exists():
            print(f"Warning: Data directory {data_dir} not found")
            return None
            
        # Model-specific feature configurations
        if model_type == 'model1':
            features = ['z_global', 'x_size', 'y_size', 'y_local']
        elif model_type == 'model2':
            features = ['x_profile', 'z_global', 'y_profile', 'y_local']
        elif model_type == 'model3':
            features = ['cluster', 'z_global', 'y_local']
        else:
            print(f"Unknown model type: {model_type}")
            return None
            
        # Create generator
        generator = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(data_dir),
            x_feature_description=features
        )
        
        return generator
        
    except Exception as e:
        print(f"Warning: Could not create data generator for {model_type}: {e}")
        return None

def calculate_real_confusion_metrics(model_path, model_type):
    """
    Calculate real TP/FP rates from model and test data.
    
    Args:
        model_path: Path to the .keras or .h5 model file
        model_type: 'model1', 'model2', or 'model3'
    
    Returns:
        Dictionary with real confusion matrix metrics or None if calculation fails
    """
    try:
        # Load model (suppress verbose output)
        import contextlib
        import io
        
        # Import QKeras layers and setup custom objects
        try:
            import qkeras
            from qkeras import QDense, QConv2D, QActivation
            from qkeras.quantizers import quantized_bits, quantized_relu
            from qkeras.qlayers import Clip
            
            # Create custom objects dict for QKeras layers
            custom_objects = {
                'QDense': QDense,
                'QConv2D': QConv2D, 
                'QActivation': QActivation,
                'quantized_bits': quantized_bits,
                'quantized_relu': quantized_relu,
                'Clip': Clip
            }
            
        except ImportError:
            custom_objects = {}
        
        # Suppress stdout during model loading to avoid verbose output
        with contextlib.redirect_stdout(io.StringIO()):
            # Load QKeras model with custom objects
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Create data generator
        test_generator = create_data_generator(model_type, 'validation')
        if test_generator is None:
            return None
            
        # Get predictions and true labels (suppress output)
        with contextlib.redirect_stdout(io.StringIO()):
            predictions = model.predict(test_generator, verbose=0).ravel()
            
        y_true = np.concatenate([y for _, y in (test_generator[i] for i in range(len(test_generator)))])
        
        # Convert to binary predictions (using 0.5 threshold)
        y_pred = (predictions > 0.5).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate rates
        signal_efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
        background_rejection = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # 1 - background_rejection
        

        
        return {
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'signal_efficiency': float(signal_efficiency),
            'background_rejection': float(background_rejection),
            'false_positive_rate': float(false_positive_rate),
            'total_samples': int(len(y_true)),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true))
        }
        
    except Exception as e:
        # Extract just the error type, not the full config dump
        error_msg = str(e).split('.')[0] if '.' in str(e) else str(e)
        if len(error_msg) > 100:  # Truncate very long error messages
            error_msg = error_msg[:100] + "..."
        print(f"Warning: Could not load model for real metrics calculation: {error_msg}")
        return None

def get_model_path_for_evaluation(model_key, quantization='6bit'):
    """
    Find the model file path for real evaluation.
    
    Args:
        model_key: 'model1', 'model2', or 'model3'
        quantization: '6bit' or other quantization level
    
    Returns:
        Path to model file or None if not found
    """
    try:
        if model_key == 'model1':
            base_path = Path("Model1")
            pattern = "quantized_model1_results_*"
            if quantization == '6bit':
                subdir = "quantized_6w0i_6a0i"
            else:
                subdir = f"quantized_{quantization}"
        elif model_key == 'model2':
            base_path = Path("Model2")
            pattern = "quantized_complicated_results_*"
            if quantization == '6bit':
                subdir = "quantized_6bit_int0"
            else:
                subdir = f"quantized_{quantization}"
        elif model_key == 'model3':
            base_path = Path("Model3")
            pattern = "combined_results_*"
            if quantization == '6bit':
                subdir = "quantized_model_6bit_0int"
            else:
                subdir = f"quantized_model_{quantization}"
        else:
            return None
            
        # Find latest results directory
        results_dirs = list(base_path.glob(pattern))
        if not results_dirs:
            return None
            
        latest_dir = sorted(results_dirs)[-1]
        model_dir = latest_dir / subdir
        
        # Look for .h5 files first (better QKeras compatibility), then .keras
        h5_files = list(model_dir.glob("*.h5"))
        if h5_files:
            return h5_files[0]
            
        keras_files = list(model_dir.glob("*.keras"))
        if keras_files:
            return keras_files[0]
            
        # Try trial subdirectories
        trial_dirs = list(model_dir.glob("trial_*"))
        for trial_dir in trial_dirs:
            h5_files = list(trial_dir.glob("*.h5"))
            if h5_files:
                return h5_files[0]
            keras_files = list(trial_dir.glob("*.keras"))
            if keras_files:
                return keras_files[0]
                
        return None
        
    except Exception as e:
        print(f"Warning: Could not find model path for {model_key}: {e}")
        return None

def load_model_metrics():
    """Load performance metrics for all models"""
    metrics = {}
    
    # Model1 - Find latest results directory
    model1_base = Path("Model1")
    model1_results_dirs = list(model1_base.glob("quantized_model1_results_*"))
    if model1_results_dirs:
        latest_model1 = sorted(model1_results_dirs)[-1]
        
        # Load 6-bit results
        model1_6bit_file = latest_model1 / "quantized_6w0i_6a0i" / "overall_results.json"
        model1_nonquant_file = latest_model1 / "non_quantized_model" / "overall_results.json"
        
        if model1_6bit_file.exists():
            with open(model1_6bit_file, 'r') as f:
                metrics['model1_6bit'] = json.load(f)
        
        if model1_nonquant_file.exists():
            with open(model1_nonquant_file, 'r') as f:
                metrics['model1_nonquant'] = json.load(f)
    
    # Model2 - Find latest results directory
    model2_base = Path("Model2")
    model2_results_dirs = list(model2_base.glob("quantized_complicated_results_*"))
    if model2_results_dirs:
        latest_model2 = sorted(model2_results_dirs)[-1]
        
        # Load 6-bit results
        model2_6bit_file = latest_model2 / "quantized_6bit_int0" / "overall_results.json"
        model2_nonquant_file = latest_model2 / "non_quantized_model" / "overall_results.json"
        
        if model2_6bit_file.exists():
            with open(model2_6bit_file, 'r') as f:
                metrics['model2_6bit'] = json.load(f)
        
        if model2_nonquant_file.exists():
            with open(model2_nonquant_file, 'r') as f:
                metrics['model2_nonquant'] = json.load(f)
    
    # Model3 - Find latest results directory
    model3_base = Path("Model3")
    model3_results_dirs = list(model3_base.glob("combined_results_*"))
    if model3_results_dirs:
        latest_model3 = sorted(model3_results_dirs)[-1]
        
        # Load 6-bit results
        model3_6bit_file = latest_model3 / "quantized_model_6bit_0int" / "evaluation_results.json"
        model3_nonquant_file = latest_model3 / "non_quantized_model" / "evaluation_results.json"
        
        if model3_6bit_file.exists():
            with open(model3_6bit_file, 'r') as f:
                metrics['model3_6bit'] = json.load(f)
        
        if model3_nonquant_file.exists():
            with open(model3_nonquant_file, 'r') as f:
                metrics['model3_nonquant'] = json.load(f)
    
    return metrics

def create_mock_models():
    """Create model architectures to calculate parameters"""
    models = {}
    
    try:
        # Model1: MLP Architecture (z_global, x_size, y_size, y_local -> [17->20->9->16->8] -> 1)
        from tensorflow.keras.layers import Input, Dense, Concatenate
        from tensorflow.keras.models import Model
        
        input1 = Input(shape=(1,), name="z_global")
        input2 = Input(shape=(1,), name="x_size")
        input3 = Input(shape=(1,), name="y_size")
        input4 = Input(shape=(1,), name="y_local")
        
        x = Concatenate()([input1, input2, input3, input4])
        x = Dense(17, activation="relu")(x)
        x = Dense(20, activation="relu")(x)
        x = Dense(9, activation="relu")(x)
        x = Dense(16, activation="relu")(x)
        x = Dense(8, activation="relu")(x)
        output = Dense(1, activation="sigmoid")(x)
        
        models['model1'] = Model(inputs=[input1, input2, input3, input4], outputs=output)
        
    except Exception as e:
        print(f"Warning: Could not create Model1 mock: {e}")
        models['model1'] = None
    
    try:
        # Model2: CNN Architecture (multi-input with profiles)
        from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
        
        x_profile_input = Input(shape=(21,), name="x_profile")
        z_global_input = Input(shape=(1,), name="z_global")
        y_profile_input = Input(shape=(13,), name="y_profile")
        y_local_input = Input(shape=(1,), name="y_local")
        
        # x_profile + z_global branch
        xz_concat = Concatenate()([x_profile_input, z_global_input])
        xz_dense = Dense(128, activation="relu")(xz_concat)
        
        # y_profile + y_local branch
        yl_concat = Concatenate()([y_profile_input, y_local_input])
        yl_dense = Dense(128, activation="relu")(yl_concat)
        
        # Merged features
        merged = Concatenate()([xz_dense, yl_dense])
        merged_dense = Dense(256, activation="relu")(merged)
        merged_dense = Dropout(0.2)(merged_dense)
        merged_dense = Dense(128, activation="relu")(merged_dense)
        merged_dense = Dense(64, activation="relu")(merged_dense)
        output = Dense(1, activation="sigmoid")(merged_dense)
        
        models['model2'] = Model(inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], outputs=output)
        
    except Exception as e:
        print(f"Warning: Could not create Model2 mock: {e}")
        models['model2'] = None
    
    try:
        # Model3: Combined CNN + MLP (more complex architecture)
        from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, MaxPooling2D, Flatten, Reshape
        
        # CNN branch for cluster data
        cluster_input = Input(shape=(13, 21, 1), name="cluster")
        conv1 = Conv2D(32, (3, 5), activation="relu")(cluster_input)
        pool1 = MaxPooling2D((2, 2))(conv1)
        flat1 = Flatten()(pool1)
        dense1 = Dense(200, activation="relu")(flat1)
        
        # MLP branch for scalar features
        z_input = Input(shape=(1,), name="z_global")
        y_input = Input(shape=(1,), name="y_local")
        scalar_concat = Concatenate()([z_input, y_input])
        dense2 = Dense(16, activation="relu")(scalar_concat)
        dense3 = Dense(32, activation="relu")(dense2)
        
        # Combine branches
        combined = Concatenate()([dense1, dense3])
        final_dense = Dense(100, activation="relu")(combined)
        output = Dense(1, activation="sigmoid")(final_dense)
        
        models['model3'] = Model(inputs=[cluster_input, z_input, y_input], outputs=output)
        
    except Exception as e:
        print(f"Warning: Could not create Model3 mock: {e}")
        models['model3'] = None
    
    return models

def print_model_summary():
    """Print comprehensive model summary"""
    
    print("=" * 80)
    print("🧠 SMART PIXELS ML MODEL SUMMARY REPORT")
    print("=" * 80)
    print("📊 MULTI-BIT QUANTIZED MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Load metrics
    metrics = load_model_metrics()
    models = create_mock_models()
    
    # Model information
    model_info = {
        'model1': {
            'name': 'Model1 (MLP)',
            'architecture': '4 inputs → [17→20→9→16→8] → 1 output',
            'type': 'Multilayer Perceptron'
        },
        'model2': {
            'name': 'Model2 (CNN)', 
            'architecture': 'Multi-input CNN with feature fusion',
            'type': 'Convolutional Neural Network'
        },
        'model3': {
            'name': 'Model3 (Hybrid)',
            'architecture': 'Combined CNN + MLP with optimization',
            'type': 'Hybrid CNN-MLP'
        }
    }
    
    for model_key in ['model1', 'model2', 'model3']:
        print(f"\n🔹 {model_info[model_key]['name']}")
        print(f"   Architecture: {model_info[model_key]['architecture']}")
        print(f"   Type: {model_info[model_key]['type']}")
        
        # Model parameters and size
        if models[model_key] is not None:
            total_params = models[model_key].count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in models[model_key].trainable_weights])
            non_trainable_params = total_params - trainable_params
            
            # Model sizes
            fp32_size = calculate_model_size_kb(models[model_key])
            quantized_2bit_size = calculate_quantized_size_kb(total_params, 2)
            quantized_3bit_size = calculate_quantized_size_kb(total_params, 3)
            quantized_4bit_size = calculate_quantized_size_kb(total_params, 4)
            quantized_6bit_size = calculate_quantized_size_kb(total_params, 6)
            
            # Calculate percentage reductions
            reduction_2bit = ((fp32_size - quantized_2bit_size) / fp32_size) * 100
            reduction_3bit = ((fp32_size - quantized_3bit_size) / fp32_size) * 100
            reduction_4bit = ((fp32_size - quantized_4bit_size) / fp32_size) * 100
            reduction_6bit = ((fp32_size - quantized_6bit_size) / fp32_size) * 100
            
            print(f"   📏 Total Parameters: {total_params:,}")
            print(f"   📏 Trainable Parameters: {trainable_params:,}")
            print(f"   📏 Non-trainable Parameters: {non_trainable_params:,}")
            print(f"   💾 FP32 Model Size: {fp32_size:.2f} KB")
            print(f"   💾 Quantized Sizes:")
            print(f"      • 2-bit: {quantized_2bit_size:.1f} KB (🗜️ {fp32_size/quantized_2bit_size:.1f}x compression, {reduction_2bit:.1f}% reduction)")
            print(f"      • 3-bit: {quantized_3bit_size:.1f} KB (🗜️ {fp32_size/quantized_3bit_size:.1f}x compression, {reduction_3bit:.1f}% reduction)")
            print(f"      • 4-bit: {quantized_4bit_size:.1f} KB (🗜️ {fp32_size/quantized_4bit_size:.1f}x compression, {reduction_4bit:.1f}% reduction)")
            print(f"      • 6-bit: {quantized_6bit_size:.1f} KB (🗜️ {fp32_size/quantized_6bit_size:.1f}x compression, {reduction_6bit:.1f}% reduction)")
        else:
            print(f"   ⚠️  Could not calculate model parameters")
        
        # Performance metrics for 6-bit quantized model
        metric_key_6bit = f"{model_key}_6bit"
        metric_key_nonquant = f"{model_key}_nonquant"
        
        if metric_key_6bit in metrics:
            data = metrics[metric_key_6bit]
            
            # Extract metrics
            if 'avg_test_accuracy' in data:  # Model1/Model2 format
                accuracy = data['avg_test_accuracy']
                roc_auc = data['avg_roc_auc']
                test_loss = data['avg_test_loss']
                trials = data['n_trials']
            else:  # Model3 format
                accuracy = data['test_accuracy']
                roc_auc = data['roc_auc']
                test_loss = data['test_loss']
                trials = 1
            
                        # Try to calculate real confusion matrix metrics
            model_path = get_model_path_for_evaluation(model_key, '6bit')
            real_metrics = None
            if model_path and model_path.exists():
                real_metrics = calculate_real_confusion_metrics(model_path, model_key)
            
            print(f"   📈 6-bit Performance (averaged over {trials} trial{'s' if trials > 1 else ''}):") 
            print(f"      • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)") 
            print(f"      • ROC AUC: {roc_auc:.4f}")
            print(f"      • Test Loss: {test_loss:.4f}")
            
            if real_metrics:
                # Use real confusion matrix metrics
                signal_eff = real_metrics['signal_efficiency']
                bg_rejection = real_metrics['background_rejection']
                print(f"      • Signal Efficiency (Real): {signal_eff:.4f} ({signal_eff*100:.2f}%)")
                print(f"      • Background Rejection (Real): {bg_rejection:.4f} ({bg_rejection*100:.2f}%)")
                print(f"      • Confusion Matrix: TP={real_metrics['tp']}, FP={real_metrics['fp']}, TN={real_metrics['tn']}, FN={real_metrics['fn']}")
                print(f"      • False Positive Rate: {real_metrics['false_positive_rate']:.4f} ({real_metrics['false_positive_rate']*100:.2f}%)")
            else:
                # Fall back to approximations
                signal_eff, bg_rejection = calculate_signal_efficiency_background_rejection(accuracy, roc_auc)
                print(f"      • Signal Efficiency (Approx): {signal_eff:.4f} ({signal_eff*100:.2f}%)")
                print(f"      • Background Rejection (Approx): {bg_rejection:.4f} ({bg_rejection*100:.2f}%)")
                print(f"      • Note: Using approximations - could not load model for exact calculation")
            
            # Performance vs non-quantized
            if metric_key_nonquant in metrics:
                nonquant_data = metrics[metric_key_nonquant]
                if 'avg_test_accuracy' in nonquant_data:
                    nonquant_acc = nonquant_data['avg_test_accuracy']
                    nonquant_auc = nonquant_data['avg_roc_auc']
                else:
                    nonquant_acc = nonquant_data['test_accuracy']
                    nonquant_auc = nonquant_data['roc_auc']
                
                acc_retention = (accuracy / nonquant_acc) * 100 if nonquant_acc > 0 else 0
                auc_retention = (roc_auc / nonquant_auc) * 100 if nonquant_auc > 0 else 0
                
                print(f"      • Accuracy Retention: {acc_retention:.1f}% vs FP32")
                print(f"      • ROC AUC Retention: {auc_retention:.1f}% vs FP32")
            
        else:
            print(f"   ⚠️  6-bit performance metrics not found")
        
        print("-" * 80)
    
    # Summary comparison table
    print("\n📋 COMPARATIVE SUMMARY TABLE")
    print("-" * 110)
    print(f"{'Model':<15} {'Params':<15} {'2bit Size':<12} {'3bit Size':<12} {'4bit Size':<12} {'6bit Size':<12} {'Accuracy':<12} {'ROC AUC':<12}")
    print("-" * 110)
    
    for model_key in ['model1', 'model2', 'model3']:
        model_name = model_info[model_key]['name'].split(' ')[0]
        
        # Parameters
        if models[model_key] is not None:
            params = f"{models[model_key].count_params():,}"
            size_2bit = f"{calculate_quantized_size_kb(models[model_key].count_params(), 2):.1f}KB"
            size_3bit = f"{calculate_quantized_size_kb(models[model_key].count_params(), 3):.1f}KB"
            size_4bit = f"{calculate_quantized_size_kb(models[model_key].count_params(), 4):.1f}KB"
            size_6bit = f"{calculate_quantized_size_kb(models[model_key].count_params(), 6):.1f}KB"
        else:
            params = "N/A"
            size_2bit = "N/A"
            size_3bit = "N/A"
            size_4bit = "N/A"
            size_6bit = "N/A"
        
        # Performance
        metric_key = f"{model_key}_6bit"
        if metric_key in metrics:
            data = metrics[metric_key]
            if 'avg_test_accuracy' in data:
                accuracy = f"{data['avg_test_accuracy']:.3f}"
                roc_auc = f"{data['avg_roc_auc']:.3f}"
            else:
                accuracy = f"{data['test_accuracy']:.3f}"
                roc_auc = f"{data['roc_auc']:.3f}"
        else:
            accuracy = "N/A"
            roc_auc = "N/A"
        
        print(f"{model_name:<15} {params:<15} {size_2bit:<12} {size_3bit:<12} {size_4bit:<12} {size_6bit:<12} {accuracy:<12} {roc_auc:<12}")
    
    print("-" * 110)
    
    # Size reduction percentage table
    print("\n📊 SIZE REDUCTION PERCENTAGES")
    print("-" * 80)
    print(f"{'Model':<15} {'FP32 Size':<12} {'2-bit Red%':<12} {'3-bit Red%':<12} {'4-bit Red%':<12} {'6-bit Red%':<12}")
    print("-" * 80)
    
    for model_key in ['model1', 'model2', 'model3']:
        model_name = model_info[model_key]['name'].split(' ')[0]
        
        if models[model_key] is not None:
            fp32_size = calculate_model_size_kb(models[model_key])
            total_params = models[model_key].count_params()
            
            # Calculate percentage reductions
            reduction_2bit = ((fp32_size - calculate_quantized_size_kb(total_params, 2)) / fp32_size) * 100
            reduction_3bit = ((fp32_size - calculate_quantized_size_kb(total_params, 3)) / fp32_size) * 100
            reduction_4bit = ((fp32_size - calculate_quantized_size_kb(total_params, 4)) / fp32_size) * 100
            reduction_6bit = ((fp32_size - calculate_quantized_size_kb(total_params, 6)) / fp32_size) * 100
            
            fp32_str = f"{fp32_size:.1f}KB"
            red_2bit_str = f"{reduction_2bit:.1f}%"
            red_3bit_str = f"{reduction_3bit:.1f}%"
            red_4bit_str = f"{reduction_4bit:.1f}%"
            red_6bit_str = f"{reduction_6bit:.1f}%"
        else:
            fp32_str = "N/A"
            red_2bit_str = "N/A"
            red_3bit_str = "N/A"
            red_4bit_str = "N/A"
            red_6bit_str = "N/A"
        
        print(f"{model_name:<15} {fp32_str:<12} {red_2bit_str:<12} {red_3bit_str:<12} {red_4bit_str:<12} {red_6bit_str:<12}")
    
    print("-" * 80)
    
    # Real vs Approximated metrics comparison
    print("\n🎆 REAL vs APPROXIMATED METRICS COMPARISON")
    print("-" * 80)
    print(f"{'Model':<15} {'Real Sig Eff':<15} {'Approx Sig Eff':<15} {'Real Bg Rej':<15} {'Approx Bg Rej':<15}")
    print("-" * 80)
    
    for model_key in ['model1', 'model2', 'model3']:
        model_name = model_info[model_key]['name'].split(' ')[0]
        metric_key = f"{model_key}_6bit"
        
        if metric_key in metrics:
            data = metrics[metric_key]
            if 'avg_test_accuracy' in data:
                accuracy = data['avg_test_accuracy']
                roc_auc = data['avg_roc_auc']
            else:
                accuracy = data['test_accuracy']
                roc_auc = data['roc_auc']
            
            # Get approximated values
            approx_sig_eff, approx_bg_rej = calculate_signal_efficiency_background_rejection(accuracy, roc_auc)
            
            # Try to get real values
            model_path = get_model_path_for_evaluation(model_key, '6bit')
            real_metrics = None
            if model_path and model_path.exists():
                real_metrics = calculate_real_confusion_metrics(model_path, model_key)
            
            if real_metrics:
                real_sig_str = f"{real_metrics['signal_efficiency']:.3f}"
                real_bg_str = f"{real_metrics['background_rejection']:.3f}"
            else:
                real_sig_str = "N/A"
                real_bg_str = "N/A"
                
            approx_sig_str = f"{approx_sig_eff:.3f}"
            approx_bg_str = f"{approx_bg_rej:.3f}"
            
            print(f"{model_name:<15} {real_sig_str:<15} {approx_sig_str:<15} {real_bg_str:<15} {approx_bg_str:<15}")
        else:
            print(f"{model_name:<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("-" * 80)
    print("\n🏆 WINNER ANALYSIS:")
    
    # Find best performing model
    best_accuracy = 0
    best_auc = 0
    best_acc_model = ""
    best_auc_model = ""
    
    for model_key in ['model1', 'model2', 'model3']:
        metric_key = f"{model_key}_6bit"
        if metric_key in metrics:
            data = metrics[metric_key]
            if 'avg_test_accuracy' in data:
                acc = data['avg_test_accuracy']
                auc = data['avg_roc_auc']
            else:
                acc = data['test_accuracy']
                auc = data['roc_auc']
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_acc_model = model_info[model_key]['name']
            
            if auc > best_auc:
                best_auc = auc
                best_auc_model = model_info[model_key]['name']
    
    print(f"   🥇 Best Accuracy: {best_acc_model} ({best_accuracy:.4f})")
    print(f"   🥇 Best ROC AUC: {best_auc_model} ({best_auc:.4f})")
    
    # Find most efficient model across all bit widths
    most_efficient_2bit = ""
    most_efficient_3bit = ""
    most_efficient_4bit = ""
    most_efficient_6bit = ""
    smallest_2bit = float('inf')
    smallest_3bit = float('inf')
    smallest_4bit = float('inf')
    smallest_6bit = float('inf')
    
    for model_key in ['model1', 'model2', 'model3']:
        if models[model_key] is not None:
            size_2bit = calculate_quantized_size_kb(models[model_key].count_params(), 2)
            size_3bit = calculate_quantized_size_kb(models[model_key].count_params(), 3)
            size_4bit = calculate_quantized_size_kb(models[model_key].count_params(), 4)
            size_6bit = calculate_quantized_size_kb(models[model_key].count_params(), 6)
            
            if size_2bit < smallest_2bit:
                smallest_2bit = size_2bit
                most_efficient_2bit = model_info[model_key]['name']
            if size_3bit < smallest_3bit:
                smallest_3bit = size_3bit
                most_efficient_3bit = model_info[model_key]['name']
            if size_4bit < smallest_4bit:
                smallest_4bit = size_4bit
                most_efficient_4bit = model_info[model_key]['name']
            if size_6bit < smallest_6bit:
                smallest_6bit = size_6bit
                most_efficient_6bit = model_info[model_key]['name']
    
    print(f"   🥇 Most Efficient Models by Quantization:")
    if most_efficient_2bit:
        print(f"      • 2-bit: {most_efficient_2bit} ({smallest_2bit:.1f} KB)")
    if most_efficient_3bit:
        print(f"      • 3-bit: {most_efficient_3bit} ({smallest_3bit:.1f} KB)")
    if most_efficient_4bit:
        print(f"      • 4-bit: {most_efficient_4bit} ({smallest_4bit:.1f} KB)")
    if most_efficient_6bit:
        print(f"      • 6-bit: {most_efficient_6bit} ({smallest_6bit:.1f} KB)")
    
    print("\n" + "=" * 80)
    print("✅ MODEL SUMMARY REPORT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    print_model_summary()