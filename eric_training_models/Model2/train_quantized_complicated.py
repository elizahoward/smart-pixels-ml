import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from qkeras.quantizers import quantized_bits, quantized_relu

# Import models
from quantized_cnn_model import build_quantized_cnn_model
from cnn_model import build_custom_cnn_model

# Import data generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import OptimizedDataGenerator4 as ODG

class QuantizedModelTrainer:
    """
    Class to train both non-quantized and quantized versions with multiple trials.
    Organized output structure with individual folders for each model configuration.
    """
    
    def __init__(self, n_epochs=150, n_trials=6):
        self.n_epochs = n_epochs
        self.n_trials = n_trials
        
        # Data directories
        self.base_dir = Path("/local/d1/smartpixML/filtering_models/shuffling_data/filtering_records1024_data_shuffled_single")
        self.train_dir = self.base_dir / "tfrecords_train"
        self.val_dir = self.base_dir / "tfrecords_validation"
        
        # Create main results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(__file__).resolve().parent / f"quantized_complicated_results_{timestamp}"
        self.results_dir.mkdir(exist_ok=True)
        print(f"Created results directory: {self.results_dir}")
        
        # Learning rate schedule config
        self.config = {
            "kind": "polynomial_decay",
            "initial_lr": 1e-3,
            "end_lr": 1e-5,
            "power": 2,
        }
        
        # Fractional bit configurations (weight_bits, integer_bits=0)
        self.fractional_bits = [2, 3, 4, 6, 8, 16, 32]
        self.bit_settings = [(bits, 0) for bits in self.fractional_bits]  # integer_bits always 0
        
        # Data generators
        self.train_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(self.train_dir),
            x_feature_description=['x_profile', 'z_global', 'y_profile', 'y_local']
        )
        self.val_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(self.val_dir),
            x_feature_description=['x_profile', 'z_global', 'y_profile', 'y_local']
        )
        
        print(f"Training generator length: {len(self.train_gen)}")
        print(f"Validation generator length: {len(self.val_gen)}")
        
        # Storage for ROC curve data (averaged over all trials)
        self.roc_data = {}
        
    def setup_learning_rate_schedule(self):
        """Setup polynomial decay learning rate schedule"""
        steps_per_epoch = len(self.train_gen)
        decay_steps = steps_per_epoch * self.n_epochs
        
        if self.config["kind"] == "polynomial_decay":
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=self.config["initial_lr"],
                decay_steps=decay_steps,
                end_learning_rate=self.config["end_lr"],
                power=self.config["power"]
            )
            return tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            return tf.keras.optimizers.Adam(learning_rate=self.config["initial_lr"])
    
    def train_non_quantized_model(self):
        """Train the non-quantized model with multiple trials"""
        print(f"\n=== Training Non-Quantized Model ===")
        print(f"Number of trials: {self.n_trials}")
        
        # Create output directory
        output_dir = self.results_dir / "non_quantized_model"
        output_dir.mkdir(exist_ok=True)
        
        # Storage for this configuration's trials
        trial_accuracies = []
        trial_val_accuracies = []
        trial_losses = []
        trial_val_losses = []
        
        # Store individual trial results
        individual_results = []
        
        # Storage for ROC data from all trials
        all_trial_roc_data = []
        
        for trial in range(self.n_trials):
            print(f"  Trial {trial + 1}/{self.n_trials}")
            
            # Build and compile model
            model = build_custom_cnn_model(dropout_rate=0.2)
            optimizer = self.setup_learning_rate_schedule()
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=150,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = model.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=self.n_epochs,
                verbose=1,
                callbacks=callbacks
            )
            
            # Store results for this trial
            trial_accuracies.append(history.history['accuracy'])
            trial_val_accuracies.append(history.history['val_accuracy'])
            trial_losses.append(history.history['loss'])
            trial_val_losses.append(history.history['val_loss'])
            
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(self.val_gen, verbose=0)
            
            # Calculate ROC AUC
            val_preds = model.predict(self.val_gen, verbose=0).ravel()
            val_true = np.concatenate([y for _, y in (self.val_gen[i] for i in range(len(self.val_gen)))])
            fpr, tpr, thresholds = roc_curve(val_true, val_preds)
            roc_auc = auc(fpr, tpr)
            
            # Store individual trial result
            trial_result = {
                'trial': trial + 1,
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'roc_auc': float(roc_auc),
                'epochs_trained': len(history.history['accuracy'])
            }
            individual_results.append(trial_result)
            
            # Save model for this trial
            trial_dir = output_dir / f"trial_{trial+1}"
            trial_dir.mkdir(exist_ok=True)
            
            model.save(trial_dir / f'non_quantized_trial{trial+1}.h5')
            model.save(trial_dir / f'non_quantized_trial{trial+1}.keras')
            
            # Save trial history
            trial_history = {
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            with open(trial_dir / f'trial_{trial+1}_history.json', 'w') as f:
                json.dump(trial_history, f, indent=2)
            
            # Save trial evaluation results
            with open(trial_dir / f'trial_{trial+1}_evaluation.json', 'w') as f:
                json.dump(trial_result, f, indent=2)
            
            # Store ROC data from this trial (will be averaged later)
            all_trial_roc_data.append({
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            })
        
        # Calculate averaged results
        min_length = min(len(acc) for acc in trial_accuracies)
        
        # Truncate all histories to the minimum length
        trial_accuracies_truncated = [acc[:min_length] for acc in trial_accuracies]
        trial_val_accuracies_truncated = [acc[:min_length] for acc in trial_val_accuracies]
        trial_losses_truncated = [loss[:min_length] for loss in trial_losses]
        trial_val_losses_truncated = [loss[:min_length] for loss in trial_val_losses]
        
        avg_acc = np.mean(trial_accuracies_truncated, axis=0)
        avg_val_acc = np.mean(trial_val_accuracies_truncated, axis=0)
        avg_loss = np.mean(trial_losses_truncated, axis=0)
        avg_val_loss = np.mean(trial_val_losses_truncated, axis=0)
        
        # Save averaged history
        np.savez(output_dir / 'averaged_history.npz', 
                 accuracy=avg_acc, val_accuracy=avg_val_acc,
                 loss=avg_loss, val_loss=avg_val_loss)
        
        # Calculate average metrics
        avg_test_accuracy = np.mean([r['test_accuracy'] for r in individual_results])
        avg_test_loss = np.mean([r['test_loss'] for r in individual_results])
        avg_roc_auc = np.mean([r['roc_auc'] for r in individual_results])
        
        # Average ROC curves across all trials
        avg_roc_data = self.average_roc_curves(all_trial_roc_data)
        self.roc_data['non_quantized'] = {
            'fpr': avg_roc_data['fpr'].tolist(),
            'tpr': avg_roc_data['tpr'].tolist(),
            'auc': avg_roc_data['auc'],
            'model_name': 'Non-quantized'
        }
        
        # Save overall results
        overall_results = {
            'model_type': 'non_quantized',
            'n_trials': self.n_trials,
            'avg_test_accuracy': float(avg_test_accuracy),
            'avg_test_loss': float(avg_test_loss),
            'avg_roc_auc': float(avg_roc_auc),
            'final_train_accuracy': float(avg_acc[-1]),
            'final_val_accuracy': float(avg_val_acc[-1]),
            'final_train_loss': float(avg_loss[-1]),
            'final_val_loss': float(avg_val_loss[-1]),
            'epochs_trained': int(min_length),
            'individual_results': individual_results
        }
        
        with open(output_dir / 'overall_results.json', 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        print(f"Non-quantized model training completed!")
        print(f"Average test accuracy: {avg_test_accuracy:.4f}")
        print(f"Average ROC AUC: {avg_roc_auc:.4f}")
        
        return overall_results
    
    def train_quantized_model(self, weight_bits, weight_integer_bits):
        """Train a quantized model with specified bit configuration and multiple trials"""
        activation_bits = 8
        activation_integer_bits = 0
        
        print(f"\n=== Testing {weight_bits}-bit quantized model (fractional bits: {weight_bits}, integer bits: {weight_integer_bits}) ===")
        print(f"  Weight quantizer: quantized_bits({weight_bits}, {weight_integer_bits}, 1)")
        print(f"  Activation quantizer: quantized_relu({activation_bits}, {activation_integer_bits})")
        print(f"  Number of trials: {self.n_trials}")
        
        # Create output directory
        config_key = f"{weight_bits}bit_int{weight_integer_bits}"
        output_dir = self.results_dir / f"quantized_{config_key}"
        output_dir.mkdir(exist_ok=True)
        
        # Storage for this configuration's trials
        trial_accuracies = []
        trial_val_accuracies = []
        trial_losses = []
        trial_val_losses = []
        
        # Store individual trial results
        individual_results = []
        
        # Storage for ROC data from all trials
        all_trial_roc_data = []
        
        for trial in range(self.n_trials):
            print(f"  Trial {trial + 1}/{self.n_trials}")
            
            # Build and compile model
            model = build_quantized_cnn_model(
                weight_bits=weight_bits,
                weight_integer_bits=weight_integer_bits,
                activation_bits=activation_bits,
                activation_integer_bits=activation_integer_bits,
                dropout_rate=0.2
            )
            optimizer = self.setup_learning_rate_schedule()
            model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=150,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = model.fit(
                self.train_gen,
                validation_data=self.val_gen,
                epochs=self.n_epochs,
                verbose=1,
                callbacks=callbacks
            )
            
            # Store results for this trial
            trial_accuracies.append(history.history['accuracy'])
            trial_val_accuracies.append(history.history['val_accuracy'])
            trial_losses.append(history.history['loss'])
            trial_val_losses.append(history.history['val_loss'])
            
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(self.val_gen, verbose=0)
            
            # Calculate ROC AUC
            val_preds = model.predict(self.val_gen, verbose=0).ravel()
            val_true = np.concatenate([y for _, y in (self.val_gen[i] for i in range(len(self.val_gen)))])
            fpr, tpr, thresholds = roc_curve(val_true, val_preds)
            roc_auc = auc(fpr, tpr)
            
            # Store individual trial result
            trial_result = {
                'trial': trial + 1,
                'weight_bits': weight_bits,
                'weight_integer_bits': weight_integer_bits,
                'activation_bits': activation_bits,
                'activation_integer_bits': activation_integer_bits,
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'roc_auc': float(roc_auc),
                'epochs_trained': len(history.history['accuracy'])
            }
            individual_results.append(trial_result)
            
            # Save model for this trial
            trial_dir = output_dir / f"trial_{trial+1}"
            trial_dir.mkdir(exist_ok=True)
            
            model.save(trial_dir / f'quantized_cnn_{config_key}_trial{trial+1}.h5')
            model.save(trial_dir / f'quantized_cnn_{config_key}_trial{trial+1}.keras')
            
            # Save trial history
            trial_history = {
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            with open(trial_dir / f'trial_{trial+1}_history.json', 'w') as f:
                json.dump(trial_history, f, indent=2)
            
            # Save trial evaluation results
            with open(trial_dir / f'trial_{trial+1}_evaluation.json', 'w') as f:
                json.dump(trial_result, f, indent=2)
            
            # Store ROC data from this trial (will be averaged later)
            all_trial_roc_data.append({
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            })
        
        # Calculate averaged results
        min_length = min(len(acc) for acc in trial_accuracies)
        
        # Truncate all histories to the minimum length
        trial_accuracies_truncated = [acc[:min_length] for acc in trial_accuracies]
        trial_val_accuracies_truncated = [acc[:min_length] for acc in trial_val_accuracies]
        trial_losses_truncated = [loss[:min_length] for loss in trial_losses]
        trial_val_losses_truncated = [loss[:min_length] for loss in trial_val_losses]
        
        avg_acc = np.mean(trial_accuracies_truncated, axis=0)
        avg_val_acc = np.mean(trial_val_accuracies_truncated, axis=0)
        avg_loss = np.mean(trial_losses_truncated, axis=0)
        avg_val_loss = np.mean(trial_val_losses_truncated, axis=0)
        
        # Save averaged history
        np.savez(output_dir / f'quantized_cnn_{config_key}_history.npz', 
                 accuracy=avg_acc, val_accuracy=avg_val_acc,
                 loss=avg_loss, val_loss=avg_val_loss)
        
        # Calculate average metrics
        avg_test_accuracy = np.mean([r['test_accuracy'] for r in individual_results])
        avg_test_loss = np.mean([r['test_loss'] for r in individual_results])
        avg_roc_auc = np.mean([r['roc_auc'] for r in individual_results])
        
        # Average ROC curves across all trials
        avg_roc_data = self.average_roc_curves(all_trial_roc_data)
        self.roc_data[config_key] = {
            'fpr': avg_roc_data['fpr'].tolist(),
            'tpr': avg_roc_data['tpr'].tolist(),
            'auc': avg_roc_data['auc'],
            'model_name': f'{weight_bits}-bit'
        }
        
        # Save overall results
        overall_results = {
            'model_type': 'quantized',
            'weight_bits': weight_bits,
            'weight_integer_bits': weight_integer_bits,
            'activation_bits': activation_bits,
            'activation_integer_bits': activation_integer_bits,
            'n_trials': self.n_trials,
            'avg_test_accuracy': float(avg_test_accuracy),
            'avg_test_loss': float(avg_test_loss),
            'avg_roc_auc': float(avg_roc_auc),
            'final_train_accuracy': float(avg_acc[-1]),
            'final_val_accuracy': float(avg_val_acc[-1]),
            'final_train_loss': float(avg_loss[-1]),
            'final_val_loss': float(avg_val_loss[-1]),
            'epochs_trained': int(min_length),
            'individual_results': individual_results
        }
        
        with open(output_dir / 'overall_results.json', 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        print(f"{weight_bits}-bit quantized model training completed!")
        print(f"Average test accuracy: {avg_test_accuracy:.4f}")
        print(f"Average ROC AUC: {avg_roc_auc:.4f}")
        
        return overall_results
    
    def average_roc_curves(self, all_trial_roc_data):
        """
        Average ROC curves across multiple trials by interpolating to common FPR points.
        """
        if not all_trial_roc_data:
            return None
        
        # Define common FPR points for interpolation
        common_fpr = np.linspace(0, 1, 100)
        
        # Interpolate all TPR curves to the common FPR points
        interpolated_tprs = []
        aucs = []
        
        for trial_data in all_trial_roc_data:
            fpr = trial_data['fpr']
            tpr = trial_data['tpr']
            auc_score = trial_data['auc']
            
            # Interpolate TPR to common FPR points
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interpolated_tprs.append(interp_tpr)
            aucs.append(auc_score)
        
        # Average the interpolated TPR curves
        mean_tpr = np.mean(interpolated_tprs, axis=0)
        mean_auc = np.mean(aucs)
        
        # Ensure the curve starts at (0,0) and ends at (1,1)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        
        return {
            'fpr': common_fpr,
            'tpr': mean_tpr,
            'auc': mean_auc
        }
    
    def plot_roc_curves_overlay(self):
        """Plot ROC curves overlay averaged across all trials of each configuration"""
        print("\nCreating ROC curves overlay...")
        
        plt.figure(figsize=(10, 10))
        
        # Define colors
        colors = ['black'] + [plt.cm.viridis(i/len(self.fractional_bits)) for i in range(len(self.fractional_bits))]
        
        # Plot each ROC curve
        for i, (config_key, roc_info) in enumerate(self.roc_data.items()):
            fpr = np.array(roc_info['fpr'])
            tpr = np.array(roc_info['tpr'])
            auc_score = roc_info['auc']
            model_name = roc_info['model_name']
            
            color = colors[0] if config_key == 'non_quantized' else colors[self.fractional_bits.index(int(config_key.split('bit')[0])) + 1]
            
            plt.plot(fpr, tpr, color=color, linewidth=2, 
                    label=f'{model_name} (AUC = {auc_score:.4f})')
        
        # Add diagonal line for random classifier
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison - All Models (Averaged Across Trials)', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Save the overlay plot
        overlay_file = self.results_dir / "roc_curves_overlay.png"
        plt.savefig(overlay_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves overlay saved to: {overlay_file}")
        
        # Save ROC data as JSON
        with open(self.results_dir / "roc_data.json", 'w') as f:
            json.dump(self.roc_data, f, indent=2)
        
        return overlay_file
    
    def create_summary_results(self, all_results):
        """Create a summary CSV of all results"""
        print("\nCreating summary results...")
        
        summary_data = []
        
        for result in all_results:
            summary_data.append({
                'model_type': result['model_type'],
                'fractional_bits': result.get('weight_bits', 'N/A'),
                'integer_bits': result.get('weight_integer_bits', 'N/A'),
                'n_trials': result['n_trials'],
                'avg_test_accuracy': result['avg_test_accuracy'],
                'avg_test_loss': result['avg_test_loss'],
                'avg_roc_auc': result['avg_roc_auc'],
                'final_train_accuracy': result['final_train_accuracy'],
                'final_val_accuracy': result['final_val_accuracy'],
                'final_train_loss': result['final_train_loss'],
                'final_val_loss': result['final_val_loss'],
                'epochs_trained': result['epochs_trained']
            })
        
        # Create DataFrame and save
        results_df = pd.DataFrame(summary_data)
        results_df.to_csv(self.results_dir / 'summary_results.csv', index=False)
        
        print(f"Summary results saved to: {self.results_dir / 'summary_results.csv'}")
        
        # Print summary table
        print("\n" + "=" * 100)
        print("FINAL RESULTS SUMMARY")
        print("=" * 100)
        print(f"{'Model Type':<15} {'Frac Bits':<10} {'Avg Acc':<10} {'Avg Loss':<12} {'Avg AUC':<10} {'Epochs':<8}")
        print("-" * 100)
        
        for result in all_results:
            model_type = result['model_type']
            frac_bits = result.get('weight_bits', 'N/A')
            avg_acc = result['avg_test_accuracy']
            avg_loss = result['avg_test_loss']
            avg_auc = result['avg_roc_auc']
            epochs = result['epochs_trained']
            
            print(f"{model_type:<15} {frac_bits:<10} {avg_acc:<10.4f} {avg_loss:<12.4f} {avg_auc:<10.4f} {epochs:<8}")
        
        # Find best configuration
        best_result = max(all_results, key=lambda x: x['avg_test_accuracy'])
        print(f"\nBEST CONFIGURATION (by accuracy):")
        if best_result['model_type'] == 'non_quantized':
            print(f"Model: Non-quantized")
        else:
            print(f"Model: {best_result['weight_bits']}-bit quantized")
        print(f"Average test accuracy: {best_result['avg_test_accuracy']:.4f}")
        print(f"Average ROC AUC: {best_result['avg_roc_auc']:.4f}")
        
        return results_df
    
    def run_full_training(self):
        """Run the complete training pipeline"""
        print(f"=" * 80)
        print(f"QUANTIZED MODEL TRAINING PIPELINE")
        print(f"=" * 80)
        print(f"Testing {len(self.bit_settings)} fractional bit configurations")
        print(f"Configurations: {self.fractional_bits} fractional bits (integer bits = 0)")
        print(f"Number of trials per configuration: {self.n_trials}")
        print(f"Number of epochs per trial: {self.n_epochs}")
        print(f"Results directory: {self.results_dir}")
        
        all_results = []
        
        # Train non-quantized model
        print(f"\n{'='*20} Training Non-quantized Model {'='*20}")
        non_quantized_result = self.train_non_quantized_model()
        all_results.append(non_quantized_result)
        
        # Train quantized models
        for weight_bits, weight_integer_bits in self.bit_settings:
            print(f"\n{'='*20} Training {weight_bits}-bit Quantized Model {'='*20}")
            quantized_result = self.train_quantized_model(weight_bits, weight_integer_bits)
            all_results.append(quantized_result)
        
        # Create ROC curves overlay
        self.plot_roc_curves_overlay()
        
        # Create summary results
        self.create_summary_results(all_results)
        
        print(f"\n" + "=" * 80)
        print("TRAINING PIPELINE COMPLETE!")
        print("=" * 80)
        print(f"All results saved to: {self.results_dir}")
        print("Files created:")
        print("- Individual model folders with .h5, .keras, and .json files")
        print("- summary_results.csv (overall metrics)")
        print("- roc_curves_overlay.png (ROC comparison)")
        print("- roc_data.json (ROC curve data)")
        
        return all_results


def main():
    """Main function to run the training pipeline"""
    # Initialize trainer with full training parameters (150 epochs, 6 trials)
    trainer = QuantizedModelTrainer(n_epochs=150, n_trials=6)
    
    # Run full training pipeline
    results = trainer.run_full_training()
    
    return results


if __name__ == "__main__":
    results = main()