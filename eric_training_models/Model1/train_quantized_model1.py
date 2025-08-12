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
from quantized_mlp_model import build_quantized_mlp_model, build_non_quantized_mlp_model

# Import data generator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import OptimizedDataGenerator4 as ODG

class QuantizedModel1Trainer:
    """
    Class to train both non-quantized and quantized MLP models with multiple trials.
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
        self.results_dir = Path(__file__).resolve().parent / f"quantized_model1_results_{timestamp}"
        self.results_dir.mkdir(exist_ok=True)
        print(f"Created results directory: {self.results_dir}")
        
        # Learning rate schedule config
        self.config = {
            "kind": "polynomial_decay",
            "initial_lr": 1e-3,
            "end_lr": 1e-5,
            "power": 2,
        }
        
        # Fractional bit configurations (weight_bits, integer_bits)
        # Testing different fractional bit values with 0 integer bits
        self.fractional_bits = [2, 3, 4, 6, 8, 16, 32]
        self.bit_settings = [(bits, 0) for bits in self.fractional_bits]  # integer_bits always 0
        
        # Data generators - Model1 uses z_global, x_size, y_size, y_local
        self.train_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(self.train_dir),
            x_feature_description=['z_global', 'x_size', 'y_size', 'y_local']
        )
        self.val_gen = ODG.OptimizedDataGenerator(
            load_records=True,
            tf_records_dir=str(self.val_dir),
            x_feature_description=['z_global', 'x_size', 'y_size', 'y_local']
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
        print(f"\n=== Training Non-Quantized MLP Model ===")
        print(f"Number of trials: {self.n_trials}")
        
        # Create output directory
        output_dir = self.results_dir / "non_quantized_model"
        output_dir.mkdir(exist_ok=True)
        
        # Storage for this configuration's trials
        trial_accuracies = []
        trial_val_accuracies = []
        trial_losses = []
        trial_val_losses = []
        individual_results = []
        all_trial_roc_data = []
        
        for trial in range(self.n_trials):
            print(f"\n--- Trial {trial+1}/{self.n_trials} ---")
            
            # Build model
            model = build_non_quantized_mlp_model()
            optimizer = self.setup_learning_rate_schedule()
            
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
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
            trial_accuracies.append(history.history['binary_accuracy'])
            trial_val_accuracies.append(history.history['val_binary_accuracy'])
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
                'model_type': 'non_quantized',
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'roc_auc': float(roc_auc),
                'epochs_trained': len(history.history['binary_accuracy'])
            }
            individual_results.append(trial_result)
            
            # Save model for this trial
            trial_dir = output_dir / f"trial_{trial+1}"
            trial_dir.mkdir(exist_ok=True)
            
            model.save(trial_dir / f'non_quantized_mlp_trial{trial+1}.h5')
            model.save(trial_dir / f'non_quantized_mlp_trial{trial+1}.keras')
            
            # Save trial history
            trial_history = {
                'accuracy': history.history['binary_accuracy'],
                'val_accuracy': history.history['val_binary_accuracy'],
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }
            
            with open(trial_dir / f'trial_{trial+1}_history.json', 'w') as f:
                json.dump(trial_history, f, indent=2)
            
            # Save trial evaluation results
            with open(trial_dir / f'trial_{trial+1}_evaluation.json', 'w') as f:
                json.dump(trial_result, f, indent=2)
            
            # Store ROC data from this trial
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
        np.savez(output_dir / f'non_quantized_model_history.npz', 
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
            'individual_results': individual_results
        }
        
        with open(output_dir / "overall_results.json", 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        print(f"\n✓ Non-quantized model training completed!")
        print(f"  Average test accuracy: {avg_test_accuracy:.4f}")
        print(f"  Average test loss: {avg_test_loss:.4f}")
        print(f"  Average ROC AUC: {avg_roc_auc:.4f}")
    
    def train_quantized_model(self, weight_bits, weight_integer_bits, activation_bits, activation_integer_bits):
        """Train quantized model with given bit configurations"""
        config_key = f"quantized_{weight_bits}w{weight_integer_bits}i_{activation_bits}a{activation_integer_bits}i"
        print(f"\n=== Training Quantized Model: {config_key} ===")
        print(f"Weight bits: {weight_bits}, Weight integer bits: {weight_integer_bits}")
        print(f"Activation bits: {activation_bits}, Activation integer bits: {activation_integer_bits}")
        print(f"Number of trials: {self.n_trials}")
        
        # Create output directory
        output_dir = self.results_dir / config_key
        output_dir.mkdir(exist_ok=True)
        
        # Storage for this configuration's trials
        trial_accuracies = []
        trial_val_accuracies = []
        trial_losses = []
        trial_val_losses = []
        individual_results = []
        all_trial_roc_data = []
        
        for trial in range(self.n_trials):
            print(f"\n--- Trial {trial+1}/{self.n_trials} ---")
            
            # Build model
            model = build_quantized_mlp_model(
                weight_bits=weight_bits,
                weight_integer_bits=weight_integer_bits,
                weight_alpha=1,
                activation_bits=activation_bits,
                activation_integer_bits=activation_integer_bits
            )
            
            optimizer = self.setup_learning_rate_schedule()
            
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["binary_accuracy"],
                run_eagerly=True
            )
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
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
            trial_accuracies.append(history.history['binary_accuracy'])
            trial_val_accuracies.append(history.history['val_binary_accuracy'])
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
                'epochs_trained': len(history.history['binary_accuracy'])
            }
            individual_results.append(trial_result)
            
            # Save model for this trial
            trial_dir = output_dir / f"trial_{trial+1}"
            trial_dir.mkdir(exist_ok=True)
            
            model.save(trial_dir / f'quantized_mlp_{config_key}_trial{trial+1}.h5')
            model.save(trial_dir / f'quantized_mlp_{config_key}_trial{trial+1}.keras')
            
            # Save trial history
            trial_history = {
                'accuracy': history.history['binary_accuracy'],
                'val_accuracy': history.history['val_binary_accuracy'],
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
        np.savez(output_dir / f'{config_key}_history.npz', 
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
            'individual_results': individual_results
        }
        
        with open(output_dir / "overall_results.json", 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        print(f"\n✓ Quantized model {config_key} training completed!")
        print(f"  Average test accuracy: {avg_test_accuracy:.4f}")
        print(f"  Average test loss: {avg_test_loss:.4f}")
        print(f"  Average ROC AUC: {avg_roc_auc:.4f}")
    
    def average_roc_curves(self, trial_roc_data):
        """Average ROC curves across trials using interpolation"""
        # Common FPR points for interpolation
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        
        for roc_data in trial_roc_data:
            # Interpolate TPR at common FPR points
            interp_tpr = np.interp(mean_fpr, roc_data['fpr'], roc_data['tpr'])
            interp_tpr[0] = 0.0  # Ensure ROC starts at (0,0)
            tprs.append(interp_tpr)
            aucs.append(roc_data['auc'])
        
        # Calculate mean and std
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0  # Ensure ROC ends at (1,1)
        mean_auc = np.mean(aucs)
        
        return {
            'fpr': mean_fpr,
            'tpr': mean_tpr,
            'auc': mean_auc
        }
    
    def run_all_experiments(self):
        """Run training for all model configurations"""
        print("=== Starting Model1 Training Experiments ===")
        print(f"Total configurations to train: {len(self.bit_settings) + 1}")  # +1 for non-quantized
        
        # Train non-quantized model first
        self.train_non_quantized_model()
        
        # Train quantized models with different bit configurations
        for weight_bits, weight_integer_bits in self.bit_settings:
            # For Model1, use same bits for activations as weights
            activation_bits = weight_bits
            activation_integer_bits = weight_integer_bits
            
            self.train_quantized_model(
                weight_bits=weight_bits,
                weight_integer_bits=weight_integer_bits,
                activation_bits=activation_bits,
                activation_integer_bits=activation_integer_bits
            )
        
        # Save consolidated ROC data
        with open(self.results_dir / "all_roc_data.json", 'w') as f:
            json.dump(self.roc_data, f, indent=2)
        
        print(f"\n=== All Model1 Training Experiments Completed! ===")
        print(f"Results saved to: {self.results_dir}")
        print(f"Total models trained: {len(self.bit_settings) + 1}")
        print("To generate plots, run:")
        print(f"python plot_model1_results.py {self.results_dir.name}")

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Model1 with various quantization settings')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs per trial')
    parser.add_argument('--trials', type=int, default=6, help='Number of trials per configuration')
    
    args = parser.parse_args()
    
    # Create trainer and run experiments
    trainer = QuantizedModel1Trainer(n_epochs=args.epochs, n_trials=args.trials)
    trainer.run_all_experiments()

if __name__ == "__main__":
    main()