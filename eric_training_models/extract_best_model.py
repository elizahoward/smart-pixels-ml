import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Reshape, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.models import Model
import kerastuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import sys

# Add parent directory to path for data generator import
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)
import OptimizedDataGenerator4 as ODG

def model_builder(hp):
    """Model builder function - must match the one used in hyperparameter tuning"""
    # ── A) Polynomial‐decay hyperparams ─────────────────────────────────────
    initial_lr = hp.Float("initial_lr", 1e-4, 1e-2, sampling="log")
    end_lr     = hp.Float("end_lr",     1e-6, 1e-4, sampling="log")
    power      = hp.Float("poly_power", 0.5, 2.5, step=0.5)

    decay_steps = len(train_gen) * 200
    lr_schedule = PolynomialDecay(
        initial_learning_rate = initial_lr,
        decay_steps           = decay_steps,
        end_learning_rate     = end_lr,
        power                 = power
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # ── B) Architecture hyperparams ──────────────────────────────────────────
    f1           = hp.Int("conv_filters", 16, 64, step=16)
    k_rows       = hp.Choice("kernel_rows", [3, 5, 7, 9])
    k_cols       = hp.Choice("kernel_cols", [3, 5, 7, 9])
    z_units      = hp.Int("z_units",      8, 64, step=8)
    y_units      = hp.Int("y_units",      8, 64, step=8)
    head_units   = hp.Int("head_units",  32,256, step=64)
    drop_rate    = hp.Float("dropout",     0.0, 0.8, step=0.1)

    # ── C) Inputs ─────────────────────────────────────────────────────────────
    vol_input = Input(shape=(13, 21), name="cluster")
    z_input   = Input(shape=(1,),     name="z_global")
    y_input   = Input(shape=(1,),     name="y_local")

    # ── D) Conv2D branch with rectangular kernel ─────────────────────────────
    x = Reshape((13, 21, 1), name="add_channel")(vol_input)
    x = Conv2D(
        filters=f1,
        kernel_size=(k_rows, k_cols),
        padding="same",
        activation="relu",
        name=f"conv2d_{k_rows}x{k_cols}"
    )(x)
    x = MaxPooling2D((2, 2), name="pool2d_1")(x)
    x = Flatten(name="flatten_vol")(x)

    # ── E) Scalar branches ───────────────────────────────────────────────────
    z_dense = Dense(z_units, activation="relu", name="dense_z")(z_input)
    y_dense = Dense(y_units, activation="relu", name="dense_y")(y_input)

    # ── F) Merge & head ──────────────────────────────────────────────────────
    merged = Concatenate(name="concat_all")([x, z_dense, y_dense])
    h = Dense(head_units, activation="relu", name="head_dense1")(merged)
    h = Dropout(drop_rate,     name="head_dropout")(h)
    h = Dense(head_units // 2, activation="relu", name="head_dense2")(h)

    output = Dense(1, activation="sigmoid", name="output")(h)

    model = Model(
        [vol_input, z_input, y_input],
        output,
        name="rect_kernel_conv2d_with_y"
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def setup_data_generators():
    """Setup data generators for training"""
    BASE_DIR        = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
    TRAIN_DIR       = BASE_DIR / "tfrecords_train"
    VALIDATION_DIR  = BASE_DIR / "tfrecords_validation"

    train_gen = ODG.OptimizedDataGenerator(
        load_records           = True,
        tf_records_dir         = str(TRAIN_DIR),
        x_feature_description  = ['cluster', 'y_local','z_global']
    )

    val_gen = ODG.OptimizedDataGenerator(
        load_records           = True,
        tf_records_dir         = str(VALIDATION_DIR),
        x_feature_description  = ['cluster','y_local','z_global']
    )
    
    return train_gen, val_gen

def load_tuner_results(tuner_dir="ktuner_dir/conv3d_zsearch"):
    """Load the tuner and get best results"""
    tuner = kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=120,
        executions_per_trial=2,
        directory="ktuner_dir",
        project_name="conv3d_zsearch"
    )
    
    # Get best hyperparameters and model
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    
    return tuner, best_hp, best_model

def analyze_tuning_results(tuner_dir="ktuner_dir/conv3d_zsearch"):
    """Analyze all trial results and find the best performing trial"""
    oracle_path = Path(tuner_dir) / "oracle.json"
    
    with open(oracle_path, 'r') as f:
        oracle_data = json.load(f)
    
    # Get all trial results
    trials = []
    for trial_id in oracle_data['end_order']:
        trial_path = Path(tuner_dir) / f"trial_{trial_id:03d}" / "trial.json"
        
        if trial_path.exists():
            with open(trial_path, 'r') as f:
                trial_data = json.load(f)
                
            # Extract metrics
            metrics = trial_data.get('metrics', {}).get('metrics', {})
            val_accuracy = metrics.get('val_accuracy', {}).get('observations', [{}])[0].get('value', [0])[0]
            val_loss = metrics.get('val_loss', {}).get('observations', [{}])[0].get('value', [float('inf')])[0]
            train_accuracy = metrics.get('accuracy', {}).get('observations', [{}])[0].get('value', [0])[0]
            train_loss = metrics.get('loss', {}).get('observations', [{}])[0].get('value', [float('inf')])[0]
            
            # Get hyperparameters
            hp_values = trial_data.get('hyperparameters', {}).get('values', {})
            
            trials.append({
                'trial_id': trial_id,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'train_loss': train_loss,
                'hyperparameters': hp_values,
                'score': trial_data.get('score', 0)
            })
    
    # Sort by validation accuracy
    trials.sort(key=lambda x: x['val_accuracy'], reverse=True)
    
    return trials

def extract_best_model(save_dir="best_model_extracted", retrain=False, epochs=120):
    """Extract the best model from tuning results"""
    
    print("🔍 Loading tuner results...")
    tuner, best_hp, best_model = load_tuner_results()
    
    print("📊 Analyzing all trial results...")
    all_trials = analyze_tuning_results()
    
    # Get the best trial
    best_trial = all_trials[0]
    
    print(f"\n🏆 Best Trial Results:")
    print(f"Trial ID: {best_trial['trial_id']}")
    print(f"Validation Accuracy: {best_trial['val_accuracy']:.4f}")
    print(f"Validation Loss: {best_trial['val_loss']:.4f}")
    print(f"Training Accuracy: {best_trial['train_accuracy']:.4f}")
    print(f"Training Loss: {best_trial['train_loss']:.4f}")
    
    print(f"\n🔧 Best Hyperparameters:")
    for hp_name, hp_value in best_trial['hyperparameters'].items():
        print(f"  • {hp_name}: {hp_value}")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Save best hyperparameters
    with open(save_path / "best_hyperparameters.json", 'w') as f:
        json.dump(best_trial['hyperparameters'], f, indent=2)
    
    # Save trial analysis
    with open(save_path / "all_trials_analysis.json", 'w') as f:
        json.dump(all_trials, f, indent=2)
    
    if retrain:
        print(f"\n🔄 Retraining best model with fresh weights...")
        
        # Setup data generators
        global train_gen, val_gen
        train_gen, val_gen = setup_data_generators()
        
        # Build fresh model with best hyperparameters
        fresh_model = model_builder(best_hp)
        
        # Train the fresh model
        history = fresh_model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
            ],
            verbose=2
        )
        
        # Save the fresh model
        fresh_model.save(save_path / "best_model_fresh.h5")
        
        # Evaluate final performance
        val_loss, val_acc = fresh_model.evaluate(val_gen, verbose=0)
        print(f"\n📈 Final Performance:")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save training history
        np.savez(save_path / "training_history.npz",
                 accuracy=history.history['accuracy'],
                 val_accuracy=history.history['val_accuracy'],
                 loss=history.history['loss'],
                 val_loss=history.history['val_loss'])
        
        return fresh_model, history
    else:
        # Save the best model from tuning
        best_model.save(save_path / "best_model_tuned.h5")
        print(f"\n💾 Best model saved to: {save_path / 'best_model_tuned.h5'}")
        return best_model, None

def plot_trial_analysis(all_trials, save_dir="best_model_extracted"):
    """Plot analysis of all trials"""
    import matplotlib.pyplot as plt
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Extract data for plotting
    trial_ids = [int(trial['trial_id']) for trial in all_trials]
    val_accuracies = [trial['val_accuracy'] for trial in all_trials]
    val_losses = [trial['val_loss'] for trial in all_trials]
    
    # Plot validation accuracy vs trial
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(trial_ids, val_accuracies, alpha=0.6)
    plt.xlabel('Trial ID')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy by Trial')
    plt.grid(True, alpha=0.3)
    
    # Highlight best trial
    best_trial_id = int(all_trials[0]['trial_id'])
    best_val_acc = all_trials[0]['val_accuracy']
    plt.scatter(best_trial_id, best_val_acc, color='red', s=100, zorder=5, label=f'Best: {best_val_acc:.4f}')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(trial_ids, val_losses, alpha=0.6)
    plt.xlabel('Trial ID')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss by Trial')
    plt.grid(True, alpha=0.3)
    
    # Highlight best trial
    best_val_loss = all_trials[0]['val_loss']
    plt.scatter(best_trial_id, best_val_loss, color='red', s=100, zorder=5, label=f'Best: {best_val_loss:.4f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / "trial_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Trial analysis plot saved to: {save_path / 'trial_analysis.png'}")

if __name__ == "__main__":
    print("🚀 Extracting best model from hyperparameter tuning results...")
    
    # Extract best model and optionally retrain
    best_model, history = extract_best_model(
        save_dir="best_model_extracted",
        retrain=True,  # Set to False if you don't want to retrain
        epochs=120
    )
    
    # Analyze all trials and create plots
    all_trials = analyze_tuning_results()
    plot_trial_analysis(all_trials)
    
    print("\n✅ Extraction complete! Check the 'best_model_extracted' directory for results.") 