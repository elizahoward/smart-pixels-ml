import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Conv2D, MaxPooling2D, Flatten, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import json
from datetime import datetime
import os
import sys
from pathlib import Path

# Add parent directory to path for data generator import
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)
import OptimizedDataGenerator4 as ODG

class CNNHyperModel(kt.HyperModel):
    """
    HyperModel class for CNN with Keras Tuner.
    """
    
    def __init__(self, cluster_shape=(13, 21, 20), z_global_length=1, y_local_length=1):
        self.cluster_shape = cluster_shape
        self.z_global_length = z_global_length
        self.y_local_length = y_local_length
    
    def build(self, hp):
        """
        Build the CNN model with hyperparameters from Keras Tuner.
        Based on the notebook structure with polynomial decay learning rate.
        
        Args:
            hp: Keras Tuner HyperParameters object
        """
        # ── A) Polynomial‐decay hyperparams ─────────────────────────────────────
        initial_lr = hp.Float("initial_lr", 1e-4, 1e-2, sampling="log")
        end_lr     = hp.Float("end_lr",     1e-6, 1e-4, sampling="log")
        power      = hp.Float("poly_power", 0.5, 2.5, step=0.5)

        # Get data generator length for decay steps calculation
        # We'll need to pass this from outside or calculate it
        decay_steps = 30 * 200  # Based on notebook: len(train_gen) * 200
        lr_schedule = PolynomialDecay(
            initial_learning_rate = initial_lr,
            decay_steps           = decay_steps,
            end_learning_rate     = end_lr,
            power                 = power
        )
        optimizer = Adam(learning_rate=lr_schedule)

        # ── B) Architecture hyperparams ──────────────────────────────────────────
        f1           = hp.Int("conv_filters", 16, 64, step=16)
        # separately tune rows and cols
        k_rows       = hp.Choice("kernel_rows", [3, 5, 7, 9])
        k_cols       = hp.Choice("kernel_cols", [3, 5, 7, 9])
        z_units      = hp.Int("z_units",      16, 64, step=16)
        y_units      = hp.Int("y_units",      16, 64, step=16)
        head_units   = hp.Int("head_units",  32, 256, step=64)
        drop_rate    = hp.Float("dropout",     0.0, 0.4, step=0.4)

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

def run_keras_tuner_search(train_gen, val_gen, max_trials=120, epochs=110, executions_per_trial=2):
    """
    Run hyperparameter search using Keras Tuner.
    Based on the notebook structure.
    
    Args:
        train_gen: Training data generator
        val_gen: Validation data generator
        max_trials (int): Maximum number of trials
        epochs (int): Number of training epochs per trial
        executions_per_trial (int): Number of executions per trial
    
    Returns:
        kt.Tuner: The tuner object with results
    """
    # Create the hypermodel
    hypermodel = CNNHyperModel()
    
    # Create the tuner using RandomSearch like in the notebook
    tuner = kt.RandomSearch(
        hypermodel,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="ktuner_dir",
        project_name="conv3d_zsearch"
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    print(f"Starting Keras Tuner search with {max_trials} trials...")
    print("Hyperparameters being tuned:")
    print("- Learning rate: Polynomial decay (initial: 1e-4 to 1e-2, end: 1e-6 to 1e-4)")
    print("- Conv2D filters: 16-64 (step 16)")
    print("- Kernel rows/cols: 3, 5, 7, 9")
    print("- Z units: 8-64 (step 8)")
    print("- Y units: 8-64 (step 8)")
    print("- Head units: 32-256 (step 64)")
    print("- Dropout: 0.0-0.8 (step 0.1)")
    
    # Run the search
    tuner.search(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return tuner

def save_tuner_results(tuner, filename=None):
    """
    Save Keras Tuner results to JSON file.
    
    Args:
        tuner: Keras Tuner object
        filename (str): Optional filename
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"keras_tuner_results_{timestamp}.json"
    
    # Get the best trial
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    
    # Get all trials
    trials = tuner.oracle.get_best_trials(num_trials=len(tuner.oracle.trials))
    
    results = []
    for trial in trials:
        trial_result = {
            'trial_id': trial.trial_id,
            'score': trial.score,
            'hyperparameters': trial.hyperparameters.values,
            'status': trial.status
        }
        results.append(trial_result)
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")
    return results

def get_best_hyperparameters(tuner):
    """
    Get the best hyperparameters from the tuner.
    
    Args:
        tuner: Keras Tuner object
    
    Returns:
        dict: Best hyperparameters and performance
    """
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    
    print(f"\nBest trial found:")
    print(f"Trial ID: {best_trial.trial_id}")
    print(f"Best validation accuracy: {best_trial.score:.4f}")
    print(f"Best hyperparameters:")
    for param, value in best_trial.hyperparameters.values.items():
        print(f"  {param}: {value}")
    
    return {
        'trial_id': best_trial.trial_id,
        'best_accuracy': best_trial.score,
        'hyperparameters': best_trial.hyperparameters.values
    }

def setup_data_generators():
    """
    Setup data generators using the same structure as the notebook.
    """
    # ─── 2) Setup data generators ────────────────────────────────────────────────
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

def main():
    """
    Main function to run Keras Tuner hyperparameter search.
    Based on the notebook structure.
    """
    print("Keras Tuner Hyperparameter Search for CNN Model")
    print("=" * 60)
    
    # Setup data generators
    print("Setting up data generators...")
    train_gen, val_gen = setup_data_generators()
    
    print(f"Training generator length: {len(train_gen)}")
    print(f"Validation generator length: {len(val_gen)}")
    
    # Run Keras Tuner search
    tuner = run_keras_tuner_search(
        train_gen, val_gen,
        max_trials=80,  # From notebook
        epochs=100,      # From notebook
        executions_per_trial=2  # From notebook
    )
    
    # Save results
    results = save_tuner_results(tuner)
    
    # Get best hyperparameters
    best_params = get_best_hyperparameters(tuner)
    
    print("\n" + "="*60)
    print("KERAS TUNER SEARCH COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_params['best_accuracy']:.4f}")
    print(f"Best trial ID: {best_params['trial_id']}")
    
    return tuner, best_params

if __name__ == "__main__":
    tuner, best_params = main() 