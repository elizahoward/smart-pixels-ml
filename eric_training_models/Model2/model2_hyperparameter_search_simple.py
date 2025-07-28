import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import os
import sys
from pathlib import Path

# Add parent directory to path for data generator import
parentdir = os.path.dirname(os.getcwd())
sys.path.insert(0, parentdir)
import OptimizedDataGenerator4 as ODG

# ─── 1) Setup data generators ────────────────────────────────────────────────
BASE_DIR        = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
TRAIN_DIR       = BASE_DIR / "tfrecords_train"
VALIDATION_DIR  = BASE_DIR / "tfrecords_validation"

train_gen = ODG.OptimizedDataGenerator(
    load_records           = True,
    tf_records_dir         = str(TRAIN_DIR),
    x_feature_description  = ['x_profile', 'z_global', 'y_profile', 'y_local']
)

val_gen = ODG.OptimizedDataGenerator(
    load_records           = True,
    tf_records_dir         = str(VALIDATION_DIR),
    x_feature_description  = ['x_profile', 'z_global', 'y_profile', 'y_local']
)

def model_builder(hp):
    """
    Model2 hyperparameter builder function.
    Tunes layer units, dropout rates, and learning rate schedule.
    """
    # ── A) Learning rate hyperparameters ─────────────────────────────────────
    initial_lr = hp.Float("initial_lr", 1e-4, 1e-2, sampling="log")
    end_lr     = hp.Float("end_lr",     1e-6, 1e-4, sampling="log")
    power      = hp.Float("poly_power", 0.5, 2.5, step=0.5)

    decay_steps = len(train_gen) * 120  # Based on Model2 training epochs
    lr_schedule = PolynomialDecay(
        initial_learning_rate = initial_lr,
        decay_steps           = decay_steps,
        end_learning_rate     = end_lr,
        power                 = power
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # ── B) Architecture hyperparameters ──────────────────────────────────────
    # xz branch units (x_profile + z_global)
    xz_units = hp.Int("xz_units", 16, 128, step=16)
    
    # yl branch units (y_profile + y_local)
    yl_units = hp.Int("yl_units", 16, 128, step=16)
    
    # Merged dense layers
    merged_dense1_units = hp.Int("merged_dense1_units", 64, 256, step=64)
    merged_dense2_units = hp.Int("merged_dense2_units", 32, 128, step=32)
    merged_dense3_units = hp.Int("merged_dense3_units", 16, 64, step=16)
    
    # Dropout rates
    dropout1_rate = hp.Float("dropout1_rate", 0.0, 0.5, step=0.1)
    dropout2_rate = hp.Float("dropout2_rate", 0.0, 0.5, step=0.1)
    dropout3_rate = hp.Float("dropout3_rate", 0.0, 0.5, step=0.1)

    # ── C) Inputs ───────────────────────────────────────────────────────────
    x_profile_input = Input(shape=(21,), name="x_profile")
    z_global_input = Input(shape=(1,), name="z_global")
    y_profile_input = Input(shape=(13,), name="y_profile")
    y_local_input = Input(shape=(1,), name="y_local")

    # ── D) x_profile + z_global branch ─────────────────────────────────────
    xz_concat = Concatenate(name="xz_concat")([x_profile_input, z_global_input])
    xz_dense = Dense(xz_units, activation="relu", name="xz_dense1")(xz_concat)

    # ── E) y_profile + y_local branch ──────────────────────────────────────
    yl_concat = Concatenate(name="yl_concat")([y_profile_input, y_local_input])
    yl_dense = Dense(yl_units, activation="relu", name="yl_dense1")(yl_concat)

    # ── F) Concatenate and dense layers ────────────────────────────────────
    merged = Concatenate(name="merged_features")([xz_dense, yl_dense])
    
    merged_dense = Dense(merged_dense1_units, activation="relu", name="merged_dense1")(merged)
    merged_dense = Dropout(dropout1_rate, name="dropout1")(merged_dense)
    
    merged_dense = Dense(merged_dense2_units, activation="relu", name="merged_dense2")(merged_dense)
    merged_dense = Dropout(dropout2_rate, name="dropout2")(merged_dense)
    
    merged_dense = Dense(merged_dense3_units, activation="relu", name="merged_dense3")(merged_dense)
    merged_dense = Dropout(dropout3_rate, name="dropout3")(merged_dense)

    # ── G) Output layer for binary classification ───────────────────────────
    output = Dense(1, activation="sigmoid", name="output")(merged_dense)

    model = Model(
        inputs=[x_profile_input, z_global_input, y_profile_input, y_local_input], 
        outputs=output, 
        name="model2_hyperparameter_tuned"
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    """
    Main function to run hyperparameter search for Model2.
    """
    print("Model2 Hyperparameter Search")
    print("=" * 50)
    
    print(f"Training generator length: {len(train_gen)}")
    print(f"Validation generator length: {len(val_gen)}")
    
    # ─── 2) Instantiate the Tuner ──────────────────────────────────────────
    tuner = kt.RandomSearch(
        model_builder,
        objective           = "val_accuracy",
        max_trials          = 70,  # As requested
        executions_per_trial= 2,
        directory           = "ktuner_dir",
        project_name        = "model2_hyperparameter_search"
    )
    
    print("Starting hyperparameter search...")
    print("Hyperparameters being tuned:")
    print("- Learning rate: Polynomial decay (initial: 1e-4 to 1e-2, end: 1e-6 to 1e-4)")
    print("- XZ branch units: 16-128 (step 16)")
    print("- YL branch units: 16-128 (step 16)")
    print("- Merged dense layers: 64-256, 32-128, 16-64 units")
    print("- Dropout rates: 0.0-0.5 (step 0.1)")
    
    # ─── 3) Run the hyperparameter search ──────────────────────────────────
    tuner.search(
        train_gen,
        validation_data = val_gen,
        epochs          = 120,  # From Model2 training
        callbacks       = [
            EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
        ]
    )
    
    # ─── 4) Retrieve best hyperparameters and model ────────────────────────
    best_hp    = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    
    print("\n" + "="*50)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*50)
    
    best_model.summary()
    
    print("\n🔎 Best hyperparameters:")
    for hp_name in ["initial_lr", "end_lr", "poly_power",
                    "xz_units", "yl_units", 
                    "merged_dense1_units", "merged_dense2_units", "merged_dense3_units",
                    "dropout1_rate", "dropout2_rate", "dropout3_rate"]:
        print(f" • {hp_name}: {best_hp.get(hp_name)}")
    
    return tuner, best_hp, best_model

if __name__ == "__main__":
    tuner, best_hp, best_model = main() 