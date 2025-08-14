#!/usr/bin/env python3
"""
Learning Rate Testing Script
Converts the testing_learnrate.ipynb notebook to a standalone Python script.
Tests different learning rate schedulers on the XY Profile Model.
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add parent directory to path to import OptimizedDataGenerator4
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import OptimizedDataGenerator4 as ODG

# ─── Configuration ─────────────────────────────────────────────────────────────
# Define paths directly instead of importing from config.py
BASE_DIR = Path("/home/youeric/PixelML/smart_pixels_ml/filtering_models/filtering_records2000")
TRAIN_DIR = BASE_DIR / "tfrecords_train"
VALIDATION_DIR = BASE_DIR / "tfrecords_validation"

# Alternative paths for different datasets (uncomment as needed)
# BASE_DIR = Path("/home/youeric/PixelML/smart_pixels_ml/shuffling_data/filtering_records1024_data_shuffled_single")
# TRAIN_DIR = BASE_DIR / "tfrecords_train"
# VALIDATION_DIR = BASE_DIR / "tfrecords_validation"

base_dir = BASE_DIR
train_dir = TRAIN_DIR
validation_dir = VALIDATION_DIR

# ─── GPU memory growth ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ─── Model definition ─────────────────────────────────────────────────────────
def CreateXYProfileModel():
    """Create the XY Profile Model for learning rate testing."""
    x_profile = tf.keras.Input(shape=(21, 1), name="x_profile")
    y_profile = tf.keras.Input(shape=(13, 1), name="y_profile")
    x_flat = tf.keras.layers.Flatten(name="flatten_x")(x_profile)
    y_flat = tf.keras.layers.Flatten(name="flatten_y")(y_profile)
    concat = tf.keras.layers.Concatenate(name="concat_xy")([x_flat, y_flat])
    hidden1 = tf.keras.layers.Dense(64, activation="relu", name="hidden_128")(concat)
    hidden2 = tf.keras.layers.Dense(16, activation="relu", name="hidden_32")(hidden1)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(hidden2)
    return tf.keras.Model(inputs=[x_profile, y_profile], outputs=output)

# ─── Data generators factory ───────────────────────────────────────────────────
def make_gens():
    """Create training and validation data generators."""
    train_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(TRAIN_DIR),
        x_feature_description=["x_profile", "y_profile"],
    )
    val_gen = ODG.OptimizedDataGenerator(
        load_records=True,
        tf_records_dir=str(VALIDATION_DIR),
        x_feature_description=["x_profile", "y_profile"],
    )
    return train_gen, val_gen

# ─── Single‐run trainer (returns full history) ────────────────────────────────
def train_and_evaluate(config):
    """Train and evaluate a model with the given configuration."""
    train_gen, val_gen = make_gens()
    steps = len(train_gen)
    epochs = 120

    model = CreateXYProfileModel()
    callbacks = [EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)]

    kind = config["type"]
    if kind == "constant":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
    elif kind == "cosine_decay":
        sched = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config["initial_lr"],
            decay_steps=steps * epochs,
            alpha=config.get("alpha", 0.0)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=sched)
    elif kind == "cosine_restarts":
        sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=config["initial_lr"],
            first_decay_steps=(steps * epochs) // config.get("restarts_divisor", 3),
            alpha=config.get("alpha", 0.0)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=sched)
    elif kind == "exponential_decay":
        sched = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["initial_lr"],
            decay_steps=(steps * epochs) // config.get("decay_divisor", 10),
            decay_rate=config.get("decay_rate", 0.96),
            staircase=config.get("staircase", True)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=sched)
    elif kind == "polynomial_decay":
        sched = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=config["initial_lr"],
            decay_steps=steps * epochs,
            end_learning_rate=config.get("end_lr", 1e-5),
            power=config.get("power", 1.0)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=sched)
    elif kind == "inverse_time_decay":
        sched = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=config["initial_lr"],
            decay_steps=(steps * epochs) // config.get("decay_divisor", 10),
            decay_rate=config.get("decay_rate", 1.0),
            staircase=config.get("staircase", True)
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=sched)
    elif kind == "piecewise":
        sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=config["boundaries"],
            values=config["values"]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=sched)
    elif kind == "reduce_on_plateau":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
        callbacks.append(ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.get("factor", 0.5),
            patience=config.get("patience", 10),
            verbose=1
        ))
    else:
        raise ValueError(f"Unknown scheduler type {kind!r}")

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        shuffle=False,
        verbose=1
    ).history

    return hist

# ─── Configurations including all schedulers ──────────────────────────────────
configs = [
    {"name": "const_1e-3", "type": "constant", "lr": 1e-3},
    {"name": "const_1e-4", "type": "constant", "lr": 1e-4},
    {"name": "const_1e-2", "type": "constant", "lr": 1e-2},
    {"name": "cosine_decay", "type": "cosine_decay", "initial_lr": 1e-3, "alpha": 0.0},
    {"name": "cosine_restarts", "type": "cosine_restarts", "initial_lr": 1e-3, "restarts_divisor": 3, "alpha": 0.0},
    {"name": "exp_decay", "type": "exponential_decay", "initial_lr": 1e-3, "decay_rate": 0.96, "decay_divisor": 10, "staircase": True},
    {"name": "poly_decay", "type": "polynomial_decay", "initial_lr": 1e-3, "end_lr": 1e-5, "power": 2.0},
    {"name": "inv_time_decay", "type": "inverse_time_decay", "initial_lr": 1e-3, "decay_rate": 1.0, "decay_divisor": 10, "staircase": True},
    {"name": "piecewise", "type": "piecewise", "boundaries": [3000, 6000], "values": [1e-3, 1e-4, 1e-5]},
    {"name": "reduce_plateau", "type": "reduce_on_plateau", "lr": 1e-3, "factor": 0.5, "patience": 10},
]

def main():
    """Main function to run the learning rate testing."""
    print(f"Using base directory: {BASE_DIR}")
    print(f"Training directory: {TRAIN_DIR}")
    print(f"Validation directory: {VALIDATION_DIR}")
    print(f"Number of configurations to test: {len(configs)}")
    print()

    # ─── Run sequentially and display progress ────────────────────────────────────
    histories = {}
    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== Running scheduler: {name} ===")
        sys.stdout.flush()
        try:
            histories[name] = train_and_evaluate(cfg)
            print(f"Completed: {name}")
        except Exception as e:
            print(f"Error running {name}: {e}")
            continue
        sys.stdout.flush()

    # ─── Plot all training & validation accuracy curves ───────────────────────────
    if histories:
        plt.figure(figsize=(12, 8))
        for name, h in histories.items():
            plt.plot(h['accuracy'], label=f'{name} train', linewidth=2)
            plt.plot(h['val_accuracy'], '--', label=f'{name} val', linewidth=2)
        
        plt.title('Training & Validation Accuracy by LR Scheduler', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='lower right', fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = Path(__file__).parent / "learning_rate_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        for name, h in histories.items():
            max_train_acc = max(h['accuracy'])
            max_val_acc = max(h['val_accuracy'])
            final_train_acc = h['accuracy'][-1]
            final_val_acc = h['val_accuracy'][-1]
            print(f"{name:15s}: Train max={max_train_acc:.4f}, final={final_train_acc:.4f} | "
                  f"Val max={max_val_acc:.4f}, final={final_val_acc:.4f}")
    else:
        print("No successful training runs to plot.")

if __name__ == "__main__":
    main()
