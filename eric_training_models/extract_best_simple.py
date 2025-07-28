import json
import kerastuner as kt
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Reshape, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.models import Model

def model_builder(hp):
    """Model builder function - must match the one used in hyperparameter tuning"""
    # Learning rate hyperparameters
    initial_lr = hp.Float("initial_lr", 1e-4, 1e-2, sampling="log")
    end_lr     = hp.Float("end_lr",     1e-6, 1e-4, sampling="log")
    power      = hp.Float("poly_power", 0.5, 2.5, step=0.5)

    # Use a default decay_steps (will be overridden when actually training)
    decay_steps = 6000  # Default value
    lr_schedule = PolynomialDecay(
        initial_learning_rate = initial_lr,
        decay_steps           = decay_steps,
        end_learning_rate     = end_lr,
        power                 = power
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # Architecture hyperparameters
    f1           = hp.Int("conv_filters", 16, 64, step=16)
    k_rows       = hp.Choice("kernel_rows", [3, 5, 7, 9])
    k_cols       = hp.Choice("kernel_cols", [3, 5, 7, 9])
    z_units      = hp.Int("z_units",      8, 64, step=8)
    y_units      = hp.Int("y_units",      8, 64, step=8)
    head_units   = hp.Int("head_units",  32,256, step=64)
    drop_rate    = hp.Float("dropout",     0.0, 0.8, step=0.1)

    # Inputs
    vol_input = Input(shape=(13, 21), name="cluster")
    z_input   = Input(shape=(1,),     name="z_global")
    y_input   = Input(shape=(1,),     name="y_local")

    # Conv2D branch
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

    # Scalar branches
    z_dense = Dense(z_units, activation="relu", name="dense_z")(z_input)
    y_dense = Dense(y_units, activation="relu", name="dense_y")(y_input)

    # Merge & head
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

def get_best_trial_info(tuner_dir="ktuner_dir/conv3d_zsearch"):
    """Get information about the best trial"""
    oracle_path = Path(tuner_dir) / "oracle.json"
    
    with open(oracle_path, 'r') as f:
        oracle_data = json.load(f)
    
    # Find the best trial by analyzing all trials
    best_trial_id = None
    best_val_accuracy = 0
    
    for trial_id in oracle_data['end_order']:
        trial_path = Path(tuner_dir) / f"trial_{trial_id:03d}" / "trial.json"
        
        if trial_path.exists():
            with open(trial_path, 'r') as f:
                trial_data = json.load(f)
                
            # Get validation accuracy
            metrics = trial_data.get('metrics', {}).get('metrics', {})
            val_accuracy = metrics.get('val_accuracy', {}).get('observations', [{}])[0].get('value', [0])[0]
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_trial_id = trial_id
    
    return best_trial_id, best_val_accuracy

def extract_best_model(save_dir="best_model", tuner_dir="ktuner_dir/conv3d_zsearch"):
    """Extract the best model from the tuning results"""
    
    print("🔍 Loading tuner...")
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
    
    # Get best trial info
    best_trial_id, best_val_accuracy = get_best_trial_info(tuner_dir)
    
    print(f"\n🏆 Best Trial Results:")
    print(f"Trial ID: {best_trial_id}")
    print(f"Validation Accuracy: {best_val_accuracy:.4f}")
    
    print(f"\n🔧 Best Hyperparameters:")
    for hp_name in ["initial_lr", "end_lr", "poly_power", "conv_filters", 
                   "kernel_rows", "kernel_cols", "z_units", "y_units", 
                   "head_units", "dropout"]:
        value = best_hp.get(hp_name)
        print(f"  • {hp_name}: {value}")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Save the best model
    model_save_path = save_path / "best_model.h5"
    best_model.save(str(model_save_path))
    
    # Save hyperparameters
    hp_save_path = save_path / "best_hyperparameters.json"
    hp_dict = {hp.name: hp.value for hp in best_hp.space}
    with open(hp_save_path, 'w') as f:
        json.dump(hp_dict, f, indent=2)
    
    print(f"\n💾 Model saved to: {model_save_path}")
    print(f"📋 Hyperparameters saved to: {hp_save_path}")
    
    return best_model, best_hp

if __name__ == "__main__":
    print("🚀 Extracting best model from hyperparameter tuning...")
    
    best_model, best_hp = extract_best_model()
    
    print("\n✅ Extraction complete!")
    print("You can now use the best model for inference or further training.") 