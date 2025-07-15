import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import sys

# --- Import the correct data generator for each dataset ---
# All are in the parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import OptimizedDataGenerator4 as ODG
import OptimizedDataGenerator4_shuffled as ODG_BATCH
import OptimizedDataGenerator4_data_shuffled as ODG_DATA

# --- Model definition (fixed hyperparameters) ---
def build_model():
    # Use best HPs from notebook or reasonable defaults
    f1 = 32
    k_rows = 7
    k_cols = 3
    z_units = 32
    y_units = 32
    head_units = 96
    drop_rate = 0.4
    initial_lr = 0.001
    end_lr = 1e-5
    power = 1.0
    decay_steps = 100 * 30  # 100 epochs * ~30 batches
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=power
    )
    optimizer = Adam(learning_rate=lr_schedule)
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
    model = Model([vol_input, z_input, y_input], output, name="rect_kernel_conv2d_with_y")
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- Utility to train and evaluate on a dataset ---
def train_and_evaluate(name, ODG_class, base_dir, results_dir, epochs=100):
    print(f"\n=== Training on {name} dataset ===")
    train_dir = Path(base_dir) / "tfrecords_train"
    val_dir   = Path(base_dir) / "tfrecords_validation"
    # Data generators
    train_gen = ODG_class(
        load_records=True,
        tf_records_dir=str(train_dir),
        x_feature_description=['cluster', 'y_local', 'z_global']
    )
    val_gen = ODG_class(
        load_records=True,
        tf_records_dir=str(val_dir),
        x_feature_description=['cluster', 'y_local', 'z_global']
    )
    # Squeeze singleton time dimension if present
    class SqueezedGenerator(tf.keras.utils.Sequence):
        def __init__(self, base_gen):
            self.base_gen = base_gen
        def __len__(self):
            return len(self.base_gen)
        def __getitem__(self, idx):
            X, y = self.base_gen[idx]
            if X['cluster'].ndim == 4 and X['cluster'].shape[1] == 1:
                X['cluster'] = X['cluster'][:, 0, :, :]
            return X, y
    train_gen = SqueezedGenerator(train_gen)
    val_gen = SqueezedGenerator(val_gen)
    # Print shape of first batch for sanity check
    X_batch, y_batch = train_gen[0]
    print(f"First batch 'cluster' shape: {X_batch['cluster'].shape}")
    print(f"First batch 'y_local' shape: {X_batch['y_local'].shape}")
    print(f"First batch 'z_global' shape: {X_batch['z_global'].shape}")
    print(f"First batch labels shape: {y_batch.shape}")
    # Model
    model = build_model()
    callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    # Final accuracy
    train_acc = history.history['accuracy'][-1]
    val_acc   = history.history['val_accuracy'][-1]
    print(f"Final training accuracy:   {train_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    # ROC curve (validation set)
    y_score = model.predict(val_gen, verbose=0).ravel()
    y_true  = np.concatenate([y for _, y in (val_gen[i] for i in range(len(val_gen)))])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)
    # Save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({name})')
    plt.legend(loc='lower right')
    plt.grid(True)
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'roc_{name}.png'))
    plt.close()
    return train_acc, val_acc, roc_auc

if __name__ == "__main__":
    # Dataset directories
    base_dirs = {
        'unshuffled':      str(Path(__file__).resolve().parent.parent / 'filtering_models' / 'filtering_records2048test'),
        'batch_shuffled':  str(Path(__file__).resolve().parent.parent / 'filtering_models' / 'filtering_records2048_shuffled'),
        'data_shuffled':   str(Path(__file__).resolve().parent.parent / 'filtering_models' / 'filtering_records2048_data_shuffled'),
    }
    ODG_classes = {
        'unshuffled':     ODG.OptimizedDataGenerator,
        'batch_shuffled': ODG_BATCH.OptimizedDataGeneratorShuffled,
        'data_shuffled':  ODG_DATA.OptimizedDataGeneratorDataShuffled,
    }
    results_dir = str(Path(__file__).resolve().parent / 'shuffling_results')
    results = {}
    for name in ['unshuffled', 'batch_shuffled', 'data_shuffled']:
        train_acc, val_acc, roc_auc = train_and_evaluate(
            name,
            ODG_classes[name],
            base_dirs[name],
            results_dir,
            epochs=100
        )
        results[name] = {'train_acc': train_acc, 'val_acc': val_acc, 'roc_auc': roc_auc}
    # Print summary
    print("\n=== Summary ===")
    for name in results:
        print(f"{name:15s} | Train Acc: {results[name]['train_acc']:.4f} | Val Acc: {results[name]['val_acc']:.4f} | ROC AUC: {results[name]['roc_auc']:.4f}")
    print(f"ROC curves saved in: {results_dir}") 