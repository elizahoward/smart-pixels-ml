#!/usr/bin/env python3
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import OptimizedDataGenerator4 as ODG  # ← your class!

# 1) Set up paths
HERE       = Path(__file__).parent
DATA_DIR   = HERE / "Data"
GEN_OUT    = HERE / "debug_samples_og"
GEN_OUT.mkdir(exist_ok=True)

# 2) Instantiate your real generator (this will write its own .tfrecords there)
gen = ODG.OptimizedDataGenerator(
    data_directory_path=str(DATA_DIR) + os.sep,
    file_count=1,                    # just process 1 file
    x_feature_description="all",     # grab every allowed feature
    tf_records_dir=str(GEN_OUT),     # where its TFRecords go
)

# 3) Pull out the very first sample that the generator built in memory:
#    gen.x_features is a dict feature_name → numpy array of shape (N, …)
sample_feats = {
    feat: gen.x_features[feat][0]
    for feat in gen.x_feature_description
}
sample_label = gen.labels[0]  # shape (…,) or scalar

# 4) Write a single‐row Parquet from that sample
#    (flatten arrays to Python lists so Pandas+PyArrow will accept them)
flat = {
    k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
    for k, v in sample_feats.items()
}
df = pd.DataFrame([flat])
df.to_parquet(GEN_OUT / "sample_from_ODG.parquet", engine="pyarrow")
print("Wrote sample_from_ODG.parquet")

# 5) Write a single‐example TFRecord from that sample
def _bytes_feature(x):
    t = tf.convert_to_tensor(x, dtype=tf.float32)
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[tf.io.serialize_tensor(t).numpy()]
    ))

feature_proto = {
    name: _bytes_feature(val)
    for name, val in sample_feats.items()
}
feature_proto["y"] = _bytes_feature(sample_label)

example = tf.train.Example(features=tf.train.Features(feature=feature_proto))
with tf.io.TFRecordWriter(str(GEN_OUT / "sample_from_ODG.tfrecord")) as w:
    w.write(example.SerializeToString())
print("Wrote sample_from_ODG.tfrecord")

print(f"\nAll outputs in `{GEN_OUT}`:")
for f in os.listdir(GEN_OUT):
    print(" ", f)
