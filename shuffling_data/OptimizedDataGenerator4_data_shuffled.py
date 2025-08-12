# python imports
import tensorflow as tf
from qkeras import quantized_bits
from typing import Union, List, Tuple
import glob
import numpy as np
import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import logging
import gc
import traceback
from pathlib import Path
import random

# custom quantizer
def QKeras_data_prep_quantizer(data, bits=4, int_bits=0, alpha=1):
    quantizer = quantized_bits(bits, int_bits, alpha=alpha)
    return quantizer(data)

class OptimizedDataGeneratorDataShuffled(tf.keras.utils.Sequence):
    def __init__(self, 
            data_directory_path: str = "./",
            is_directory_recursive: bool = False,
            file_type: str = "parquet",
            data_format: str = "3D",
            batch_size: int = 32,
            file_count = None,
            labels_list: Union[List,str] = None,
            to_standardize: bool = False,
            normalization: Union[list,int] = 1,
            input_shape: Tuple = (1, 13, 21),
            transpose = None,
            files_from_end = False,
            tag: str = "",
            x_feature_description: Union[list,str] = ['cluster'],
            filteringBIB: bool = True,
            load_records: bool = False,
            tf_records_dir: str = None,
            time_stamps = [19],
            quantize: bool = False,
            max_workers: int = 1,
            shuffle_data: bool = True,
            random_seed: int = 42,
            ):
        super().__init__() 
        if tf_records_dir is None:
            raise ValueError(f"tf_records_dir is None")
        if shuffle_data:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.file_offsets = [0]
        allowed_features = ['cluster', 'x_profile', 'y_profile', 'x_size', 'y_size', 'y_local', 'z_global', 'total_charge', 'adjusted_hit_time', 'adjusted_hit_time_30ps_gaussian', 'adjusted_hit_time_60ps_gaussian']
        if isinstance(x_feature_description, str) and x_feature_description == "all":
            self.x_feature_description=allowed_features
        elif isinstance(x_feature_description, str):
            raise Exception("x_feature_description must be a list of features or 'all'")
        else:
            invalid_items = [item for item in x_feature_description if item not in allowed_features]
            if invalid_items:
                raise Exception(f"The following features are not allowed in x_feature_description: {invalid_items}\nAllowed features include: {allowed_features}")
            self.x_feature_description = x_feature_description
        if load_records:
            if not os.path.isdir(tf_records_dir):
                raise ValueError(f"Directory {tf_records_dir} does not exist.")
            else:
                self.tf_records_dir = tf_records_dir
        else:
            self.normalization = normalization
            self.time_stamps = np.arange(0,20) if time_stamps == -1 else time_stamps
            len_xy, ntime = 13*21, 20
            idx = [[i*(len_xy),(i+1)*(len_xy)] for i in range(ntime)]
            self.time_stamps = np.array([ np.arange(idx[i][0], idx[i][1]).astype("str") for i in self.time_stamps]).flatten().tolist()
            if time_stamps != -1 and data_format != '2D':
                assert len(time_stamps) == input_shape[0]
            self.max_workers = max_workers
            if file_type not in ["csv", "parquet"]:
                raise ValueError("file_type can only be \"csv\" or \"parquet\"!")
            self.file_type = file_type
            self.recon_files_bib = [
                f for f in glob.glob(
                    data_directory_path + "recon" + data_format + "bib*." + file_type, 
                    recursive=is_directory_recursive
                ) if tag in f
            ]
            self.recon_files_sig = [
                f for f in glob.glob(
                    data_directory_path + "recon" + data_format + "sig*." + file_type, 
                    recursive=is_directory_recursive
                ) if tag in f
            ]
            self.recon_files_sig.sort()
            self.recon_files_bib.sort()
            if file_count != None:
                if not files_from_end:
                    self.recon_files_bib_subset = self.recon_files_bib[:file_count]
                    self.recon_files_sig_subset = self.recon_files_sig[:file_count]
                else:
                    self.recon_files_bib_subset = self.recon_files_bib[-file_count:]
                    self.recon_files_sig_subset = self.recon_files_sig[-file_count:]
            else:
                self.recon_files_bib_subset = self.recon_files_bib
                self.recon_files_sig_subset = self.recon_files_sig
            all_files = self.recon_files_bib_subset + self.recon_files_sig_subset
            all_types = ['bib'] * len(self.recon_files_bib_subset) + ['sig'] * len(self.recon_files_sig_subset)
            print(f"Loading {len(all_files)} files (BIB: {len(self.recon_files_bib_subset)}, SIG: {len(self.recon_files_sig_subset)})")
            # Load all data
            labels_list = self.x_feature_description if labels_list is None else labels_list
            labels_df = pd.DataFrame()
            recon_df = pd.DataFrame()
            ylocal_df = pd.DataFrame()
            z_loc_df = pd.DataFrame()
            eh_pairs = pd.DataFrame()
            hit_time_df = pd.DataFrame()
            hit_time_30_df = pd.DataFrame()
            hit_time_60_df = pd.DataFrame()
            type_list = []
            for i, file in enumerate(all_files):
                file_type_indicator = all_types[i]
                tempDf = pd.read_parquet(file, columns=self.time_stamps)
                recon_df = pd.concat([recon_df,tempDf])
                type_list.extend([file_type_indicator]*tempDf.shape[0])
                file = file.replace(f"recon{data_format}","labels")
                if not filteringBIB: 
                    labels_df = pd.concat([labels_df,pd.read_parquet(file, columns=labels_list)])
                else:
                    if file_type_indicator == 'sig':
                        labels_df = pd.concat([labels_df, pd.DataFrame({'signal': [1] * tempDf.shape[0]})])
                    else:
                        labels_df = pd.concat([labels_df, pd.DataFrame({'signal': [0] * tempDf.shape[0]})])
                ylocal_df = pd.concat([ylocal_df,pd.read_parquet(file, columns=['y-local'])])
                eh_pairs = pd.concat([eh_pairs,pd.read_parquet(file, columns=['number_eh_pairs'])])
                z_loc_df = pd.concat([z_loc_df,pd.read_parquet(file, columns=['z-global'])])
                hit_time_df = pd.concat([hit_time_df,pd.read_parquet(file, columns=['adjusted_hit_time'])])
                hit_time_30_df = pd.concat([hit_time_30_df,pd.read_parquet(file, columns=['adjusted_hit_time_30ps_gaussian'])])
                hit_time_60_df = pd.concat([hit_time_60_df,pd.read_parquet(file, columns=['adjusted_hit_time_60ps_gaussian'])])
            # Remove NaNs
            has_nans = np.any(np.isnan(recon_df.values), axis=1)
            has_nans = np.arange(recon_df.shape[0])[has_nans]
            recon_df_raw = recon_df.drop(has_nans)
            labels_df_raw = labels_df.drop(has_nans)
            ylocal_df_raw = ylocal_df.drop(has_nans)
            z_loc_df_raw = z_loc_df.drop(has_nans)
            eh_pairs_raw = eh_pairs.drop(has_nans)
            hit_time = hit_time_df.drop(has_nans).values
            hit_time_30 = hit_time_30_df.drop(has_nans).values
            hit_time_60 = hit_time_60_df.drop(has_nans).values
            # Remove from type_list as well
            type_list = [t for i, t in enumerate(type_list) if i not in has_nans]
            self.dataPoints = len(labels_df_raw)
            recon_values = recon_df_raw.values    
            nonzeros = abs(recon_values) > 0
            recon_values[nonzeros] = np.sign(recon_values[nonzeros])*np.log1p(abs(recon_values[nonzeros]))/math.log(2)
            if to_standardize:
                recon_values[nonzeros] = self.standardize(recon_values[nonzeros])
            recon_values = recon_values.reshape((-1, *input_shape))            
            if transpose is not None:
                recon_values = recon_values.transpose(transpose)
            clusters = recon_values
            if isinstance(time_stamps, list) and len(time_stamps) == 1:
                clusters = recon_values.reshape((recon_values.shape[0],13,21))
            
            # Calculate profiles - sum across the appropriate axes
            if len(time_stamps) == 1:
                # Single timestamp case - sum across spatial dimensions
                y_profiles = np.sum(clusters, axis = 2)  # Sum across x dimension
                x_profiles = np.sum(clusters, axis = 1)  # Sum across y dimension
            else:
                # Multiple timestamps case - sum across time and spatial dimensions
                # Reshape to (batch_size, time_steps, y, x) for proper summing
                clusters_reshaped = clusters.reshape((clusters.shape[0], len(time_stamps), 13, 21))
                y_profiles = np.sum(clusters_reshaped, axis=(1, 3))  # Sum across time and x dimensions
                x_profiles = np.sum(clusters_reshaped, axis=(1, 2))  # Sum across time and y dimensions
            
            print(f"x_profiles.shape: {x_profiles.shape}")   # Debug print
            print(f"y_profiles.shape: {y_profiles.shape}")   # Debug print
            
            # Calculate sizes
            if len(time_stamps) == 1:
                bool_arr = x_profiles != 0
                x_sizes = np.sum(bool_arr, axis = 1)/21 
                bool_arr = y_profiles != 0
                y_sizes = np.sum(bool_arr, axis = 1)/13
            else:
                # For multiple timestamps, calculate sizes across all timestamps
                clusters_reshaped = clusters.reshape((clusters.shape[0], len(time_stamps), 13, 21))
                x_profiles_3d = np.sum(clusters_reshaped, axis=(1, 2))  # Sum across time and y
                y_profiles_3d = np.sum(clusters_reshaped, axis=(1, 3))  # Sum across time and x
                bool_arr = x_profiles_3d != 0
                x_sizes = np.sum(bool_arr, axis = 1)/21 
                bool_arr = y_profiles_3d != 0
                y_sizes = np.sum(bool_arr, axis = 1)/13
            
            # Reshape profiles to match expected output format (like in original code)
            if len(time_stamps) == 1:
                y_profiles = y_profiles.reshape((-1, 13))
                x_profiles = x_profiles.reshape((-1, 21))
            else:
                # For multiple timestamps, ensure proper 2D output
                y_profiles = y_profiles.reshape((-1, 13))
                x_profiles = x_profiles.reshape((-1, 21))
            
            y_locals = ylocal_df_raw.values/8.5
            z_locs = z_loc_df_raw.values/65
            eh_pairs = eh_pairs_raw.values/150000
            self.labels = labels_df_raw.values
            self.x_features = {}
            if 'cluster' in self.x_feature_description:
                self.x_features['cluster'] = clusters
            if 'x_profile' in self.x_feature_description:
                self.x_features['x_profile'] = x_profiles
            if 'y_profile' in self.x_feature_description:
                self.x_features['y_profile'] = y_profiles
            if 'x_size' in self.x_feature_description:
                self.x_features['x_size'] = x_sizes
            if 'y_size' in self.x_feature_description:
                self.x_features['y_size'] = y_sizes
            if 'y_local' in self.x_feature_description:
                self.x_features['y_local'] = y_locals
            if 'z_global' in self.x_feature_description:
                self.x_features['z_global'] = z_locs
            if 'total_charge' in self.x_feature_description:
                self.x_features['total_charge'] = eh_pairs
            if 'adjusted_hit_time' in self.x_feature_description:
                self.x_features['adjusted_hit_time'] = hit_time
            if 'adjusted_hit_time_30ps_gaussian' in self.x_feature_description:
                self.x_features['adjusted_hit_time_30ps_gaussian'] = hit_time_30
            if 'adjusted_hit_time_60ps_gaussian' in self.x_feature_description:
                self.x_features['adjusted_hit_time_60ps_gaussian'] = hit_time_60
            # DATA-LEVEL SHUFFLING
            if shuffle_data:
                print(f"Shuffling all data points (seed={random_seed})...")
                indices = np.arange(self.dataPoints)
                np.random.shuffle(indices)
                self.labels = self.labels[indices]
                for k in self.x_features:
                    self.x_features[k] = self.x_features[k][indices]
                type_list = [type_list[i] for i in indices]
                # Print shuffling stats
                print(f"First 20 types after shuffling: {type_list[:20]}")
                alternations = 0
                max_run = 1
                current_run = 1
                for i in range(1, len(type_list)):
                    if type_list[i] != type_list[i-1]:
                        alternations += 1
                        current_run = 1
                    else:
                        current_run += 1
                        max_run = max(max_run, current_run)
                print(f"Shuffling metrics:")
                print(f"  - Alternations (switches between BIB/SIG): {alternations}")
                print(f"  - Longest contiguous run of same type: {max_run}")
                print(f"  - Alternation rate: {alternations/(len(type_list)-1)*100:.1f}%")
            self.tf_records_dir = tf_records_dir
            os.makedirs(self.tf_records_dir, exist_ok=True)
            with open(f'{self.tf_records_dir}/x_features.txt', "w") as f:
                for x_feature in self.x_feature_description:
                    f.write(f"{x_feature} ")
            self.batch_size = batch_size
            self.save_batches_parallel()
        self.tfrecord_filenames = np.sort(np.array(tf.io.gfile.glob(os.path.join(self.tf_records_dir, "*.tfrecord"))))
        self.quantize = quantize
        self.epoch_count = 0
        self.on_epoch_end()
    def standardize(self, x, norm_factor_pos=1.7, norm_factor_neg=2.5):
        out = (x - self.dataset_mean)/self.dataset_std
        out[out > 0] = out[out > 0]/norm_factor_pos
        out[out < 0] = out[out < 0]/norm_factor_neg
        return out
    def save_batches_parallel(self):
        num_batches = round(math.ceil(self.labels.shape[0]/self.batch_size))
        paths_or_errors = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_batch = {executor.submit(self.save_single_batch, i): i for i in range(num_batches)}
            for future in tqdm(as_completed(future_to_batch), total=num_batches, desc="Saving batches as TFRecords"):
                result = future.result()
                paths_or_errors.append(result)
        for res in paths_or_errors:
            if "Error" in res:
                print(res)
    def save_single_batch(self, batch_index):
        try:
            filename = f"batch_{batch_index}.tfrecord"
            TFRfile_path = os.path.join(self.tf_records_dir, filename)
            X, y = self.prepare_batch_data(batch_index)
            serialized_example = self.serialize_example(X,y)
            with tf.io.TFRecordWriter(TFRfile_path) as writer:
                writer.write(serialized_example)
            return TFRfile_path
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error saving batch {batch_index}: {e} \n{tb}"
    def prepare_batch_data(self, batch_index):
        index = batch_index * self.batch_size
        y = self.labels[index:index+self.batch_size] / self.normalization 
        X = []
        for x_feature in self.x_feature_description:
            X.append(self.x_features[x_feature][index:index+self.batch_size])
        return X, y
    def serialize_example(self, X, y):
        y = tf.cast(y, tf.float32)
        feature = {
            'y': self._bytes_feature(tf.io.serialize_tensor(y)),
        }
        for x_feature, x_feature_name in zip(X, self.x_feature_description):
            x_feature = tf.cast(x_feature, tf.float32)
            feature[x_feature_name] = self._bytes_feature(tf.io.serialize_tensor(x_feature))
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def __getitem__(self, batch_index):
        tfrecord_path = self.tfrecord_filenames[batch_index]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        for data in parsed_dataset:
            X_batch, y_batch = data
            y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])
            for x_feature in X_batch.keys():
                X_batch[x_feature] = tf.reshape(X_batch[x_feature], [-1, *X_batch[x_feature].shape[1:]])
            return X_batch, y_batch
    def _parse_tfrecord_fn(self, example):
        feature_description = {
            'y': tf.io.FixedLenFeature([], tf.string),
        }
        for x_feature in self.x_feature_description:
            feature_description[x_feature] = tf.io.FixedLenFeature([], tf.string)
        example = tf.io.parse_single_example(example, feature_description)
        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)
        X = {}
        for x_feature in self.x_feature_description:
             X[x_feature]= tf.io.parse_tensor(example[x_feature], out_type=tf.float32)
        return X, y
    def __len__(self):
        files=[f for f in os.listdir(self.tf_records_dir) if f.endswith(".tfrecord")]
        num_batches = len(files)
        return num_batches
    def on_epoch_end(self):
        gc.collect()
        self.epoch_count += 1
        if self.epoch_count == 1:
            logging.warning(f"Quantization is {self.quantize} in data generator. This may affect model performance.") 