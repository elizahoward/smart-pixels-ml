# python imports
import tensorflow as tf
from qkeras import quantized_bits
from typing import Union, List, Tuple
import glob
import numpy as np
import pandas as pd
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import datetime
import random 
import logging
import gc
import traceback

import utils


# custom quantizer

# @tf.function
def QKeras_data_prep_quantizer(data, bits=4, int_bits=0, alpha=1):
    """
    Applies QKeras quantization.
    Args:
        data (tf.Tensor): Input data (tf.Tensor).
        bits (int): Number of bits for quantization.
        int_bits (int): Number of integer bits.
        alpha (float): (don't change)
    Returns::
        tf.Tensor: Quantized data (tf.Tensor).
    """
    quantizer = quantized_bits(bits, int_bits, alpha=alpha)
    return quantizer(data)


class OptimizedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
            data_directory_path: str = "./",
            labels_directory_path: str = "./",
            is_directory_recursive: bool = False,
            file_type: str = "csv",
            data_format: str = "2D",
            muon_collider: bool = False,
            batch_size: int = 32,
            file_count = None,
            labels_list: Union[List,str] = None,
            to_standardize: bool = False,
            normalization: Union[list,int] = 1,
            input_shape: Tuple = (13,21),
            transpose = None,
            include_y_local: bool = False,
            include_z_loc: bool = False,
            files_from_end = False,
            shuffle=False,
            current=False,
            sample_delta_t=200,
            tag: str = "",
            filteringBIB: bool = False,

            # Added in Optimized datagenerators 
            load_from_tfrecords_dir: str = None,
            tfrecords_dir: str = None,
            use_time_stamps = -1,
            seed: int = None,
            quantize: bool = False,
            max_workers: int = 1,
                 
            **kwargs,
            ):
        super().__init__() 

        """
        Data Generator to streamline data input to the network direct from the directory.
        Args:
        data_directory_path:
        labels_directory_path: 
        is_directory_recursive: 
        file_type: Default: "csv"
                   Adapt the data loader according to file type. For now, it only supports csv and parquet file formats.
        data_format: Default: 2D
                     Used to refer to the relevant "recon" files, 2D for 2D pixel array, 3D for time series input,
        batch_size: Default: 32
                    The no. of data points to be included in a single batch.
        file_count: Default: None
                    To limit the no. of .csv files to be used for training.
                    If set to None, all files will be considered as legitimate inputs.
        labels_list: Default: "cotAlpha"
                     Input column name or list of column names to be used as label input to the neural network.
        to_standardize: If set to True, it ensures that batches are normalized prior to being used as inputs
                        for training.
                        Default: False
        input_shape: Default: (13,21) for image input to a 2D feedforward neural network.
                    To reshape the input array per the requirements of the network training.
        current: Default False, calculate the current instead of the integrated charge
        sample_delta_t: how long an "ADC bin" is in picoseconds
        
        load_from_tfrecords_dir: Directory to load prepared data from TFRecords.
        tfrecords_dir: Directory to save TFRecords.
        use_time_stamps: which of the 20 time stamps to train on. default -1 is to train on all of them
        seed: Random seed for shuffling.
        quantize: Whether to quantize the data.
        """
        self.normalization = normalization
        self.filteringBIB = filteringBIB

        # decide on which time stamps to load
        self.use_time_stamps = np.arange(0,20) if use_time_stamps == -1 else use_time_stamps
        len_xy, ntime = 13*21, 20
        idx = [[i*(len_xy),(i+1)*(len_xy)] for i in range(ntime)] # 20 time stamps of length 13*21
        self.use_time_stamps = np.array([ np.arange(idx[i][0], idx[i][1]).astype("str") for i in self.use_time_stamps]).flatten().tolist()
        if use_time_stamps != -1 and data_format != '2D':
            assert len(use_time_stamps) == input_shape[0]

        self.max_workers = max_workers
        self.shuffle = shuffle
        if shuffle:
            self.seed = seed if seed is not None else 13
            self.rng = np.random.default_rng(seed = self.seed)
        
        if file_type not in ["csv", "parquet"]:
            raise ValueError("file_type can only be \"csv\" or \"parquet\"!")
        self.file_type = file_type

        if not muon_collider:
            self.recon_files = [
                f for f in glob.glob(
                    data_directory_path + "recon" + data_format + "*." + file_type, 
                    recursive=is_directory_recursive
                ) if tag in f
            ]
            self.recon_files.sort()
        else:
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
            if not muon_collider:
                if not files_from_end:
                    self.recon_files = self.recon_files[:file_count]
                else:
                    self.recon_files = self.recon_files[-file_count:]
            else:
                if not files_from_end:
                    self.recon_files = self.recon_files_bib[:file_count]+self.recon_files_sig[:file_count]
                else:
                    self.recon_files = self.recon_files_bib[-file_count:]+self.recon_files_sig[-file_count:]
        else:
            if not muon_collider:
                    self.recon_files = self.recon_files
            else:
                    self.recon_files = self.recon_files_bib+self.recon_files_sig

        
        self.include_y_local = include_y_local
        self.include_z_loc = include_z_loc

        self.file_offsets = [0]
        self.dataset_mean = None
        self.dataset_std = None

        # If data is already prepared load and use that data
        if load_from_tfrecords_dir is not None:
            if not os.path.isdir(load_from_tfrecords_dir):
                raise ValueError(f"Directory {load_from_tfrecords_dir} does not exist.")
            else:
                self.tfrecords_dir = load_from_tfrecords_dir
        else:
            utils.safe_remove_directory(tfrecords_dir)
            self.batch_size = batch_size
            self.labels_list = labels_list
            self.input_shape = input_shape
            self.transpose = transpose
            self.to_standardize = to_standardize
            
            labels_df = pd.DataFrame()
            recon_df = pd.DataFrame()
            ylocal_df = pd.DataFrame()
            z_loc_df = pd.DataFrame()

            for file in self.recon_files:
                tempDf = pd.read_parquet(file, columns=self.use_time_stamps)
                recon_df = pd.concat([recon_df,tempDf])
                file = file.replace(f"recon{data_format}","labels")
                if not self.filteringBIB:
                    labels_df = pd.concat([labels_df,pd.read_parquet(file, columns=self.labels_list)])
                else:
                    if "sig" in file:
                        labels_df = pd.concat([labels_df, pd.DataFrame({'signal': [1] * tempDf.shape[0]})])
                    else:
                        labels_df = pd.concat([labels_df, pd.DataFrame({'signal': [0] * tempDf.shape[0]})])
                ylocal_df = pd.concat([ylocal_df,pd.read_parquet(file, columns=['y-local'])])
                z_loc_df = pd.concat([z_loc_df,pd.read_parquet(file, columns=['hit_z'])])

            has_nans = np.any(np.isnan(recon_df.values), axis=1)
            has_nans = np.arange(recon_df.shape[0])[has_nans]
            recon_df_raw = recon_df.drop(has_nans)
            labels_df_raw = labels_df.drop(has_nans)
            ylocal_df_raw = ylocal_df.drop(has_nans)
            z_loc_df_raw = z_loc_df.drop(has_nans)

            recon_values = recon_df_raw.values    

            nonzeros = abs(recon_values) > 0
            
            recon_values[nonzeros] = np.sign(recon_values[nonzeros])*np.log1p(abs(recon_values[nonzeros]))/math.log(2)
            
            if self.to_standardize:
                recon_values[nonzeros] = self.standardize(recon_values[nonzeros])
            
            recon_values = recon_values.reshape((-1, *self.input_shape))            
                        
            if self.transpose is not None:
                recon_values = recon_values.transpose(self.transpose)
            
            self.recon_df = recon_values

            clusters = recon_values.reshape((recon_values.shape[0],13,21))

            y_profiles = np.sum(clusters, axis = 2)
            x_profiles = np.sum(clusters, axis = 1)

            bool_arr = x_profiles != 0
            self.x_size_df = np.sum(x_profiles, axis = 1)/21

            bool_arr = y_profiles != 0
            self.y_size_df = np.sum(y_profiles, axis = 1)/13

            self.labels_df = labels_df_raw.values
            self.ylocal_df = ylocal_df_raw.values/8.5
            self.z_loc_df = z_loc_df_raw.values/65

    
            if tfrecords_dir is None:
                raise ValueError(f"tfrecords_dir is None")
                
            self.tfrecords_dir = tfrecords_dir    
            os.makedirs(self.tfrecords_dir, exist_ok=True)
            self.save_batches_parallel() # save all the batches
            
            
        self.tfrecord_filenames = np.sort(np.array(tf.io.gfile.glob(os.path.join(self.tfrecords_dir, "*.tfrecord"))))
        self.quantize = quantize
        self.epoch_count = 0
        self.on_epoch_end()

    def process_file_parallel(self):
        file_infos = [(afile, self.use_time_stamps, self.file_type, self.input_shape, self.transpose) for afile in self.recon_files]
        results = []
        with ProcessPoolExecutor(self.max_workers) as executor:
            futures = [executor.submit(self._process_file_single, file_info) for file_info in file_infos]
            for future in tqdm(as_completed(futures), total=len(file_infos), desc="Processing Files..."):
                results.append(future.result())

        for amean, avariance, amin, amax, num_rows in results:
            self.file_offsets.append(self.file_offsets[-1] + num_rows)

            if self.dataset_mean is None:
                self.dataset_max = amax
                self.dataset_min = amin
                self.dataset_mean = amean
                self.dataset_std = avariance
            else:
                self.dataset_max = max(self.dataset_max, amax)
                self.dataset_min = min(self.dataset_min, amin)
                self.dataset_mean += amean
                self.dataset_std += avariance

        self.dataset_mean = self.dataset_mean / len(self.recon_files)
        self.dataset_std = np.sqrt(self.dataset_std / len(self.recon_files)) 
            
        self.file_offsets = np.array(self.file_offsets)

    @staticmethod
    def _process_file_single(file_info):
        afile, use_time_stamps, file_type, input_shape, transpose = file_info
        if file_type == "csv":
            adf = pd.read_csv(afile).dropna()
        elif file_type == "parquet":
            adf = pd.read_parquet(afile, columns=use_time_stamps).dropna()
    
        x = adf.values
        nonzeros = abs(x) > 0
        x[nonzeros] = np.sign(x[nonzeros]) * np.log1p(abs(x[nonzeros])) / math.log(2)
        amean, avariance = np.mean(x[nonzeros], keepdims=True), np.var(x[nonzeros], keepdims=True) + 1e-10
        centered = np.zeros_like(x)
        centered[nonzeros] = (x[nonzeros] - amean) / np.sqrt(avariance)
        x = x.reshape((-1, *input_shape))
        if transpose is not None:
            x = x.transpose(transpose)
        amin, amax = np.min(centered), np.max(centered)
        len_adf = len(adf)
        del adf
        gc.collect()
        
        return amean, avariance, amin, amax, len_adf

    def standardize(self, x, norm_factor_pos=1.7, norm_factor_neg=2.5):
        """
        Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network.
        Args:
            x: Batch of inputs to be normalized.
        Returns:
            The inputs, normalized. 
        """
        out = (x - self.dataset_mean)/self.dataset_std
        out[out > 0] = out[out > 0]/norm_factor_pos
        out[out < 0] = out[out < 0]/norm_factor_neg
        return out

    def save_batches_parallel(self):
        """
        Saves data batches as multiple TFRecord files.
        """
        num_batches = round(math.ceil(self.labels_df.shape[0]/self.batch_size)) # Total num of batches
        paths_or_errors = []

        # The max_workers is set to 1 because processing large batches in multiple threads can significantly
        # increase RAM usage. Adjust 'max_workers' based on your system's RAM capacity and requirements.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_batch = {executor.submit(self.save_single_batch, i): i for i in range(num_batches)}
            
            for future in tqdm(as_completed(future_to_batch), total=num_batches, desc="Saving batches as TFRecords"):
                result = future.result()
                paths_or_errors.append(result)
            
        for res in paths_or_errors:
            if "Error" in res:
                print(res)  
                
    def save_single_batch(self, batch_index):
        """
        Serializes and saves a single batch to a TFRecord file.
        Args:
            batch_index (int): Index of the batch to save.
        Returns:
            str: Path to the saved TFRecord file or an error message.
        """
        
        try:
            filename = f"batch_{batch_index}.tfrecord"
            TFRfile_path = os.path.join(self.tfrecords_dir, filename)

            if self.filteringBIB:
                x_size, y_size, y_local, z_loc, y = self.prepare_batch_data(batch_index)
                serialized_example = self.serialize_example(y, x_size=x_size, y_size=y_size, y_local=y_local, z_loc=z_loc)
            else:
                print("HERE: Saving tfrecords is going wrong!!")
                X, y = self.prepare_batch_data(batch_index)
                serialized_example = self.serialize_example(y, X=X)

            with tf.io.TFRecordWriter(TFRfile_path) as writer:
                writer.write(serialized_example)
            return TFRfile_path
        
        except Exception as e:
            tb = traceback.format_exc()
            return f"Error saving batch {batch_index}: {e} \n{tb}" 
    
    def prepare_batch_data(self, batch_index):
        """
        Used to fetch a batch of inputs (X,y) for the network's training.
        """
        index = batch_index * self.batch_size # absolute *event* index
        
        y = self.labels_df[index:index+self.batch_size] / self.normalization 

        # print(f'start_index: {index}\t end_index: {batch_size}')
        if self.filteringBIB:
            x_size = self.x_size_df[index:index+self.batch_size]
            y_size = self.y_size_df[index:index+self.batch_size]
            y_local = self.ylocal_df[index:index+self.batch_size]
            z_loc = self.z_loc_df[index:index+self.batch_size]
            
            return x_size, y_size, y_local, z_loc, y
        else:
            X = self.recon_df[index:index+self.batch_size]

            return X, y

    
    def serialize_example(self, y, **kwargs):
        """
        Serializes a single example (featuresand labels) to TFRecord format. 
        
        Args:
        - X: Training data
        - y: labelled data
        
        Returns:
        - string (serialized TFRecord example).
        """

        # X and y are float32 (maybe we can reduce this
        y = tf.cast(y, tf.float32)

        feature = {
            'y': self._bytes_feature(tf.io.serialize_tensor(y)),
        }

        if 'X' in kwargs:
            X = kwargs['X']
            X = tf.cast(X, tf.float32)
            feature['X'] = self._bytes_feature(tf.io.serialize_tensor(X))

        if 'x_size' in kwargs:
            x_size = kwargs['x_size']
            x_size = tf.cast(x_size, tf.float32)
            feature['x_size'] = self._bytes_feature(tf.io.serialize_tensor(x_size))

        if 'y_size' in kwargs:
            y_size = kwargs['y_size']
            y_size = tf.cast(y_size, tf.float32)
            feature['y_size'] = self._bytes_feature(tf.io.serialize_tensor(y_size))

        if 'y_local' in kwargs:
            y_local = kwargs['y_local']
            y_local = tf.cast(y_local, tf.float32)
            feature['y_local'] = self._bytes_feature(tf.io.serialize_tensor(y_local))

        if 'z_loc' in kwargs:
            z_loc = kwargs['z_loc']
            z_loc = tf.cast(z_loc, tf.float32)
            feature['z_loc'] = self._bytes_feature(tf.io.serialize_tensor(z_loc))

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def _bytes_feature(value):
        """
        Converts a string/byte value into a Tf feature of bytes_list
        
        Args: 
        - string/byte value
        
        Returns:
        - tf.train.Feature object as a bytes_list containing the input value.
        """
        if isinstance(value, type(tf.constant(0))): # check if Tf tensor
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def __getitem__(self, batch_index):
        """
        Load the batch from a pre-saved TFRecord file instead of processing raw data.
        Each file contains exactly one batch.
        quantization is done here: Helpful for pretraining without the quantization and the later training with quantized data.
        shuffling is also done here.
        TODO: prefetching (un-done)
        """
        assert self.filteringBIB == True, "Unexpected change in variable!"
        
        tfrecord_path = self.tfrecord_filenames[batch_index]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        for data in parsed_dataset:
            ''' Add the reshaping in saving'''
            X_batch, y_batch = data

            y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])

            if isinstance(X_batch, tuple):
                X_batch = list(X_batch)
                for batch in X_batch:
                    batch = tf.reshape(batch, [-1, *batch.shape[1:]])
            else:
                X_batch = tf.reshape(batch, [-1, *X_batch.shape[1:]])                
            
            """if self.filteringBIB:
                (x_size_batch, y_size_batch, y_local_batch, z_loc_batch), y_batch = data

                x_size_batch = tf.reshape(x_size_batch, [-1, *x_size_batch.shape[1:]])
                y_size_batch = tf.reshape(y_size_batch, [-1, *y_size_batch.shape[1:]])
                y_local_batch = tf.reshape(y_local_batch, [-1, *y_local_batch.shape[1:]])
                z_loc_batch = tf.reshape(z_loc_batch, [-1, *z_loc_batch.shape[1:]])

                y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])

                X_batch = [x_size_batch, y_size_batch, y_local_batch, z_loc_batch]

                return X_batch, y_batch

            else:
                if self.include_y_local and not self.include_z_loc:
                    (X_batch, y_local_batch), y_batch = data
                elif not self.include_y_local and self.include_z_loc:
                    (X_batch, z_loc_batch), y_batch = data
                elif self.include_y_local and self.include_z_loc:
                    (X_batch, y_local_batch, z_loc_batch), y_batch = data
                else:
                    X_batch, y_batch = data

                X_batch = tf.reshape(X_batch, [-1, *X_batch.shape[1:]])
                y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])
                if self.include_y_local:
                    y_local_batch = tf.reshape(y_local_batch, [-1, *y_local_batch.shape[1:]])
                if self.include_z_loc:
                    z_loc_batch = tf.reshape(z_loc_batch, [-1, *z_loc_batch.shape[1:]])
                if self.quantize:
                    X_batch = QKeras_data_prep_quantizer(X_batch, bits=4, int_bits=0, alpha=1)

                if self.shuffle:
                    indices = tf.range(start=0, limit=tf.shape(X_batch)[0], dtype=tf.int32)
                    shuffled_indices = tf.random.shuffle(indices, seed=self.seed)

                    X_batch = tf.gather(X_batch, shuffled_indices)
                    y_batch = tf.gather(y_batch, shuffled_indices)
                    if self.include_y_local:
                        y_local_batch = tf.gather(y_local_batch, shuffled_indices)
                    
                del raw_dataset, parsed_dataset
                if self.include_y_local and not self.include_z_loc:
                    X_batch = [X_batch, y_local_batch]
                if not self.include_y_local and self.include_z_loc:
                    X_batch = [X_batch, z_loc_batch]
                if self.include_y_local and self.include_z_loc:
                    X_batch = [X_batch, y_local_batch, z_loc_batch]"""
            return X_batch, y_batch
            
    
    def _parse_tfrecord_fn(self, example):
        """
        Parses a single TFRecord example.
        
        Returns:
        - X: as a float32 tensor.
        - y: as a float32 tensor.
        """
        feature_list = example.keys

        feature_list.remove('y')

        feature_description = {
            'y': tf.io.FixedLenFeature([], tf.string),
        }

        for feature in feature_list:
            feature_description[feature] = tf.io.FixedLenFeature([], tf.string)

        example = tf.io.parse_single_example(example, feature_description)

        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)

        if len(feature_list)>1:
            X = []
            for feature in feature_list:
                X.append(tf.io.parse_tensor(example[feature], out_type=tf.float32))
        else:
            X = tf.io.parse_tensor(example['X'], out_type=tf.float32)

        return X, y
        
        """if self.include_y_local or self.filteringBIB:
            feature_description['y_local'] = tf.io.FixedLenFeature([], tf.string)

        if self.include_z_loc or self.filteringBIB:
            feature_description['z_loc'] = tf.io.FixedLenFeature([], tf.string)

        if self.filteringBIB:
            feature_description['x_size'] = tf.io.FixedLenFeature([], tf.string)
            feature_description['y_size'] = tf.io.FixedLenFeature([], tf.string)
        else:
            feature_description['X'] = tf.io.FixedLenFeature([], tf.string)

        example = tf.io.parse_single_example(example, feature_description)
    
        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)

        if self.filteringBIB:
            x_size = tf.io.parse_tensor(example['x_size'], out_type=tf.float32)
            y_size = tf.io.parse_tensor(example['y_size'], out_type=tf.float32)
            y_local = tf.io.parse_tensor(example['y_local'], out_type=tf.float32)
            z_loc = tf.io.parse_tensor(example['z_loc'], out_type=tf.float32)
            return (x_size, y_size, y_local, z_loc), y
        else:
            print(self.filteringBIB)
            print("HERE")
            X = tf.io.parse_tensor(example['X'], out_type=tf.float32)
            if self.include_y_local and not self.include_z_loc:
                y_local = tf.io.parse_tensor(example['y_local'], out_type=tf.float32)
                return (X, y_local), 
            elif not self.include_y_local and self.include_z_loc:
                z_loc = tf.io.parse_tensor(example['z_loc'], out_type=tf.float32)
                return (X, z_loc), y
            elif self.include_y_local and self.include_z_loc:
                y_local = tf.io.parse_tensor(example['y_local'], out_type=tf.float32)
                z_loc = tf.io.parse_tensor(example['z_loc'], out_type=tf.float32)
                return (X, y_local, z_loc), y
            else:
                return X, y"""

    def __len__(self):
        if len(self.file_offsets) != 1: # used when TFRecord files are created during initialization
            num_batches = self.file_offsets[-1] // self.batch_size
        else: # used during loading saved TFRecord files
            num_batches = len(os.listdir(self.tfrecords_dir))
        return num_batches

    def on_epoch_end(self):
        '''
        This shuffles the file ordering so that it shuffles the ordering in which the TFRecord
        are loaded during the training for each epochs.
        '''
        gc.collect()
        self.epoch_count += 1
        # Log quantization status once
        if self.epoch_count == 1:
            logging.warning(f"Quantization is {self.quantize} in data generator. This may affect model performance.")

        if self.shuffle:
            self.rng.shuffle(self.tfrecord_filenames)
            self.seed += 1 # So that after each epoch the batch is shuffled with a different seed (deterministic)
