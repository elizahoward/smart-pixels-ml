#!/usr/bin/env python3
"""
Script to generate TF records for multiple batch sizes and timestamp configurations.
Creates folders for batch sizes 512, 1024, and 2048 with both single timestamp and all timestamps.
"""

from pathlib import Path
import os, shutil
import glob
import time

import sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parentdir)
import OptimizedDataGenerator4_data_shuffled as ODG

def create_tf_records(batch_size, is_single_timestamp, base_output_dir, random_seed=42):
    """
    Create TF records for a specific batch size and timestamp configuration.
    
    Args:
        batch_size (int): Batch size for the records (512, 1024, or 2048)
        is_single_timestamp (bool): True for single timestamp, False for all timestamps
        base_output_dir (str): Base directory for output
        random_seed (int): Random seed for shuffling
    """
    
    # Configuration parameters
    data_directory_path = "/local/d1/smartpixML/PixelSim2/MuonColliderSim/Simulation_Output/"
    is_directory_recursive = False
    file_type = "parquet"
    data_format = "3D"
    normalization = 1
    file_fraction = .8  # fraction of files used for training
    to_standardize = False
    transpose = None
    x_feature_description = "all"
    filteringBIB = True
    shuffle_data = True
    
    # Configure timestamp-specific parameters
    if is_single_timestamp:
        input_shape = (1, 13, 21)  # Only last timestamp
        time_stamps = [19]  # Only the last timestamp
        timestamp_suffix = "single"
    else:
        input_shape = (20, 13, 21)  # All 20 timestamps
        time_stamps = list(range(20))  # All 20 timestamps (0-19)
        timestamp_suffix = "all"
    
    # Create directory name
    directory_name = f'filtering_records{batch_size}_data_shuffled_{timestamp_suffix}'
    records_dir = os.path.join(base_output_dir, directory_name)
    
    print(f"\n{'='*60}")
    print(f"CREATING: {directory_name}")
    print(f"Batch size: {batch_size}")
    print(f"Timestamps: {'Single (19)' if is_single_timestamp else 'All (0-19)'}")
    print(f"Random seed: {random_seed}")
    print(f"{'='*60}")
    
    # Create records directory
    if not os.path.exists(records_dir):
        os.makedirs(records_dir)
    else:
        # Clean existing directory
        print(f"Directory {records_dir} already exists. Cleaning...")
        files = os.listdir(records_dir)
        for filename in files:
            file_path = os.path.join(records_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    # Create subdirectories for train and validation
    tf_dir_train = Path(records_dir, "tfrecords_train").resolve()
    tf_dir_validation = Path(records_dir, "tfrecords_validation").resolve()
    
    os.makedirs(tf_dir_train, exist_ok=True)
    os.makedirs(tf_dir_validation, exist_ok=True)
    
    # Calculate file counts
    total_files = len(glob.glob(
        data_directory_path + "recon" + data_format + "bib*." + file_type, 
        recursive=is_directory_recursive
    ))
    file_count = round(file_fraction * total_files)
    
    print(f"Total files: {total_files}")
    print(f"Training files: {file_count}")
    print(f"Validation files: {total_files - file_count}")
    
    # Create training generator
    print("Creating training generator...")
    start_time = time.time()
    training_generator = ODG.OptimizedDataGeneratorDataShuffled(
        data_directory_path=data_directory_path,
        is_directory_recursive=is_directory_recursive,
        file_type=file_type,
        data_format=data_format,
        batch_size=batch_size,
        to_standardize=to_standardize,
        normalization=normalization,
        file_count=file_count,
        input_shape=input_shape,
        transpose=transpose,
        time_stamps=time_stamps,
        tf_records_dir=str(tf_dir_train),
        x_feature_description=x_feature_description,
        filteringBIB=filteringBIB,
        shuffle_data=shuffle_data,
        random_seed=random_seed,
    )
    print(f"Training generator created in {time.time() - start_time:.2f} seconds")
    
    # Create validation generator
    print("Creating validation generator...")
    start_time = time.time()
    validation_generator = ODG.OptimizedDataGeneratorDataShuffled(
        data_directory_path=data_directory_path,
        is_directory_recursive=is_directory_recursive,
        file_type=file_type,
        data_format=data_format,
        batch_size=batch_size,
        to_standardize=to_standardize,
        normalization=normalization,
        file_count=total_files - file_count,
        files_from_end=True,
        input_shape=input_shape,
        transpose=transpose,
        time_stamps=time_stamps,
        tf_records_dir=str(tf_dir_validation),
        x_feature_description=x_feature_description,
        filteringBIB=filteringBIB,
        shuffle_data=shuffle_data,
        random_seed=random_seed + 1,  # Use different seed for validation
    )
    print(f"Validation generator created in {time.time() - start_time:.2f} seconds")
    
    print(f"✓ TF records created successfully for {directory_name}")
    print(f"  Training records: {tf_dir_train}")
    print(f"  Validation records: {tf_dir_validation}")
    
    return records_dir

def main():
    """Main function to generate all TF record configurations."""
    
    # Create main output directory
    base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_batches_shuffled")
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Created main output directory: {base_output_dir}")
    else:
        print(f"Using existing output directory: {base_output_dir}")
    
    # Configuration matrix
    batch_sizes = [512, 1024, 2048]
    timestamp_configs = [
        (True, "single timestamp (19)"),
        (False, "all timestamps (0-19)")
    ]
    
    total_configs = len(batch_sizes) * len(timestamp_configs)
    current_config = 0
    
    print(f"\n{'='*80}")
    print(f"GENERATING TF RECORDS FOR {total_configs} CONFIGURATIONS")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Timestamp configurations: Single timestamp and All timestamps")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*80}")
    
    start_total_time = time.time()
    created_dirs = []
    
    # Generate records for each configuration
    for batch_size in batch_sizes:
        for is_single_timestamp, timestamp_desc in timestamp_configs:
            current_config += 1
            
            print(f"\n[{current_config}/{total_configs}] Processing batch size {batch_size} with {timestamp_desc}")
            
            try:
                records_dir = create_tf_records(
                    batch_size=batch_size,
                    is_single_timestamp=is_single_timestamp,
                    base_output_dir=base_output_dir,
                    random_seed=42
                )
                created_dirs.append(records_dir)
                
            except Exception as e:
                print(f"ERROR: Failed to create records for batch size {batch_size} with {timestamp_desc}")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    total_time = time.time() - start_total_time
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Successfully created {len(created_dirs)} out of {total_configs} configurations")
    print(f"\nCreated directories:")
    for i, dir_path in enumerate(created_dirs, 1):
        print(f"  {i}. {os.path.basename(dir_path)}")
    
    print(f"\nAll records are stored in: {base_output_dir}")

if __name__ == "__main__":
    main() 