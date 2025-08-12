#!/usr/bin/env python3
"""
Script to find the best model from KerasTuner hyperparameter search results.
Analyzes all trial directories and identifies the model with the highest validation accuracy.
"""

import os
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_trial_data(trial_dir):
    """Load trial data from a trial directory."""
    trial_json_path = os.path.join(trial_dir, "trial.json")
    
    if not os.path.exists(trial_json_path):
        return None
    
    try:
        with open(trial_json_path, 'r') as f:
            trial_data = json.load(f)
        return trial_data
    except Exception as e:
        print(f"Error loading trial data from {trial_dir}: {e}")
        return None

def analyze_all_trials(search_dir):
    """Analyze all trials in the search directory."""
    trials_data = []
    
    # Get all trial directories
    trial_dirs = glob.glob(os.path.join(search_dir, "trial_*"))
    trial_dirs.sort()  # Sort to ensure consistent ordering
    
    print(f"Found {len(trial_dirs)} trial directories")
    
    for trial_dir in trial_dirs:
        trial_data = load_trial_data(trial_dir)
        if trial_data is None:
            continue
            
        # Extract trial information
        trial_id = trial_data.get('trial_id', 'unknown')
        status = trial_data.get('status', 'unknown')
        score = trial_data.get('score', 0.0)
        best_step = trial_data.get('best_step', 0)
        
        # Extract hyperparameters
        hyperparams = trial_data.get('hyperparameters', {}).get('values', {})
        
        # Extract metrics
        metrics = trial_data.get('metrics', {}).get('metrics', {})
        
        # Get final values for key metrics
        final_accuracy = None
        final_loss = None
        final_val_accuracy = None
        final_val_loss = None
        
        if 'accuracy' in metrics:
            observations = metrics['accuracy'].get('observations', [])
            if observations:
                final_accuracy = observations[-1]['value'][0]
                
        if 'loss' in metrics:
            observations = metrics['loss'].get('observations', [])
            if observations:
                final_loss = observations[-1]['value'][0]
                
        if 'val_accuracy' in metrics:
            observations = metrics['val_accuracy'].get('observations', [])
            if observations:
                final_val_accuracy = observations[-1]['value'][0]
                
        if 'val_loss' in metrics:
            observations = metrics['val_loss'].get('observations', [])
            if observations:
                final_val_loss = observations[-1]['value'][0]
        
        trial_info = {
            'trial_id': trial_id,
            'status': status,
            'score': score,
            'best_step': best_step,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'final_val_accuracy': final_val_accuracy,
            'final_val_loss': final_val_loss,
            'trial_dir': trial_dir,
            **hyperparams
        }
        
        trials_data.append(trial_info)
    
    return trials_data

def find_best_model(trials_data, metric='final_val_accuracy'):
    """Find the best model based on the specified metric."""
    if not trials_data:
        print("No trial data available")
        return None
    
    # Filter completed trials
    completed_trials = [t for t in trials_data if t['status'] == 'COMPLETED' and t[metric] is not None]
    
    if not completed_trials:
        print("No completed trials found")
        return None
    
    # Sort by the specified metric
    if 'accuracy' in metric:
        # Higher is better for accuracy
        best_trial = max(completed_trials, key=lambda x: x[metric])
    else:
        # Lower is better for loss
        best_trial = min(completed_trials, key=lambda x: x[metric])
    
    return best_trial

def print_trial_summary(trial_data):
    """Print a summary of trial data."""
    print(f"\n{'='*60}")
    print(f"BEST MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Trial ID: {trial_data['trial_id']}")
    print(f"Status: {trial_data['status']}")
    print(f"Score: {trial_data['score']:.6f}")
    print(f"Best Step: {trial_data['best_step']}")
    print(f"Final Training Accuracy: {trial_data['final_accuracy']:.6f}")
    print(f"Final Training Loss: {trial_data['final_loss']:.6f}")
    print(f"Final Validation Accuracy: {trial_data['final_val_accuracy']:.6f}")
    print(f"Final Validation Loss: {trial_data['final_val_loss']:.6f}")
    print(f"Trial Directory: {trial_data['trial_dir']}")
    
    print(f"\nHYPERPARAMETERS:")
    print(f"{'='*30}")
    hyperparam_keys = [k for k in trial_data.keys() if k not in [
        'trial_id', 'status', 'score', 'best_step', 'final_accuracy', 
        'final_loss', 'final_val_accuracy', 'final_val_loss', 'trial_dir'
    ]]
    
    for key in sorted(hyperparam_keys):
        print(f"{key}: {trial_data[key]}")

def save_results_to_csv(trials_data, output_file):
    """Save trial results to CSV file."""
    if not trials_data:
        print("No data to save")
        return
    
    df = pd.DataFrame(trials_data)
    
    # Reorder columns to put key metrics first
    key_cols = ['trial_id', 'status', 'score', 'final_val_accuracy', 'final_val_loss', 
                'final_accuracy', 'final_loss', 'best_step']
    other_cols = [col for col in df.columns if col not in key_cols + ['trial_dir']]
    
    df_reordered = df[key_cols + other_cols + ['trial_dir']]
    
    df_reordered.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

def copy_best_model(best_trial, output_dir):
    """Copy the best model files to a designated output directory."""
    if not best_trial:
        print("No best trial to copy")
        return
    
    trial_dir = best_trial['trial_dir']
    trial_id = best_trial['trial_id']
    
    # Create output directory
    best_model_dir = os.path.join(output_dir, f"best_model_trial_{trial_id}")
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Copy all files from the best trial
    import shutil
    for file_name in os.listdir(trial_dir):
        src_file = os.path.join(trial_dir, file_name)
        dst_file = os.path.join(best_model_dir, file_name)
        
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
        elif os.path.isdir(src_file):
            shutil.copytree(src_file, dst_file)
    
    # Create a summary file
    summary_file = os.path.join(best_model_dir, "model_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"BEST MODEL SUMMARY\n")
        f.write(f"{'='*50}\n")
        f.write(f"Trial ID: {trial_id}\n")
        f.write(f"Validation Accuracy: {best_trial['final_val_accuracy']:.6f}\n")
        f.write(f"Validation Loss: {best_trial['final_val_loss']:.6f}\n")
        f.write(f"Training Accuracy: {best_trial['final_accuracy']:.6f}\n")
        f.write(f"Training Loss: {best_trial['final_loss']:.6f}\n")
        f.write(f"Best Step: {best_trial['best_step']}\n")
        f.write(f"\nHYPERPARAMETERS:\n")
        f.write(f"{'='*30}\n")
        
        hyperparam_keys = [k for k in best_trial.keys() if k not in [
            'trial_id', 'status', 'score', 'best_step', 'final_accuracy', 
            'final_loss', 'final_val_accuracy', 'final_val_loss', 'trial_dir'
        ]]
        
        for key in sorted(hyperparam_keys):
            f.write(f"{key}: {best_trial[key]}\n")
    
    print(f"Best model files copied to: {best_model_dir}")
    return best_model_dir

def main():
    """Main function to analyze hyperparameter search results."""
    # Get the search directory (current directory)
    search_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Analyzing hyperparameter search results in: {search_dir}")
    
    # Analyze all trials
    trials_data = analyze_all_trials(search_dir)
    
    if not trials_data:
        print("No trial data found!")
        return
    
    print(f"\nAnalyzed {len(trials_data)} trials")
    
    # Find best model by validation accuracy
    best_trial = find_best_model(trials_data, 'final_val_accuracy')
    
    if best_trial:
        print_trial_summary(best_trial)
        
        # Create output directory for results
        output_dir = os.path.join(search_dir, "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to CSV
        csv_file = os.path.join(output_dir, "all_trials_results.csv")
        save_results_to_csv(trials_data, csv_file)
        
        # Copy best model files
        best_model_dir = copy_best_model(best_trial, output_dir)
        
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        print(f"Best model files available in: {best_model_dir}")
        
        # Print top 5 models
        completed_trials = [t for t in trials_data if t['status'] == 'COMPLETED' and t['final_val_accuracy'] is not None]
        top_5 = sorted(completed_trials, key=lambda x: x['final_val_accuracy'], reverse=True)[:5]
        
        print(f"\nTOP 5 MODELS BY VALIDATION ACCURACY:")
        print(f"{'='*60}")
        for i, trial in enumerate(top_5, 1):
            print(f"{i}. Trial {trial['trial_id']}: {trial['final_val_accuracy']:.6f} (Val Loss: {trial['final_val_loss']:.6f})")
    
    else:
        print("No valid trials found!")

if __name__ == "__main__":
    main() 