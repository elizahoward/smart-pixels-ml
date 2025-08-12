import os
import numpy as np

# Paths to the folders
base_dir = os.path.dirname(os.path.abspath(__file__))
new_dir = os.path.join(base_dir, 'signal_tracklists_new')
old_dir = os.path.join(base_dir, 'signal_tracklists_old')

# Function to load all data from a folder
def load_all_tracklists(folder):
    data = []
    for fname in os.listdir(folder):
        if fname.endswith('.txt'):
            fpath = os.path.join(folder, fname)
            arr = np.loadtxt(fpath)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            data.append(arr)
    if data:
        return np.vstack(data)
    else:
        return np.empty((0, 9))

# Load data
new_data = load_all_tracklists(new_dir)
old_data = load_all_tracklists(old_dir)

# Column names (from project scripts)
colnames = [
    'cota', 'cotb', 'p', 'flp', 'localx', 'localy', 'pT', 'hittime', 'pid'
]

# Function to print mean and covariance
def print_stats(data, label):
    print(f'===== {label} =====')
    if data.shape[0] == 0:
        print('No data found.')
        return
    means = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    print('Means:')
    for i, name in enumerate(colnames):
        print(f'  {name:8s}: {means[i]: .5f}')
    print('\nCovariance matrix:')
    for i, name in enumerate(colnames):
        row = ' '.join(f'{cov[i, j]: .5f}' for j in range(len(colnames)))
        print(f'{name:8s}: {row}')
    print()

print_stats(new_data, 'signal_tracklists_new')
print_stats(old_data, 'signal_tracklists_old') 