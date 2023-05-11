import numpy as np
import os

def FindMaxShape(batchPath):
    # Find all data files
    data_files = [f for f in os.listdir(batchPath) if "data" in f]
    
    # Initialize variables to store maximum dimensions
    x_max, y_max, z_max = 0, 0, 0
    
    # Loop through each batch
    for batch_file in data_files:
        # Load batch data and decompress
        batch_data = np.unpackbits(np.load(os.path.join(batchPath, batch_file)), axis=-1).astype(bool)
        
        # Loop through each voxel grid in batch
        for voxel_grid in batch_data:
            # Get dimensions of voxel grid
            x, y, z = voxel_grid.shape
            
            # Update maximum dimensions if necessary
            x_max = max(x_max, x)
            y_max = max(y_max, y)
            z_max = max(z_max, z)
    return x_max, y_max, z_max


def FindFirstShape(batchPath):
    # Find all data files
    data_files = [f for f in os.listdir(batchPath) if "data" in f]
    
    # Loop through each batch
    batch_data = np.unpackbits(np.load(os.path.join(batchPath, data_files[0])), axis=-1).astype(bool)
    
    # Get dimensions of voxel grid
    x_max, y_max, z_max = batch_data[0].shape
    return x_max, y_max, z_max


def get_dir_size(path='.'):
    total_size = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total_size += entry.stat().st_size
            elif entry.is_dir():
                total_size += get_dir_size(entry.path)
    return total_size

# batches a list of file paths
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]