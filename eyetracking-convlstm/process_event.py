import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import cv2
import pandas as pd
from convlstmbak import ConvLSTM
import tqdm as tqdm
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import median_filter
height = N = 180  # input y size
width = M = 240  # input x size

chunk_size = 500

def create_samples(data,sequence, stride):
    num_samples = data.shape[0]

    chunk_num = num_samples // chunk_size
    # Create start indices for each chunk
    chunk_starts = np.arange(chunk_num) * chunk_size

    # For each start index, create the indices of subframes within the chunk
    within_chunk_indices = np.arange(sequence) + np.arange(0, chunk_size - sequence + 1, stride)[:, None]

    # For each chunk start index, add the within chunk indices to get the complete indices
    indices = chunk_starts[:, None, None] + within_chunk_indices[None, :, :]

    # Reshape indices to be two-dimensional
    indices = indices.reshape(-1, indices.shape[-1])

    subframes = data[indices]
    return subframes


def load_filenames(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]
    
def get_data(data_dir, output_dir,seq, stride):
    # Sort the file paths
    sorted_data_file_paths = sorted(data_dir)
    data = None
    for file_path in tqdm.tqdm(sorted_data_file_paths):
        with tables.open_file(file_path, 'r') as h5_file:
            data = h5_file.root.vector[:]
            extended_data = create_samples(data, seq, stride)
        output_file_name = file_path.split("/")[4:5][0]
        output_file = os.path.join(output_dir, output_file_name)
        with tables.open_file(output_file, mode='w') as f:
            filters = tables.Filters(complevel=5, complib='blosc')
            f.create_carray('/', 'vector', obj=extended_data, filters=filters)

data_dir = "/DATA/pupil_st/data_ts_500"
output_dir_train = "/DATA/pupil_st/data_ts_pro/train/"
output_dir_val = "/DATA/pupil_st/data_ts_pro/val/"
# Load filenames from the provided lists
train_filenames = load_filenames('train_files.txt')
val_filenames = load_filenames('val_files.txt')

# Get the data file paths and target file paths
data_train = [os.path.join(data_dir, f + '.h5') for f in train_filenames]
data_val = [os.path.join(data_dir, f + '.h5') for f in val_filenames]

seq = 40
stride = 1
stride_val = 40
get_data(data_train, output_dir_train, seq, stride)
get_data(data_val, output_dir_val, seq, stride_val)
