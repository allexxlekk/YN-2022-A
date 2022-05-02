"""Contains functions to prepare data before loading them to the network."""
import numpy as np


def center_dataset(dataset):
    """Centers the data. i.e. subtracts the mean from each element"""
    print("Centering Data...")
    c_dataset = []
    center_function = lambda x: x - x.mean()
    for vector in dataset:
        tmp_c = center_function(vector)
        c_dataset.append(tmp_c)
    return np.array(c_dataset)


def normalize_dataset(dataset):
    """Normalizes the data. i.e. updated values range from [0,1]"""
    print("Normalizing Data...")
    n_dataset = []
    normalize_function = lambda x: (x - np.min(x)) / np.ptp(x)
    for vector in dataset:
        tmp_n = normalize_function(vector)
        n_dataset.append(tmp_n)
    return np.array(n_dataset)


def standardize_dataset(dataset):
    """Standardizes the data."""
    print("Standardizing Data...")
    s_dataset = []
    standardize_function = lambda x: (x - np.mean(x)) / np.std(x)
    for vector in dataset:
        tmp_s = standardize_function(vector)
        s_dataset.append(tmp_s)
    return np.array(s_dataset)
