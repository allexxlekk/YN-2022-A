"""Contains functions to load and transform the data from the train and testing data files."""

import re
import numpy as np

# Functions that produce:

# Input Vector for the Neural Network.
def sanitize_input(data):
    """Gets the encoded text from a data file and extracts all useful information into a list."""
    tmp_line = re.sub("<[0-9]*>", "", data)  # Removes unecessary information.
    return re.split(" +", tmp_line.strip())


def vectorize_input(data_list):
    """Converts list to a numpy array with the count of each element."""
    in_vector = np.zeros((8520,), dtype=np.float64)
    for number in data_list:
        in_vector[int(number)] += 1
    return in_vector


def get_input_vector(data):
    """Converts encoded text from the data file into a numpy array."""
    return vectorize_input(sanitize_input(data))


# Expected Output Vector for the Neural Network.u
def get_output_vector(line):
    """Converts a result label from the data file into a numpy vector"""
    tmp_line = re.split(" ", line)
    return np.array(tmp_line, np.float64)


# Training or Testing Dataset.
def get_dataset(filename):
    """Returns all the input and output vectors from a file."""
    x = []
    y = []
    with open(filename, "r") as f:
        for line in f:
            line = line.split(",")
            x.append(get_input_vector(line[0]))
            y.append(get_output_vector(line[1]))
    X = np.array(x)
    Y = np.array(y)
    return [X, Y]
