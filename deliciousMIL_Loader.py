"""Contains functions to load and transform the data from the train and testing data files."""

import re
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

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


# Word Embedding Section


def get_input_vector_E(data):
    """Converts line into a sentence."""
    tmp_line = re.sub("<[0-9]*>", "", data)
    return re.sub(" +", " ", tmp_line).strip()


# Training or Testing Dataset w/ word embedding.
def get_dataset_E(filename):
    """Returns created one hot encoded input vectors with padding, output vectors from a file."""
    X = []
    y = []

    with open(filename, "r") as f:
        for line in f:
            line = line.split(",")
            X.append(get_input_vector_E(line[0]))
            y.append(get_output_vector(line[1]))

        padding_size = 0
        # Get the padding size and represent sentences as one hot encoded words based on vocabulary size.
        for sentence in X:
            no_words = len(sentence.split())
            if no_words > padding_size:
                padding_size = no_words

    Y = np.array(y)
    return [X, Y, padding_size]
