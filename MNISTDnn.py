import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from plotting import *

# Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Cast to np.float32
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

# Dataset variables
train_size = x_train.shape[0]
test_size = x_test.shape[0]
num_features = 784 # image 28x28=784
num_classes = 10 # digits 0..9

# Compute to categorical classes
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Reshape the input data
x_train = x_train.reshape(train_size, num_features)
y_test = x_test.reshape(test_size, num_features)

# Model variables
hidden_layer_size = 50
nodes = [num_features, hidden_layer_size, num_classes] # input, hidden, output
epochs = 1

class Model:
    def __init__(self):
        



    