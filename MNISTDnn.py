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

# Reshape the input data from matrix-format to vector-format
x_train = x_train.reshape(train_size, num_features)
x_test = x_test.reshape(test_size, num_features)

# Model variables
nodes = [num_features, 800, 400, num_classes] # input, hidden, output
epochs = 100

class Model:
    def __init__(self):
        # Weights (Matrices)
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=[nodes[0], nodes[1]], stddev=0.01), name="W1")
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=[nodes[1], nodes[2]], stddev=0.01), name="W2")
        self.W3 = tf.Variable(tf.random.truncated_normal(shape=[nodes[2], nodes[3]], stddev=0.01), name="W3")
        # Biases (Vectors)
        self.b1 = tf.Variable(tf.constant(0.0, shape=[nodes[1]]), name="b1")
        self.b2 = tf.Variable(tf.constant(0.0, shape=[nodes[2]]), name="b2")
        self.b3 = tf.Variable(tf.constant(0.0, shape=[nodes[3]]), name="b3")
        self.variables = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
        # Model variables
        self.learning_rate = 0.001
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = tf.losses.CategoricalCrossentropy()
        self.metric_fn = tf.metrics.CategoricalAccuracy()
        self.current_loss_val = None

    def get_variables(self):
        return {var.name: var.numpy() for var in self.variables}

    def get_num_trainable_parameter(self, nodes):
        num_weights = 0
        num_biases = 0
        last_layer_num_nodes = nodes[0]
        for idx, layer_num_nodes in enumerate(nodes[1:]):
            if idx == 0:
                print("Input to Hidden Layer:")
            elif idx == len(nodes)-1:
                print("Hidden to Output Layer:")
            else:
                print("Hidden to Hidden Layer")
            weights_layer = last_layer_num_nodes * layer_num_nodes
            biases_layer = layer_num_nodes
            print("\tWights: ", weights_layer)
            print("\tBiases: ", biases_layer)
            num_weights += weights_layer
            num_biases += biases_layer
            last_layer_num_nodes = layer_num_nodes
        trainable_parameters = num_weights + num_biases
        print("Overall trainable parameters: ", trainable_parameters)



    def predict(self, x):
        input_layer = x
        # h = (x*W1) + b1
        hidden_layer1 = tf.math.add(tf.linalg.matmul(input_layer, self.W1), self.b1)
        # using Relu activation function
        hidden_layer1_act = tf.nn.relu(hidden_layer1)
        hidden_layer2 = tf.math.add(tf.linalg.matmul(hidden_layer1_act, self.W2), self.b2)
        # using Relu activation function
        hidden_layer2_act = tf.nn.relu(hidden_layer2)
        output_layer = tf.math.add(tf.linalg.matmul(hidden_layer2_act, self.W3), self.b3)
        # using Softmax activation function to get probability values
        output_layer_act = tf.nn.softmax(output_layer)
        return output_layer_act

    def loss(self, y_true, y_pred):
        loss_val = self.loss_fn(y_true, y_pred)
        self.current_loss_val = loss_val.numpy()
        return loss_val

    def update_variables(self, x_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = self.predict(x_train)
            loss = self.loss(y_train, y_pred)
        gradients = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        return loss
    
    def compute_metrics(self, x, y):
        y_pred = self.predict(x)
        self.metric_fn.update_state(y, y_pred)
        metric_val = self.metric_fn.result()
        self.metric_fn.reset_states()
        return metric_val

    def fit(self, x_train, y_train, x_test, y_test, epochs=10):
        # Create empty lists to save loss and metric values
        train_losses, train_metrics = [], []
        test_losses, test_metrics = [], []
        # Iterate over number of epochs
        for epoch in range(epochs):
            # Compute loss and metric value for the train set
            train_loss = self.update_variables(x_train, y_train).numpy()
            train_metric = self.compute_metrics(x_train, y_train).numpy()
            train_losses.append(train_loss)
            train_metrics.append(train_metric)
            # Compute loss and metric value for the test set
            test_loss = self.loss(self.predict(x_test), y_test).numpy()
            test_metric = self.compute_metrics(x_test, y_test).numpy()
            test_losses.append(test_loss)
            test_metrics.append(test_metric)
            print("Epoch: ", epoch+1, " of ", epochs,
                    " - Train Loss: ", round(train_loss, 4),
                    " - Train Metric: ", round(train_metric, 4),
                    " - Test Loss: ", round(test_loss, 4),
                    " - Test Metric: ", round(test_metric, 4))
        #Visualization of loss and metric values
        display_convergence_error(train_losses, test_losses)
        display_convergence_acc(train_metrics, test_metrics)

    
    def evaluate(self, x, y):
        loss = self.loss(self.predict(x), y).numpy()
        metric = self.compute_metrics(x, y).numpy()
        print("Loss: ", round(loss, 4), " Metric: ", round(metric, 4))

model = Model()
model.get_num_trainable_parameter(nodes)
model.fit(x_train, y_train, x_test, y_test, epochs=epochs)
model.evaluate(x_test, y_test)

        

        



    