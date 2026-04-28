import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import IPython.display as ipd
from neuralnet import *
from layers import *
from losses import *

from torchvision import datasets

imgres = 28
num_classes = 10

train_dataset = datasets.MNIST(root="./data", train=True, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, download=True)

x_train = train_dataset.data.numpy().astype(float) / 255.0
x_test = test_dataset.data.numpy().astype(float) / 255.0
y_train = train_dataset.targets.numpy()
y_test = test_dataset.targets.numpy()

x_train = x_train.reshape(x_train.shape[0], imgres**2)
x_test = x_test.reshape(x_test.shape[0], imgres**2)

y_train = np.eye(num_classes)[y_train]


# -------------------------
# 1. Helper functions
# -------------------------
def predict(nn, X):
    y_pred = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        out = nn.forward(X[i])
        y_pred[i] = np.argmax(out)
    return y_pred

def accuracy(nn, X, y_idx):
    y_pred = predict(nn, X)
    return np.mean(y_pred == y_idx)

def train_sgd(nn, X, Y, X_test=None, y_test_idx=None, alpha=0.01, n_epochs=60):
    train_accs = []
    test_accs = []

    n = X.shape[0]

    for epoch in range(n_epochs):
        indices = np.random.permutation(n)

        for i in indices:
            nn.backprop_descent(X[i], Y[i], alpha)

        train_acc = accuracy(nn, X, np.argmax(Y, axis=1))
        train_accs.append(train_acc)

        if X_test is not None and y_test_idx is not None:
            test_acc = accuracy(nn, X_test, y_test_idx)
            test_accs.append(test_acc)
            print("Epoch", epoch + 1, "Train accuracy:", train_acc, "Test accuracy:", test_acc)
        else:
            print("Epoch", epoch + 1, "Train accuracy:", train_acc)

    return train_accs, test_accs


# -------------------------
# 2. Network 1
# Single hidden layer: 20 neurons -> softmax output
# -------------------------
nn1 = NeuralNet(x_train.shape[1], softmax_est_crossentropy_deriv)
nn1.add_layer(20, leaky_relu, leaky_relu_deriv)
nn1.add_layer(10, softmax, None)

train_accs_1, test_accs_1 = train_sgd(
    nn1, x_train, y_train, x_test, y_test,
    alpha=0.01, n_epochs=60
)

final_test_acc_1 = accuracy(nn1, x_test, y_test)
print("Final test accuracy, Network 1:", final_test_acc_1)


# -------------------------
# 3. Network 2
# Hidden layer: 20 neurons -> hidden layer: 40 neurons -> softmax output
# -------------------------
nn2 = NeuralNet(x_train.shape[1], softmax_est_crossentropy_deriv)
nn2.add_layer(20, leaky_relu, leaky_relu_deriv)
nn2.add_layer(40, leaky_relu, leaky_relu_deriv)
nn2.add_layer(10, softmax, None)

train_accs_2, test_accs_2 = train_sgd(
    nn2, x_train, y_train, x_test, y_test,
    alpha=0.01, n_epochs=60
)

final_test_acc_2 = accuracy(nn2, x_test, y_test)
print("Final test accuracy, Network 2:", final_test_acc_2)
