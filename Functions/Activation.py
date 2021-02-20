import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.maximum(0, x)

# activation sigmoid(Z) return A, activation cache(Z)
def activation_sigmoid(z):
    A = sigmoid(z)
    activation_cache = (z)
    return A, activation_cache

def activation_softmax(z):
    A = softmax(z)
    activation_cache = (z)
    return A, activation_cache


# activation relu(z) return A activation cache(Z)
def activation_relu(z):
    A = relu(z)
    activation_cache = z
    return A, activation_cache
