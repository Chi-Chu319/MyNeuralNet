import numpy as np
from Functions.Activation import sigmoid


def initialize_parameters(layers_dims):
    L = len(layers_dims)
    parameters = {}
    for i in range(1, L):
        parameters['w' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])*0.01
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        return parameters


def forward_prop(training_x, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    caches = {}

    linear = np.dot(w1, training_x) + b1
    y_hat = sigmoid(linear)

    caches['w1'] = w1
    caches['b1'] = b1
    caches['z1'] = linear
    caches['a1'] = y_hat

    return y_hat, caches


def compute_cost(training_y, y_hat):
    m = training_y.shape[1]
    entropy = np.multiply(training_y, np.log(y_hat)) + np.multiply((1 - training_y), np.log(1 - y_hat))
    # print(entropy)
    # print('here is the shape of entropy', str(entropy.shape))
    cost = np.sum(-entropy)/m
    return cost


def back_prop(caches, training_x, training_y):
    grads = {}
    w1 = caches['w1']
    b1 = caches['b1']
    z1 = caches['z1']
    y_hat = caches['a1']
    m = y_hat.shape[1]

    dz1 = y_hat - training_y
    dw1 = np.dot(dz1, training_x.T)*(1/m)
    db1 = np.sum(dz1, axis=1, keepdims=True)/m

    grads['dw1'] = dw1
    grads['db1'] = db1

    return grads


def update_param(parameters, grads, learning_rate):
    w1 = parameters['w1']
    b1 = parameters['b1']

    dw1 = grads['dw1']
    db1 = grads['db1']

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1

    parameters['w1'] = w1
    parameters['b1'] = b1

    return parameters


def compute_cost_regularized(training_y, y_hat, caches, regularization_param):
    w1 = caches['w1']

    m = training_y.shape[1]
    entropy = np.multiply(training_y, np.log(y_hat)) + np.multiply((1 - training_y), np.log(1 - y_hat))
    # print(entropy)
    # print('here is the shape of entropy', str(entropy.shape))
    cost = np.sum(-entropy)/m + regularization_param * np.square(np.linalg.norm(w1))/(2 * m)
    return cost
