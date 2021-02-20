import numpy as np
from Functions.Activation import *
import math

# np.seterr(divide='ignore', invalid='ignore')  # prevent runningtime warnings


# ANN functions
# initialize parameters(layer dims) return parameters : finished
def L_initialize_parameters(layers_dims):
    L = len(layers_dims)
    # print('len of layer_dims ' + str(L))
    parameters = {}
    for i in range(1, L):
        parameters['w' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])*0.01
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        # print('i: ' + str(i))
        # print('w: ' + str(parameters['w' + str(i)].shape))
        # print('b: ' + str(parameters['b' + str(i)].shape))
    return parameters


# linear forward(A_prev, w, b) return Z, linear cache(A_prev, w, b)
def linear_forward(A_prev, w, b):
    z = np.dot(w, A_prev) + b
    linear_cache = (A_prev, w, b)
    return z, linear_cache


# linear activation forward(A_prev, w, b, activation_type) return A, cache(linear cache, activation cache)
def linear_activation_forward(A_prev, w, b, activation_type):
    if activation_type == 'softmax':
        z, linear_cache = linear_forward(A_prev, w, b)
        a, activation_cache = activation_softmax(z)
        cache = (linear_cache, activation_cache)
        return a, cache
    elif activation_type == 'sigmoid':
        z, linear_cache = linear_forward(A_prev, w, b)
        a, activation_cache = activation_sigmoid(z)
        cache = (linear_cache, activation_cache)
        return a, cache
    elif activation_type == 'relu':
        z, linear_cache = linear_forward(A_prev, w, b)
        a, activation_cache = activation_relu(z)
        cache = (linear_cache, activation_cache)
        return a, cache


# L model forward(X, parameters) return y_hat, caches(*cache, cache)
def L_model_forward(X, parameters, hidden_activation, output_activation):
    # print('w1: ' + str(parameters['w1'].shape))
    # print('b1: ' + str(parameters['b1'].shape))
    # print('w2: ' + str(parameters['w2'].shape))
    # print('b2: ' + str(parameters['b2'].shape))
    L = len(parameters) // 2
    A = X
    caches = ()

    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['w' + str(i)], parameters['b' + str(i)], hidden_activation)
        caches = (*caches, cache)

    AL, cache = linear_activation_forward(A, parameters['w' + str(L)], parameters['b' + str(L)], output_activation)
    caches = (*caches, cache)
    return AL, caches


# compute cost_regularized(AL, y, parameters, regularization_param) return cost
def L_compute_cost_regularized(AL, y, parameters, regularization_param, output_activation):
    # return divide by zero encountered in log when try to do np.log(0)
    # some elements in AL or (1-AL) may be 0 after gradient descent.

    m = len(parameters) // 2
    frobenius_norm_squared = 0
    if output_activation == 'sigmoid':
        cross_entropy = np.multiply(y, np.log(AL)) + np.multiply((1-y), np.log(1 - AL))
    elif output_activation == 'softmax':
        cross_entropy = np.multiply(y, np.log(AL))

    for i in range(1, m+1):
        frobenius_norm_squared += np.sum(np.square(parameters['w' + str(i)]))

    frobenius_norm = np.sqrt(frobenius_norm_squared)

    cost = np.sum(-cross_entropy)/m + regularization_param * frobenius_norm / (2 * m)
    return cost


# linear backward_regularized(dz, linear cache, regularization param) return da, dw, db
def linear_backward_regularized(dz, linear_cache, regularizarion_param):
    A_prev = linear_cache[0]
    m = A_prev.shape[1]
    w = linear_cache[1]
    b = linear_cache[2]

    dw = np.dot(dz, A_prev.T)/m + regularizarion_param * w
    da_prev = np.dot(w.T, dz)
    db = np.sum(dz, axis=1, keepdims=True)/m
    return da_prev, dw, db


# activation sigmoid backward(da, activation cache) return dz
def activation_sigmoid_backward(da, activation_cache):
    z = activation_cache
    dz = np.multiply(da, np.multiply(sigmoid(z), 1 - sigmoid(z)))
    return dz


def activation_softmax_backward(da, activation_cache):
    z = activation_cache
    dz = np.multiply(da, np.multiply(softmax(z), 1 - softmax(z)))
    return dz


# activation relu backward(da, activation cache) return dz
def activation_relu_backward(da, activation_cache):
    z = activation_cache
    dz = np.multiply(da, np.int64(relu(z) > 0))
    return dz


# linear activation backward regularized(da, cache, activation_type, regularization_param) return grads{}
def linear_activation_backward(da, cache, activation_type, regularization_param):
    linear_cache, activation_cache = cache[0], cache[1]

    if activation_type == 'softmax':
        dz = activation_softmax_backward(da, activation_cache)
        da_prev, dw, db = linear_backward_regularized(dz, linear_cache, regularization_param)
        return da_prev, dw, db
    elif activation_type == 'sigmoid':
        dz = activation_sigmoid_backward(da, activation_cache)
        da_prev, dw, db = linear_backward_regularized(dz, linear_cache, regularization_param)
        return da_prev, dw, db
    if activation_type == 'relu':
        dz = activation_relu_backward(da, activation_cache)
        da_prev, dw, db = linear_backward_regularized(dz, linear_cache, regularization_param)
        return da_prev, dw, db


# L model backward(y_hat, Y, caches, regularization_param) return grads
def L_model_backward(y_hat, Y, caches, regularization_param, hidden_activation, output_activation):
    L = len(caches)
    dAL = - np.divide(Y, y_hat) + np.divide((1 - Y), (1 - y_hat))
    grads = {}

    da_prev, dw, db = linear_activation_backward(dAL, caches[L - 1], output_activation, regularization_param)
    grads['da' + str(L - 1)] = da_prev
    grads['dw' + str(L)] = dw
    grads['db' + str(L)] = db

    for i in reversed(range(0, L-1)):
        da_prev, dw, db = linear_activation_backward(da_prev, caches[i], hidden_activation, regularization_param)
        grads['da' + str(i)] = da_prev
        grads['dw' + str(i + 1)] = dw
        grads['db' + str(i + 1)] = db
    return grads


# L layer model(X, Y, num_iter, layers_dims, learning_rate, regularization param) return parameters
def L_layer_model(X, Y, parameters, num_iter, layer_dims, regularization_param, hidden_activation, output_activation, optimizer):
    if parameters == {}:
        parameters = L_initialize_parameters(layer_dims)
    m = X.shape[1]
    K = 1
    cost = 0
    print_epoch = optimizer.print_epoch
    print_iter = optimizer.print_iter
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    mini_batch_size = optimizer.minibatch_size
    mini_batch_num = math.floor(m/mini_batch_size)

    for k in range(0, mini_batch_num):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_num * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, mini_batch_num * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    for i in range(1, num_iter + 1):
        for mini_batch in mini_batches:
            X, Y = mini_batch
            AL, caches = L_model_forward(X, parameters, hidden_activation, output_activation)
            min_nonzero = np.min(AL[np.nonzero(AL)])  # replace zero values in an array with lowest value in the
            # array except for the 0
            AL[AL == 0] = min_nonzero
            max_nonone = np.max(AL[np.nonzero(1 - AL)])  # replace one values in an array with highest value in the
            # array except for the 1
            AL[AL == 1] = max_nonone
            cost = L_compute_cost_regularized(AL, Y, parameters, regularization_param, output_activation)

            if print_epoch != 0:
                if K == 1 or K % print_epoch == 0:
                    print('cost for the ' + str(K) + 'th epoch is ' + str(cost))

            grads = L_model_backward(AL, Y, caches, regularization_param, hidden_activation, output_activation)
            parameters = optimizer.update_param(parameters, grads, i)

            K += 1

        if print_iter != 0:
            if i == 1 or i % print_iter == 0:
                print('cost for the ' + str(i) + 'th iteration is ' + str(cost))

    return parameters


