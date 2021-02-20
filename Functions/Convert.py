import numpy as np


def convert_into_binary(y, nL):
    binary_y = np.zeros((nL, y.shape[1]))
    for i in range(0, y.shape[1]):
        binary_y[:, i][y[:, i]] = 1
    return binary_y


def convert_into_int_threshold(y_hat, threshold): # i is the index of row; j is the index of column
    number_matrix = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    for i in range(0, y_hat.shape[0]):
        for j in range(0, y_hat.shape[1]):
            y_hat[i,j] = 1 if y_hat[i, j] >= threshold else 0  #if statement in one line
    prediction = np.zeros(y_hat.shape[1])
    for i in range(0, y_hat.shape[1]):
        prediction[i] = y_hat[:, i].dot(number_matrix)
    return prediction


def max_in_matrix(x, dimension): # dimension is in which direction the function take the maximum value (1,0)
    length = x.shape[1 - dimension]
    x_max = np.zeros(x.shape)

    horizontal, vertical = np.where(x == np.amax(x, axis=dimension))

    for i in range(length):
        x_max[horizontal[i], vertical[i]] = 1
    return x_max


# input size should be (m, height, length)
# output size should be (length*height, m)
def image_to_vector(X):
    return X.reshape(X.shape[0], -1).T

