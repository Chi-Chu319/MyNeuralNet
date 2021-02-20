from sklearn import datasets
import matplotlib.pyplot as plt
from Functions import *
import numpy as np
from random import randint


iris = datasets.load_iris()
digits = datasets.load_digits()


'''
print(digits.data)  # features
print('\n')
print(digits.target)   # Ground truth
print('\n')
print(digits.images[0])  # image matrix

example = digits.images[0]
'''

'''
plt.imshow(example)
plt.colorbar()    # gives a color bar with respect to the numbers in the matrix
plt.show()   # must be added to show the image

plt.imshow(example, cmap="gray")   # show the image in gray scale
plt.show()

plt.imshow(example, cmap="gray_r", vmin=0, vmax=16) # reversed gray scale #range of values
plt.show()
'''

# Linear regression Max accuracy:
'''
No regularization:
learning_rate = 0.05
num_iter = 12000
training set acc 95.2
test set acc 86.6
with regularization:
training 93
test 87.8

mark: model is too simple,
bottleneck is rhe number of training set.
'''


num_iter = 12000
# splitting up training and test set
# print("shape of data" + str(digits.data.shape))
digits.data = digits.data.T
set_size = int(round(0.9 * digits.data.shape[1], 0))
# print('set size is ' + str(set_size))

training_x, training_y, test_x, test_y = digits.data[:, :set_size], digits.target[:set_size], digits.data[:, set_size:]\
    , digits.target[set_size:]


m = training_x.shape[1]
n = training_x.shape[0]
Layers_dims = (n, 10)
parameters = initialize_parameters(Layers_dims)
training_y = convert_into_binary(training_y)
test_y = convert_into_binary(test_y)
regularization_param = 0.0009
learning_rate = 0.05

# y_hat, caches = forward_prop(training_x, parameters)  # caches stores w, b, z
# # print('here is training y' + str(training_y))
# print('size of training y', str(training_y.shape))
# # print("here is y hat" + str(y_hat))
# print('here is the size of y hat', str(y_hat.shape))
# # print(np.multiply(training_y, np.log(y_hat)))
# cost = compute_cost(training_y, y_hat)


for i in range(0, num_iter):
    y_hat, caches = forward_prop(training_x, parameters)  # caches stores w, b, z
    # cost = compute_cost(training_y, y_hat)
    cost = compute_cost_regularized(training_y, y_hat, caches, regularization_param)
    if i == 0:
        print('cost for ' + str(i + 1) + 'th iteration is ' + str(cost))
    if (i+1) % 100 == 0:
        print('cost for ' + str(i + 1) + 'th iteration is ' + str(cost))
    #
    grads = back_prop_regularized(caches, training_x, training_y, regularization_param)
    parameters = update_param(parameters, grads, learning_rate)

training_accuracy, predict = model_performance(training_x, training_y, parameters)

print('training test set accuracy is ' + str(training_accuracy))

# performance on test set, compute accuracy
test_accuracy, prediction_y = model_performance(test_x, test_y, parameters)

print('test set accuracy is ' + str(test_accuracy))

test_y = convert_into_int(test_y)

# print(prediction_y)
# training_y = convert_into_int(training_y)
# print(training_y)

test_samples(digits.images[set_size:], prediction_y, test_y)

mislabelled_index = bad_samples(digits.images, prediction_y, test_y, set_size)

print(mislabelled_index)
