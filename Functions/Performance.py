import numpy as np
import matplotlib.pyplot as plt
from Functions.Convert import convert_into_int_threshold
from Functions.Convert import max_in_matrix
from Functions.Function_ANN import L_model_forward
from Functions.Function_Classifier import forward_prop
from random import randint


def back_prop_regularized(caches, training_x, training_y,  regularization_param):
    grads = {}
    y_hat = caches['a1']
    w1 = caches['w1']
    b1 = caches['b1']
    z1 = caches['z1']
    m = y_hat.shape[1]

    dz1 = y_hat - training_y
    dw1 = np.dot(dz1, training_x.T) * (1 / m) + regularization_param * w1
    db1 = np.sum(dz1, axis=1, keepdims=True)/m

    grads['dw1'] = dw1
    grads['db1'] = db1

    return grads



def test_samples(images, prediction_y, test_y):
    while True:
        Input = input('enter a number ranging from 1 to ' + str(len(prediction_y)) + 'to see the labelling: ')
        if Input == 'exit':
            break
        else:
            random = int(Input) - 1
        # random = randint(0, len(prediction_y)-1)  # random integer number
        example = images[random]
        prediction = prediction_y[random]

        title = 'prediction is ' + str(int(prediction)) + '; ground truth is ' + str(int(test_y[random]))

        plt.ion()  # turn the interactive mode on
        plt.imshow(example, cmap='gray_r')
        plt.title(title)
        plt.show()
        plt.pause(3)
        plt.close()


def model_performance(test_x, test_y, parameters):
    y_hat, caches = forward_prop(test_x, parameters)  # caches stores w, b, z
    prediction_y = convert_into_int_threshold(y_hat, 1)
    test_y = convert_into_int_threshold(test_y, 1)

    difference = prediction_y - test_y
    num_dif = 0
    for i in range(0, len(test_y)):
        if difference[i] != 0:
            num_dif += 1

    accuracy = str((1 - num_dif / len(test_y)) * 100) + '%'
    return accuracy, prediction_y


# L model performance(X, Y, parameters) return accuracy
def L_model_performance(X, Y, parameters, hidden_activation, output_activation):
    y_hat, caches = L_model_forward(X, parameters, hidden_activation, output_activation)
    if output_activation == 'sigmoid':
        # convert into int using threshold
        y_hat = convert_into_int_threshold(y_hat, threshold=0.5)
    elif output_activation == 'softmax':
        # select the max one and convert into int
        y_hat = max_in_matrix(y_hat, 0)
        y_hat = convert_into_int_threshold(y_hat, 1)

    difference = y_hat - Y
    num_dif = 0
    for i in range(0, len(Y)):
        if difference[i] != 0:
            num_dif += 1

    accuracy = str((1 - num_dif / len(Y)) * 100) + '%'
    return accuracy, y_hat


def bad_samples(images, prediction_y, test_y, set_size):
    difference = test_y - prediction_y
    list_images = []
    list_prediction = []
    list_GT = []
    list_index = []

    for i in range(0, len(difference)):
        if difference[i] != 0:
            list_images.append(images[i+set_size])
            list_prediction.append(prediction_y[i])
            list_GT.append(test_y[i])
            list_index.append(i + 1)

    rand1 = randint(0, len(list_prediction) - 1)  # random integer number
    rand2 = randint(0, len(list_prediction) - 1)  # random integer number
    rand3 = randint(0, len(list_prediction) - 1)  # random integer number
    title1 = 'prediction is ' + str(int(list_prediction[rand1])) + '; ground truth is ' + str(int(list_GT[rand1]))
    title2 = 'prediction is ' + str(int(list_prediction[rand2])) + '; ground truth is ' + str(int(list_GT[rand2]))
    title3 = 'prediction is ' + str(int(list_prediction[rand3])) + '; ground truth is ' + str(int(list_GT[rand3]))

    # print(len(list_images))
    # print(list_images[1].shape)
    # print(rand1)
    # print(list_images[1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))  # size of subgraph
    ax1.imshow(list_images[rand1], cmap='gray_r')
    ax1.set_title(title1)
    ax2.imshow(list_images[rand2], cmap='gray_r')
    ax2.set_title(title2)
    ax3.imshow(list_images[rand3], cmap='gray_r')
    ax3.set_title(title3)
    plt.suptitle('mislabelled data')
    plt.show()
    plt.pause(3)

    return list_index
