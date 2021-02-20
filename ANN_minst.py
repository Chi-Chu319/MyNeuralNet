import tensorflow as tf
from Functions.Function_ANN import *
from Functions.Performance import *
from Functions.Convert import *
from Functions.Optimizer import *
import os.path
import matplotlib.pyplot as plt

'''
bug log

1. convert to binary was not working functional.
   originally, len() was used inside the convert to binary method, which caused some dimension mis match for y
   the value expected for a single row of y is [0,1,0,0,0,0,0,0,0,0] which represents 1, whereas the method used to 
   return values such as [0,1,1,1,1,1,1,1,1,1]
   is has been fixed with shape()
   
2. the input feature x was not unrolled into vector correctly.
   previous code to reshape the x was x_tran.reshape(x_train.shape[2]*x_train.shape[1], x_train.shape[0])
   input (m, width, length)
   desired output (length*width, m) for a single image matrix, it should unroll it with width first.
   it turns out that the above code will not unroll the matrix in a desired way
   what .shape() returns are just real number values, the reshape() method will just one row by another read the value
   and kind squeeze it into the desired shape.
   therefore, x_train.reshape(x_train.shape[0],-1)
   with this, the function first unroll each single image into row vectors (m, width*length)
   then, take the transpose to shape them into column vectors.
   
3. when training using existing parameters, adam and other optimizer other than GD may not be good choices,
   because it uses bias correction to amplify the first few iterations, so, a surge of cost may be observed
   at the beginning, and only after, it will behave correctly.
   So, GD is actually a better option.
     
'''

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (60000,28,28), (60000,)
# x_train = x_train[:3000]
# y_train = y_train[:3000]
# x_test = x_test[:800]
# y_test = y_test[:800]

# print(np.max(x_train))# 255

# normalization
x_train = x_train/255
x_test = x_test/255

images = np.append(x_train, x_test, axis=0)

# reshape to fit the network
x_train = image_to_vector(x_train)# (784, 60000)
x_test = image_to_vector(x_test)# (10, 60000)
y_train = y_train.reshape(1, y_train.shape[0])# (784, 10000)
y_test = y_test.reshape(1, y_test.shape[0])# (10, 10000)

# convert y's into binary matrix
y_train = convert_into_binary(y_train, 10)
y_test = convert_into_binary(y_test, 10)


m = x_train.shape[1]
n = x_train.shape[0]
layer_dims = (n, 100, 10)
L = len(layer_dims)
# optimizers
#optimizer = Adam(layer_num=L, minibatch_size=2048, learning_rate=0.0005, print_epoch=20, print_iter=2)
optimizer = Gradient_descent(minibatch_size=2048, learning_rate=0.05, print_epoch=10, print_iter=2)
hidden_activation = 'relu'
output_activation = 'softmax'
# path to the weights
dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './mini_batch')
# folder_name ='.\\mini_batch\\'


# loading pre-trained weights from files
parameters = {}
if os.path.exists(os.path.join(dir_path, 'w1.txt')):
    for i in range(1, L):
        print("loading from pre-trained weights.")
        w_filename = os.path.join(dir_path, 'w' + str(i) + '.txt')
        b_filename = os.path.join(dir_path, 'b' + str(i) + '.txt')
        parameters['w' + str(i)] = np.loadtxt(w_filename, dtype=float)
        b = np.loadtxt(b_filename, dtype=float)
        parameters['b' + str(i)] = b.reshape(b.shape[0], 1)
else:
    print("no pre-trained weights.")


# training
parameters = L_layer_model(x_train, y_train, parameters=parameters, num_iter=1, layers_dims=layer_dims, regularization_param=0.0003,
                           hidden_activation=hidden_activation, output_activation=output_activation, optimizer=optimizer)

# thresholding the ys
y_train = convert_into_int_threshold(y_train, 1)
y_test = convert_into_int_threshold(y_test, 1)

# getting the performance info of training set
training_accuracy, predict = L_model_performance(x_train, y_train, parameters, hidden_activation=hidden_activation,
                                                 output_activation=output_activation)
print('training set accuracy is ' + str(training_accuracy))

# getting the performance info of test set
test_accuracy, prediction_y = L_model_performance(x_test, y_test, parameters, hidden_activation=hidden_activation,
                                                  output_activation=output_activation)
print('test set accuracy is ' + str(test_accuracy))

opt = input('Would you like to save the current model[y/n]: ')
if opt == 'y':
    for key, value in parameters.items():
        filename = os.path.join(dir_path, key + '.txt')
        np.savetxt(filename, value, fmt='%1.10f')
else:
    pass

mislabelled_index = bad_samples(images, prediction_y, y_test, m)
print('mislabelled data index: ' + str(mislabelled_index))

test_samples(images[m:], prediction_y, y_test)
