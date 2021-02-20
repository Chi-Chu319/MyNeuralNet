from sklearn import datasets

from Functions.Function_ANN import L_layer_model
from Functions.Performance import *
from Functions.Convert import convert_into_binary
from Functions.Convert import convert_into_int_threshold
from Functions.Optimizer import *


'''
2 layer ANN (64, 30, 10)
num_iter = 16000
learning rate = 0.05
training: 100%
test: 91.6%
2 layer ANN (64, 30, 10) with regularization
num_iter = 16000
regularization_param = 0.01
learning rate = 0.05
training:100%
test:91.6%
3 layer ANN (64, 30, 30, 10)
num_iter = 16000
learning rate = 0.05
training:100%
test:92.2%
3 layer ANN (64, 30, 30, 10) with regularization
num_iter = 16000
learning rate = 0.05
regularization_param = 0.01
training:100%
test:93.8%
3 layer ANN (64, 30, 30, 10) with regularization softmax
num_iter = 16000
learning rate = 0.05
regularization_param = 0.01
training:100%
test:94.4%
'''
# np.seterr(divide='ignore', invalid='ignore') # the only function it has is to turn off the warning

# pre_processing of data
digits = datasets.load_digits()
Digits = digits.data.T  # shape(n, m)
Targets = convert_into_binary(np.reshape(digits.target, (1, -1)), 10)  # shape(10, m)
images = digits.images  # shape(m. 8. 8)

# dimension setup
total_set_size = Digits.shape[1]
n = Digits.shape[0]
output_dim = Targets.shape[0]
set_size = int(round(total_set_size * 0.7, 0))

# data split
training_x, training_y, test_x, test_y = Digits[:, :set_size], Targets[:, :set_size], Digits[:, set_size:]\
    , Targets[:, set_size:]


layers_dims = (n, 30, 30, 10)
iter_num = 8000
#learning_rate = 0.05
regularization_param = 0.007
print_iter = 250
output_activation = 'softmax'
# optimizer = Gradient_descent(set_size, learning_rate=0.1)
# optimizer = GD_with_momentum(minibatch_size=set_size, learning_rate=1.2, layer_num=len(layers_dims))
# optimizer = RMSprop(minibatch_size=set_size, learning_rate=0.001, layer_num=len(layers_dims))
optimizer = Adam(minibatch_size=set_size, learning_rate=0.005, layer_num=len(layers_dims))

parameters = L_layer_model(training_x, training_y, parameters ={}, num_iter=iter_num, layer_dims=layers_dims, regularization_param=regularization_param,
                           hidden_activation='relu', output_activation=output_activation, optimizer=optimizer)

training_y = convert_into_int_threshold(training_y, 1)
test_y = convert_into_int_threshold(test_y, 1)

training_accuracy, predict = L_model_performance(training_x, training_y, parameters, hidden_activation='relu',
                                                 output_activation=output_activation)
print('training set accuracy is ' + str(training_accuracy))

test_accuracy, prediction_y = L_model_performance(test_x, test_y, parameters, hidden_activation='relu',
                                                  output_activation=output_activation)

print('test set accuracy is ' + str(test_accuracy))

mislabelled_index = bad_samples(digits.images, prediction_y, test_y, set_size)

print('mislabelled data index: ' + str(mislabelled_index))

test_samples(digits.images[set_size:], prediction_y, test_y)
