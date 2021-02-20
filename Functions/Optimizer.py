import numpy as np


# the optimizer parent class define all shared variables taken by all the optimization algorithm.
class Optimizer:
    def __init__(self, minibatch_size, learning_rate, print_epoch, print_iter):
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.print_epoch = print_epoch
        self.print_iter = print_iter


class Gradient_descent(Optimizer):
    def __init__(self, minibatch_size, learning_rate, print_epoch, print_iter):
        super(Gradient_descent, self).__init__(minibatch_size, learning_rate, print_epoch, print_iter)

    def update_param(self, parameters, grads, t):
        gradients = grads
        L = len(parameters) // 2

        for i in range(1, L + 1):
            w = parameters['w' + str(i)]
            b = parameters['b' + str(i)]
            gradw = gradients['dw' + str(i)]
            gradb = gradients['db' + str(i)]
            w = w - self.learning_rate * gradw
            b = b - self.learning_rate * gradb
            parameters['w' + str(i)] = w
            parameters['b' + str(i)] = b
        return parameters


# calculate the gradient and return it to the up date param function.
# V and S are predefined as 0 then updated using grads.
class GD_with_momentum(Optimizer):
    def __init__(self, minibatch_size, learning_rate, layer_num, print_epoch, print_iter, beta1=0.9):
        super(GD_with_momentum, self).__init__(minibatch_size, learning_rate, print_epoch, print_iter)
        # set the V and S initially to 0 ,dictionary
        self.V = {}
        for i in range(1, layer_num):
            self.V['Vdw' + str(i)] = 0
            self.V['Vdb' + str(i)] = 0
        self.beta1 = beta1

    def update_param(self, parameters, grads, t):
        L = len(parameters) // 2

        for i in range(1, L + 1):
            w = parameters['w' + str(i)]
            b = parameters['b' + str(i)]
            dw = grads['dw' + str(i)]
            db = grads['db' + str(i)]
            gradw = self.beta1 * self.V['Vdw' + str(i)] + (1 - self.beta1) * dw
            gradb = self.beta1 * self.V['Vdb' + str(i)] + (1 - self.beta1) * db
            # bias correction
            gradw = gradw / (1 - np.power(self.beta1, t))
            gradb = gradb / (1 - np.power(self.beta1, t))
            w = w - self.learning_rate * gradw
            b = b - self.learning_rate * gradb
            parameters['w' + str(i)] = w
            parameters['b' + str(i)] = b
        return parameters


class RMSprop(Optimizer):
    def __init__(self, minibatch_size, learning_rate, layer_num, print_epoch, print_iter, beta2=0.99, epsilon=10e-8):
        super(RMSprop, self).__init__(minibatch_size, learning_rate, print_epoch, print_iter)
        self.S = {}
        for i in range(1, layer_num):
            self.S['Vdw' + str(i)] = 0
            self.S['Vdb' + str(i)] = 0
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_param(self, parameters, grads, t):
        L = len(parameters) // 2

        for i in range(1, L + 1):
            w = parameters['w' + str(i)]
            b = parameters['b' + str(i)]
            dw = grads['dw' + str(i)]
            db = grads['db' + str(i)]
            gradw = self.beta2 * self.S['Vdw' + str(i)] + (1 - self.beta2) * np.power(dw, 2)
            gradb = self.beta2 * self.S['Vdb' + str(i)] + (1 - self.beta2) * np.power(db, 2)
            # bias correction
            gradw = gradw / (1 - np.power(self.beta2, t))
            gradb = gradb / (1 - np.power(self.beta2, t))
            w = w - self.learning_rate * dw/np.sqrt(gradw + self.epsilon)
            b = b - self.learning_rate * db/np.sqrt(gradb + self.epsilon)
            parameters['w' + str(i)] = w
            parameters['b' + str(i)] = b
        return parameters


class Adam(Optimizer):
    def __init__(self, minibatch_size, learning_rate, layer_num, print_epoch = -1, print_iter = -1, beta1=0.9, beta2=0.99, epsilon=10e-8):
        super(Adam, self).__init__(minibatch_size, learning_rate, print_epoch, print_iter)
        self.V = {}
        for i in range(1, layer_num):
            self.V['Vdw' + str(i)] = 0
            self.V['Vdb' + str(i)] = 0
        self.S = {}
        for i in range(1, layer_num):
            self.S['Vdw' + str(i)] = 0
            self.S['Vdb' + str(i)] = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_param(self, parameters, grads, t):
        L = len(parameters) // 2

        for i in range(1, L + 1):
            w = parameters['w' + str(i)]
            b = parameters['b' + str(i)]
            dw = grads['dw' + str(i)]
            db = grads['db' + str(i)]
            gradw1 = self.beta1 * self.V['Vdw' + str(i)] + (1 - self.beta1) * dw
            gradb1 = self.beta1 * self.V['Vdb' + str(i)] + (1 - self.beta1) * db
            gradw2 = self.beta2 * self.S['Vdw' + str(i)] + (1 - self.beta2) * np.power(dw, 2)
            gradb2 = self.beta2 * self.S['Vdb' + str(i)] + (1 - self.beta2) * np.power(db, 2)
            # bias correction
            gradw1 = gradw1 / (1 - np.power(self.beta2, t))
            gradb1 = gradb1 / (1 - np.power(self.beta2, t))
            gradw2 = gradw2 / (1 - np.power(self.beta2, t))
            gradb2 = gradb2 / (1 - np.power(self.beta2, t))
            w = w - self.learning_rate * gradw1 / np.sqrt(gradw2 + self.epsilon)
            b = b - self.learning_rate * gradb1 / np.sqrt(gradb2 + self.epsilon)
            parameters['w' + str(i)] = w
            parameters['b' + str(i)] = b
        return parameters
