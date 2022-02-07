import numpy as np
from activation_functions import linear_der, sigmoid, sigmoid_der, relu, relu_der, linear, tanh, tanh_der


class Layer:
    """
    A neural network layer, with a set of neurons and incoming weights
    """

    def __init__(self, prev_layer_neurons, neurons, act_func, wr_lower,
                 wr_higher, lr) -> None:
        self.prev_layer_neurons = prev_layer_neurons
        self.neurons = neurons
        self.in_weights = np.random.uniform(wr_lower, wr_higher,
                                            (prev_layer_neurons, neurons))

        self.biases = np.random.uniform(wr_lower, wr_higher, (neurons, 1))
        self.activations = np.zeros(neurons)
        self.act_func = act_func

        self.der_act_func = None
        if act_func == sigmoid:
            self.der_act_func = sigmoid_der
        elif act_func == relu:
            self.der_act_func = relu_der
        elif act_func == linear:
            self.der_act_func = linear_der
        elif act_func == tanh:
            self.der_act_func = tanh_der

        self.lr = lr
