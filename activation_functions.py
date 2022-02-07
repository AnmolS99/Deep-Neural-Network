import numpy as np


def sigmoid(x):
    """
    Sigmoid function
    """
    return 1 / (1 + (np.exp(-x)))


def sigmoid_der(v):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(v) * (1 - sigmoid(v))


def relu(x):
    """
    ReLU function
    """
    return np.maximum(x, 0)


def relu_der(v):
    """
    Derivative of ReLU function
    """
    return np.where(v <= 0, 0, 1)


def linear(x):
    """
    Linear function
    """
    return x


def linear_der(x):
    """
    Derivative of linear function
    """
    return np.ones_like(x)


def tanh(x):
    """
    Tanh function
    """
    return np.tanh(x)


def tanh_der(x):
    """
    Derivative of tanh function
    """
    return 1 - np.square(tanh(x))


def softmax(x):
    """
    Softmax function
    """
    exp_matrix = np.exp(x)
    return exp_matrix / exp_matrix.sum(0)
