import numpy as np

def sigmoid(x):
    """
    Sigmoid function
    """
    return 1/(1+(np.exp(-x)))

def sigmoid_der(v):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(v)*(1-sigmoid(v))

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

def softmax(x):
    """
    Softmax function
    """
    exp_matrix = np.exp(x)
    return exp_matrix / exp_matrix.sum(0)


if __name__ == "__main__":
    pass