import numpy as np

class Layer:
    """
    A neural network layer, with a set of neurons and incoming weights
    """

    def __init__(self, prev_layer_neurons, neurons, act_func, der_act_func, lr) -> None:
        self.prev_layer_neurons = prev_layer_neurons
        self.neurons = neurons
        # Initializing the weights matrix with random weights between -0.5 and 0.5
        # To get the same weights all the time REMOVE IF NOT TESTING
        np.random.seed(42)
        self.in_weights = np.random.uniform(-0.5, 0.5, (prev_layer_neurons, neurons))

        self.biases = np.random.random((neurons, 1)) - 0.5
        self.activations = np.zeros(neurons)
        self.act_func = act_func
        self.der_act_func = der_act_func
        self.lr = lr
    
if __name__ == "__main__":
    pass