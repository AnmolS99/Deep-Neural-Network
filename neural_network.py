import numpy as np
import matplotlib.pyplot as plt
import config_parser
from layer import Layer
from datagen import DataGenerator
from activation_functions import sigmoid, sigmoid_der, relu, relu_der, softmax
from loss_functions import cross_entropy, cross_entropy_der, mse, mse_der

class NeuralNetwork:
    """
    A neural network consisting of an input layer, output layer and an optional amount of hidden layers
    """
    def __init__(self, num_features, layers, loss_func, num_classes, regularizer, reg_rate, include_softmax=True) -> None:
        self.layers = []
        prev_layer_neurons = num_features 
        
        # Adding all the layers
        for layer_neurons, layer_act_func, lr in layers:
            self.layers.append(Layer(prev_layer_neurons, layer_neurons, layer_act_func, lr))
            prev_layer_neurons = layer_neurons
        
        # Setting loss function and the derivative of the loss function
        self.loss_func = loss_func
        if loss_func == mse:
            self.loss_func_der = mse_der
        elif loss_func == cross_entropy:
            self.loss_func_der = cross_entropy_der

        self.num_classes = num_classes
        self.regularizer = regularizer
        self.reg_rate = reg_rate
        self.include_softmax = include_softmax
    

    def forward_pass(self, minibatch_x, minibatch_y):

        # Multiplying the input neurons with the weights to get the sum into every neuron for every case
        # The resulting matrix has a column for each case in the minibatch, and a row for each neuron in
        # this layer (the first layer)
        sum_1 = np.einsum("ij,ki->jk", self.layers[0].in_weights, minibatch_x)

        # Adding the biases
        sum_1 = sum_1 + self.layers[0].biases

        # Saving the sum
        self.layers[0].sum = sum_1

        # Applying this layer's activation function
        self.layers[0].activations = self.layers[0].act_func(sum_1)

        # If we have more than 1 layers (layers in this context meaning hidden layers + output layer)
        if len(self.layers) > 1:

            for i in range(1, len(self.layers)):
                
                prev_layer_activations = self.layers[i-1].activations

                # Multiplying previous layer activations with current layer in-weights to get the sum
                sum = np.einsum("ij,ik->jk", self.layers[i].in_weights, prev_layer_activations)

                # Adding the biases
                sum = sum + self.layers[i].biases

                # Saving the sum into this layer
                self.layers[i].sum = sum

                # Applying this layer's activation function
                self.layers[i].activations = self.layers[i].act_func(sum)

        if self.include_softmax:
            output = softmax(self.layers[-1].activations)
        else:
            output = self.layers[-1].activations

        return output, self.loss_func(output, self.one_hot(minibatch_y))
    

    def backward_pass(self, output, minibatch_x, minibatch_y):
        num_cases = output.shape[1]

        # Assuming softmax layer, so output will be the output from softmax layer
        if self.include_softmax:

            # Computing the initial jacobian J_L_S
            j_l_s =  self.loss_func_der(output, self.one_hot(minibatch_y))

            # Computing the jacobian J_S_Z where S stands for the softmax output, and Z is
            # the output of the final layer (output layer)
            
            # Iterating through the cases
            j_s_z = np.empty((num_cases, output.shape[0], output.shape[0]))
            for case in range(num_cases):
                s_vector = output[:,case]

                # Creating the J_soft jacobian for each case
                j_s_z_tmp = np.zeros((len(s_vector), len(s_vector)))

                # Computing the effect of z_i on s_j
                for i in range(len(s_vector)):
                    s_i = s_vector[i]
                    for j in range(len(s_vector)):
                        s_j = s_vector[j]
                        if i == j:
                            j_s_z_tmp[j, i] = s_i - s_i**2
                        else:
                            j_s_z_tmp[j, i] = -s_j*s_i
                
                # Adding each J_soft of each case in the minibatch
                j_s_z[case] = j_s_z_tmp
            
            # Computing the initial jacobian j_l_z, where eac row represents that case's j_l_z
            # Iterating through the cases
            j_l_z = np.empty((num_cases, output.shape[0]))
            for case in range(num_cases):
                j_l_s_case = j_l_s[:, case]
                j_s_z_case = j_s_z[case]
                j_l_z[case] = np.dot(j_l_s_case.T, j_s_z_case)

        else:
            # Computing the initial jacobian j_l_z, where eac row represents that case's j_l_z
            j_l_z =  self.loss_func_der(output, self.one_hot(minibatch_y))
            
        for n in range((len(self.layers) - 1), -1, -1):
            
            # Finding j_z_w
            if n == 0:
                y = minibatch_x.T
            else:
                y = self.layers[n-1].activations
            
            j_z_sum_diag = self.layers[n].der_act_func(self.layers[n].sum).T
            j_z_sum = np.eye(j_z_sum_diag.shape[1]) * j_z_sum_diag[:,np.newaxis,:]
            j_z_w = np.einsum("ik,kj->kij", y, j_z_sum_diag)
            
            # Calculating j_l_w
            j_l_w = np.einsum("kj,kij->kij", j_l_z, j_z_w)
            
            # Using j_l_w to update the weights
            self.layers[n].in_weights = self.layers[n].in_weights - self.layers[n].lr * (sum(self.regularization(j_l_w, n)) / len(j_l_w))
                
            # Finding j_z_w_b
            y_b = np.ones((1, num_cases))
            j_z_w_b = np.einsum("ik,kj->kj", y_b, j_z_sum_diag)

            # Calculating j_l_w_b
            j_l_w_b = np.einsum("kj,kj->kj", j_l_z, j_z_w_b)

            # Using j_l_w_b to update the weights
            self.layers[n].biases = self.layers[n].biases - self.layers[n].lr * np.array(sum(j_l_w_b) / len(j_l_w_b)).reshape(-1, 1)
            
            # Calculating j_z_y
            j_z_y = np.einsum("kii,ij->kij", j_z_sum, self.layers[n].in_weights.T)
            
            # Calculating j_l_y
            j_l_y = np.einsum("ki,kij->kj", j_l_z, j_z_y)
            
            # Passing the Jacobian of the loss with the respect to the prevoius layer, to the previous layer
            j_l_z = j_l_y
    
    def regularization(self, j_l_w, n):
        """
        Method that takes in a list of j_l_w matrices, and returns a list of regularized j_l_w matrices 
        """
        if self.regularizer == "l1":
            return j_l_w + (self.reg_rate * np.sign(self.layers[n].in_weights))
        elif self.regularizer == "l2":
            return j_l_w + (self.reg_rate * self.layers[n].in_weights)
        else:
            return j_l_w



    def one_hot(self, x):
        """
        Function that converts array of targets to array of one-hot-targets (which are arrays)
        """
        one_hot = np.eye(self.num_classes)[x]
        return one_hot



def test_data_images():
    cp = config_parser.ConfigParser("config_images.ini")
    dg, nn, epochs, batch_size = cp.create_nn()

    train, valid, test = dg.generate_imageset(flatten=True)
    batch_x, batch_y = dg.unzip(train)
    batch_valid_x, batch_valid_y = dg.unzip(valid)
    batch_test_x, batch_test_y = dg.unzip(valid)

    num_batches = (dg.dataset_size * dg.train_frac) // batch_size

    minibatches_x = np.split(batch_x, num_batches)
    minibatches_y = np.split(batch_y, num_batches)

    loss_train_list = []
    loss_valid_list = []
    loss_test_list = []
    for _ in range(epochs):
        for i in range(len(minibatches_x)):
            minibatch_x = minibatches_x[i]
            minibatch_y = minibatches_y[i]

            output, loss = nn.forward_pass(minibatch_x, minibatch_y)
            nn.backward_pass(output, minibatch_x, minibatch_y)
            loss_train_list.append(loss)

            output_valid, loss_valid = nn.forward_pass(batch_valid_x, batch_valid_y)
            loss_valid_list.append(loss_valid)
    
    output_test, loss_test = nn.forward_pass(batch_test_x, batch_test_y)

    print(output)
    print(minibatch_y)
    print("Average loss of test batch: " + str(loss_test))

    loss_train_list = np.array(loss_train_list)
    loss_valid_list = np.array(loss_valid_list)
    plt.plot(loss_train_list, label="Train")
    plt.plot(loss_valid_list, label="Validate")
    plt.xlabel("Minibatch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def test_xor():
    cp = config_parser.ConfigParser("config_xor.ini")
    dg, nn, epochs, batch_size = cp.create_nn()

    minibatch_xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    minibatch_xor_y = np.array([0, 1, 1, 0])

    for _ in range(1000):
        output2, loss2 = nn.forward_pass(minibatch_xor_x, minibatch_xor_y)
        nn.backward_pass(output2, minibatch_xor_x, minibatch_xor_y)
    print(output2)
    print(minibatch_xor_y)

if __name__ == "__main__":
    test_data_images()
    #test_xor()

    # MÅ IMPLEMENTERE RESTERENDE ACTIVATION FUNCTIONS
    # MÅ LEGGE TIL VERBOSE FLAG
    # MÅ LEGGE TIL WEIGHT RANGES I CONFIG FOR HVERT LAG