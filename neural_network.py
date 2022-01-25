from re import S
import numpy as np
from layer import Layer
from datagen import DataGenerator
from activation_functions import sigmoid, sigmoid_der, relu, relu_der, softmax
from loss_functions import cross_entropy, cross_entropy_der, mse, mse_der

class NeuralNetwork:
    """
    A neural network consisting of an input layer, output layer and an optional amount of hidden layers
    """

    def __init__(self, num_features, layers, loss_func, loss_func_der, num_classes, include_softmax=True) -> None:
        self.layers = []
        prev_layer_neurons = num_features 
        
        # Adding all the layers
        for layer_neurons, layer_act_func, layer_der_act_func, lr in layers:
            self.layers.append(Layer(prev_layer_neurons, layer_neurons, layer_act_func, layer_der_act_func, lr))
            prev_layer_neurons = layer_neurons
    
        self.loss_func = loss_func
        self.loss_func_der = loss_func_der
        self.num_classes = num_classes
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
            j_l_w = np.empty((num_cases, self.layers[n].prev_layer_neurons, self.layers[n].neurons))
            for case in range(num_cases):
                j_l_z_case =  j_l_z[case]
                j_z_w_case = j_z_w[case]
                j_l_w[case] = j_l_z_case * j_z_w_case
            
            # Using J_L_W to update the weights
            # Iterating through the cases
            for case in range(num_cases):
                j_l_w_case = j_l_w[case]
                self.layers[n].in_weights = self.layers[n].in_weights - self.layers[n].lr * j_l_w_case
                
            # Finding j_z_w_b
            # Iterating through the cases
            j_z_w_b = np.empty((num_cases, self.layers[n].neurons))
            for case in range(num_cases):
                y_case = 1
                j_z_sum_case = np.diag(self.layers[n].der_act_func(self.layers[n].sum[:, case]))
                j_z_w_b_case = np.outer(y_case, np.diag(j_z_sum_case))
                j_z_w_b[case] = j_z_w_b_case

            # Calculating j_l_w_b
            j_l_w_b = np.empty((num_cases, self.layers[n].neurons))
            for case in range(num_cases):
                j_l_z_case =  j_l_z[case]
                j_z_w_b_case = j_z_w_b[case]
                j_l_w_b[case] = j_l_z_case * j_z_w_b_case
            
            # Using j_l_w_b to update the weights
            # Iterating through the cases
            for case in range(num_cases):
                j_l_w_b_case = j_l_w_b[case]
                self.layers[n].biases = self.layers[n].biases - self.layers[n].lr * j_l_w_b_case.reshape(-1, 1)
            
            # Calculating j_z_y
            # Iterating through the cases
            j_z_y = np.empty((num_cases, self.layers[n].neurons, self.layers[n].prev_layer_neurons))
            for case in range(num_cases):
                j_z_sum_case = np.diag(self.layers[n].der_act_func(self.layers[n].sum[:, case]))
                j_z_y_case = np.dot(j_z_sum_case, self.layers[n].in_weights.T)
                j_z_y[case] = j_z_y_case
            
            # Calculating j_l_y
            # Iterating through the cases
            j_l_y = np.empty((num_cases, self.layers[n].prev_layer_neurons))
            for case in range(num_cases):
                j_l_z_case = j_l_z[case]
                j_z_y_case = j_z_y[case]
                j_l_y[case] = np.dot(j_l_z_case, j_z_y_case)
            
            # Passing the Jacobian of the loss with the respect to the prevoius layer, to the previous layer
            j_l_z = j_l_y



    def one_hot(self, x):
        """
        Function that converts array of targets to array of one-hot-targets (which are arrays)
        """
        one_hot = np.eye(self.num_classes)[x]
        return one_hot

if __name__ == "__main__":
    #n = 10
    #nn = NeuralNetwork(num_features=n**2, hidden_layers=[(5, sigmoid, sigmoid_der)], output_layer_neurons=4, 
    #    output_layer_act_func=sigmoid, output_layer_der_act_func=sigmoid_der, loss_func=cross_entropy, 
    #    loss_func_der=cross_entropy_der, num_classes=4, include_softmax=True, lr=0.5)

    #dg = DataGenerator(n, dataset_size=10)
    #train, valid, test = dg.generate_imageset(flatten=True)
    #minibatch_x, minibatch_y = dg.unzip(train)
    
    #for _ in range(10000):
    #    output, loss = nn.forward_pass(minibatch_x, minibatch_y)
    #    bp = nn.backward_pass(output, minibatch_x, minibatch_y)
    #print(output)

    nn2 = NeuralNetwork(num_features=2, layers=[(2, sigmoid, sigmoid_der, 0.5), (1, sigmoid, sigmoid_der, 0.5)], 
        loss_func=mse, loss_func_der=mse_der, num_classes=2, include_softmax=False)

    minibatch_xor_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    minibatch_xor_y = np.array([0, 1, 1, 0])

    for _ in range(10000):
        output2, loss2 = nn2.forward_pass(minibatch_xor_x, minibatch_xor_y)
        nn2.backward_pass(output2, minibatch_xor_x, minibatch_xor_y)
    print(output2)

