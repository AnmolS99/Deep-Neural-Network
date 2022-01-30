import configparser

from activation_functions import relu, sigmoid
from loss_functions import cross_entropy, mse
import neural_network

class ConfigParser:

    def __init__(self, config_file) -> None:
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def parse_act_func(self, act_func):
        if act_func == "sigmoid":
            return sigmoid
        elif act_func == "relu":
            return relu
    
    def parse_loss_func(self, loss_func):
        if loss_func == "cross_entropy":
            return cross_entropy
        elif loss_func == "mse":
            return mse
    
    def create_nn(self):
        loss_func = self.parse_loss_func(self.config["globals"]["loss"])
        include_softmax = self.config["globals"]["include_softmax"].lower() == "true"
        num_features = int(self.config["globals"]["input"])
        num_classes = int(self.config["globals"]["num_classes"])

        layers = []
        for section in self.config.sections()[2:]:
            neurons = int(self.config[section]["neurons"]) 
            layer_act_func = self.parse_act_func(self.config[section]["activation_function"])
            layer_lr = float(self.config[section]["lr"])
            layers.append((neurons, layer_act_func, layer_lr))

        return neural_network.NeuralNetwork(num_features, layers, loss_func, num_classes, include_softmax)
    
if __name__ == "__main__":
    cp = ConfigParser("config_nn.ini")
    cp.create_nn()

    # BRUK SECTIONS TIL Å FINNE UT HVOR MANGE LAYERS DET ER