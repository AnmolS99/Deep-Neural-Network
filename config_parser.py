import configparser

from activation_functions import linear, relu, sigmoid, tanh
import datagen
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
        elif act_func == "linear":
            return linear
        elif act_func == "tanh":
            return tanh
    
    def parse_loss_func(self, loss_func):
        if loss_func == "cross_entropy":
            return cross_entropy
        elif loss_func == "mse":
            return mse
    
    def create_nn(self):
        # Parsing global variables
        loss_func = self.parse_loss_func(self.config["globals"]["loss"])
        include_softmax = self.config["globals"]["include_softmax"].lower() == "true"
        num_features = int(self.config["globals"]["input"])
        num_classes = int(self.config["globals"]["num_classes"])
        regularizer = self.config["globals"]["regularizer"].lower()
        reg_rate = float(self.config["globals"]["reg_rate"])
        epochs = int(self.config["globals"]["epochs"])
        batch_size = int(self.config["globals"]["batch_size"])

        # Parsing datagenerator variables
        image_dimension = int(self.config["data_generator"]["image_dimension"])
        dataset_size = int(self.config["data_generator"]["dataset_size"])
        l_lower_frac = float(self.config["data_generator"]["l_lower_frac"])
        l_higher_frac = float(self.config["data_generator"]["l_higher_frac"])
        width_lower_frac = float(self.config["data_generator"]["width_lower_frac"])
        width_higher_frac = float(self.config["data_generator"]["width_higher_frac"])
        centering = self.config["data_generator"]["centering"].lower() == "true"
        noise_percentage = float(self.config["data_generator"]["noise_percentage"])
        train_frac = float(self.config["data_generator"]["train_frac"])
        valid_frac = float(self.config["data_generator"]["valid_frac"])
        test_frac = float(self.config["data_generator"]["test_frac"])

        # Parsing the layer vairables for each layer
        layers = []
        for section in self.config.sections()[2:]:
            neurons = int(self.config[section]["neurons"]) 
            layer_act_func = self.parse_act_func(self.config[section]["activation_function"])
            layer_wr_lower = float(self.config[section]["wr_lower"])
            layer_wr_higher = float(self.config[section]["wr_higher"])
            layer_lr = float(self.config[section]["lr"])
            layers.append((neurons, layer_act_func, layer_wr_lower, layer_wr_higher, layer_lr))
        
        dg = datagen.DataGenerator(image_dimension, dataset_size, l_lower_frac, l_higher_frac, width_lower_frac, width_higher_frac, centering,
            noise_percentage, train_frac, valid_frac, test_frac)
        nn = neural_network.NeuralNetwork(num_features, layers, loss_func, num_classes, regularizer, reg_rate, include_softmax)
        return dg, nn, epochs, batch_size
    
if __name__ == "__main__":
    cp = ConfigParser("config_nn.ini")
    cp.create_nn()

    # BRUK SECTIONS TIL Å FINNE UT HVOR MANGE LAYERS DET ER