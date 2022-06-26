# Deep Neural Network 🧠

This project was part of the Deep Learning (IT3030) course at NTNU spring 2022. The aim of this project was to create a deep neural network library from scratch and implement the backpropagation algorithm. Part of the project was also to generate data and test the neural network on it, as well as defining a format for config-files where neural net and data format could be specified.

## Data generation

The data generated consists of 2D-images of four different shapes: circle, square, triangle and cross. It is possible to specify:

- The image dimensions, n x n pixels (works best for n between 10 and 50)
- The length and width range of shapes as fractions of the whole image (length fractions between 0.20 and 0.40, and width fractions between 0.02 and 0.04 works best)
- Noise percentage (how much random noise there should be in an image)
- Whether or not the shapes should be centered
- The size of the dataset
- The size of the train, valid and test set as fractions of the whole dataset

Examples of 20x20 images with 1% noise and without centering are given below:

### Circle

<img src="images/circle20.png" alt="drawing" width="200"/>

### Square

<img src="images/square20.png" alt="drawing" width="200"/>

### Triangle

<img src="images/triangle20.png" alt="drawing" width="200"/>

### Cross

<img src="images/cross20.png" alt="drawing" width="200"/>

## Configuration files ⚙️

In the configuration files it is possible to specify the data that the neural network is to be trained and tested on (more info [here](#data-generation)), as well as the neural network architecture. For the neural net it is possible to specify:

- Loss function (mean-squared-error or cross-entropy)
- Whether or not to have a softmax layer
- Number of different target classes in the data (default is 4 as the data contains four shapes)
- Regularizer (L1, L2 or None), as well as regularization rate
- Number of epochs for training
- Batch size

For each hidden layer + output layer, you can specify:

- Number of neurons
- Activation function (sigmoid, tanh, relu or linear)
- Initial weight ranges
- Learning rate

Config files are located in the [configs](/configs/) folder.

## Installation 📦

To install the required packages, use the following command: `pip install -r requirements.txt`

## Running a model

To run a model, it (and the data) first needs to be specified. This is done in a [configuration file](#configuration-files), in the [configs](/configs/) folder. You could either modify the `config_main.ini` file, or create your own config-file and specify the path to it in the `main()` function in the `main.py` file. To create, train and test the model, run the main.py file: `python main.py`.

## Results

" Not having centered shapes (as opposed to MNIST) made it much more difficult to predict "
