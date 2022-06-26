import math
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line


class DataGenerator:
    """
    Generates images of different shapes and sizes
    """

    def __init__(self,
                 n,
                 dataset_size=100,
                 l_lower_frac=1 / 5,
                 l_higher_frac=2 / 5,
                 width_lower_frac=1 / 50,
                 width_higher_frac=1 / 25,
                 centering=False,
                 noise_percentage=0.01,
                 train_frac=0.70,
                 valid_frac=0.20,
                 test_frac=0.10) -> None:
        self.n = n
        self.dataset_size = dataset_size
        self.l_lower_frac = l_lower_frac
        self.l_higher_frac = l_higher_frac
        self.width_lower_frac = width_lower_frac
        self.width_higher_frac = width_higher_frac
        self.centering = centering
        self.noise_percentage = noise_percentage
        self.train_frac = train_frac
        self.valid_frac = valid_frac
        self.test_frac = test_frac

    def create_circle(self, flatten=False):
        """
        Creates circle
        """
        n = self.n  # n is image dimension
        img_array = np.full((n, n), 0, dtype=int)
        # r for radius
        r_lower_frac = self.l_lower_frac / 2
        r_higher_frac = self.l_higher_frac / 2
        if n < 25:
            r = random.randint(2, 3)
        else:
            r = random.randint(int(n * r_lower_frac), int(n * r_higher_frac))
        # width of the circle
        width = max(
            random.randint(int(n * self.width_lower_frac),
                           int(n * self.width_higher_frac)), 0.5)
        # max distance from centre
        max_dist_centre = int(n * r_higher_frac) + int(
            n * self.width_higher_frac)
        if self.centering:
            # x-coordinate of circle centre
            rx = n // 2
            # y-coordinate of circle centre
            ry = n // 2
        else:
            # x-coordinate of circle centre
            rx = random.randint(max_dist_centre + 1,
                                n - max(max_dist_centre, 1) - 2)
            # y-coordinate of circle centre
            ry = random.randint(max_dist_centre + 1,
                                n - max(max_dist_centre, 1) - 2)
        # Creating the circle
        for row in range(n):
            for col in range(n):
                dist = math.sqrt(((rx - row)**2) + ((ry - col)**2))
                if abs(dist - r) < width:
                    img_array[row, col] = 1
        # Adding noise to the image
        img_array = self.add_noise(img_array)
        # Flattening the image array if this option is set to True
        if flatten:
            return img_array.flatten()
        else:
            return img_array

    def create_square(self, flatten=False):
        """
        Creating square
        """
        n = self.n  # Image dimension
        img_array = np.full((n, n), 0, dtype=int)
        # Length of the sides
        l = random.randint(int(n * self.l_lower_frac),
                           int(n * self.l_higher_frac))
        width = random.randint(max(int(n * self.width_lower_frac), 1),
                               max(int(n * self.width_higher_frac), 1))
        if self.centering:
            # x-coordinate of top left corner of the square
            rx = int(n / 2 - l / 2)
            # y-coordinate of top left corner of the square
            ry = int(n / 2 - l / 2)
        else:
            # x-coordinate of top left corner of the square
            rx = random.randint(
                0, n - int(n * self.l_higher_frac) -
                int(n * self.width_higher_frac) - 1)
            # y-coordinate of top left corner of the square
            ry = random.randint(
                0, n - int(n * self.l_higher_frac) -
                int(n * self.width_higher_frac) - 1)
        # Generating the top line of the square
        for i in range(width):
            np.put(img_array[rx + i], list(range(ry, ry + l)), 1)
        # Generating the bottom line of the square
        for i in range(width):
            np.put(img_array[rx + l + i], list(range(ry, ry + l)), 1)
        # Generating the left line of the square
        for i in range(width):
            np.put(img_array[:, ry + i], list(range(rx, rx + l)), 1)
        # Generating the right line of the square
        for i in range(width):
            np.put(img_array[:, ry + l + i], list(range(rx, rx + l + width)),
                   1)
        # Adding noise to the image
        img_array = self.add_noise(img_array)
        # Flattening the image array if this option is set to True
        if flatten:
            return img_array.flatten()
        else:
            return img_array

    def create_cross(self, flatten=False):
        """
        Creating cross
        """
        n = self.n
        img_array = np.full((n, n), 0, dtype=int)
        l = random.randint(int(n * self.l_lower_frac),
                           int(n * self.l_higher_frac))
        width = random.randint(max(int(n * self.width_lower_frac), 1),
                               max(int(n * self.width_higher_frac), 1))
        if self.centering:
            # x-coordinate of centre of cross
            rx = n // 2
            # y-coordinate of centre of cross
            ry = n // 2
        else:
            # x-coordinate of centre of cross
            rx = random.randint(max(int(n * self.l_lower_frac), 1),
                                n - max(l // 2, 1) - 1)
            # y-coordinate of centre of cross
            ry = random.randint(max(int(n * self.l_lower_frac), 1),
                                n - max(l // 2, 1) - 1)
        # Generating the horizontal line of the cross
        for i in range(width):
            np.put(img_array[rx - width // 2 + i],
                   list(range(ry - l // 2, ry + l // 2 + 1)), 1)
        # Generating the vertical line of the cross
        for i in range(width):
            np.put(img_array[:, ry - width // 2 + i],
                   list(range(rx - l // 2, rx + l // 2 + 1)), 1)
        # Adding noise to the image
        img_array = self.add_noise(img_array)
        # Flattening the image array if this option is set to True
        if flatten:
            return img_array.flatten()
        else:
            return img_array

    def create_triangle(self, flatten=False):
        """
        Creating triangle
        """
        n = self.n
        img_array = np.full((n, n), 0, dtype=int)
        # Length of the sides
        l = random.randint(int(n * self.l_lower_frac),
                           int(n * self.l_higher_frac))
        width = random.randint(max(int(n * self.width_lower_frac), 1),
                               max(int(n * self.width_higher_frac), 1))
        if self.centering:
            # x-coordinate of top left corner
            rx = int(n / 2 - l / 2)
            # y-coordinate of top left corner
            ry = int(n / 2 - l / 2)
        else:
            # x-coordinate of top left corner
            rx = random.randint(
                0, n - int(n * self.l_higher_frac) -
                int(n * self.width_higher_frac) - 1)
            # y-coordinate of top left corner
            ry = random.randint(
                0, n - int(n * self.l_higher_frac) -
                int(n * self.width_higher_frac) - 1)
        # Generating the bottom line of the triangle
        for i in range(width):
            np.put(img_array[rx + l + i], list(range(ry, ry + l + width)), 1)
        # Generating the left line of the triangle
        for i in range(width):
            np.put(img_array[:, ry + i], list(range(rx, rx + l)), 1)
        # Generating the hypotenus of the triangle using line function from skimage
        rows, cols = line(rx, ry, rx + l, ry + l)
        for i in range(width):
            img_array[rows + i, cols] = 1
            img_array[rows, cols + i] = 1
        # Adding noise to the image
        img_array = self.add_noise(img_array)
        # Flattening the image array if this option is set to True
        if flatten:
            return img_array.flatten()
        else:
            return img_array

    def add_noise(self, img_array):
        """
        Adding noise to image using probability
        """
        for row in range(img_array.shape[0]):
            for col in range(img_array.shape[1]):
                img_array[row, col] = random.choices(
                    [img_array[row, col], 1 - img_array[row, col]],
                    weights=[1 - self.noise_percentage,
                             self.noise_percentage])[0]
        return img_array

    def show_image(self, img_array):
        """
        Displays an image given as a matrix
        """
        plt.imshow(img_array)
        plt.show()

    def get_shape(self, shape_num):
        """
        Converts from shape number to shape string
        """
        if shape_num == 0:
            return "circle"
        elif shape_num == 1:
            return "square"
        elif shape_num == 2:
            return "cross"
        elif shape_num == 3:
            return "triangle"

    def generate_random_image(self, num_images=1, flatten=False):
        """
        Generating image with random shape:
            circle = 0
            square = 1
            cross = 2
            triangle = 3
        """
        for _ in range(num_images):
            case_number = random.randint(0, 3)
            if case_number == 0:
                return (self.create_circle(flatten=flatten), case_number)
            elif case_number == 1:
                return (self.create_square(flatten=flatten), case_number)
            elif case_number == 2:
                return (self.create_cross(flatten=flatten), case_number)
            elif case_number == 3:
                return (self.create_triangle(flatten=flatten), case_number)

    def show_images(self, batch_x, batch_y, pred):
        """
        Displays images from an imageset
        """
        for i in range(len(batch_x)):
            case = batch_x[i]
            tmp = np.split(case, self.n)
            print("Target: " + str(batch_y[i]) + " (" +
                  str(self.get_shape(batch_y[i])) + ")")
            print("Prediction: " + str(pred[:, i]) + "\n")
            self.show_image(tmp)

    def generate_imageset(self, flatten=False):
        """
        Generates a set of images split into training, valid and test sets
        """
        all_cases = []
        for _ in range(self.dataset_size):
            all_cases.append(self.generate_random_image(flatten=flatten))
        train = all_cases[:round(self.dataset_size * self.train_frac)]
        valid = all_cases[round(self.dataset_size *
                                self.train_frac):round(self.dataset_size *
                                                       (self.train_frac +
                                                        self.valid_frac))]
        test = all_cases[round(self.dataset_size *
                               (self.train_frac + self.valid_frac)
                               ):round(self.dataset_size *
                                       (self.train_frac + self.valid_frac +
                                        self.test_frac))]
        return train, valid, test

    def unzip(self, image_set):
        """
        Method that takes in image set, that is a list of images on the format (img_arr, shape), and returns
        a list of img_arr and a list of corresponding shapes
        """
        image_set_shapes = list(map(list, zip(*image_set)))
        return np.array(image_set_shapes[0]), np.array(image_set_shapes[1])
