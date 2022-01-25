import math
import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line

class DataGenerator:
    """
    Generates images of different shapes
    """

    def __init__(self, n, dataset_size=100, l_lower_frac=1/5, l_higher_frac=2/5, width_lower_frac=1/50, width_higher_frac=1/25, centering=False, noise_percentage=0.01, train_frac = 0.70, valid_frac = 0.20, test_frac = 0.10) -> None:
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
        n = self.n
        img_array = np.full((n, n), 0, dtype=int)
        # r for radius
        r_lower_frac = self.l_lower_frac / 2
        r_higher_frac = self.l_higher_frac / 2
        if n < 25:
            r = random.randint(2, 3)
        else:
            r = random.randint(int(n*r_lower_frac), int(n*r_higher_frac))
        # width of the circle
        width = max(random.randint(int(n*self.width_lower_frac), int(n*self.width_higher_frac)), 0.5)
        # max distance from centre
        max_dist_centre = int(n*r_higher_frac) + int(n*self.width_higher_frac)
        if self.centering:
            # x-coordinate of circle centre
            rx = n//2
            # y-coordinate of circle centre
            ry = n//2
        else:
            # x-coordinate of circle centre
            rx = random.randint(max_dist_centre + 1, n - max(max_dist_centre, 1) - 2)
            # y-coordinate of circle centre
            ry = random.randint(max_dist_centre + 1, n - max(max_dist_centre, 1) - 2)
        for row in range(n):
            for col in range(n):
                dist = math.sqrt(((rx - row) ** 2) + ((ry - col) ** 2))
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
        n = self.n
        img_array = np.full((n, n), 0, dtype=int)
        # length of the sides
        l = random.randint(int(n*self.l_lower_frac), int(n*self.l_higher_frac))
        width = random.randint(max(int(n*self.width_lower_frac), 1), max(int(n*self.width_higher_frac), 1))
        if self.centering:
            # x-coordinate of top left corner
            rx = int(n/2 - l/2)
            # y-coordinate of top left corner
            ry = int(n/2 - l/2)
        else:
            # x-coordinate of top left corner
            rx = random.randint(0, n - int(n*self.l_higher_frac) - int(n*self.width_higher_frac) - 1)
            # y-coordinate of top left corner
            ry = random.randint(0, n - int(n*self.l_higher_frac) - int(n*self.width_higher_frac) - 1)
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
            np.put(img_array[:, ry + l + i], list(range(rx, rx + l + width)), 1)
        # Adding noise to the image
        img_array = self.add_noise(img_array)
        # Flattening the image array if this option is set to True
        if flatten:
            return img_array.flatten()
        else:
            return img_array
    
    def create_cross(self, flatten=False):
        n = self.n
        img_array = np.full((n, n), 0, dtype=int)
        l = random.randint(int(n*self.l_lower_frac), int(n*self.l_higher_frac))
        width = random.randint(max(int(n*self.width_lower_frac), 1), max(int(n*self.width_higher_frac), 1))
        if self.centering:
            # x-coordinate of centre of cross
            rx = n//2
            # y-coordinate of centre of cross
            ry = n//2
        else:
            # x-coordinate of centre of cross
            rx = random.randint(max(int(n*self.l_lower_frac), 1),  n - max(l//2, 1) - 1)
            # y-coordinate of centre of cross
            ry = random.randint(max(int(n*self.l_lower_frac), 1),  n - max(l//2, 1) - 1)
        # Generating the horizontal line of the square
        for i in range(width):
            np.put(img_array[rx - width//2 + i], list(range(ry - l//2, ry + l//2 + 1)), 1)
        # Generating the bottom line of the square
        for i in range(width):
            np.put(img_array[:, ry - width//2 + i], list(range(rx - l//2, rx + l//2 + 1)), 1)
        # Adding noise to the image
        img_array = self.add_noise(img_array)
        # Flattening the image array if this option is set to True
        if flatten:
            return img_array.flatten()
        else:
            return img_array
    
    def create_triangle(self, flatten=False):
        n = self.n
        img_array = np.full((n, n), 0, dtype=int)
        # length of the sides
        l = random.randint(int(n*self.l_lower_frac), int(n*self.l_higher_frac))
        width = random.randint(max(int(n*self.width_lower_frac), 1), max(int(n*self.width_higher_frac), 1))
        if self.centering:
            # x-coordinate of top left corner
            rx = int(n/2 - l/2)
            # y-coordinate of top left corner
            ry = int(n/2 - l/2)
        else:
            # x-coordinate of top left corner
            rx = random.randint(0, n - int(n*self.l_higher_frac) - int(n*self.width_higher_frac) - 1)
            # y-coordinate of top left corner
            ry = random.randint(0, n - int(n*self.l_higher_frac) - int(n*self.width_higher_frac) - 1)
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
        for row in range(img_array.shape[0]):
            for col in range(img_array.shape[1]):
                img_array[row, col] = random.choices([img_array[row, col], 	1-img_array[row, col]], weights=[1-self.noise_percentage, self.noise_percentage])[0]
        return img_array
    
    def show_image(self, img_array):
        plt.imshow(img_array)
        plt.show()

    def generate_random_image(self, flatten=False):
        """
        Generating image with random shape:
            circle = 0
            square = 1
            cross = 2
            triangle = 3
        """
        case_number = random.randint(0, 3)
        if case_number == 0:
            return (self.create_circle(flatten=flatten), case_number)
        elif case_number == 1:
            return (self.create_square(flatten=flatten), case_number)
        elif case_number == 2:
            return (self.create_cross(flatten=flatten), case_number)
        elif case_number == 3:
            return (self.create_triangle(flatten=flatten), case_number)

    def generate_imageset(self, flatten=False):
        all_cases = []
        for _ in range(self.dataset_size):
            all_cases.append(self.generate_random_image(flatten=flatten))
        train = all_cases[:round(self.dataset_size*self.train_frac)]
        valid = all_cases[round(self.dataset_size*self.train_frac):round(self.dataset_size*(self.train_frac + self.valid_frac))]
        test = all_cases[round(self.dataset_size*(self.train_frac + self.valid_frac)):round(self.dataset_size*(self.train_frac + self.valid_frac + self.test_frac))]
        return train, valid, test
    
    def unzip(self, image_set):
        """
        Method that takes in image set, that is a list of images on the format (img_arr, shape), and returns
        a list of img_arr and a list of corresponding shapes
        """
        image_set_shapes = list(map(list, zip(*image_set)))
        return np.array(image_set_shapes[0]), np.array(image_set_shapes[1])

dg = DataGenerator(n=10, dataset_size=100)
train, valid, test = dg.generate_imageset(flatten=True)
x, y = dg.unzip(test)
