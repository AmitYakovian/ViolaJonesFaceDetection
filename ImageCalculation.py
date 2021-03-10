import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import time
from numba import njit


class IntegralImage:

    WINDOW_SIZE = (24, 24)

    @staticmethod
    def convert_image_to_numpy_array(image):
        data = np.array(image) / 255
        return data

    @staticmethod
    def open_image(img_path):
        return Image.open(img_path)

    @staticmethod
    def resize_image(img: Image.Image):
        resized = img.resize(IntegralImage.WINDOW_SIZE, Image.ANTIALIAS)
        return resized

    @staticmethod
    def covert_image_to_grayscale(image, coeff=2.2):
        """

        :return: an array with the same shape but of the gray image

        """
        # create an array of values between 0-1
        origin = np.array(image).astype(np.float32)
        # print(type(origin))
        # conversion of the algorithm
        num = origin ** (1. / coeff)
        # print("origin.shape[2]: ", origin.shape)
        # makes an average for the three numbers
        arr = np.sum(num, axis=2) / origin.shape[2]
        return Image.fromarray(arr)

    @staticmethod
    def show_grayscale_image(grayscale):
        plt.imshow(Image.fromarray(np.uint8(grayscale * 255.)), cmap="gray")
        plt.waitforbuttonpress()

    @staticmethod
    @njit
    def get_integral_image2(image_array: np.ndarray):
        integral_image = np.copy(image_array)
        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                integral_image[i][j] += (image_array[i, 0:j].sum() if j > 0 else 0) + (integral_image[i-1][j] if i > 0 else 0)

        return integral_image

    @staticmethod
    def get_integral_image(img: np.ndarray) -> np.ndarray:
        integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
        # return np.pad(integral, (1, 1), 'constant', constant_values=(0, 0))[:-1, :-1]
        return integral

    @staticmethod
    @njit
    def get_area_value(integral_image, top_left, width, height):

        # print("integarl:", top_left)

        width -= 1
        height -= 1
        tl = integral_image[top_left[0]-1, top_left[1]-1] if top_left[0] > 0 and top_left[1] >= 1 else 0
        tr = integral_image[top_left[0]-1, top_left[1]+width] if top_left[0] > 0 and top_left[1] + width <= integral_image.shape[1] else 0
        bl = integral_image[top_left[0]+height, top_left[1]-1] if top_left[0]+height < integral_image.shape[0] and top_left[1] >= 1 else 0
        br = integral_image[top_left[0]+height, top_left[1]+width]

        # print("tl:", tl, "tr:", tr, "bl:", bl, "br:", br)
        # print("num", tl + br - tr - bl)
        return tl + br - tr - bl


if __name__ == '__main__':

    b = np.ones((275, 183))
    d = np.ones((1000, 1000))
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    c = IntegralImage.get_integral_image(b)
    # print(a)
    before = time.time()
    print(IntegralImage.get_integral_image(b))
    print(time.time()- before)
    before = time.time()
    # print(IntegralImage.get_area_value(c, (1, 0), 2, 2))
    # print(time.time() - before)
