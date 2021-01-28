import numpy as np
import ImageCalculation


class Feature:
    def __init__(self, x_pos, y_pos, width, height, weight=None):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.height = height
        self.weight = weight

    def get_prediction(self, integral_image: np.ndarray):

        print(integral_image)
        print(self.structure)

        white_sum = sum([ImageCalculation.IntegralImage.get_area_value(integral_image, (area[0][0]+self.x_pos, area[1][0]+self.y_pos), self.rect_width, self.rect_height) for area in self.structure["white"]])
        black_sum = sum([ImageCalculation.IntegralImage.get_area_value(integral_image, (area[0][0]+self.x_pos, area[1][0]+self.y_pos), self.rect_width, self.rect_height) for area in self.structure["black"]])

        print(white_sum, black_sum)
        return white_sum - black_sum


class Feature2v(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None): 
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width // 2
        self.rect_height = height
        # self.structure = {"white": [((0, self.width // 2), (0, self.height))],
        #                   "black": [((self.width // 2 , self.width), (0, self.height))]}
        self.structure = {"white": [((0, self.height), (0, self.width // 2))],
                          "black": [((0, self.height), (self.width // 2 , self.width))]}

    @staticmethod
    def check_arguments(width, height):
        return width % 2 == 0

class Feature2v2(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None): 
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width // 2
        self.rect_height = height
        # self.structure = {"white": [((self.width // 2 , self.width), (0, self.height))],
        #                   "black": [((0, self.width // 2), (0, self.height))]}
        self.structure = {"white": [((0, self.height), (self.width // 2 , self.width))],
                          "black": [((0, self.height), (0, self.width // 2))]}

    @staticmethod
    def check_arguments(width, height):
        return width % 2 == 0


class Feature2h(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None):
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width 
        self.rect_height = height // 2
        # self.structure = {"white": [((0, self.width), (0, self.height//2))],
        #                   "black": [((0, self.width), (self.height//2, self.height))]}
        self.structure = {"white": [((0, self.height//2), (0, self.width))],
                          "black": [((self.height//2, self.height), (0, self.width))]}

    @staticmethod
    def check_arguments(width, height):
        return height % 2 == 0

class Feature2h2(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None):
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width 
        self.rect_height = height // 2
        # self.structure = {"white": [((0, self.width), (self.height//2, self.height))],
        #                   "black": [((0, self.width), (0, self.height//2))]}
        self.structure = {"white": [((self.height//2, self.height), (0, self.width))],
                          "black": [((0, self.height//2), (0, self.width))]}

    @staticmethod
    def check_arguments(width, height):
        return height % 2 == 0




class Feature3h(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None):
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width 
        self.rect_height = height // 3
        # self.structure = {"white": [((0, self.width), (0, self.height//3)), ((0, self.width), (self.height//3 * 2, self.height))],
        #                   "black": [((0, self.width), (self.height//3 , self.height//3 * 2))]}
        self.structure = {"white": [((0, self.height//3), (0, self.width)), ((self.height//3 * 2, self.height), (0, self.width))],
                          "black": [((self.height//3 , self.height//3 * 2), (0, self.width))]}

    @staticmethod
    def check_arguments(width, height):
        return height % 3 == 0


class Feature3h2(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None):   
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width 
        self.rect_height = height // 3
        # self.structure = {"white": [((0, self.width), (self.height//3 , self.height//3 * 2))],
        #                   "black": [((0, self.width), (0, self.height//3)), ((0, self.width), (self.height//3 * 2, self.height))]}
        self.structure = {"white": [((self.height//3 , self.height//3 * 2), (0, self.width))],
                          "black": [((0, self.height//3), (0, self.width)), ((self.height//3 * 2, self.height), (0, self.width))]}

    @staticmethod
    def check_arguments(width, height):
        return height % 3 == 0


class Feature3v(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None):
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width // 3
        self.rect_height = height 
        # self.structure = {"white": [((0, self.width//3), (0, self.height)), ((self.width//3 * 2, self.width), (0, self.height))],
        #                   "black": [((self.width // 3, self.width // 3 * 2), (0 , self.height))]}
        self.structure = {"white": [((0, self.height), (0, self.width//3)), ((0, self.height), (self.width//3 * 2, self.width))],
                          "black": [((0, self.height), (self.width // 3, self.width // 3 * 2))]}

    @staticmethod
    def check_arguments(width, height):
        return width % 3 == 0

class Feature3v2(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None): 
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width // 3
        self.rect_height = height 
        # self.structure = {"white": [((self.width // 3, self.width // 3 * 2), (0 , self.height))],
        #                   "black": [((0, self.width//3), (0, self.height)), ((self.width//3 * 2, self.width), (0, self.height))]}
        self.structure = {"white": [((0 , self.height), (self.width // 3, self.width // 3 * 2))],
                          "black": [((0, self.height), (0, self.width//3)), ((0, self.height), (self.width//3 * 2, self.width), )]}

    @staticmethod
    def check_arguments(width, height):
        return width % 3 == 0


class Feature4(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None):
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width // 2
        self.rect_height = height // 2
        # self.structure = {"white": [((self.width // 2, self.width), (0 , self.height // 2)), ((0, self.width // 2), (self.height // 2 , self.height))],
        #                   "black": [((0, self.width//2), (0, self.height//2)), ((self.width//2, self.width), (self.height // 2, self.height))]}
        self.structure = {"white": [((0 , self.height // 2), (self.width // 2, self.width)), ((self.height // 2 , self.height), (0, self.width // 2))],
                          "black": [((0, self.height//2), (0, self.width//2)), ((self.height // 2, self.height), (self.width//2, self.width))]}

    @staticmethod
    def check_arguments(width, height):
        return width % 2 == 0 and height % 2 == 0


class Feature42(Feature):
    def __init__(self, x_pos, y_pos, width, height, weight=None):
        super().__init__(x_pos, y_pos, width, height, weight)
        self.rect_width = width // 2
        self.rect_height = height // 2
        # self.structure = {"white": [((0, self.width//2), (0, self.height//2)), ((self.width//2, self.width), (self.height // 2, self.height))],
        #                   "black": [((self.width // 2, self.width), (0 , self.height // 2)), ((0, self.width // 2), (self.height // 2 , self.height))]}
        self.structure = {"white": [((0, self.height//2), (0, self.width//2)), ((self.height // 2, self.height), (self.width//2, self.width))],
                          "black": [((0 , self.height // 2), (self.width // 2, self.width)), ((self.height // 2 , self.height), (0, self.width // 2))]}

    @staticmethod
    def check_arguments(width, height):
        return width % 2 == 0 and height % 2 == 0


def generate_all_features(feature_type, window_size):
    all_features = list()
    min_height = 1
    min_width = 1
    min_x_pos = 0
    min_y_pos = 0
    for height in range(min_height, window_size[0]+1):
        for width in range(min_width, window_size[1]+1):
            for x_pos in range(min_x_pos, window_size[0]):
                for y_pos in range(min_x_pos, window_size[1]):
                    if (feature_type.check_arguments(width, height) and x_pos + height <= window_size[0] and y_pos + width <= window_size[1]):
                        all_features.append(feature_type(x_pos, y_pos, width, height))

    return all_features





def main():
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # , [7, 8, 9]
    # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # f = Feature42(0, 0, 4, 2)
    # print(a)
    # # print(ImageCalculation.IntegralImage.get_integral_image(a))
    # print(f.get_prediction(ImageCalculation.IntegralImage.get_integral_image(a)))
    # print(Feature3h.check_arguments(2,2))

    print(len(generate_all_features(Feature2h, (20, 20))))
    Feature2h2.hi()



if __name__ == "__main__":
    main()