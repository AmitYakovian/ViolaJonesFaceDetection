import dill
import numpy as np
import glob
import random
from collections import namedtuple
from sklearn.utils import shuffle
import HaarFeatures
from ImageCalculation import IntegralImage

# from HaarFeatures import Feature2v, Feature2v2, Feature2h, Feature2h2, Feature3h, Feature3h2, Feature3v, Feature3v2, \
#      Feature4, Feature42

# ClassifierResult = namedtuple('ClassifierResult', [('threshold', float), ('polarity', int),
#                                                    ('classification_error', float),
#                                                    ('classifier', HaarFeatures.Feature)])
ClassifierResult = namedtuple('ClassifierResult', ['threshold', 'polarity', 'error', 'classifier'])

# WeakClassifier = namedtuple('WeakClassifier', [('threshold', float), ('polarity', int),
#                                                ('alpha', float),
#                                                ('classifier', Callable[[np.ndarray], float])])
WeakClassifier = namedtuple('WeakClassifier', ['threshold', 'polarity', 'alpha', 'classifier'])


class Model:

    def __init__(self):
        self.layers = []  # 5 layers

    def predict(self, integral_image: np.ndarray):
        for layer in self.layers:
            prediction = .0
            for weak_model in layer:
                prediction += weak_model.get_prediction(integral_image) / len(layer)
            if prediction < 0.5:
                return False
            else:
                continue

        return True

    @staticmethod
    def normalize_weights(images_weights) -> np.ndarray:
        return images_weights / images_weights.sum()

    @staticmethod
    def build_weak_classifiers(num_features: int, integral_images, labels: list, features):

        global ClassifierResult, WeakClassifier

        negative = len([label for label in labels if float(label) < .5])  # number of negative examples
        positive = len([label for label in labels if float(label) > .5])  # number of positive examples

        # Initialize the weights
        images_weights = np.zeros_like(labels)
        index = 0
        for label in labels:
            if float(label) < .5:
                images_weights[index] = 1. / (2. * negative)
            else:
                images_weights[index] = 1. / (2. * positive)
            index += 1

        labels = np.array(labels)
        weak_classifiers = []  # type: #list[WeakClassifier]
        for t in range(num_features):
            # Normalize the weights
            images_weights = Model.normalize_weights(images_weights)

            # Select best weak classifier for this round
            best = ClassifierResult(polarity=0, threshold=0, error=float('inf'), classifier=None)
            for f in features:
                result = Model.apply_feature(f, integral_images, labels, images_weights)
                if result.error < best.error:
                    best = result

            # After the best classifier was found
            beta = best.error / (1 - best.error)
            alpha = np.log(1. / beta)

            # Build the weak classifier
            classifier = WeakClassifier(threshold=best.threshold, polarity=best.polarity,
                                        classifier=best.classifier, alpha=alpha)

            # Update the weights for misclassified examples
            for index, (image, label) in enumerate(zip(integral_images, labels)):
                h = Model.weak_classifier(classifier.classifier, image, classifier.threshold,
                                          classifier.polarity)  # 0 or 1
                err = np.abs(h - label)  # 0 -> succeeded 1 -> error
                # decreases for a strong image and increases for a weak image
                images_weights[index] = images_weights[index] * np.power(beta, 1 - err)

            # Register this weak classifier
            weak_classifiers.append(classifier)

        return weak_classifiers

    @staticmethod
    def model_layers(models_num: int, feature_num_for_model: list, integral_images: list, labels: list, features: list):
        models = {}
        for i in range(1, models_num + 1):
            models[i] = Model.build_weak_classifiers(feature_num_for_model[i - 1], integral_images, labels,
                                                              features)
        return models

    @staticmethod
    def weak_classifier(feature, integral_image, threshold, polarity):
        # return 1 if threshold * polarity > feature(image) * polarity, else 0
        return (np.sign(threshold * polarity - feature.get_prediction(integral_image) * polarity) + 1) // 2

    @staticmethod
    def calculate_sums_for_threshold(labels: list, weights: list):
        """

        :param labels: list
        :param weights: list
        :return: 4 sums used to calculate the threshold of a specific feature:
        T+  The total sum of positive example weights
        T-  The total sum of negative example weights
        S+  List: for each index, the sum of positive weights below current index
        S-  List: for each index, the sum of negative weights below current index
        """

        t_plus, t_minus = .0, .0
        s_pluses, s_minuses = [], []
        s_minus, s_plus = .0, .0

        for label, weight in zip(labels, weights):
            if label < 1:
                s_minus += weight
                t_minus += weight
            else:
                s_plus += weight
                t_plus += weight
            s_minuses.append(s_minus)
            s_pluses.append(s_plus)

        return t_minus, t_plus, s_minuses, s_pluses

    @staticmethod
    def determine_threshold(feature_results, t_minus, t_plus, s_minuses, s_pluses):
        error = float('inf')
        picked_threshold, polarity = 0, 0

        for feature_result, s_minus, s_plus in zip(feature_results, s_minuses, s_pluses):
            error_1 = s_plus + (t_minus - s_minus)
            error_2 = s_minus + (t_plus - s_plus)

            if error_1 < error:
                polarity = -1
                error = error_1
                picked_threshold = feature_result

            elif error_2 < error:
                polarity = 1
                error = error_2
                picked_threshold = feature_result

        return picked_threshold, polarity

    @staticmethod
    def get_threshold_and_polarity(feature_results, labels, weights):
        sort_array = np.argsort(list(feature_results))
        feature_results, labels, weights = feature_results[sort_array], labels[sort_array], weights[sort_array]

        # get sums
        t_minus, t_plus, s_minuses, s_pluses = Model.calculate_sums_for_threshold(labels, weights)

        threshold, polarity = Model.determine_threshold(feature_results, t_minus, t_plus, s_minuses, s_pluses)

        return threshold, polarity

    @staticmethod
    def apply_feature(feature, integral_images, labels, image_weights):

        feature_results = np.array([feature.get_prediction(im) for im in integral_images])
        threshold, polarity = Model.get_threshold_and_polarity(feature_results, labels, image_weights)

        error_sum = 0
        for weight, integral_image, label in zip(image_weights, integral_images, labels):
            result = Model.weak_classifier(feature, integral_image, threshold, polarity)
            error_sum += weight * np.abs(label - result)

        return ClassifierResult(error=error_sum, threshold=threshold, polarity=polarity, classifier=feature)


    def save_model(self):
        with open(r"model.dill", 'wb') as f:
            dill.dump(model, f)


def load_model():
    with open(r"model.dill", 'rb') as f:
        model = dill.load(open(r"model.dill", 'rb'))
    return model


def load_dataset(path_to_faces, path_to_backgrounds):
    backgrounds = glob.glob(path_to_backgrounds)
    faces = glob.glob(path_to_faces)

    return faces, backgrounds


def prepare_dataset(faces, backgrounds):
    # prepare faces

    faces = random.sample(faces, len(backgrounds))

    xs = []
    ys = []
    for face_path in faces:
        img = IntegralImage.open_image(face_path)
        img = IntegralImage.resize_image(img)
        img = img.convert('L')
        np_array = IntegralImage.convert_image_to_numpy_array(img)
        integral_image = IntegralImage.get_integral_image(np_array)
        xs.append(integral_image)
        ys.append(1)

    for background_path in backgrounds:
        img = IntegralImage.open_image(background_path)
        grayscale = IntegralImage.covert_image_to_grayscale(img)
        img = IntegralImage.resize_image(grayscale)
        np_array = IntegralImage.convert_image_to_numpy_array(img)
        integral_image = IntegralImage.get_integral_image(np_array)
        xs.append(integral_image)
        ys.append(0)

    xs, ys = shuffle(xs, ys)

    return xs, ys


def main():
    faces, backgrounds = load_dataset(r"C:\Users\HP\Desktop\final_project\dataset\face_images\*.png",
                                      r"C:\Users\HP\Desktop\final_project\dataset\background_images\*.jpg")
    xs, ys = prepare_dataset(faces, backgrounds)

    # xs = dill.load(open(r"xs1.dill", 'rb'))
    #
    # ys = dill.load(open(r"ys1.dill", 'rb'))

    features = HaarFeatures.all_features()
    strong_classifier = Model.model_layers(6, [2, 10, 25, 50, 50, 50], xs, ys, features)


def main1():
    faces, backgrounds = load_dataset(r"C:\Users\HP\Desktop\final_project\dataset\face_images\*.png",
                                      r"C:\Users\HP\Desktop\final_project\dataset\background_images\*.jpg")
    xs, ys = prepare_dataset(faces, backgrounds)

    with open(r"xs1.dill", 'wb') as f:
            dill.dump(xs, f)

    with open(r"ys1.dill", 'wb') as f:
            dill.dump(ys, f)


def main2():
    xs = dill.load(open(r"xs1.dill", 'rb'))

    ys = dill.load(open(r"ys1.dill", 'rb'))

    print(ys)
    print(len(xs))

if __name__ == '__main__':
    main()
